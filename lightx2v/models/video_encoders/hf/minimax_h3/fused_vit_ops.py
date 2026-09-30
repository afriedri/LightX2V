"""Fused elementwise/norm Triton kernels for the MiniMax-H3 video-VAE ViT decoder block (FP32 sensitive dtype, FP16 GEMMs).

Replaces, per block:
  norm1/norm2: FP32 RMSNorm (affine) + .to(fp16)                       -> _rms_rows (one pass, writes fp16)
  q/k: .to(fp32) + RMSNorm(64, no affine) + .to(fp16) + rotary (fp16)   -> _qk_norm_rope (one kernel for q and k)
  residual: h + out.to(fp32) * scale                                     -> _gated_add
  SwiGLU:  a * silu(g)  (fp16, chunk of the proj output)                -> _swiglu
Every intermediate rounding of the eager path is reproduced (fp16 rounding after each fp16 op, no FMA contraction).
The residual/gate and SwiGLU kernels are bitwise equal to eager; the RMS statistics are FP32 row reductions in a
different order than torch's kernel (block output rel. error ~1.5e-6; decoded 8-bit video differs by at most 1 level).
Disable with LIGHTX2V_H3_VAE_FUSED=0.
"""
import types
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _rms_rows(x, w, out, cols: tl.constexpr, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    m = offs < cols
    v = tl.load(x + row * cols + offs, m, other=0.0)
    ms = tl.sum(v * v, 0) / cols
    r = 1.0 / tl.sqrt_rn(ms + eps)
    y = (v * r) * tl.load(w + offs, m, other=0.0)
    tl.store(out + row * cols + offs, y.to(tl.float16), m)


@triton.jit
def _qk_norm_rope(q, k, cos, sin, qo, ko, heads: tl.constexpr, eps, ROT: tl.constexpr, HEADS_PER: tl.constexpr,
                  TOKENS):
    # one program = HEADS_PER heads of one token, for q and k. D = 64, rotary dim ROT (48), cos/sin [T, ROT] fp32
    # (shared by every batch item: token index modulo TOKENS)
    tok = tl.program_id(0)
    pos = tok % TOKENS
    hb = tl.program_id(1)
    d = tl.arange(0, 64)
    h = hb * HEADS_PER + tl.arange(0, HEADS_PER)
    base = tok * heads * 64 + h[:, None] * 64 + d[None, :]
    half: tl.constexpr = ROT // 2
    rot_mask = d < ROT
    c = tl.load(cos + pos * ROT + d, rot_mask, other=0.0).to(tl.float16).to(tl.float32)
    s = tl.load(sin + pos * ROT + d, rot_mask, other=0.0).to(tl.float16).to(tl.float32)
    partner = tl.where(d < half, d + half, d - half)
    sign = tl.where(d < half, -1.0, 1.0)
    for which in tl.static_range(2):
        src = q if which == 0 else k
        dst = qo if which == 0 else ko
        v = tl.load(src + base).to(tl.float32)
        ms = tl.sum(v * v, 1) / 64
        r = 1.0 / tl.sqrt_rn(ms + eps)
        n = (v * r[:, None]).to(tl.float16).to(tl.float32)
        # partner values (normalized) for the rotation
        pv = tl.load(src + tok * heads * 64 + h[:, None] * 64 + partner[None, :]).to(tl.float32)
        pn = (pv * r[:, None]).to(tl.float16).to(tl.float32)
        rotated = pn * sign[None, :]            # -second / first, exact in fp16
        a = (n * c[None, :]).to(tl.float16).to(tl.float32)
        b = (rotated * s[None, :]).to(tl.float16).to(tl.float32)
        y = tl.where(rot_mask[None, :], (a + b), n)
        tl.store(dst + base, y.to(tl.float16))


@triton.jit
def _gated_add(h, o, scale, out, n, cols: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    a = tl.load(h + offs, m, other=0.0)
    b = tl.load(o + offs, m, other=0.0).to(tl.float32)
    s = tl.load(scale + offs % cols, m, other=0.0)
    tl.store(out + offs, a + b * s, m)


@triton.jit
def _swiglu(p, out, rows, inner: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    n = rows * inner
    m = offs < n
    r = offs // inner
    c = offs % inner
    a = tl.load(p + r * (2 * inner) + c, m, other=0.0).to(tl.float32)
    g = tl.load(p + r * (2 * inner) + inner + c, m, other=0.0).to(tl.float32)
    sg = (g / (1.0 + tl.exp(-g))).to(tl.float16).to(tl.float32)
    tl.store(out + offs, (a * sg).to(tl.float16), m)


def rms_to_half(x, weight, eps):
    x2 = x.reshape(-1, x.shape[-1])
    out = torch.empty(x2.shape, device=x.device, dtype=torch.float16)
    _rms_rows[(x2.shape[0],)](x2, weight, out, x2.shape[1], eps, BLOCK=triton.next_power_of_2(x2.shape[1]),
                              num_warps=8, enable_fp_fusion=False)
    return out.view(x.shape)


def qk_norm_rope(q, k, cos, sin, eps):
    # q,k: [B,T,H,64] fp16 (views of [B,T,H*64]); cos/sin: [1 or B,T,1,ROT] fp32, identical for every batch item
    B, T, H, Dh = q.shape
    q = q.contiguous(); k = k.contiguous()
    qo = torch.empty_like(q); ko = torch.empty_like(k)
    rot = cos.shape[-1]
    c = cos[:1].reshape(T, rot).contiguous(); s = sin[:1].reshape(T, rot).contiguous()
    HP = 8
    _qk_norm_rope[(B * T, H // HP)](q, k, c, s, qo, ko, H, eps, ROT=rot, HEADS_PER=HP, TOKENS=T, num_warps=2,
                                    enable_fp_fusion=False)
    return qo, ko


def gated_add(h, o, scale):
    out = torch.empty_like(h)
    n = h.numel()
    _gated_add[(triton.cdiv(n, 2048),)](h, o, scale, out, n, h.shape[-1], BLOCK=2048, num_warps=8, enable_fp_fusion=False)
    return out


def swiglu(p):
    inner = p.shape[-1] // 2
    rows = p.numel() // p.shape[-1]
    out = torch.empty(p.shape[:-1] + (inner,), device=p.device, dtype=p.dtype)
    _swiglu[(triton.cdiv(rows * inner, 2048),)](p, out, rows, inner, BLOCK=2048, num_warps=8, enable_fp_fusion=False)
    return out


def _linear(module, x):
    """nn.Linear, or the FP8 W8A8 path when the module carries ``_fp8`` (per-channel E4M3 weight, per-token act)."""
    fp8 = getattr(module, "_fp8", None)
    if fp8 is None:
        return module(x)
    from aiter.ops.gfx1201 import fp8_gemm
    x2 = x.reshape(-1, x.shape[-1])
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    x8, xs = fp8_gemm.quant_rows(x2)
    weight8, weight_scales, bias = fp8
    out = torch.empty((x2.shape[0], weight8.shape[0]), device=x.device, dtype=x.dtype)
    fp8_gemm.mm8_out(x8, xs, weight8, weight_scales, out, bias=bias)
    return out.view(*x.shape[:-1], weight8.shape[0])


def _attn_forward(self, hidden_states, rotary_emb=None):
    packed = getattr(self, "_packed_qkv", None)
    if packed is not None:
        qkv = _linear(packed, hidden_states)
        query, key, value = qkv.split(self.inner_dim, dim=-1)
    else:
        query = _linear(self.to_q, hidden_states)
        key = _linear(self.to_k, hidden_states)
        value = _linear(self.to_v, hidden_states)
    query = query.unflatten(2, (self.heads, self.dim_head))
    key = key.unflatten(2, (self.heads, self.dim_head))
    value = value.unflatten(2, (self.heads, self.dim_head))
    cos, sin = rotary_emb
    query, key = qk_norm_rope(query, key, cos, sin, self.norm_q.eps)
    hidden_states = self.calculate.apply(query, key, value, max_seqlen_q=query.shape[1],
                                         max_seqlen_kv=key.shape[1]).view(query.shape[0], query.shape[1], self.inner_dim)
    return _linear(self.to_out[0], hidden_states)


def _block_forward(self, hidden_states, rotary_emb=None):
    n = rms_to_half(hidden_states, self.norm1.weight, self.norm1.eps)
    a = self.attn(n, rotary_emb)
    hidden_states = gated_add(hidden_states, a, self.scale1)
    n = rms_to_half(hidden_states, self.norm2.weight, self.norm2.eps)
    ff = self.ff.net
    y = _linear(ff[2], swiglu(_linear(ff[0].proj, n)))
    return gated_add(hidden_states, y, self.scale2)


def eligible(block):
    a = block.attn
    return (block.infer_dtype == torch.float16 and block.sensitive_layer_dtype == torch.float32 and a.to_qkv is None
            and a.dim_head == 64 and a.heads % 8 == 0 and block.norm1.weight is not None
            and block.norm1.weight.dtype == torch.float32 and a.norm_q.weight is None)


def _pack_qkv(attn):
    """One [3*inner, dim] Linear whose weight/bias storage the original to_q/to_k/to_v parameters view (no extra memory)."""
    linears = (attn.to_q, attn.to_k, attn.to_v)
    weight = torch.cat([linear.weight.data for linear in linears], dim=0)
    bias = torch.cat([linear.bias.data for linear in linears], dim=0) if linears[0].bias is not None else None
    packed = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None, device="meta", dtype=weight.dtype)
    packed.weight = torch.nn.Parameter(weight, requires_grad=False)
    if bias is not None:
        packed.bias = torch.nn.Parameter(bias, requires_grad=False)
    rows = attn.inner_dim
    for index, linear in enumerate(linears):
        linear.weight = torch.nn.Parameter(weight[index * rows:(index + 1) * rows], requires_grad=False)
        if bias is not None:
            linear.bias = torch.nn.Parameter(bias[index * rows:(index + 1) * rows], requires_grad=False)
    attn._packed_qkv = packed


def _quantize_linear(linear):
    """Attach per-output-channel E4M3 weights (quantized on the weight's device) for the FP8 decode path."""
    from aiter.ops.gfx1201 import fp8_gemm
    weight = linear.weight.data
    if weight.device.type != "cuda":
        return False
    weight8, scales = fp8_gemm.quant_rows(weight.contiguous())
    linear._fp8 = (weight8, scales, linear.bias.data.contiguous() if linear.bias is not None else None)
    return True


def _release_weight(linear):
    # The FP8 path never reads the FP16 weight again; free it (the VAE stays resident during the DiT).
    linear.weight = torch.nn.Parameter(torch.empty(0, device=linear.weight.device, dtype=linear.weight.dtype),
                                       requires_grad=False)


def patch_block(block, fp8=False):
    """fp8=True (LIGHTX2V_H3_VAE_FP8=1): Q/K/V (packed), attention-out and FFN Linears run as FP8 W8A8
    (per-output-channel E4M3 weights, per-token E4M3 activations); the FP16 weights are released."""
    if not eligible(block):
        return False
    if fp8:
        _pack_qkv(block.attn)
        linears = [block.attn._packed_qkv, block.attn.to_out[0], block.ff.net[0].proj, block.ff.net[2]]
        if all(_quantize_linear(linear) for linear in linears):
            for linear in linears + [block.attn.to_q, block.attn.to_k, block.attn.to_v]:
                _release_weight(linear)
    block.attn.forward = types.MethodType(_attn_forward, block.attn)
    block.forward = types.MethodType(_block_forward, block)
    return True
