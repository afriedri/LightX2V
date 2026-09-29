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
def _qk_norm_rope(q, k, cos, sin, qo, ko, heads: tl.constexpr, eps, ROT: tl.constexpr, HEADS_PER: tl.constexpr):
    # one program = HEADS_PER heads of one token, for q and k. D = 64, rotary dim ROT (48), cos/sin [T, ROT] fp32
    tok = tl.program_id(0)
    hb = tl.program_id(1)
    d = tl.arange(0, 64)
    h = hb * HEADS_PER + tl.arange(0, HEADS_PER)
    base = tok * heads * 64 + h[:, None] * 64 + d[None, :]
    half: tl.constexpr = ROT // 2
    rot_mask = d < ROT
    c = tl.load(cos + tok * ROT + d, rot_mask, other=0.0).to(tl.float16).to(tl.float32)
    s = tl.load(sin + tok * ROT + d, rot_mask, other=0.0).to(tl.float16).to(tl.float32)
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
    # q,k: [1,T,H,64] fp16 (views of contiguous [1,T,H*64]); cos/sin: [1,T,1,ROT] fp32
    B, T, H, Dh = q.shape
    q = q.contiguous(); k = k.contiguous()
    qo = torch.empty_like(q); ko = torch.empty_like(k)
    rot = cos.shape[-1]
    c = cos.reshape(T, rot).contiguous(); s = sin.reshape(T, rot).contiguous()
    HP = 8
    _qk_norm_rope[(B * T, H // HP)](q, k, c, s, qo, ko, H, eps, ROT=rot, HEADS_PER=HP, num_warps=2, enable_fp_fusion=False)
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


def _attn_forward(self, hidden_states, rotary_emb=None):
    query = self.to_q(hidden_states)
    key = self.to_k(hidden_states)
    value = self.to_v(hidden_states)
    query = query.unflatten(2, (self.heads, self.dim_head))
    key = key.unflatten(2, (self.heads, self.dim_head))
    value = value.unflatten(2, (self.heads, self.dim_head))
    cos, sin = rotary_emb
    query, key = qk_norm_rope(query, key, cos, sin, self.norm_q.eps)
    hidden_states = self.calculate.apply(query, key, value, max_seqlen_q=query.shape[1],
                                         max_seqlen_kv=key.shape[1]).view(query.shape[0], query.shape[1], self.inner_dim)
    return self.to_out[0](hidden_states)


def _block_forward(self, hidden_states, rotary_emb=None):
    n = rms_to_half(hidden_states, self.norm1.weight, self.norm1.eps)
    a = self.attn(n, rotary_emb)
    hidden_states = gated_add(hidden_states, a, self.scale1)
    n = rms_to_half(hidden_states, self.norm2.weight, self.norm2.eps)
    ff = self.ff.net
    y = ff[2](swiglu(ff[0].proj(n)))
    return gated_add(hidden_states, y, self.scale2)


def eligible(block):
    a = block.attn
    return (block.infer_dtype == torch.float16 and block.sensitive_layer_dtype == torch.float32 and a.to_qkv is None
            and a.dim_head == 64 and a.heads % 8 == 0 and block.norm1.weight is not None
            and block.norm1.weight.dtype == torch.float32 and a.norm_q.weight is None)


def patch_block(block):
    if not eligible(block):
        return False
    block.attn.forward = types.MethodType(_attn_forward, block.attn)
    block.forward = types.MethodType(_block_forward, block)
    return True
