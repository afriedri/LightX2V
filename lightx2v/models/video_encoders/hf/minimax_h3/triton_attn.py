"""Dense FP16 flash-attention forward (Triton) for the MiniMax-H3 video-VAE ViT decoder on RDNA4 (gfx1201).

Non-causal, no mask, head_dim 64. Reads q/k/v in the decoder's natural [B, T, H, D] layout (no transposes) and
writes [B, T, H*D] directly (no output transpose/reshape copy). FP32 online softmax (exp2) and FP32 accumulation;
P is rounded to FP16 before the PV product, like the flash kernels. Same accuracy class as torch SDPA/aotriton
(max |err| vs an FP32 reference identical on random inputs), but not bitwise equal to it.
On R9600D (24 CUs) at B=2, T=1797, H=32: ~1.06 ms vs 1.84 ms for SDPA + output reshape (~50 vs ~29 TFLOPS).
Enabled with LIGHTX2V_H3_VAE_TRITON_ATTN=1 (default 0).
"""
import torch
import triton
import triton.language as tl

_LOG2E = 1.4426950408889634


@triton.jit
def _fa_fwd(Q, K, V, O, T, H: tl.constexpr, sm_scale_log2, D: tl.constexpr, BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr, SQ: tl.constexpr, SK: tl.constexpr, SV: tl.constexpr):
    # SQ/SK/SV: token (row) strides of q/k/v in elements (heads contiguous inside a row); O is [B, T, H*D]
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    stride_t = H * D
    bt = b.to(tl.int64) * T
    base = bt * stride_t + h * D
    Q += bt * SQ + h * D
    K += bt * SK + h * D
    V += bt * SV + h * D
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < T
    q = tl.load(Q + offs_m[:, None] * SQ + offs_d[None, :], mask_m[:, None], other=0.0)
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    n_full = (T // BLOCK_N) * BLOCK_N
    for start in range(0, n_full, BLOCK_N):
        kn = start + offs_n
        k = tl.load(K + kn[None, :] * SK + offs_d[:, None])
        s = tl.dot(q, k) * sm_scale_log2
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.math.exp2(s - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(V + kn[:, None] * SV + offs_d[None, :])
        acc = tl.dot(p.to(tl.float16), v, acc)
        m_i = m_new
    if n_full < T:
        kn = n_full + offs_n
        mk = kn < T
        k = tl.load(K + kn[None, :] * SK + offs_d[:, None], mk[None, :], other=0.0)
        s = tl.dot(q, k) * sm_scale_log2
        s = tl.where(mk[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.math.exp2(s - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(V + kn[:, None] * SV + offs_d[None, :], mk[:, None], other=0.0)
        acc = tl.dot(p.to(tl.float16), v, acc)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(O + base + offs_m[:, None] * stride_t + offs_d[None, :], acc.to(tl.float16), mask_m[:, None])


def flash_attn_bthd(q, k, v, block_m=128, block_n=32, num_warps=4, waves_per_eu=2):
    """q, k, v: [B, T, H, D] fp16, D = 64. Row-strided views (e.g. splits of a packed QKV projection) are read in
    place as long as each token row holds the H*D values contiguously. Returns [B, T, H*D]."""
    B, T, H, D = q.shape

    def rows(x):
        if x.stride(3) == 1 and x.stride(2) == D and x.stride(0) == T * x.stride(1):
            return x, x.stride(1)
        x = x.contiguous()
        return x, H * D

    (q, sq), (k, sk), (v, sv) = rows(q), rows(k), rows(v)
    o = torch.empty((B, T, H * D), device=q.device, dtype=q.dtype)
    _fa_fwd[(triton.cdiv(T, block_m), B * H)](q, k, v, o, T, H, (D ** -0.5) * _LOG2E, D=D, BLOCK_M=block_m,
                                              BLOCK_N=block_n, SQ=sq, SK=sk, SV=sv, num_warps=num_warps,
                                              num_stages=1, waves_per_eu=waves_per_eu)
    return o
