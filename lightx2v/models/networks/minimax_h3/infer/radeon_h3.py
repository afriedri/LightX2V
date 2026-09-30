import os
import sys

import torch
from torch.profiler import record_function
from lightx2v.common.ops.mm.mm_weight import MMWeight, MMWeightTP, unwrap_tp_weight
from aiter.ops.gfx1201.h3 import H3TP2Workspace
from aiter.ops.gfx1201.modulation import modulate


def enabled(name):
    return os.environ.get("RADEON_CORESW_" + name, "1") != "0"


_FP8_STEP_OFF = False


def _fp8_step_gate(owner):
    """RADEON_CORESW_FP8_STEPS="first:last" (1-based, inclusive) limits FP8 to those denoising steps; BF16 elsewhere."""
    global _FP8_STEP_OFF
    calls = owner.__dict__.get("_radeon_block_calls", 0)
    owner._radeon_block_calls = calls + 1
    window = os.environ.get("RADEON_CORESW_FP8_STEPS", "").strip()
    block_window = os.environ.get("RADEON_CORESW_FP8_BLOCKS", "").strip()
    if not window and not block_window:
        _FP8_STEP_OFF = False
        return
    blocks = int(getattr(owner, "blocks_num", 0) or owner.config.get("num_layers", 50) or 50)
    step, block = calls // blocks + 1, calls % blocks
    off = False
    if window:
        first, _, last = window.partition(":")
        off = not (int(first or 1) <= step <= int(last or 10**9))
    if block_window:
        # RADEON_CORESW_FP8_BLOCKS="first:last" (0-based, inclusive) limits FP8 to those transformer blocks
        first, _, last = block_window.partition(":")
        off = off or not (int(first or 0) <= block <= int(last or 10**9))
    _FP8_STEP_OFF = off


def fp8_enabled(part):
    """RADEON_CORESW_FP8: "0" (default) = BF16 GEMMs; "1"/"all" = FP8 W8A8 for ffn,qkv,out; or a comma list of parts."""
    if _FP8_STEP_OFF:
        return False
    value = os.environ.get("RADEON_CORESW_FP8", "0").strip().lower()
    if value in ("", "0", "none", "off"):
        return False
    return value in ("1", "all") or part in value.split(",")


def _fp8_weight(owner, name, weight):
    quantizer = getattr(owner, "_radeon_fp8_weights", None)
    if quantizer is None:
        from aiter.ops.gfx1201.fp8_gemm import WeightQuantizer
        quantizer = owner._radeon_fp8_weights = WeightQuantizer()
    return quantizer(name, weight)


def _fp8_shadow(owner, tag, fp8_out, bf16_fn):
    """RADEON_FP8_SHADOW=K: every K-th block call, log the FP8 output's relative error against the BF16 kernel."""
    every = int(os.environ.get("RADEON_FP8_SHADOW", "0") or 0)
    if every <= 0:
        return
    counts = owner.__dict__.setdefault("_radeon_fp8_shadow_count", {})
    count = counts.get(tag, 0)
    counts[tag] = count + 1
    if count % every:
        return
    reference = bf16_fn()
    outs = fp8_out if isinstance(fp8_out, (tuple, list)) else (fp8_out,)
    refs = reference if isinstance(reference, (tuple, list)) else (reference,)
    errors = []
    for out, ref in zip(outs, refs):
        diff = (out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-30)
        errors.append(f"{diff.item():.3e}")
    print(f"[RADEON FP8 SHADOW][rank={os.environ.get('RANK', '0')}] {tag} call={count} rel_err={','.join(errors)} "
          f"ref_rms={refs[0].float().pow(2).mean().sqrt().item():.3e}", file=sys.stderr, flush=True)


def norm_modulation(norm, inputs, scale, shift, indices, fp8=False):
    """RMSNorm + AdaLN modulation. NORM_MODULATION=1 fuses both into one pass (1-ulp BF16 differences on ~2e-6 of outputs).
    fp8=True returns (E4M3 [S,D], FP32 per-token scales) of the same BF16 modulated value."""
    from lightx2v.common.ops.norm.rms_norm_weight import RMSWeightNative

    if fp8:
        weight = norm._get_actual_weight() if type(norm) is RMSWeightNative else None
        if not (isinstance(weight, torch.Tensor) and weight.dtype == torch.bfloat16 and inputs.dtype == torch.bfloat16
                and inputs.ndim == 2 and inputs.is_contiguous()):
            raise ValueError("RADEON_CORESW_FP8 requires the fused BF16 RMSNorm+modulation path")
        from aiter.ops.gfx1201.fp8_gemm import rms_modulate_fp8
        return rms_modulate_fp8(inputs, weight, norm.eps, scale, shift, indices)

    if enabled("NORM_MODULATION") and enabled("LOSSLESS") and type(norm) is RMSWeightNative and inputs.ndim == 2 and inputs.is_contiguous():
        weight = norm._get_actual_weight()
        if isinstance(weight, torch.Tensor) and weight.dtype == torch.bfloat16 and inputs.dtype == torch.bfloat16:
            from aiter.ops.gfx1201.norm_modulation import rms_modulate

            return rms_modulate(inputs, weight, norm.eps, scale, shift, indices)
    return apply_modulation(norm.apply(inputs), scale, shift, indices)


def apply_modulation(inputs, scale, shift, indices):
    if enabled("LOSSLESS"):
        return modulate(inputs, scale, shift, indices)
    return inputs * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)


def project_attention(owner, module, inputs, residual, gate, indices, workspace):
    if fp8_enabled("out") and enabled("OUT_PROJECTION") and enabled("LOSSLESS") and workspace is not None:
        from aiter.ops.gfx1201 import fp8_gemm
        weight = local_weight(module, owner.tp_group, "row", (3584, 5376)).t()
        weight8, weight_scales = _fp8_weight(owner, "out", weight)
        inputs8, input_scales = fp8_gemm.quant_rows(inputs)
        if os.environ.get("RADEON_FP8_SHADOW"):
            from aiter.ops.gfx1201.out_projection import project_out
            rows = min(8192, inputs.shape[0])
            out8 = torch.empty((rows, 5376), device=inputs.device, dtype=torch.bfloat16)
            fp8_gemm.mm8_out(inputs8[:rows], input_scales[:rows], weight8, weight_scales, out8)

            def reference():
                out = torch.empty((rows, 5376), device=inputs.device, dtype=torch.bfloat16)
                project_out(inputs[:rows], weight.t(), out)
                return out
            _fp8_shadow(owner, "out", out8, reference)
        return workspace.projection.run8(inputs8, input_scales, weight8, weight_scales, residual, gate, indices)
    if enabled("OUT_PROJECTION"):
        weight = local_weight(module, owner.tp_group, "row", (3584, 5376))
        if enabled("LOSSLESS"):
            return workspace.attention_output(inputs, weight, residual, gate, indices)
        from aiter.ops.gfx1201.out_projection import project_out
        output = torch.empty_like(residual)
        project_out(inputs, weight, output)
        torch.distributed.all_reduce(output, group=owner.tp_group)
    else:
        output = module.apply(inputs)
    return residual + gate.index_select(0, indices) * output


def feed_forward(owner, weights, inputs, residual, gate, indices, workspace):
    if isinstance(inputs, tuple):
        return feed_forward8(owner, weights, inputs, residual, gate, indices, workspace)
    custom_up, custom_down = enabled("FFN_UP"), enabled("FFN_DOWN")
    if custom_up:
        up_weight = local_weight(weights.in_proj, owner.tp_group, "col", (5376, 14336))
    if custom_down:
        down_weight = local_weight(weights.out_proj, owner.tp_group, "row", (7168, 5376))
    if custom_up and custom_down and enabled("LOSSLESS"):
        return workspace.feed_forward(inputs, up_weight.t(), down_weight, residual, gate, indices)
    if custom_up:
        from aiter.ops.gfx1201.ffn_up import fused_up
        activation = fused_up(inputs, up_weight.t())
    else:
        value, activation_gate = weights.in_proj.apply(inputs).chunk(2, dim=-1)
        activation = value * torch.nn.functional.silu(activation_gate)
    if custom_down:
        from aiter.ops.gfx1201.ffn_down import down_out
        output = torch.empty_like(residual)
        down_out(activation, down_weight, output)
        torch.distributed.all_reduce(output, group=owner.tp_group)
    else:
        output = weights.out_proj.apply(activation)
    return residual + gate.index_select(0, indices) * output


def feed_forward8(owner, weights, inputs, residual, gate, indices, workspace):
    from aiter.ops.gfx1201 import fp8_gemm
    inputs8, input_scales = inputs
    up_weight = local_weight(weights.in_proj, owner.tp_group, "col", (5376, 14336)).t()
    down_weight = local_weight(weights.out_proj, owner.tp_group, "row", (7168, 5376)).t()
    up8, up_scales = _fp8_weight(owner, "ffn_up", up_weight)
    down8, down_scales = _fp8_weight(owner, "ffn_down", down_weight)
    if os.environ.get("RADEON_FP8_SHADOW"):
        from aiter.ops.gfx1201.ffn_up import fused_up
        from aiter.ops.gfx1201.ffn_down import down_out
        rows = min(8192, inputs8.shape[0])
        x = (inputs8[:rows].float() * input_scales[:rows, None]).to(torch.bfloat16)
        act8 = torch.empty((rows, 7168), device=x.device, dtype=torch.bfloat16)
        fp8_gemm.up8_swiglu_out(inputs8[:rows], input_scales[:rows], up8, up_scales, act8)
        q, qs = fp8_gemm.quant_rows(act8)
        out8 = torch.empty((rows, 5376), device=x.device, dtype=torch.bfloat16)
        fp8_gemm.mm8_out(q, qs, down8, down_scales, out8)

        def reference():
            act = fused_up(x, up_weight)
            out = torch.empty((rows, 5376), device=x.device, dtype=torch.bfloat16)
            down_out(act, down_weight.t(), out)
            return act, out
        _fp8_shadow(owner, "ffn(act,out;in=dequant)", (act8, out8), reference)
    return workspace.ffn.run8(inputs8, input_scales, up8, up_scales, down8, down_scales, residual, gate, indices)


def report_status(owner):
    enabled = bool(owner.config.get("radeon_gfx1201_h3", False))
    if getattr(owner, "_radeon_reported_status", None) == enabled:
        return
    rank = os.environ.get("RANK", "0")
    status = "ENABLED" if enabled else "DISABLED"
    detail = ("awaiting first fused block submission" if enabled else
              "block fusion OFF; library FFN/projection + RCCL remain active; "
              "set radeon_gfx1201_h3=true to enable; attention dispatch is independent")
    print(f"========== [RADEON OP][rank={rank}][{status}] {detail} ==========",
          file=sys.stderr, flush=True)
    owner._radeon_reported_status = enabled


def local_weight(module, group, split, shape):
    if not isinstance(module, MMWeightTP) or module.tp_size != 2 or module.tp_group is not group or module.split_dim != split:
        raise ValueError("Radeon H3 requires matching TP2 weight partitions")
    inner = unwrap_tp_weight(module)
    if type(inner) is not MMWeight or inner.has_lora_branch or inner.has_diff:
        raise ValueError("Radeon H3 supports unquantized BF16 weights without LoRA/diff only")
    if getattr(inner, "bias", None) is not None or getattr(module, "_row_split_bias", None) is not None:
        raise ValueError("Radeon H3 fused projections require no bias")
    if split == "row" and not module.reduce_output:
        raise ValueError("Radeon H3 row projections require SUM")
    if split == "col" and module.lora_column_chunks != 2:
        raise ValueError("Radeon H3 requires paired local SwiGLU halves")
    weight = inner._get_actual_weight()
    if tuple(weight.shape) != shape or weight.dtype != torch.bfloat16 or weight.stride() != (1, shape[0]):
        raise ValueError(f"Unexpected Radeon H3 weight layout: {weight.shape}, {weight.stride()}, {weight.dtype}")
    return weight


def infer_block(owner, weights, hidden_states, pre_infer_out, modulation):
    _fp8_step_gate(owner)
    return _infer_block(owner, weights, hidden_states, pre_infer_out, modulation)


def _infer_block(owner, weights, hidden_states, pre_infer_out, modulation):
    if owner.tp_size != 2 or owner.seq_p_group is not None or pre_infer_out.sequence_parallel_state is not None:
        raise ValueError("Radeon H3 block fusion requires TP2 without sequence parallelism")
    if hidden_states.dtype != torch.bfloat16 or owner.hidden_size != 5376 or owner.num_heads != 28 or owner.head_dim != 128:
        raise ValueError("Radeon H3 block fusion requires BF16 H3 TP2 geometry")
    workspace = None
    if enabled("LOSSLESS") and (enabled("OUT_PROJECTION") or (enabled("FFN_UP") and enabled("FFN_DOWN"))):
        workspace = getattr(owner, "_radeon_workspace", None)
        if workspace is None:
            workspace = H3TP2Workspace(hidden_states.shape[0], hidden_states.device, owner.tp_group)
            owner._radeon_workspace = workspace
        if workspace.rows != hidden_states.shape[0] or workspace.device != hidden_states.device:
            raise ValueError("Radeon H3 workspace shape/device changed; recreate the infer instance")
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
    indices = pre_infer_out.adaln_indices
    with record_function("radeon_h3.norm_modulation.attn"):
        normed = norm_modulation(weights.norm1, hidden_states, scale_msa, shift_msa, indices, fp8=fp8_enabled("qkv"))
    with record_function("radeon_h3.attention"):
        attention = norm_rope_attention(owner, weights.attn, normed, pre_infer_out)
        if attention is None and isinstance(normed, tuple):
            raise ValueError("RADEON_CORESW_FP8=qkv requires the fused norm/RoPE/Sage attention path")
        if attention is None:
            attention = owner._attention(weights.attn, normed, pre_infer_out, project_output=False)
    with record_function("radeon_h3.out_projection_reduce_residual"):
        hidden_states = project_attention(owner, weights.attn.to_out, attention, hidden_states, gate_msa, indices, workspace)
    del attention, normed
    with record_function("radeon_h3.norm_modulation.ffn"):
        normed = norm_modulation(weights.norm2, hidden_states, scale_mlp, shift_mlp, indices,
                                 fp8=fp8_enabled("ffn") and workspace is not None)
    with record_function("radeon_h3.ffn_reduce_residual"):
        output = feed_forward(owner, weights.ff, normed, hidden_states, gate_mlp, indices, workspace)
    if not getattr(owner, "_radeon_active_reported", False):
        rank = os.environ.get("RANK", "0")
        print(f"========== [RADEON OP][rank={rank}][ACTIVE] TP2 fused block submitted; "
              f"device={hidden_states.device} shape={tuple(hidden_states.shape)}; "
              f"switches={dict((name, enabled(name)) for name in ('LOSSLESS', 'ATTENTION', 'OUT_PROJECTION', 'FFN_UP', 'FFN_DOWN', 'QKV_PROJECTION', 'NORM_MODULATION'))}; "
              f"fp8={[part for part in ('qkv', 'out', 'ffn') if fp8_enabled(part)]}; "
              "asynchronous submission only, confirm completion in GPU trace ==========",
              file=sys.stderr, flush=True)
        owner._radeon_active_reported = True
    return output


def _qk_rope_inputs(owner, weights, query, key, value, rotary_emb):
    from lightx2v.common.ops.norm.rms_norm_weight import RMSWeightNative
    from lightx2v.common.ops.rope.torch_rope import TorchRealRope

    def fallback(reason):
        reported = getattr(owner, "_radeon_qk_fallbacks", set())
        if reason not in reported:
            print(f"[RADEON OP][rank={os.environ.get('RANK', '0')}][FALLBACK] QK norm/RoPE: {reason}",
                  file=sys.stderr, flush=True)
            reported.add(reason)
            owner._radeon_qk_fallbacks = reported
        return None

    if owner.tp_size != 2 or owner.seq_p_group is not None or owner.num_heads != 28 or owner.head_dim != 128:
        return fallback("requires TP2 local H28 D128 without SP")
    if type(weights.norm_q) is not RMSWeightNative or type(weights.norm_k) is not RMSWeightNative:
        return fallback("requires torch_native RMSNorm")
    if weights.norm_q.eps != 1e-5 or weights.norm_k.eps != 1e-5:
        return fallback("requires QK norm eps=1e-5")
    if type(weights.rope) is not TorchRealRope or weights.rope.layout != "split_half" or weights.rope.compute_dtype != torch.float32:
        return fallback("requires FP32 split-half TorchRealRope")
    if query.ndim not in (2, 3) or query.shape[0] < 1:
        return fallback("unsupported QKV shape")
    rows = query.shape[0]
    if any(tuple(tensor.shape) not in ((rows, 3584), (rows, 28, 128)) for tensor in (query, key, value)):
        return fallback("requires local QKV [S,3584] or [S,28,128]")
    if query.numel() >= 2**31:
        return fallback("int32 index limit")
    if not isinstance(rotary_emb, tuple) or len(rotary_emb) != 2:
        return fallback("requires full cosine/sine tuple")
    cosine, sine = rotary_emb
    if any(not isinstance(tensor, torch.Tensor) or tensor.shape != (rows, 96) or tensor.dtype != torch.float32
           for tensor in (cosine, sine)):
        return fallback("requires full FP32 cosine/sine [S,96]")
    query_weight = weights.norm_q._get_actual_weight()
    key_weight = weights.norm_k._get_actual_weight()
    if any(not isinstance(tensor, torch.Tensor) or tensor.shape != (128,) or tensor.dtype != torch.bfloat16
           for tensor in (query_weight, key_weight)):
        return fallback("requires BF16 norm weights [128]")
    if any(tensor.dtype != torch.bfloat16 for tensor in (query, key, value)):
        return fallback("requires BF16 QKV")
    tensors = (query, key, value, cosine, sine, query_weight, key_weight)
    if any(not tensor.is_cuda or tensor.device != query.device or not tensor.is_contiguous() or tensor.requires_grad for tensor in tensors):
        return fallback("requires contiguous same-device inference tensors")
    if not torch.cuda.get_device_properties(query.device).gcnArchName.startswith("gfx1201"):
        return fallback("requires gfx1201")
    query, key, value = (tensor.view(rows, 28, 128) for tensor in (query, key, value))
    return query, key, value, query_weight, key_weight, cosine, sine


def prepare_qk_rope(owner, weights, query, key, value, rotary_emb):
    if not enabled("LOSSLESS") or not owner.config.get("radeon_gfx1201_qk_norm_rope", owner.config.get("radeon_gfx1201_h3", False)):
        return None
    from aiter.ops.gfx1201 import qk_norm, rope

    inputs = _qk_rope_inputs(owner, weights, query, key, value, rotary_emb)
    if inputs is None:
        return None
    query, key, value, query_weight, key_weight, cosine, sine = inputs
    with record_function("radeon_h3.qk_norm_single_wave"):
        query = qk_norm(query, query_weight)
        key = qk_norm(key, key_weight)
    with record_function("radeon_h3.rope"):
        query, key = rope(query, key, cosine, sine)
    route = "QK_NORM_SINGLE_WAVE + ROPE"
    if getattr(owner, "_radeon_qk_active", None) != route:
        print(f"[RADEON OP][rank={os.environ.get('RANK', '0')}][ACTIVE] {route}; "
              "original BF16 norm rounding; asynchronous submission only",
              file=sys.stderr, flush=True)
        owner._radeon_qk_active = route
    return query, key, value


def _local_qkv_weight(module, group):
    if not isinstance(module, MMWeightTP) or module.tp_size != 2 or module.tp_group is not group or module.split_dim != "col":
        return None
    inner = unwrap_tp_weight(module)
    if type(inner) is not MMWeight or inner.has_lora_branch or inner.has_diff or getattr(inner, "bias", None) is not None:
        return None
    weight = inner._get_actual_weight()
    if not isinstance(weight, torch.Tensor) or tuple(weight.shape) != (5376, 3584) or weight.dtype != torch.bfloat16 or weight.stride() != (1, 5376):
        return None
    return weight.t()


def project_qkv(owner, weights, hidden_states):
    """Q/K/V projections. QKV_PROJECTION=1 runs all three in one Triton launch instead of three hipBLASLt GEMMs.
    A (E4M3, scales) tuple input selects the FP8 W8A8 kernel (RADEON_CORESW_FP8=qkv)."""
    if isinstance(hidden_states, tuple):
        from aiter.ops.gfx1201 import fp8_gemm
        inputs8, input_scales = hidden_states
        local = [_local_qkv_weight(module, owner.tp_group) for module in (weights.to_q, weights.to_k, weights.to_v)]
        if any(weight is None for weight in local):
            raise ValueError("RADEON_CORESW_FP8=qkv requires TP2 BF16 Q/K/V weights")
        quantized = [_fp8_weight(owner, name, weight) for name, weight in zip(("q", "k", "v"), local)]
        outputs = fp8_gemm.qkv8(inputs8, input_scales, [w for w, _ in quantized], [s for _, s in quantized])
        if os.environ.get("RADEON_FP8_SHADOW"):
            from aiter.ops.gfx1201.qkv_projection import qkv_projection
            rows = min(8192, inputs8.shape[0])
            x = (inputs8[:rows].float() * input_scales[:rows, None]).to(torch.bfloat16)
            _fp8_shadow(owner, "qkv(in=dequant)", tuple(o[:rows] for o in outputs), lambda: qkv_projection(x, *local))
        return outputs
    if enabled("QKV_PROJECTION") and hidden_states.ndim == 2 and hidden_states.dtype == torch.bfloat16 and hidden_states.is_contiguous():
        local = [_local_qkv_weight(module, owner.tp_group) for module in (weights.to_q, weights.to_k, weights.to_v)]
        if all(weight is not None for weight in local):
            from aiter.ops.gfx1201.qkv_projection import qkv_projection

            return qkv_projection(hidden_states, *local)
    return weights.to_q.apply(hidden_states), weights.to_k.apply(hidden_states), weights.to_v.apply(hidden_states)


def norm_rope_attention(owner, weights, hidden_states, pre_infer_out):
    """Projected QKV -> fused QK norm/RoPE/Sage quant -> attention core; None selects the unfused path."""
    if not enabled("LOSSLESS") or not owner.config.get("radeon_gfx1201_qk_norm_rope", owner.config.get("radeon_gfx1201_h3", False)):
        return None
    if not enabled("ATTENTION"):
        return None
    rank = os.environ.get("RANK", "0")
    if (pre_infer_out.sequence_parallel_state is not None or type(weights.calculate).__name__ != "RadeonCoreswAttnWeight"
            or (owner.use_fused_qkv and weights.has_fused_qkv)):
        if not getattr(owner, "_radeon_fused_qk_fallback", False):
            print(f"[RADEON OP][rank={rank}][FALLBACK] fused QK norm/RoPE attention: "
                  "requires gfx1201_sage_attn, no SP and separate Q/K/V projections", file=sys.stderr, flush=True)
            owner._radeon_fused_qk_fallback = True
        return None
    query, key, value = project_qkv(owner, weights, hidden_states)
    inputs = _qk_rope_inputs(owner, weights, query, key, value, pre_infer_out.rotary_emb)
    if inputs is None:
        return None
    query, key, value, query_weight, key_weight, cosine, sine = inputs
    with record_function("radeon_h3.norm_rope_attention"):
        from aiter.ops.gfx1201.sage_attention import gfx1201_norm_rope_attention

        output = gfx1201_norm_rope_attention(query, key, value, query_weight, key_weight, cosine, sine)
    if not getattr(owner, "_radeon_fused_qk_active", False):
        print(f"[RADEON OP][rank={rank}][ACTIVE] QK_NORM + ROPE + SAGE_QUANT fused HIP prepare -> attention core; "
              "bitwise equal to separate kernels; asynchronous submission only", file=sys.stderr, flush=True)
        owner._radeon_fused_qk_active = True
    return output.view(query.shape[0], 28 * 128)
