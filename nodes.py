# ComfyUI-MiniMaxH3-FP16Safe v6.3.0
#
# Make MiniMax H3 (comfy/ldm/minimax, PR #15224) numerically stable in fp16
# compute on GPUs without bf16/fp8 hardware (V100 sm_70 etc.), at near-fp16
# speed. MiniMax H3 officially supports only bf16/fp32 -- its activations
# (residual stream, gated-silu products, fc2 outputs) genuinely reach ~5e5,
# far beyond the fp16 limit (+-65504). Forcing fp16 compute => NaN => black
# frames. This plugin keeps the fp32 residual stream, but runs every big
# matmul on fp16 Tensor Cores with power-of-2 scaling compensations:
#
#   * residual stream: fp32 accumulation (_dit_block_forward)
#   * RMSNorm (210x): fp32 compute, I/O dtype preserved
#   * attention: always-fp16 SDPA via power-of-2 input scaling
#     (q/k restored by RMSNorm homogeneity, v unscaled by output multiply;
#     internal fp32 accumulate + max-subtract keep huge logits stable)
#   * MLP (v6): FULLY fp16 -- fc1 output (measured max ~585) stays fp16,
#     gated-silu scales the gate branch by a power of 2 so the product stays
#     bounded (~2340), fc2 runs fp16 with input scaling; NO fp32 intermediate
#     tensors at all (kills the ~850GB/step cast bandwidth on long seqs)
#   * long sequences: chunked MLP so activation peak stays ~9GB regardless of
#     seq length (avoids lowvram weight-eviction thrash on 16GB cards)
#   * video VAE: fp16 stream, only norms/scores/silu upcast to fp32
#
# Verified magnitudes (real model, extreme timestep):
#   max|P@V|=706  max|out_proj|~39k  max|fc1|=585  max|silu_act|=59k
#   max|fc2|=501k  -> every stage is either fp16-safe or 2-power-scaled.

import math

import torch
import comfy.model_management
from comfy.ldm.modules import attention as attn_mod

# --- defensive imports ---
try:
    import comfy.ops as _comfy_ops
except Exception:
    _comfy_ops = None
try:
    import comfy.rmsnorm as _comfy_rmsnorm
except Exception:
    _comfy_rmsnorm = None
try:
    import comfy.quant_ops as _comfy_quant_ops
except Exception:
    _comfy_quant_ops = None
try:
    import comfy.ldm.minimax.model as mm_model
    import comfy.ldm.minimax.vae as mm_vae
    MINIMAX_AVAILABLE = True
    _IMPORT_ERR = None
except Exception as e:
    mm_model = None
    mm_vae = None
    MINIMAX_AVAILABLE = False
    _IMPORT_ERR = e

ck = getattr(_comfy_quant_ops, "ck", None)
_ORIG_OPT = getattr(attn_mod, "optimized_attention", None)


def _rms_norm_fallback(x, weight, eps):
    var = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * weight


def _fp32_opt(q, k, v, heads, *args, **kwargs):
    """fp32 attention (used by the VAE path only; the DiT uses fp16 SDPA)."""
    if _ORIG_OPT is not None:
        qf, kf, vf = q.float(), k.float(), v.float()
        kwargs["attn_precision"] = torch.float32
        return _ORIG_OPT(qf, kf, vf, heads, *args, **kwargs)
    b, _, _, dh = q.shape
    qf, kf, vf = (t.reshape(-1, t.shape[-2], dh).float() for t in (q, k, v))
    sim = torch.einsum("b i d, b j d -> b i j", qf, kf) * (dh ** -0.5)
    out = torch.einsum("b i j, b j d -> b i d", torch.softmax(sim, dim=-1), vf)
    return out.reshape(b, heads, -1, dh)


class _FP32RMSNorm(torch.nn.Module):
    """RMSNorm computing in fp32, keeping I/O dtype. Exposes .weight/.eps/.bias."""
    def __init__(self, orig):
        super().__init__()
        self.orig = orig
        self.weight = orig.weight
        self.eps = orig.eps
        if hasattr(orig, "bias"):
            self.bias = orig.bias

    def forward(self, x, *args, **kwargs):
        return self.orig(x.float(), *args, **kwargs).to(x.dtype)


class _FP32LinearWrap(torch.nn.Module):
    """Linear always computing in fp32 (protects the Qwen text path)."""
    def __init__(self, orig):
        super().__init__()
        self.orig = orig

    def forward(self, x, *args, **kwargs):
        return self.orig(x.float(), *args, **kwargs)


def _is_rmsnorm(mod):
    return type(mod).__module__ == "comfy.ops" and type(mod).__name__ == "RMSNorm"


def _wrap_rmsnorms(root):
    replaced = 0
    for name, mod in list(root.named_modules()):
        if _is_rmsnorm(mod):
            parent_name, attr = name.rsplit(".", 1) if "." in name else ("", name)
            parent = root.get_submodule(parent_name) if parent_name else root
            setattr(parent, attr, _FP32RMSNorm(mod))
            replaced += 1
    return replaced


def _fp16_safe(h):
    """Downcast to fp16 ONLY when it cannot overflow; else keep fp32."""
    if h.dtype == torch.float16:
        return h
    try:
        if h.abs().max().item() <= 60000.0:
            return h.to(torch.float16)
    except Exception:
        pass
    return h


def _fp16_scaled(h, threshold=60000.0):
    """V3: downcast to fp16 ALWAYS, scaling by a power of 2 when needed.

    Returns (fp16_tensor, scale) with scale = 2**k (k>=0). When |h| exceeds the
    fp16-safe threshold, h is divided by scale before the cast, so the fp16
    matmuls never overflow even for huge residuals. The scale is a power of two,
    so the division is exact (no precision loss); the caller compensates:
      * qkv_proj is linear -> q' = q / scale, and RMSNorm(q') == RMSNorm(q)
        (normalization cancels the scaling), so q/k are restored for free;
      * v does NOT pass RMSNorm -> the SDPA output must be multiplied back by scale.
    """
    if h.dtype == torch.float16:
        return h, 1.0
    try:
        m = h.abs().max().item()
        if m <= threshold:
            return h.to(torch.float16), 1.0
        k = int(math.ceil(math.log2(m / threshold)))
        s = float(1 << k)
        return (h / s).to(torch.float16), s
    except Exception:
        pass
    return h, 1.0  # caller must fall back to fp32 when dtype is still fp32


def _qkv_scale_check(self, x, x_scale):
    """V4: qkv_proj output can exceed fp16 range even when its input h is small
    (the linear layer amplifies by the weight norm). Detect an fp16 overflow of
    q/k/v and redo qkv_proj with a larger power-of-2 scale on x.
    V4: single .item() (v3 did 3) -- q/k/v share one qkv tensor, so one abs-max
    over the full qkv covers all three. Halves the per-layer GPU sync count.
    Returns (q, k, v in fp16, final x_scale)."""
    qkv = self.qkv_proj(x)
    try:
        if qkv.abs().max().item() > 60000.0 and x_scale < 1e6:
            qkv_max = qkv.abs().max().item()
            extra = int(math.ceil(math.log2(qkv_max / 60000.0)))
            x_scale2 = x_scale * float(1 << extra)
            xh2 = (x.float() / x_scale2).to(torch.float16)
            qkv = self.qkv_proj(xh2)
            x_scale = x_scale2
    except Exception:
        pass
    q, k, v = qkv.split(self.heads * self.head_dim, dim=-1)
    return q, k, v, x_scale


# ---- DiT Attention (v6.3): ALWAYS-fp16 SDPA with FIXED power-of-2 scale ----
# Zero per-layer scans. x (fp16 or fp32) is always divided by 256 (2^8, exact)
# before qkv_proj, so qkv output is bounded by |x|/256 * ||W_qkv||. For |x| up
# to 60000 (fp16 stream bound) and realistic weight norms this stays ~700, and
# even |x|=1e6 (fp32, extreme) stays fp16-safe. q/k are restored exactly by
# RMSNorm homogeneity (logits see O(1) values); v stays scaled through the
# linear chain SDPA -> out_proj and is unscaled in fp32 after out_proj (small
# [s, 3072] tensor). Bounds: |P@v| <= 706 (measured) -> scaled 2.76, out_proj
# out ~39k/256 = 152 -> fp16 can never overflow.
_ATTN_FIXED_SCALE = 256.0   # 2^8


def _dit_attn_forward(self, x, rope_freqs=None, transformer_options={}):
    s = x.shape[0]
    if x.dtype == torch.float16:
        x_h = x * (1.0 / _ATTN_FIXED_SCALE)              # exact in fp16
    else:
        x_h = (x * (1.0 / _ATTN_FIXED_SCALE)).to(torch.float16)
    qkv = self.qkv_proj(x_h)
    q, k, v = qkv.split(self.heads * self.head_dim, dim=-1)
    v = v.view(s, self.heads, self.head_dim)
    if rope_freqs is not None:
        q = q.view(1, s, self.heads, self.head_dim)
        k = k.view(1, s, self.heads, self.head_dim)
        qw = comfy.model_management.cast_to(self.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(self.k_norm.weight, device=x.device)
        rot = rope_freqs.shape[-3] * 2
        if ck is not None:
            if comfy.model_management.in_training:
                q, k = ck.rms_rope_split_half(
                    q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
            else:
                ck.rms_rope_split_half_(
                    q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
        else:
            q = self.q_norm(q[0]).unsqueeze(0)
            k = self.k_norm(k[0]).unsqueeze(0)
            print("[MiniMaxH3-FP16Safe] WARNING: comfy_kitchen missing, rope skipped.")
        q = q[0]
        k = k[0]
    else:
        q = self.q_norm(q.view(s, self.heads, self.head_dim))
        k = self.k_norm(k.view(s, self.heads, self.head_dim))
    # [1, heads, s, hd]; q/k restored to O(1) by RMSNorm, v still /256
    q = q.transpose(0, 1).unsqueeze(0)
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)
    if q.dtype == torch.float16:
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)   # fp16 Tensor-Core SDPA
    else:
        out = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
    out = out.transpose(1, 2).reshape(s, -1)                              # [s, heads*hd]
    proj = self.out_proj(out)                                             # fp16, bounded <= ~152
    proj = proj.float() * _ATTN_FIXED_SCALE                               # unscale v in fp32
    if not torch.isfinite(proj).all():                                    # insurance (never fires)
        proj = self.out_proj(out.float()).float() * _ATTN_FIXED_SCALE
    return proj                                                           # fp32 for the stream


# ---- DiT block / refiner block: fp32 residual stream, fp16 inner compute ----
import os as _os
import time as _time
_ENV_PROFILE = _os.environ.get("MINIMAXH3_PROFILE", "").strip().lower() in ("1", "true", "yes", "on")
_PROFILE = _ENV_PROFILE          # 进程内当前开关状态, 每次 patch 重新按 env/节点开关决定
_prof = {"attn": 0.0, "mlp": 0.0, "other": 0.0, "n": 0}


def _prof_block(i, kind, dt_attn, dt_mlp, dt_other):
    _prof["attn"] += dt_attn
    _prof["mlp"] += dt_mlp
    _prof["other"] += dt_other
    _prof["n"] += 1
    idx = i if i >= 0 else _prof["n"]        # fallback: call counter
    if idx in (2, 5, 10, 25, 50, 100):
        alloc = torch.cuda.memory_allocated() / 2**30
        print(f"[MiniMaxH3-FP16Safe][PROF] block {idx} ({kind}): attn累计={_prof['attn']:.1f}s "
              f"mlp累计={_prof['mlp']:.1f}s other累计={_prof['other']:.1f}s "
              f"当前allocated={alloc:.1f}GB", flush=True)


def _dit_block_forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}):
    _t = _time.time()
    _mss = mm_model._mod_scale_shift
    _mg = mm_model._mod_gate
    x = x.float()
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
    h = _mss(self.norm1(x), shift_msa, scale_msa, mod_segments)
    _t1 = _time.time()
    # V4: _fp16_safe pre-check kept -- it downcasts safe residuals to fp16 here
    # so q_norm/k_norm (_FP32RMSNorm) output stays fp16 and SDPA stays on the
    # fast fp16 path; removing it made norm output fp32 -> SDPA fp32 -> NaN.
    attn_out = self.attn(_fp16_safe(h), rope_freqs=rope_freqs,
                         transformer_options=transformer_options).float()
    _t2 = _time.time()
    x = _mg(x, gate_msa, attn_out, mod_segments)
    h = _mss(self.norm2(x), shift_mlp, scale_mlp, mod_segments)
    mlp_out = self.mlp(_fp16_safe(h)).float()
    _t3 = _time.time()
    out = _mg(x, gate_mlp, mlp_out, mod_segments)
    _t4 = _time.time()
    if _PROFILE:
        idx = getattr(self, "_dbg_index", -1)
        _prof_block(idx, "dit", _t2 - _t1, _t3 - _t2, (_t1 - _t) + (_t4 - _t3))
    return out


def _refiner_block_forward(self, x, transformer_options={}):
    x = x.float()
    a = self.attn(_fp16_safe(self.norm1(x)), transformer_options=transformer_options).float()
    x = x.add_(a)
    m = self.mlp(_fp16_safe(self.norm2(x))).float()
    return x.add_(m)


# ---- DiT MLP (v6): fully-fp16 MLP with power-of-2 gate scaling ----
# Measured magnitudes (real model, extreme timestep): max|fc1|=585 (fp16-safe),
# max|silu_act|=59k (close to fp16 limit!), max|fc2|=501k (~8.5x input).
# v5 chunking fixed the VRAM cliff (110->97s) but the per-layer fp32 cast of the
# fc1 output (5.6GB fp16 -> 11.3GB fp32, x50 layers/step) is ~30s of pure extra
# bandwidth. v6 keeps the WHOLE MLP in fp16:
#   * fc1 out (max ~585) stays fp16;
#   * gated-silu: scale the gate branch b by a power of 2 so that
#     act = silu(a) * (b/2^k) stays well inside fp16 (target act <= ~585*4);
#   * fc2 runs fp16; its output ~8.5x input, so scale the act input when needed;
#     unscale everything in fp32 at the end (small [s,3072] tensor).
#   * when the residual was scaled (x_scale != 1, rare) silu must see true
#     values -> fall back to the chunked-fp32 path (correct, slow, rare).
_MLP_CHUNK = 16384            # rows per chunk


def _mlp_chunked_fp32(self, x_h, x_scale, s):
    """Correct-but-slow fp32 chunked path (used when x was power-of-2 scaled)."""
    outs = []
    for i in range(0, s, _MLP_CHUNK):
        xc = x_h[i:i + _MLP_CHUNK]
        y = self.fc1(xc).float()
        if x_scale != 1.0:
            y.mul_(x_scale)
        a, b = y.chunk(2, dim=-1)
        act = torch.nn.functional.silu(a)
        act.mul_(b)
        del y
        try:
            amax = act.abs().max().item()
        except Exception:
            amax = 0.0
        if amax > 4000.0:
            sc = 1 << int(math.ceil(math.log2(max(amax / 4000.0, 1.0))))
            outs.append(self.fc2((act / sc).to(torch.float16)).float() * sc)
        else:
            outs.append(self.fc2(act.to(torch.float16)).float())
    return torch.cat(outs, dim=0)


def _mlp_forward(self, x):
    s = x.shape[0]
    x_h, x_scale = _fp16_scaled(x)
    if x_scale != 1.0:
        # rare: silu is NOT linear, must see true values -> fp32 chunked path
        out = _mlp_chunked_fp32(self, x_h, x_scale, s)
        if not torch.isfinite(out).all():
            outs = []
            for i in range(0, s, _MLP_CHUNK):
                xc = x_h[i:i + _MLP_CHUNK]
                y = self.fc1(xc).float()
                if x_scale != 1.0:
                    y.mul_(x_scale)
                a, b = y.chunk(2, dim=-1)
                act = torch.nn.functional.silu(a)
                act.mul_(b)
                del y
                outs.append(self.fc2(act).float())
            out = torch.cat(outs, dim=0)
        return out
    # fast path: fully fp16 MLP, ZERO per-chunk syncs (v6.2).
    # Fixed conservative scales (no .item() scans):
    #   * bs=16 on the gate branch: measured max|fc1|=585 -> act = silu(a)*(b/16)
    #     <= 585*36.6 = 21.4k (fp16-safe), with ~3.4x headroom for outliers;
    #   * fs=8 on the act input: fc2 out ~8.5x input -> <= 21.4k/8*8.5 = 22.7k.
    # Both are powers of 2 (exact in fp16); unscale in fp32 at the end. The
    # isfinite fuse below catches any unexpected overflow (rare fp32 retry).
    _BS = 16.0
    _FS = 8.0
    outs = []
    for i in range(0, s, _MLP_CHUNK):
        xc = x_h[i:i + _MLP_CHUNK]
        y16 = self.fc1(xc)                       # fp16 [c, 2I], max ~585 (safe)
        a, b = y16.chunk(2, dim=-1)
        act = torch.nn.functional.silu(a) * (b * (1.0 / _BS))        # fp16 gated-silu
        outs.append(self.fc2(act * (1.0 / _FS)).float() * _FS * _BS) # fp16 fc2, fp32 unscale
    out = torch.cat(outs, dim=0)
    if not torch.isfinite(out).all():            # fuse: rare fp32 chunked retry
        outs = []
        for i in range(0, s, _MLP_CHUNK):
            xc = x_h[i:i + _MLP_CHUNK]
            y = self.fc1(xc).float()
            if x_scale != 1.0:
                y.mul_(x_scale)
            a, b = y.chunk(2, dim=-1)
            act = torch.nn.functional.silu(a)
            act.mul_(b)
            del y
            outs.append(self.fc2(act).float())
        out = torch.cat(outs, dim=0)
    return out


# ---- Video VAE (same as v1: fp16 stream, fp32 norms/scores/silu) ----
def _vae_ff_forward(self, x):
    gate, x = self.w1(x).chunk(2, dim=-1)
    act = torch.nn.functional.silu(gate.float()) * x.float()
    return self.w2(act.clamp(-65504.0, 65504.0).to(x.dtype))


def _vae_attn_forward(self, x, rotary_pos_emb=None):
    batch_size, seq_len, _ = x.shape
    qkv = self.to_qkv(x)
    qkv = qkv.view(batch_size, seq_len, -1, 3 * self.dim_head)
    query, key, value = torch.chunk(qkv, 3, dim=-1)
    if _comfy_rmsnorm is not None:
        query = _comfy_rmsnorm.rms_norm(query.float(), self.norm_q.weight, self.norm_q.eps).to(x.dtype)
        key = _comfy_rmsnorm.rms_norm(key.float(), self.norm_k.weight, self.norm_k.eps).to(x.dtype)
    else:
        query = _rms_norm_fallback(query.float(), self.norm_q.weight, self.norm_q.eps).to(x.dtype)
        key = _rms_norm_fallback(key.float(), self.norm_k.weight, self.norm_k.eps).to(x.dtype)
    if rotary_pos_emb is not None and ck is not None:
        rot = rotary_pos_emb.shape[-3] * 2
        query[..., :rot], key[..., :rot] = ck.apply_rope_split_half(
            query[..., :rot], key[..., :rot], rotary_pos_emb)
    out = _fp32_opt(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
                    self.heads, skip_reshape=True).nan_to_num_(0.0).to(x.dtype)
    return self.to_out(out)


def _vae_tb_forward(self, x, rotary_pos_emb=None):
    cti = _comfy_ops.cast_to_input if _comfy_ops is not None else (lambda w, i: w.to(i.dtype))
    if _comfy_rmsnorm is not None:
        n1 = _comfy_rmsnorm.rms_norm(x.float(), self.norm1.weight, self.norm1.eps).to(x.dtype)
    else:
        n1 = _rms_norm_fallback(x.float(), self.norm1.weight, self.norm1.eps).to(x.dtype)
    a = self.attn(n1, rotary_pos_emb)
    x = x.addcmul_(a, cti(self.scale1, x))
    if _comfy_rmsnorm is not None:
        n2 = _comfy_rmsnorm.rms_norm(x.float(), self.norm2.weight, self.norm2.eps).to(x.dtype)
    else:
        n2 = _rms_norm_fallback(x.float(), self.norm2.weight, self.norm2.eps).to(x.dtype)
    f = self.ff(n2)
    return x.addcmul_(f, cti(self.scale2, x))


def _is_minimax_dit(inner):
    if inner is None or not hasattr(inner, "modules"):
        return False
    if any(type(m).__name__ == "MiniMaxH3Model" for m in inner.modules()):
        return True
    for m in inner.modules():
        if all(hasattr(m, a) for a in ("qkv_proj", "out_proj", "q_norm", "k_norm")):
            return True
    return False


def _is_minimax_vae(inner):
    if inner is None or not hasattr(inner, "modules"):
        return False
    if any(type(m).__name__ == "MiniMaxH3VideoVAE" for m in inner.modules()):
        return True
    for m in inner.modules():
        if all(hasattr(m, a) for a in ("to_qkv", "norm_q", "to_out")):
            return True
    return False


class MiniMaxH3FP16Safe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"model": ("MODEL",)},
            "optional": {
                "vae": ("VAE",),
                "fix_vae": ("BOOLEAN", {"default": True}),
                "debug_nan": ("BOOLEAN", {"default": False}),
                "profile": ("BOOLEAN", {"default": False, "tooltip": "打印每阶段耗时(block 2/5/10/25/50)与显存, 定位速度瓶颈"}),
            },
        }

    RETURN_TYPES = ("MODEL", "VAE")
    FUNCTION = "patch"
    CATEGORY = "MiniMaxH3"

    def patch(self, model, vae=None, fix_vae=True, debug_nan=False, profile=False):
        if not MINIMAX_AVAILABLE:
            print("[MiniMaxH3-FP16Safe] comfy.ldm.minimax backend not found (%s). "
                  "Update ComfyUI to a build that includes PR #15224." % _IMPORT_ERR)
            return (model, vae)

        # ---- 强制 fp16 计算: MiniMaxH3 官方 supported=[bf16,fp32] 不含 fp16,
        # V100 无 bf16 硬件 -> aki 兜底 cast 成 fp32 (慢 4x, 无 Tensor Core)。
        # 必须在 patch 时强制 fp16, 否则 qkv/out/fc1/fc2 全是 fp32 matmul。
        try:
            model.set_model_compute_dtype(torch.float16)
            print("[MiniMaxH3-FP16Safe] forced compute dtype -> fp16 (weights cast to fp16, Tensor Core ON)")
        except Exception as e:
            print("[MiniMaxH3-FP16Safe] WARNING: set_model_compute_dtype(fp16) failed: %s" % e)

        # ---- DiT (UNet) ----
        global _PROFILE, _prof
        # 节点开关可真正关闭; 只有环境变量显式开启时 profile 才强制打开
        _PROFILE = _ENV_PROFILE or bool(profile)
        _prof = {"attn": 0.0, "mlp": 0.0, "other": 0.0, "n": 0}   # 每次 patch 重置累计, 避免跨多次运行叠加
        inner = getattr(model, "model", None)
        if _is_minimax_dit(inner):
            mm_model.Attention.forward = _dit_attn_forward
            mm_model.DiTBlock.forward = _dit_block_forward
            mm_model.RefinerBlock.forward = _refiner_block_forward
            mm_model.MLP.forward = _mlp_forward
            target = getattr(inner, "diffusion_model", None) or inner
            n = _wrap_rmsnorms(target)
            wrapped_cond = 0
            for m in target.modules():
                if hasattr(m, "condition_proj") and isinstance(m.condition_proj, torch.nn.Module):
                    m.condition_proj = _FP32LinearWrap(m.condition_proj)
                    wrapped_cond += 1
            # 总是给 block 打索引 (debug_nan 与 profile 都依赖它)
            try:
                for m in target.modules():
                    if isinstance(m, mm_model.MiniMaxH3Model):
                        for i, blk in enumerate(m.blocks):
                            blk._dbg_index = i
                        break
            except Exception:
                pass
            print("[MiniMaxH3-FP16Safe][V6.3-FIXEDATTN] DiT patched: fp32 residual stream + fp16 SDPA attention "
                  "(fixed /256 scale, zero scans) + fully-fp16 MLP (fixed-scale) + %d RMSNorm(s) + %d condition_proj. "
                  "(profile=%s)" % (n, wrapped_cond, _PROFILE))

            if debug_nan:
                self._install_nan_debug(mm_model, target)
        else:
            print("[MiniMaxH3-FP16Safe] MODEL is not MiniMax H3; DiT left unchanged.")

        # ---- Video VAE ----
        if vae is not None and fix_vae:
            vinner = getattr(vae, "first_stage_model", None) or vae
            if _is_minimax_vae(vinner):
                mm_vae.Attention.forward = _vae_attn_forward
                mm_vae.TransformerBlock.forward = _vae_tb_forward
                mm_vae.FeedForward.forward = _vae_ff_forward
                print("[MiniMaxH3-FP16Safe] Video VAE patched: attention + transformer-block norms + gated-silu upcast to fp32.")
            else:
                print("[MiniMaxH3-FP16Safe] VAE is not MiniMax H3 video VAE; left unchanged.")

        return (model, vae)

    @staticmethod
    def _install_nan_debug(mm_model, target=None):
        base = mm_model.DiTBlock.forward
        state = {"reported": False}
        if target is not None:
            try:
                for m in target.modules():
                    if isinstance(m, mm_model.MiniMaxH3Model):
                        for i, blk in enumerate(m.blocks):
                            blk._dbg_index = i
                        break
            except Exception:
                pass

        def checking(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}):
            idx = getattr(self, "_dbg_index", "?")
            if not state["reported"] and not torch.isfinite(x).all():
                print("[MiniMaxH3-FP16Safe][DEBUG] DiTBlock %s INPUT already non-finite "
                      "(finite %.4f) -> NaN originates BEFORE the blocks (embedding/refiner/context)" % (
                          idx, torch.isfinite(x).float().mean().item()))
                state["reported"] = True
            out = base(self, x, t_emb, mod_segments, rope_freqs, transformer_options)
            if not state["reported"] and not torch.isfinite(out).all():
                print("[MiniMaxH3-FP16Safe][DEBUG] non-finite value first seen at DiTBlock %s "
                      "(finite ratio %.4f)" % (idx, torch.isfinite(out).float().mean().item()))
                state["reported"] = True
            return out

        mm_model.DiTBlock.forward = checking
        print("[MiniMaxH3-FP16Safe] debug_nan enabled.")


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3FP16Safe": MiniMaxH3FP16Safe,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3FP16Safe": "MiniMax H3 FP16 Safe",
}
