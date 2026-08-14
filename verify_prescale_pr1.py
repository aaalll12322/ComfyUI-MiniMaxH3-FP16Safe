# -*- coding: utf-8 -*-
# PR #1 verification: _ATTN_FIXED_SCALE 256 -> 16
# Reproduces, on a real model + real activations, the three claims behind the
# fixed-scale change (see DEVELOPMENT.md §3):
#   A) eps term quantification: rms_norm(q/s) = q/sqrt(ms_q + eps*s^2),
#      verifying that at s=256 the eps*s^2 term is NOT negligible vs ms(q)
#   B) monotone 1/s^2 convergence: for the same activation, smaller prescale
#      approaches the exact rms_norm(q) (no plateau)
#   C) overflow headroom: at s=16, out_proj's fp16 output max|.| vs 65504
#
# Usage:
#   python verify_prescale_pr1.py [--comfy <ComfyUI root>] [--plugin <plugin dir>]
#                                 [--model <ckpt>] [--lora <lora>] [small|medium]
# Defaults resolve under the ComfyUI root: models/diffusion_models/<fl2va fp8>,
# models/loras/minimax-h3/<turbo 4step>; adjust to what you have locally.
import sys, time, os, math
import numpy as np
import torch
import torch.nn.functional as F
import logging
logging.basicConfig(level=logging.ERROR)
sys.stdout.reconfigure(line_buffering=True)

def _parse_args():
    args = sys.argv[1:]
    def pick(flag, default):
        if flag in args:
            return args[args.index(flag) + 1]
        return default
    comfy_root = pick("--comfy", r"C:\ComfyUI")
    plugin_dir = pick("--plugin", os.path.join(os.path.dirname(os.path.abspath(__file__))))
    model = pick("--model", os.path.join(comfy_root, "models", "diffusion_models",
                                         "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"))
    lora = pick("--lora", os.path.join(comfy_root, "models", "loras", "minimax-h3",
                                       "minimax_h3_turbo_4step_pruned_comfyui.safetensors"))
    size = pick("--size", "small")
    return comfy_root, plugin_dir, model, lora, size

COMFY_ROOT, PLUGIN_DIR, MODEL, LORA, SIZE = _parse_args()
sys.path.insert(0, os.path.abspath(COMFY_ROOT))
sys.path.insert(0, PLUGIN_DIR)
import comfy.utils, comfy.sd, comfy.model_management
import comfy.quant_ops
import comfy.ldm.minimax.model as mm_model
import nodes as plugin

ck = comfy.quant_ops.ck

SCALES = [256.0, 64.0, 32.0, 16.0]
CAPTURED = []   # (self, x, rope_freqs)


def _capture_attn(orig_fwd):
    def wrapper(self, x, rope_freqs=None, transformer_options={}):
        CAPTURED.append((self, x.detach().clone(), rope_freqs))
        return orig_fwd(x, rope_freqs=rope_freqs, transformer_options=transformer_options)
    return wrapper


def _rmsnorm_manual(t, w, eps):
    """fp32 RMSNorm: out = t / sqrt(ms + eps) * w"""
    ms = t.float().pow(2).mean(-1, keepdim=True)
    return t.float() * torch.rsqrt(ms + eps) * w.float()


def _rope_manual(x, table, rot_dim):
    """split-half RoPE, fp32. table: [1,S,1,rot/2,2,2] = [[c,-s],[s,c]]"""
    xf = x.float()
    hd = xf.shape[-1]
    if rot_dim <= 0 or rot_dim > hd:
        return xf
    xt = xf[..., :rot_dim].reshape(*xf.shape[:-1], rot_dim // 2, 2)
    c = table[0, :, 0, :, 0, 0].reshape(1, xf.shape[1], 1, rot_dim // 2)
    s = table[0, :, 0, :, 1, 0].reshape(1, xf.shape[1], 1, rot_dim // 2)
    q0, q1 = xt[..., 0], xt[..., 1]
    n0 = c * q0 - s * q1
    n1 = s * q0 + c * q1
    rot = torch.stack([n0, n1], dim=-1).reshape(*xf.shape[:-1], rot_dim)
    return torch.cat([rot, xf[..., rot_dim:]], dim=-1)


def _attn_scale(self, x, scale, rope_freqs=None, fp32_ref=False):
    """s-series: replicates the plugin's _dit_attn_forward exactly (real fp16 path).
    fp32_ref=True: unprescaled fp32 reference (fp32 weights, no scaling, manual fp32 norm+rope)."""
    s = x.shape[0]
    if fp32_ref:
        x_h = x.float()
        qkv = F.linear(x_h, comfy.model_management.cast_to(self.qkv_proj.weight.float(), device=x.device),
                       comfy.model_management.cast_to(self.qkv_proj.bias.float(), device=x.device) if self.qkv_proj.bias is not None else None)
        q, k, v = qkv.split(self.heads * self.head_dim, dim=-1)
        v = v.view(s, self.heads, self.head_dim)
        q = q.view(1, s, self.heads, self.head_dim)
        k = k.view(1, s, self.heads, self.head_dim)
        q = _rmsnorm_manual(q, comfy.model_management.cast_to(self.q_norm.weight, device=x.device), self.q_norm.eps)
        k = _rmsnorm_manual(k, comfy.model_management.cast_to(self.k_norm.weight, device=x.device), self.k_norm.eps)
        if rope_freqs is not None:
            rot = rope_freqs.shape[-3] * 2
            q = _rope_manual(q, rope_freqs, rot)
            k = _rope_manual(k, rope_freqs, rot)
        q = q[0].transpose(0, 1).unsqueeze(0)
        k = k[0].transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(s, -1)
        proj = F.linear(out, comfy.model_management.cast_to(self.out_proj.weight.float(), device=x.device),
                        comfy.model_management.cast_to(self.out_proj.bias.float(), device=x.device) if self.out_proj.bias is not None else None)
        return proj.float(), None
    # ---- real fp16 path (plugin logic) ----
    if x.dtype == torch.float16:
        x_h = x * (1.0 / scale)
    else:
        x_h = (x * (1.0 / scale)).to(torch.float16)
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
                q, k = ck.rms_rope_split_half(q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
            else:
                ck.rms_rope_split_half_(q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
        else:
            q = self.q_norm(q[0]).unsqueeze(0)
            k = self.k_norm(k[0]).unsqueeze(0)
        q = q[0]
        k = k[0]
    else:
        q = self.q_norm(q.view(s, self.heads, self.head_dim))
        k = self.k_norm(k.view(s, self.heads, self.head_dim))
    q = q.transpose(0, 1).unsqueeze(0)
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)
    if q.dtype == torch.float16:
        out = F.scaled_dot_product_attention(q, k, v)
    else:
        out = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
    out = out.transpose(1, 2).reshape(s, -1)
    proj = self.out_proj(out)
    max_h16 = proj.abs().max().item()
    proj = proj.float() * scale
    if not torch.isfinite(proj).all():
        proj = F.linear(out.float(), comfy.model_management.cast_to(self.out_proj.weight.float(), device=x.device),
                        comfy.model_management.cast_to(self.out_proj.bias.float(), device=x.device) if self.out_proj.bias is not None else None).float() * scale
    return proj, max_h16


def load_and_patch():
    print("=== loading model + lora ===", flush=True)
    if not os.path.exists(MODEL):
        print(f"model not found: {MODEL}\npass --model / --comfy", flush=True)
        sys.exit(1)
    sd = comfy.utils.load_torch_file(MODEL)
    model = comfy.sd.load_diffusion_model_state_dict(sd, model_options={})
    del sd
    if os.path.exists(LORA):
        lsd = comfy.utils.load_torch_file(LORA)
        loaded = comfy.sd.load_lora_for_models(model, None, lsd, 1.0, 1.0)
        model = loaded[0] if isinstance(loaded, tuple) else loaded
        del lsd
        print("lora loaded", flush=True)
    diff = model.model.diffusion_model
    print("=== plugin patch (fp16 compute) ===", flush=True)
    node = plugin.MiniMaxH3FP16Safe()
    model = node.patch(model)[0]   # v6.6.0: patch 返回 clone, 必须接收返回值
    diff = model.model.diffusion_model
    import types as _t
    n = 0
    for m in diff.modules():
        if isinstance(m, mm_model.Attention):
            orig = m.forward
            m.forward = _t.MethodType(_capture_attn(orig), m)
            n += 1
    print(f"capture wrapper: {n} Attention layers", flush=True)
    return diff


def run(diff, video_shape, audio_t, text_len, ts):
    T, H, W = video_shape
    torch.manual_seed(0)
    audio = torch.randn(1, 32, 2, T * 4, device='cuda', dtype=torch.float16) * 0.5
    ctx = torch.randn(1, text_len, 5120, device='cuda', dtype=torch.float16)
    ctx[0, :, ::113] += 8000.0
    payload = {"audio_scale": 1.0, "layout": None}
    x = torch.randn(1, 24, T, H, W, device='cuda', dtype=torch.float16)
    t = torch.tensor([ts], device='cuda', dtype=torch.float32)
    seq = T * H * W + text_len + T * 8
    print(f"seq ~ {seq} (capturing real activations)", flush=True)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        o = diff([x, audio], t, ctx, transformer_options={}, minimax_payload=payload)
    torch.cuda.synchronize()
    print(f"forward done: {time.time()-t0:.1f}s, captured {len(CAPTURED)} layers", flush=True)


def main():
    torch.set_grad_enabled(False)
    diff = load_and_patch()
    if SIZE == "medium":
        run(diff, (30, 22, 38), 30, 634, 500.0)
    else:
        run(diff, (24, 16, 16), 24, 634, 500.0)
    print(f"captured {len(CAPTURED)} layers", flush=True)

    # ===== A) eps term quantification =====
    print("\n===== A) RMSNorm eps*s^2 term vs ms(q) (qk_norm_eps=1e-5) =====", flush=True)
    print(f"{'layer':>5} {'ms(q) med':>9} {'ms(q) min':>9} | {'eps*256^2/ms':>12} {'eps*64^2/ms':>11} {'eps*16^2/ms':>11}", flush=True)
    ms_list = []
    for idx, (self_m, x, rope) in enumerate(CAPTURED):
        xf = x.float()
        qkv = F.linear(xf, comfy.model_management.cast_to(self_m.qkv_proj.weight.float(), device=x.device),
                       comfy.model_management.cast_to(self_m.qkv_proj.bias.float(), device=x.device) if self_m.qkv_proj.bias is not None else None)
        q, _, _ = qkv.split(self_m.heads * self_m.head_dim, dim=-1)
        qf = q.reshape(-1, q.shape[-1])
        ms = (qf * qf).mean(-1)
        ms_list.append(ms)
        med = ms.median().item()
        mn = ms.min().item()
        eps = self_m.q_norm.eps
        print(f"{idx:>5} {med:>9.2f} {mn:>9.2f} | {eps*256*256/med:>12.4f} {eps*64*64/med:>11.4f} {eps*16*16/med:>11.4f}", flush=True)
    med_all = torch.cat(ms_list).median().item()
    print(f"summary: ms(q) all-layer median = {med_all:.2f}", flush=True)

    # ===== B) eps-term math check (pure fp32, rope-independent) =====
    print("\n===== B) eps term: rms_norm(q/s) vs rms_norm(q) (fp32, no rope) =====", flush=True)
    print(f"{'layer':>5} {'ref|q|max':>10} | " + " | ".join([f"s={s:>4.0f} rel" for s in SCALES]) + " | 1/s^2 mono", flush=True)
    rel_by_scale = {s: [] for s in SCALES}
    for idx, (self_m, x, rope) in enumerate(CAPTURED):
        xf = x.float()
        qkv = F.linear(xf, comfy.model_management.cast_to(self_m.qkv_proj.weight.float(), device=x.device),
                       comfy.model_management.cast_to(self_m.qkv_proj.bias.float(), device=x.device) if self_m.qkv_proj.bias is not None else None)
        q, _, _ = qkv.split(self_m.heads * self_m.head_dim, dim=-1)
        qf = q.reshape(-1, q.shape[-1])
        eps = self_m.q_norm.eps
        ms = (qf * qf).mean(-1, keepdim=True)
        ref = qf * torch.rsqrt(ms + eps)
        ref_max = ref.abs().max().item()
        line = f"{idx:>5} {ref_max:>10.2f} | "
        prev = None
        mono = True
        for s in SCALES:
            q_s = qf * torch.rsqrt(ms + eps * s * s)
            rmax = max(ref.abs().max().item(), 1e-6)
            rel = (q_s - ref).abs().max().item() / rmax
            rel_by_scale[s].append(rel)
            line += f" {rel:.2e}"
            if prev is not None and rel > prev * 1.25:
                mono = False
            prev = rel
        line += f" | {'YES' if mono else 'NO'}"
        print(line, flush=True)
    print("\nsummary B (mean norm-layer rel err):", flush=True)
    for s in SCALES:
        v = np.mean(rel_by_scale[s])
        ratio = v / np.mean(rel_by_scale[SCALES[0]]) if np.mean(rel_by_scale[SCALES[0]]) > 0 else float('nan')
        print(f"  s={s:>5.0f}: mean rel = {v:.3e}  (vs s=256: {ratio:.1f}x)", flush=True)

    # ===== B2) real fp16 chain (ck rope): pairwise scale differences =====
    print("\n===== B2) real fp16 chain (ck rope): scale pairwise diffs =====", flush=True)
    print(f"{'layer':>5} {'|d(256,16)|':>12} {'|d(64,16)|':>11} {'|d(32,16)|':>11} {'|d(16)|max':>10} {'d(256,16)/max':>13}", flush=True)
    for idx, (self_m, x, rope) in enumerate(CAPTURED):
        outs = {}
        for s in SCALES:
            out, _ = _attn_scale(self_m, x, s, rope_freqs=rope, fp32_ref=False)
            outs[s] = out
        d256 = (outs[256.0] - outs[16.0]).abs().max().item()
        d64 = (outs[64.0] - outs[16.0]).abs().max().item()
        d32 = (outs[32.0] - outs[16.0]).abs().max().item()
        omax = outs[16.0].abs().max().item()
        print(f"{idx:>5} {d256:>12.3f} {d64:>11.3f} {d32:>11.3f} {omax:>10.2f} {d256/max(omax,1e-6):>13.4f}", flush=True)

    # ===== C) overflow headroom =====
    print("\n===== C) overflow headroom: out_proj fp16 output max|.| (limit 65504) =====", flush=True)
    print(f"{'layer':>5} {'s=256':>10} {'s=64':>10} {'s=32':>10} {'s=16':>10} {'s=16 headroom':>13}", flush=True)
    worst = 0.0
    for idx, (self_m, x, rope) in enumerate(CAPTURED):
        vals = []
        for s in SCALES:
            _, max_h16 = _attn_scale(self_m, x, s, rope_freqs=rope, fp32_ref=False)
            vals.append(max_h16 if max_h16 is not None else 0.0)
        worst = max(worst, vals[-1])
        print(f"{idx:>5} " + "".join([f" {v:>10.1f}" for v in vals]) + f" {65504/vals[-1]:>13.1f}x", flush=True)
    print(f"worst s=16: max|.| = {worst:.1f} -> headroom {65504/worst:.1f}x  (>1 safe; plugin isfinite fuse as backstop)", flush=True)
    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
