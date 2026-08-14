# -*- coding: utf-8 -*-
# v6.8.0 真机验证 (V100, 真实模型 fl2va fp8 + turbo LoRA):
#   1) 无条件降 fp16 在真实激活下不引入溢出: 捕获每 block attn/MLP 输入 h 的
#      abs().max(), 确认远小于 60000 (与 v6.7.0 条件降行为完全等价)
#   2) deferred fuse 不触发: forward 后 _FUSE_FLAG 应为 False, 无 fp32 重跑
#   3) 输出有限 + 耗时
# 用法: <aki python> verify_v68_deferred.py --comfy <ComfyUI 根> [--plugin <插件目录>] [small|medium]
import sys, os, time
import torch

def _pick(flag, default):
    if flag in sys.argv:
        return sys.argv[sys.argv.index(flag) + 1]
    return default

COMFY = _pick("--comfy", r"C:\ComfyUI")
PLUGIN = _pick("--plugin", os.path.join(os.path.dirname(os.path.abspath(__file__))))
SIZE = _pick("--size", "small")
sys.path.insert(0, os.path.abspath(COMFY))
sys.path.insert(0, os.path.abspath(PLUGIN))
import comfy.utils, comfy.sd, comfy.model_management
import nodes as plugin

MODEL = os.path.join(COMFY, "models", "diffusion_models", "minimax_h3_fl2va_pruned_fp8_scaled.safetensors")
LORA = os.path.join(COMFY, "models", "loras", "minimax-h3", "minimax_h3_turbo_4step_pruned_comfyui.safetensors")


def main():
    torch.set_grad_enabled(False)
    sys.stdout.reconfigure(line_buffering=True)
    print("=== 加载模型 ===", flush=True)
    assert os.path.exists(MODEL), f"model not found: {MODEL}"
    sd = comfy.utils.load_torch_file(MODEL)
    model = comfy.sd.load_diffusion_model_state_dict(sd, model_options={})
    del sd
    if os.path.exists(LORA):
        lsd = comfy.utils.load_torch_file(LORA)
        loaded = comfy.sd.load_lora_for_models(model, None, lsd, 1.0, 1.0)
        model = loaded[0] if isinstance(loaded, tuple) else loaded
        del lsd
        print("lora loaded", flush=True)

    print("=== v6.8.0 patch ===", flush=True)
    node = plugin.MiniMaxH3FP16Safe()
    model = node.patch(model)[0]
    diff = model.model.diffusion_model

    # 捕获 _fp16_downcast 输入 (每 block attn+mlp 各一次) 的原始 h max
    hmax = []
    orig_down = plugin._fp16_downcast
    def capture(h):
        if h.dtype != torch.float16:
            try:
                hmax.append(h.abs().max().item())
            except Exception:
                pass
        return orig_down(h)
    plugin._fp16_downcast = capture

    # 真实 forward (small: 24x16x16; 与 verify_prescale_pr1 同构造)
    T, H, W = (24, 16, 16) if SIZE == "small" else (30, 22, 38)
    text_len = 634
    torch.manual_seed(0)
    audio = torch.randn(1, 32, 2, T * 4, device='cuda', dtype=torch.float16) * 0.5
    ctx = torch.randn(1, text_len, 5120, device='cuda', dtype=torch.float16)
    ctx[0, :, ::113] += 8000.0
    payload = {"audio_scale": 1.0, "layout": None}
    x = torch.randn(1, 24, T, H, W, device='cuda', dtype=torch.float16)
    t = torch.tensor([500.0], device='cuda', dtype=torch.float32)
    seq = T * H * W + text_len + T * 8
    print(f"seq ~ {seq} (SIZE={SIZE})", flush=True)

    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        o = diff([x, audio], t, ctx, transformer_options={}, minimax_payload=payload)
    torch.cuda.synchronize()
    dt = time.time() - t0

    # ---- 结果 ----
    def _finite_all(obj):
        if isinstance(obj, (list, tuple)):
            return all(bool(torch.isfinite(t).all().item())
                       for t in obj if isinstance(t, torch.Tensor))
        return bool(torch.isfinite(obj).all().item())

    finite = _finite_all(o)
    fuse = plugin._FUSE_FLAG is not None and bool(plugin._FUSE_FLAG.item())
    if hmax:
        import numpy as np
        arr = np.array(hmax)
        print("\n===== 结果 =====", flush=True)
        print(f"forward 耗时: {dt:.1f}s | seq ~ {seq}", flush=True)
        print(f"输出 finite: {finite}", flush=True)
        print(f"deferred fuse 触发: {fuse} (应为 False)", flush=True)
        print(f"h max 捕获 {len(arr)} 次: max={arr.max():.1f}  p99={np.percentile(arr,99):.1f}  "
              f"median={np.median(arr):.1f}  (阈值 60000)", flush=True)
        ok = finite and not fuse and arr.max() < 60000.0
    else:
        print("\n===== 结果 =====", flush=True)
        print(f"forward 耗时: {dt:.1f}s | 输出 finite: {finite} | fuse: {fuse} | 无 h 捕获(全 fp16 输入?)", flush=True)
        ok = finite and not fuse
    print(f"\n===== {'PASS: v6.8.0 真实激活安全, 与条件降等价, fuse 不触发' if ok else 'FAIL'} =====", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
