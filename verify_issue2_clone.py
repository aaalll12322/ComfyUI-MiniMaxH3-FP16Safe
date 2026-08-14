# -*- coding: utf-8 -*-
# Issue #2 修复验证: patch 必须在 clone 上执行, 不污染 UNETLoader 缓存的原始 ModelPatcher。
# 检查:
#   1) patch 返回的 model 与传入的 model 不是同一个对象 (clone 生效)
#   2) 原始 model 的 model_options["object_patches"] 不含 manual_cast_dtype (未被污染)
#   3) 原始 model 的 force_cast_weights 保持 False
#   4) clone 后的 model 上 fp16 状态正确设置
# 用法: python verify_issue2_clone.py [--comfy <ComfyUI root>] [--plugin <plugin dir>]
import sys, os
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
    return comfy_root, plugin_dir, model

COMFY_ROOT, PLUGIN_DIR, MODEL = _parse_args()
sys.path.insert(0, os.path.abspath(COMFY_ROOT))
sys.path.insert(0, PLUGIN_DIR)
import torch
import comfy.utils, comfy.sd
import nodes as plugin

def main():
    print("=== 加载模型 ===", flush=True)
    sd = comfy.utils.load_torch_file(MODEL)
    model = comfy.sd.load_diffusion_model_state_dict(sd, model_options={})
    del sd

    # 模拟 UNETLoader 缓存对象: 记录原始状态
    orig_op = model.object_patches
    orig_force = model.force_cast_weights
    print(f"patch 前: object_patches keys={list(orig_op.keys())} force_cast_weights={orig_force}", flush=True)

    print("=== 执行 patch ===", flush=True)
    node = plugin.MiniMaxH3FP16Safe()
    patched = node.patch(model)[0]

    # 1) clone 生效?
    is_clone = patched is not model
    print(f"[1] patch 返回新对象 (clone): {is_clone}", flush=True)

    # 2) 原始对象未被污染?
    leaked = "manual_cast_dtype" in model.object_patches
    print(f"[2] 原始 model object_patches 含 manual_cast_dtype: {leaked} "
          f"(应为 False)", flush=True)
    print(f"    原始 model object_patches keys = {list(model.object_patches.keys())}", flush=True)

    # 3) 原始 force_cast_weights 保持?
    print(f"[3] 原始 model force_cast_weights = {model.force_cast_weights} "
          f"(应为 {orig_force})", flush=True)

    # 4) clone 上 fp16 状态正确?
    p_op = patched.object_patches
    p_force = patched.force_cast_weights
    print(f"[4] clone object_patches manual_cast_dtype = {p_op.get('manual_cast_dtype', None)} "
          f"(应为 torch.float16)", flush=True)
    print(f"    clone force_cast_weights = {p_force} (应为 True)", flush=True)

    ok = (is_clone and not leaked and model.force_cast_weights == orig_force
          and p_op.get("manual_cast_dtype") == torch.float16 and p_force)
    print(f"\n===== {'PASS: 原始缓存对象未被污染, fp16 只作用于 clone' if ok else 'FAIL'} =====", flush=True)
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
