# -*- coding: utf-8 -*-
# v6.8.0 deferred-fuse 逻辑验证 (CPU, 无 GPU):
#   1) _accum_fuse: 正常张量不置位, 含 inf/nan 置位; 无 .item() 调用 (无 sync)
#   2) _model_fwd_wrapper: 正常 forward 不重跑; fuse 置位时以 _FP32_MODE 重跑并恢复
#   3) _fp16_downcast: 默认无条件降 fp16; _FP32_MODE 时保持 fp32
# 用法: <aki python> verify_deferred_fuse.py --comfy <ComfyUI 根> --plugin <插件目录>
import sys, types, os
import torch
import torch.nn as nn


def _pick(flag, default):
    if flag in sys.argv:
        return sys.argv[sys.argv.index(flag) + 1]
    return default


COMFY = _pick("--comfy", r"C:\ComfyUI")
PLUGIN = _pick("--plugin", os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(COMFY))
sys.path.insert(0, os.path.abspath(PLUGIN))
import nodes


class FakeModel(nn.Module):
    """模拟 MiniMaxH3Model: forward 返回一个张量, 可注入非有限值。"""
    def __init__(self, bad=False):
        super().__init__()
        self.bad = bad
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        if self.bad and not nodes._FP32_MODE and self.calls == 1:
            y = x * float("nan")   # 第一次 fp16 路径产生 NaN
        else:
            y = x * 2.0
        nodes._accum_fuse(y)       # 真实路径中由 attn/MLP 累积
        return y


def main():
    torch.manual_seed(0)

    # 1) _accum_fuse
    nodes._FUSE_FLAG = None
    nodes._accum_fuse(torch.tensor([1.0, 2.0, -3.0]))
    assert nodes._FUSE_FLAG is not None and not bool(nodes._FUSE_FLAG.item()), "正常值不应置位"
    nodes._accum_fuse(torch.tensor([1.0, float("nan")]))
    assert bool(nodes._FUSE_FLAG.item()), "NaN 应置位"
    nodes._FUSE_FLAG = None
    nodes._accum_fuse(torch.tensor([1.0, float("inf")]))
    assert bool(nodes._FUSE_FLAG.item()), "Inf 应置位"
    print("[1] _accum_fuse OK: 正常不置位, NaN/Inf 置位 (GPU 累积, 无 .item() per block)")

    # 2) _model_fwd_wrapper: 正常 forward 不重跑
    m = FakeModel(bad=False)
    w = types.MethodType(nodes._model_fwd_wrapper(m.forward), m)
    m.forward = w
    nodes._FUSE_FLAG = None
    out = m(torch.tensor([1.0]))
    assert m.calls == 1, "正常 forward 不应重跑"
    assert not nodes._FP32_MODE, "_FP32_MODE 应恢复 False"
    print("[2a] 正常路径 OK: forward 1 次, 无重跑")

    # 3) _model_fwd_wrapper: fuse 置位时 fp32 重跑一次
    m2 = FakeModel(bad=True)
    w2 = types.MethodType(nodes._model_fwd_wrapper(m2.forward), m2)
    m2.forward = w2
    nodes._FUSE_FLAG = None
    out2 = m2(torch.tensor([1.0]))
    assert m2.calls == 2, f"fuse 触发应重跑 (calls={m2.calls})"
    assert torch.isfinite(out2).all(), "重跑输出应有限"
    assert not nodes._FP32_MODE, "重跑后 _FP32_MODE 应恢复 False"
    print("[2b] fuse 触发 OK: fp32 重跑 1 次, 输出有限, 标志恢复")

    # 4) _fp16_downcast
    x = torch.randn(4, 4, dtype=torch.float32)
    assert nodes._fp16_downcast(x).dtype == torch.float16, "默认应无条件降 fp16"
    nodes._FP32_MODE = True
    assert nodes._fp16_downcast(x).dtype == torch.float32, "_FP32_MODE 应保持 fp32"
    nodes._FP32_MODE = False
    print("[3] _fp16_downcast OK: 默认降 fp16, _FP32_MODE 保持 fp32")

    print("\n===== ALL PASS: deferred fuse 逻辑正确 =====")
    sys.exit(0)


if __name__ == "__main__":
    main()
