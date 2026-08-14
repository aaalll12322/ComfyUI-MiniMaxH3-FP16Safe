# -*- coding: utf-8 -*-
# v6.7.0 structure-clone 验证: 无 GPU, 构造小模块树验证:
#   1) 结构独立: 克隆树模块对象与原树不同 (forward patch 打克隆树不污染原树)
#   2) 参数共享: 克隆树参数是原树参数的同一对象 (权重 cast/offload 照常)
#   3) forward 隔离: 改克隆树 forward 不影响原树 forward
#   4) named_modules 结构一致 (_wrap_rmsnorms / debug_nan 遍历依赖)
#   5) 数值一致: 参数共享 => 两树输出逐位一致
# 用法: <aki python> verify_structure_clone.py [--comfy <ComfyUI 根>] [--plugin <插件目录>]
# 默认 --comfy C:\ComfyUI, --plugin 为脚本所在目录 (插件仓库内)
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


class Leaf(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(4, 4))

    def forward(self, x):
        return x @ self.w


class Mid(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = Leaf()
        self.b = Leaf()


class Root(nn.Module):
    def __init__(self):
        super().__init__()
        self.mid = Mid()
        self.extra = nn.Sequential(Leaf(), Leaf())

    def forward(self, x):
        return self.mid.a(x) + self.mid.b(x) + self.extra[0](x) + self.extra[1](x)


def main():
    torch.manual_seed(0)
    root = Root()
    orig_fwd_a = root.mid.a.forward.__func__
    orig_fwd_root = root.forward.__func__

    clone = nodes._structure_clone(root)

    # 1) 结构独立
    assert clone is not root, "顶层对象应独立"
    assert clone.mid is not root.mid
    assert clone.mid.a is not root.mid.a
    assert clone.mid.b is not root.mid.b
    assert clone.extra is not root.extra
    assert clone.extra[0] is not root.extra[0]
    assert clone.extra[1] is not root.extra[1]
    print("[1] 结构独立 OK: 全部模块对象为新实例")

    # 2) 参数共享
    assert clone.mid.a.w is root.mid.a.w
    assert clone.mid.b.w is root.mid.b.w
    assert clone.extra[0].w is root.extra[0].w
    print("[2] 参数共享 OK: 权重为同一 Parameter 对象")

    # 3) forward 隔离 (用 __func__ 比较: bound method 每次访问是新对象, is 恒 False)
    def hacked(self, x):
        return x * 0.0

    clone.mid.a.forward = types.MethodType(hacked, clone.mid.a)
    assert root.mid.a.forward.__func__ is Leaf.forward, "原树 forward 被污染!"
    assert clone.mid.a.forward.__func__ is hacked
    assert root.forward.__func__ is Root.forward
    print("[3] forward 隔离 OK: 改克隆树不影响原树")

    # 4) named_modules 结构一致
    n_root = [n for n, _ in root.named_modules()]
    n_clone = [n for n, _ in clone.named_modules()]
    assert n_root == n_clone, f"结构不一致: {n_root} vs {n_clone}"
    print(f"[4] 结构一致 OK: {len(n_root)} 个模块路径完全相同")

    # 5) 数值一致 (clone 的 forward 未被 hack 前)
    clone2 = nodes._structure_clone(root)
    x = torch.randn(2, 4)
    torch.manual_seed(1)
    o1 = root(x)
    torch.manual_seed(1)
    o2 = clone2(x)
    assert torch.equal(o1, o2), "参数共享下输出应逐位一致"
    print("[5] 数值一致 OK: 两树输出逐位相同")

    print("\n===== ALL PASS: structure clone 正确 =====")
    sys.exit(0)


if __name__ == "__main__":
    main()
