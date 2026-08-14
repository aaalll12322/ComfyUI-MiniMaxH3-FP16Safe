# DEVELOPMENT.md — 开发与设计文档

> **AI 辅助开发声明**：本项目由作者使用 AI 助手（DeepSeek-V4-Flash）辅助完成开发。作者本人没有计算机专业背景，代码编写、调试与文档撰写均借助 AI 完成。项目公开的目的是分享思路与成果，代码按现状提供（MIT License）；如果你在阅读或使用中发现问题，欢迎提交 Issue，我们会在能力范围内协助解决。

本文件面向**想理解插件内部机制、自行验证、或参与贡献**的开发者。它解释了：

1. MiniMax H3 在 fp16 下为什么 NaN（根因）
2. 本插件的整体设计（fp32 残差流 + fp16 Tensor Core 大矩阵乘）
3. 每个关键算子的数学原理（含固定缩放的设计推导）
4. 如何复现验证（数值级 + 真实模型级）
5. 已知问题与边界（避免重复踩坑）

> 使用说明见 [README.md](README.md)。本文档不涉及任何个人环境信息，所有数值均为真实模型实测（模型、LoRA 均为 MiniMax H3 官方生态公开资源）。

---

## 1. 背景：MiniMax H3 为什么在 fp16 下 NaN

MiniMax H3（`comfy/ldm/minimax`，ComfyUI PR #15224）官方声明：

```
supported_inference_dtypes = [bf16, fp32]
```

**不支持 fp16**。根因是激活值动态范围远超 fp16 表示能力：

| 张量 | 实测量级（真实模型、极端时间步） | fp16 上限 |
|---|---|---|
| 残差流 | 可达 ~5×10⁵ | ±65504 |
| gated-silu 产物 | ~5.9×10⁴ | ±65504 |
| fc2（MLP 输出） | ~5×10⁵ | ±65504 |
| P@V（attention 输出） | ~7×10² | 安全 |
| fc1（MLP 激活） | ~5.85×10² | 安全 |

V100（sm_70）**没有 bf16 硬件**，官方路径只能回落 fp32（约慢 4×）。强行设 fp16 计算 → 采样 NaN → 黑帧。

**本插件的思路：残差流全程 fp32（稳定累加），大矩阵乘保持 fp16（Tensor Core 速度）**，并用 2 的幂缩放补偿会溢出的算子——在硬件上"手动模拟 bf16 的宽指数范围"。

---

## 2. 整体设计

| 组件 | 处理方式 |
|---|---|
| 残差流（DiTBlock / RefinerBlock） | fp32 累加（`_dit_block_forward` / `_refiner_block_forward`） |
| RMSNorm（210 个） | fp32 计算，I/O dtype 保持（`_FP32RMSNorm` 包装） |
| attention | 全程 fp16 SDPA，输入固定 ÷2⁸ 缩放（见 §3） |
| MLP | 全 fp16 + 固定缩放（gate ÷2⁴、fc2 输入 ÷2³，见 §4） |
| 长序列 | 分块 MLP（16384 行/块），激活峰值与 seq 无关 |
| Qwen 文本路径（condition_proj） | fp32（防真实 context 溢出） |
| 视频 VAE | 不处理（原生 fp16 路径实测有限） |

**保险丝机制**：fp16 快速路径后检查 `torch.isfinite()`，罕见溢出时自动用 fp32 重算该段——保证永不 NaN。固定缩放的数学上界使保险丝几乎不触发，仅作安全网。

**v6.5.0 起为实例级 patch**：只替换当前模型实例内模块的 forward，不再修改类方法。工作流中删除节点后，新加载的其他 MiniMax 模型保持原生行为（不"幽灵生效"）。

**v6.7.0 起补上结构克隆**：`ModelPatcher.clone()` 只隔离 dtype 状态、**不深拷贝模型实例**——实例级 forward patch 打在共享模块上，仍会残留在缓存的 UNETLoader 输出里（删节点重跑仍是插件行为）。现在 patch 前先递归重建模块树（权重参数共享、模块实例隔离），缓存对象永远保持原生 forward（见 §6.1）。

---

## 3. attention 的固定缩放（核心数学）

### 3.1 缩放链

输入 `x`（fp32 残差流，或 fp16）先除以固定 scale `s = 256`（2 的幂，fp16 中除法精确），再进 `qkv_proj`：

```
x_h = x / s            # fp16, 精确
q, k, v = qkv_proj(x_h)
```

- **q/k**：经 RMSNorm 后恢复 O(1) 量级（见 §3.2），logits 用原始量级。
- **v**：带缩放穿过线性链 `SDPA → out_proj`，末尾在 fp32 乘回 `s`。

因为 `s` 是 2 的幂，`x/s` 在 fp16 中**无损**；`P@V ≤ 706`（实测），`out_proj` 输出 ≤ ~39k/s（fp16 永不溢出）。

### 3.2 RMSNorm 齐次性与 eps 项（为何 s 不能太大）

RMSNorm 的 scale 齐次性只在 `eps → 0` 时成立：

```
rms_norm(q/s) = (q/s) / sqrt( ms(q/s) + eps )
              = q / sqrt( ms(q) + eps·s² )        ← 精确展开
```

即缩放后做 RMSNorm，**等价于**不缩放、但把 eps 放大了 s² 倍。当 `eps·s²` 与 `ms(q)` 同量级时，attention logits 被系统性软化（分母偏大）。

MiniMax H3 的 `qk_norm_eps = 1e-5`。真实模型实测（深层 block）：

| prescale s | eps·s² / ms(q) | 影响 |
|---|---|---|
| 256 | **4% ~ 9%**（ms(q) 中位 7~13） | 不可忽略 |
| 16 | 0.02% ~ 0.04% | 可忽略 |

### 3.3 误差随 1/s² 单调下降（实测）

同一激活、无采样发散下，`rms_norm(q/s)` 相对 `rms_norm(q)` 的误差（真实模型 52 层平均）：

| prescale | 平均 rel err | vs s=256 |
|---|---|---|
| /256 | 1.35e-2 | 1.0x |
| /64 | 8.99e-4 | 0.067x |
| /32 | 2.25e-4 | 0.017x |
| **/16** | **5.62e-5** | **0.004x ≈ 1/256** |

误差比 ≈ 240x ≈ (256/16)²，与 1/s² 理论精确吻合、无平台。早期 block（残差流能量低、ms(q) 小）在 s=256 时误差最大（3.8e-2 ~ 1.05e-1），s=16 时降到 2.7e-4 ~ 4.9e-4。

### 3.4 溢出下界：s 越小越安全，还是越危险？

直觉上 s 小 → `x/s` 大 → 溢出风险高；但**绑定约束是 `out_proj` 的输出**（fp16 流中最大张量），而它正比于 `P@V`，又正比于 `v` 的缩放：

```
out_proj_out ≈ ||W_out|| · P@V(v/s)  ∝  1/s
```

s 越小，`out_proj` 的 fp16 输出越小。实测（小 seq，52 层最坏）：

| prescale | out_proj fp16 输出 max | 余量（vs 65504） |
|---|---|---|
| /256 | 120.7 | 543x |
| /16 | 1931.0 | **33.9x** |

即便 480p/10s 规模（out_proj 输出 ~39k 量级）s=16 仍有 ~27× 余量；1344×768/124 帧极端峰值 1.43×10⁵ 时仍有 7.3×，且保险丝兜底。**因此 s=16 是纯收益：精度 +240 倍，溢出余量依然充足。**

> 结论：`_ATTN_FIXED_SCALE = 16`（原为 256）。这是社区贡献的优化（PR #1），数学推导与实测见上。

---

## 4. MLP 的固定缩放（v6 起全 fp16）

| 阶段 | 处理 | 依据 |
|---|---|---|
| fc1 | 保持 fp16（max≈585） | 远小于 65504 |
| gated-silu | gate 分支固定 ÷16（2⁴） | silu 产物 ≤ ~21.4k，3.4× 裕量 |
| fc2 | 输入固定 ÷8（2³） | 输出 ≤ ~22.7k |
| 还原 | 末尾 fp32 乘回 | 精确 |

全部 2 的幂缩放（fp16 无损），零 fp32 中间张量、零逐块扫描——消除了长序列上每层 ~850GB 的 cast 带宽。

长序列（seq > 16384）自动分块，激活峰值 ~9GB 与 seq 无关，避免 16GB 卡 lowvram 换出雪崩。

---

## 5. 如何验证

### 5.1 数值级（无 GPU 采样，仅数学）

验证 §3.2/§3.3：对真实激活的 q（fp32 权重投影、不缩放），比较

```
ref  = rms_norm(q)              = q / sqrt(ms(q) + eps)
q_s  = rms_norm(q/s) 精确形式    = q / sqrt(ms(q) + eps·s²)
```

遍历各 s，确认 rel err 随 1/s² 单调、无平台。此验证不依赖 rope 实现，纯 fp32，任何机器可复现。

### 5.2 真实模型级

1. 加载 MiniMax H3 检查点（fp8_scaled / int8_convrot / bf16 均可）+ turbo LoRA；
2. 插件 `patch()` 后，在 attention 入口捕获真实激活 `x` 与 `rope_freqs`；
3. 对同一 `x` 分别以 s=256/64/32/16 跑完整 fp16 链路（qkv → RMSNorm+rope → SDPA → out_proj → 乘回）；
4. 检查：早期 block 输出差异 d(256,16)/max 应在 1%~8% 量级（eps 效应可见），深层收敛到 fp16 ulp 步长；s=16 时 `out_proj` 输出 max 应远低于 65504。

（完整可运行脚本可参考仓库验证脚本 `verify_prescale_pr1.py` 的结构。）

### 5.3 最终质量

数值验证之外，建议对比真实出片（同 seed 同参数，仅 scale 不同）：
- 目标输出 PSNR 应 > 40dB（s=16 vs s=256）；
- 肉眼无可见差异；
- s=16 渲染全程无保险丝触发（控制台无 fp32 重算告警）。

---

## 6. 已知问题与边界

### 6.1 [已修复 v6.6.0 + v6.7.0] 删除节点后补丁残留（Issue #2：dtype 侧 + forward 侧）

**现象（v6.5.0 及之前）**：`patch()` 调用 `model.set_model_compute_dtype(torch.float16)`，该调用把 `manual_cast_dtype` 写入 ModelPatcher 的 `object_patches` 并设置 `force_cast_weights`——这是 **ModelPatcher 上的持久状态**。ComfyUI 缓存 UNETLoader 输出，因此：

```
带节点跑一次 → 删除节点 → 同进程重跑
```

模型**仍被强制 fp16 compute，但不再有本插件的缩放补偿**（forward patch 是实例级的、随新加载的模型实例消失）→ 回到 fp16 NaN/黑帧的条件。

**修复（v6.6.0，dtype 侧）**：`patch()` 现在**先 `model.clone()` 再在其上做全部修改**（`set_model_compute_dtype` + 实例级 forward patch），返回 clone：

- `clone()` 深拷贝 `model_options` / `object_patches` / `force_cast_weights` → **缓存中的原始 ModelPatcher 保持干净**，删除节点后重跑模型回到原生 fp32 compute；
- 模型权重实例共享（clone 不复制权重），但 forward patch 是实例级、且 `comfy.ops` 的权重自动 cast 保证 fp16 输入下计算正确；
- 已验证（`verify_issue2_clone.py`）：patch 后原始对象 `object_patches` 为空、`force_cast_weights` 保持 `False`，fp16 状态只存在于 clone 上。

**修复（v6.7.0，forward 侧——v6.6.0 的盲区）**：`ModelPatcher.clone()`（`model_patcher.py`）只是新建一个 ModelPatcher 包装，**`self.model`（模块树）仍然共享**。因此即便 dtype 状态隔离了，`m.forward = MethodType(...)` 这类实例级补丁仍是直接写在共享模块对象上的——**缓存的原始对象（M0）的模块树也被改掉**。用户实测：删除插件节点 + 模型计算Dtype 节点后重跑，行为与删除前一致，只有重新加载模型（新模块树）才恢复原生。

- 解决：patch 时用 `_structure_clone()` **递归重建模块树**——逐节点浅拷贝、重建 `_modules` 容器，得到一棵**结构独立、权重参数共享**的新树；所有补丁（forward / RMSNorm 包装 / condition_proj / debug 索引）只落在克隆树上；
- 参数共享 ⇒ ComfyUI 权重 cast 与 lowvram 换入换出照常工作（内存增量仅模块对象本身，约几 MB；耗时毫秒级）；
- 已验证（`verify_structure_clone.py`，无 GPU 小模块树 5 项断言）：结构独立 / 参数共享 / forward 隔离 / 结构一致 / 数值一致，全过；真实 ComfyUI 上删节点重跑行为干净。

> 若因外部原因 `clone()` 或结构克隆失败，插件会回退为就地 patch 并打印 WARNING（此时缓存对象仍可能保留 fp16 compute dtype 或 forward 补丁，建议重启服务）。

### 6.2 边界

- **视频 VAE**：保持原生 fp16 路径（v6.4.0 实测含 ±38 outlier 解码仍有限，与 fp32 差异 0.5%）。若个别 checkpoint 解码异常，优先怀疑 VAE 权重本身。
- **Qwen 文本路径**：`condition_proj` 强制 fp32（防真实 context 溢出），速度影响可忽略（文本 token 远少于视频 token）。
- **非 MiniMax 模型**：`_is_minimax_dit()` 检测不命中时插件直接透传，不做任何修改。

---

## 7. 版本历史（开发视角）

| 版本 | 关键决策 |
|---|---|
| v6.0.0 | MLP 全 fp16 + gate 固定缩放，消除 fp32 cast 带宽；数学上不可能溢出 |
| v6.3.0 | attention 固定 ÷256 缩放零扫描（RMSNorm 齐次性还原 q/k）；每层同步 11→2 次 |
| v6.4.0 | 视频 VAE 恢复原生 fp16 路径（实测有限，无需升精度） |
| v6.5.0 | 实例级 patch（forward 层面不再幽灵生效）；移除旧节点 ID 别名 |
| v6.6.0 | **attention 固定缩放 256 → 16**（PR #1，精度 +240×，溢出余量仍 ≥7×）；**Issue #2 修复**（patch 在 clone 上执行，缓存对象不再被 dtype 污染） |
| v6.7.0 | **结构克隆隔离（Issue #2 forward 侧）**：`ModelPatcher.clone()` 不深拷贝模型实例，forward patch 仍会残留进缓存对象；递归重建模块树（参数共享、实例隔离），缓存对象保持原生 forward |

## License

MIT
