# ComfyUI-MiniMaxH3-FP16Safe

**MiniMax H3 音视频 DiT 的 fp16 稳定性 + 速度优化插件（ComfyUI 自定义节点）· v6.0.0**

在 **V100（sm_70，无 bf16/fp8 硬件）** 或任何强制 fp16 计算的机器上，让 MiniMax H3（`comfy/ldm/minimax`，PR #15224）以**接近纯 fp16 的速度**获得**接近 fp32 的数值稳定性**——在硬件上"手动模拟 bf16 的宽指数范围"。

---

## 解决的问题

MiniMax H3 官方 `supported_inference_dtypes = [bf16, fp32]`，**明确不支持 fp16**。原因：模型的激活值（残差流、MLP 的 gated-silu 产物、fc2 输出）真实值可达 **50 万**，远超 fp16 上限 ±65504。V100 没有 bf16 硬件，官方路径只能回落到 fp32（慢约 4×）。

强行用"模型计算Dtype"节点设 fp16 计算 → 采样 NaN → 黑图。本插件在 fp16 计算下，把**会溢出的算子自动升 fp32 或用 2 的幂缩放补偿**，其余大矩阵乘保持 fp16（Tensor Core）。

---

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/<你的用户名>/ComfyUI-MiniMaxH3-FP16Safe.git
# 或手动复制整个文件夹到 ComfyUI/custom_nodes/
```

重启 ComfyUI。工作流中在 `MiniMaxH3` 分类下找到 **"MiniMax H3 FP16 Safe"** 节点。

> 若已安装过旧版 `ComfyUI-MiniMaxH3-FP16Fix-v2`：删除旧文件夹，本插件节点 ID 与旧版兼容，已保存的工作流无需改动。

---

## 使用

```
UNET加载器 ──> MiniMax H3 FP16 Safe ──> model ──> 采样器（KSampler 等）
     │              ▲  │
     │              │  └──> vae ──> VAE Decode（视频解码）
     └── vae ───────┘
```

- **`model`（必接）**：UNET 加载器的输出，可接在"模型计算Dtype"节点之后（插件也会强制 fp16）。
- **`vae`（推荐）**：视频 VAE 加载器输出。连接后插件对视频 VAE 打 fp16 稳定补丁（防解码端 NaN/黑图）。
- **`fix_vae`**（默认开）：是否应用视频 VAE 补丁。
- **`debug_nan`**（默认关）：打印第一个出现非有限值的 DiTBlock 索引，用于故障定位。
- **`profile`**（默认关）：打印 block 2/5/10/25/50 的 attention/MLP/其他 累计耗时与当前显存，用于速度瓶颈定位。

> 音频 VAE 与文本编码器不受本插件影响，按原工作流接线即可。

---

## 工作原理

思路：**残差流全程 fp32（稳定累加），大矩阵乘保持 fp16（Tensor Core 速度）**。

| 组件 | 处理 |
|---|---|
| 残差流（50 层 DiTBlock） | fp32（`_dit_block_forward`），不再有 fp16 累积溢出 |
| RMSNorm（210 个） | fp32 计算，I/O dtype 不变 |
| attention | **全程 fp16 SDPA**（2 的幂缩放补偿：残差流超阈值时先除以 2 的幂再降 fp16，q/k 由 RMSNorm 缩放不变性自动还原、v 输出乘回；SDPA 内部 fp32 累加 + max-subtract 抗溢出）；out_proj fp16 + 溢出回退 fp32 |
| MLP（v6） | **全 fp16**：fc1 输出（实测 max≈585）保持 fp16；gated-silu 的 gate 分支按 2 的幂缩放（`act = silu(a)·(b/2^k) ≤ ~2340`）；fc2 fp16；末尾 fp32 还原（小张量）。**零 fp32 中间张量** |
| 长序列 | **分块 MLP**：按 16384 行切块单遍处理，激活峰值与 seq 无关（~9GB），避免 16GB 卡上 lowvram 换出雪崩 |
| Qwen 文本路径（condition_proj） | fp32（防真实 context max≈21k 溢出） |
| 视频 VAE | fp16 流，仅 norm/attention 分数/silu 升 fp32 |

**保险丝机制**：fp16 快速路径后检查 `torch.isfinite()`，罕见溢出时自动用 fp32 重算该段——**保证永不 NaN**。

**为什么 v6 能全链路 fp16（V100 无 fp32 Tensor Core）**：
- 实测极端时间步下：P@V 输出 ~706、fc1 ~585（远小于 65504，可安全 fp16）；唯一超限的是 fc2 输出（~50 万）与 gated-silu 产物（~59k）——两者都用 **2 的幂缩放**（除法/乘法在 fp16 中无损）解决。
- v4/v5 时代 fc1 输出要升 fp32 算 silu，长序列每层多 ~850GB 搬运（480p/10s 约 +30s/步）；v6 的 gate 缩放让 silu 也在 fp16 完成，同时 act 有上界（≤2340）→ **数学上不可能溢出**。
- 残差流超阈值时 `_fp16_scaled` 仍会整体缩放（罕见），此时 MLP 自动切到 fp32 分块路径保证正确。

---

## 性能

实测（352×608、124 帧、5 秒视频，V100）：

| 配置 | 每步耗时 |
|---|---|
| 纯 fp16（无插件） | 8s（但 NaN → 黑图） |
| 纯 fp32 | 35s |
| 插件 v2 | 10s |

大 seq（480p/10s，seq≈9.85 万，V100，DiT offload 环境）：

| 配置 | 每步耗时 | 说明 |
|---|---|---|
| 纯 fp16（无插件） | 63s | NaN → 不可用 |
| 插件 v4 | 110s | 激活峰值 17GB+ 超 16GB → lowvram 换出雪崩 |
| 插件 v5（分块） | 97s | 峰值压回 ~9GB，雪崩消除 |
| **插件 v6（全 fp16 MLP）** | **~65-70s（预期）** | 再消除 fp32 cast 带宽 |

已验证模型：`minimax_h3_fl2va_pruned_fp8_scaled`、`minimax_h3_ref2va_pruned_int8_convrot`（均正常出片）。

---

## 控制台输出（预期）

```
[MiniMaxH3-FP16Safe] forced compute dtype -> fp16 (weights cast to fp16, Tensor Core ON)
[MiniMaxH3-FP16Safe][V6-FP16MLP] DiT patched: fp32 residual stream + fp16 SDPA attention + fully-fp16 MLP (gate-scaled, chunked) + 210 RMSNorm(s) + 1 condition_proj. (profile=False)
[MiniMaxH3-FP16Safe] Video VAE patched: attention + transformer-block norms + gated-silu upcast to fp32.
```

若 DiT 一行打印 "MODEL is not MiniMax H3"，说明检测未命中，请检查模型是否为 MiniMax H3 系列。

---

## 故障排查

| 现象 | 处理 |
|---|---|
| 节点找不到 / `IMPORT FAILED` | 确认 ComfyUI 含 PR #15224（`comfy/ldm/minimax` 存在）；升级 ComfyUI 后重试 |
| 采样仍有 NaN | 打开 `debug_nan`，把 `[MiniMaxH3-FP16Safe][DEBUG]` 输出发来（带 block 索引与输入/输出区分） |
| 采样正常但解码黑 | 确认 VAE 输入已连到本节点（未连则视频 VAE 无补丁） |
| 速度慢于预期 | 打开 `profile`，把 `[MiniMaxH3-FP16Safe][PROF]` 输出发来（每阶段耗时定位瓶颈） |
| VAE 报 dtype 不匹配 | 属旧版 bug，v6 已修复；确认加载的是本插件 |

---

## 兼容性

- **ComfyUI**：需包含 PR #15224（`comfy/ldm/minimax/` 模块）。
- **显卡**：任意（fp16 溢出是数值问题，与硬件无关）；V100（sm_70）效果最明显。
- **模型**：MiniMax H3 系列检查点（fp8_scaled / int8_convrot / bf16 均可）。
- **不修改任何 ComfyUI 核心文件**，升级不覆盖；卸载即删除文件夹。

---

## 版本历史

| 版本 | 要点 |
|---|---|
| v1 | 稳定版：attention/MLP 关键段升 fp32 |
| v2 | 提速：fp16 SDPA + 缩放 fc2（10s/步 @ 352×608） |
| v3 | 2 的幂缩放补偿，全链路 fp16（长 seq 不再退化） |
| v4 | qkv 检查 3→1 次同步 |
| v5 | 分块 MLP，长 seq 激活峰值与 seq 无关（110→97s/步） |
| **v6** | **全 fp16 MLP（gate 缩放），消除 fp32 cast 带宽；数学上不可能溢出** |

## License

MIT
