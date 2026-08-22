# ComfyUI-MiniMaxH3-FP16Safe

**fp16 stability + speed plugin for the MiniMax H3 audio/video DiT (ComfyUI custom node) · v6.8.0**

中文版见 [README.md](README.md) / Chinese version: [README.md](README.md) · Development/design notes: [DEVELOPMENT_EN.md](DEVELOPMENT_EN.md) / [DEVELOPMENT.md](DEVELOPMENT.md)

> **AI-assisted development notice**: This project was developed by the author with the help of an AI assistant (DeepSeek-V4-Flash). The author has no computer-science background; code and docs were produced with AI assistance. The project is shared as-is under MIT. If you run into issues, feel free to open an Issue and we will help as far as we can.

Run MiniMax H3 (`comfy/ldm/minimax`, PR #15224) at **near-fp16 speed** with **near-fp32 numerical stability** on **V100 (sm_70, no bf16/fp8 hardware)** or any machine forced into fp16 compute — effectively "emulating bf16's wide exponent range in hardware".

---

## The Problem

MiniMax H3 officially declares `supported_inference_dtypes = [bf16, fp32]` and **explicitly does not support fp16**. The reason: real activations (residual stream, gated-silu products, fc2 outputs) reach values of **~500k**, far beyond the fp16 limit of ±65504. On V100 there is no bf16 hardware, so the official path falls back to fp32 (~4× slower).

Forcing fp16 compute via the "Model Compute Dtype" node → NaN during sampling → black frames. This plugin keeps fp16 compute but **automatically upcasts overflowing ops to fp32 or compensates with power-of-2 scaling**, so all large matmuls stay on fp16 Tensor Cores.

---

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/aaalll12322/ComfyUI-MiniMaxH3-FP16Safe.git
# or copy the whole folder into ComfyUI/custom_nodes/ manually
```

Restart ComfyUI. Find the **"MiniMax H3 FP16 Safe"** node under the `MiniMaxH3` category.

> If you previously installed the older `ComfyUI-MiniMaxH3-FP16Fix-v2`: delete the old folder. Workflows saved with the old node ID `MiniMaxH3FP16SafeV2` must re-select this plugin's node before saving.

---

## Usage

```
UNET Loader ──> MiniMax H3 FP16 Safe ──> model ──> Sampler (KSampler, etc.)
```

- **`model` (required)**: output of the UNET Loader. Can be chained after a "Model Compute Dtype" node (the plugin also forces fp16 itself).
- **`debug_nan`** (default off): prints the index of the first DiTBlock where non-finite values appear, for fault diagnosis.
- **`profile`** (default off): prints cumulative attention/MLP/other timings and current VRAM for blocks 2/5/10/25/50, to locate speed bottlenecks.

> The audio VAE, video VAE and text encoder are all unaffected; wire them as in your original workflow (since v6.4.0 the video VAE has been verified safe in native fp16, no patch needed).

---

## Example Workflow

[`examples/minimax_h3_r2v_fp16safe.json`](examples/minimax_h3_r2v_fp16safe.json) in this repo provides a complete **ref2va (reference-image-to-video)** example: the official ComfyUI ref2va template with this plugin's node inserted.

```
UNET Loader ──▶ LoraLoader (optional) ──▶ MiniMax H3 FP16 Safe ──▶ model ──▶ Sampler
```

**Wiring notes (important)**:

- **LoRA must be connected *before* the plugin node**: the plugin's `patch()` clones the model and rewrites the forward pass (v6.6.0/v6.7.0 isolation). A LoRA must be merged into the model first, then handed to the plugin for fp16-safing. If placed after the plugin, the LoRA bypasses the fp16-safing scope and risks polluting the cached model object.
- The example LoRA is `minimax-h3\minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors` (FL2V turbo 4-step, sample path) — replace it as needed or just delete the `LoraLoader` node.
- The sample reference images `red_superboy_on_city_roof.png` / `mecha_dragon_lightning.png` are not bundled; provide your own (put them in `ComfyUI/input/`).

**Required models**:

| Purpose | File |
|---|---|
| UNET | `minimax_h3_ref2va_pruned_fp8_scaled.safetensors` (fp8 recommended on V100) |
| Text encoder | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` |

---

## How It Works

Idea: **fp32 residual stream (stable accumulation) + fp16 big matmuls (Tensor Core speed)**.

| Component | Handling |
|---|---|
| Residual stream (50 DiTBlocks) | fp32 (`_dit_block_forward`), no more fp16 accumulation overflow |
| RMSNorm (210×) | fp32 compute, I/O dtype preserved |
| Attention | **Always-fp16 SDPA (v6.3 fixed-scale, zero scans)**: input is always divided by 16 (exact power of 2; since v6.6.0, /256 → /16, +240× accuracy) before qkv_proj → qkv has a hard upper bound; q/k are restored exactly by RMSNorm homogeneity (logits use the original magnitude), v flows scaled through the linear chain SDPA→out_proj and is unscaled in fp32 afterwards; **mathematically cannot overflow, zero `.item()` scans** |
| MLP | **Fully fp16 + fixed-scale, zero scans**: fc1 output (measured max ≈585) stays fp16; gated-silu fixed ÷16 (act ≤ 21.4k, 3.4× headroom), fc2 fixed ÷8 (output ≤ 22.7k), all exact powers of 2; fp32 unscale at the end. **Zero fp32 intermediate tensors, zero per-chunk scans** |
| Long sequences | **Chunked MLP**: processed in 16384-row chunks in a single pass; activation peak is independent of seq length (~9GB), avoiding lowvram weight-eviction thrash on 16GB cards |
| Qwen text path (condition_proj) | fp32 (real context max ≈21k would overflow) |
| Video VAE | Not patched (native fp16 path; since v6.4.0 verified finite with ±38 outliers, ~0.5% output diff vs fp32) |

**Fuse mechanism**: the fp16 fast path is followed by a `torch.isfinite()` check; on a rare overflow the segment is recomputed in fp32 — **guaranteed no NaN** (the fixed scales have mathematical bounds, so the fuse almost never fires; it is only a safety net).

**Why v6 can go fully fp16 (V100 has no fp32 Tensor Cores)**:
- Measured at an extreme timestep: P@V output ~706, fc1 ~585 (far below 65504, safely fp16); only fc2 output (~500k) and gated-silu products (~59k) exceed the limit — both are handled with **fixed power-of-2 scaling** (lossless in fp16).
- Early implementations upcast the fc1 output to fp32 for silu, costing ~850GB of extra traffic per layer per step on long sequences (+~30s/step); the current gate scaling does silu in fp16 with a bounded act → **mathematically cannot overflow**.
- When the residual stream exceeds the threshold (rare), the MLP automatically falls back to the chunked-fp32 path for correctness.
- The current version replaced every per-layer `.item()` scan with fixed power-of-2 scaling (syncs per layer 11→2), eliminating the sync-amplification problem in low-VRAM offload environments.

---

## Performance

Measured (352×608, 124 frames, 5s video, V100):

| Config | Time per step |
|---|---|
| Pure fp16 (no plugin) | 8s (but NaN → black frames) |
| Pure fp32 | 35s |
| Plugin (fp16 compute) | ~10s |

Long seq (480p/10s, seq ≈ 98.5k, V100, DiT offload environment, real 4-step sampling):

| Config | Time per step | Notes |
|---|---|---|
| Pure fp16 (no plugin) | 63s | NaN → unusable |
| **Plugin v6.8.0 (zero per-block syncs + fixed-scale)** | **71-74s (ref2va fp8)** | fp32 residual stream + fp16 Tensor Cores, near the no-plugin baseline |

Verified models: `minimax_h3_fl2va_pruned_fp8_scaled`, `minimax_h3_ref2va_pruned_fp8_scaled`, `minimax_h3_ref2va_pruned_int8_convrot` (all produce valid video; fp8 is ~9s/step faster than int8 on V100 — fp8 recommended).

---

## Want More Speed? Try the Sol-Attn Sparse Plugin (experimental)

Already running MiniMax H3 with this plugin on V100 and want more speed? You can try the companion plugin **[ComfyUI-MiniMaxH3-SolAttn-V100](https://github.com/aaalll12322/ComfyUI-MiniMaxH3-SolAttn-V100)** (Sol-Attn sparsity, arXiv 2607.24027):

- **One node = embedded FP16Safe (v6.8.0 logic) + Sol-Attn sparsity** (keep-or-drop kernel), self-contained — no need to chain this plugin
- **Measured on real V100 (small-scale scenes)**: 480p/10s from 71-74 s/step → **43 s/step (~1.7×)**; 960×544/5s 33 s → **24 s/step (+27%)**, quality visually ≈ dense
- **Stability note**: sparse routing is sensitive to workflow/sequence shape — results and quality may vary across resolutions and frame counts, and it has only been validated on a small set of real runs so far. Try a small resolution first and compare quality before using it for production renders
- Recommended config (default): `tau=0.75, end_percent=0.9, dense_blocks="0-1,-1", topk_blocks=32, h3_prefix_tokens=1024`

> This plugin makes H3 "numerically correct and stable"; the SolAttn plugin further "skips useless compute" on top of it — complementary, each usable standalone. **SolAttn is experimental; for production renders, stick with this plugin (dense, full compute).**

---

## Expected Console Output

```
[MiniMaxH3-FP16Safe] patching on a cloned ModelPatcher (cache object left untouched, Issue #2)
[MiniMaxH3-FP16Safe] structure-cloned model tree (params shared, module instances isolated)
[MiniMaxH3-FP16Safe] forced compute dtype -> fp16 (weights cast to fp16, Tensor Core ON)
[MiniMaxH3-FP16Safe][V6.8-NOSYNC] DiT patched (instance-level, 156 modules): fp32 residual stream + fp16 SDPA attention (fixed /16 scale, zero scans) + fully-fp16 MLP (fixed-scale) + 210 RMSNorm(s) + 1 condition_proj. zero per-block GPU syncs (deferred fuse, fwd_wrapped=1). (profile=False)
```

If the DiT line prints "MODEL is not MiniMax H3", the detection did not match — check whether your model is from the MiniMax H3 family.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Node missing / `IMPORT FAILED` | Confirm ComfyUI includes PR #15224 (`comfy/ldm/minimax` exists); update ComfyUI and retry |
| Sampling still NaN | Enable `debug_nan` and share the `[MiniMaxH3-FP16Safe][DEBUG]` output (block index + input/output distinction) |
| Black frames / unchanged behavior after removing the node and re-running in the same process | v6.6.0 fixed the dtype residue (patch runs on a clone); v6.7.0 fixed the forward residue (structure-cloned module tree at patch time — cached objects keep their native forward, no model reload needed). If still broken, restart the ComfyUI server to clear the cache |
| Sampling fine but decode is black | Behavior of pre-v6.4.0 versions; the current node has no VAE port (VAE runs natively). If still black, update ComfyUI and verify the video VAE loads properly |
| Slower than expected | Enable `profile` and share the `[MiniMaxH3-FP16Safe][PROF]` output (per-stage timings) |
| Node red/missing (old workflows) | v6.5.0 removed the legacy alias `MiniMaxH3FP16SafeV2`; re-select the node manually in old workflows |

---

## Compatibility

- **ComfyUI**: requires PR #15224 (`comfy/ldm/minimax/` module).
- **GPU**: any (fp16 overflow is a numerical issue, independent of hardware); most beneficial on V100 (sm_70).
- **Models**: MiniMax H3 family checkpoints (fp8_scaled / int8_convrot / bf16 all work).
- **Does not modify any ComfyUI core files** — upgrades do not overwrite it; uninstall = delete the folder.

---

## Version History

| Version | Highlights |
|---|---|
| v6.0.0 | Fully fp16 MLP (gate scaling), fp32 cast bandwidth eliminated; mathematically cannot overflow |
| **v6.3.0** | **Attention fixed /256 scaling, zero scans (q/k restored by RMSNorm, v scaled through linear chain); per-layer syncs 11→2; ref2va fp8 480p/10s measured 75-78s/step** |
| v6.4.0 | Video VAE back to native fp16 path (fp32 upcast removed): measured finite decode (even with ±38 outliers), ~0.5% output diff vs fp32 path, ~30% faster (14.5s→10.1s) |
| **v6.5.0** | **Slimmer node + instance-level patching**: removed vae in/out ports and fix_vae (VAE handled natively by ComfyUI); patch now applies only to the current model instance (class methods untouched), deleting the node from a workflow no longer leaves a "ghost" patch active |
| **v6.6.0** | **Attention fixed scale 256→16 (PR #1, +240× accuracy, still ≥7× overflow headroom) + Issue #2 fix (patch runs on a clone, so removing the node no longer leaves fp16 compute on the cached model)** |
| **v6.7.0** | **Structure-clone isolation (Issue #2, forward side)**: the module tree is recursively rebuilt at patch time (weights shared, module instances isolated), so cached objects keep their native forward — no model reload needed after removing the node |
| **v6.8.0** | **Zero per-block GPU syncs (deferred fuse)**: the 2 magnitude probes are dropped (real-model measurement: attention/MLP input peak 78.5 vs fp16 limit 65504, 830× headroom — unconditional downcast is bit-identical to the conditional one), and the isfinite fuses now accumulate a flag on-GPU and are checked once per forward (a trip re-runs the forward in fp32). ref2va fp8 480p/10s measured 75-78 → 71-74 s/step |

## License

MIT
