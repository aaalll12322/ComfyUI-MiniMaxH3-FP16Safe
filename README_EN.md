# ComfyUI-MiniMaxH3-FP16Safe

**fp16 stability + speed plugin for the MiniMax H3 audio/video DiT (ComfyUI custom node) · v6.3.0**

中文版见 [README.md](README.md) / Chinese version: [README.md](README.md)

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
     │              ▲  │
     │              │  └──> vae ──> VAE Decode (video)
     └── vae ───────┘
```

- **`model` (required)**: output of the UNET Loader. Can be chained after a "Model Compute Dtype" node (the plugin also forces fp16 itself).
- **`vae` (recommended)**: output of the video VAE Loader. When connected, the plugin applies an fp16-stability patch to the video VAE (prevents NaN/black frames on the decode side).
- **`fix_vae`** (default on): whether to apply the video VAE patch.
- **`debug_nan`** (default off): prints the index of the first DiTBlock where non-finite values appear, for fault diagnosis.
- **`profile`** (default off): prints cumulative attention/MLP/other timings and current VRAM for blocks 2/5/10/25/50, to locate speed bottlenecks.

> The audio VAE and the text encoder are unaffected; wire them as in your original workflow.

---

## How It Works

Idea: **fp32 residual stream (stable accumulation) + fp16 big matmuls (Tensor Core speed)**.

| Component | Handling |
|---|---|
| Residual stream (50 DiTBlocks) | fp32 (`_dit_block_forward`), no more fp16 accumulation overflow |
| RMSNorm (210×) | fp32 compute, I/O dtype preserved |
| Attention | **Always-fp16 SDPA (v6.3 fixed-scale, zero scans)**: input is always divided by 256 (exact power of 2) before qkv_proj → qkv has a hard upper bound; q/k are restored exactly by RMSNorm homogeneity (logits use the original magnitude), v flows scaled through the linear chain SDPA→out_proj and is unscaled in fp32 afterwards; **mathematically cannot overflow, zero `.item()` scans** |
| MLP | **Fully fp16 + fixed-scale, zero scans**: fc1 output (measured max ≈585) stays fp16; gated-silu fixed ÷16 (act ≤ 21.4k, 3.4× headroom), fc2 fixed ÷8 (output ≤ 22.7k), all exact powers of 2; fp32 unscale at the end. **Zero fp32 intermediate tensors, zero per-chunk scans** |
| Long sequences | **Chunked MLP**: processed in 16384-row chunks in a single pass; activation peak is independent of seq length (~9GB), avoiding lowvram weight-eviction thrash on 16GB cards |
| Qwen text path (condition_proj) | fp32 (real context max ≈21k would overflow) |
| Video VAE | fp16 stream; only norms/attention scores/silu upcast to fp32 |

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
| **Plugin v6.3.0 (fixed-scale, zero scans)** | **75-78s (ref2va fp8)** | fp32 residual stream + fp16 Tensor Cores, near the no-plugin baseline |

Verified models: `minimax_h3_fl2va_pruned_fp8_scaled`, `minimax_h3_ref2va_pruned_fp8_scaled`, `minimax_h3_ref2va_pruned_int8_convrot` (all produce valid video; fp8 is ~9s/step faster than int8 on V100 — fp8 recommended).

---

## Expected Console Output

```
[MiniMaxH3-FP16Safe] forced compute dtype -> fp16 (weights cast to fp16, Tensor Core ON)
[MiniMaxH3-FP16Safe][V6.3-FIXEDATTN] DiT patched: fp32 residual stream + fp16 SDPA attention (fixed /256 scale, zero scans) + fully-fp16 MLP (fixed-scale) + 210 RMSNorm(s) + 1 condition_proj. (profile=False)
[MiniMaxH3-FP16Safe] Video VAE patched: attention + transformer-block norms + gated-silu upcast to fp32.
```

If the DiT line prints "MODEL is not MiniMax H3", the detection did not match — check whether your model is from the MiniMax H3 family.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Node missing / `IMPORT FAILED` | Confirm ComfyUI includes PR #15224 (`comfy/ldm/minimax` exists); update ComfyUI and retry |
| Sampling still NaN | Enable `debug_nan` and share the `[MiniMaxH3-FP16Safe][DEBUG]` output (block index + input/output distinction) |
| Sampling fine but decode is black | Make sure the VAE input is connected to this node (otherwise the video VAE is unpatched) |
| Slower than expected | Enable `profile` and share the `[MiniMaxH3-FP16Safe][PROF]` output (per-stage timings) |
| VAE dtype mismatch | Old-version bug, fixed in the current version; make sure you are loading this plugin |

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

## License

MIT
