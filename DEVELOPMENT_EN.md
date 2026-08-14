# DEVELOPMENT.md — Development & Design Notes

> **AI-assisted development notice**: This project was developed by the author with the help of an AI assistant (DeepSeek-V4-Flash). The author has no computer-science background; the code, debugging, and documentation were all produced with AI assistance. The project is shared openly to exchange ideas and results; the code is provided as-is (MIT License). If you find any problems while reading or using it, feel free to open an Issue and we will help as far as we can.

This document is for **developers who want to understand the plugin's internals, verify it themselves, or contribute**. It covers:

1. Why MiniMax H3 produces NaN in fp16 (root cause)
2. Overall design (fp32 residual stream + fp16 Tensor-Core matmuls)
3. The math behind each key op (including the fixed-scale design derivation)
4. How to reproduce the verification (numeric-level + real-model-level)
5. Known issues and boundaries (so you don't step on the same rakes)

> Usage instructions are in [README.md](README.md). This document contains no personal environment info; all numbers are measured on a real model (the checkpoint and LoRA are public MiniMax H3 ecosystem resources).

---

## 1. Background: why MiniMax H3 NaNs in fp16

MiniMax H3 (`comfy/ldm/minimax`, ComfyUI PR #15224) officially declares:

```
supported_inference_dtypes = [bf16, fp32]
```

**fp16 is not supported.** The root cause is that activation dynamic range far exceeds fp16's representable range:

| Tensor | Measured magnitude (real model, extreme timestep) | fp16 limit |
|---|---|---|
| Residual stream | up to ~5×10⁵ | ±65504 |
| gated-silu product | ~5.9×10⁴ | ±65504 |
| fc2 (MLP output) | ~5×10⁵ | ±65504 |
| P@V (attention output) | ~7×10² | safe |
| fc1 (MLP activation) | ~5.85×10² | safe |

V100 (sm_70) has **no bf16 hardware**, so the official path falls back to fp32 (~4× slower). Forcing fp16 compute → NaN during sampling → black frames.

**The plugin's idea: keep the residual stream in fp32 (stable accumulation) while every big matmul stays on fp16 Tensor Cores**, compensating the overflowing ops with power-of-2 scaling — effectively "emulating bf16's wide exponent range in hardware".

---

## 2. Overall design

| Component | Handling |
|---|---|
| Residual stream (DiTBlock / RefinerBlock) | fp32 accumulation (`_dit_block_forward` / `_refiner_block_forward`) |
| RMSNorm (210×) | fp32 compute, I/O dtype preserved (`_FP32RMSNorm` wrapper) |
| attention | fully fp16 SDPA, fixed ÷2⁸ input scaling (see §3) |
| MLP | fully fp16 + fixed scaling (gate ÷2⁴, fc2 input ÷2³, see §4) |
| Long sequences | chunked MLP (16384 rows/chunk), activation peak independent of seq |
| Qwen text path (condition_proj) | fp32 (protects against real-context overflow) |
| Video VAE | untouched (native fp16 path measured finite) |

**Fuse mechanism**: after the fp16 fast path, `torch.isfinite()` is checked; on a rare overflow the segment is recomputed in fp32 — NaN is never produced. The fixed-scaling mathematical upper bounds make the fuse almost never trip; it is purely a safety net.

**Since v6.5.0 the patch is instance-level**: only the modules inside the *current* model instance have their forward replaced; class methods are untouched. After deleting the node from a workflow, other MiniMax models loaded later keep native behavior (no "ghost patching").

---

## 3. The fixed scaling of attention (core math)

### 3.1 The scaling chain

Input `x` (fp32 residual stream, or fp16) is first divided by a fixed scale `s = 256` (a power of 2 — exact division in fp16), then fed to `qkv_proj`:

```
x_h = x / s            # fp16, exact
q, k, v = qkv_proj(x_h)
```

- **q/k**: restored to O(1) magnitude by RMSNorm (see §3.2); logits use the original scale.
- **v**: travels scaled through the linear chain `SDPA → out_proj`, then is multiplied back by `s` in fp32 at the end.

Because `s` is a power of 2, `x/s` is **lossless** in fp16; `P@V ≤ 706` (measured), `out_proj` output ≤ ~39k/s (fp16 can never overflow).

### 3.2 RMSNorm homogeneity and the eps term (why s must not be too large)

RMSNorm's scale-homogeneity only holds as `eps → 0`:

```
rms_norm(q/s) = (q/s) / sqrt( ms(q/s) + eps )
              = q / sqrt( ms(q) + eps·s² )        ← exact expansion
```

That is, normalizing after scaling is **equivalent** to normalizing without scaling but with eps amplified by s². When `eps·s²` is comparable to `ms(q)`, the attention logits are systematically softened (denominator too large).

MiniMax H3 uses `qk_norm_eps = 1e-5`. Measured on a real model (deep blocks):

| prescale s | eps·s² / ms(q) | impact |
|---|---|---|
| 256 | **4% ~ 9%** (ms(q) median 7~13) | not negligible |
| 16 | 0.02% ~ 0.04% | negligible |

### 3.3 Error decreases as 1/s², monotonically (measured)

On identical activations with no sampling divergence, the error of `rms_norm(q/s)` vs `rms_norm(q)` (real model, mean over 52 layers):

| prescale | mean rel err | vs s=256 |
|---|---|---|
| /256 | 1.35e-2 | 1.0x |
| /64 | 8.99e-4 | 0.067x |
| /32 | 2.25e-4 | 0.017x |
| **/16** | **5.62e-5** | **0.004x ≈ 1/256** |

The error ratio ≈ 240× ≈ (256/16)², matching the 1/s² theory exactly with no plateau. Early blocks (low residual energy, small `ms(q)`) show the largest error at s=256 (3.8e-2 ~ 1.05e-1), dropping to 2.7e-4 ~ 4.9e-4 at s=16.

### 3.4 Overflow lower bound: is smaller s safer or more dangerous?

Intuition says smaller s → larger `x/s` → higher overflow risk; but the **binding constraint is `out_proj`'s output** (the largest tensor in the fp16 stream), and it scales with `P@V`, which scales with `v`:

```
out_proj_out ≈ ||W_out|| · P@V(v/s)  ∝  1/s
```

The smaller s is, the *smaller* `out_proj`'s fp16 output is. Measured (small seq, worst of 52 layers):

| prescale | out_proj fp16 output max | headroom (vs 65504) |
|---|---|---|
| /256 | 120.7 | 543x |
| /16 | 1931.0 | **33.9x** |

Even at 480p/10s scale (out_proj output ~39k), s=16 keeps ~27× headroom; at the extreme 1344×768/124-frame peak of 1.43×10⁵ it still has 7.3×, with the fuse as backstop. **So s=16 is pure win: +240× accuracy, still ample overflow headroom.**

> Conclusion: `_ATTN_FIXED_SCALE = 16` (was 256). This is a community contribution (PR #1); derivation and measurements above.

---

## 4. MLP fixed scaling (fully fp16 since v6)

| Stage | Handling | Basis |
|---|---|---|
| fc1 | kept fp16 (max≈585) | far below 65504 |
| gated-silu | gate branch fixed ÷16 (2⁴) | silu product ≤ ~21.4k, 3.4× margin |
| fc2 | input fixed ÷8 (2³) | output ≤ ~22.7k |
| restore | multiply back in fp32 at the end | exact |

All power-of-2 scalings (lossless in fp16), zero fp32 intermediate tensors, zero per-block `.item()` scans — eliminating the ~850GB/step cast bandwidth on long sequences.

Long sequences (seq > 16384) are chunked automatically; activation peak stays ~9GB regardless of seq, avoiding lowvram eviction thrash on 16GB cards.

---

## 5. How to verify

### 5.1 Numeric level (no sampling, math only)

Verify §3.2/§3.3: on real activations' q (projected with fp32 weights, unscaled), compare

```
ref  = rms_norm(q)             = q / sqrt(ms(q) + eps)
q_s  = rms_norm(q/s) exact form = q / sqrt(ms(q) + eps·s²)
```

Sweep s and confirm rel err decreases monotonically as 1/s² with no plateau. This check does not depend on the rope implementation and is pure fp32 — reproducible on any machine.

### 5.2 Real-model level

1. Load a MiniMax H3 checkpoint (fp8_scaled / int8_convrot / bf16 all work) + turbo LoRA;
2. After `patch()`, capture the real activation `x` and `rope_freqs` at every Attention input;
3. Run the full fp16 chain (qkv → RMSNorm+rope → SDPA → out_proj → restore) on the same `x` with s=256/64/32/16;
4. Check: early-block output difference d(256,16)/max should be in the 1%~8% range (eps effect visible), deep blocks converge to fp16 ulp steps; at s=16 the `out_proj` output max should be far below 65504.

(A runnable script skeleton can be found in the repo's verification script `verify_prescale_pr1.py`.)

### 5.3 Final quality

Beyond numerics, compare real renders (same seed/params, only the scale differs):
- target output PSNR should be > 40dB (s=16 vs s=256);
- no visible difference to the eye;
- s=16 render never trips the fuse (no fp32 recompute warnings in the console).

---

## 6. Known issues and boundaries

### 6.1 [FIXED v6.6.0] Compute dtype persists after node removal (Issue #2)

**Symptom (v6.5.0 and earlier)**: `patch()` calls `model.set_model_compute_dtype(torch.float16)`, which writes `manual_cast_dtype` into the ModelPatcher's `object_patches` and sets `force_cast_weights` — this is **persistent state on the ModelPatcher**. ComfyUI caches the UNETLoader output, so:

```
run once with the node → remove the node → re-run in the same process
```

the model is **still forced to fp16 compute, but without the plugin's scaling compensation** (the forward patches are instance-level and disappear with newly-loaded model instances) → back to the fp16 NaN/black-frame condition.

**Fix (v6.6.0)**: `patch()` now **`clone()`s the ModelPatcher first and applies everything** (`set_model_compute_dtype` + instance-level forward patches) on the clone, returning it:

- `clone()` deep-copies `model_options` / `object_patches` / `force_cast_weights` → the **cached original ModelPatcher stays clean**, so re-running after removing the node goes back to native fp32 compute;
- the weight instance is shared (clone does not copy weights), but the forward patches are instance-level and `comfy.ops` auto-casts weights so fp16 inputs compute correctly;
- verified (`verify_issue2_clone.py`): after patch, the original object's `object_patches` is empty and `force_cast_weights` stays `False`; the fp16 state exists only on the clone.

> If `clone()` fails for external reasons, the plugin falls back to in-place patching and prints a WARNING (the cached object may then retain the fp16 compute dtype; a server restart is advised).

### 6.2 Boundaries

- **Video VAE**: native fp16 path kept (v6.4.0 measured finite even with ±38 outliers; 0.5% diff vs fp32). If a particular checkpoint decodes badly, suspect the VAE weights themselves first.
- **Qwen text path**: `condition_proj` is forced fp32 (protects against real-context overflow); speed impact negligible (text tokens ≪ video tokens).
- **Non-MiniMax models**: `_is_minimax_dit()` misses → the plugin passes through untouched.

---

## 7. Version history (development view)

| Version | Key decisions |
|---|---|
| v6.0.0 | MLP fully fp16 + gate fixed scaling, removes fp32 cast bandwidth; mathematically can't overflow |
| v6.3.0 | attention fixed ÷256 scaling, zero scans (q/k restored by RMSNorm homogeneity); per-layer syncs 11→2 |
| v6.4.0 | Video VAE back to native fp16 path (measured finite, no upcast needed) |
| v6.5.0 | instance-level patch (no more ghost patching at forward level); drop legacy node ID alias |
| v6.6.0 | **attention fixed scale 256 → 16** (PR #1, +240× accuracy, still ≥7× overflow headroom); **Issue #2 fixed** (patch runs on a clone; cached ModelPatcher no longer polluted with fp16 compute dtype) |

## License

MIT
