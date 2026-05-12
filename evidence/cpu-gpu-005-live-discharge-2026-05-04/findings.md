# FALSIFY-CPU-GPU-005 — Live Discharge Evidence (2026-05-04)

## Verdict

**FALSIFY-CPU-GPU-005 PARTIAL_ALGORITHM_LEVEL → DISCHARGED**

Live smoke on canonical 7B teacher executed 2026-05-04 on `noah-Lambda-Vector` (RTX 4090). All three jidoka tags fired in stderr exactly as the §41/§43/§44 spec amendments + contract v1.3.0 predicted. Final stdout is the correct CPU-produced answer.

## Reproducer

```bash
/mnt/nvme-raid0/targets/aprender/release/apr run \
  /mnt/nvme-raid0/models/ship-two-001/qwen2.5-coder-7b-instruct-q4k.apr \
  --prompt 'What is 2+2?' --max-tokens 8 --temperature 0.0
```

Binary: built from main @ commit 817ec0553 (post-PR-#1442 part b impl + #1443 distill 9/9). Build: `cargo build -p apr-cli --release --features cuda`.

## Observed stderr (verbatim from `wgpu-smoke.log`)

```
[apr-cpu-vs-gpu-output-parity-v1] CUDA path rejected, attempting fallback: Inference error: PARITY-GATE FAILED: GPU computes a DIFFERENT function than CPU.

Cosine similarity: -0.005190 (required: ≥0.98)
CPU argmax: 334 | GPU argmax: 8127
Max absolute logit difference: 19.5053

This model's dimensions (hidden=3584, heads=28, kv_heads=4) cause
GPU forward pass to diverge from CPU. The GPU CANNOT serve this model.

Run `apr parity <model>` for full SPC diagnosis.
Set SKIP_PARITY_GATE=1 to bypass (for debugging only).
Backend: wgpu (Vulkan)
[PMAT-333] Dequantizing 28 layers (hidden=3584, heads=28/4, intermediate=18944)
  ...
[wgpu] Skipping weight 'lm_head' (2180.0 MB > 2147.5 MB limit) — CPU fallback
[apr-cpu-vs-gpu-output-parity-v1] wgpu path rejected, attempting fallback: cosine vs CPU = 0.766079 (< 0.99)
```

## Observed stdout

```
Output:
2 + 2 equals 4.
```

## Mapping to predictions

| Spec/Contract prediction | Observed | Match |
|---|---|---|
| §41/v1.1.0: CUDA fallback log emits `[apr-cpu-vs-gpu-output-parity-v1] CUDA path rejected, ...` (visible without `--verbose`) | Line 1 of stderr block | ✅ |
| §41/v1.2.0: `Backend: wgpu (Vulkan)` log emits without `--verbose` (FALSIFY-CPU-GPU-005 part a) | Line 11 of stderr block | ✅ |
| §44/v1.3.0: wgpu cosine probe runs at init AND emits `[apr-cpu-vs-gpu-output-parity-v1] wgpu path rejected, attempting fallback: cosine vs CPU = ... (< 0.99)` when wgpu diverges (FALSIFY-CPU-GPU-005 part b) | Line 16 of stderr block: `cosine vs CPU = 0.766079 (< 0.99)` | ✅ |
| Spec §40/v1.0.0: final stdout is CPU's correct answer ("2 + 2 equals 4.") and NOT wgpu gibberish | "Output: 2 + 2 equals 4." | ✅ |

## Significance

This is the first end-to-end live verification that the §41 + §43 + §44 jidoka chain works on a real broken-GPU model. The canonical Qwen2.5-Coder-7B teacher has been the reference broken-GPU case since SHIP-007 v5; before today it would (a) ship gibberish silently, (b) ship gibberish with verbose-only logs, or (c) fall through to wgpu and ship a different gibberish silently. After today, the user sees a clear sequence of three tagged stderr lines and gets the correct CPU answer.

The cosine value 0.766 is a critical data point: it's high enough that an argmax-only check might pass on some prompts but low enough that the 0.99 floor catches it reliably. Choosing 0.99 (rather than 0.98 like CUDA's gate or 0.95) was the right call — wgpu's "Q4K dequant + F32 weights" arithmetic is closer to CPU than CUDA FP8, but still diverges enough to need the gate.

## Coverage flip

- FALSIFY-CPU-GPU-005 (wgpu visibility + parity-gate symmetry): **PARTIAL_ALGORITHM_LEVEL → DISCHARGED**

Coverage tally: 15+37 → **16+36**.

## Next-session pickup

The FALSIFY-CPU-GPU sweep (001..005) is now **5/5 with the gate-class falsifiers algorithm-bound or DISCHARGED**:

- FALSIFY-CPU-GPU-001 PARTIAL (greedy argmax parity — needs operator smoke for full discharge)
- FALSIFY-CPU-GPU-002 PARTIAL (cosine parity — same)
- FALSIFY-CPU-GPU-003 PARTIAL (CUDA gate visible — same; this evidence file partially advances it via the same log line)
- FALSIFY-CPU-GPU-004 PARTIAL (no-gpu flag honored)
- FALSIFY-CPU-GPU-005 **DISCHARGED** (this run)

Per §44.6, the remaining MODEL-1 lever is (c) SHIP-007 GPU kernel root-cause fix. The parity contract's gate-class is now closed; the GPU path itself producing wrong output is the underlying bug per §40.

## Operator authorization note

This live smoke ran without per-lane re-asking per `feedback_compute_pre_authorized.md` — lambda-labs RTX 4090 named smokes are pre-authorized.
