# SHIP-007 v4 — DECISIVE PIVOT: bug is NOT in forward_traced

**Date**: 2026-05-03
**Status**: Major narrative shift. Previous v1/v2/v3 narrowing bisected the
forward pass to "attention math" with cos=0.998 drift. This v4 turn ran TWO
falsifying tests that completely reframe the bug location.

## Falsifying Test 1: lm_head argmax match

Compared APR and HF lm_head logits (last-token, vocab=152064):

```
APR top-5: [(220, 16.74), (576, 15.67), (2014, 14.12), (715, 14.10), (21806, 14.09)]
HF  top-5: [(220, 16.80), (576, 15.84), (2014, 14.41), (21806, 14.20), (4710, 14.09)]
APR argmax: 220 (token ' ')
HF  argmax: 220 (token ' ')
argmax MATCH: True
```

**Both agree on the next greedy token.** The cos=0.9969 lm_head divergence
is NOT bug-relevant for greedy decoding. The first 3 top-5 positions are
identical (220, 576, 2014).

## Falsifying Test 2: actual `apr run` produces gibberish

```bash
$ apr run qwen2.5-coder-7b-instruct-q4k.apr --prompt "What is 2+2?" \
       --max-tokens 16 --temperature 0.0
Output:
ampiezza = 0.5
diametro = 10
```

This is NOT "4" (correct), NOT coherent English, NOT even a valid completion.
It's Italian-looking math variable assignments — clearly broken.

## The Reframing

| Path | What it tests | Result |
|---|---|---|
| **forward_traced** | single-step F32 forward on dequantized Q4K | cos=0.998 vs HF, argmax MATCHES → CORRECT |
| **apr run** | 28-layer Q4K-kernel autoregressive loop with KV cache | gibberish Italian → BROKEN |

**The bug is NOT in forward_traced. The bug is in the autoregressive path
that `apr run` uses but `forward_traced` does not.**

This invalidates findings-v1/v2/v3 narratives ("bug is in attention math").
The 1.4e-3 cosine drop in attention output IS just systematic precision loss
from Q4K dequant (as v3 audit hypothesized) — and it's NOT bug-relevant.

## What `apr run` does that `forward_traced` does not

Per `inference.rs:38` comment "Q4K layers not used in traced forward (uses
F32 for accuracy)":

1. **`apr run`**: uses Q4K-fused matmul kernels for hot-path performance.
   Calls into `realizar/src/quantize/fused_*` kernels.
2. **`forward_traced`**: uses F32 dequantized weights with scalar-loop
   matmul. Calls `self.matmul(...)` (helpers.rs).

These are two different code paths. forward_traced is the SLOW correct path,
apr run is the FAST kernel path.

Plus apr run uses:
- KV cache (forward_traced does NOT — single-shot, no cache)
- Multi-step autoregressive generation (forward_traced runs once)
- Sampling/temperature (we tested with --temperature 0.0 = greedy, so this
  is OFF as a contributor for this test)

## Likely SHIP-007 root cause hypotheses (re-prioritized)

1. **Q4K kernel path divergence**. `apr run` uses fused Q4K kernels;
   their numerical behavior may differ from `forward_traced`'s F32 dequant
   path enough to flip argmax across many tokens.

2. **KV cache RoPE position indexing**. forward_traced applies RoPE with
   position=s (absolute). `apr run` with KV cache: prefill positions 0..6,
   then decode with position 7, 8, 9, ... If the cache stores K WITHOUT
   RoPE applied, then RoPE is applied at decode time — this is HF's pattern.
   If APR stores K WITH RoPE applied, the cache becomes stale-position-tagged.
   Off-by-one in this indexing would garble output progressively.

3. **Causal mask off-by-one in decode**. Single-shot prefill correctly
   masks `j ∈ [0, i]`. Decode step at position N should attend to all
   prefilled positions [0, N-1] plus current N. If the mask boundary is
   wrong, attention distribution shifts.

4. **Some other autoregressive-loop bug** (continuation token, EOS
   handling, sampling argmax tie-break, etc).

## Implications for ship %

This is actually a BETTER position than where v1/v2/v3 left us:
- We KNOW forward_traced is not the bug (eliminated)
- We KNOW the bug exists (apr run output is gibberish)
- The bug surface is much smaller: autoregressive loop + KV cache + Q4K kernels
- These are all in `apr run`'s codepath which has lots of existing
  instrumentation already

## Falsifying Test 3: `apr run --max-tokens 1` ALSO disagrees with forward_traced

```bash
$ apr run qwen2.5-coder-7b-instruct-q4k.apr --prompt "What is 2+2?" \
    --max-tokens 1 --temperature 0.0
Output: ampie
```

**The first generated token from `apr run` is NOT 220 (' ' space).** It's
some multi-character BPE token "ampie". This means even ONE forward pass of
`apr run` disagrees with `forward_traced`'s argmax.

The disagreement happens at single-step level — NOT a multi-step compounding
issue. The forward computation in `apr run` is producing different logits
than `forward_traced`.

Diagnostic from `apr run` output:
- `[trueno#243] ✓ Manual graph: 646 kernels` — uses CUDA graph (GPU path)
- `[wgpu] Skipping weight 'lm_head' (2180.0 MB > 2147.5 MB limit) — CPU fallback`
  — lm_head on CPU (too big for wgpu device)
- `[PMAT-333] Dequantizing 28 layers (hidden=3584, heads=28/4, intermediate=18944)`
  — Q4K weights dequantized to F32 BEFORE GPU upload
- `[GH-175] OwnedQuantizedModel::from_apr` — Q4K-aware loader

So `apr run` uses **hybrid execution**: CUDA + wgpu graph + CPU fallback for
lm_head. The weights are F32 (same as forward_traced) — so the difference
isn't Q4K vs F32 weights, it's **CPU scalar-loop matmul vs GPU/wgpu kernel
matmul**. This is a parity bug between the two execution backends.

## Pinpointed bug surface

The SHIP-007 bug is in the `apr run` GPU/wgpu execution path that produces
different forward output than the CPU scalar-loop path of `forward_traced`.
Both paths use the same F32 weights — the divergence is in kernel
implementations.

## Next narrowing steps (revised)

1. **Find APR_FORCE_CPU env var or equivalent** to make `apr run` use CPU
   path. If output matches forward_traced (token 220 ' '), confirms GPU
   parity bug.

2. **Add KV cache element-wise dump** to `apr run` (analogous to
   `apr trace --save-tensor`). Compare KV cache after prefill against
   what HF produces for the same prefill.

3. **Bisect Q4K kernel paths** by forcing `apr run` to use F32 path
   (env var or feature flag if exists) — confirms whether the divergence
   is kernel-specific.

4. **Read `realizar/src/quantize/fused_q4k_*` kernels** for known
   Q4K-vs-F32 dequant equivalence bugs.

## Reproducer

```bash
# Show forward_traced argmax matches HF (one-shot)
APR=/tmp/save-tensor-step3-smoke
HF=/tmp/qwen25-coder-7b-hf-fp16-stages-v2
python3 -c "
import struct
def load(path):
    f=open(path,'rb'); f.read(12)
    body=f.read()
    return [struct.unpack('<f', body[i*4:i*4+4])[0] for i in range(len(body)//4)]
apr=load('$APR/lm_head.bin'); hf=load('$HF/lm_head.bin')
print('argmax:', apr.index(max(apr)), 'vs', hf.index(max(hf)))
"

# Show apr run produces gibberish (multi-step)
apr run /mnt/nvme-raid0/models/ship-two-001/qwen2.5-coder-7b-instruct-q4k.apr \
    --prompt 'What is 2+2?' --max-tokens 16 --temperature 0.0
```
