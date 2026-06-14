# KV-cache / paged-attention — adversarial bug-hunt + hand-triage (2026-06-14)

Adversarial bug-hunt over the KV-cache / paged-attention layer (5 dims: block alloc, position
indexing, GQA stride, eviction/reuse, quantized KV, prefill/decode). **27 findings → 9 marked
REAL, 1 uncertain, 17 refuted.** Each REAL verdict is being hand-re-checked before any fix (the
standing lesson — skeptics over-confirm ~⅓). Triage below.

## HAND-VERIFIED so far

### [1] (claimed CRITICAL) "decode attends to its own just-written K/V" — MISCHARACTERIZED
`gpu/scheduler/kv_forward_block.rs:70-74`. The finding called the current token attending to its
own K[N]/V[N] a "causal violation" — **that is wrong**: causal decode attends to `0..=N`
*including* the diagonal (a token sees itself). So the stated mechanism is a false alarm.
HOWEVER there IS a real, different anomaly: `append` (streaming_kv.rs:93-95) increments the
SHARED `valid_positions` ONLY on `layer == num_layers-1`, and `get_valid` reads that counter.
So during token N's forward, layers `0..L-2` read `valid_positions=N` → attend to `0..N-1`
(**EXCLUDING** the just-appended self), while the last layer reads `N+1` → includes self.
That per-layer INCONSISTENCY (earlier layers miss self-attention) is the real bug, not what the
finding said. Impact-limited: this is the **wgpu scheduler** decode path, which frequently fails
the cpu-vs-gpu parity gate (cosine < 0.99) and falls back to CPU. Fix is non-trivial (get_valid
should reflect the per-block physical length incl. the just-appended token, while valid_positions
tracks sequence length) + needs wgpu-path validation. DEFERRED — verify path liveness + design.

### [2] (low) GQA non-divisible head ratio — UNREACHABLE (agreed)
`aprender-gpu/.../incremental.rs:200` integer-division head mapping differs from the reference
only when `num_heads % num_kv_heads != 0`; no production model has that. Add a divisibility
assertion to the contract gate (defense-in-depth), not a runtime fix. Low priority.

## REAL — pending hand-verification (promising; verify next, likely fixable)

- **[6] (critical) Quantized-KV RMW scale asymmetry** `paged_kv/mod_quantized.rs:117-139`:
  partial-block `write_quantized_q8` dequantizes a block with the OLD scale, modifies some
  elements, re-quantizes with a NEW scale → the untouched elements in first/last partial blocks
  get re-encoded under a different scale (corruption at block boundaries). [7] is the Q4 twin
  (worse: 4-bit amplifies). HIGH-VALUE if the quantized paged path is live — verify next.
- **[3] (medium) / [4] (high) Stale data on block reuse**: `compact_sequence` (contiguous.rs:107)
  and `QuantizedPagedKvCache::allocate_sequence` (mod_quantized_paged.rs:54) reset `num_tokens`/
  `ref_count` but DON'T clear `keys`/`values` → a reused block carries a prior request's KV.
  num_tokens=0 means it *shouldn't* be read, so impact hinges on whether any reader trusts the
  buffer beyond num_tokens — verify the read paths before fixing.
- **[5] (medium) Prefix-cache page refs not validated on eviction** `mod_compute_prefix.rs:192`.
- **[8] (medium) Q8 quantize silent clamp** `mod_compute_prefix.rs:315` — clamps to [-127,127]
  without flagging saturation (precision-loss masking).
- **[9] (high) Causal mask at chunk boundaries** `apr_transformer/cache_attention.rs:67` —
  `seq_len = cache_len + 1` mask range; verify whether cache_len is off by one at chunk seams.

## UNCERTAIN (1)
- "Stale KV in extended blocks (extend)" `mod_paged.rs:106` — same clear-on-reuse family as
  [3]/[4]; verify together.

## Method / lesson
Hunt + skeptic, then HAND-VERIFY. The headline "CRITICAL" [1] was mischaracterized (self-
attention is not a causal violation) — fixing it as stated would have been wrong. Verify each
REAL finding (and whether its path is live: several are wgpu-scheduler / paged paths that may
fall back to CPU) before investing. Promising clear bugs to verify+fix next: [6]/[7] quantized
RMW scale asymmetry, [3]/[4] block-reuse clear.
