# Row 14: NEON for the qwen35 CPU gated-delta step

Read at origin/main 11f844a772. `Q:` cites are
`crates/aprender-serve/src/gguf/inference/forward/forward_qwen35.rs`. No build.

## Today
`delta_rule_head` (Q:280) runs per value head, per token, in four scalar passes over
the head's `head_v_dim x head_k_dim` state `s_h`:
1. `s_h *= exp(gate)`: read + write all of `s_h`.
2. `delta[j] = (v[j] - dot(row_j, k)) * beta`: read all of `s_h`, plus a
   `vec![0.0; head_v_dim]` heap allocation per head per token.
3. `row_j += k * delta[j]`: read + write all of `s_h`.
4. `out[j] = dot(row_j, q) * scale`: read all of `s_h`.
So 4 reads and 2 writes of the state, all scalar. Callers: the per-token path (Q:261)
and the batched prefill (Q:1341), which the doc comment says share this body on
purpose (#4228), so one change covers both.

## Finding R14-1: the four passes fuse into one pass per row
Row j's step 3 needs only `delta[j]`, and `delta[j]` needs only row j after step 1.
Step 4 on row j needs only row j after step 3. So per row:
`row *= g; d = (v[j] - dot(row, k)) * beta; row += k * d; out[j] = dot(row, q) * scale`.
One read and one write of the state instead of four and two, and no `delta` vector.
This is a scalar refactor with the same per-element operations; only the dot-product
summation order changes once SIMD lands.

## Finding R14-2: the in-tree SIMD helper does not fit the inner loop
`simd_dot` in serve (`crates/aprender-serve/src/inference/simd.rs:98`) builds two trueno
`Vector`s from slices per call. Called once per row (`head_v_dim` times per head per
token), its per-call setup is paid at row length `head_k_dim` [U, value in the 0.73
config]. The fused row kernel wants a slice-level NEON body: 4 or 8 `float32x4_t`
accumulators, `vfmaq_f32` for both dots and the axpy, `vmulq_n_f32` for the gate.
`brick/attention.rs:74` and `:147` (`simd_dot`, `simd_axpy`, aprender-compute,
`pub(crate)`) are slice-level; exposing them or a fused `delta_row` there is the
choice for the ticket. No `vfmaq_f32` exists in aprender-serve today.

## Finding R14-3: a second gated-delta body exists
`gated_delta_rule_head` (`crates/aprender-serve/src/gpu/scheduler/linear_attn.rs:671`,
called at `:665`) is a separate implementation of the same step. If row 14 speeds up
only Q:280, the two drift in summation order. The ticket should either route both
through the one fused body or name which is the reference.

## Draft for the row-14 ticket (lands with or after P3, same NEON idiom)
1. Fuse (R14-1) in scalar first, no SIMD. Each element gets the same operations in
   the same order, and each dot is still one left-to-right sum per row, so every
   qwen35 parity test passes bit for bit (Rust does not contract to FMA on its own).
2. A `delta_row` with a NEON body on aarch64 and an AVX2 body on x86 (R5 lists
   gated-delta as scalar on both), scalar elsewhere, dispatched once per call, not per
   row.
3. Trace: the step reports which body ran, so F-R5-1's "trace says NEON while scalar
   runs" cannot recur here.

## Falsifiers
| id | claim | how |
|---|---|---|
| F14-1 | the fused scalar body equals the four-pass body bit for bit | random state, k, v, q, beta, gate; `head_k_dim` 128 and an odd 37 |
| F14-2 | the NEON body equals the scalar body within a stated bound | max abs diff over 1000 tokens of recurrence, bound from BPM thresholds; state carried, so error growth is caught |
| F14-3 | a body that skips the gate is caught | gate = -10 must shrink the state; a NEON body without `vmulq_n_f32` fails |
| F14-4 | the dispatched body is the one the trace names | on aarch64 the trace says `neon`; with the NEON path disabled by a test hook it says `scalar` |
| F14-5 | E1 on C4 (gx10 CPU) still passes for Qwen3.5 | E1 cosine >= 0.995 |
| F14-6 | E2 on C4 moves | gated-delta step time per token, before vs after, in the E2 receipt |

## Open
- `head_k_dim` and `head_v_dim` for the 0.73 Qwen3.5-4B config [U].
- Which body `linear_attn.rs:671` serves (CUDA scheduler CPU side?) [U].
