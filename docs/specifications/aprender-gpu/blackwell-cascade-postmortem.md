# Blackwell GB10 Phase 3 Cascade Post-Mortem (PMAT-698e..m + PMAT-700-B)

**Date:** 2026-05-19
**Outcome:** Phase 3 distillation pipeline runs end-to-end on Blackwell GB10 sm_121 — first time.
**Duration:** Single session, 9 PRs landed.
**Run dir:** `gx10:/home/noah/runs/distill-smoke-20260519-191042/`

## TL;DR

8 PRs hunted a "Blackwell training is broken" symptom across 7 distinct defects. The actual root cause — a `warm!` macro that hardcoded a cache key, causing every "pre-warmed" kernel to silently collide on one hashmap entry — was a one-character bug invisible to all 5 prior single-kernel fixes. Diagnostic logging surfaced it in one pass; one-character fix unblocked the entire pipeline.

## Cascade

| # | PR | Defect | Class |
|---|-----|--------|-------|
| 1 | #1804 | PTX GEMM pre-warm wastes VRAM when cuBLAS active on sm_121 | Independent real bug |
| 2 | #1808 | `max_position_embeddings=32768` × 14 heads = 60 GB workspace per block | Independent real bug |
| 3 | #1809 | `apr distill` pipeline rejects APR-format weights as "invalid SafeTensors" | Independent real bug |
| 4 | #1810 | `pre_warm_lora_backward` short-circuits at `lora_rank==0`, missing shared kernels | Defense-in-depth |
| 5 | #1813 | `rms_norm_gamma_reduce` stage 2 not in backward pre-warm | Defense-in-depth |
| 6 | #1815 | (no defect — diagnostic logging) | Diagnostic infrastructure |
| **7** | **#1817** | **`warm!` macro hardcoded `"silu_forward"` as cache key for ALL kernels** | **Root cause** |
| 8 | #1820 | rmsnorm pre-warm key missing eps suffix; rope forward not pre-warmed | Hygiene |
| 9 | #1823 | Smoke setup: same input + distinct labels = unlearnable | Contract semantics |

## Five lessons extracted

### Lesson 1 — Symptom-similarity is a SIGNAL, not a search direction

When the second, third, and fourth iterations of "fix one missing kernel pre-warm" all surfaced the same downstream error, that was the cascade telling us the symptom-generator was upstream of every individual kernel. Stop adding kernels; instrument the cache. PMAT-698i's `[FWD-CACHE] Compiling '{name}'` logging in `get_or_compile` surfaced 11+ "pre-warmed" kernels all JIT-compiling at runtime, proving the cache held one entry — and that one entry was under the wrong key.

### Lesson 2 — Macros that take a cache-key argument deserve audit, not skim

The `warm!` macro definition was 6 lines:

```rust
macro_rules! warm {
    ($key:expr, $kernel:expr) => {{
        let ptx = $kernel.emit_ptx_for_target(&target);
        self.get_or_compile("silu_forward", &ptx)?;   // <-- HARDCODED
        count += 1;
    }};
}
```

Every caller passes `$key`. The body ignores it. This is plausibly a copy-paste from a working call site that originally only handled silu_forward, generalized to take an argument but not generalized to actually use it. Code review would catch this; auto-formatter and clippy do not. **In any cache-key-driven pattern, verify `$key` is substituted into the call, not just passed in.**

The mirror function in `cuda_backward/cache.rs` had the correct pattern (`let key = $key; cache.get_or_compile(&key, ...)`). Symmetry between forward and backward caches was BROKEN by this one line — and visible only at runtime.

### Lesson 3 — Pre-warm contracts have two halves: key + body

Even with the macro fixed, three kernels still cache-missed (PMAT-698k):

| Pre-warm key | Runtime key |
|--------------|-------------|
| `batched_rmsnorm_fwd_{h}` | `batched_rmsnorm_fwd_{h}_eps{bits:08x}` |
| (not present) | `batched_rope_fwd_{nh}_{hd}_{s}_th{bits:08x}` |

The PRE-WARM and RUNTIME each construct cache keys via separate `format!()` calls. They MUST stay synchronized. A property test asserting `pre_warm_keys() ⊇ runtime_keys()` for a known model config would catch this regression before merge.

### Lesson 4 — Blackwell sm_121 surfaces OLD bugs as NEW failures

Bugs in pre-warm machinery existed on sm_89 (RTX 4090) too — but JIT-on-demand at runtime SUCCEEDED there. sm_121's stricter behavior (JIT during active GPU work corrupts the stream) turned the same bug into a hard failure. **Cross-architecture validation will surface latent defects that single-architecture testing hides.**

This is the Blackwell pattern in microcosm: not "Blackwell is broken," but "Blackwell exposes pre-existing fragility we had been getting away with."

### Lesson 5 — Smoke contracts test the smoke, not the pipeline

PMAT-698m fixed a degenerate batch construction that had been silently invalidating the smoke contract — same input + distinct labels = unlearnable. The pipeline had been "working" all along in the sense of executing forward + backward + optimizer, but the convergence assertion (`final_loss < initial_loss`) could never pass on this data. **A contract that's logically impossible to satisfy under any execution is a contract bug.** The fixture-path test (`falsify_pipeline_001_end_to_end_loss_decreases`) was accidentally passing because the fixture's gradient happened to not blow up — but in cuda path with a real teacher, CE diverged immediately.

## Effort accounting

- 8 PRs landed
- ~7 hours of single-session debugging (autonomous mode, primary author + Claude)
- ~50 lines of net production code across all 8 PRs
- ~250 lines of comments + spec doc
- 1 root cause + 4 defense-in-depth + 2 independent + 1 contract semantics + 1 diagnostic

## What would have shortened this

Three interventions would have caught the root cause in one PR instead of seven:

1. **Diagnostic logging from the start.** Cache get_or_compile should ALWAYS log the kernel name when it compiles (per PMAT-698i). The cost is a single eprintln; the benefit is unmissable.

2. **Property test: pre-warm covers runtime keys.** A snapshot-style test that runs both `pre_warm_for_model` and a representative forward pass, then asserts the post-call cache contains every key the forward pass requested. This is the structural invariant that the cascade kept violating in different ways.

3. **Differential architecture testing.** Run the full forward pass on both sm_89 and sm_121 in CI (or at least on a recurring nightly), asserting cache state matches across architectures. Many of the missing-pre-warm gaps would have surfaced months ago.

## References

- `feedback_macro_cache_key_audit.md` (project memory rule)
- `project_blackwell_gb10_phase3_e2e_working.md` (project memory victory log)
- `docs/specifications/aprender-gpu/blackwell-backend-fix-spec.md` (SPEC-BLACKWELL-FIX-001 / PMAT-700)
- `evidence/distill-phase-3-post-698j/launch-victory.log` (the actual successful run)
