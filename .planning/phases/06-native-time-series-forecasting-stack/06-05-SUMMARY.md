---
phase: 06-native-time-series-forecasting-stack
plan: 05
subsystem: forecasting
tags: [chronos, chronos-bolt, t5, zero-shot, safetensors, gemm, blis, parity, contracts, pv, weights, time-series, rust]

requires:
  - phase: 06-native-time-series-forecasting-stack
    provides: "plan 06-01 landed crates/aprender-forecast (dates, types, test_support's load_json/read_csv/equation_tolerance/constant_u64, the strict parse_date), the committed Chronos oracle fixtures (chronos_bolt_tiny_fixture.json, chronos_probes.json, peyton_tiny_oracle.json) and the MEASURED [profile.dev.package.aprender-forecast] opt-level = 3 decision"
  - phase: 06-native-time-series-forecasting-stack
    provides: "plan 06-03 established the contract shape to copy (kind: kernel on the setfit-apr-v1 template), the contract-read bar pattern with the contract name written in full at every call site, the region-scoped tolerance-literal scan, and the observed-induced-RED discipline for a qa_gate falsification"
  - phase: 06-native-time-series-forecasting-stack
    provides: "plan 06-04 added `np` to the same `pub mod` block in src/lib.rs this plan extends, and established the ban-by-grep discipline (a ported file names its own bans in prose but never writes the banned identifier)"
provides:
  - "just fetch-chronos-tiny — a pinned, VERIFY-ALWAYS, sha256-checked weight fetch into the gitignored /models/, with a locally derived f16 and a proven tamper failure"
  - "crates/aprender-forecast/build.rs — CHRONOS_MODEL_DIR presence becomes cfg(chronos_weights), so weight-dependent tests are COUNTED skips when unarmed"
  - "crates/aprender-forecast/src/bolt.rs — the Chronos-Bolt T5 zero-shot forward, spike-007 verbatim minus the rayon arm, with the spike-005 ladder tensors merged into one forward body"
  - "crates/aprender-forecast/src/safetensors.rs — F32/F16/BF16 decode from one byte slice, shared by embedded and on-disk weights"
  - "crates/aprender-forecast/src/chronos.rs — the Chronos door: validate() (weights-free) then forecast(), ModelLoadError, ChronosArgs/ChronosResponse, load_csv"
  - "contracts/chronos-bolt-parity-v1.yaml — 18 equations, 18 obligations, 18 FALSIFY-CHRONOS tests, 2 Kani harnesses, qa_gate F-FORECAST-CHRONOS-001; pv validate 0 errors, pv status 18/18/18/2"
  - "The SC4 parity ladder: 7 weight-gated tests + 2 ungated, every bar contract-read, measured 9.5367e-7 on Peyton against the frozen 1.0e-6"
  - "The two-sided counted-skip proof OBSERVED in both directions (unarmed 7 ignored with the arming reason; armed 0 ignored)"
  - "The D-13 memory-clause amendment on the record, with its cost asserted rather than hidden (69.18 MB resident, 2.00x)"
affects: [06-06, 06-07, 06-08, 06-09]

actuals:
  tokens: 41400
  tasks: 4
  commits: 3

tech-stack:
  added: []
  patterns:
    - "A weight-dependent claim is proven in BOTH directions: unarmed it must show a non-zero ignored count with a visible reason, armed it must show zero ignored AND a minimum count of matching test lines"
    - "A numeric bar is ARCHITECTURE-KEYED when it was measured on one architecture: the test selects the equation on cfg!(target_arch) and PRINTS which equation, which value and which ARCH, so a green run always says where it ran"
    - "A tolerance that has never been measured is labelled PROVISIONAL in the contract and carries an obligation naming what the first real run must do with it — headroom is recorded as headroom, never as evidence"
    - "A rollout/scheme rung ships with its CONTROL: the superseded algorithm must FAIL against the same oracle by a stated margin, otherwise the rung proves only that some algorithm was implemented"
    - "A ported file names its own bans in prose but never writes the banned identifier, so `! grep -q` stays a real file-wide ban (extended here from 06-04's D-10 case to the rayon toggle, the print-a-skip style and the superseded contract key)"

key-files:
  created:
    - crates/aprender-forecast/src/bolt.rs
    - crates/aprender-forecast/src/safetensors.rs
    - crates/aprender-forecast/src/chronos.rs
    - crates/aprender-forecast/build.rs
    - crates/aprender-forecast/tests/fixtures/chronos_bolt_tiny_config.json
    - contracts/chronos-bolt-parity-v1.yaml
  modified:
    - justfile
    - Cargo.lock
    - README.md
    - crates/aprender-forecast/Cargo.toml
    - crates/aprender-forecast/README.md
    - crates/aprender-forecast/src/lib.rs
    - crates/aprender-forecast/tests/fixtures/README.md

key-decisions:
  - "D-13's memory clause AMENDED by the human's blocking decision `amend-memory-clause`: the shipped Bolt keeps BOTH weight layouts, because dot8 (D-14's single-row kernel) reads contiguous [out, in] rows while gemm_blis (D-14's multi-row kernel) reads the [in, out] transpose. The cost is asserted and printed (69.18 MB resident, 2.00x), not hidden"
  - "The enforced f16 sha256 is f5dc2ef5..., NOT the plan's f9a033b4...: RESEARCH assumption A3 FAILED and re-pinning to the value this toolchain reproduces is what keeps the check fail-closed"
  - "ONE forward body with three wrappers (D-08 merge): spike-005's ladder-returning forward and spike-007's timing-returning forward are unified into Bolt::forward_full, because two copies of the arithmetic the 9.54e-7 parity was measured through is two things to keep in sync"
  - "The validate()/forecast() split is a recorded D-08 structural deviation justified by D-11 + D-18: the door's refusals must be provable with no weights present, or the refusal claim itself becomes weight-gated"
  - "relative_buckets_match_fixture_over_grid is anchored on the 18 offsets spike 005 RECORDED against transformers' reference formula plus structural properties over the whole grid — the committed fixtures publish no bucket table, and a Rust-vs-Rust comparison would have been theatre (06-03's rule)"
  - "The x86_64 bar ships as a SEPARATE PROVISIONAL equation rather than as a loosened single bar, so SC4's 1e-6 is still asserted where it was measured and the unmeasured value is visibly unmeasured"

patterns-established:
  - "Contract-read bars with the contract name written in full at every call site (8 equation_tolerance + 6 constant_u64 sites), so `grep -c` is a real link check"
  - "Region-scoped tolerance-literal scan over `mod parity`, refusing to pass vacuously (it asserts it can LOCATE the module first) — and it FOUND a real finding on its first run"
  - "Every gate touched was proven to fire: clippy by a needless_bool mutation in bolt.rs, the literal scan by an injected const, the contract link by a YAML-only induced RED, the fetch recipe by a one-byte tamper"

requirements-completed: [SC4]

coverage:
  - id: D1
    description: "`just fetch-chronos-tiny` fetches amazon/chronos-bolt-tiny at the pinned revision into the gitignored /models/, verifies all three sha256s on EVERY run (not only on download), derives f16 locally, and exits non-zero naming the file when one does not match"
    requirement: SC4
    verification:
      - kind: other
        ref: "second, all-files-present `just fetch-chronos-tiny` run: rc=0, echoes all three pins, zero occurrences of `download` — /tmp/p06-05-t1d.log"
        status: pass
      - kind: other
        ref: "tamper control: one byte flipped at offset 4096 of an f32 model in a mktemp copy dir -> `just chronos_dir=$T fetch-chronos-tiny` rc=1, `FAIL: sha256 mismatch — a supply-chain event, not a cache miss` naming computed ddf18f9d... vs pinned 75068728... — /tmp/p06-05-t1e.log"
        status: pass
    human_judgment: false
  - id: D2
    description: "No model weight is committable: `git status --porcelain -- models/` is empty after a fetch and the weights are git-ignored via the root-anchored /models/ rule (CB-510)"
    verification:
      - kind: other
        ref: "`git status --porcelain -- models/` prints 0 lines; `git check-ignore -q models/chronos-bolt-tiny/f32/model.safetensors` exits 0"
        status: pass
    human_judgment: false
  - id: D3
    description: "The Chronos-Bolt zero-shot forward reproduces chronos-forecasting 2.3.1 on the Peyton Manning 64-step direct forecast within the SC4 bar, with every intermediate rung (mask, patch count, loc/scale, six hidden states) also barred"
    requirement: SC4
    verification:
      - kind: unit
        ref: "crates/aprender-forecast/src/bolt.rs#parity::peyton_ladder_matches_oracle_f32 — measured max|delta| 9.5367e-7 against `chronos-bolt-parity-v1.quantiles_abs_f32 = 1e-6 (ARCH=aarch64)`"
        status: pass
      - kind: unit
        ref: "crates/aprender-forecast/src/bolt.rs#parity::air_and_short100_ladders_match_oracle — air 1.8311e-4 = 1.532e-6 relative to scale 119.5490; short100 9.5367e-7 = 1.263e-6 relative to scale 0.7554"
        status: pass
    human_judgment: false
  - id: D4
    description: "The 365-step rollout reproduces the oracle AND the superseded median-only rollout does not — the control that pins which pipeline the fixture came from"
    requirement: SC4
    verification:
      - kind: unit
        ref: "bolt.rs#parity::rollout_365_matches_oracle_and_median_only_does_not — 1.907e-5 (bar 5e-5), first 64 steps 9.537e-7, 46 forwards read from the contract; median-only control 5.108e-1 against the SAME oracle (must exceed 1e-1)"
        status: pass
    human_judgment: false
  - id: D5
    description: "The six edge probes reproduce the oracle with identical attention masks, `constant` on an absolute bar because its scale is a floor rather than a datum"
    verification:
      - kind: unit
        ref: "bolt.rs#parity::edge_probes_match_oracle — nan_gaps 1.907e-6, five_points 1.907e-6, constant 4.768e-7 (abs bar 1e-5), huge_scale 1.000e0 at scale 7.36e5, tiny_len_130_rollout 2.861e-6, air_rollout_24 1.221e-4 at scale 119.55"
        status: pass
    human_judgment: false
  - id: D6
    description: "The locally derived f16 weights forecast within 2 % of the series standard deviation of the pinned f32 weights (SC4's f16 literal), with both dtypes asserted so the test cannot pass by comparing a file to itself"
    requirement: SC4
    verification:
      - kind: unit
        ref: "chronos.rs#parity::f16_weights_within_two_percent_of_std — 9.5463e-4 = 0.1095 % of the series std 0.8718 (bar 0.02 x std = 1.744e-2); dtypes F32 and F16, n_params equal"
        status: pass
    human_judgment: false
  - id: D7
    description: "The Chronos door refuses at every documented boundary as Validation (never Internal) with the fix named, and does so WITHOUT any weights present; a rolled-forward horizon carries a warning naming the rollout with the documented forward count"
    verification:
      - kind: unit
        ref: "chronos.rs#tests — 12 weights-free cases (too few points, horizon 0, horizon > 1024, long horizon without the flag naming `allow_long_horizon`, long horizon WITH the flag accepted, all-null y, impossible date, unsorted ds, freq H, trailing content in ds, strict schema, config fixture shape)"
        status: pass
      - kind: unit
        ref: "chronos.rs#parity::door_reports_warning_and_forwards_for_365 (warning contains `rollout`, forwards 46, rollouts 5; a native-horizon call has NO warning key, forwards 1, rollouts 0) and #parity::door_context_used_is_capped_at_2048 (n_history 3000 -> context_used 2048)"
        status: pass
    human_judgment: false
  - id: D8
    description: "D-14's routing is proven BEHAVIOURALLY in both clauses — a grep showing `dot8` exists cannot show which kernel a row reaches — and D-13's amended dual weight layout has its memory cost asserted and printed"
    verification:
      - kind: unit
        ref: "bolt.rs#tests::single_row_routing_is_dot8_at_production_defaults — bit-identical to dot8 on 20/20 seeds, different from the gemv path on 20/20"
        status: pass
      - kind: unit
        ref: "bolt.rs#tests::multi_row_routing_is_gemm_blis_at_production_defaults — bit-identical to gemm_blis on 20/20 seeds, different from the dot8 path on 20/20"
        status: pass
      - kind: unit
        ref: "bolt.rs#tests::weight_layout_is_dual_and_its_cost_is_stated — 8,652,672 f32 params = 34.61 MB; resident with BOTH layouts 17,295,232 f32 = 69.18 MB (2.00x); the config-derived count cross-checked against the model card and the safetensors header"
        status: pass
    human_judgment: false
  - id: D9
    description: "Weight-dependent tests are COUNTED skips when unarmed and real tests when armed, observed in BOTH directions (D-18, threat T-06-13)"
    verification:
      - kind: other
        ref: "unarmed `env -u CHRONOS_MODEL_DIR cargo test -p aprender-forecast --lib -- bolt::parity chronos::parity` -> `test result: ok. 1 passed; 0 failed; 7 ignored`, all 7 printing the arming reason"
        status: pass
      - kind: other
        ref: "armed `CHRONOS_MODEL_DIR=$PWD/models/chronos-bolt-tiny/f32 ...` -> `test result: ok. 8 passed; 0 failed; 0 ignored`, 9 matching `test (bolt|chronos)::parity::` lines, 84 s wall"
        status: pass
    human_judgment: false
  - id: D10
    description: "Every Chronos tolerance is read from contracts/chronos-bolt-parity-v1.yaml at test time (D-15); the contract validates with 0 errors and is not hollow"
    verification:
      - kind: other
        ref: "pv validate contracts/chronos-bolt-parity-v1.yaml -> rc=0, `0 error(s), 0 warning(s)`; pv status -> Equations 18 / Proof obligations 18 / Falsification tests 18 / Kani harnesses 2 / qa_gate F-FORECAST-CHRONOS-001 — no count is 0"
        status: pass
      - kind: other
        ref: "OBSERVED induced-RED: quantiles_abs_f32 1.0e-6 -> 1.0e-9 in the YAML ALONE turns peyton_ladder red quoting `9.5367431640625e-7 over the contract bar 1e-9`; byte-identical revert restores green — /tmp/p06-05-induced-red.log, /tmp/p06-05-restored.log"
        status: pass
      - kind: other
        ref: "8 `equation_tolerance(\"chronos-bolt-parity-v1\"` + 6 `constant_u64(\"chronos-bolt-parity-v1\"` sites; region-scoped literal scan of `mod parity` CLEAN and proven to fire by an injected `const MUTATION_PROBE: f64 = 1.0e-6;` (rc=1) with a byte-identical revert (rc=0)"
        status: pass
    human_judgment: false
  - id: D11
    description: "The x86_64 quantile bar is a SEPARATE, explicitly PROVISIONAL equation whose contract entry obliges the first real x86_64 run to record its measurement and tighten the bar (REVIEW-06-02)"
    verification: []
    human_judgment: true
    rationale: "No x86_64 run has happened. `quantiles_abs_f32_nonaarch64: 5.0e-6` is headroom chosen so a first CI run reports a NUMBER rather than a failure of unknown size, and the contract says so in those words — but nobody has yet confirmed that a 5x allowance is the right shape for the x86_64 accumulation order, and only a human (or the first CI run of 06-08's chronos leg) can close that. Until then this deliverable is a stated unknown, not a proven one."

duration: 46min
completed: 2026-09-06
status: complete
---

# Phase 06 Plan 05: Chronos-Bolt Zero-Shot Forward and Door Summary

**The Chronos-Bolt T5 now reproduces `chronos-forecasting` 2.3.1 in-tree to 9.5367e-7 on Peyton Manning against a frozen, contract-read 1.0e-6 — proven by a nine-rung ladder that is a COUNTED skip without weights and a real test with them, both directions observed.**

D-13 decision: amend-memory-clause
D-13 decision (verbatim): amend-memory-clause

## Performance

- **Duration:** 46 min for the continuation segment (Tasks 3-4). Task 1 was executed by a previous
  executor and committed at 2026-09-06T06:30:48Z; the plan then STOPPED at Task 2's blocking human
  `checkpoint:decision` and this agent resumed after the answer, so the plan's wall clock includes a
  human decision wait that is not executor time.
- **Started (this segment):** ~2026-09-06T12:55:00Z
- **Completed:** 2026-09-06T13:41:11Z
- **Tasks:** 4 (1 by the previous executor, 1 human decision, 2 here)
- **Files created/modified:** 13

## Accomplishments

- **`contracts/chronos-bolt-parity-v1.yaml`** — 18 equations (9 carrying a `float_tolerance`, 9
  structural invariants), 18 proof obligations, 18 `FALSIFY-CHRONOS-0NN` tests each naming the exact
  `cargo test` filter or shell run that discharges it, 2 Kani harnesses under the verbatim
  DECLARED-NOT-EXECUTED house comment, the `F-FORECAST-CHRONOS-001` qa_gate and a top-level
  `constants:` map. `pv validate` 0 errors; `pv status` **18 / 18 / 18 / 2** — no count is 0, so
  PROVABILITY-001 fires (RESEARCH F4's hollow-contract trap avoided).
- **The SC4 ladder is complete and every rung reproduces the spike measurement.** Peyton
  `quantiles_64` **9.5367e-7** against the frozen **1.0e-6**; the 365-step rollout **1.907e-5**
  against 5e-5; f16 **0.1095 %** of the series std against the 2 % bar; all six edge probes inside
  contract with attention masks EQUAL.
- **The median-only control is what makes the rollout rung evidence.** The shipped nine-path
  re-quantiled scheme lands at 1.907e-5; the pre-2025 median-only scheme lands at **5.108e-1**
  against the *identical* oracle — a 27,000x separation. Without it the rung would prove only that
  some rollout was implemented.
- **The counted-skip proof was OBSERVED in both directions**, not asserted: unarmed
  `ok. 1 passed; 0 failed; 7 ignored` with all seven printing
  ``ignored, CHRONOS_MODEL_DIR unset or has no model.safetensors — run `just fetch-chronos-tiny` to arm``;
  armed `ok. 8 passed; 0 failed; 0 ignored` with 9 matching parity lines.
- **D-14's routing is proven behaviourally in both clauses.** `single_row_routing_is_dot8_at_production_defaults`
  and `multi_row_routing_is_gemm_blis_at_production_defaults` each assert bit-identity to the
  mandated kernel on 20/20 seeds AND a difference from the other kernel on 20/20 — because a
  bit-identity assertion alone would be vacuous on a host where the two kernels happened to agree.
- **D-13's amendment is on the record with its cost asserted**, not footnoted: 8,642,560 of the tiny
  model's 8,652,672 f32 parameters are matrices held twice, so residency is 69.18 MB rather than
  34.61 MB — exactly 2.00x, recomputed from the config alone and printed.
- **The root `readme_contract` drift gate is now fully green** — 15/15, up from 11/15 before Phase 6.

## Task Commits

1. **Task 1: `just fetch-chronos-tiny` (pinned, sha256-checked, f16 derived) + `build.rs` cfg emission + the committed config fixture** — `21732582c` (feat) *(previous executor)*
2. **Task 2: DECISION — D-13's memory clause vs D-14's single-row routing** — no commit; the human returned `amend-memory-clause`, recorded on this file's `D-13 decision:` line before any `bolt.rs` byte was written and committed with Task 3
3. **Task 3: Port `bolt.rs` + `safetensors.rs` and build the Chronos door with a weights-free `validate()` seam** — `18021082d` (feat)
4. **Task 4: The contract, the gated Bolt parity ladder and the two-sided counted-skip proof** — `92584e902` (test)

## Files Created/Modified

| File | What it does |
|---|---|
| `justfile` | **Task 1.** `chronos_rev` / `chronos_dir` / `chronos_abs` vars + the `fetch-chronos-tiny` shebang recipe: conditional download, UNCONDITIONAL hashing of all three files, f16 re-derivation, both f16 shas documented in the header |
| `crates/aprender-forecast/build.rs` | **NEW, Task 1.** `cargo::rustc-check-cfg=cfg(chronos_weights)`, `rerun-if-env-changed=CHRONOS_MODEL_DIR`, and `rustc-cfg=chronos_weights` iff the dir holds a `model.safetensors` |
| `crates/aprender-forecast/tests/fixtures/chronos_bolt_tiny_config.json` | **NEW, Task 1.** The 1.1 KB hyper-parameter config (D-18 permits it; weights are never committed) — what makes every door test weights-free |
| `crates/aprender-forecast/src/safetensors.rs` | **NEW.** `Tensor` / `Weights` / `load_bytes` — F32/F16/BF16 to f32 from one byte slice, shared by the embedded copy and an on-disk file |
| `crates/aprender-forecast/src/bolt.rs` | **NEW.** `Config`, `linear`, `dot8`, `linear_fast`, `proj`, `t5_layer_norm`, `relative_bucket`, `transpose`, `SCALE_FLOOR`, `Attn`/`Block`/`Stack`/`Residual`/`Bolt`/`Forward`, `Bolt::{load, attention, position_bias, run_stack, forward, forward_stages, forward_ladder, forward_full, predict}`, `torch_quantile`; plus `mod tests` (4) and `mod parity` (5) |
| `crates/aprender-forecast/src/chronos.rs` | **NEW.** `CHRONOS_MIN_POINTS/MAX_POINTS/MAX_HORIZON`, `Model`, `ModelLoadError`, `load_model_from_bytes/dir`, `ChronosArgs`, `Validated`, `ChronosResponse`, `validate`, `forecast`, `load_csv`; plus `mod tests` (12) and `mod parity` (3) |
| `contracts/chronos-bolt-parity-v1.yaml` | **NEW.** The frozen SC4 bar and the weights/skip/routing invariants |
| `crates/aprender-forecast/src/lib.rs` | `pub mod bolt; pub mod chronos; pub mod safetensors;` added to the block 06-04 also appends to, with a doc paragraph saying why Chronos is a SECOND door and not a `model:` arm of the first |
| `crates/aprender-forecast/Cargo.toml`, `Cargo.lock` | **Task 1.** `half = { workspace = true }` for the F16/BF16 decode; no new registry package |
| `crates/aprender-forecast/README.md`, `.../tests/fixtures/README.md` | **Task 1.** The repo id, revision, all three sha256s, the Apache-2.0 licence and both env vars |
| `README.md` | Provable-contract count 1778 -> 1789 (see deviation 3) |

## Measured results

### The ladder (aarch64, `quantiles_abs_f32`, ARCH printed by the test)

| Series | loc rel | scale rel | embeds first/last/REG | encoder first/REG | decoder hidden | quantiles_64 |
|---|---|---|---|---|---|---|
| peyton (2905) | 1.158e-7 | 2.051e-7 | 4.40e-7 / 2.38e-7 / **0.0e0** | 1.13e-6 / 6.56e-7 | 4.768e-6 | **9.5367e-7** (bar 1e-6) |
| air (144) | 0.0e0 | 1.276e-7 | 2.83e-7 / 2.68e-7 / **0.0e0** | 7.15e-7 / 2.98e-7 | 1.907e-6 | 1.8311e-4 = **1.532e-6 rel** (bar 4e-6) |
| short100 | 0.0e0 | 7.891e-8 | 2.38e-7 / 3.43e-7 / **0.0e0** | 2.68e-7 / 2.68e-7 | 1.431e-6 | 9.5367e-7 = **1.263e-6 rel** |

Bars: `loc_scale_rel` 2e-6, `hidden_states_abs` 2e-5. The REG embedding is **exactly 0.0e0** on all
three series, as it must be — it is a slice of `shared.weight`, not a computed value, so a non-zero
difference there would mean the REG index moved rather than that a kernel drifted.

**Tightest margin on the whole ladder:** Peyton's `quantiles_64` at 9.5367e-7 against 1.0e-6 —
**4.6 %**. That is the SC4 literal and it is the number REVIEW-06-02 is about.

### Rollout, probes, f16

| Rung | Measured | Bar |
|---|---|---|
| rollout 365, all steps | **1.907e-5** | 5.0e-5 |
| rollout 365, first 64 steps | 9.537e-7 | (matches the direct forecast, as it must) |
| forwards / rollouts | **46 / 5** | read from `constants.rollout_forwards_for_365` |
| median-only CONTROL | **5.108e-1** | must EXCEED 1e-1 |
| f16 vs f32 | **9.5463e-4 = 0.1095 % of std 0.8718** | 2 % of std |
| probe nan_gaps (300 -> 64) | 1.907e-6 | 5e-6 x scale 0.7479 = 3.74e-6 |
| probe five_points (5 -> 64) | 1.907e-6 | 5e-6 x scale 0.8052 = 4.03e-6 |
| probe constant (64 -> 64) | 4.768e-7 | **absolute** 1e-5 |
| probe huge_scale (200 -> 64) | 1.000e0 | 5e-6 x scale 7.36e5 = 3.68e0 |
| probe tiny_len_130_rollout | 2.861e-6 | 5e-6 x scale 0.7096 = 3.55e-6 |
| probe air_rollout_24 | 1.221e-4 | 5e-6 x scale 119.55 = 5.98e-4 |
| `relative_bucket` | 18 recorded offsets EXACT, 4097 grid points x 2 modes inside [0, 32) | — |
| door context cap | n_history 3000 -> context_used **2048** | `constants.context_length` |

`tiny_len_130_rollout` is the thinnest probe margin (2.861e-6 against 3.55e-6, 1.24x) and is the one
to watch: it is the shortest series that triggers a rollout block, so it carries both the
short-context and the rollout error together.

### The two-sided skip proof (verbatim)

```
UNARMED: test result: ok. 1 passed; 0 failed; 7 ignored; 0 measured; 76 filtered out; finished in 0.00s
ARMED:   test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 76 filtered out; finished in 71.30s
```

Every one of the seven unarmed lines reads
``... ignored, CHRONOS_MODEL_DIR unset or has no model.safetensors — run `just fetch-chronos-tiny` to arm``.
The armed run produced **9** matching `^test (bolt|chronos)::parity::` lines (bar: >= 8) in an 84 s
wall. **CI pays none of that**: CI is unarmed, so the seven expensive rungs are counted skips there
and only the two ungated ones run.

### The f16 sha and the second fetch run

```
f32/config.json        278f0086...785e0  pin 278f0086...785e0
f32/model.safetensors  75068728...030a32  pin 75068728...030a32
f16/model.safetensors  f5dc2ef5...a74a67  pin f5dc2ef5...a74a67
      (spike 007 recorded f9a033b4... — same tensors, __metadata__ key order differs)
```

rc=0, zero occurrences of `download` — the file-present path re-hashes rather than trusting.
**Tamper control** (one byte flipped at offset 4096 of an f32 model in a `mktemp` copy dir): rc=1,
`FAIL: sha256 mismatch — a supply-chain event, not a cache miss`, naming computed
`ddf18f9d00def10fba92408c7807b38060f3d0f0081612cf13fd4feddeba8754` against the pin. The real
`models/` tree was never touched.

## Decisions Made

**D-13 AMENDED — the human returned `amend-memory-clause`.** The shipped `Bolt` keeps BOTH weight
layouts. D-13's sentence "Only transposed `[in, out]` weights are kept in memory" cannot hold beside
D-14's "single rows through the 8-accumulator dot", because `dot8` reads a contiguous `[out, in]`
row. Deleting the untransposed copies would force single rows onto `gemv`, violating D-14's second
clause, and would change the float accumulation order through a 12-layer T5 — under which the
9.54e-7 measurement was taken. Confirmed rather than assumed: this tree's own armed run reproduces
**9.5367e-7**, so the frozen 1.0e-6 bar remains evidence.

The cost is **disclosed and asserted**, per the decision's own terms:
`weight_layout_is_dual_and_its_cost_is_stated` prints
`8652672 f32 parameters = 34.61 MB; resident with BOTH layouts 17295232 f32 = 69.18 MB (2.00x)`,
recomputed from the committed config with no weights loaded, and cross-checks the derived parameter
count against the 8,652,672 that the model card AND the safetensors header both report. SC4's
"< 30 MB" is a BINARY-SIZE bar (embedded f16 bytes) and is untouched by an in-memory duplicate; the
contract states that distinction in `weights_dual_layout_documented` rather than leaving it to be
re-derived.

**The x86_64 bar is a separate PROVISIONAL equation, not a loosened single one.** SC4 names 1e-6 and
aarch64 asserts 1e-6. `quantiles_abs_f32_nonaarch64: 5.0e-6` is labelled, in the contract's own
words, PROVISIONAL AND UNMEASURED, with an obligation (FALSIFY-CHRONOS-002) naming what the first
x86_64 run must do with it in BOTH directions — tighten it to the measurement, or, if the
measurement exceeds 5e-6, investigate rather than raise the bar to fit the observation.

**`relative_buckets_match_fixture_over_grid` is anchored on the recorded offsets, not on itself.**
The committed fixtures publish no bucket table, so 06-03's rule applies: where the oracle publishes
no number, do not write a Rust-vs-Rust test. The reference here is the **18 offsets spike 005
recorded in `RUN-OUTPUT.md` §1** against transformers' `_relative_position_bucket`, plus structural
properties (index bound, causal-never-looks-forward, bidirectional upper half) over the whole
`[-2048, 2048]` grid in both modes. That asymmetry is stated in the test's doc comment.

**One forward body, three wrappers.** Spike 005's `forward` returns the ladder tensors; spike 007's
returns stage timings. Keeping both as separate bodies would be two copies of the arithmetic the
parity number was measured through. `Bolt::forward_full` is the one body; `forward` (quantiles),
`forward_stages` (007's exact signature, for 06-07's `--bench`) and `forward_ladder` (005's struct)
are thin wrappers. No operation, order or rounding moved.

**The `validate()` / `forecast()` split — a recorded D-08 structural deviation.** The spike's
`forecast()` validation prefix (its first 14 statements, through `future_days`) is extracted into
`pub fn validate(cfg, args) -> Result<Validated, ForecastError>`, with **every message verbatim**.
Justification is D-11 + D-18 together: D-11 says the boundary refuses rather than defaults, and D-18
says a claim must not pass only because weights were absent — the converse of which is that a *door*
claim must not need weights to be checked at all. Without the seam, all twelve refusal tests would
have been weight-gated and CI (which is unarmed) would assert none of them. `forecast` calls
`validate` first and is otherwise unchanged, so no refusal exists in two places (OPS-03).

**The two-positional-filter test commands were KEPT.** The plan's `<review_dispositions>` records
codex's HIGH finding that `cargo test -- bolt::parity chronos::parity` forwards the second filter as
a free argument, and records the orchestrator's refutation by experiment (modern libtest UNIONS
positional filters). This executor did not re-litigate it and did not "fix" it: both the unarmed and
the armed runs above show tests from BOTH modules, and the armed run's `>= 8` matching-line count is
what would go red if a filter were ever silently dropped.

## Deviations from Plan

### Carried forward from Task 1 (classified by the previous executor)

**1. [Rule 3 - Blocking] The enforced f16 sha256 is `f5dc2ef5...`, NOT the plan's `f9a033b4...`**
- **Found during:** Task 1 (fetch recipe), by the previous executor.
- **Issue:** RESEARCH assumption **A3 FAILED**. `safetensors` 0.8.0 cannot reproduce spike-007's f16
  file byte-for-byte.
- **Measured, not guessed:** the two f16 files have the same 11,640-byte header length, the same 101
  tensor entries in the same order, and a **byte-identical 17,305,344-byte body**. They differ only
  in the order of the two keys inside `__metadata__` (the spike's writer emits `converted` first,
  0.8.0 emits `format` first) regardless of the dict order passed in.
- **Fix:** re-pin to the value this toolchain reproduces. Loosening the f16 check to a warning would
  have deleted the very property REVIEW-06-03 asked for; re-pinning keeps it fail-closed.
- **Both values are pinned, printed and documented** in the justfile header, the crate README and
  now `contracts/chronos-bolt-parity-v1.yaml`, so the plan's `f9a033b4...` acceptance grep still
  holds and the discrepancy is on the record rather than in a commit message.
- **Committed in:** `21732582c`.

### This segment

**2. [Rule 1 - Bug] The plan's `cfg_attr` acceptance grep cannot coexist with `cargo fmt`**
- **Found during:** Task 4, running the acceptance criteria.
- **Issue:** the criterion is
  ``grep -c 'cfg_attr(not(chronos_weights), ignore' <bolt.rs> <chronos.rs>` totals >= 7``. rustfmt in
  this repo formats a two-argument `cfg_attr` **vertically regardless of line width** — measured: a
  98-character single-line form (with a deliberately shortened reason) was still split into four
  lines by `cargo fmt`, and `cargo fmt --all -- --check` is itself a gate. The two requirements are
  not simultaneously satisfiable.
- **Fix:** keep the RESEARCH-F6-verbatim, fully informative reason (which the plan's own `<verify>`
  greps as `ignored, CHRONOS_MODEL_DIR unset`, and which `key_links` specifies), and verify the
  criterion's INTENT with a multiline-aware count instead: `grep -c '^        not(chronos_weights),$'`
  gives **bolt 4 + chronos 3 = 7** gated tests. The stronger evidence is the unarmed run itself,
  which reports exactly **7 ignored** with the reason printed on every line — that is the property
  the criterion was proxying for. A shortened message chosen to satisfy a grep would have been
  optimising the measurement rather than the thing measured.
- **Verification:** unarmed `7 ignored`; the plan's `grep -q 'ignored, CHRONOS_MODEL_DIR unset'`
  passes; `cargo fmt --all -- --check` rc=0.
- **Committed in:** `92584e902`.

**3. [Rule 1 - Bug] `README.md` provable-contract count 1778 -> 1789**
- **Found during:** Task 4, running the `readme_contract` drift gate after adding a contract.
- **Issue:** this plan adds `contracts/chronos-bolt-parity-v1.yaml`, moving
  `find contracts -name '*.yaml' | wc -l` to 1789 while the README claims table still said 1778.
  `FALSIFY-README-007` was the **last remaining** failure in that target (14 passed / 1 failed).
- **Fix:** one-line correction, applying 06-01's own stated rule — the plan that widens a drift gate
  closes it. `readme_contract` is now **15 passed / 0 failed**, the first time in this phase (it was
  11/15 before Phase 6 and 12/15 after 06-01).
- **Note:** the fix also absorbs the 8 counts of pre-existing drift 06-01 declined to touch, because
  it is the same single number; there is no way to close the part this plan caused without closing
  the rest. `README.md` was not in the plan's `files_modified`.
- **Committed in:** `92584e902`.

**4. [Rule 2 - Missing critical] `bolt::tests::multi_row_routing_is_gemm_blis_at_production_defaults` added**
- **Found during:** Task 4, writing `gemm_routing_blis`'s falsification test.
- **Issue:** the plan specifies a behavioural proof for D-14's SECOND clause (single rows -> `dot8`)
  but leaves the FIRST clause (multi-row -> `gemm_blis`) to a source grep, which is exactly the
  weakness REVIEW-06-04 identified for the other clause. `FALSIFY-CHRONOS-014` would otherwise have
  had no runnable test.
- **Fix:** the symmetric twin — bit-identical to `linear_fast` on 20/20 seeds, different from
  `linear` on 20/20, count printed.
- **Committed in:** `92584e902`.

**5. [Rule 1 - Bug, self-inflicted] Three banned identifiers written in my own prose**
- **Found during:** Task 3 and Task 4, running the acceptance greps.
- **Issue:** the plan bans `PARALLEL_GEMM` and `println!("SKIP` in `bolt.rs`, and
  `weights_transposed_only` in the contract — as *file-wide* greps. My own explanatory prose wrote
  all three while explaining that they were removed, turning three green criteria red.
- **Fix:** reword to name the ban without writing the token (06-04's discipline, extended here to
  three more cases), and say so in the text so the next reader knows the omission is deliberate.
- **Recorded because** the general rule was already in the plan and was still violated three times
  in one session, which is precisely the pattern CLAUDE.md's Verification Discipline #7 describes.

**6. [Rule 1 - Bug] `door_context_used_is_capped_at_2048` first slice-panicked on the real CSV**
- **Found during:** Task 4, the first armed run (1 of 9 tests failed).
- **Issue:** the test took the first 3000 rows of `peyton_manning.csv`, which has **2905**.
- **Fix:** a synthetic 3000-point daily series built from `dates::days_from_civil` / `format_ymd`.
  Only the LENGTH is under test, so a synthetic series is the honest input; the plan asked for 3000
  points and 3000 points is what it now gets.
- **Committed in:** `92584e902`.

**7. [Out of scope - fixed, mechanical] Two D-08 clippy edits in the ported `bolt.rs`**
- `clippy::needless_for_each` on the two score/context zeroing loops
  (`x.iter_mut().for_each(|s| *s = 0.0)` -> `for s in x.iter_mut() { *s = 0.0; }`). Same zeroing,
  same order, no arithmetic touched. Each is annotated in the source as a D-08 clippy edit.
- `SCALE_FLOOR` was introduced as a named `pub const` for the instance-norm `1e-5` floor, because
  the region-scoped literal scan (correctly) flagged the bare literal in the parity module's
  loc/scale denominator. One definition now serves the forward and the test.

---

**Total deviations:** 7 — 1 carried from Task 1 (Rule 3 blocking, the failed A3 assumption), 4
auto-fixed here (2 Rule 1 bugs, 1 Rule 1 self-inflicted prose defect, 1 Rule 2 missing test), 1
acceptance-criterion conflict resolved in favour of the measured property, 1 mechanical D-08 batch.
**Impact on plan:** No bar was loosened and no rung was thinned. One acceptance grep was replaced by
a stronger measurement of the same property; one gate the plan did not own (`readme_contract`) went
from red to green.

## Gates run

| Gate | Result |
|---|---|
| `pv validate contracts/chronos-bolt-parity-v1.yaml` | **rc=0**, `0 error(s), 0 warning(s)` |
| `pv status contracts/chronos-bolt-parity-v1.yaml` | Equations **18** / obligations **18** / falsification **18** / Kani **2** / qa_gate `F-FORECAST-CHRONOS-001` — no count 0 |
| `cargo test -p aprender-forecast --lib chronos::tests` | **ok. 12 passed; 0 failed; 0 ignored** (bar: >= 11) |
| `cargo test -p aprender-forecast --lib bolt::tests -- --nocapture` | **ok. 4 passed; 0 failed; 0 ignored** |
| UNARMED `-- bolt::parity chronos::parity` | **ok. 1 passed; 0 failed; 7 ignored**, arming reason on all 7 |
| ARMED `-- bolt::parity chronos::parity` | **ok. 8 passed; 0 failed; 0 ignored**, 9 parity lines, 84 s |
| `cargo test -p aprender-forecast --lib` (unarmed, what CI runs) | **ok. 77 passed; 0 failed; 7 ignored** (60 before this plan) |
| Induced-RED through the contract ALONE (`quantiles_abs_f32` 1.0e-6 -> 1.0e-9) | **1 failed** quoting `9.5367431640625e-7 over the contract bar 1e-9`; byte-identical revert -> **1 passed** |
| Region-scoped tolerance-literal scan of `mod parity` | **CLEAN**; found 5 real findings on its first run, and proven to still fire by an injected `const MUTATION_PROBE: f64 = 1.0e-6;` (rc=1) with a byte-identical revert (rc=0) |
| `cargo clippy -p aprender-forecast --all-targets --no-deps -- -D warnings` | **rc=0**; engagement re-proven IN bolt.rs by a `needless_bool` mutation (rc=101, lint named) and a byte-identical revert (rc=0) |
| `cargo fmt --all -- --check` | **rc=0** |
| `just fetch-chronos-tiny`, second files-present run | **rc=0**, all three pins echoed, 0 occurrences of `download` |
| Tamper control (1 byte flipped, copy dir) | **rc=1**, `FAIL: sha256 mismatch — a supply-chain event, not a cache miss` |
| `git status --porcelain -- models/` / `git check-ignore` | **0 lines** / **exit 0** |
| `cargo test -p aprender-core --test readme_contract` | **ok. 15 passed; 0 failed** (was 14/1 before deviation 3, 11/4 before Phase 6) |
| D-05 guard: `crates/aprender-compute` | **untouched** in the working tree AND across `83e003f90..HEAD` |
| Task 3 acceptance criteria | **28/28** |
| Task 4 acceptance criteria | **23/23** (see deviation 2 for the one grep form replaced) |

`--no-deps` is 06-01's recorded, measured scoping (D-ITEM-06-01-b), not this plan loosening a gate.

## Issues Encountered

- **The armed ladder costs 84 s on this host, dominated by `aprender-compute` being built at
  `opt-level = 0`.** The `[profile.dev.package.aprender-forecast] opt-level = 3` override 06-01
  measured does NOT reach `trueno`, and `gemm_blis` unoptimised makes one 129-token Peyton forward
  take **1.68 s** (measured directly with a throwaway probe) against 137 ms in the spike's release
  build. The 46-forward 365-step rollout is therefore ~78 s of the 84 s wall and trips nextest's
  60 s SLOW threshold. **This is deliberately NOT fixed here:** CI is unarmed, so CI pays none of it,
  and the root `Cargo.toml` is closed for this phase (06-01 Task 2's decision, respected by 06-03 and
  06-04). Anyone who wants a fast armed run should add
  `[profile.dev.package.aprender-compute] opt-level = 3` as its own measured decision, not as a
  side-effect of this plan. Logged for 06-09.
- **The `rtk` Bash hook rewrites `cargo test` and strips both the `test result:` line and `println!`
  output**, exactly as 06-01 and 06-03 recorded. Every verification command here was run through
  `rtk proxy cargo ...`.
- **`.pv/contracts.idx`, `.pv/contracts.idx.mtime` and `.pv/lint-previous.json` go dirty on every
  `pv` run** and were restored to HEAD rather than committed, following 06-03 Deviation 4.

## Known Stubs

**None introduced by this plan.** No hardcoded empty value flows to a response, no placeholder text,
and no test can pass without its input: fixtures are `expect`ed, the armed model is `expect`ed
(absence when `chronos_weights` is set is a defect, never a skip), and the seven weight-dependent
tests are `cfg_attr`-ignored rather than early-returning, so an unarmed run reports them as a
non-zero COUNT rather than as a silent green.

Two things are deliberately NOT asserted and are recorded rather than hidden:

- **`quantiles_abs_f32_nonaarch64 = 5.0e-6` has never been measured.** It is labelled PROVISIONAL in
  the contract, selected only off aarch64, and carries FALSIFY-CHRONOS-002 naming what the first
  x86_64 run owes it. Surfaced in `coverage.D11` for human confirmation.
- **`relative_bucket` has no Python-generated oracle table.** The 18 anchor offsets come from
  spike 005's recorded run against the reference formula, not from a committed fixture; the test's
  doc comment says so.

## Threat Flags

None. No file created or modified here introduces security-relevant surface outside the plan's
`<threat_model>`. This library opens no socket, spawns no process and reads only files the caller
names; the one third-party-input boundary (Hugging Face -> `/models/`) is T-06-06 as registered and
is mitigated exactly as the register requires — pinned revision, sha256 on the f32 model and config,
verify-always rather than fetch-if-missing, a proven failing direction, f16 derived locally, and the
weights never reachable from a request path.

- **T-06-02** (DoS via context/rollout): `CHRONOS_MAX_POINTS` 20 000, context truncated to 2048 and
  REPORTED as `context_used` (asserted), `CHRONOS_MAX_HORIZON` 1024, horizon > 64 gated behind
  `allow_long_horizon` — all four asserted, all four weights-free.
- **T-06-04** (non-finite / all-null `y`): `is_finite` refusal and the `>= 2 observed values`
  refusal, both asserted; NaN is handled as missing by the instance norm, which `nan_gaps` probes.
- **T-06-13** (a parity claim that only passed because weights were absent): the two-sided proof
  above, in both directions, plus the `>= 8` matching-line count on the armed run.
- **T-06-SC** (cargo/python supply chain): `Cargo.lock` gained no new registry package —
  `half` was already a resolved workspace dependency.

## User Setup Required

None for the library. To run the armed ladder locally: `just fetch-chronos-tiny` (needs `uv` and
either network access to huggingface.co or the local HF cache snapshot), then
`CHRONOS_MODEL_DIR=$PWD/models/chronos-bolt-tiny/f32 cargo test -p aprender-forecast --lib -- bolt::parity chronos::parity`.
No credentials; the weights are Apache-2.0 and are never committed.

## Next Phase Readiness

**Ready for 06-06.** Everything 06-07's server crate embeds and calls is now proven here, which was
this plan's stated purpose.

What later plans inherit:

- **`chronos::{Model, load_model_from_bytes, load_model_from_dir, ChronosArgs, ChronosResponse,
  validate, forecast, load_csv}`** — 06-07 owns the transport (tool schema, `build.rs` OUT_DIR
  staging of `CHRONOS_EMBED_DIR`, `EMBEDDED_WEIGHTS`/`EMBEDDED_CONFIG`, `resolve_model`, routes) and
  re-implements no numeric path and no refusal.
- **`contracts/chronos-bolt-parity-v1.yaml`** is the frozen bar and the place any Chronos tolerance
  moves; `constants` is read by tests via `constant_u64`.
- **`just fetch-chronos-tiny` is verify-always**, so 06-08's `chronos-gate` should invoke it
  UNCONDITIONALLY rather than only when the f32 model is missing (REVIEW-06-03), and the two-sided
  skip commands in that gate are the two runs recorded verbatim above.
- **`Bolt::forward_stages` keeps spike-007's exact signature** for 06-07's `--bench`, and the
  `fast`/`attn_gemm`/`row1_gemv` toggles survive in the shipped struct for its A/B measurements.

**Three things to watch:**

1. **The provisional x86_64 bar.** 06-08's CI leg is the first x86_64 run; whoever lands it owes the
   contract a measurement (FALSIFY-CHRONOS-002), in either direction.
2. **`tiny_len_130_rollout` at 1.24x margin** is the thinnest probe and the first to move if the
   rollout or the short-context path changes.
3. **The 84 s armed wall** is an `aprender-compute` debug-profile artefact, not a Chronos cost. If
   06-08 wires an armed leg anywhere with a time budget, measure before assuming.

---
*Phase: 06-native-time-series-forecasting-stack*
*Completed: 2026-09-06*

## Self-Check: PASSED

All six `key-files.created` entries exist on disk (`src/bolt.rs`, `src/safetensors.rs`,
`src/chronos.rs`, `build.rs`, `tests/fixtures/chronos_bolt_tiny_config.json`,
`contracts/chronos-bolt-parity-v1.yaml`); all three task commits (`21732582c`, `18021082d`,
`92584e902`) are present in `git log --all`; and this file carries exactly one bare-token
`D-13 decision:` line plus its `(verbatim)` twin, as 06-09 Task 3 requires.
