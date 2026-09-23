# Receipt — PMAT-3968: per-shape GPU conformance, and a whitelist that states what loads

Branch `feat/3968-gpu-whitelist-shape-conformance`. It is based on the 0.69.1 freeze `c619dddd4`, because the
whitelisted IQ types and their kernels exist only on the release line until the #4046 merge-back. Author aprender-dd
(claude-opus-5-5). The 11 commits are `f78e9317c..30edfa0ea`.

About 7,100 of the ~8,050 added lines are evidence JSON (two censuses, two receipts), all machine-written. The
reviewable code is:
- `scripts/lib/gguf_census.py`;
- `crates/aprender-serve/src/cuda/executor/shape_conformance_tests.rs`;
- `scripts/check_gpu_shape_conformance.sh` and `scripts/gpu_shape_conformance.sh`;
- the whitelist change in `crates/aprender-serve/src/gguf/dtype.rs`, `crates/aprender-gpu/src/driver/context.rs`,
  `crates/aprender-serve/src/capability.rs` and `contracts/apr-model-capability-v1.yaml`.

## Claim 1: the rows are derived, never hand-listed
- `gguf_census.py` reads every held GGUF's header and writes one (qtype, k, n) per 2-D tensor. There is one census
  per host (lambda, gx10).
- The scanner's stdout IS the committed file, with no transform:
  `python3 scripts/lib/gguf_census.py scan '~/models' '~/.apr/models' '~/.cache/apr/models' --host <host>`.
  It records its roots, depth, missing roots and every depth-skipped GGUF. `--depth N` counts subdirectory levels.
  Before round 2's fix it read only files directly in a root and skipped nested models silently: lambda 23 → 27 files,
  gx10 14 → 18, and the union 163 → **188** rows. The 25 new rows (F32, Q8_0, BF16) had never been tested.
- The harness builds its rows from the committed censuses, filtered by the device-independent whitelist. A newly
  held shape is therefore tested as soon as it is censused.

## Claim 2: each row is measured through the production dispatch
- The path is `BoundWeight::bind` + `CudaExecutor::bound_gemv`, against the CPU decoder.
- The bound is condition-aware: |err| / Σ|w||x| ≤ 1e-5.
- The workspace is initialised as production does (`init_workspace`), with `q8_activation_valid = false` before each
  call.
- Q6_K's DP4A path quantizes activations to Q8_1, so the CPU oracle is fed the same Q8_1-quantized x (`q8_1_mirror`).
  Two harness defects in that mirror were found and fixed, each by a measured run:
  - (i) a scaled-integer input made exact .5 rounding ties. Run 3 left 16 rows at 1–2e-5; the input became a
    golden-ratio sequence.
  - (ii) the kernel stores the Q8_1 scale as f16 (`q8.rs` `cvt_f16_f32`) but the mirror used f32. Run 4 showed 27
    rows at 1.1–3.9e-5, falling with k. The mirror now dequantizes with the f16 scale.
  - Run 5: every Q6_K row was at most 2.0e-8. Neither defect was in the kernel.
- A panic in one row (a kernel precondition, a launch error) is recorded as that row's failure and does not lose the
  other rows.
- Negative control per type: block 0 of row 0 is corrupted in the GPU copy only, and the control must go RED. A
  panicking control counts as blind.

## Claim 3: IQ3_S/IQ2_S are refused at cc ≥ 12 (cop ruling (b), 2026-09-23)
- On GB10 sm_121, IQ3_S (21) and IQ2_S (22) are whitelisted but never load (ModuleLoad 218; the sm_12x PTX patcher,
  #4096).
- `gpu_unsupported_quant_qtype(q)` is now `_on(q, device_cc_major())`:
  - `GPU_QTYPES_UNLOADABLE_AT_CC = [21, 22]` applies at `GPU_QTYPE_EXCLUSION_MIN_CC_MAJOR = 12`;
  - `device_cc_major()` asks `max_compute_capability_major()` once per process. That function is new, needs no
    context, and takes the max over visible devices, so it fails closed;
  - every production gate (apr run/serve, the construction gate, the hybrid gate) calls this one function.
- The capability contract rows carry `gpu_excluded_from_cc_major: 12`. FALSIFY-CAP's row test requires the value to
  equal the code's.
- `apr capability` renders the same fact: `no … refused on THIS device` at cc ≥ 12, and `yes … except compute
  capability >= 12` elsewhere. `--json` carries `this_device_cc_major` (from `realizar::gguf::device_cc_major`, the
  function the gate calls). The packaged contract mirror `crates/apr-cli/contracts/…` is synced (`capability_mirror`
  3/3).
- A capability query that ERRORS on a present device fails closed (`i32::MAX`: every exclusion applies). `None` means
  only that no driver or device exists.
- The whitelist-exactness tests now assert the device-independent list, `_on(q, None)`. Without that they would fail
  on a sm_12x box.

## Claim 4: the guard, and its stale-exclusion must-RED
`check_gpu_shape_conformance.sh` joins the whitelist, the exclusions (parsed from the same `dtype.rs`), the censuses
and the receipts.
- **RED** when any of these holds:
  - a held whitelisted shape is unproven on a host;
  - a receipt is missing, stale, failed or blind;
  - an excluded type is not exercised on a host where it is excluded;
  - the host capability is unknown while exclusions exist;
  - an exclusion is **stale**: every held shape of the type passes at cc ≥ 12 with a RED control, so #4096 has
    landed and the exclusion must go.
- **REFUSES** when the whitelist or an exclusion does not parse.
- A receipt whose `excluded_types` differs from what the tree's exclusions imply at its host's capability is a STALE
  receipt (RED). A change to the exclusion policy without a re-run is caught, as a census change is.
- `--self-test` has 17 rows, two of them for the nvidia-smi `gpu`-line capability fallback the lambda receipt uses. The RED rows require their reason text, not only rc 1. Mutant (stale branch disabled):
  exactly `RED-stale-exclusion` BROKE.

## Measured
Both hosts were re-run at `cd0457c82` against the regenerated 188-row censuses:
- **lambda** (RTX 4090 8.9): `cc_major 8`, `excluded_types []`; 188/188; worst 1.3e-7; 17/17 negative controls RED.
- **gx10** (GB10 12.1): `cc_major 12`, `excluded_types [21, 22]`.
  - The 3 excluded rows ran and failed (the stated RED). 0 passed, so the exclusion is not stale.
  - The other 185 passed, worst 1.3e-7.
  - 15/17 controls are RED; the 2 that are not are the excluded types, whose kernels do not load.
- `check_gpu_shape_conformance.sh` over both hosts: **PASS**. This is its first green on its real target.
- `cargo test -p aprender-serve --lib` over pmat785, prose_whitelist, every_quant_row_agrees and gemv_entry_name:
  37 passed with cuda, 12 passed on default features. `apr-cli` capability tests 5/5; `capability_mirror` 3/3; clippy
  `-D warnings` clean on `aprender-serve --features cuda` and `apr-cli`.

## Quorum round 1 (agy quota exhausted on every family, so Claude Code Sonnet 5 lanes, degraded: same-family)
Lane 1: FAIL. Its findings, all fixed in `b7f378ebe`/`7383d6f7b`:
- `apr capability` ignored the exclusion;
- the `gpu`-line fallback had no self-test row;
- the capability-query error failed open;
- the lambda receipt predated the exclusion code.

Also fixed: the packaged mirror, which the lane did not flag.

## Quorum round 2 (Sonnet 5 lanes, degraded: same-family)
- Lane 1: PASS, with three non-blocking findings.
  - Its "receipt not bound to the code" point is now the guard's exclusion-policy STALE check.
  - Its untested fail-closed branch in `device_cc_major` stays a code-read claim: a driver error cannot be planted
    without a mock seam.
- Lane 2: FAIL, on census provenance.
  - The committed censuses were not what the committed scanner writes.
  - Depth-skipped models were silent.
  - Both are fixed above, and the fix found 25 untested rows.
  - Its `rcp.approx` point is noted, not emulated. The Q8_1 mirror uses exact `1/x` where the kernel uses
    `rcp.approx` (≤1 ulp). It matters only on near-ties, and Q6_K's measured margin is ~500× (2.0e-8 vs 1e-5). If a
    future shape erodes that margin, this is the first suspect.

A fresh round of three lanes judges the new head.

## Not done
- The guard is not yet wired into a workflow. Wiring it is a `.github/workflows` edit, which needs the standing
  approval, and on CI it must be judged against committed receipts only (no GPU there).
- When #4096 lands: re-run `gpu_shape_conformance.sh gx10`. The guard then goes RED on the stale exclusion. Remove
  21/22 from `GPU_QTYPES_UNLOADABLE_AT_CC` and the two contract fields.
- The PR targets `main` after the #4046 merge-back. Until then the base is the 0.69.1 line.
