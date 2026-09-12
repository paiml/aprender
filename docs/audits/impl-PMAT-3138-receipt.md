---
status: in-flight
merged: "[U] — opens against main after train #3127 lands; this line is amended with the squash sha"
ticket: PMAT-3138
row: 67-fleet (follow-up to #3104; 06x-release-schedule.md §2 E/F)
issue: 3138
epic: 3078
model: "orchestrator claude-fable-5-1 (direct; no worker dispatched — the measurement ran on gx10 under setsid, the fixes are four files + ci.yml)"
tokens_used: "orchestrator [U] (not instrumented)"
wall_clock_s: "08:20Z–09:10Z wall for measurement + fixes ≈ 3000 s; the gx10 sweeps themselves: 499 + 638 + 84 + 833 + 25 + 36 s"
orch_model: claude-fable-5-1
orch_class: orchestration
orch_decision: "direct — one operator-priority ticket, the measurement is a shell harness on gx10 and the fixes are four files + ci.yml; no worker or lane would have been cheaper than the round-trips it saved"
fable_binding: true
quota_age_h: 0
quota_mark: A
k_measured_at_set: 4
---
# impl-PMAT-3138 — workspace-test on ARM64: gx10 runs the quick tier 3–4× faster than intel; four aarch64-only reds fixed; the tree-reader step builds 20 packages, not 686 binaries (#3138)

Receipt for the comparison in the title: `evidence/ci/arm64-workspace-test-2026-09-12/gx10-summary.txt` (per-step rc and seconds on gx10) and `evidence/ci/arm64-workspace-test-2026-09-12/gx10-tiers.txt` (the Starting/Summary lines of every tier), against intel's run 34680214617 attempt 1 (step timestamps in the Verification table below). Measured, not a target.

## Identity
ticket PMAT-3138 · kind code+ci · operator instruction 2026-09-12 08:20Z: "STOP the entire build system, and unclog gx10, then continue". gx10 read 0 % busy while PRs queued on intel because every long job pinned `X64`.

## What lands
- `.github/workflows/ci.yml` `workspace-test`: `runs-on: [self-hosted, Linux, clean-room]` (was `X64`). The tree-reader step derives `-p` from its 102 targets (20 packages) instead of `--workspace --lib --tests` (686 test binaries linked to run 41). `pr-review-receipt` keeps `X64` (#3132).
- `crates/aprender-serve/examples/bench_simd_dot.rs`: the AVX2/FMA/AVX-VNNI kernels move into a `#[cfg(target_arch = "x86_64")] mod x86`; other targets get a `main` that says so. `cargo check --workspace --all-targets --locked` was RED on aarch64 (`the feature named avx2 is not valid for this target`, ×4).
- `crates/aprender-compute/src/blis/parallel.rs`: `gemm_blis_parallel_shared_b` — the AVX-512 guard was `cfg(x86_64)`-only and the microkernel call sits in a `cfg(x86_64)` `if` with no other arm, so on aarch64 every full 8×32 tile was skipped (FALSIFY-SHARED-B-001: max diff 39.2 at 256³; 100×96 passed only under the 8 M-flop cut). Non-x86 takes the plain BLIS path; the scalar arm covers a full tile wherever the microkernel is absent. Decomposed into `shared_b_path_available`, `shared_b_thread_count`, `SharedBBlock::{run_slice, run_panels, tile, scalar_tile}` — the pre-commit complexity gate (cyclomatic 30 / cognitive 25) blocks any edit to the 175-line original. Not on any inference path: the only caller is `examples/blis_benchmark.rs`.
- `crates/aprender-compute/Cargo.toml` + `tests/falsification_tests/section_a.rs`: A-013 (NEON ≥ 2× scalar) behind a new `bench-gates` feature; the placeholder covers every other configuration. It is a wall-clock ratio, x86-vacuous, and 0.87× on gx10 because LLVM autovectorises the "scalar" baseline (`iter().zip().map().collect()`).
- `crates/aprender-compute/tests/fma_correctness_f017.rs` F022: relative-error bound 1e-5 → 1e-4, derived (`n·eps/(2k)`, k = 4 NEON lanes → 7.5e-5; measured 1.9e-5 on gx10; a scalar sequential sum ≈ 3e-4 still fails).
- `docs/roadmaps/roadmap.yaml`: PMAT-3138 (minted from #3138).

routes:
  ph0  class=orchestration  route=self  w=11.11  basis=first-run[U]   # STOP: cancel 5 runs, rerun one job, inventory gx10
  ph1  class=measure  route=self  w=11.11  basis=first-run[U]   # arm-sweep.sh / -2 / -3 / -5 on gx10 (setsid, container = CI image)
  ph2  class=impl  route=self  w=11.11  basis=first-run[U]   # four aarch64 fixes + ci.yml scoping + runs-on
  ph3  class=verify  route=self  w=11.11  basis=first-run[U]   # x86 (AVX-512 box) + gx10 re-verification, guards, lint

verification:
  cmd="cargo fmt --all -- --check"  claimed_exit=0  rerun_exit=0  log_path=evidence/ci/arm64-workspace-test-2026-09-12/fmt.log  sha256=ac8582433c5eba13abd955caf91e8d716ce92cd77971a68eb4cf3cae698d171a
  cmd="bash scripts/check_runner_labels.sh && bash scripts/check_no_timing_in_required.sh"  claimed_exit=0  rerun_exit=0 (both rc 0)  log_path=evidence/ci/arm64-workspace-test-2026-09-12/guards.log  sha256=42fdf9e8118584215862df453732ef614b4f04a2bd5dca46bb9842c36e76f0fc
  cmd="cargo nextest run --profile ci --no-fail-fast -p aprender-compute --lib --tests --features parallel -E 'test(/blis::/) | test(f022_fma) | test(a013)'  [x86_64, avx512f]"  claimed_exit=0  rerun_exit=0 (306 passed)  log_path=evidence/ci/arm64-workspace-test-2026-09-12/x86-compute-tests.log  sha256=26bd469fb2b01c7c44336309db6caa79bd30fe0a4ca44a62e7773e89117d7b87
  cmd="cargo clippy -p aprender-compute --lib --features parallel -- -D warnings  [x86_64]"  claimed_exit=0  rerun_exit=0  log_path=evidence/ci/arm64-workspace-test-2026-09-12/x86-clippy.log  sha256=d4c756ad12014c9e82f8acbca2c2e85974215a709739f74da1689b86a5470de7
  cmd="cargo check -p aprender-serve --example bench_simd_dot  [x86_64]"  claimed_exit=0  rerun_exit=0  log_path=evidence/ci/arm64-workspace-test-2026-09-12/x86-bench-simd-dot-check.log  sha256=a97f10e12a61a6a2a1e5752a0d8cc7b62087bb367986f0645b95c7268199c9d5
  cmd="gx10: cargo nextest run --profile ci --no-fail-fast -p aprender-compute --lib --tests --features parallel -E 'test(shared_b) | test(f022_fma) | test(a013)'  [aarch64, after the fixes]"  claimed_exit=0  rerun_exit=0 (4 passed; the same four were 3 FAIL + 1 vacuous before)  log_path=evidence/ci/arm64-workspace-test-2026-09-12/gx10-fix-verify.txt  sha256=1338e624cb462932fa676c4803130f45dbd0252c7be950359f10acedf5823990
  cmd="gx10: cargo check --workspace --all-targets --locked  [aarch64, after the bench_simd_dot gate; was rc 101]"  claimed_exit=0  rerun_exit=0 (check2_rc=0 in 84 s)  log_path=evidence/ci/arm64-workspace-test-2026-09-12/gx10-check2-tail.txt  sha256=2935c8ccae0da02aa20eb83cbc5caef19dbba856fe8148c443025a22cabc7d54
  cmd="gx10: quick step 1 / step 2 / FULL lib / GPU crates / compute lib — the Starting/Summary lines of every tier"  claimed_exit=0  rerun_exit=n/a (measurement; 59,359/59,362 pre-fix, 10,325/10,325, 82,085/82,085, rc 0, rc 0)  log_path=evidence/ci/arm64-workspace-test-2026-09-12/gx10-tiers.txt  sha256=727c3c726c5565ec03ba4e24ffae3fc2034e13ebacbaebbf5057b8baff7249fd
  cmd="gx10: arm-sweep summary (tier decision, per-step rc and seconds)"  claimed_exit=0  rerun_exit=n/a (measurement)  log_path=evidence/ci/arm64-workspace-test-2026-09-12/gx10-summary.txt  sha256=b4285607a7f912c546065cbb777c8edc0f8e6b9489707a74287a3b07151f5efa

## Verification
Measurement harness: `gx10:~/eph-work/arm-sweep{,-2,-3,-5}.sh`, results `gx10:~/eph-work/arm-sweep/20260912T082818Z/summary.txt` — same `localhost:5000/sovereign-ci:stable` image and env block as ci.yml, train tree a6148ab72, `--no-fail-fast`, `CARGO_BUILD_JOBS=12` (CI uses 8).
| check | intel (run 34680214617 attempt 1) | gx10 |
|---|---|---|
| quick step 1, 9 selected crates | 34.5 min, 59,590/59,590 (tests 1603 s) | 638 s, 59,359/59,362 (tests 532 s) — the 3 reds below |
| quick step 2, 102 tree-reader targets | 27.4 min, 10,329/10,329 (tests 1016 s) | 402 s, 10,325/10,325 (tests 246 s) |
| `cargo check --workspace --all-targets --locked` | cancelled at 1.5 min | rc 101 (bench_simd_dot) → rc 0 in 84 s after the gate |
| FULL: workspace `--lib` −3 crates | ~60 min (nightly) | 833 s, 82,085/82,085 |
| FULL: `-p aprender-gpu -p aprender-cuda-edge --lib` | — | 25 s, rc 0 |
| FULL: `cargo test -p aprender-compute --lib` | — | 36 s, rc 0 |
| the four fixes, `-p aprender-compute --lib --tests --features parallel -E 'test(shared_b) \| test(f022_fma) \| test(a013)'` | 4/4 on this AVX-512 box (48 cores) | 4/4 (`fix-verify.txt`, rc 0) |
| `cargo nextest run -p aprender-compute --lib --features parallel -E 'test(/blis::/)'` after the decomposition | 304/304 | — |
| `cargo clippy -p aprender-compute --lib --features parallel -- -D warnings` | clean | — |
| `cargo run -p aprender-serve --example bench_simd_dot` | (x86: benches) | rc 0, prints the x86-only notice |
| `scripts/check_runner_labels.sh`, `scripts/check_no_timing_in_required.sh`, `cargo fmt --all -- --check` | rc 0 | — |
| tree-reader `-p` derivation dry-run on the train's touched set | 102 targets → 20 packages | — |

## Mutation (RED, then GREEN)
- bench_simd_dot: `check.txt` RED (4 errors) → `check2.txt` rc 0 after the gate. Reverting the module gate restores the 4 errors on aarch64.
- shared-B GEMM: `step1.txt` RED (max diff 39.2, 3 tries) → `fix-verify.txt` 4/4. On x86 the same 304 blis tests are green before and after (the x86 path is unchanged by construction: same microkernel, same blocking).
- A-013: `step1.txt` RED (0.87× < 2×) → with `bench-gates` off the ratio is not compiled; with it on the ratio is measured and RED on gx10 — which is the finding, not a defect to hide.
- F022: `step1.txt` RED (1.9e-5 > 1e-5) → GREEN at 1e-4. A scalar sequential sum's ≈3e-4 stays RED under the new bound, so the bound still excludes an outcome.
- tree-reader `-p` scoping: `[U]` until the PR's first run — its step-2 log must show `packages built: 20` and far fewer `Compiling` lines than 421.

## Jidoka
- The first draft of this PR "fixed" step 1 by adding an env block it already had (a `grep -v '-e '` in my own view had hidden it). Re-read the raw block before diagnosing; the actual waste was step 2's `--workspace`.
- The first verification run on gx10 (`blis-adhoc.txt`, 161/161) proved nothing: without `--features parallel` the shared-B tests are not compiled. Feature unification in the 9-crate CI selection is what compiles them there.
- `tar … | ssh 'tar xf - && … &'` extracted nothing: the trailing `&` backgrounded the whole chain and tar read `/dev/null`. Copy with scp, launch in a second command.
- The Bash tool is zsh: `read -ra` and `${PIPESTATUS}` fail silently; the derivation dry-run runs under `bash -c`.

## Gaps
- The arm64 `sovereign-ci:stable` image runs as uid 0: 17 root-owned per-PR target dirs (100 GB) on gx10 that the host reclaim step cannot delete — chowned by hand today; the image needs `USER 1000` (separate infra change, not in this PR).
- `pr-review-receipt` stays `X64` until #3132 (reject-76-drop on arm64).
- The queue-run `estimatedTimeToMerge` and the 80/80/50 packing targets are unreachable while every PR sits behind one serialized train; the 0.68 DIRTY wave after #3127 is what fills three boxes.

## Estimates
basis: first-run[U] — no prior measured row for kind=ci+code on repo=aprender at this shape (measurement + fixes + refactor on two boxes).
