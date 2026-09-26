# DEBT-RATCHET-070-075 — tech-debt floors per release, 0.70 → 0.75

Status: DRAFT, 2026-09-26 (aprender-01). Assigned by the cop on the operator's request.

## Rule

Every row names four things: the command that derives it, the value measured today, the gate
that enforces it (`file:line`), and one floor per release. **A floor only ever tightens.** For
a percentage or score that means the floor rises. For a debt count, the ceiling falls. A release
whose measured value misses its row's floor does not cut. The cut takes the floor from this table
unless the measured value is already better. In that case the gate's baseline file is regenerated
to the measured value, because a ratchet that stays behind the tree allows regressions.

"Unmeasured" is written as unmeasured. A row with no measured value gets its first floor from the
first release that measures it. That release sets the floor and does not gate on it.

## Two REDs on 2026-09-26 and what fixes them (they block row 1 and row 2)

| red | root cause (evidence) | fix branch |
|---|---|---|
| coverage-nightly, run 36204739015 (yoga-build3, `c68deb494`), 76% | earlyoom (`--prefer rustc\|cargo`, swap 0) SIGTERMed the aprender-serve lib test binary `realizar-5c448889829f98ca` at VmRSS 15185 MiB, with 1317/28861 MiB available. A signalled binary writes no `.profraw`, and under `--ignore-run-fail` all of aprender-serve read as 0%. Lines hit fell 855274 → 728040. Nothing regressed. The Makefile nonetheless said "REGRESSION: coverage went DOWN". The seven tests still running at the kill allocate 6.7–34.4 GB KV caches (kv_001/002/002b, ultra/super/mega `long_context_memory_bound`, `large_vocab`). | `a01/cov-nightly-oom-kill`. `scripts/coverage_killed_binaries.sh` (with a case-table self-test) makes `make coverage` exit 1 with "DID NOT MEASURE" instead of publishing a partial TOTAL. The seven tests are listed in `scripts/coverage-skips.txt`; they still run in workspace-test. |
| toolchain-ceiling `clippy-current-stable`, red since 09-23 | Rust 1.98.1 added `clippy::chunks_exact_to_as_chunks`, which is denied by `#![deny(clippy::all)]` at `crates/aprender-serve/src/lib.rs:60`. It fires at 10 sites: `float16_dot.rs` ×3, `forward_qwen35.rs`, and the iq2_s/iq2_xxs/iq3_s/iq3_xxs/iq4_nl/iq4_xs dequant. | `a01/clippy-ceiling-198`. The fix rewrites the sites to `as_chunks::<N>()`, which has been stable since 1.88 (MSRV is 1.91). Stable-1.98.1 `clippy --all-targets -D warnings` gives rc=0. |
| toolchain-ceiling `clippy-feature-matrix` (cuda, full axes) | `borrow_deref_ref` under the pinned 1.93 toolchain with `--features cuda`. It is only reachable with that feature, so the pin gates never compiled it. | same branch (see receipt) |

A second RED earlier, run 36076984673 on 09-25 (87%), had **no** signal kill: 75 `test result: ok` lines against 101. Its cause is unmeasured, and it is not claimed to be the same defect.

## The table

Release columns are floors (≥) or ceilings (≤). "0.70 (now)" is the value measured on 2026-09-26.

| # | row | derive | now (2026-09-26) | gate | 0.70 | 0.71 | 0.72 | 0.73 | 0.74 | 0.75 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | line coverage % | coverage-nightly `TOTAL:` line (`make coverage`) | **90.21** (855274/948085, run 36117229983, 09-25). The 09-26 run did not measure (see above). | `Makefile:523` `COV_FLOOR`, `Makefile:510` `COV_THRESHOLD` | ≥88 | ≥89 | ≥90 | ≥92 | ≥93 | **≥95** |
| 2a | clippy on the pin | `make lint` (pin from `rust-toolchain.toml`), `-D warnings` | 0 | `Makefile:169`, CI lint | 0 | 0 | 0 | 0 | 0 | 0 |
| 2b | clippy on current stable | `bash scripts/check_clippy_current_stable.sh` | 10 on `main` → 0 with `a01/clippy-ceiling-198` | `.github/workflows/toolchain-ceiling.yml:88` | 0 | 0 | 0 | 0 | 0 | 0 |
| 2c | clippy feature matrix (pin) | `bash scripts/check_clippy_feature_matrix.sh` | red on cuda and full | `.github/workflows/toolchain-ceiling.yml:112` | 0 | 0 | 0 | 0 | 0 | 0 |
| 2d | `allow(clippy::…)` attributes, pedantic debt (count) | `grep -rhoE '#!?\[allow\(clippy::[a-z_]+' crates --include='*.rs' \| wc -l` | 4021 | **none yet**. Gate 0.71: add a `clippy_allow_count` row to `scripts/check_baseline_ratchets.sh` | ≤4021 | ≤3900 | ≤3650 | ≤3350 | ≤3050 | ≤2750 |
| 3a | surviving mutants on the PR diff | `cargo mutants --in-diff` (CI mutants section) | cap 0 | `ci/sections.yml:3258` `MUTANTS_MAX_MISSED` | 0 | 0 | 0 | 0 | 0 | 0 |
| 3b | global mutation score % | `cargo mutants --no-times --timeout 300 --in-place -- --all-features` (nightly, per crate) | **unmeasured** (the published 85.3% had no date or commit) | **none yet**. Gate 0.71: nightly job + baseline file | measure | = 0.70 value | +2 | +2 | +2 | +2 |
| 4a | `#[contract(` attributes on functions | `grep -rhoE '#\[contract\(' crates --include='*.rs' \| wc -l` | 190 (129 files) | **none yet**. Gate 0.71: `check_baseline_ratchets.sh` row | ≥190 | ≥220 | ≥260 | ≥300 | ≥350 | ≥400 |
| 4b | contracts anchored (SHACL) | `bash scripts/check_ont_ratchet.sh --write` → `contracts/lint-baseline.json` `.ont.contracts_anchored` | 8 / 1889 | `Makefile:1413` (`check_ont_ratchet.sh --check`) | ≥8 | ≥25 | ≥60 | ≥120 | ≥200 | ≥297 |
| 4c | contracts shaped (SHACL) | same, `.ont.contracts_shaped` | 6 | `Makefile:1413` | ≥6 | ≥20 | ≥50 | ≥100 | ≥170 | ≥250 |
| 4d | unanchored-but-bindable (count) | same, `.ont.unanchored_but_bindable` | 297 | `Makefile:1413` | ≤297 | ≤280 | ≤240 | ≤180 | ≤100 | ≤0 |
| 4e | contracts without `valid_under` (count) | same, `.ont.contracts_without_valid_under` | 386 | `Makefile:1413` | ≤386 | ≤350 | ≤300 | ≤230 | ≤150 | ≤60 |
| 4f | `pv lint contracts/` score | `pv lint contracts/` (via `scripts/pv_bin.sh`) | **unmeasured**. The 09-26 run hit the tool's 330 s cap while building pv | **none yet**. Gate 0.71 | measure | = 0.70 value | rises | rises | rises | rises |
| 5a | CB-200: functions below TDG `min_grade` B (count) | `pmat comply` CB-200 (pmat 3.41.1 pin) | 599 | `.pmat-gates.toml:103`, mirrored in `scripts/cb200_baseline.txt`, checked by `scripts/check_baseline_ratchets.sh:144` | ≤599 | ≤570 | ≤530 | ≤480 | ≤430 | ≤380 |
| 5b | functions over the hook complexity (cyc >30 or cog >25), count | `bash scripts/check_complexity_ratchet.sh --update` → rows in `scripts/complexity_baseline.txt` | 667 | `ci/sections.yml:1800` | ≤667 | ≤640 | ≤600 | ≤550 | ≤500 | ≤450 |
| 5c | per-function cyclomatic target | `.pmat-gates.toml:28` `max_complexity = 10` (aspirational, new code) | 10 | pre-commit hook | 10 | 10 | 10 | 10 | 10 | 10 |

### How the floors were chosen

- **Coverage** runs from the enforced 88 today to the operator's 95 by 0.75. The step size is
  slowest where the measured value already sits (90.2), so 0.71 and 0.72 can be met by fixing
  the nightly alone. The steps are ~2 points a release after that.
- **Debt counts** (2d, 5a, 5b) fall about 5–10% a release. That is the rate a normal release's
  refactors reach without a dedicated debt sprint. If a release beats the rate, the next ceiling
  is lowered to what was measured.
- **SHACL rows** (4b–4e) reach "every bindable contract anchored" (297) by 0.75. That is the
  README-ONT-001 / ONT-001 direction. The 1464 `formal_prose` contracts stay out of scope.
- **Unmeasured rows** (3b, 4f) set their floor at 0.70 and gate from 0.71. They are measured
  and stated, never estimated.

## Gates missing today (work items, one per row)

1. 2d, 4a: add two count rows to `scripts/check_baseline_ratchets.sh` with baseline files, and include them in its case table.
2. 3b: a nightly per-crate `cargo mutants` job and a `mutation_score_baseline.txt`. This modifies CI, so it needs a check-in first.
3. 4f: run `pv lint` in the nightly and record the score in `contracts/lint-baseline.json`.
4. Coverage nightly: the fix above, plus giving yoga swap or moving coverage to a host with more memory. The nightly must not be killed quietly again (row 1 depends on it measuring at all).

## Milestone per release

Each release milestone (0.70 … 0.75) carries one DEBT-RATCHET issue. Its body is that release's
column, with each floor as a checkbox. It is closed by the release receipt that quotes each
measured value next to its floor. The cop mints the issues.
