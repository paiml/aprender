# PMAT-4152 implementation receipt: clippy over every workspace member

Issue: paiml/aprender#4152 (0.70.0). The scope was ruled by the cop (aprender-cf, 2026-09-24): "(a) + ratchet".

## The defect, measured
Every clippy gate (Makefile:161/230/246, `cargo clippy -- -D warnings`) lints the ROOT FACADE package only. On main
aa7c6ef03, `cargo clippy --workspace --all-targets --no-deps -- -D warnings` (GPU trio excluded, as CI does) gives
**87 failing targets and 3807 diagnostics across 15 crates**, none of them ever seen by a gate. The top lints:
- 1969 × `Result::unwrap` and 319 × `Option::unwrap` (the `.clippy.toml` ban);
- 1280 × `excessive_precision`, all aprender-serve;
- 82 × deprecated `criterion::black_box`;
- 73 × `expect_fun_call`;
- 16 × `undocumented_unsafe_blocks`.

## (a) aprender-serve is clippy `--all-targets -D warnings` clean (9248b9c49)
- **`excessive_precision`** (scoped narrower in quorum round 1, below). Inheriting `[workspace.lints]` is not possible. Cargo forbids inherit + override, and the
  workspace brings `unsafe_code = "deny"` (serve has hundreds of `unsafe` blocks) and `pedantic`. A Cargo
  `[lints]` allow was tried and **measured to have no effect**, because `lib.rs`'s `#![deny(clippy::all)]`
  re-enables the lint at source level. So the allow sits in `lib.rs` next to the crate's other allows, mirroring the
  workspace's rationale. 1280 → 0.
- **`undocumented_unsafe_blocks`, which this crate sets to `deny`.** Every flagged block was REVIEWED; none got a
  blanket allow. Two were **unsound**, and are fixed:
  - `tests/falsification_tests.rs`: `simd_axpy` checked AVX2 only, but called a `#[target_feature(enable = "avx2",
    enable = "fma")]` fn. On an AVX2-without-FMA CPU that is UB. It checks both now.
  - `examples/bench_simd_dot.rs`: its avx2+fma kernels were called with **no runtime detection at all**. `main`
    now refuses on a CPU without AVX2+FMA.
- **Other sites.**
  - `examples/bench_manual_threads.rs`: `from_raw_parts_mut` ×3 carving `&mut` row blocks out of one Vec →
    `chunks_mut`. No `unsafe` left.
  - `tests/falsification_crux_c_34.rs`: the old SAFETY comment cited an `ENV_LOCK` that does not exist. `ENV_SAFETY`
    now states the real invariant: all 12 tests are `#[serial(env_force_loading)]`, on current-thread runtimes.
  - `src/memory.rs`: `PinnedRegion::new`'s contract (valid for `len`; the Vec outlives the region).
  - 4 small test-code lints.
- **Verification.** `cargo clippy -p aprender-serve --all-targets --no-deps -- -D warnings` → rc 0. Affected tests:
  lib 48, `beat_fail_closed_garbage` 3, `falsification_tests` 12, `falsification_crux_c_34` 4, all passing.

## (b) the member-wide ratchet (37860c93b, c984b7e82, 188fe43d8)
`scripts/check_clippy_member_ratchet.sh` + `scripts/clippy_member_baseline.txt` (49 keys, 3262 findings, clippy 0.1.93).
- **Census.** `cargo clippy --workspace --all-targets --no-deps` (GPU trio excluded) `--keep-going
  --message-format=json`, keyed `<crate>|<target kind>|<lint>`. It runs WITHOUT `-D warnings`, so a failing target
  cannot hide its dependents. That is why the census (3262) differs from the `-D warnings` count (3807, of which serve's
  1300 are now fixed).
- **Rule.** A NEW key or a ROSE count fails. A STALE key (fixed but not shrunk) fails until the baseline is shrunk, so
  progress is recorded and never left as slack. The baseline file may only shrink against origin/main
  (`lib_baseline_ratchet.sh keyed`). The guard fetches the comparand itself when the job has not, with the exact
  command the library's refusal prescribes.
- **Refusals, never verdicts.**
  - Clippy that did not finish: an unfinished census hides every finding in the targets it never reached. The guard
    names each error-level diagnostic with its file:line.
  - A partial workspace: every wanted member (76, the GPU trio excluded) must be checked, compared BY NAME.
  - A different clippy version, recorded in the baseline: the lint set is not monotonic.
- **First green on a real run.** 76 members checked; 3262 == baseline; 2m28s warm on lambda (2m51s on the re-run at 6fb583ee4).
- **Planted mutants (both KILLED, restored after each).**
  - An undocumented `unsafe` block in aprender-serve's lib → RED: "clippy::undocumented_unsafe_blocks at
    crates/aprender-serve/src/memory.rs:406", the only cause (serve's `deny` fails the target, and the unfinished
    census is refused).
  - One more banned `.unwrap()` in aprender-graph's lib (a warn-level lint) → RED: "ROSE
    aprender-graph|lib|clippy::disallowed_methods: 171 -> 173", plus two NEW pedantic keys.
- **Case table.** `--self-test`, 11 rows, no cargo: unchanged green; rise; new key; stale; key gone; unfinished;
  no build-finished; note-level ignored; the version-only package-id form (a real parse bug found on the first
  census: aprender-core came out as "0.69.0"); an excluded dependency standing in for an unchecked member; a full member set.
- **Wiring, with no workflow file edited.**
  - `ci/explicit-test-commands.d/460-…self-test.cmd` and `470-…ratchet.cmd`, run by workspace-test-shard.
  - `make lint-members`; `tier3` runs the pair inline (so `bashrs make lint` gains no MAKE012 warning).
  - `check_explicit_test_commands` PASS, `check_tree_reader_tests` PASS, `check_bashrs_gate` PASS; the guard is
    bashrs-clean.

## Quorum round 1 (lane 1, claude-sonnet-5: FAIL, three findings, all verified and fixed in 6fb583ee4)
1. **The vacuity check counted, it did not compare.** `checked_members` counted every workspace package with a
   compiler artifact, and an excluded GPU-trio member built as a DEPENDENCY counted too. The first green run printed
   "79 members" against 76 wanted. The test was `got -lt want`, so an over-count could hide a missing member.
   It is now a set difference of names (`missing_members`), and each missing member is named. Two new self-test
   rows. The planted count-test mutant turns the "excluded dependency" row RED. Re-run on real output: "76
   workspace members", green.
2. **`excessive_precision` was allowed crate-wide.** The only sites are the five `IQ*_EXPECTED` test consts in
   `quantize/iq*.rs`, which hold ggml's `dequantize_row_*` output as printed with `%.9e`. The allow now sits on those
   five consts and nowhere else. `lib.rs` is back to its prior allows. Verified: serve `--all-targets -D warnings`
   rc 0. Mutant (one const's allow removed) → rc 101, 257 `excessive_precision` errors at `iq2_s.rs:114`.
3. **`dot_avx_vnni`'s SAFETY comments named the wrong invariant.** They cited only AVX2+FMA. The raw VPDPBUSD
   `asm!` needs AVX-VNNI, which the enclosing `if has_avx_vnni()` confirms. The comments now say so.

## Follow-up filed
**#4161** (0.71.0) is the unwrap sweep. It has 2137 UNIQUE sites (1810 `Result`, 327 `Option`), per crate:
qa-runner 1038, test-cli 444, graph 235, apr-cli 143, qa-report 78, tsp 74, core 44, qa-cli 42, rag 24, test-showcase
13, train-shell 1, facade 1. The baseline header cites it.

## Cost, stated
The 470 fragment adds one full workspace clippy to one workspace-test shard: 2m28s warm on lambda (8 jobs). A cold
clean-room runner will take longer, and that has not been measured here.
