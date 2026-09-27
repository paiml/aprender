# impl receipt — #4522 R2 resource ceilings (RSS + VRAM)

Ask (cop): memory and VRAM ceilings in perf-matrix and contracts, `CiAssertions.max_memory_mb` live, dogfood C1
RSS WARN→FAIL, a case table with a planted over-budget run that goes RED. Builds on R1 (aprender-52) `resources{}`.

| Piece | What |
|---|---|
| `scripts/perf-matrix.yaml` arms.R (+ vendored copy, byte-identical) | `rss_model_multiple: 3`, `rss_overhead_bytes: 536870912` (F-CHAOS-001), KV added for `cpu` class, `vram_device_fraction: 0.95`, `vram_device_mib: {lambda: 24564}` (evidence/task-132/rtx4090-370m-residency.json). `require_resources: false` until R1 emits the block. threshold_class policy, author aprender-5d |
| `scripts/perf_gate.sh` `arm_r_resources` | Arm R, phase both. Reads `resources.peak_rss_bytes`, `resources.vram_peak_bytes`, `resources.null_reasons.<field>`. Null needs a non-blank reason; 0/negative/non-number FAIL; no model bytes ⇒ ceiling uncomputable ⇒ FAIL; `rss >= ceil` FAIL (strict <, as F-CHAOS-001); `vram > ceil` FAIL; host with no measured capacity ⇒ n/a |
| `scripts/check_rss_budget.sh` | Dogfood C1. One pinned `apr run` under GNU `time -v`; budget read from arms.R. Exit 0/1/2; 2 (cannot measure) is a FAIL, never a skip |
| `apr profile --ci --assert-memory <MB>` | `CiAssertions.max_memory_mb` was `#[allow(dead_code)]`. Now: flag → dispatch → assertion `peak_rss` against VmHWM from /proc/self/status. Unmeasured ⇒ FAIL. JSON gains `peak_rss_mb` |
| `contracts/apr-resource-ceilings-v1.yaml` | 5 obligations, 5 falsifiers. `pv validate` (pv 0.70.0 817d63361): valid, 0 warnings |
| apr-dogfood SKILL C1/C1b | C1 calls the script; C1b runs `--assert-memory`. The old C1 could not fail (WARN, `apr inspect`, `-size -1G` matched only empty files, symlinked ~/models not descended) |
| `docs/audits/surface_audit.csv` | New row for the flag, gated in C1b (145/839, 17.3%); 77 citations of extended_commands.rs (+4 after line 219) and dispatch_analysis.rs (+1/+2 after 1332/1356) re-anchored; `apr profile` row 165→168 (was stale) |

Measured:
- `perf_gate.sh --selftest`: 121 passed, 0 broken. The 12 new armr_* rows include the planted over-budget run: 15e9 B against a 4.7e9 B model goes RED. Mutant `rss>=ceil` → `rss>=ceil*10`: 2 rows BROKE (planted_over_budget, at_ceiling), rc 1.
- `check_rss_budget.sh --self-test`: 8/8 ok. Mutant budget ×10: 2 rows BROKE.
- `cargo test -p apr-cli --lib -- gh4522 test_parse_profile_ci_mode test_ci_profile`: 29 passed. Mutant `mb <= max_mb` → `true`: 2 tests FAILED, rc 101, file restored (cmp).
- `cargo clippy -p apr-cli --lib -- -D warnings`: rc 0.
- `DOGFOOD_BASE_REF=HEAD scripts/check_dogfood_coverage.sh`: PASS.
- Live, lambda, apr built from this tree (`scripts/apr_bin.sh` → /mnt/nvme-raid0/targets/aprender-5d-guard/release/apr), model Qwen2.5-0.5B-f16-notemplate.gguf (994154304 B):
  - C1: `peak RSS 2581299200 B < budget 3519333824 B`, rc 0.
  - C1b: `PASS peak_rss: 2448.6 MB (expected <= 3356.0 MB)`, rc 0.
  - Planted `--assert-memory 100`: `FAIL peak_rss: 2448.2 MB`, rc 5.

[U] **Done-when is not met yet.** It needs a nightly and an rc receipt with `resources{}` non-null. That comes from R1; Arm R reports n/a until then. Flip `require_resources: true` in the PR where R1 emits the block.
[U] **VRAM ceiling is lambda-only.** No other host has a measured capacity in the tree. gx10 is unified memory, so RSS is its ceiling.
[U] **Coverage gate against origin/main** fails G2.2/G2.3 on this base: main has 864 rows and this base has 838. That is branch age, not this diff. The batch's merge with main must union the ledger.
