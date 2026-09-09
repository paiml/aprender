# Handoff Report — Level 2 Root Aggregation for Segment 1

## 1. Observation
- Audited Segment 1 (`decisions_findings_scope_discovery_criteria`) of `docs/specifications/PP-066-release-spec.md` (lines 1–170, v1.5).
- Evaluated Level 1 candidate reviews `agg_1.md` and `agg_2.md` located in `/home/noah/src/aprender-worktrees/pp-066-spec/.agents/segment_decisions_findings_scope_discovery_criteria/`.
- Verified repository files at HEAD:
  - `scripts/pv_bin.sh`: lines 1–6 declare sourcing requirement; script terminates at line 662 with `export PV` and contains no CLI argument dispatcher (`exec "$PV" "$@"`). File mode is 100644.
  - `scripts/perf_gate.sh`: line 620 defines `arm_a_self_regression()`, replacing former `arm_a_scaling` under PP-31.
  - `crates/apr-cli/src/error.rs`: line 106 defines `CliError::FeatureDisabled(_) => 9`; lines 284–287 test exit code 9.
  - `crates/apr-cli/Cargo.toml`: defines two binaries (`apr`, `apr-corpus-ingest`) without `default-run`.
  - `crates/aprender-gpu/Cargo.toml`: lacks a `metal` feature (removed in PR #2849). `metal_shaders.rs` lines 8–11 document raw shader string constants only.
  - `pmat comply check`: executed live, verifying `CB-1700` passes (`✓`), while `CB-1701` and `CB-2100` fail (`✗`).
  - `PP-LLAMA-001-MASTER.md`: v3.1 committed at HEAD on `main`; line 364 formats row 22 with markdown bolding (`| **22** |`).
  - `evidence/parity/LEDGER.md`: 7 occurrences of `CONFORMANT` exist in prose comments and headers; zero rows in data cells are conformant.
  - Contracts count: `grep -l 'registry: true' contracts/*.yaml | wc -l` yields 501 matching files.

## 2. Logic Chain
1. **Verification Theater in C12 / S0-18**: Because `scripts/pv_bin.sh` is an environment-sourcing helper that exports `PV` and ignores arguments, running `scripts/pv_bin.sh lint contracts/ ...` executes in a subshell and exits 0 unconditionally without invoking `pv lint`. Thus, C12 passes vacuously, recreating the exact verification theater defect documented in F-26.
2. **Broken and Pseudo-Code Release Gates**: C1, C7, C9, and C13 contain unexecutable shell syntax (`≥ 2`, missing script name in C7, `wc -l = ticket count`, searching `empty`, Unicode ellipsis `…`). Additionally, `grep -c CONFORMANT` in C1 matches 7 lines in prose, yielding a vacuous pass unless table cells are structurally regex-matched.
3. **Step-0 Discovery Desynchronization & Multi-Host Execution**: S0 mints a ticket to falsify "eight premises" while §3 defines 23 premises. Commands S0-1 (`grep -n '^| 22'`), S0-2 (`--status all`), S0-4 (`cargo run -p apr-cli`), S0-12 (`--features metal`), and S0-14/19 (missing `-E`) fail upon execution.
4. **Omissions and Namespace Collisions**: Criterion C14 is missing from §4 despite partition references (`C0..C14`). Deliverable `T-2` is missing from the §2 Scope table and has a mismatched minting string in §5 (`T-5a`). Identifiers `D-1` and `T-1` collide across tracks.

## 3. Caveats
- Host-specific hardware discovery premises (`S0-3`, `S0-4`, `S0-12`, `S0-21`, `S0-22`) require bare-metal execution on physical hosts (`intel`, `mini`, `gx10`, `lambda`). They cannot all run locally inside an isolated container without remote SSH dispatch.
- Verification of sibling repository infrastructure (`../infra/machines/clean-room` for C8) assumes existence of the `../infra` checkout.

## 4. Conclusion
Segment 1 provides strong architectural direction (R-0 backend registry, contract discipline) but contains critical execution defects, vacuous gates, and syntax errors that would cause release automation and Step-0 discovery to fail or pass fraudulently. All 20 verified findings, impact analyses, and remediations have been synthesized into the definitive unit report.

## 5. Verification Method
- Review the generated unit report:
  `view_file` on `/home/noah/src/aprender-worktrees/pp-066-spec/.agents/segment_decisions_findings_scope_discovery_criteria/unit_report_decisions_findings_scope_discovery_criteria.md`
- Verify compliant top-level headers:
  `grep -E '^# ' /home/noah/src/aprender-worktrees/pp-066-spec/.agents/segment_decisions_findings_scope_discovery_criteria/unit_report_decisions_findings_scope_discovery_criteria.md`
- Independently verify tool commands against the repository:
  - `pmat comply check | grep -E 'CB-(1700|1701|2100)'`
  - `grep -nE '^\|\s*(\*\*)?22' docs/specifications/PP-LLAMA-001-MASTER.md`
  - `head -n 6 scripts/pv_bin.sh`
