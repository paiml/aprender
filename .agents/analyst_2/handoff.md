# Handoff Report — Analyst 2 (Candidate 2)

## 1. Observation
- Inspected Segment 2 of `PP-066-release-spec.md` (lines 167–665) covering §5 through Appendix A.
- Found that `scripts/pv_bin.sh` does not forward CLI arguments when executed directly (`bash scripts/pv_bin.sh validate non_existent_file.yaml` exits 0), rendering §5 R-0 A and §4 C12 commands vacuous passes.
- Identified four 0-day slack blocker pairs in §5 (`R-4` vs `master 0b`, `P-0.6` vs `P-0.3`, `P-1.2` vs `P-1.1`, and `T-0` vs `T-1`) that contradict C10 and G-4's 6-day minimum slack invariant.
- Found a physical contention and queue order inversion on `gx10`: `master 15` (blocked by October 2 rows 6 and 12) is assigned queue slot 1 ahead of `T-0` and `S-3` which expire on September 26.
- Observed premise count desynchronization: §9 targets `9/9 Step-0 premises answered`, but §3 enumerates 23 premises (`S0-1..S0-23`).
- Verified code facts: `CliError::FeatureDisabled` maps to 9 (`crates/apr-cli/src/error.rs:106`), `accel.rs:28` uses `cfg!(any(feature = "cuda", feature = "wgpu"))`, `accel.rs:114` tests 3 surfaces, 13 MSL kernels in `metal_shaders.rs`, `Q4K_GEMV_SHADER` at `basic_ops.rs:555`, and double `[lib] name = "aprender"`.
- Wrote full evaluation report to `/home/noah/src/aprender-worktrees/pp-066-spec/.agents/segment_implementation_tickets_future_lanes_governance/handoff_2.md`.

## 2. Logic Chain
1. `scripts/pv_bin.sh` was written as a sourceable script (`. scripts/pv_bin.sh`) and ends with `export PV` without executing `exec "$PV" "$@"`. Calling it as an executable with arguments (`scripts/pv_bin.sh validate ...`) drops the arguments and exits 0, silently bypassing verification.
2. In §4 C10 and §5 G-4, the specification mandates `expires(blocker) + 6d <= expires(blockee)`. Defining tickets with identical expiry dates to their blockers creates immediate CI failures under `check_dag_invariants.sh`.
3. In §2 and §9, `gx10` is a single-queue physical resource ordered by expiry. If slot 1 (`master 15`) cannot execute until October, then queue slots 2 and 3 (`T-0` and `S-3`, due September 26) are either starved or inverted.
4. Step-0 was expanded across revisions to 23 premises. Leaving §9 target at `9/9` creates an incomplete gate.
5. Aligning ticket handles between spec cards and `pmat work add` strings ensures unambiguous audit receipt paths.

## 3. Caveats
- Did not modify `PP-066-release-spec.md` or repository source code directly, as the assignment was document evaluation and review.
- The `0_66-review.md` findings referenced in §11 were assessed based on citations in the spec and current HEAD state.

## 4. Conclusion
Segment 2 is architecturally sound and sets a high benchmark for Toyota Way governance and contract rigor. However, the four critical execution defects (pv_bin.sh vacuity, DAG slack violations, gx10 queue inversion, and premise count drift) must be remediated in the specification before ticket #1 opens. The complete Candidate 2 evaluation report is stored in `handoff_2.md`.

## 5. Verification Method
- Inspect report: `view_file /home/noah/src/aprender-worktrees/pp-066-spec/.agents/segment_implementation_tickets_future_lanes_governance/handoff_2.md`
- Test `pv_bin.sh` vacuity: `bash scripts/pv_bin.sh validate non_existent_file.yaml; echo exit=$?` (yields 0)
- Verify `FeatureDisabled` mapping: `grep -n "FeatureDisabled.*=>" crates/apr-cli/src/error.rs` (line 106)
- Verify `accel.rs` checks: `grep -n "build_has_accelerator" crates/apr-cli/src/accel.rs`
- Verify MSL kernel count: `grep -c "kernel void" crates/aprender-gpu/src/backend/metal_shaders.rs` (13)
