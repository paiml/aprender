# Handoff Report — Aggregator 2 (Segment 2)

## 1. Observation
- Evaluated two candidate reviews for Segment 2 (`implementation_tickets_future_lanes_governance`) of `docs/specifications/PP-066-release-spec.md` (v1.5):
  - Candidate 2: `.agents/segment_implementation_tickets_future_lanes_governance/handoff_2.md`
  - Candidate 4: `.agents/segment_implementation_tickets_future_lanes_governance/handoff_4.md`
- Both candidates agreed on:
  - `scripts/pv_bin.sh` dropping arguments and exiting 0 when executed directly in R-0 (line 208) and C12 (line 159). Live test `bash scripts/pv_bin.sh validate non_existent.yaml` exited with `0`.
  - Four 0-day slack blocker pairs in §5: `R-4` vs `master 0b` (both 2026-09-19), `P-0.6` vs `P-0.3` (both 2026-09-19), `P-1.2` vs `P-1.1` (both 2026-10-10), and `T-0` vs `T-1` (both 2026-09-26).
  - Step-0 premise count desynchronization between §9 target (`9/9`) and §3 expansion (`S0-1..S0-23`, 23 premises).
  - Track T card handle vs `pmat work add` string desynchronization (`T-2` minting `T-5a`, `T-3` minting `T-7`).
  - Missing `REG-10` cross-reference in the prior-art register (§12) for llama.cpp and Ollama.
  - Markdown backtick syntax error in R-5 (line 275).
  - Stale line citations in §10: `aprender_ml` at line 666 (not 509); `FeatureDisabled` mapped at line 106 (not 86-90); 104 Cargo manifests under `git ls-files '*/Cargo.toml'` (not 106); `finetune.rs` sequence length clamp hardcoded at line 717 (not 317).
  - Scope nuance for wasm32 in `aprender-serve`: only safetensors memory-mapping is gated out, not the entire crate.
- Candidate 2 uniquely contributed:
  - Deadlock and expiry inversion on the single `gx10` queue: `master 15` is blocked by October 2 rows (6 & 12) yet assigned queue slot 1 ahead of September 26 rows `T-0` and `S-3`, violating G-4's queue invariant `queue_pos(a) < queue_pos(b) => expires(a) <= expires(b)`.
  - Crate name allow-list count ambiguity (51 vs 52 rows).
  - Duplicate section heading in §11.
- Candidate 4 uniquely contributed:
  - Non-executable acceptance command in G-4: `pmat comply check --rule obligation-dag` fails CLI argument parsing (`error: unexpected argument '--rule' found`).
  - Missing source files: `docs/specifications/pp-066-dag.yaml` and `scripts/render_dag.py` do not exist.
  - Semantic tooling mismatch in R-0: `pv validate` only validates YAML contract syntax, not CLI JSON output.
  - Non-existent `claims-cite` command in D-1 (the in-tree script is `scripts/check_perf_claims_cite_receipts.sh`).
  - Internal fixture count contradiction: §4 C11 states 12 fixtures, while R-0, §9, and Appendix A specify 14 fixtures (`FX-1..FX-14`).
  - Missing prefill throughput kill threshold for S-2 in §7.
  - Missing discrimination clauses across cards R-7, T-0, G-3, D-1, T-1, B-A1.
  - `MetalBackend` struct location in `crates/aprender-gpu/src/backend/mod.rs:67`, not `metal_shaders.rs`.
  - Authoritative filename in §10: `PP-LLAMA-001-MASTER.md`, not `PP-LLAMA-001-MASTER-v3.md`.
  - Master expiry date citation update in §10: rows 19 and 21 converted to `derived` in v3.0.24.
- Candidate 4 false positive filtered out:
  - Claimed typo `repnt` in row `P-1.2` (line 308). Inspection of the file confirms it already reads `repoint`.

## 2. Logic Chain
1. Step-0 and ticket acceptance commands that exit 0 without running verification create silent passes ("contract theater"). `scripts/pv_bin.sh` when executed directly drops all arguments and exits 0, so R-0 and C12 cannot verify contracts unless the script is patched or sourced.
2. Invariant checkers cannot succeed if the input data violates their declared rules. The DAG checker mandates `expires(blocker) + 6d <= expires(blockee)`, so the four 0-day slack blocker pairs in §5 will fail immediately upon execution of `check_dag_invariants.sh`.
3. Physical hardware cannot run tasks out of dependency order without violating queue policies. Because `master 15` depends on October 2 rows, placing it in queue slot 1 ahead of September 26 tasks starves `gx10` and causes `T-0` and `S-3` to blow their deadlines.
4. CLI commands specified in card acceptance criteria must match the actual argument schema of the binary. Since `pmat comply check` has no `--rule` flag, G-4's acceptance command cannot execute.
5. All findings were verified against git HEAD; false positives were eliminated; consensus and high-confidence individual findings were unified into `agg_2.md`.

## 3. Caveats
- The exact scheduling adjustment for `gx10` depends on whether `master 15`'s dependency on rows 6 and 12 can be relaxed or if training tickets `T-0` and `S-3` can be rescheduled to October.
- The choice between patching `scripts/pv_bin.sh` to forward arguments or editing the spec to source the script (`. scripts/pv_bin.sh && "$PV" ...`) is an implementation decision for the ticket owner.

## 4. Conclusion
The aggregated review for Segment 2 has been synthesized and written to:
`/home/noah/src/aprender-worktrees/pp-066-spec/.agents/segment_implementation_tickets_future_lanes_governance/agg_2.md`
It incorporates 15 major potential mistakes and improvements across execution, scheduling, scope, and contract discipline, alongside 10 minor corrections and typos, with all claims verified against the codebase.

## 5. Verification Method
To independently verify the findings in `agg_2.md`:
1. Check `scripts/pv_bin.sh` behavior:
   ```bash
   bash scripts/pv_bin.sh validate non_existent.yaml; echo exit=$?
   ```
   (Outputs `exit=0`).
2. Check `pmat comply check` CLI flags:
   ```bash
   pmat comply check --rule obligation-dag docs/specifications/pp-066-dag.yaml --min-slack-days 6
   ```
   (Outputs `error: unexpected argument '--rule' found`).
3. Verify existence of DAG files:
   ```bash
   ls docs/specifications/pp-066-dag.yaml scripts/render_dag.py
   ```
4. Verify non-existence of `claims-cite`:
   ```bash
   which claims-cite || git grep -i 'claims-cite'
   ```
5. Check C11 fixture count vs R-0 fixture count:
   ```bash
   sed -n '157,159p;208p' docs/specifications/PP-066-release-spec.md
   ```
6. Check `repnt` absence (false-positive verification):
   ```bash
   grep -n 'repnt' docs/specifications/PP-066-release-spec.md
   ```
