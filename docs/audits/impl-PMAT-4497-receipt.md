# impl receipt — #4497 APR-OBS-001 OBS-10 loop admission

Spec: APR-OBS-001 §5.1 (admission), §7 row OBS-10, §9 `loop_admission`. Done-when: "a consumer with a planted write
path to a RED rule is refused; a consumer before night 14 is refused".

| Piece | What |
|---|---|
| `scripts/check_loop_admission.sh` | Judges every consumer in the registry against §5.1: prerequisites P (OBS-01/03/04/05 contracts present), nights N (≥14 consecutive UTC nights of admissible `apr-perf-ledger-v1` rows on host×backend, ending today or yesterday; admissible = every §2.1 identity field non-empty, `gpu_proof` on non-cpu), writes W (declared globs never reach a protected path). Emits `{"loop_admission":[{consumer,admitted,reason}]}`. Exit 1 iff an `acting=yes` consumer is refused or the registry is empty. bash+jq, no Python (R-5). bashrs: 0 errors |
| `docs/obs/loop-admission-consumers.tsv` | `prm-rex-10`, `arb-self-001`, both `acting=no`, host unassigned |
| `contracts/apr-loop-admission-v1.yaml` | 4 obligations, 6 falsifiers. `pv validate`: valid, 0 warnings |

Measured:
- `--self-test`: 17/17 rows ok, rc 0. Rows include both done-when cases (RED-rule write path → REFUSED; 13 nights → REFUSED).
- Mutants, each run on a scratch copy (worktree clean afterwards), all killed:
  M1 `MIN_NIGHTS` 14→13 → the 13-nights row goes RED;
  M2 RED-rule path removed from PROTECTED → the done-when-1 row goes RED;
  M3 identity-field check dropped → the missing-`model_sha256` row goes RED;
  M4 acting-refusal no longer RED → 12 rows go RED.
- Quorum r1 (sonnet): a `writes=unassigned` placeholder read as a declared glob; now refused like an empty field (self-test row added).
- Self-test found a real defect before commit: `read` with `IFS=$'\t'` merges an empty column, so an undeclared `writes` was ADMITTED. Fixed by splitting on `\x1f`.
- Live run on this tree: both consumers refused, reason "prerequisite rows not GREEN" (no OBS contract is on main yet), rc 0, 0 admitted. The gate fails closed.
- Wiring: `guard_tree.sh --list --no-cargo` prints `scripts/check_loop_admission.sh`. No CI edit.

[U] GREEN for a prerequisite row is proxied by its contract existing in `contracts/` (§3 ships each contract with its
row). When the OBS rows land, this can tighten to running their gates. [U] "a CI job the consumer cannot modify":
the gate refuses a declared write path to the gate, guard_tree.sh, ci.yml and ci/sections.yml. It cannot see a
consumer's real credentials; declared writes are what the registry holds.
