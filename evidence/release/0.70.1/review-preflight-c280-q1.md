# Review receipt: the 0.70.1 preflight change (C280.4, C282/C283 "Q1 resmoke")

- Reviewed patch: `git diff cc4463f7a6 -- scripts/check_publish_preflight.sh contracts/model-capability-ladder-v1.yaml`, 271 lines, sha256 `ebd2139e4bedbdde96e78f40bedd423fd28a35491bf552097f4c84eaa33691a6`.
- Preflight script reviewed and committed: sha256 `423a6fad5ab4a9600f9e09790b1e25d7a912599d4763c42703b8284c33230979`.
- What it does: (a) R8 under a recorded scope prints the readiness verdict as evidence (C280.4); (b) with no `--scope`, a scope the contract records for this release engages by itself, judged at HEAD (Q1 resmoke, C283); (c) the R7 evidence call passes `--scope none`, so the old failures still print (C280.6).
- Author: claude-opus-5-5 (cop aprender-27). Lanes, none the author's model: claude-sonnet-5-5 (plan mode), agy gemini-3.1-pro-high, agy gpt-oss-120b-medium. Packet: the patch, the machine evidence below, the rulings.
- Round 1, 2026-10-03: verify 12:27:53Z -> 12:32:37Z, lanes 12:32:37Z -> 12:34:30Z. gpt-oss-120b-medium answered only "**VERDICT: PASS**" (18 bytes, no reasons); a verdict without reasons is not a review, so it was discarded and the same packet was asked again at 12:37Z with a demand for cited reasons. Its second answer is below.

## Verdicts

| lane | verdict |
|---|---|
| claude-sonnet-5-5 | VERDICT: PASS |
| gemini-3.1-pro-high | VERDICT: PASS |
| gpt-oss-120b-medium (second answer) | VERDICT: PASS |

## Machine evidence (verify.sh summary, verbatim)

```
START 2026-10-03T12:27:53Z worktree HEAD cc4463f7a609f04c0c4fd1ba4229a985ddaeeead preflight sha256 423a6fad5ab4a960
SELFTEST rc=0 --- 85/85 rows --- broke=0
GUARD check_release_models_t1 rc=0 PASS  41 row(s): the model matrix runs at T-1 on both hosts, every failure to prove the release STOPs before the tag, and R7 refuses the sam
GUARD check_release_autopilot_dogfood_close rc=0 PASS  17 row(s): autopilot never inherits a dogfood GO, judges R5 at T-1 with the T-4 function, and closes the epic before the milestone (#3
BASHRS Summary: 0 error(s), 109 warning(s), 147 info(s) Summary: 0 error(s), 109 warning(s), 190 info(s) 
PV-EXTRACT-CHECK rc=0 "shapes_n":66 "check":[] 
SCOPE IDENTICAL outside=NONE NONE-UNTRACKED 
PATCH 271 lines sha256 ebd2139e4bedbdde
KILLED mut_auto auto-never rc=1 broke=10: auto_scope_is_printed auto_scope_r8_readiness_is_evidence auto_scope_r8_warn_is_evidence auto_scope_r7_judges_the_scope_at_head auto_scope_red_refuses auto_scope_decline_refuses auto_scope_pass_names_both auto_scope_two_records_refuse auto_scope_unreadable_contract_refuses auto_scope_keeps_the_matrix_as_evidence 
KILLED mut_auto auto-unusable-ignored rc=1 broke=2: auto_scope_two_records_refuse auto_scope_unreadable_contract_refuses 
KILLED mut_auto auto-any-release rc=1 broke=2: auto_scope_other_release_r7_is_matrix auto_scope_two_records_refuse 
KILLED mut_auto auto-unreadable-is-none rc=1 broke=1: auto_scope_unreadable_contract_refuses 
KILLED mut_auto auto-silent rc=1 broke=1: auto_scope_is_printed 
KILLED mut_auto auto-no-contract-refuses rc=1 broke=6: all_rules_hold rc_on_release_not_main_passes prepublish_open_obligations_pass versioned_sibling_devdep_acyclic_passes pathed_sibling_devdep_passes r8_pass_is_named 
KILLED mut_auto auto-writes-bytecode rc=1 broke=6: auto_scope_r8_readiness_is_evidence auto_scope_r8_warn_is_evidence auto_scope_r7_judges_the_scope_at_head auto_scope_pass_names_both auto_scope_keeps_the_matrix_as_evidence scope_on_a_recorded_release_keeps_matrix 
KILLED mut_auto auto-cut-not-head rc=1 broke=7: auto_scope_is_printed auto_scope_r8_readiness_is_evidence auto_scope_r8_warn_is_evidence auto_scope_r7_judges_the_scope_at_head auto_scope_decline_refuses auto_scope_pass_names_both auto_scope_keeps_the_matrix_as_evidence 
KILLED mut_auto evidence-bare rc=1 broke=2: auto_scope_keeps_the_matrix_as_evidence scope_on_a_recorded_release_keeps_matrix 
KILLED mut_auto reader-unusable-is-none rc=1 broke=2: auto_scope_two_records_refuse auto_scope_unreadable_contract_refuses 
KILLED mut_auto reader-name-garbled rc=1 broke=8: auto_scope_is_printed auto_scope_r8_readiness_is_evidence auto_scope_r8_warn_is_evidence auto_scope_r7_judges_the_scope_at_head auto_scope_red_refuses auto_scope_decline_refuses auto_scope_pass_names_both auto_scope_keeps_the_matrix_as_evidence 
KILLED mut_auto gate-rrc-ignored rc=1 broke=2: auto_scope_two_records_refuse auto_scope_unreadable_contract_refuses 
KILLED mut_auto reader-why-dropped rc=1 broke=1: auto_scope_unreadable_contract_refuses 
KILLED mut_r8 scope-flag-alone rc=1 broke=5: r8_unrecorded_scope_name_refuses r8_scope_without_a_contract_refuses r8_scope_of_another_release_refuses r8_scope_of_another_release_no_enforce_refuses r8_two_records_for_one_release_refuse 
KILLED mut_r8 never-scoped rc=1 broke=11: r8_scoped_fail_is_evidence r8_scoped_prints_the_wrapper_rc r8_scoped_prints_the_wrapper_rows r8_scoped_no_enforce_pass_is_evidence r8_scoped_could_not_judge_is_evidence r8_scoped_caller_error_is_evidence r8_scoped_pass_is_printed_too r8_scoped_gate_names_readiness auto_scope_r8_readiness_is_evidence auto_scope_r8_warn_is_evidence auto_scope_pass_names_both 
KILLED mut_r8 any-name rc=1 broke=1: r8_unrecorded_scope_name_refuses 
KILLED mut_r8 no-evidence-print rc=1 broke=3: r8_scoped_prints_the_wrapper_rows r8_scoped_pass_is_printed_too auto_scope_r8_warn_is_evidence 
KILLED mut_r8 falls-through rc=1 broke=10: r8_scoped_fail_is_evidence r8_scoped_prints_the_wrapper_rc r8_scoped_prints_the_wrapper_rows r8_scoped_no_enforce_pass_is_evidence r8_scoped_could_not_judge_is_evidence r8_scoped_caller_error_is_evidence r8_scoped_gate_names_readiness auto_scope_r8_readiness_is_evidence auto_scope_r8_warn_is_evidence auto_scope_pass_names_both 
KILLED mut_r8 writes-bytecode rc=1 broke=13: r8_scoped_prints_the_wrapper_rc r8_scoped_prints_the_wrapper_rows r8_scoped_no_enforce_pass_is_evidence r8_scoped_could_not_judge_is_evidence r8_scoped_caller_error_is_evidence r8_scoped_pass_is_printed_too r8_scoped_gate_names_readiness auto_scope_r8_readiness_is_evidence auto_scope_r8_warn_is_evidence auto_scope_r7_judges_the_scope_at_head auto_scope_pass_names_both auto_scope_keeps_the_matrix_as_evidence scope_on_a_recorded_release_keeps_matrix 
KILLED mut_r8 absent-wrapper-ok-under-scope rc=1 broke=1: r8_scoped_wrapper_absent_refuses 
KILLED mut_r8 pass-line-silent rc=1 broke=2: r8_scoped_gate_names_readiness auto_scope_pass_names_both 
DONE 2026-10-03T12:32:37Z
```

## Lane output: sonnet

```
VERDICT: PASS

I found no concrete defect. I read the patch and the evidence you supplied, and checked `SCOPE` handling in the file directly. I did not re-run the self-test, `bashrs` or the mutants; those results are from your evidence.

1. **Scope of the diff (check 1).** The only files touched are `scripts/check_publish_preflight.sh` (its self-test is in the same file), one `emergency_scopes` entry in the contract, and `evidence/`. The `crates/`, `src/`, `Cargo.toml` and `Cargo.lock` comparison reads IDENTICAL, and `pv extract --check` returns `"check": []`.

2. **Nothing relaxed without a record for this version (checks 2, 6).**
   - `scope_recorded` returns 0 only when the judge's own reader reports no problem and the name matches `SCOPE` exactly. Every other case returns 1 and leaves R8 unchanged. That covers no contract, a contract with no entry, an entry for another release, two entries, a wrong name, and a failed load. Self-test rows `r8_*` and `auto_scope_other_release_*` cover each of these.
   - `SCOPE` is initialised at `check_publish_preflight.sh:949` and set only by `--scope` (line 952) or the committed contract (line 436), so the environment cannot engage it. The `SCOPE=... row` calls in the self-test are function-scoped test seams.

3. **Nothing beyond the record is relaxed (check 3).** R1–R6 and the rest of `gate()` are untouched. R7 under a scope still refuses on RED and on a decline. The wrapper-absent refusal comes before the new branch, so a missing wrapper still refuses even with a record. Only the wrapper's rc and verdict are demoted to evidence.

4. **Unusable record (check 4).**
   - `recorded_scope_name` maps any load or parse failure, a missing `ladder` key, a missing `yaml` module, two records, or a nameless record to rc 2. `gate()` then prints a FAIL and sets `fails=1`. The self-test covers two records and an unparseable YAML file.
   - An unexpected python crash exits 1, which reads as "no record". That sends the run to the full gate, which is stricter, so it fails closed.

5. **Cut (check 5).** `--cut-commit` is still accepted only with `--scope` (line 957), so an auto-engaged scope always judges at HEAD.

6. **Bash correctness (check 7).** `rec_out="$(...)"; rec_rc=$?` is correct because `rec_out` and `rec_rc` are declared on a separate `local` line. `REC_NAME` and `REC_WHY` are reset on every call. `R8_SCOPED` is reset at the top of `gate()`. `PYTHONDONTWRITEBYTECODE` is set in both python calls, and the mutants `auto-writes-bytecode` and `writes-bytecode` are killed.

7. **Self-test and mutants (check 8).** Rows exist both ways for (a), (b) and (c): `r8_scoped_*` against `r8_unrecorded_*`, and `auto_scope_*` against `auto_scope_other_release_*`, `no_contract`, `two_records` and `unreadable`. The `evidence-bare` mutant is killed. All listed mutants are killed, and each kill has `rc=1` plus at least one BROKE row.

8. **Lint (check 9).** `bashrs` reports 0 errors and 109 warnings, equal to `cc4463f`. Both `PERF002` disables sit on single-shot lines that are not loop bodies, so they are honest.

Non-blocking notes (not reasons to FAIL):
- **Comment drift.** The comment above `emergency_scopes:` in the contract still reads as 0.69.1-only. This is acceptable under C280.2's "nothing else changes".
- **Missing PyYAML.** A box without PyYAML now refuses the publish with the message "ladder contract could not be loaded". That fails closed, but the message is not obviously about a missing dependency.
- **Pre-existing `--scope none`.** Passing `--scope none` explicitly makes `SCOPE=none`, which engages the scoped R7 path calling the judge with `--scope none`. `scope_recorded` rejects the name, so R8 stays enforced. This was already there and is not weaker.
rc=0

```

## Lane output: gemini

```
VERDICT: PASS

1. **Diff Scope and Rules**: The patch strictly adheres to C280.5. The diff touches only `scripts/check_publish_preflight.sh` and `contracts/model-capability-ladder-v1.yaml`.
2. **Contract Edit**: The new entry for `0.70.1` precisely matches C280.2's specifications, including the exact verbatim quote and thinking modes ("off", "on") present in the `0.69.1` entry.
3. **R8 Logic (`C280.4`)**: The script perfectly scopes R8. `scope_recorded` validates exact matches directly against the contract. A missing wrapper rightfully halts execution prior to this check. If found, the wrapper runs, prints evidence, safely returns `0`, and signals `gate()` to note release-readiness was not the gate.
4. **Q1 "resmoke" Auto-engage**: Implemented correctly via `recorded_scope_name`. If `--scope` is absent, the gate asks the judge's reader for the exact scope. Unusable records securely exit non-zero (refusing release as `fails=1`), preventing any silent fallbacks.
5. **'+' Evidence Fix**: The `ev="$(cd "$root" && bash "$judge" --version "$version" --scope none 2>&1)"` execution successfully commands the judge to disregard its own auto-scope logic (from #4086), correctly preserving the full model matrix failures as evidence (C280.6).
6. **Lint & Security**: The two `PERF002` bashrs disables are honest as neither execution resides inside a loop block. `sys.argv[1]` routing and exception boundaries in the Python scripts are robust. Error codes correctly align across all execution flows. Variable states (e.g. `R8_SCOPED`) do not leak.

```

## Lane output: oss2

```
**VERDICT: PASS**

1. **contracts/model‑capability‑ladder‑v1.yaml:97‑103** – A single new `emergency_scopes` entry for `crux-smoke` / 0.70.1 is added exactly as required (date, quote, hosts, thinking). No other parts of the file are changed, satisfying C280 §2 and the “nothing else changes” rule.

2. **scripts/check_publish_preflight.sh:43‑45** – Comments updated to describe the behavior under a recorded emergency scope, matching the specification (C280 §4). No functional code is altered here, so documentation remains accurate.

3. **scripts/check_publish_preflight.sh:68‑71** – The model‑matrix call now uses `--scope none` to force the full ladder when a recorded scope is engaged, implementing the required “evidence only” behavior (C280 §4). The change is correctly scoped and does not affect other code paths.

4. **scripts/check_publish_preflight.sh:295‑313** – New `scope_recorded` and `recorded_scope_name` helper functions correctly invoke `crux_smoke_scope.recorded_scope` and handle the three possible outcomes (recorded, none, unusable). The logic follows the contract spec and the self‑test suite validates all three branches.

5. **Self‑test rows (see `selftest()` in the same script)** – Comprehensive rows added for:
   - Scoped vs. unscoped executions,
   - Evidence printing of R8 under a recorded scope,
   - Failure, warning, and pass cases,
   - Auto‑scope activation when no `--scope` flag is supplied.  
   All 85 self‑test rows pass, and every mutant listed in the evidence is killed, confirming full coverage and no regression.

*No concrete defects were found; all changes adhere to the operator rulings, preserve existing behavior when no recorded scope exists, and introduce the required evidence‑only handling for the 0.70.1 emergency scope.*

```

## Lane output: gpt-oss-120b-medium, first answer (discarded: no reasons)

```
**VERDICT: PASS**

```
