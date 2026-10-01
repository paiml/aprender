---
schema_version: 1
open_count: 20
waived_count: 2
fixed_count: 0
total_count: 22
last_updated: 2026-09-21T16:45:55.552Z
---

# Broken Windows Ledger

> Cross-phase defect register. With `workflow.windows_enforce` enabled, `/gsd-ship` blocks while `open_count > 0`.
> Waive with `gsd-tools windows waive <id> "<reason>"` (reason required).
> Mark fixed with `gsd-tools windows fixed <id>`.

| id | phase | kind | file | line | description | status | reason | recorded_at | resolved_at |
|----|-------|------|------|------|-------------|--------|--------|-------------|-------------|
| 1 | 05 | deviation | crates/aprender-core/src/calibration_tests.rs |  | 05-04 T2: the plan's named RED mutation (bin by predicted-class probability) is a no-op since pred IS the argmax; substituted true-label-confidence and unweighted-bins mutations | waived | Not a defect: the plan's named RED mutation is provably a no-op (predicted class IS the argmax), and 05-04 shipped two mutations that DO discriminate, with recorded values. Documented in 05-04-SUMMARY.md Deviations. Nothing is left open in the code. | 2026-08-17T04:43:03.419Z | 2026-08-17T04:44:08.106Z |
| 2 | 05 | deviation | crates/aprender-core/src/stats/tests_claims_stats.rs |  | 05-04 T3: the plan's RNG-guard criterion lists 'sample' as an RNG symbol; it is the statistical noun and the pre-existing f32 API's parameter name, so the guard bans real RNG symbols instead | waived | Not a defect: banning the token 'sample' would ban the pre-existing f32 ttest_1samp(sample: &[f32]) API and catch no RNG. The shipped guard bans rand::/thread_rng/SeedableRng/StdRng/gen_range/bootstrap/resample/shuffle/random( and passes. Documented in 05-04-SUMMARY.md Deviations. | 2026-08-17T04:43:13.479Z | 2026-08-17T04:44:14.174Z |
| 3 | 06 | deviation | crates/aprender-forecast/src/prophet.rs |  | objective_at_python_map binds 3 of 7 fixtures: the four spike-003 oracles publish no log_posterior_at_map_unnormalized, so no rung-2 test exists for them (06-03 Deviation 1; stated in contracts/prophet-parity-v1.yaml) | open |  | 2026-09-06T05:38:09.726Z |  |
| 4 | 06 | unrun-verify | contracts/chronos-bolt-parity-v1.yaml |  | quantiles_abs_f32_nonaarch64 = 5.0e-6 is PROVISIONAL and UNMEASURED; the first x86_64 run must record its max\|delta\| and tighten the bar (FALSIFY-CHRONOS-002) | open |  | 2026-09-06T13:46:12.140Z |  |
| 5 | 06 | unrun-verify | .planning/phases/06-native-time-series-forecasting-stack/deferred-items.md |  | D-18 clause 2: no CI leg exercises the embedded-weights build; quantiles_abs_f32_nonaarch64 stays PROVISIONAL 5.0e-6 pending one x86_64 just chronos-gate run (same obligation as REVIEW-06-02 and ledger entry #4) | open |  | 2026-09-06T23:48:33.107Z |  |
| 6 | 06 | deviation | crates/aprender-mcp-forecast/src/lib.rs |  | Plan 06-10 stated 22 existing e2e::refuses_ cases; measured baseline was 21, so post-change is 23 not the plan's >=24 bar. Intent (2 cases added) met; number in plan text was stale. | open |  | 2026-09-07T02:44:08.056Z |  |
| 7 | 06 | unmet-truth | crates/aprender-forecast/src/types.rs |  | SC1's 2 s bar is not guaranteed for holiday-carrying requests: no payload statistic bounds the wall (a 25 000-cell request walls at 4.2 s reproducibly). MAX_HOLIDAY_DESIGN_COST caps WORK; residual wall-clock exposure is FIT_BUDGET_SECS (15 s). Human decision required — see 06-11-SUMMARY coverage D7. | open |  | 2026-09-07T03:31:33.474Z |  |
| 8 | 06 | deviation | justfile |  | 06-12 T3 asked to change forecast-holiday-bench's defaults to the at-the-bound geometry; 06-11 deviation 2 had already done so ((800+200)x50 = 50 000 cells = MAX_HOLIDAY_DESIGN_COST). No change made; plan text was stale. | open |  | 2026-09-07T03:57:44.081Z |  |
| 9 | 06 | unmet-truth | contracts/forecast-tool-boundary-v1.yaml |  | Cost axis C-07 (holidays[].name byte amplification) carries unbounded_pending_06_15; held open by the DECLARED-RED types::tests::no_cost_axis_is_pending. Owed by 06-15 T2. | open |  | 2026-09-07T15:58:32.207Z |  |
| 10 | 06 | unmet-truth | contracts/forecast-tool-boundary-v1.yaml |  | Cost axis C-08 (unbudgeted NeuralProphet training work) carries unbounded_pending_06_15; held open by the DECLARED-RED types::tests::no_cost_axis_is_pending. Owed by 06-15 T3. | open |  | 2026-09-07T15:58:32.289Z |  |
| 11 | 06 | unrun-verify | justfile | 656 | chronos-coldstart's 150 ms SC4 bar uses [ $med -ge 150 ]; an erroring token takes the not-taken branch and prints COLD START OK. Closed today only by the upstream sed digits-only parse, not by the bar (D-ITEM-06-16). | open |  | 2026-09-07T17:54:46.394Z |  |
| 12 | 06 | deviation | Makefile |  | make contract-audit-phase6's success line contains the literal BIND-, so grep -c 'BIND-' == 0 is unsatisfiable; four consecutive plans each fixed it locally. Needs a shared scripts/assert_no_audit_findings.sh with a case table. | open |  | 2026-09-07T18:40:10.361Z |  |
| 13 | 06 | deviation | CHANGELOG.md |  | The two breaking 0.63.0 surface changes (prophet::feature_row's fourth parameter, safetensors::load removal) are documented in the crate README but not yet in the root CHANGELOG; release-time follow-up. | open |  | 2026-09-07T18:40:10.438Z |  |
| 14 | 05 | deviation | scripts/setfit_bench_gate_doctor.py |  | Probe helper extracted to a sibling .py not in 05-15 files_modified; bashrs cannot parse a quoted heredoc body (12 phantom shell errors). Load-bearing for the door probe. | open |  | 2026-09-11T22:23:42.703Z |  |
| 15 | 05 | lint-warning | crates/aprender-compute/src |  | cargo clippy -p aprender-train --features setfit -- -D warnings exits 101 on pre-existing aprender-compute / aprender-present-terminal findings; the 05-15 plan verification line cannot pass on this tree. | open |  | 2026-09-11T22:23:42.783Z |  |
| 16 | 05 | unmet-truth | crates/aprender-train/src/train/setfit/bench_row.rs | 37 | Module doc claims canonical bytes are key-sorted (no preserve_order). Measured false: file-order reproduces both committed digests, sorted reproduces neither. | open |  | 2026-09-11T22:23:42.862Z |  |
| 17 | 05 | deviation | crates/aprender-train/src/train/setfit/bench_row.rs | 37 | The bench row seal is build-graph dependent: serde_json/preserve_order (via pmcp) makes to_canonical_bytes emit declaration order in apr-cli and key-sorted in aprender-train, so the same committed row verifies in one binary and is refused in the other (D-ITEM-05-17-A) | open |  | 2026-09-12T01:12:32.984Z |  |
| 18 | 05 | deviation | crates/aprender-test-lib/src/brick/pipeline.rs | 1965 | uuid_v4() is a timestamp, not a UUID; test_uuid_v4_generates_unique_ids fails 5/5 on this host and its verdict depends on clock resolution (D-ITEM-05-17-B) | open |  | 2026-09-12T01:12:33.076Z |  |
| 19 | 06.1 | deviation | contracts/forecast-tool-boundary-v1.yaml |  | regressor_prior_scale_min is a representability bound, not the usability bound its rationale claimed; fit collapses below ~1e-8 | open |  | 2026-09-21T14:23:10.699Z |  |
| 20 | 06.1 | unrun-verify | crates/aprender-image/src/lib.rs |  | cargo fmt --all --check and full clippy -D warnings are red on the committed tree (pre-existing); phase used rustfmt per-file and clippy --no-deps instead | open |  | 2026-09-21T14:23:10.777Z |  |
| 21 | 06.1 | deviation | crates/aprender-forecast/src/invariance.rs |  | 06.1-04 verifies t1c and t3b are unsatisfiable by construction: both forbid any #[ignore] / any ignored test in the invariance module, but wave 1 deliberately ships capture_baseline as #[ignore]. Measured red on the untouched wave-1 tree. Intent-preserving substitutes pass. | open |  | 2026-09-21T16:45:55.473Z |  |
| 22 | 06.1 | deviation | .planning/phases/06.1-forecast-exogenous-inputs-prophet-regressors-neuralprophet-e/06.1-04-PLAN.md |  | 06.1-04 verify t2c python probe is blind to #[ignore] (it anchors test bodies at fn; the attribute precedes fn) and prefix-matches a renamed fn. Measured green on a part A carrying #[ignore]. Fixed in the in-tree Rust twin only; the verify body was kept verbatim. | open |  | 2026-09-21T16:45:55.552Z |  |

````json
[
  {
    "id": 1,
    "kind": "deviation",
    "phase": "05",
    "file": "crates/aprender-core/src/calibration_tests.rs",
    "line": null,
    "description": "05-04 T2: the plan's named RED mutation (bin by predicted-class probability) is a no-op since pred IS the argmax; substituted true-label-confidence and unweighted-bins mutations",
    "status": "waived",
    "reason": "Not a defect: the plan's named RED mutation is provably a no-op (predicted class IS the argmax), and 05-04 shipped two mutations that DO discriminate, with recorded values. Documented in 05-04-SUMMARY.md Deviations. Nothing is left open in the code.",
    "recorded_at": "2026-08-17T04:43:03.419Z",
    "resolved_at": "2026-08-17T04:44:08.106Z"
  },
  {
    "id": 2,
    "kind": "deviation",
    "phase": "05",
    "file": "crates/aprender-core/src/stats/tests_claims_stats.rs",
    "line": null,
    "description": "05-04 T3: the plan's RNG-guard criterion lists 'sample' as an RNG symbol; it is the statistical noun and the pre-existing f32 API's parameter name, so the guard bans real RNG symbols instead",
    "status": "waived",
    "reason": "Not a defect: banning the token 'sample' would ban the pre-existing f32 ttest_1samp(sample: &[f32]) API and catch no RNG. The shipped guard bans rand::/thread_rng/SeedableRng/StdRng/gen_range/bootstrap/resample/shuffle/random( and passes. Documented in 05-04-SUMMARY.md Deviations.",
    "recorded_at": "2026-08-17T04:43:13.479Z",
    "resolved_at": "2026-08-17T04:44:14.174Z"
  },
  {
    "id": 3,
    "kind": "deviation",
    "phase": "06",
    "file": "crates/aprender-forecast/src/prophet.rs",
    "line": null,
    "description": "objective_at_python_map binds 3 of 7 fixtures: the four spike-003 oracles publish no log_posterior_at_map_unnormalized, so no rung-2 test exists for them (06-03 Deviation 1; stated in contracts/prophet-parity-v1.yaml)",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-06T05:38:09.726Z",
    "resolved_at": null
  },
  {
    "id": 4,
    "kind": "unrun-verify",
    "phase": "06",
    "file": "contracts/chronos-bolt-parity-v1.yaml",
    "line": null,
    "description": "quantiles_abs_f32_nonaarch64 = 5.0e-6 is PROVISIONAL and UNMEASURED; the first x86_64 run must record its max|delta| and tighten the bar (FALSIFY-CHRONOS-002)",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-06T13:46:12.140Z",
    "resolved_at": null
  },
  {
    "id": 5,
    "kind": "unrun-verify",
    "phase": "06",
    "file": ".planning/phases/06-native-time-series-forecasting-stack/deferred-items.md",
    "line": null,
    "description": "D-18 clause 2: no CI leg exercises the embedded-weights build; quantiles_abs_f32_nonaarch64 stays PROVISIONAL 5.0e-6 pending one x86_64 just chronos-gate run (same obligation as REVIEW-06-02 and ledger entry #4)",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-06T23:48:33.107Z",
    "resolved_at": null
  },
  {
    "id": 6,
    "kind": "deviation",
    "phase": "06",
    "file": "crates/aprender-mcp-forecast/src/lib.rs",
    "line": null,
    "description": "Plan 06-10 stated 22 existing e2e::refuses_ cases; measured baseline was 21, so post-change is 23 not the plan's >=24 bar. Intent (2 cases added) met; number in plan text was stale.",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-07T02:44:08.056Z",
    "resolved_at": null
  },
  {
    "id": 7,
    "kind": "unmet-truth",
    "phase": "06",
    "file": "crates/aprender-forecast/src/types.rs",
    "line": null,
    "description": "SC1's 2 s bar is not guaranteed for holiday-carrying requests: no payload statistic bounds the wall (a 25 000-cell request walls at 4.2 s reproducibly). MAX_HOLIDAY_DESIGN_COST caps WORK; residual wall-clock exposure is FIT_BUDGET_SECS (15 s). Human decision required \u2014 see 06-11-SUMMARY coverage D7.",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-07T03:31:33.474Z",
    "resolved_at": null
  },
  {
    "id": 8,
    "kind": "deviation",
    "phase": "06",
    "file": "justfile",
    "line": null,
    "description": "06-12 T3 asked to change forecast-holiday-bench's defaults to the at-the-bound geometry; 06-11 deviation 2 had already done so ((800+200)x50 = 50 000 cells = MAX_HOLIDAY_DESIGN_COST). No change made; plan text was stale.",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-07T03:57:44.081Z",
    "resolved_at": null
  },
  {
    "id": 9,
    "kind": "unmet-truth",
    "phase": "06",
    "file": "contracts/forecast-tool-boundary-v1.yaml",
    "line": null,
    "description": "Cost axis C-07 (holidays[].name byte amplification) carries unbounded_pending_06_15; held open by the DECLARED-RED types::tests::no_cost_axis_is_pending. Owed by 06-15 T2.",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-07T15:58:32.207Z",
    "resolved_at": null
  },
  {
    "id": 10,
    "kind": "unmet-truth",
    "phase": "06",
    "file": "contracts/forecast-tool-boundary-v1.yaml",
    "line": null,
    "description": "Cost axis C-08 (unbudgeted NeuralProphet training work) carries unbounded_pending_06_15; held open by the DECLARED-RED types::tests::no_cost_axis_is_pending. Owed by 06-15 T3.",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-07T15:58:32.289Z",
    "resolved_at": null
  },
  {
    "id": 11,
    "kind": "unrun-verify",
    "phase": "06",
    "file": "justfile",
    "line": 656,
    "description": "chronos-coldstart's 150 ms SC4 bar uses [ $med -ge 150 ]; an erroring token takes the not-taken branch and prints COLD START OK. Closed today only by the upstream sed digits-only parse, not by the bar (D-ITEM-06-16).",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-07T17:54:46.394Z",
    "resolved_at": null
  },
  {
    "id": 12,
    "kind": "deviation",
    "phase": "06",
    "file": "Makefile",
    "line": null,
    "description": "make contract-audit-phase6's success line contains the literal BIND-, so grep -c 'BIND-' == 0 is unsatisfiable; four consecutive plans each fixed it locally. Needs a shared scripts/assert_no_audit_findings.sh with a case table.",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-07T18:40:10.361Z",
    "resolved_at": null
  },
  {
    "id": 13,
    "kind": "deviation",
    "phase": "06",
    "file": "CHANGELOG.md",
    "line": null,
    "description": "The two breaking 0.63.0 surface changes (prophet::feature_row's fourth parameter, safetensors::load removal) are documented in the crate README but not yet in the root CHANGELOG; release-time follow-up.",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-07T18:40:10.438Z",
    "resolved_at": null
  },
  {
    "id": 14,
    "kind": "deviation",
    "phase": "05",
    "file": "scripts/setfit_bench_gate_doctor.py",
    "line": null,
    "description": "Probe helper extracted to a sibling .py not in 05-15 files_modified; bashrs cannot parse a quoted heredoc body (12 phantom shell errors). Load-bearing for the door probe.",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-11T22:23:42.703Z",
    "resolved_at": null
  },
  {
    "id": 15,
    "kind": "lint-warning",
    "phase": "05",
    "file": "crates/aprender-compute/src",
    "line": null,
    "description": "cargo clippy -p aprender-train --features setfit -- -D warnings exits 101 on pre-existing aprender-compute / aprender-present-terminal findings; the 05-15 plan verification line cannot pass on this tree.",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-11T22:23:42.783Z",
    "resolved_at": null
  },
  {
    "id": 16,
    "kind": "unmet-truth",
    "phase": "05",
    "file": "crates/aprender-train/src/train/setfit/bench_row.rs",
    "line": 37,
    "description": "Module doc claims canonical bytes are key-sorted (no preserve_order). Measured false: file-order reproduces both committed digests, sorted reproduces neither.",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-11T22:23:42.862Z",
    "resolved_at": null
  },
  {
    "id": 17,
    "kind": "deviation",
    "phase": "05",
    "file": "crates/aprender-train/src/train/setfit/bench_row.rs",
    "line": 37,
    "description": "The bench row seal is build-graph dependent: serde_json/preserve_order (via pmcp) makes to_canonical_bytes emit declaration order in apr-cli and key-sorted in aprender-train, so the same committed row verifies in one binary and is refused in the other (D-ITEM-05-17-A)",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-12T01:12:32.984Z",
    "resolved_at": null
  },
  {
    "id": 18,
    "kind": "deviation",
    "phase": "05",
    "file": "crates/aprender-test-lib/src/brick/pipeline.rs",
    "line": 1965,
    "description": "uuid_v4() is a timestamp, not a UUID; test_uuid_v4_generates_unique_ids fails 5/5 on this host and its verdict depends on clock resolution (D-ITEM-05-17-B)",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-12T01:12:33.076Z",
    "resolved_at": null
  },
  {
    "id": 19,
    "kind": "deviation",
    "phase": "06.1",
    "file": "contracts/forecast-tool-boundary-v1.yaml",
    "line": null,
    "description": "regressor_prior_scale_min is a representability bound, not the usability bound its rationale claimed; fit collapses below ~1e-8",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-21T14:23:10.699Z",
    "resolved_at": null
  },
  {
    "id": 20,
    "kind": "unrun-verify",
    "phase": "06.1",
    "file": "crates/aprender-image/src/lib.rs",
    "line": null,
    "description": "cargo fmt --all --check and full clippy -D warnings are red on the committed tree (pre-existing); phase used rustfmt per-file and clippy --no-deps instead",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-21T14:23:10.777Z",
    "resolved_at": null
  },
  {
    "id": 21,
    "kind": "deviation",
    "phase": "06.1",
    "file": "crates/aprender-forecast/src/invariance.rs",
    "line": null,
    "description": "06.1-04 verifies t1c and t3b are unsatisfiable by construction: both forbid any #[ignore] / any ignored test in the invariance module, but wave 1 deliberately ships capture_baseline as #[ignore]. Measured red on the untouched wave-1 tree. Intent-preserving substitutes pass.",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-21T16:45:55.473Z",
    "resolved_at": null
  },
  {
    "id": 22,
    "kind": "deviation",
    "phase": "06.1",
    "file": ".planning/phases/06.1-forecast-exogenous-inputs-prophet-regressors-neuralprophet-e/06.1-04-PLAN.md",
    "line": null,
    "description": "06.1-04 verify t2c python probe is blind to #[ignore] (it anchors test bodies at fn; the attribute precedes fn) and prefix-matches a renamed fn. Measured green on a part A carrying #[ignore]. Fixed in the in-tree Rust twin only; the verify body was kept verbatim.",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-21T16:45:55.552Z",
    "resolved_at": null
  },
  {
    "id": 23,
    "kind": "deviation",
    "phase": "06.1",
    "file": ".planning/phases/06.1-forecast-exogenous-inputs-prophet-regressors-neuralprophet-e/06.1-06-PLAN.md",
    "line": null,
    "description": "06.1-06 verify t1a check 3 (\"the prophet arm contains ZERO holiday bounds\") is UNSATISFIABLE on a correct hoist: the regressor name ceiling inside the prophet arm deliberately reuses MAX_HOLIDAY_NAME_LEN (landed by 06.1-03, which says so in its own comment). Measured red at forecast.rs:507/511 after a correct hoist. Repaired with a structurally-bounded exemption for that one block plus two CONTROLs; four broken-input probes observed red.",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-21T22:24:48.215Z",
    "resolved_at": null
  },
  {
    "id": 24,
    "kind": "deviation",
    "phase": "06.1",
    "file": ".planning/phases/06.1-forecast-exogenous-inputs-prophet-regressors-neuralprophet-e/06.1-06-PLAN.md",
    "line": null,
    "description": "06.1-06 verify t1a check 5 (no second np_event_design_cost ceiling) did not skip comment lines, so it went red on the hoisted site's own prose forbidding that constant. Measured. Narrowed to non-comment lines; re-probed red on a real `const MAX_NP_EVENT_DESIGN_COST`.",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-21T22:24:48.215Z",
    "resolved_at": null
  },
  {
    "id": 25,
    "kind": "unmet-truth",
    "phase": "06.1",
    "file": "crates/aprender-forecast/src/np.rs",
    "line": 539,
    "description": "SC3's per-column event recovery bar (event_effect_recovery_rel, 0.10) does NOT hold at the epoch budget np::door_epochs configures when n_lags > 0. MEASURED by 06.1-05 over four seeds; MEASURED here that the 320 AR cap -- the only budget at which the bar was observed to hold -- binds for no series of 51 points or more, and that even at 320 one seed at 2400 points reads 0.1663. Raising the budget to a flat 320 would also push C-08 over its bound at 5000+ points with lags. Blocks plan 06.1-06 Task 2 step 8; surfaced at the blocking-human checkpoint.",
    "status": "resolved",
    "reason": "Resolved by plan 06.1-06 Task 2 on the operator checkpoint ruling: the door now REFUSES holidays together with n_lags > 0, so SC3 recovery bar is no longer asserted over a configuration it fails. Recorded in equations.event_effect_recovery_rel preconditions with the rejected alternative and its numbers (the 320 cap binds only at n <= 50; one seed still reads 0.1663 at 320; a flat 320 prices at 191.7 % of C-08 at 5000 points, turning accepted requests into refusals).",
    "recorded_at": "2026-09-21T22:24:48.215Z",
    "resolved_at": "2026-09-21T23:03:16.632Z"
  },
  {
    "id": 26,
    "kind": "deviation",
    "phase": "06.1",
    "file": ".planning/phases/06.1-forecast-exogenous-inputs-prophet-regressors-neuralprophet-e/06.1-06-PLAN.md",
    "line": null,
    "description": "06.1-06 verify t2a counts tests with `.*(neuralprophet|np_).*(holiday|event)`, which requires the ARM token BEFORE the holiday/event token. MEASURED: 5 new cases named both but in the other order and only 1 matched. Resolved by renaming the tests to lead with the arm, keeping the verify body verbatim; the regex is still order-sensitive for any future case.",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-21T23:03:16.632Z",
    "resolved_at": null
  },
  {
    "id": 27,
    "kind": "deviation",
    "phase": "06.1",
    "file": ".planning/phases/06.1-forecast-exogenous-inputs-prophet-regressors-neuralprophet-e/06.1-06-PLAN.md",
    "line": null,
    "description": "06.1-06 Task 2 step 4's reconciliation premise is FALSE: published event components do NOT equal the events-ON minus events-OFF forecast difference, because the block's RNG draws shift the shuffle stream and the two fits diverge. MEASURED through the door: a near-constant level shift of -1.7864 (MAD 0.0703 over 32 event-free rows). The test now measures that shift from the zero-roll-up rows, asserts it is a level and not a shape difference, and reconciles to 0.0023-0.0212 against the 0.10 bar.",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-21T23:03:16.632Z",
    "resolved_at": null
  }
]
````
