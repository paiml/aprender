Integrated receipt for the #4033 stack on branch feat/4040-nightly, base = #4046 head 6db770d2f. Counts are MEASURED at the commit this receipt ships in.

## What is in the stack
- **#4037** carry-forward:
  - ladder_carry: the closure derived from cargo metadata; embedded inputs from a balanced-paren parse of include_str!/include_bytes! (concat!/env!-aware) and existing-path build.rs literals; measurement inputs from receipt-schema writers.
  - M2 wiring into check_model_ladder.sh (equiv_lines covers ladder AND CRUX receipt shas).
- **#4051** timing stamps, per apr CALL (ladder) and per engine LINE (CRUX), with lock_wait_s.
  - This is a stated NARROWING of "per gate": apr qa runs its gates in ONE process, so its inner gates carry apr's own duration_ms, not separate stamps.
  - The ladder runs one qa per rung; there are no per-thinking-leg qa calls to stamp.
- **#4040** nightly:
  - certify_nightly.sh runs full ladder + full CRUX on BOTH lanes (gpu and cpu: every rung claims both). A RED night is re-measured on the next run, never cached.
  - check_model_ladder.sh --nightly admits the newest GREEN, fresh, upstream and COHERENT nightly (green re-derived from its receipts; a future t_end is refused), then binds it to the cut via the carry, or it is STALE.
  - The version-only-bump carry.

## Tables (every mutant killed by its NAMED row)
- ladder_carry_cases.py: 45 rows / 27 mutants
- nightly_admission_cases.py: 16 rows / 11 mutants
- check_certify_nightly.sh: 14 rows / 12 mutants
- crux_stamps_cases.py: 10 rows / 8 mutants
- check_crux_inference_judge.sh: 165 ok with mutants
- check_model_ladder.sh --self-test: 155/0. It includes:
  - the END-TO-END nightly rows through the real gate: e2e-green (positive control), e2e-timing-green, e2e-timing-unstamped (the #4051 rule ON through --nightly), e2e-cpu-lane-missing, e2e-ancestor-stale, and mutant dirs-unlinked;
  - the version-bump case pair;
  - the --scope/--nightly wiring rows.
- check_crux_ollama_in_lock.sh runs the REAL crux_inference_dogfood.sh with STUB engines and stub apr. There is no GPU and no real model. It asserts 160/160 engine rows stamped, and the no-stamp mutant names all 160.

## Not done here, stated
- Wiring --nightly into the release path (autopilot T-1 / check_publish_preflight) is M4 on feat/4045-release-gate-normal (--scope release requires --nightly). The T-1 call-site switch follows once the timer (paiml/infra#959) has produced a green nightly on each host. The cop gated it: not before 0.69.1 publishes, and after one manual run per host.
- The carry is whole-receipt, not per cell. Per-architecture resolution is #4049 (parked).
- The saving:
  - measured carry rate on main's last 20 PRs is 7/20 (posted on #4037);
  - a carried cut skips the full ladder (~127 host-min on gx10) plus full CRUX (~70 min on lambda);
  - a delta that touches the closure re-measures in full;
  - the realized minutes need the nightly to exist; they are measured per release under #4045 M7.
- check_release_models_t1.sh exits 2 on this branch AND on the #4046 head alone. It is pre-existing in the merge-back, not from this diff.
