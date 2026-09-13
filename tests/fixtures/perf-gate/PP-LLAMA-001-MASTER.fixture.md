# PP-LLAMA-001 — MASTER (test fixture)

**This is not the specification.** It is the input `scripts/spec_conformance.sh`
is tested against while `docs/specifications/PP-LLAMA-001-MASTER.md` is being
written, and it mirrors the §6 and §12 tables the real master must carry. When
the master lands, the guard reads THAT file by glob and this one keeps only its
selftest job: proving the parser reads the shape the master promises.

Use it explicitly:

    SPEC_CONFORMANCE_SPEC=tests/fixtures/perf-gate/PP-LLAMA-001-MASTER.fixture.md \
      bash scripts/spec_conformance.sh

## §6 Invariants

Column order is load-bearing: the parser reads `status` by header name and takes
the selftest names from the LAST cell. A name may carry a surface prefix — `pg:`
for `scripts/perf_gate.sh --list-selftests`, `sh:<script>:` for that script's own
case table, `rs:<crate>:` for a `#[test] fn` under `crates/<crate>/src`. The name
is always the part after the LAST colon.

| id | rule | must-fire | must-not-fire | status | producer · selftest |
|---|---|---|---|---|---|
| PP-1 | the expected cell set is enumerated in the matrix, and every cell is MEASURED, UNMEASURED-unexpired or NA | a cell's receipt is deleted | an NA cell with no receipt | ARMED | `scripts/perf_gate.sh` · `pg:cellset_missing` / `pg:cellset_na_ok` |
| PP-2 | every server fact comes from GET /v1/effective-config | server_config absent | server_config present and agreeing | ARMED | `scripts/perf_gate.sh` · `pg:config_missing` / `pg:config_present` |
| PP-3 | a ratio is representable only inside its band, beside a same-run baseline | a bare scalar agg_ratio at v3 | a paired ratios object | ARMED | `scripts/lib/bench_receipt.py` · `pg:ratio_bare` / `pg:ratio_paired` / `rs:aprender-test-lib:ratio_bare__a_scalar_ratio_is_unrepresentable` / `rs:aprender-test-lib:ratio_paired__a_same_run_baseline_joins` |
| PP-4 | agg, dec and prefill on every band; absence is fatal at v3 | prefill removed at c=4 | a v2 receipt read as historical | ARMED | `scripts/perf_gate.sh` · `pg:band_metric_absent` / `pg:historical_cited` |
| PP-5 | a timeout is fatal to the host's ratio; the drain is recorded | timeouts=1 | drain_ms present on every band | ARMED | `scripts/perf_gate.sh` · `pg:timeout_fatal` / `pg:drain_ok` |
| PP-6 | every arm runs at the phase perf-matrix.yaml declares for it | a release arm FAILs a merge | the same arm reports at merge | ARMED | `scripts/perf_gate.sh` · `pg:phase_guard_b_merge` / `pg:phase_guard_b_release` / `pg:phase_guard_a_merge` |
| PP-7 | raw per-request rows survive into the receipt | samples[] emptied | samples[] present | ARMED | `scripts/perf_gate.sh` · `pg:samples_stripped` / `pg:samples_ok` |
| PP-8 | both sides of a band were driven at the same client concurrency | comparator driven at c=1 under a c=4 label | both sides at c=4 | ARMED | `scripts/lib/parity_block.py` · `pg:client_conc_mismatch` / `pg:client_conc_ok` |
| PP-9 | a cell, once run, is spent | two RECORDED rows share a spend key | a new commit starts a new row | ARMED | `scripts/spec_conformance.sh` · `sh:scripts/spec_conformance.sh:respend_same_key` / `sh:scripts/spec_conformance.sh:respend_new_commit` |
| PP-10 | no request is admitted after the window closes | a sample issued at window_ms | the drain recorded instead | ARMED | `scripts/perf_gate.sh` · `pg:post_close_request` / `pg:drain_recorded` |
| PP-11 | the tokenization method is stated | method empty | method present | ARMED | `scripts/perf_gate.sh` · `pg:tokenization_absent` / `pg:tokenization_ok` |
| PP-12 | no speed claim without a receipt citation | a bare ratio literal in prose | the same literal beside its receipt | ARMED | `scripts/check_no_claim_literals.sh` · `sh:scripts/check_no_claim_literals.sh:claim_unreceipted` / `sh:scripts/check_no_claim_literals.sh:claim_receipted` |
| PP-13 | a server fact may not be inferred from a harness flag | compute_class disagrees with the server's | the two agree | ARMED | `scripts/perf_gate.sh` · `pg:inferred_field` / `pg:reported_field` |
| PP-14 | autofit may not overrule an explicit argument | autofit applied beside explicit args | autofit applied with none given | ARMED | `scripts/perf_gate.sh` · `pg:autofit_override` / `pg:autofit_ok` |
| PP-15 | an accelerator knob is a quantity, never a boolean | a boolean --gpu on a serve line | --gpu-layers with a number | ARMED | `scripts/check_comparator_flags.sh` · `sh:scripts/check_comparator_flags.sh:boolean_flag` / `sh:scripts/check_comparator_flags.sh:quantity_flag` |
| PP-16 | the declared compute class is one a build in this tree reaches | a class outside reachable_by | an NA host, decided and dated | ARMED | `scripts/perf_gate.sh` · `pg:class_unreachable` / `pg:class_na` |
| PP-17 | a ratio names the band it describes | a top-level ratios object | ratios inside a band with concurrency | ARMED | `scripts/lib/bench_receipt.py` · `pg:claim_bandless` / `pg:claim_named` |
| PP-18 | the served and driving binaries' commits are ancestors of the commit under test | a subject commit off the branch | both contained | ARMED | `scripts/perf_gate.sh` · `pg:ancestor_fail` / `pg:ancestor_ok` |
| PP-19 | a perf workflow cannot run concurrently with itself | a bench job with no concurrency group | the group declared | ARMED | `scripts/check_perf_concurrency_groups.sh` · `sh:scripts/check_perf_concurrency_groups.sh:isolation_breach` / `sh:scripts/check_perf_concurrency_groups.sh:isolation_ok` |
| PP-20 | the comparator pin carries an expiry and it is fresh | an expired pin | a fresh one | ARMED | `scripts/check_llama_pin.sh` · `sh:scripts/check_llama_pin.sh:pin_stale` / `sh:scripts/check_llama_pin.sh:pin_fresh` |
| PP-21 | a release receipt is signed and covers the commit under test | an unsigned receipt | a signed, fresh one | ARMED | `scripts/lib/receipt_sig.py` · `pg:sig_missing` / `pg:sig_ok` |
| PP-22 | two bands join only when their join keys agree | window differs between the lanes | the keys match | ARMED | `crates/aprender-test-lib/src/perf_gate/join.rs` · `pg:join_mismatch` / `pg:join_ok` / `rs:aprender-test-lib:join_mismatch__c4_against_c16_is_refused` / `rs:aprender-test-lib:join_ok__matching_keys_join` |
| PP-23 | a single stream cannot beat its own roofline | decode above the roofline | an aggregate above it under batching | ARMED | `scripts/perf_gate.sh` · `pg:roofline_exceeded` / `pg:roofline_aggregate_ok` |
| PP-24 | bands are derived from what both servers admitted | unequal admission, band still compared | the band marked NA | ARMED | `scripts/perf_gate.sh` · `pg:admission_unequal` / `pg:admission_na` |
| PP-25 | one client drives both lanes | the baseline names another client | the same client sha on both | ARMED | `scripts/perf_gate.sh` · `pg:client_mismatch` / `pg:client_ok` |
| PP-26 | a batched decode reproduces the m=1 token stream | witness absent at c>1 | the band marked INVALID-CORRECTNESS with no throughput | ARMED | `scripts/perf041_batched_parity_probe.py` · `pg:batch_invariance_fail` / `pg:batch_invariance_ok` / `sh:scripts/perf041_batched_parity_probe.py:witness_constant_token_m3` / `sh:scripts/perf041_batched_parity_probe.py:witness_identical_128_ok` |
| PP-27 | the stream is live, and the client witnesses it | stream_mode replayed | stream_mode live with a live witness | ARMED | `scripts/perf_gate.sh` · `pg:stream_replayed` / `pg:stream_live` / `pg:stream_absent` |
| PP-28 | the sampler is pinned and every completion reaches n_predict | short_of_n_predict > 0 | zero short completions | ARMED | `scripts/perf_gate.sh` · `pg:sampler_unpinned` / `pg:sampler_pinned` |
| PP-29 | every ARMED row's cases exist by name on their surface | a named case deleted | the full table | ARMED | `scripts/spec_conformance.sh` · `sh:scripts/spec_conformance.sh:conformance_missing` / `sh:scripts/spec_conformance.sh:conformance_ok` |
| PP-30 | the run states when it started and by which clock | started_utc absent | both present | ARMED | `scripts/perf_gate.sh` · `pg:timestamp_absent` / `pg:timestamp_ok` |
| PP-31 | a ratchet compares quantities, not scaling efficiency | agg(1) below the seeded baseline | agg(1) improved by 20% | ARMED | `scripts/perf_gate.sh` · `pg:self_regress_fail` / `pg:agg1_improve_ok` |
| PP-32 | an A/B record carries two shas of the same code, never a comparator | a comparator field in an A/B record | a code delta with two shas | ARMED | `crates/aprender-test-lib/src/perf_gate/ab.rs` · `rs:aprender-test-lib:abrecord_comparator__a_comparator_field_does_not_parse` / `rs:aprender-test-lib:abrecord_ok__a_code_delta_with_two_shas_parses` |
| PP-33 | every number a gate compares against lives in perf-matrix.yaml | a float threshold in a reader | the clean tree | ARMED | `scripts/check_thresholds_in_matrix.sh` · `sh:scripts/check_thresholds_in_matrix.sh:threshold_outside_matrix` / `sh:scripts/check_thresholds_in_matrix.sh:threshold_in_matrix` |

## §12 Owed work

`expires` is a DATE only on a ROOT row — one nothing blocks. Every other row's
expiry is DERIVED: the latest among its transitive blockers. A blocked row that
types a date is refused, because a typed expiry can fall before the work it
waits on, and the gate would then be red for a reason nobody can clear.

| row | what is owed | owner | blocked_by | expires |
|---|---|---|---|---|
| row-0a | thread the declared phase into every arm | perf-gate | — | 2026-09-30 |
| row-0b | pin the sampler on the wire | perf-gate | — | 2026-10-07 |
| row-6 | the effective-config route | serve | row-0b | derived |
| row-7 | the JOIN fixture and its refusals | perf-gate | row-0a | derived |
| row-15 | gx10 shakedown | perf-gate | row-6, row-7 | derived |
| row-18 | the reference measurement | perf-gate | row-15 | derived |
| row-21 | gx10 c=1 fixed cost | serve | row-15 | derived |
