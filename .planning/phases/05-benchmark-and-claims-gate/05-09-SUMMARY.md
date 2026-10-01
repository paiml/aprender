---
phase: 05-benchmark-and-claims-gate
plan: 09
subsystem: cli
tags: [benchmark, eval-03, eval-05, setfit, lora, resource-protocol, apr-cli, bashrs]

requires:
  - phase: 02-deterministic-pair-and-data-protocol
    provides: "read_attested_canonical / read_selection_manifest / Selection::replay — the three-call ingest both methods walk"
  - phase: 03-setfit-identity-and-lock
    provides: "create_selection_lock -> mint_test_token -> CanonicalTestAccess::grant, consumed here and reimplemented nowhere"
  - phase: 04-apr-artifact-and-production-parity
    provides: "reload_verified_run_from_apr, SetFitRun::into_artifact_bytes, setfit_train::atomic_write / refuse_existing_output"
  - phase: 05-benchmark-and-claims-gate
    provides: "05-05 BenchRow/RunManifest + the rewritten resource protocol; 05-06 ClassifyPipeline::load_adapter + predict_proba_tokenized + run_classify_core; 05-08 assemble_quality_block"
provides:
  - "`apr setfit bench run` — one cell in, one digest-committed row out, for both methods"
  - "`--record <ROW_FILE>` — verified transport ingest with NO execution (the D-09 GPU-host path)"
  - "`--cold-probe` / `--cold-probe-base` / `--probe-text` — the dedicated fresh child the resource protocol requires"
  - "`--base-model <FILE>` — the LoRA base, so base_model_bytes and deployable_total_bytes are computable"
  - "`ClassifyOutcome` from `run_classify_core` — epochs_completed, checkpoint dir, pipeline-read GPU identity"
  - "`entrenar::train::setfit::apr_evaluate::row_predictions_from_lora` — the LoRA entrance to ONE metric assembly"
  - "`data_tweeteval::dataset_revision_from_manifest` — the row's dataset_revision, from the file's one reader"
  - "`scripts/run_bench_cells.sh` — resume-capable, single-writer, halt-on-first-evidence-failure 40-cell driver"
affects: [05-10 claims gate, 05-11 the 40 remote 9B cells, 05-12 the 40 local SetFit cells]

actuals:
  tokens: 61000
  tasks: 3
  commits: 4

tech-stack:
  added: []
  patterns:
    - "Mechanism derived from the FORM THAT PARSED, never from cfg!(target_os) — the two benchmark hosts are deliberately different, so a parent that assumed its own platform would mislabel every transported number"
    - "Guard-regex case table with must-match AND must-not-match rows, where the must-not-match rows are the NEIGHBOURING lines of the real output rather than obviously unrelated text"
    - "A unit conversion pinned in BOTH directions, so a parser with the scaling backwards cannot satisfy an equality written only one way round"
    - "Read the resolution site, not the flag's name: `--output` sounded like a file and resolves as a directory"
    - "One ingest for both methods, so 'identical sampled IDs' is a fact about there being one file rather than about two readers agreeing"

key-files:
  created:
    - crates/apr-cli/src/commands/setfit_bench.rs
    - crates/apr-cli/src/commands/setfit_bench_tests.rs
    - scripts/run_bench_cells.sh
  modified:
    - crates/apr-cli/src/setfit_commands.rs
    - crates/apr-cli/src/dispatch_analysis.rs
    - crates/apr-cli/src/commands/mod.rs
    - crates/apr-cli/src/commands/setfit_train.rs
    - crates/apr-cli/src/commands/data_tweeteval.rs
    - crates/apr-cli/src/commands/finetune.rs
    - crates/apr-cli/Cargo.toml
    - crates/aprender-train/src/train/setfit/apr_evaluate.rs
    - Cargo.lock

key-decisions:
  - "`--cold-probe` is `hide_short_help`, not `hide`: it is machinery an operator must never type, and an auditor of the resource protocol must be able to find without reading the source. `-h` omits it; `--help` lists it."
  - "The peak-RSS mechanism is derived from the `/usr/bin/time` FORM that parsed, not from `cfg!(target_os)`. D-09 puts the two methods on two hosts, so a parent that assumed its own platform would mislabel every row that travelled."
  - "`--base-model <FILE>` is a separate flag from `--model-dir <DIR>` because they are different kinds of thing — a checkout directory versus one `.apr` whose byte length the row records."
  - "`run_classify_core` returns `Option<ClassifyOutcome>` rather than `()`. The `None` arms are the two paths that deliberately do not train, so the distinction lives in the type rather than in a boolean a caller can forget."
  - "The LoRA adapter is resolved BY NAME from the completed epoch, never as `the newest directory`: `ClassifyTrainer::save_checkpoint`'s result is discarded (`let _ = ...`), so a silently-failed final write must surface as a missing file rather than as an earlier epoch being attested."
  - "A `best/` directory existing refuses the row. 05-06 established that directory — not `save_every` — is the real model-selection surface, so its presence directly contradicts `no_selection_attestation`."
  - "`--force` on the LoRA ledger starts a FRESH ledger rather than appending, so `candidates_trained == 1` stays a true statement about the recorded run instead of a count of everything the directory ever saw."
  - "`row_predictions_from_lora` is a second ENTRANCE to one assembly, not a second assembly. Widening the SetFit credential to admit a LoRA pipeline would have made the witness a container for a claim; a per-method assembly would have been a second definition of F_avg."

requirements-completed: [EVAL-03, EVAL-05]

coverage:
  - id: D1
    description: "`apr setfit bench run` exposes three mutually exclusive modes with clap-enforced conflicts, and `--cold-probe` is in `--help` but not `-h`"
    requirement: "EVAL-03"
    verification:
      - kind: unit
        ref: "crates/apr-cli/src/commands/setfit_bench_tests.rs#setfit_bench_run_record_conflicts_with_every_execution_flag"
        status: pass
      - kind: unit
        ref: "crates/apr-cli/src/commands/setfit_bench_tests.rs#setfit_bench_run_cold_probe_conflicts_with_the_bench_dir_and_the_execution_flags"
        status: pass
      - kind: unit
        ref: "crates/apr-cli/src/commands/setfit_bench_tests.rs#setfit_bench_run_long_help_lists_the_machinery_flags"
        status: pass
    human_judgment: false
  - id: D2
    description: "The two-platform max-RSS parser is pinned by a must-match / must-not-match case table including the bytes-vs-kilobytes conversion in both directions"
    requirement: "EVAL-05"
    verification:
      - kind: unit
        ref: "crates/apr-cli/src/commands/setfit_bench_tests.rs#setfit_bench_max_rss_parser_case_table_must_match"
        status: pass
      - kind: unit
        ref: "crates/apr-cli/src/commands/setfit_bench_tests.rs#setfit_bench_max_rss_parser_pins_the_unit_conversion_in_both_directions"
        status: pass
      - kind: unit
        ref: "crates/apr-cli/src/commands/setfit_bench_tests.rs#setfit_bench_max_rss_parser_case_table_must_not_match"
        status: pass
      - kind: unit
        ref: "crates/apr-cli/src/commands/setfit_bench_tests.rs#setfit_bench_vm_hwm_parser_case_table"
        status: pass
    human_judgment: false
  - id: D3
    description: "A non-contracted cell key is a typed refusal naming setfit-benchmark-claims-v1, and every contracted cell resolves"
    requirement: "EVAL-03"
    verification:
      - kind: unit
        ref: "crates/apr-cli/src/commands/setfit_bench_tests.rs#setfit_bench_refuses_a_non_contracted_seed_naming_the_contract"
        status: pass
      - kind: unit
        ref: "crates/apr-cli/src/commands/setfit_bench_tests.rs#setfit_bench_refuses_an_empty_or_degenerate_shot_count"
        status: pass
      - kind: unit
        ref: "crates/apr-cli/src/commands/setfit_bench_tests.rs#setfit_bench_accepts_every_contracted_cell"
        status: pass
    human_judgment: false
  - id: D4
    description: "`--record` round-trips a valid transported row, is idempotent on an identical digest, and refuses a differing digest, a doctored payload and a misfiled name as DISTINCT errors"
    requirement: "EVAL-03"
    verification:
      - kind: unit
        ref: "crates/apr-cli/src/commands/setfit_bench_tests.rs#setfit_bench_record_ingests_a_valid_transported_row"
        status: pass
      - kind: unit
        ref: "crates/apr-cli/src/commands/setfit_bench_tests.rs#setfit_bench_record_is_idempotent_on_an_identical_digest"
        status: pass
      - kind: unit
        ref: "crates/apr-cli/src/commands/setfit_bench_tests.rs#setfit_bench_record_refuses_a_differing_digest_for_a_recorded_cell"
        status: pass
      - kind: unit
        ref: "crates/apr-cli/src/commands/setfit_bench_tests.rs#setfit_bench_record_refuses_a_doctored_digest"
        status: pass
      - kind: unit
        ref: "crates/apr-cli/src/commands/setfit_bench_tests.rs#setfit_bench_record_refuses_a_filename_that_disagrees_with_the_payload"
        status: pass
    human_judgment: false
  - id: D5
    description: "A completed cell's row file is write-once, its digest is the one the manifest records, and the row's lock_hash is recomputable from the committed lock bytes"
    requirement: "EVAL-03"
    verification:
      - kind: unit
        ref: "crates/apr-cli/src/commands/setfit_bench_tests.rs#setfit_bench_emit_row_refuses_an_existing_row_without_force"
        status: pass
      - kind: unit
        ref: "crates/apr-cli/src/commands/setfit_bench_tests.rs#setfit_bench_emit_row_records_the_digest_the_row_file_carries"
        status: pass
      - kind: unit
        ref: "crates/apr-cli/src/commands/setfit_bench_tests.rs#setfit_bench_row_lock_hash_is_the_digest_of_the_committed_lock_file"
        status: pass
    human_judgment: false
  - id: D6
    description: "LoRA rows split adapter bytes from deployable bytes, attest exactly one candidate, and the ledger refuses a second without --force"
    requirement: "EVAL-05"
    verification:
      - kind: unit
        ref: "crates/apr-cli/src/commands/setfit_bench_tests.rs#setfit_bench_lora_rows_split_adapter_bytes_from_deployable_bytes"
        status: pass
      - kind: unit
        ref: "crates/apr-cli/src/commands/setfit_bench_tests.rs#setfit_bench_lora_ledger_refuses_a_second_candidate_without_force"
        status: pass
      - kind: unit
        ref: "crates/apr-cli/src/commands/setfit_bench_tests.rs#setfit_bench_lora_ledger_force_starts_a_fresh_ledger_rather_than_appending"
        status: pass
      - kind: unit
        ref: "crates/apr-cli/src/commands/setfit_bench_tests.rs#setfit_bench_lora_ledger_is_appended_before_the_training_call_in_source_order"
        status: pass
    human_judgment: false
  - id: D7
    description: "The driver is sequential by construction, pins its binary, holds an atomic single-writer lock, resumes by digest, and halts on the first evidence-class failure with a distinct exit code"
    requirement: "EVAL-03"
    verification:
      - kind: unit
        ref: "crates/apr-cli/src/commands/setfit_bench_tests.rs#driver_has_no_parallel_dispatch_construct"
        status: pass
      - kind: unit
        ref: "crates/apr-cli/src/commands/setfit_bench_tests.rs#driver_holds_a_single_writer_lock_with_distinct_failure_exit_codes"
        status: pass
      - kind: unit
        ref: "crates/apr-cli/src/commands/setfit_bench_tests.rs#driver_resume_is_hash_based_not_a_bare_existence_check"
        status: pass
      - kind: unit
        ref: "crates/apr-cli/src/commands/setfit_bench_tests.rs#driver_never_reads_a_status_through_a_pipe"
        status: pass
      - kind: command
        ref: "bashrs lint scripts/run_bench_cells.sh -> rc=0, 0 errors, 0 warnings"
        status: pass
    human_judgment: false
  - id: D8
    description: "The SetFit and LoRA execution paths are NOT exercised end-to-end here — each costs a real training run"
    verification:
      - kind: manual
        ref: "05-12 (SetFit, local CPU) and 05-11 (LoRA, lambda-vector GPU) execute them for real"
        status: deferred
    human_judgment: true
    rationale: "A stub row labelled `real` would be worse than no test: it would put a green tick beside a claim nothing measured. The plan says so explicitly, and this SUMMARY says so rather than implying coverage the suite does not have."

duration: 175min
completed: 2026-08-18
status: complete
---

# Phase 5 Plan 09: `apr setfit bench run` — One Cell In, One Verified Row Out Summary

**`apr setfit bench run` now executes one benchmark cell for either method through single library doors, measures the rewritten EVAL-05 resource protocol with cold latency and inference peak RSS taken in a dedicated fresh child under `/usr/bin/time`, ingests a transported row without executing anything, and is driven across the 40-cell matrix by a sequential, single-writer, digest-resuming script that halts on the first evidence-class failure.**

## Performance

- **Duration:** ~175 min
- **Tasks:** 3
- **Commits:** 4
- **Files:** 12 (3 created, 9 modified)

## Task Commits

1. **Task 1: CLI wiring + resource measurement module** — `159f8ca25` (feat)
2. **Task 2: SetFit cell path + row emission + `--record`** — `923c48fe7` (feat)
3. **Task 3: LoRA cell path + the 40-cell driver** — `dd236bf40` (feat)
4. **Follow-on fix found by reading the resolution site** — `6fbdf71e6` (fix)

## Precondition, checked before any LoRA work

Task 3's plan text says: *read 05-06-SUMMARY.md first and use the exact `load_adapter` / `predict_proba_tokenized` entries it recorded. If that summary reports the preflight did not succeed, STOP.*

`05-06-SUMMARY.md` reports the preflight **SUCCEEDED**, and records the route explicitly: `Transformer::from_apr` → `ClassifyPipeline::from_model` → `ClassifyPipeline::load_adapter(dir/"model.adapter.apr")` → `predict_proba_tokenized(&[u32]) -> Vec<f32>`, with a two-sided control (fresh-process vs in-process max |diff| **0.000000000**; with-adapter vs without-adapter **0.194929659**). This plan calls those entries by name and invents no reload. Proceeding was therefore correct, not assumed.

## What the resource protocol actually does now

Three distinct surfaces, three separate fields, three mechanism strings:

| surface | where it is measured | mechanism |
|---|---|---|
| TRAIN peak RSS | inside the training process, sampler opened before `SetFitRun::prepare` and closed the moment training ends | `vm_hwm` (Linux `/proc/self/status`) or `sysinfo_sampled_<hz>` with the **achieved** rate |
| COLD latency + INFERENCE peak RSS | a dedicated fresh child that loads the written artifact and classifies **once** | `child_max_rss_time_l` (macOS) or `child_max_rss_vm_hwm` (Linux) |
| WARM median + throughput | against the RELOADED model in the measuring process | n/a (wall clock) |

The child is spawned under `/usr/bin/time` (`-l` / `-v`) through `std::env::current_exe`, never a bare `apr`. Its `ExitStatus` is reaped off the `Output` **on its own line** — never through a pipe.

### The parser hazard, and why it has a case table

macOS `/usr/bin/time -l` reports the maximum RSS in **BYTES**; GNU `/usr/bin/time -v` reports it in **KILOBYTES**. The identical numeral means two different quantities depending on which block it came from, and D-09 puts the two methods on two different hosts — so this is not a hypothetical.

The mechanism is therefore **derived from the form that parsed**, not from `cfg!(target_os)`. A parent that assumed its own platform would mislabel every transported number. The must-not-match rows are the *neighbours* of the wanted lines, because that is where a loose pattern actually goes wrong:

- macOS `average shared memory size` — same column layout, ends in `size`
- macOS `peak memory footprint` — a different metric with the same value
- GNU `Average resident set size (kbytes): 0` — one word from the maximum
- `Maximum resident set size (bytes): N` — a unit this parser has never seen, refused rather than read as kB

And the conversion is pinned **in both directions**: `4096` in the macOS block is 4096 bytes, `4096` in the GNU block is 4,194,304, and the ratio is asserted as exactly 1024 in that direction — an equality written only one way round is satisfied by a parser with the scaling backwards.

## Deviations from Plan

### Auto-fixed issues

**1. [Rule 3 — Blocking] `run_classify_core` returned `()`, so no LoRA row could be built from it**

- **Found during:** Task 3
- **Issue:** The plan says the LoRA path calls `run_classify_core`. That function returned `Result<()>`, and it holds — and then drops — every value a LoRA row attests: `TrainResult.epochs_completed`, the checkpoint directory, the resolved `TransformerConfig`/`ClassifyConfig` needed to rebuild the pipeline for reload, and the pipeline-read GPU identity.
- **Fix:** It now returns `Result<Option<ClassifyOutcome>>`. `None` is the two paths that deliberately do not train (`--plan-only`; no selection and no `--data`), so a benchmark cell's typed refusal for those lives in the type rather than in a boolean. The CLI flag path discards the value.
- **Why not re-derive in the adapter:** a second pipeline construction and a second `TrainingConfig` in `setfit_bench.rs` is exactly the "two descriptions of one run" that `ClassifyRun` was extracted to prevent (OPS-03).
- **Files:** `crates/apr-cli/src/commands/finetune.rs`
- **Committed in:** `dd236bf40`

**2. [Rule 3 — Blocking] `RowPredictions` has no public constructor, and the LoRA method can never mint the SetFit credential**

- **Found during:** Task 3
- **Issue:** `assemble_quality_block` takes `RowPredictions`, whose fields are private by design — a SetFit value of it is EVIDENCE that a measurement went through the credentialed load ladder. The LoRA baseline has no `setfit-apr-v1` artifact and never will, so that ladder is unreachable for it.
- **Fix:** Added `entrenar::train::setfit::apr_evaluate::row_predictions_from_lora`, a second ENTRANCE to the one assembly. It validates shapes against `ordered_labels`, bounds-checks the truth vector, refuses an empty split, and records `artifact_hash` rather than trusting it. It cannot forge a SetFit claim: the hash it stamps is the adapter's, and the row carrying it declares `method: lora`, which `BenchRow::from_bytes` independently requires to agree with a `MethodEvidence::Lora` block.
- **Rejected alternatives, stated in the doc comment:** a second assembly for LoRA would be a second definition of `F_avg`/MCC/the confusion matrix/the calibration diagnostics, invisible because both would look right; widening `ReloadedSetFitCredential` would make the witness a container for a claim.
- **Files:** `crates/aprender-train/src/train/setfit/apr_evaluate.rs`
- **Committed in:** `dd236bf40`

**3. [Rule 3 — Blocking] `benchmark-manifest.json` had one reader, and the row needs `dataset_revision` from it**

- **Found during:** Task 2
- **Issue:** The row records the pinned upstream revision the directory was prepared from. `data_tweeteval.rs` is the single reader of that file, and its own header records why a second reader is a defect.
- **Fix:** `dataset_revision_from_manifest` was added to the OWNER rather than the consumer, so the file still has one reader.
- **Files:** `crates/apr-cli/src/commands/data_tweeteval.rs`
- **Committed in:** `923c48fe7`

**4. [Rule 2 — Missing critical] Nothing checked that the cell key and the selection manifest described the SAME draw**

- **Found during:** Task 2
- **Issue:** `Selection::replay` proves the manifest describes THIS dataset. The row records `shots`/`seed` from the FLAGS. Nothing connected the two — so `--seed 13` against a manifest drawn at seed 17 would publish a row filed under seed 13 whose rows are seed 17's, and **every paired delta in the report would compare two different draws while looking correctly paired**. That is precisely the comparison PF-007 exists to forbid.
- **Fix:** `read_phase2` refuses a `--seed` or `--shots` that disagrees with the replayed selection. Both methods go through it.
- **Committed in:** `923c48fe7` / `dd236bf40`

**5. [Rule 2 — Missing critical] The pairing key had two derivations**

- **Found during:** Task 3
- **Issue:** 05-06's LoRA path records `hex(replayed.semantic_hash())`; the SetFit path as first written read the manifest envelope's `semantic_hash` string field. Those should be equal — and if they ever were not, one method's rows would pair on a value the other's never carried, and the report would silently drop every delta rather than fail.
- **Fix:** `read_phase2` derives it from the replayed selection (05-06's spelling) and **cross-checks it against the envelope**, refusing a disagreement by name.
- **Committed in:** `dd236bf40`

**6. [Rule 2 — Missing critical] A silently-failed final checkpoint write would have been attested as the trained model**

- **Found during:** Task 3
- **Issue:** `ClassifyTrainer` writes checkpoints with `let _ = self.save_checkpoint(...)` — the result is discarded. A resolver that took "the newest `epoch-*` directory" would therefore hash an EARLIER epoch's adapter and record it as the run's output.
- **Fix:** The adapter is resolved BY NAME from `epoch-{epochs_completed - 1}` and its absence is a typed refusal that says the final write failed silently. A `best/` directory existing is a separate refusal, because that is the actual model-selection surface (05-06's finding) and its presence contradicts `no_selection_attestation`.
- **Committed in:** `dd236bf40`

**7. [Rule 1 — Bug] The driver passed a FILE path to `apr data select --output`, which takes a DIRECTORY**

- **Found during:** post-Task-3 verification, by reading the resolution site rather than the flag's name
- **Issue:** `run_select` resolves it as `output.unwrap_or(data).join(SELECTION_MANIFEST_FILE)`. The driver would have written `.../s8-seed13.json/selection-manifest.json` and pointed every subsequent `--selection` at a path that does not exist — surfacing forty cells later as "file not found", long after the sweep had been started.
- **Fix:** Per-cell selection DIRECTORIES; the consumer reads `selection-manifest.json` from the same directory; the generator asserts the file APPEARED (an `apr data select` exiting 0 is not evidence that this path now holds a manifest); a driver-gate test pins the two path spellings together.
- **Committed in:** `6fbdf71e6`

---

**Total deviations:** 7 auto-fixed (1 bug, 3 missing critical, 3 blocking). No Rule 4 architectural decision arose.

### Additions beyond the plan's artifact list, and why

- **`--base-model <FILE>`.** The plan's flag list has `--model-dir <DIR>` only, but a LoRA row records `base_model_sha256`, `base_model_bytes` and `deployable_total_bytes = base + adapter`, none of which is computable without naming the base. Reusing `--model-dir` would have made one flag mean a checkout directory for one method and a single `.apr` for the other.
- **`--cold-probe-base <BASE>`.** The cold probe must reload the LoRA pair, not a standalone `setfit-apr-v1`. Its presence selects the reload route; `requires = "cold_probe"` means it cannot be passed alone.
- **`--cold-probe` uses `hide_short_help`, not `hide`.** The plan calls it "a hidden measurement mode" and its acceptance criterion requires `--help` to list it. `hide_short_help` satisfies both readings: out of `-h`, in `--help`.

### Ordering note

`--record` landed in the Task-1 commit rather than Task-2's. It is a pure function of the transported bytes and depends on nothing Task 2 builds; its tests are in the Task-2 commit. Recorded because the commit boundary does not match the plan's task boundary.

## The bashrs finding, measured with a two-sided control

`bashrs lint scripts/run_bench_cells.sh` reported `IDEM002: Non-idempotent rm - add -f flag` at column 32 of a **comment line**. The rule matches the substring `rm`, and the word it fired on was **"form"**.

Verified rather than assumed, and stated as a control per CLAUDE.md rule 7:

```
# reword "canonical form here" -> "canonical path here"   ->  0 errors, 0 warnings
# restore  "canonical form here"                          ->  IDEM002 at 127:35-37
```

Nothing else in the file changed between the two runs. The comment was reworded rather than suppressed with `--ignore IDEM002`, because a suppression at that scope would also hide a real `rm`. The false positive is recorded here so the next reader does not re-derive it.

Final lint state: **rc=0, 0 errors, 0 warnings, 38 infos** (the infos are `SC1091` "not following sourced file", `SC1012` `\n` in single quotes inside `printf` format strings, and `REL003` "`read` without `-t`" — all correct-as-written for this script).

## What the driver guarantees, and how each is gated

| guarantee | mechanism | gate |
|---|---|---|
| single writer | `set -o noclobber` + redirect (O_EXCL, atomic; scoped to a subshell so `>` keeps working afterwards) | `driver_holds_a_single_writer_lock_with_distinct_failure_exit_codes` |
| no parallelism | no `--jobs`, no `xargs -P`, no line-terminal `&` | `driver_has_no_parallel_dispatch_construct` |
| resume is hash-based | the row's own `semantic_hash` must be the digest the run manifest recorded | `driver_resume_is_hash_based_not_a_bare_existence_check` |
| halt on evidence failure | `is_evidence_failure` classifies the refusal text; `EXIT_EVIDENCE=3` vs `EXIT_TRANSIENT=4` vs `EXIT_LOCKED=5` | same test |
| rc never through a pipe | every `rc=$?` follows a non-pipeline command | `driver_never_reads_a_status_through_a_pipe` (which also asserts it had something to scan) |
| coverage is not vacuous | `total -ne 40` is a hard failure | `driver_covers_exactly_the_contracted_matrix` |

A stale lock is **reported, never stolen**: a driver that broke another's lock because the pid looked dead would be doing exactly the concurrent write the lock prevents, on the one occasion the pid check was wrong.

## Verification, as measured

| command | result |
|---|---|
| `cargo test -p apr-cli --lib --features setfit setfit_bench` | **47 passed**, 0 failed, rc=0 |
| `cargo test -p apr-cli --lib --features setfit finetune` | **81 passed**, rc=0 (no regression from the `ClassifyOutcome` change) |
| `cargo test -p aprender-train --lib --features setfit apr_evaluate` | **22 passed**, rc=0 |
| `cargo check -p apr-cli --features setfit --lib` | rc=0 |
| `bashrs lint scripts/run_bench_cells.sh` | rc=0, **0 errors, 0 warnings** |
| `cargo clippy -p apr-cli --features setfit --lib --tests` | 0 findings in changed files |
| `cargo clippy -p aprender-train --features setfit --lib` | 0 findings in changed files |

Acceptance greps, run comment-filtered where the criterion says so:

| grep | result |
|---|---|
| `--jobs\|xargs -P\|&$` in `run_bench_cells.sh`, comments stripped | **0** |
| `\|\s*grep.*\$\?` in `setfit_bench.rs`, comments stripped | **0** |
| `current_exe` in `setfit_bench.rs` | 3 |
| `Selection::replay` | 3 |
| `load_setfit_apr` | 3 |
| `ExecutionBackend::identity` | 1 |
| library doors combined (`Selection::replay`, `load_setfit_apr`, `evaluate_rows_from_artifact`, `assemble_quality_block`, `ExecutionBackend::identity`, `run_classify_core`, `load_adapter`, `predict_proba_tokenized`) | 28 |
| `sysinfo = { workspace = true }` in `crates/apr-cli/Cargo.toml` | 1 (no version literal) |

**One honest qualification on `ExecutionBackend::identity`.** The CLI cannot call it: no `ExecutionBackend` value is reachable out-of-crate. The SetFit row's `backend_identity` is `ClassifyResponse::backend()`, which core's own doc states IS `ExecutionBackend::identity` called on the value the encode invocation RETURNED — there is no parameter, no setter and no configuration path to that field. The grep matches the comment that says so. That is the correct route and the only honest one; it is recorded here rather than left to look like a literal call.

## Known Stubs

None. The two method paths are complete implementations. The intermediate `setfit_cell`/`lora` refusal placeholders that existed in commit `159f8ca25` were replaced by the real implementations in `923c48fe7` and `dd236bf40` respectively, and no refusal constant survives.

## Threat Flags

None beyond the plan's `<threat_model>`. Each mitigated threat has a named passing test:

| threat | mitigation | test |
|---|---|---|
| T-05-09-01 spoofed backend identity | read from execution on both sides | `setfit_bench_lora_backend_identity_cannot_fabricate_a_gpu` |
| T-05-09-02 tampered transported row | digest + schema + cell + filename verified before recording | `setfit_bench_record_refuses_a_doctored_digest`, `..._refuses_a_filename_that_disagrees_with_the_payload` |
| T-05-09-03 uncontracted cell | typed refusal naming the contract; 42 explicitly excluded | `setfit_bench_refuses_a_non_contracted_seed_naming_the_contract` |
| T-05-09-04 repudiated resource numbers | mechanism string mandatory per row | `setfit_bench_names_all_four_mechanism_strings`, `..._sampled_mechanism_always_carries_a_nonzero_interval` |
| T-05-09-05 oversized reads | bounded from the stat'd length before the read | `read_bounded`'s two-stage cap (declared length, then `take(cap + 1)`) |
| T-05-09-06 mislabelled resource numbers | fresh child, separate train field, form-derived mechanism, two-platform case table | D2's four tests |
| T-05-09-07 undeclared second LoRA candidate | append-only ledger before training; second refuses; count asserted | D6's four tests |
| T-05-09-08 concurrent writers | atomic noclobber lock, no parallel construct, atomic-rename row writes | D7's tests + `setfit_bench_creates_files_only_through_the_shared_atomic_writer` |

## Issues Encountered

### libtest's 2 MiB stack cannot BUILD `apr`'s clap tree

The first clap tests aborted with `has overflowed its stack` inside clap's own `Command` construction, before any assertion ran — `apr` has 103 subcommands and the tree build recurses deep. Every clap-tree test now runs on a 32 MiB thread via `on_roomy_stack`, and the parse result crosses the boundary as an `ErrorKind` (a `Copy` discriminant) so assertions name a KIND rather than matching rendered prose.

### `--probe-text` alone parsed successfully when `--bench-dir` was also present

Measured, not theorised: clap's `requires` did not fire when the required argument was in a `conflicts_with` relationship with another supplied flag. The test now exercises `requires` without the conflicting flag, and adds the positive half so the two refusals are not satisfied by a parser that refuses everything.

### Out of scope, left alone

`cargo fmt -p apr-cli` and `-p aprender-train` reformat files this plan does not touch — `commands/serve/handlers.rs`, `train/setfit/apr_reload.rs`, `bench_row.rs`, `bench_row_tests.rs` (131 lines across the four). All were reverted with targeted `git checkout --` on each file; none is in this plan's commits. They are pre-existing formatting drift and want their own change.

## Next Phase Readiness

**Ready for 05-10 (the claims gate):**
- Rows land at `{bench-dir}/rows/{method}-s{shots}-seed{seed}.json` with the run manifest at `{bench-dir}/run-manifest.json`.
- The SetFit lock record is COMMITTED at `{bench-dir}/locks/setfit-s{shots}-seed{seed}.lock.json` and the row's `lock_hash` is the SHA-256 of exactly those bytes — so the gate recomputes rather than trusts.
- The LoRA candidate ledger is at `{bench-dir}/ledger/lora-s{shots}-seed{seed}.jsonl`, one line, digest and line count both on the row.

**Ready for 05-11 / 05-12 (the 80 cells):**
- `scripts/run_bench_cells.sh METHOD BENCH_DIR DATA_DIR MODEL_OR_BASE` is resume-capable and lint-clean.
- `--record` ingests the GPU host's rows with full verification and no execution.

**Carried forward, unchanged and NOT closed by this plan:**
- **05-06's blocker for 05-11 stands.** On CPU, `apr finetune --task classify` never touches the LoRA adapters — a graph cut in `ClassificationHead::mean_pool` means the backward pass never enters the transformer, so it is a linear probe on a frozen encoder. Whether the GPU path shares that cut is a different question this plan does not answer and must not be read as answering. The cheapest check remains re-running `cpu_training_moves_the_head_but_not_the_lora_adapters_pre_existing` under `--features cuda` on lambda-vector: if it FAILS there, the GPU path trains adapters and the matrix is safe to schedule.
- **Neither method's execution path has been run end-to-end.** That is deliberate (see coverage D8) and is 05-11's and 05-12's work.

## Self-Check: PASSED

Files verified present:
- `crates/apr-cli/src/commands/setfit_bench.rs` — FOUND (98.1K)
- `crates/apr-cli/src/commands/setfit_bench_tests.rs` — FOUND (63.9K)
- `scripts/run_bench_cells.sh` — FOUND (15.4K, executable)

`must_haves.artifacts` contents verified:
- `setfit_bench.rs` contains `pub(crate) fn run` — 1 match
- `setfit_commands.rs` contains `BenchCommands` — 2 matches
- `run_bench_cells.sh` contains `set -euo pipefail` and `. scripts/apr_bin.sh || exit 1` — 1 each

Commits verified in `git log`: `159f8ca25`, `923c48fe7`, `dd236bf40`, `6fbdf71e6` — all 4 FOUND.

Working tree clean at the point of this check.

---
*Phase: 05-benchmark-and-claims-gate*
*Completed: 2026-08-18*
