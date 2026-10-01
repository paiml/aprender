# The claims gate's row-supplied input surface, enumerated by class

**Plan:** 05-15 (gap closure for verifier gap 1, EVAL-04) · **Derived:** 2026-09-11
**Derived from:** `crates/aprender-train/src/train/setfit/bench_row.rs` (the schema) and
`crates/aprender-train/src/train/setfit/bench_gate.rs` (every consumer), read at
`bfa6a25a3`, by inspection of the two files — not from a SUMMARY and not from the plan.

## Why this file exists

Verifier gap 1 found ONE unvalidated row-supplied string
(`payload.evidence.setfit.lock.lock_record_path`). Verifier gap 2 found a second
(`payload.selection_manifest_hash`). Patching the two that were measured and stopping is the
recurrence mode this round exists to break, so the whole surface is enumerated here and every
entry is put in one of three classes:

- **(i) recomputed from committed bytes** — the gate opens a file at a location the row cannot
  choose and derives the value itself.
- **(ii) compared against a contract-derived value** — the gate holds a constant (or a
  contract-declared conjunct) and refuses a row that disagrees.
- **(iii) trusted as written** — the gate reads the value and carries it, comparing it against
  nothing.

**A (iii) entry with no reason in its `evidence` cell is the defect this artifact exists to
prevent.** Every (iii) row below either states which plan moves it, or states why no committed
bytes and no contract-derived value exist to check it against.

Line numbers are as of `bfa6a25a3`, BEFORE this plan's Task 2 edit. Where a row says "closed in
this round", Task 2 is the edit that closes it and the line number is the pre-edit site.

---

## The row envelope (`BenchRow`)

| field | consumed by | class | evidence |
|---|---|---|---|
| `semantic_hash` | `BenchRow::from_bytes` (`bench_row.rs:598`) | **(i)** | `sha256_hex(&row.payload.to_canonical_bytes()?)` is recomputed and compared BEFORE the value is returned, so no caller can hold a row whose digest disagrees with its payload. |
| `volatile.created_at` | nothing | **(iii)** | Deliberately outside the digest (`bench_row.rs:277-288`): re-running a cell on a newer build must not change the row's identity. The gate never reads it, so no published number depends on it. Carried. |
| `volatile.tool_version` | nothing | **(iii)** | Same reason as `created_at`. The gate never reads it. Carried. |

## `BenchRowPayload` — the hashed payload

| field | consumed by | class | evidence |
|---|---|---|---|
| `schema_version` | `BenchRow::from_bytes` (`bench_row.rs:591`) | **(ii)** | Compared against `BENCH_ROW_SCHEMA_VERSION`; checked FIRST, because a foreign schema's digest convention is not necessarily this one's. |
| `contract_id` | read at `bench_row.rs:485` (declaration), never compared | **(iii)** | **The THIRD unvalidated field, found by this enumeration.** See "The `contract_id` grep, accounted for" below. **Closed in this round** by Task 2 layer 3b: a class-(ii) comparison against `CLAIMS_CONTRACT_ID` in `verify_row_evidence`, refusing with the existing `RowSchemaRefused`. |
| `method` | `CellKey::is_contracted` (`bench_row.rs:607` via `:261`); `BenchRow::from_bytes` tag/block agreement (`:613`); `verify_row_evidence` slot check (`bench_gate.rs:765`) | **(ii)** | Compared against `BENCH_METHODS`, against the evidence block's own tag, and against the manifest slot the row was filed under — three independent statements of the same fact. |
| `shots` | `CellKey::is_contracted` (`bench_row.rs:262`); slot check (`bench_gate.rs:765`) | **(ii)** | Compared against `BENCH_SHOTS` and against the manifest slot. |
| `seed` | `CellKey::is_contracted` (`bench_row.rs:263`); slot check (`bench_gate.rs:765`) | **(ii)** | Compared against `BENCH_SEEDS` and against the manifest slot. 42 is deliberately not in that list. |
| `dataset_revision` | `aggregate` reads nothing from it; no gate consumer | **(iii)** | Carried. The pinned revision lives in `tweet-eval-stance-benchmark-v1.yaml`, but the DATASET is not under `bench_dir` — there are no committed bytes in the benchmark directory to fingerprint a revision against. A test (`bench_metrics_label_order_is_evidence_from_the_pinned_dataset_revision`) pins the revision at build time; the gate does not check it at read time. |
| `dataset_fingerprint` | no gate consumer | **(iii)** | Carried. A whole-dataset fingerprint from the attested `PreparedDataset`; the dataset is not committed under `bench_dir`, so no bytes exist to recompute it from. |
| `model_revision` | no gate consumer | **(iii)** | Carried. The encoder checkout is not under `bench_dir`. |
| `selection_manifest_hash` | `verify_pairing` (`bench_gate.rs:902`) compares it to the OTHER method's row; `verify_provenance`'s LoRA arm (`:1028`) compares it to each ledger line | **(iii)** | **Gap 2.** Both consumers compare it to another row-supplied value, never to the committed `selections/` manifests — which the gate never opens (spot-checks F and G: `rm -rf selections/` leaves the verdict at rc=0). **Closed in 05-16**, which recomputes the committed manifest's own `semantic_hash` and refuses a disagreement. |
| `backend_identity` | `aggregate` (`bench_gate.rs:1435`), published as `backends` | **(iii)** | Carried. Read from execution at emission time (Ph4 D-12); nothing in the benchmark directory records the backend independently, so there is nothing to recompute against. It is published as a distinct-value list rather than compared. |
| `host.hostname` | `aggregate` (`bench_gate.rs:1431`) | **(iii)** | Carried. A recorded fact about where the cell ran; no committed bytes attest it. Published so the renderer can say resource figures are never pooled across hosts (D-09). |
| `host.os` | `aggregate` (`bench_gate.rs:1433`) | **(iii)** | Carried, same reason as `hostname`. |
| `host.arch` | `aggregate` (`bench_gate.rs:1433`) | **(iii)** | Carried, same reason as `hostname`. |

## `payload.quality` (`QualityBlock`)

| field | consumed by | class | evidence |
|---|---|---|---|
| `f_avg` | `aggregate` (`bench_gate.rs:1405`), the HEADLINE | **(iii)** | Spot-check D: doctoring it to 0.99 with all digests repaired yields rc=0 and moves the published mean. **05-17 moves it to (i)** by recomputing `(F1[1] + F1[2]) / 2` in closed form from the row's own `confusion_matrix`. |
| `f_avg_bits` | `aggregate` (`bench_gate.rs:1411`), published per seed | **(iii)** | Carried beside `f_avg` so a decimal rendering can never become the compared value. **05-17 moves it to (i)** with `f_avg`: the bits must equal `f_avg.to_bits()`. |
| `macro_f1` | `aggregate` (`bench_gate.rs:1406`) | **(iii)** | **05-17 moves it to (i)** — recomputable from `confusion_matrix`. |
| `macro_f1_bits` | `aggregate` (`bench_gate.rs:1412`) | **(iii)** | **05-17 moves it to (i)** with `macro_f1`. |
| `per_class_precision` | no gate consumer (published in the row only) | **(iii)** | **05-17 moves it to (i)** — recomputable from `confusion_matrix`. |
| `per_class_recall` | no gate consumer | **(iii)** | **05-17 moves it to (i)**, same basis. |
| `per_class_f1` | no gate consumer | **(iii)** | **05-17 moves it to (i)**, same basis; it is what `f_avg` and `macro_f1` reduce. |
| `mcc` | `aggregate` (`bench_gate.rs:1407`) | **(iii)** | **05-17 moves it to (i)** — Matthews correlation is closed-form over the confusion matrix. |
| `mcc_bits` | `aggregate` (`bench_gate.rs:1413` via `SeedValue`) | **(iii)** | **05-17 moves it to (i)** with `mcc`. |
| `confusion_matrix` | no gate consumer | **(iii)** | **STAYS (iii), and it is the base the 05-17 recomputation rests on.** No committed file carries per-row predictions, so the matrix itself cannot be recomputed — it can only be made the single source the derived metrics must agree with. Recorded explicitly so 05-17's gain is not overstated: it converts eight independently-forgeable numbers into one. |
| `n_test_rows` | no gate consumer | **(iii)** | **05-17 moves it to (i)**: it must equal the sum of `confusion_matrix`. |
| `ordered_labels` | no gate consumer | **(iii)** | Carried. The declared label map; the contract's class-label map lives in `tweet-eval-stance-benchmark-v1.yaml` at its pinned revision, which the gate does not load. A contract-derived comparison is possible in principle and is not taken in this round; recorded as open rather than closed. |
| `ece_top_label_validation` | no gate consumer | **(iii)** | **CANNOT move class.** See "Still trusted, with reason". |
| `ece_top_label_validation_bits` | no gate consumer | **(iii)** | Cannot move — the value it mirrors cannot. |
| `brier_multiclass_validation` | no gate consumer | **(iii)** | **CANNOT move class.** See "Still trusted, with reason". |
| `brier_multiclass_validation_bits` | no gate consumer | **(iii)** | Cannot move — the value it mirrors cannot. |
| `calibration_split` | no gate consumer; SET (not checked) at emission by `bench_metrics.rs:262` | **(iii)** | The library holds the contract-derived constant `CALIBRATION_SPLIT` (`bench_row.rs:145`), and `bench_metrics::check_evidence` uses it only at EMISSION time to select which split's probabilities reach the calibration functions — never when the gate READS a committed row. **Closed in this round** by Task 2: a one-line class-(ii) comparison. |

## `payload.resource` (`ResourceBlock`)

| field | consumed by | class | evidence |
|---|---|---|---|
| `train_wall_ms` | `aggregate` (`bench_gate.rs:1419`) | **(iii)** | Carried. A measurement; no committed bytes and no contract-derived value exist to check it against. Its credibility comes from the disclosed threat model ("this report proves consistency, not truth"), not from a check. |
| `cold_latency_ms` | `aggregate` (`bench_gate.rs:1425`) | **(iii)** | Carried, same reason. Its measurement BOUNDARY is checked (next row); its VALUE is not. |
| `cold_measured_in_child_process` | no gate consumer | **(iii)** | The contract pins it to the literal `true` (`setfit-benchmark-claims-v1.yaml`, `resource_protocol`: "a row may not carry a cold latency measured in a process that also trained"). Set unconditionally by the CLI writer (`setfit_bench.rs:1488`, `:1920`); never checked on read. **Closed in this round** by Task 2: a one-line class-(ii) comparison against `true`. |
| `warm_latency_ms_median` | `aggregate` (`bench_gate.rs:1426`) | **(iii)** | Carried, same reason as `train_wall_ms`. |
| `throughput_rows_per_sec` | `aggregate` (`bench_gate.rs:1427`) | **(iii)** | Carried, same reason. |
| `throughput_batch_size` | `aggregate` (`bench_gate.rs:1428`), published as `throughput_batch_sizes` | **(iii)** | **Carried, with reason — the one of these four NOT closed in this round.** The contract does not pin a value: it says only "the batch size that pass used". The writer's `THROUGHPUT_BATCH_SIZE = 32` lives in `apr-cli` (`setfit_bench.rs:405`) and is a PRODUCER choice, not a contract constant; importing it into the verifier would refuse a legitimately different batch size while calling it a contract violation. What makes throughput comparable is the field's PRESENCE beside the number, which the schema already enforces (non-`Option`, `deny_unknown_fields`), and the renderer publishes the distinct set so an unequal batch size is visible rather than averaged away. |
| `warmup_count` | no gate consumer | **(iii)** | The contract pins it to 3 ("the number of warmup classifies discarded before the warm measurement (3)") and the library holds `WARMUP_COUNT = 3` (`bench_row.rs:148`). Used by the CLI writer at emission (`setfit_bench.rs:1492`); never checked on read. **Closed in this round** by Task 2: a one-line class-(ii) comparison. |
| `train_peak_rss_bytes` | `aggregate` (`bench_gate.rs:1420`) | **(iii)** | Carried, same reason as `train_wall_ms`. |
| `train_peak_rss_mechanism` | `aggregate` (`bench_gate.rs:1429`) then `mechanism_class` (`:1465`) | **(iii)** | Carried. It IS classified — `mechanism_class` maps it to exact / sampled / `Unrecognised`, and `Unrecognised` is never comparable to anything including itself — but the class is PUBLISHED to the renderer rather than refused. An unrecognised mechanism therefore does not invalidate the report; it makes the report say the figure is not comparable. Recorded as a deliberate design choice, not an oversight. |
| `inference_peak_rss_bytes` | `aggregate` (`bench_gate.rs:1421`) | **(iii)** | Carried, same reason as `train_wall_ms`. |
| `inference_peak_rss_mechanism` | `aggregate` (`bench_gate.rs:1430`) then `mechanism_class` (`:1468`) | **(iii)** | Carried, same reason as `train_peak_rss_mechanism`. |
| `peak_rss_sample_interval_hz` | no gate consumer | **(iii)** | Carried. The contract says it is present EXACTLY when a mechanism is a sampled one; the gate does not enforce that conditional. Recorded as open — closing it would be a cross-field consistency rule (present iff `mechanism_class(...) == SampledLowerBound`), which is a different shape from the one-line constant comparisons this round closes, and it is not selected here. |
| `artifact_bytes` | `aggregate` (`bench_gate.rs:1422`) | **(iii)** | Carried. The artifact it measures (a ~90 MB `.apr` per cell) is deliberately NOT in the index, so no committed bytes exist to re-measure. This is the same reason `evidence_table_hash` and `apr_artifact_sha256` cannot move. |
| `deployable_total_bytes` | `aggregate` (`bench_gate.rs:1423`) | **(iii)** | Carried, same reason as `artifact_bytes`. |

## `payload.evidence` — `setfit` arm (`SetfitEvidence`, `BenchLockRef`)

| field | consumed by | class | evidence |
|---|---|---|---|
| `evidence_table_hash` | no gate consumer | **(iii)** | Carried. The hash is OF the APR artifact's evidence table, and the APR is not committed under `bench_dir` (40 × ~90 MB, deliberately not in the index), so no committed bytes exist to recompute it from. The verifier's independent check that all 40 are distinct is a property of the data, not a gate. |
| `apr_artifact_sha256` | no gate consumer | **(iii)** | Carried, same reason: the artifact it names is not in the index. |
| `lock.lock_hash` | `verify_provenance` (`bench_gate.rs:943-944`) | **(i)** | `sha256_hex(&bytes)` over the bytes actually read from the committed lock record, compared to the row's claim; a disagreement is `ProvenanceMismatch` naming both the cell and the file. |
| `lock.role` | `verify_provenance` (`bench_gate.rs:956`) | **(ii)** | Compared against `LOCK_ROLES` (`["written", "consumed"]`), the `apr eval` `LockRow` vocabulary. |
| `lock.rule` | `verify_provenance` (`bench_gate.rs:966`) | **(ii)** | Compared against `SelectionRule::MaxMetricLowestIndexTieBreak.tag()` — a lock written under an unrecognised rule is a selection nothing can replay. |
| `lock.lock_record_path` | `verify_provenance` (`bench_gate.rs:940-941`) | **(iii)** | **GAP 1, the measured hole.** `bench_dir.join(PathBuf::from(row_string))` with no validation: `Path::join` with an absolute component DISCARDS the base, and `..` is never resolved. Spot-check E drove it end-to-end through `apr setfit bench report` — committed lock deleted, row pointing outside the tree, digests repaired — and the door exited 0 while printing its own attestation that provenance had been recomputed from the committed lock bytes. **Closed in this round** by Task 2: `resolve_committed_evidence_path` refuses it syntactically (absolute / rooted / `..` / empty) and then by canonical containment (symlinks), as `BenchGateError::EvidencePathEscape`. |

## `payload.evidence` — `lora` arm (`LoraEvidence`), deferred scope

The active scope is one method (D-19), so no LoRA row reaches the shipped door today. These
entries are enumerated anyway: the class of every field is what a restored arm INHERITS, and the
point of this artifact is that a future field cannot hide behind the ones already measured.

| field | consumed by | class | evidence |
|---|---|---|---|
| `base_model_sha256` | no gate consumer | **(iii)** | Carried. The base model is an ~18 GB checkpoint that is not, and will not be, under `bench_dir`. |
| `base_model_bytes` | no gate consumer | **(iii)** | Carried, same reason. Mandatory in the schema because an adapter alone is not deployable, but not re-measurable from committed bytes. |
| `adapter_sha256` | no gate consumer | **(iii)** | Carried. The adapter is not committed under `bench_dir`. |
| `epochs_requested` | `verify_lora_attestation` (`bench_gate.rs:1056`) | **(ii)** | Compared against `epochs_completed` — the contract's own `no_selection_attestation` conjunct, so the comparison is contract-derived rather than invented here. |
| `epochs_completed` | `verify_lora_attestation` (`bench_gate.rs:1056`) | **(ii)** | Same conjunct, other side. A run that ended somewhere other than where it was told to ended at a checkpoint somebody chose. |
| `early_stopping_disabled` | `verify_lora_attestation` (`bench_gate.rs:1067`) | **(ii)** | Compared against the contract's literal `true`. |
| `val_split` | `verify_lora_attestation` (`bench_gate.rs:1074`) | **(ii)** | Compared against the contract's literal `0.0`. |
| `no_selection_attestation` | `verify_lora_attestation` (`bench_gate.rs:1084`) | **(ii)** | Compared against the contract's literal `true`. |
| `candidate_ledger_sha256` | `verify_provenance` (`bench_gate.rs:984-985`) | **(i)** | Recomputed over the committed ledger's bytes. |
| `candidates_trained` | `verify_provenance` (`bench_gate.rs:1001-1002`); `verify_lora_attestation` (`:1091`) | **(i)** and **(ii)** | COUNTED from the ledger's non-empty lines and compared to the row's claim, and separately compared against `CONTRACTED_CANDIDATES_TRAINED = 1`. The strongest entry in this table: the value is both derived from bytes and bounded by the contract. |
| `candidate_ledger_path` | `verify_provenance` (`bench_gate.rs:981-982`) | **(iii)** | **The SAME hole as `lock_record_path`, on the deferred arm** — identical `bench_dir.join(row_string)` with no validation. **Closed in this round** by Task 2 through the SAME helper at `EvidenceKind::Ledger`, so a restored LoRA arm inherits the fix instead of re-opening it. |

## The run manifest (`RunManifest` / `RunManifestPayload` / `CellEntry`)

The manifest is producer-written too, and the gate reads it before any row byte.

| field | consumed by | class | evidence |
|---|---|---|---|
| `semantic_hash` | `RunManifest::from_bytes` (`bench_row.rs:904`); `verify_manifest_digest` (`bench_gate.rs:786-790`) | **(i)** | Recomputed from the payload's own canonical bytes at BOTH doors, because a manifest can also be built in memory and no gate may depend on which constructor its argument came through. |
| `volatile.created_at` / `volatile.tool_version` | nothing | **(iii)** | Never hashed, never read. Carried for the same reason as the row's volatile block. |
| `payload.schema_version` | `RunManifest::from_bytes` (`bench_row.rs:897`) | **(ii)** | Compared against `RUN_MANIFEST_SCHEMA_VERSION`. |
| `payload.contract_id` | declared at `bench_row.rs:750`, stamped by `declare_for` at `:853`, never compared on read | **(iii)** | **The manifest half of the third unvalidated field.** A manifest declaring a foreign contract is accepted, and `aggregate` then stamps the published `RunAggregate` with `setfit-benchmark-claims-v1` regardless (`bench_gate.rs:1486`). **Closed in this round** by Task 2 layer 3b. |
| `payload.cells[]` (the SEQUENCE) | `RunManifest::from_bytes` (`bench_row.rs:919`); `verify_run_scoped` (`bench_gate.rs:644`) | **(ii)** | SEQUENCE equality against `RunManifest::expectation_for(scope)`, which is derived in code from the contract's constants — so cardinality, membership and order are one comparison. This is what refuses spot-check C's self-consistent 39-cell manifest. |
| `cells[].method` / `.shots` / `.seed` | the sequence comparison above | **(ii)** | They ARE the compared sequence; no cell key can be present that the contract does not derive. |
| `cells[].status` | `verify_entry_complete` (`bench_gate.rs:809-821`) | **(ii)** | Compared against `CellStatus::Complete`; a `pending` entry in a pre-declared table is what makes a selectively omitted cell visible. |
| `cells[].row_sha256` | `verify_row_evidence` (`bench_gate.rs:756`) | **(i)** | Compared against the digest recomputed from the ROW FILE's own payload bytes. Distinct from the row's self-consistency check: this one catches a self-consistent row SUBSTITUTED for the one the run recorded. |

---

## The `contract_id` grep, accounted for

The value of this artifact is that a reviewer can re-run the grep and land on the same table, so
every hit is accounted for here — not only the interesting ones.

`grep -n CLAIMS_CONTRACT_ID` over the two files returns exactly FIVE sites:

| site | what it is | why it is not a comparison |
|---|---|---|
| `bench_row.rs:57` | `pub const CLAIMS_CONTRACT_ID: &str = "setfit-benchmark-claims-v1";` | The declaration itself. A constant compares nothing. |
| `bench_row.rs:853` | `contract_id: CLAIMS_CONTRACT_ID.to_string()` inside `declare_for` | The WRITER. It STAMPS the manifest it builds; it never reads one. |
| `bench_gate.rs:97` | the `use super::bench_row::{...}` import line | An import. It brings the name into scope for the two sites below. |
| `bench_gate.rs:418` | interpolated into `ExpectationSetMismatch`'s `Display` | It NAMES the contract inside a CELL-COUNT refusal message. The refusal is about the cell sequence; the contract id is prose in the message. |
| `bench_gate.rs:1486` | `contract_id: CLAIMS_CONTRACT_ID.to_string()` inside `aggregate` | The output STAMP on the published `RunAggregate`. It asserts the contract the numbers were computed under; it does not check what the inputs declared. |

Separately, the `contract_id` FIELD is declared at three sites, read and carried, never compared:

| site | what it is | why it is not a comparison |
|---|---|---|
| `bench_row.rs:485` | `BenchRowPayload::contract_id` | A `String` field. `deny_unknown_fields` proves the KEY is spelled right; nothing constrains the VALUE. |
| `bench_row.rs:750` | `RunManifestPayload::contract_id` | Same: a `String` the parser accepts unconditionally. |
| `bench_gate.rs:1286` | `RunAggregate::contract_id` | The OUTPUT struct's field, populated by the `:1486` stamp above. An output field cannot validate an input. |

**Zero of those eight sites compares a row's or a manifest's declared `contract_id` against the
constant.** A row declaring a foreign contract is therefore accepted today, and `aggregate`
publishes it stamped `setfit-benchmark-claims-v1` regardless — the published payload would assert
a contract the inputs never claimed. Task 2 closes it.

---

## Closed in this round

| entry | from | to | closed by |
|---|---|---|---|
| `payload.evidence.setfit.lock.lock_record_path` | (iii) | (i) — resolved at a location the row cannot choose, then recomputed | 05-15 Task 2, `resolve_committed_evidence_path` |
| `payload.evidence.lora.candidate_ledger_path` | (iii) | (i) — the SAME helper at `EvidenceKind::Ledger` | 05-15 Task 2 |
| `payload.contract_id` | (iii) | (ii) — compared against `CLAIMS_CONTRACT_ID` | 05-15 Task 2 layer 3b |
| `RunManifestPayload.contract_id` | (iii) | (ii) — same comparison, manifest side | 05-15 Task 2 layer 3b |
| `payload.quality.calibration_split` | (iii) | (ii) — compared against `CALIBRATION_SPLIT` | 05-15 Task 2 |
| `payload.resource.warmup_count` | (iii) | (ii) — compared against `WARMUP_COUNT` | 05-15 Task 2 |
| `payload.resource.cold_measured_in_child_process` | (iii) | (ii) — compared against the contract's literal `true` | 05-15 Task 2 |

Scheduled, not done here:

| entry | from | to | closed by |
|---|---|---|---|
| `payload.selection_manifest_hash` | (iii) | (i) — recomputed from the committed selection manifest | 05-16 (gap 2) |
| `payload.quality.f_avg` (+ `_bits`), `macro_f1` (+ `_bits`), `mcc` (+ `_bits`), `per_class_precision`, `per_class_recall`, `per_class_f1`, `n_test_rows` | (iii) | (i) — recomputed in closed form from the row's own `confusion_matrix` | 05-17 |

## Still trusted, with reason

Every remaining (iii) entry, grouped by the reason it stays one. None is left implicit.

**A. No committed bytes exist to recompute against — the artifact is deliberately not in the
index.**
`payload.evidence.setfit.evidence_table_hash`, `payload.evidence.setfit.apr_artifact_sha256`,
`payload.resource.artifact_bytes`, `payload.resource.deployable_total_bytes`,
`payload.evidence.lora.base_model_sha256`, `payload.evidence.lora.base_model_bytes`,
`payload.evidence.lora.adapter_sha256`.
The SetFit artifacts are ~90 MB each across 40 cells and the LoRA base model is ~18 GB; they are
outside the index by design, so the benchmark directory holds no bytes a gate could hash. Closing
these would require committing the artifacts, which is a different decision from this round's.

**B. No committed bytes and no contract-derived value — a measurement is what it is.**
`payload.resource.train_wall_ms`, `cold_latency_ms`, `warm_latency_ms_median`,
`throughput_rows_per_sec`, `train_peak_rss_bytes`, `inference_peak_rss_bytes`.
These are timings and kernel high-water marks. Nothing recomputes a wall clock. Their standing
rests on the report's own disclosed residual — it proves consistency, not truth — and that
disclosure is in-band rather than inferred.

**C. Present but classified rather than refused, deliberately.**
`payload.resource.train_peak_rss_mechanism`, `payload.resource.inference_peak_rss_mechanism`.
`mechanism_class` maps each to exact / sampled / `Unrecognised`, and `Unrecognised` is never
comparable to anything including itself — but the class is published to the renderer rather than
made a refusal, so an unknown mechanism degrades a COMPARISON rather than invalidating a RUN.

**D. The contract pins no value.**
`payload.resource.throughput_batch_size` — the contract says "the batch size that pass used".
Comparing against the producer's `THROUGHPUT_BATCH_SIZE = 32` would import a producer choice into
the verifier. See its table row above.

**E. A cross-field conditional this round does not close.**
`payload.resource.peak_rss_sample_interval_hz` — contracted as present exactly when the mechanism
is a sampled one. A one-line constant comparison cannot express it; it needs a two-field rule.
Recorded as open.

**F. The identifying facts of the run, which nothing under `bench_dir` attests.**
`payload.dataset_revision`, `payload.dataset_fingerprint`, `payload.model_revision`,
`payload.backend_identity`, `payload.host.hostname`, `payload.host.os`, `payload.host.arch`,
`payload.quality.ordered_labels`.
`ordered_labels` is the one of these that a contract-derived comparison COULD reach — the class
map is in `tweet-eval-stance-benchmark-v1.yaml` at its pinned revision — but the gate does not
load that contract, so closing it means adding a second contract to this module's compile-time
surface. Not selected this round; recorded as open rather than as impossible.

**G. Outside the digest by design, and read by nothing.**
`volatile.created_at`, `volatile.tool_version`, on both the row and the manifest. Excluding them
from the digest is what lets a re-run on a newer build produce a byte-identical row identity. The
gate reads neither, so no published number can depend on them.

**H. CANNOT move class — no committed file carries the inputs.**
`payload.quality.ece_top_label_validation` (+ `_bits`) and
`payload.quality.brier_multiclass_validation` (+ `_bits`).
Both are computed from PER-ROW PREDICTED PROBABILITIES over the validation split
(`bench_metrics.rs:236-243`). The row commits the test-split `confusion_matrix`, which is a tally
of ARGMAX decisions — it discards the probability vector entirely — and no committed file under
`bench_dir` carries validation-split probabilities. So unlike `f_avg`/`macro_f1`/`mcc`, which
05-17 recomputes from the matrix, these two have no recomputable basis at all and stay (iii) with
that reason written down rather than left implicit. Closing them would require committing a
per-row probability file per cell, which is a schema change, not a gate change.

**I. The base the 05-17 recomputation rests on.**
`payload.quality.confusion_matrix`. It cannot itself be recomputed (same reason as H: no committed
predictions), so 05-17's gain is a REDUCTION of the forgeable surface — eight independently
doctorable numbers collapse to one — not an elimination of it. Stated here so the next round's
claim is not read as stronger than it is.
