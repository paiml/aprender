//! Claims-gate tests (plan 05-10, EVAL-04).
//!
//! Every test name starts `bench_gate_`, and the module path itself contains `bench_gate`, so
//! `cargo test -p aprender-train --lib --features setfit bench_gate` selects exactly this file.
//! The COUNT matters as much as the status: a name-filtered `cargo test` that matches nothing
//! prints `test result: ok. 0 passed` and exits 0 (CR-02), so the Make floor reads the matched
//! count out of the log rather than trusting `ok`.
//!
//! # How every negative here is built, and the ONE rule that governs the list
//!
//! Each is built by MUTATING a programmatically-generated VALID run, so the mutation is the only
//! difference between a green run and a red one — a hand-written broken fixture proves only that
//! some bytes are refused, never that THIS defect is what refused them.
//!
//! **THE SCOPE A NEGATIVE IS MUTATED AT IS PART OF WHAT IT PROVES.** Since the claims contract's
//! 2.0.0 narrowing (D-19) the ACTIVE expectation set is 40 cells, one method. A proof taken at
//! one scope does NOT transfer to another (CLAUDE.md Verification Discipline rule 4), which is
//! why the `scope` column below exists and why two entries deliberately overlap in shape while
//! differing in scope. The Makefile banner at the `setfit-bench-gate` leg carries the same
//! account; they must agree.
//!
//! Three scopes appear:
//!
//! * **`verify_run` / ACTIVE 40** — through the PUBLIC two-argument door, over the shipped
//!   expectation set. This is what a user runs.
//! * **`verify_run_scoped` / DEFERRED 80** — the two-method scope, `#[cfg(test)]`-gated. The
//!   shapes that exist ONLY in a two-method design (an unpaired pair, a forged LoRA ledger)
//!   need rows production code cannot construct. `D-ITEM-05-15` restores this arm; these
//!   negatives are retained, contract-bound and unexercised by any shipped door.
//! * **resolver helper, directly** — `resolve_committed_evidence_path` called with an evidence
//!   kind and a declared string. This holds the resolver's two stages; it does NOT hold the
//!   wiring that makes the resolver reachable from a row at all.
//!
//! # The negative inventory
//!
//! | # | doctored shape | asserted variant | scope | added by |
//! |---|---|---|---|---|
//! | 1 | a cell left `pending` (selective omission) | `incomplete_cell` | `verify_run` / 40 | 05-10, re-mutated 05-11 |
//! | 2 | a required evidence block removed from a row | `row_schema_refused` | `verify_run` / 40 | 05-10, re-mutated 05-11 |
//! | 3 | one pair's two rows on different selection manifests | `unpaired_selection` | deferred 80 | 05-10 |
//! | 4 | a payload byte edited without resealing | `row_digest_mismatch` | `verify_run` / 40 | 05-10, re-mutated 05-11 |
//! | 5 | post-test selection (lock rule; epochs_completed) | `post_test_selection` | `verify_run` / 40 and deferred 80 | 05-10, re-mutated 05-11 |
//! | 6 | FORGED PROVENANCE (two-line ledger; edited lock file) | `provenance_mismatch` | deferred 80 | 05-10 |
//! | 7 | a manifest declaring a second method's cell | `expectation_set_mismatch` | `verify_run` / 40 | 05-11 |
//! | 8 | a second method's ROW in a declared slot | `row_slot_mismatch` | `verify_run` / 40 | 05-11 |
//! | 9 | an escaping `lock_record_path` — absolute, `..`, last-component symlink | `evidence_path_escape` | `verify_run` / 40 | 05-15 |
//! | 10 | thirteen path shapes x two evidence kinds | `evidence_path_escape`, `evidence_file_missing`, `evidence_read_failed`, and three ACCEPTANCE rows | resolver helper | 05-15 |
//! | 11 | a deleted committed lock record | `evidence_file_missing` | `verify_run` / 40 | 05-15 |
//! | 12 | a row declaring a foreign `contract_id` | `row_schema_refused` | `verify_run` / 40 | 05-15 |
//! | 13 | a MANIFEST declaring a foreign `contract_id` | `row_schema_refused` | `verify_run` / 40 | 05-15 |
//! | 14 | a contract-pinned constant a row may not choose (three fields) | `row_schema_refused` | `verify_run` / 40 | 05-15 |
//! | 15 | TWO cells carrying an escaping path | `evidence_path_escape` on the FIRST in contract order | `verify_run` / 40 | 05-15 |
//! | 16 | the SELECTION-BINDING case table: a deleted `selections/` tree, a deleted per-cell directory, a zero-byte manifest, a row hash doctored to 64 zeros, a row hash doctored to another cell's real hash, three transplanted manifests WITH the row hash doctored to match, the same transplant WITHOUT it, and an unsealed manifest payload — plus one ACCEPTANCE row | `evidence_file_missing`, `evidence_read_failed`, `selection_manifest_mismatch`, `selection_manifest_cell_mismatch`, and `Ok` | `verify_run` / 40 | 05-16 |
//! | 17 | a transplanted manifest at BOTH ENDS of the shot axis (s16 under s8, s32 under s64) | `selection_manifest_cell_mismatch` | `verify_run` / 40 | 05-16 |
//! | 18 | a pairing key agreeing in its first characters — last byte flipped, truncated to 16, upper-cased | `selection_manifest_mismatch` | `verify_run` / 40 | 05-16 |
//! | 19 | TWO cells carrying a broken selection binding | `selection_manifest_mismatch` on the FIRST in contract order | `verify_run` / 40 | 05-16 |
//! | 20 | the DEFERRED-scope pairing negative re-asserted after the fixture change | `unpaired_selection`, NOT one of entry 16's tags | deferred 80 | 05-16 |
//! | 21 | the QUALITY CROSS-CHECK case table: spot-check D replayed (`f_avg` -> 0.99 with the row envelope digest, the manifest `row_sha256` and the manifest envelope digest all repaired), `macro_f1`, `mcc`, one `per_class_f1` element, `n_test_rows`, `f_avg_bits` with the decimal untouched, the decimal with the bits untouched, `ece_top_label_validation_bits`, a ragged matrix, a 4x4 matrix against three labels and an all-zero matrix — plus one ACCEPTANCE row | `quality_cross_check_mismatch`, `row_schema_refused`, and `Ok` | `verify_run` / 40 | 05-17 |
//! | 22 | spot-check D replayed through the SINGLE-CELL door | `quality_cross_check_mismatch`, with a passing control on the untouched cell | `verify_cell` | 05-17 |
//! | 23 | the synthetic fixture's two dispersion properties and its `n_test_rows`-against-matrix-total identity | pairwise-distinct ascending `f_avg`, bit-identical at `zero_variance_shots`, `n_test_rows == sum(matrix)` | fixture | 05-17 |
//!
//! Entries 9 and 10 overlap by design: three of entry 10's rows are the same SHAPES as entry
//! 9's, proven at a different scope. Deleting either for duplicating the other is exactly the
//! scope transfer rule 4 forbids. Entry 20 is the same discipline pointed the other way: it
//! asserts that 05-16's step-6 addition did not PRE-EMPT a step-5 negative and quietly change
//! which rule that negative observes. 05-17 extends this table when it adds its own.
//!
//! Entry 16's transplant rows are the ones most easily made vacuous, and the reason is stated
//! at the table itself: without the accompanying row-hash doctoring the HASH check fires first
//! and the cell-key check those rows exist to prove is never reached. The inverting row (the
//! same transplant WITHOUT the doctoring, expecting `selection_manifest_mismatch`) is what
//! keeps that honest.
//!
//! Every negative in this file runs in a default `cargo test -p aprender-train --lib --features
//! setfit` invocation: no `#[ignore]`, no extra feature, no network, no fixture file. The
//! deferred-scope entries run too — what is deferred is the PRODUCTION arm, not the test.
//!
//! # What these negatives do NOT prove
//!
//! They prove the gate detects INCONSISTENT evidence, and since 05-15 that a row cannot point
//! the gate at bytes outside the benchmark directory. They prove nothing about TRUTHFUL
//! provenance: a producer holding both the rows and the lock/ledger files can still emit a
//! mutually consistent forgery. That residual is stated in `bench_gate.rs`'s module doc and in
//! the contract's own `selection_safety_evidence.residual_risk`, and it is repeated here so a
//! reader of the test list does not conclude more from a green suite than it supports.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use aprender_contrastive_data::dedup::ExclusionRecord;
use aprender_contrastive_data::ledger::AccessRecord;
use aprender_contrastive_data::manifest::{
    SelectedExampleRecord, SelectionManifest, SelectionPayload, VolatileMetadata,
    SELECTION_SCHEMA_VERSION,
};
use serde::Deserialize;
use tempfile::TempDir;

use super::*;
use crate::train::setfit::bench_row::ExpectationScope;
use crate::train::setfit::bench_row::{
    BenchLockRef, BenchRowPayload, HostIdentity, LoraEvidence, QualityBlock, ResourceBlock,
    SetfitEvidence, BENCH_ROW_SCHEMA_VERSION, CALIBRATION_SPLIT, WARMUP_COUNT,
};

// ===========================================================================================
// The synthetic run builder
// ===========================================================================================

/// The knobs the doctored negatives and the zero-variance test turn.
#[derive(Debug, Clone, Copy, Default)]
struct RunSpec {
    /// When set, every seed at this shot level produces the IDENTICAL paired delta, which is
    /// what `paired_ci95_df9` refuses with `ZeroVarianceDifferences`.
    zero_variance_shots: Option<u32>,
}

/// Position of `value` in `list`, as an `f64` multiplier.
fn index_of<T: PartialEq + Copy>(list: &[T], value: T) -> usize {
    list.iter().position(|item| *item == value).unwrap_or(0)
}

/// The three labels every synthetic row declares, in the pinned dataset's own order.
fn synthetic_labels() -> Vec<String> {
    vec!["none".to_string(), "against".to_string(), "favor".to_string()]
}

/// The synthetic confusion matrix for one cell — row-major `[true][predicted]`, three classes.
///
/// **Since 05-17 this matrix is the SOURCE of every quality number the fixture publishes.**
/// Before it, `synthetic_quality` wrote a fixed `[[10,1,1],[1,10,1],[1,1,10]]` beside
/// `n_test_rows: 35` (that matrix totals **36**), `macro_f1 = f_avg - 0.05`, `mcc = f_avg - 0.10`
/// and the constants `[0.6, 0.5, 0.4]` for all three per-class vectors — so every synthetic row
/// was internally inconsistent with the matrix printed next to it. A closed-form cross-check
/// added over that fixture would have gone red for the right reason at the wrong time: the
/// failure would have been the fixture, not the gate.
///
/// Two dispersion properties are load-bearing for tests that predate this plan, so they are
/// built into the matrix's shape rather than left to the caller — and each is ASSERTED by
/// `bench_gate_the_synthetic_fixture_keeps_its_two_dispersion_properties` rather than argued
/// here:
///
/// * **At a non-degenerate shot level the ten seeds must give ten DISTINCT `f_avg` values**, or
///   `seed_dispersion_ci95` takes the zero-variance branch everywhere and the interval arm of
///   the aggregate is never exercised. `seed_index` moves counts from the `(against, none)` cell
///   onto `against`'s diagonal, which makes `F1[against]` STRICTLY INCREASING in the seed index
///   while `F1[favor]` is untouched. Strict monotonicity is what makes the ten distinct;
///   *increasing* is what keeps `min` at the first seed and `max` at the last, which
///   `bench_gate_aggregate_recomputes_the_closed_form_summary_from_the_rows` asserts.
/// * **At `zero_variance_shots` the ten seeds must be BIT-IDENTICAL per method**, so the seed
///   index is pinned to `0` at that shot level and the matrix is a function of `(method, shots)`
///   alone. Identical matrices produce identical `f64`s by construction, and the paired delta
///   over them is therefore bit-identical too.
///
/// `shots_index` moves `favor`'s diagonal, so the four shot levels differ from one another;
/// `method` moves the same `against` cell the seed does, which keeps SetFit above LoRA exactly
/// as the arithmetic generator this replaced did.
fn synthetic_confusion(spec: RunSpec, cell: CellKey) -> Vec<Vec<u64>> {
    let shots_index = index_of(&BENCH_SHOTS, cell.shots) as u64;
    let seed_index = if spec.zero_variance_shots == Some(cell.shots) {
        0
    } else {
        index_of(&BENCH_SEEDS, cell.seed) as u64
    };
    let method_offset: u64 = match cell.method {
        Method::Setfit => 0,
        Method::Lora => 6,
    };
    // Written so no subtraction can underflow at any (shots, seed, method) in the contract's
    // own axes: `seed_index <= 9 < 11 + method_offset`, and `method_offset <= 6 < 27`.
    vec![
        vec![30, 5, 5],
        vec![(11 + method_offset) - seed_index, (27 + seed_index) - method_offset, 2],
        vec![3, 4, 33 + shots_index],
    ]
}

/// The synthetic headline metric for one cell, DERIVED from that cell's confusion matrix.
///
/// Kept as a named function because the aggregate tests recompute the published mean, std and
/// interval from it by hand — the property EVAL-04 asks for is that a reader can re-derive the
/// number from the stored rows, and this is the test-side half of that.
fn synthetic_f_avg(spec: RunSpec, cell: CellKey) -> f64 {
    synthetic_quality(spec, cell).f_avg
}

/// The relative spelling of one cell's committed selection manifest.
///
/// Stated here as a LITERAL rather than through `selection_manifest_path`, deliberately: the
/// production helper is the thing under test, and a fixture that built its paths by calling it
/// would agree with it by construction and could never go red on a drift.
/// `bench_gate_layout_constants_agree_with_the_shipped_row_filename_grammar` cross-pins the two.
fn selection_manifest_rel(shots: u32, seed: u32) -> String {
    format!("selections/s{shots}-seed{seed}/selection-manifest.json")
}

/// That manifest's absolute path under a benchmark directory.
fn selection_manifest_at(root: &Path, shots: u32, seed: u32) -> PathBuf {
    root.join(selection_manifest_rel(shots, seed))
}

/// A minimal, VALID [`ExclusionRecord`].
///
/// Built through the type's own public `Deserialize` impl because its four fields are PRIVATE
/// and it exposes no constructor and no `Default` (`aprender-contrastive-data/src/dedup.rs:67`).
/// Widening another crate's API so a test could build one would change production surface for a
/// fixture's convenience; deserializing the empty record uses only what is already public.
fn synthetic_exclusions() -> ExclusionRecord {
    serde_json::from_str(
        "{\"excluded_train_ids\":[],\"groups\":[],\"reduced_pools\":{},\
         \"normalization_version\":\"nfc-trim-ws-v1\"}",
    )
    .expect("the exclusion record's public Deserialize accepts an empty record")
}

/// A real, SEALED selection manifest for one `(shots, seed)` cell.
///
/// SEALED THROUGH THE SHIPPED SERIALIZER: `semantic_hash` is
/// `sha256(payload.to_canonical_bytes())`, the same bytes and the same function
/// `SelectionManifest::from_bytes` verifies against. A fixture that computed the digest any
/// other way would be testing the fixture's arithmetic rather than the gate's.
///
/// The forty manifests must be DISTINCT — the committed set carries 40 distinct
/// `semantic_hash` values — so `ordered_examples` varies with `(shots, seed)`. A fixture whose
/// manifests collided would make the TRANSPLANT negatives pass for the wrong reason: a donor
/// manifest identical to the target's would satisfy the hash check by accident and the
/// cell-key check would never be the thing that refused.
fn synthetic_selection_manifest(shots: u32, seed: u32) -> SelectionManifest {
    let payload = SelectionPayload {
        schema_version: SELECTION_SCHEMA_VERSION,
        algorithm_version: 1,
        profile: "canonical".to_string(),
        dataset_fingerprint: "0".repeat(64),
        validation_fingerprint: "1".repeat(64),
        // The SAME label order the synthetic rows carry in `quality.ordered_labels`.
        label_names: vec!["none".to_string(), "against".to_string(), "favor".to_string()],
        normalization_version: "nfc-trim-ws-v1".to_string(),
        // THE WIDENING, in the fixture as in the gate: `root_seed` is `u64` and
        // `CellKey::seed` is `u32`.
        root_seed: u64::from(seed),
        shots_per_class: shots,
        ordered_examples: (0_usize..3)
            .map(|index| SelectedExampleRecord {
                id: format!("train:s{shots}-seed{seed}-{index}"),
                label: index,
                exact_hash: sha256_hex(format!("exact:{shots}:{seed}:{index}").as_bytes()),
                normalized_hash: sha256_hex(format!("norm:{shots}:{seed}:{index}").as_bytes()),
            })
            .collect(),
        exclusions: synthetic_exclusions(),
        access_ledger: vec![AccessRecord {
            role: "train".to_string(),
            profile: "canonical".to_string(),
            purpose: "select".to_string(),
            fingerprint_hex: "0".repeat(64),
        }],
        ledger_hash: "2".repeat(64),
    };
    let semantic_hash =
        sha256_hex(&payload.to_canonical_bytes().expect("the selection payload serializes"));
    SelectionManifest {
        semantic_hash,
        volatile: VolatileMetadata {
            created_at: String::new(),
            tool_version: "bench-gate-fixture".to_string(),
        },
        payload,
    }
}

/// The pairing key both methods of a `(shots, seed)` cell consume.
///
/// Since 05-16 this is the REAL sealed digest of the manifest the builder commits for that
/// cell, not a label. Before 05-16 it was the string `selection-manifest-s{shots}-seed{seed}`
/// and no file existed for it, because nothing on the gate path ever opened one — which is
/// precisely verifier gap 2.
fn synthetic_selection_hash(shots: u32, seed: u32) -> String {
    synthetic_selection_manifest(shots, seed).semantic_hash
}

/// The committed lock record's bytes for a SetFit cell.
///
/// The gate never PARSES this file — the contract's rule is `sha256(bytes(lock_record_path)) ==
/// lock.lock_hash` — so synthetic bytes are the honest fixture here: using a real
/// `SelectionLock` would test the lock module, which has its own suite, and would hide the fact
/// that this gate's guarantee is a digest over bytes rather than a re-parse.
fn synthetic_lock_bytes(cell: CellKey) -> Vec<u8> {
    format!(
        "{{\"schema\":\"selection-lock-v1\",\"cell\":\"{}\",\"chosen_artifact_sha256\":\"{}\"}}",
        cell.render(),
        "0".repeat(64)
    )
    .into_bytes()
}

/// One append-only ledger line for a LoRA cell.
fn synthetic_ledger_line(cell: CellKey) -> String {
    format!(
        "{{\"timestamp\":\"1970-01-01T00:00:00+00:00\",\"selection_manifest_hash\":\"{}\",\
         \"epochs_requested\":3,\"seed\":{},\"config_hash\":\"{}\"}}",
        synthetic_selection_hash(cell.shots, cell.seed),
        cell.seed,
        "1".repeat(64)
    )
}

/// A synthetic `QualityBlock`, every headline carrying its bits sibling.
///
/// EVERY recomputable field is a closed-form function of [`synthetic_confusion`]'s output,
/// computed by routing to the SAME shipped surfaces `assemble_quality_block` routes to
/// ([`MultiClassMetrics::from_predictions_with_min_classes`], [`f1_average_for_classes`],
/// [`matthews_corrcoef`]). No arithmetic is written here: a fixture that computed the metric a
/// second way and a gate that computed it a third would be two chances to disagree.
///
/// The two calibration diagnostics stay constants, and that is not an oversight — they are the
/// two quality fields NO committed file makes recomputable (they need per-row probability
/// vectors), so they are the two the closed-form cross-check deliberately does not reach.
fn synthetic_quality(spec: RunSpec, cell: CellKey) -> QualityBlock {
    let ordered_labels = synthetic_labels();
    let confusion_matrix = synthetic_confusion(spec, cell);
    // THROUGH THE PRODUCTION RECOMPUTATION ITSELF, so the fixture and the gate hold exactly one
    // definition of each number rather than two that have to be kept in step. The doctored rows
    // are what prove the gate; the control's job is only to be a run the gate must accept.
    let recomputed = quality_from_confusion_matrix(&confusion_matrix, &ordered_labels)
        .expect("the synthetic confusion matrix is a well-formed 3x3 over three labels");
    QualityBlock {
        f_avg: recomputed.f_avg,
        f_avg_bits: recomputed.f_avg.to_bits(),
        macro_f1: recomputed.macro_f1,
        macro_f1_bits: recomputed.macro_f1.to_bits(),
        per_class_precision: recomputed.per_class_precision,
        per_class_recall: recomputed.per_class_recall,
        per_class_f1: recomputed.per_class_f1,
        mcc: recomputed.mcc,
        mcc_bits: recomputed.mcc.to_bits(),
        confusion_matrix,
        n_test_rows: recomputed.n_rows,
        ordered_labels,
        ece_top_label_validation: 0.05,
        ece_top_label_validation_bits: 0.05_f64.to_bits(),
        brier_multiclass_validation: 0.30,
        brier_multiclass_validation_bits: 0.30_f64.to_bits(),
        calibration_split: CALIBRATION_SPLIT.to_string(),
    }
}

/// A synthetic `ResourceBlock`.
///
/// The two methods carry DIFFERENT train mechanisms on purpose — `vm_hwm` for the CPU SetFit
/// host and `sysinfo_sampled_10hz` for the GPU LoRA host — because D-09 puts them on two hosts
/// and the mixed-mechanism case is the one the renderer has to label. A fixture where both sides
/// happened to agree would let the comparability machinery go untested.
fn synthetic_resource(cell: CellKey) -> ResourceBlock {
    let seed_index = index_of(&BENCH_SEEDS, cell.seed);
    #[allow(clippy::cast_precision_loss)]
    let jitter = seed_index as f64;
    match cell.method {
        Method::Setfit => ResourceBlock {
            train_wall_ms: 1_000 + seed_index as u64,
            cold_latency_ms: 12.0 + jitter,
            cold_measured_in_child_process: true,
            warm_latency_ms_median: 4.0 + jitter,
            throughput_rows_per_sec: 200.0 + jitter,
            throughput_batch_size: 32,
            warmup_count: WARMUP_COUNT,
            train_peak_rss_bytes: 500_000_000 + seed_index as u64,
            train_peak_rss_mechanism: "vm_hwm".to_string(),
            inference_peak_rss_bytes: 200_000_000 + seed_index as u64,
            inference_peak_rss_mechanism: MECHANISM_CHILD_MAX_RSS_VM_HWM.to_string(),
            peak_rss_sample_interval_hz: None,
            artifact_bytes: 90_000_000,
            deployable_total_bytes: 90_000_000,
        },
        Method::Lora => ResourceBlock {
            train_wall_ms: 90_000 + seed_index as u64,
            cold_latency_ms: 400.0 + jitter,
            cold_measured_in_child_process: true,
            warm_latency_ms_median: 60.0 + jitter,
            throughput_rows_per_sec: 20.0 + jitter,
            throughput_batch_size: 32,
            warmup_count: WARMUP_COUNT,
            train_peak_rss_bytes: 40_000_000_000 + seed_index as u64,
            train_peak_rss_mechanism: "sysinfo_sampled_10hz".to_string(),
            inference_peak_rss_bytes: 19_000_000_000 + seed_index as u64,
            inference_peak_rss_mechanism: MECHANISM_CHILD_MAX_RSS_VM_HWM.to_string(),
            peak_rss_sample_interval_hz: Some(10),
            artifact_bytes: 40_000_000,
            deployable_total_bytes: 18_040_000_000,
        },
    }
}

/// The host a cell ran on. Two hosts by design (D-09), never pooled.
fn synthetic_host(method: Method) -> HostIdentity {
    match method {
        Method::Setfit => HostIdentity {
            hostname: "local-cpu".to_string(),
            os: "macos".to_string(),
            arch: "aarch64".to_string(),
        },
        Method::Lora => HostIdentity {
            hostname: "lambda-vector".to_string(),
            os: "linux".to_string(),
            arch: "x86_64".to_string(),
        },
    }
}

/// The full payload for one cell, with its evidence block already digest-consistent with the
/// lock or ledger bytes the builder is about to commit.
fn synthetic_payload(spec: RunSpec, cell: CellKey) -> BenchRowPayload {
    let evidence = match cell.method {
        Method::Setfit => MethodEvidence::Setfit(SetfitEvidence {
            evidence_table_hash: "a".repeat(64),
            apr_artifact_sha256: "b".repeat(64),
            lock: BenchLockRef {
                lock_hash: sha256_hex(&synthetic_lock_bytes(cell)),
                role: "written".to_string(),
                rule: SelectionRule::MaxMetricLowestIndexTieBreak.tag().to_string(),
                lock_record_path: format!(
                    "{LOCKS_DIR}/{}-s{}-seed{}.lock.json",
                    cell.method.tag(),
                    cell.shots,
                    cell.seed
                ),
            },
        }),
        Method::Lora => {
            let ledger = format!("{}\n", synthetic_ledger_line(cell));
            MethodEvidence::Lora(LoraEvidence {
                base_model_sha256: "c".repeat(64),
                base_model_bytes: 18_000_000_000,
                adapter_sha256: "d".repeat(64),
                epochs_requested: 3,
                epochs_completed: 3,
                early_stopping_disabled: true,
                val_split: 0.0,
                no_selection_attestation: true,
                candidate_ledger_sha256: sha256_hex(ledger.as_bytes()),
                candidates_trained: 1,
                candidate_ledger_path: format!(
                    "{LEDGER_DIR}/{}-s{}-seed{}.jsonl",
                    cell.method.tag(),
                    cell.shots,
                    cell.seed
                ),
            })
        }
    };

    BenchRowPayload {
        schema_version: BENCH_ROW_SCHEMA_VERSION,
        contract_id: CLAIMS_CONTRACT_ID.to_string(),
        method: cell.method,
        shots: cell.shots,
        seed: cell.seed,
        dataset_revision: "4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66".to_string(),
        dataset_fingerprint: "e".repeat(64),
        model_revision: "f".repeat(40),
        selection_manifest_hash: synthetic_selection_hash(cell.shots, cell.seed),
        backend_identity: match cell.method {
            Method::Setfit => "cpu:trueno:simd".to_string(),
            Method::Lora => "gpu:cuda:cublas".to_string(),
        },
        host: synthetic_host(cell.method),
        quality: synthetic_quality(spec, cell),
        resource: synthetic_resource(cell),
        evidence,
    }
}

/// Which scope a synthetic run on disk was built for, READ FROM THE DISK.
///
/// Detected rather than threaded through thirty call sites: a second method's row file can
/// only exist if the directory was built for the deferred scope, so the disk already carries
/// the answer and a parameter would just be a second, drift-prone statement of it.
fn scope_of(root: &Path) -> ExpectationScope {
    let deferred_marker =
        root.join(ROWS_DIR).join(row_file_name(CellKey::new(Method::Lora, 8, 13)));
    if deferred_marker.exists() {
        ExpectationScope::DeferredTwoMethod
    } else {
        ExpectationScope::Active
    }
}

/// Write a complete, VALID 40-cell ACTIVE-scope benchmark directory.
///
/// Returns the temp dir; the manifest is derived from what is on disk by [`manifest_for`], so a
/// mutation helper can re-derive it after editing a row rather than fighting `record`'s
/// deliberate refusal to overwrite a differing digest.
fn write_valid_run(spec: RunSpec) -> TempDir {
    write_valid_run_scoped(spec, ExpectationScope::Active)
}

/// Write a complete, VALID benchmark directory for the DEFERRED two-method scope (80 cells).
///
/// The two doctored shapes that exist ONLY in a two-method design — an unpaired selection hash
/// and a forged LoRA provenance — need rows production code cannot construct, which is exactly
/// why the deferred scope is retained rather than deleted.
fn write_valid_run_deferred(spec: RunSpec) -> TempDir {
    write_valid_run_scoped(spec, ExpectationScope::DeferredTwoMethod)
}

fn write_valid_run_scoped(spec: RunSpec, scope: ExpectationScope) -> TempDir {
    let dir = TempDir::new().expect("a temp dir");
    let root = dir.path();
    fs::create_dir_all(root.join(ROWS_DIR)).expect("rows dir");
    fs::create_dir_all(root.join(LOCKS_DIR)).expect("locks dir");
    fs::create_dir_all(root.join(LEDGER_DIR)).expect("ledger dir");

    for cell in RunManifest::expectation_for(scope) {
        // THE SELECTION MANIFEST, one per `(shots, seed)` and SHARED by the pair.
        //
        // Written per cell rather than per pair, which is idempotent: both methods of a
        // `(shots, seed)` resolve the SAME path and the same bytes, because
        // `selection_manifest_path` is built from shots and seed alone. That sharing IS the
        // pairing design — one selection, two methods — and it is what keeps the deferred-scope
        // `unpaired_selection` negative constructible after this change.
        let manifest_path = selection_manifest_at(root, cell.shots, cell.seed);
        fs::create_dir_all(manifest_path.parent().expect("the manifest has a parent"))
            .expect("selection dir");
        fs::write(
            &manifest_path,
            synthetic_selection_manifest(cell.shots, cell.seed)
                .to_file_bytes()
                .expect("the selection manifest serializes"),
        )
        .expect("selection manifest write");

        match cell.method {
            Method::Setfit => {
                let relative = format!(
                    "{LOCKS_DIR}/{}-s{}-seed{}.lock.json",
                    cell.method.tag(),
                    cell.shots,
                    cell.seed
                );
                fs::write(root.join(relative), synthetic_lock_bytes(cell)).expect("lock write");
            }
            Method::Lora => {
                let relative = format!(
                    "{LEDGER_DIR}/{}-s{}-seed{}.jsonl",
                    cell.method.tag(),
                    cell.shots,
                    cell.seed
                );
                fs::write(root.join(relative), format!("{}\n", synthetic_ledger_line(cell)))
                    .expect("ledger write");
            }
        }
        let row = BenchRow::new(synthetic_payload(spec, cell));
        fs::write(
            root.join(ROWS_DIR).join(row_file_name(cell)),
            row.to_file_bytes().expect("row serializes"),
        )
        .expect("row write");
    }
    dir
}

/// Declare a manifest and record whatever rows are actually on disk.
///
/// A missing row file leaves its cell `pending`, which is exactly how selective omission looks
/// to the gate — so negative 1 is produced by DELETING a file rather than by hand-editing a
/// status field.
fn manifest_for(root: &Path) -> RunManifest {
    let scope = scope_of(root);
    let mut manifest = RunManifest::declare_for(scope);
    for cell in RunManifest::expectation_for(scope) {
        let path = root.join(ROWS_DIR).join(row_file_name(cell));
        let Ok(bytes) = fs::read(&path) else {
            continue;
        };
        let Ok(row) = BenchRow::from_bytes(&bytes) else {
            // A doctored row that no longer parses cannot have its digest recorded honestly.
            // Recording the file's own digest keeps the manifest a truthful statement about
            // what is on disk and lets the ROW-level refusal be the thing under test.
            manifest.record(cell, &sha256_hex(&bytes)).expect("recording a fresh cell");
            continue;
        };
        manifest.record(cell, &row.semantic_hash).expect("recording a fresh cell");
    }
    manifest
}

/// Read a row file as untyped JSON, so a REQUIRED field can be removed (which the typed struct
/// cannot express).
fn read_row_value(root: &Path, cell: CellKey) -> serde_json::Value {
    let bytes = fs::read(root.join(ROWS_DIR).join(row_file_name(cell))).expect("row read");
    serde_json::from_slice(&bytes).expect("row is JSON")
}

/// Write an untyped row value back, pretty, with the trailing newline the writer uses.
fn write_row_value(root: &Path, cell: CellKey, value: &serde_json::Value) {
    let mut bytes = serde_json::to_vec_pretty(value).expect("row serializes");
    bytes.push(b'\n');
    fs::write(root.join(ROWS_DIR).join(row_file_name(cell)), bytes).expect("row write");
}

/// Mutate a row's typed payload and RESEAL it, so the envelope digest stays honest and the
/// defect under test is the one the mutation introduced rather than a digest mismatch.
fn reseal_row(root: &Path, cell: CellKey, mutate: impl FnOnce(&mut BenchRowPayload)) {
    let path = root.join(ROWS_DIR).join(row_file_name(cell));
    let bytes = fs::read(&path).expect("row read");
    let row = BenchRow::from_bytes(&bytes).expect("the valid fixture parses");
    let mut payload = row.payload;
    mutate(&mut payload);
    let resealed = BenchRow::new(payload);
    fs::write(&path, resealed.to_file_bytes().expect("row serializes")).expect("row write");
}

/// Verify, and require a refusal — reported in one line.
///
/// `expect_err` would print the `Ok` value's `Debug`, and a `VerifiedRunSet`'s `Debug` is eighty
/// rows of nested structs. A diagnostic nobody can read is a diagnostic that does not exist, so
/// the accepted case reports the COUNT and the doctored shape's name and stops there.
fn refuse(manifest: &RunManifest, root: &Path, doctored: &str) -> BenchGateError {
    match verify_scoped(manifest, root) {
        Ok(set) => panic!(
            "the doctored run `{doctored}` was ACCEPTED ({} rows verified). The gate did not \
             refuse it, so this negative is proving nothing",
            set.len()
        ),
        Err(error) => error,
    }
}

/// Verify against whichever scope the directory on disk was built for.
///
/// The ACTIVE-scope cases go through the PUBLIC two-argument `verify_run`, so what they prove
/// is proven about the shipped door and not about a test-only entry point.
fn verify_scoped(manifest: &RunManifest, root: &Path) -> Result<VerifiedRunSet, BenchGateError> {
    match scope_of(root) {
        ExpectationScope::Active => verify_run(manifest, root),
        scope => verify_run_scoped(manifest, root, scope),
    }
}

/// The cell every negative doctors, so the failures are directly comparable.
const TARGET_SETFIT: CellKey = CellKey::new(Method::Setfit, 16, 29);
/// Its LoRA partner.
const TARGET_LORA: CellKey = CellKey::new(Method::Lora, 16, 29);

// ===========================================================================================
// The positive control
// ===========================================================================================

#[test]
fn bench_gate_accepts_a_complete_valid_run() {
    let dir = write_valid_run(RunSpec::default());
    let manifest = manifest_for(dir.path());
    let verified = verify_run(&manifest, dir.path()).expect("the valid synthetic run verifies");
    assert_eq!(
        verified.len(),
        EXPECTED_CELLS,
        "a verified set is the whole contracted matrix or it is nothing"
    );
    assert!(!verified.is_empty());
    assert!(verified.row(TARGET_SETFIT).is_some());
}

#[test]
fn bench_gate_verified_rows_are_in_the_contract_order() {
    let dir = write_valid_run(RunSpec::default());
    let manifest = manifest_for(dir.path());
    let verified = verify_run(&manifest, dir.path()).expect("verifies");
    let keys: Vec<CellKey> = verified.rows().iter().map(|(cell, _)| *cell).collect();
    assert_eq!(keys, RunManifest::expectation());
}

// ===========================================================================================
// DOCTORED NEGATIVE 1 — a selectively omitted cell
// ===========================================================================================

#[test]
fn bench_gate_refuses_a_missing_cell_naming_it() {
    let dir = write_valid_run(RunSpec::default());
    fs::remove_file(dir.path().join(ROWS_DIR).join(row_file_name(TARGET_SETFIT)))
        .expect("remove the row");
    let manifest = manifest_for(dir.path());

    let error = refuse(&manifest, dir.path(), "an omitted cell is a refusal");
    assert_eq!(error.variant_tag(), "incomplete_cell");
    assert_eq!(error.cell(), Some(TARGET_SETFIT.render().as_str()));
    assert!(
        error.to_string().contains(&TARGET_SETFIT.render()),
        "the refusal must name the cell: {error}"
    );
}

// ===========================================================================================
// DOCTORED NEGATIVE 2 — a trimmed row (a required block removed)
// ===========================================================================================

#[test]
fn bench_gate_refuses_a_trimmed_row_whose_evidence_block_was_removed() {
    let dir = write_valid_run(RunSpec::default());
    let mut value = read_row_value(dir.path(), TARGET_SETFIT);
    // Remove the `lock` sub-block. A `setfit` row without it can still be READ as JSON and
    // still names a plausible cell — which is exactly why the refusal has to be structural.
    value
        .get_mut("payload")
        .and_then(|p| p.get_mut("evidence"))
        .and_then(|e| e.get_mut("setfit"))
        .and_then(serde_json::Value::as_object_mut)
        .expect("the fixture carries a setfit evidence block")
        .remove("lock");
    write_row_value(dir.path(), TARGET_SETFIT, &value);
    let manifest = manifest_for(dir.path());

    let error = refuse(&manifest, dir.path(), "a trimmed row is a refusal");
    assert_eq!(error.variant_tag(), "row_schema_refused");
    assert_eq!(error.cell(), Some(TARGET_SETFIT.render().as_str()));
}

// ===========================================================================================
// DOCTORED NEGATIVE 3 — a mismatched sampled-ID hash in one pair
// ===========================================================================================

#[test]
fn bench_gate_refuses_a_pair_measured_on_different_selection_manifests() {
    let dir = write_valid_run_deferred(RunSpec::default());
    // Reseal, so the row is internally perfect. The ONLY defect is that the two halves of one
    // pair consumed different sampled IDs — which is PF-007's incomparable comparison, and it
    // is invisible to every per-row check.
    reseal_row(dir.path(), TARGET_LORA, |payload| {
        payload.selection_manifest_hash = "a-different-draw-entirely".to_string();
    });
    let manifest = manifest_for(dir.path());

    let error = refuse(&manifest, dir.path(), "an unpaired pair is a refusal");
    assert_eq!(error.variant_tag(), "unpaired_selection");
    let rendered = error.to_string();
    assert!(rendered.contains("shots 16"), "{rendered}");
    assert!(rendered.contains("seed 29"), "{rendered}");
    assert!(rendered.contains("a-different-draw-entirely"), "{rendered}");
}

// ===========================================================================================
// DOCTORED NEGATIVE 4 — bit-flipped row bytes
// ===========================================================================================

#[test]
fn bench_gate_refuses_a_row_whose_payload_bytes_were_edited() {
    let dir = write_valid_run(RunSpec::default());
    let mut value = read_row_value(dir.path(), TARGET_SETFIT);
    // Edit a payload field and DO NOT reseal: the envelope still claims the old digest.
    *value
        .get_mut("payload")
        .and_then(|p| p.get_mut("dataset_revision"))
        .expect("the fixture carries a dataset revision") =
        serde_json::Value::String("0000000000000000000000000000000000000000".to_string());
    write_row_value(dir.path(), TARGET_SETFIT, &value);
    let manifest = manifest_for(dir.path());

    let error = refuse(&manifest, dir.path(), "edited bytes are a refusal");
    assert_eq!(error.variant_tag(), "row_digest_mismatch");
    assert_eq!(error.cell(), Some(TARGET_SETFIT.render().as_str()));
}

// ===========================================================================================
// DOCTORED NEGATIVE 5 — post-test selection, on both sides
// ===========================================================================================

#[test]
fn bench_gate_refuses_a_setfit_row_whose_lock_rule_is_not_the_committed_one() {
    let dir = write_valid_run(RunSpec::default());
    reseal_row(dir.path(), TARGET_SETFIT, |payload| {
        if let MethodEvidence::Setfit(evidence) = &mut payload.evidence {
            evidence.lock.rule = "best_observed_on_test".to_string();
        }
    });
    let manifest = manifest_for(dir.path());

    let error = refuse(&manifest, dir.path(), "an unknown rule is a refusal");
    assert_eq!(error.variant_tag(), "post_test_selection");
    assert!(
        error.to_string().contains("lock.rule"),
        "the refusal must name the failing conjunct: {error}"
    );
}

#[test]
fn bench_gate_refuses_a_lora_row_that_completed_fewer_epochs_than_it_requested() {
    let dir = write_valid_run_deferred(RunSpec::default());
    reseal_row(dir.path(), TARGET_LORA, |payload| {
        if let MethodEvidence::Lora(evidence) = &mut payload.evidence {
            evidence.epochs_completed = 1;
        }
    });
    let manifest = manifest_for(dir.path());

    let error = refuse(&manifest, dir.path(), "a short run is a refusal");
    assert_eq!(error.variant_tag(), "post_test_selection");
    let rendered = error.to_string();
    assert!(rendered.contains("epochs_completed"), "{rendered}");
    assert!(rendered.contains(&TARGET_LORA.render()), "{rendered}");
}

#[test]
fn bench_gate_refuses_every_conjunct_of_the_lora_attestation_separately() {
    // Each conjunct closes a DIFFERENT route to a post-hoc choice, and any one alone is
    // satisfiable while the claim is false — so each is exercised rather than one standing in
    // for the set.
    let mutations: Vec<(&str, fn(&mut LoraEvidence))> = vec![
        ("early_stopping_disabled", |e| {
            e.early_stopping_disabled = false;
        }),
        ("val_split", |e| e.val_split = 0.1),
        ("no_selection_attestation", |e| {
            e.no_selection_attestation = false;
        }),
    ];
    for (conjunct, mutate) in mutations {
        let dir = write_valid_run_deferred(RunSpec::default());
        reseal_row(dir.path(), TARGET_LORA, |payload| {
            if let MethodEvidence::Lora(evidence) = &mut payload.evidence {
                mutate(evidence);
            }
        });
        let manifest = manifest_for(dir.path());
        let Err(error) = verify_scoped(&manifest, dir.path()) else {
            panic!("the `{conjunct}` conjunct must be refused, and it was not");
        };
        assert_eq!(
            error.variant_tag(),
            "post_test_selection",
            "`{conjunct}` must be refused as post-test selection, got: {error}"
        );
        assert!(
            error.to_string().contains(conjunct),
            "the refusal must name the failing conjunct `{conjunct}`: {error}"
        );
    }
}

// ===========================================================================================
// DOCTORED NEGATIVE 6 — FORGED PROVENANCE, two sub-cases
// ===========================================================================================

#[test]
fn bench_gate_refuses_a_ledger_carrying_a_second_candidate_the_row_does_not_declare() {
    let dir = write_valid_run_deferred(RunSpec::default());
    let ledger_path = dir.path().join(format!(
        "{LEDGER_DIR}/{}-s{}-seed{}.jsonl",
        TARGET_LORA.method.tag(),
        TARGET_LORA.shots,
        TARGET_LORA.seed
    ));
    // TWO lines, and the row's `candidate_ledger_sha256` is UPDATED to match the new bytes —
    // so the digest check passes and only COUNTING catches it. That is the sharp form of this
    // forgery: a producer who edits both the file and the field it claims.
    let forged =
        format!("{}\n{}\n", synthetic_ledger_line(TARGET_LORA), synthetic_ledger_line(TARGET_LORA));
    fs::write(&ledger_path, forged.as_bytes()).expect("ledger write");
    reseal_row(dir.path(), TARGET_LORA, |payload| {
        if let MethodEvidence::Lora(evidence) = &mut payload.evidence {
            evidence.candidate_ledger_sha256 = sha256_hex(forged.as_bytes());
        }
    });
    let manifest = manifest_for(dir.path());

    let error = refuse(&manifest, dir.path(), "a two-line ledger is a refusal");
    assert_eq!(error.variant_tag(), "provenance_mismatch");
    let rendered = error.to_string();
    assert!(rendered.contains("candidates_trained = 1"), "{rendered}");
    assert!(rendered.contains("2 ledger line(s)"), "{rendered}");
    assert!(
        rendered.contains(&ledger_path.display().to_string()),
        "the refusal must name the FILE whose bytes disagreed: {rendered}"
    );
}

#[test]
fn bench_gate_refuses_a_setfit_row_whose_committed_lock_file_was_edited() {
    let dir = write_valid_run(RunSpec::default());
    let lock_path = dir.path().join(format!(
        "{LOCKS_DIR}/{}-s{}-seed{}.lock.json",
        TARGET_SETFIT.method.tag(),
        TARGET_SETFIT.shots,
        TARGET_SETFIT.seed
    ));
    let mut bytes = fs::read(&lock_path).expect("lock read");
    // One byte, inside the committed evidence. The row's `lock_hash` is untouched, so the row
    // and the file now disagree — and the row alone still looks perfect.
    let last = bytes.len() - 2;
    bytes[last] = b'9';
    fs::write(&lock_path, &bytes).expect("lock write");
    let manifest = manifest_for(dir.path());

    let error = refuse(&manifest, dir.path(), "an edited lock is a refusal");
    assert_eq!(error.variant_tag(), "provenance_mismatch");
    let rendered = error.to_string();
    assert!(rendered.contains("lock_hash"), "{rendered}");
    assert!(
        rendered.contains(&lock_path.display().to_string()),
        "the refusal must name the FILE whose bytes disagreed: {rendered}"
    );
}

#[test]
fn bench_gate_refuses_a_ledger_transplanted_from_another_cell() {
    let dir = write_valid_run_deferred(RunSpec::default());
    let ledger_path = dir.path().join(format!(
        "{LEDGER_DIR}/{}-s{}-seed{}.jsonl",
        TARGET_LORA.method.tag(),
        TARGET_LORA.shots,
        TARGET_LORA.seed
    ));
    let other = CellKey::new(Method::Lora, 16, 31);
    let transplanted = format!("{}\n", synthetic_ledger_line(other));
    fs::write(&ledger_path, transplanted.as_bytes()).expect("ledger write");
    reseal_row(dir.path(), TARGET_LORA, |payload| {
        if let MethodEvidence::Lora(evidence) = &mut payload.evidence {
            evidence.candidate_ledger_sha256 = sha256_hex(transplanted.as_bytes());
        }
    });
    let manifest = manifest_for(dir.path());

    let error = refuse(&manifest, dir.path(), "a transplanted ledger is a refusal");
    assert_eq!(error.variant_tag(), "provenance_mismatch");
    assert!(error.to_string().contains("different selection manifest"), "{error}");
}

// ===========================================================================================
// GAP 1 — THE ACTIVE 40-CELL SCOPE ESCAPE SWEEP (plan 05-15 task 2)
//
// Verifier gap 1's second `missing:` bullet names THREE bounds — an absolute path, a `..`
// traversal, and a symlink out of `bench_dir` — and requires a RED-turning negative for each,
// RE-MUTATED at the ACTIVE scope. The helper-level case table further down drives
// `resolve_committed_evidence_path` DIRECTLY, which is a different scope: it holds the
// resolver's two stages, not the wiring that makes the resolver reachable from a row at all.
// CLAUDE.md Verification Discipline rule 4 — extending a guard's SCOPE requires re-mutating in
// the new scope — is why both exist and why neither may be deleted for duplicating the other.
// ===========================================================================================

/// One escaping path SHAPE, as something the materializer builds rather than a hardcoded
/// literal — the two temp directories' real names are only known at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EscapeShape {
    /// An ABSOLUTE declared path. `Path::join` DISCARDS the base when its argument is
    /// absolute, so the benchmark directory stops being part of the resolution at all.
    Absolute,
    /// A RELATIVE declared path that climbs out with `..`. `Path::join` never resolves `..`;
    /// the kernel does, at open time, after the gate has stopped looking.
    ParentTraversal,
    /// A relative path INSIDE the benchmark directory whose LAST component is a symlink to a
    /// file outside it. No syntactic check can see this one — only canonicalization can.
    LastComponentSymlink,
}

/// The three bounds gap 1 names, as a TABLE. A fourth bound is added as a row here, never as
/// another test function (CLAUDE.md Verification Discipline rule 7).
const ACTIVE_SCOPE_ESCAPE_SHAPES: [(&str, EscapeShape, &str); 3] = [
    (
        "absolute",
        EscapeShape::Absolute,
        "an absolute component makes `Path::join` discard the benchmark directory entirely",
    ),
    (
        "parent_traversal",
        EscapeShape::ParentTraversal,
        "`Path::join` never resolves `..`, so the climb happens at open time, unseen",
    ),
    (
        "last_component_symlink",
        EscapeShape::LastComponentSymlink,
        "the declared string is entirely well-formed; only the canonical form leaves the tree",
    ),
];

/// Build the declared string for one shape, creating whatever filesystem object it needs.
///
/// `outside_file` already holds the cell's OWN attested lock bytes, and the in-tree lock record
/// has already been deleted by the caller. That pairing is load-bearing: it reproduces
/// spot-check E exactly, so the PRE-FIX gate hashes the escape target, MATCHES, and returns
/// `Ok`. Aim any of these rows at `/etc/passwd` instead and the pre-fix run refuses with
/// `provenance_mismatch` — still red, but red for the wrong reason, proving the escape was
/// DETECTED rather than that it SUCCEEDED.
fn materialize(shape: EscapeShape, bench_dir: &Path, outside_file: &Path) -> String {
    match shape {
        EscapeShape::Absolute => outside_file.display().to_string(),
        EscapeShape::ParentTraversal => {
            // Built from the two temp dirs' REAL names at runtime. They are siblings under
            // `$TMPDIR`, so one `..` from the bench dir lands beside the scratch dir.
            let outside_dir = outside_file.parent().expect("the scratch file has a parent");
            format!(
                "../{}/{}",
                outside_dir.file_name().expect("the scratch dir has a name").to_string_lossy(),
                outside_file.file_name().expect("the scratch file has a name").to_string_lossy(),
            )
        }
        EscapeShape::LastComponentSymlink => {
            let relative = format!("{LOCKS_DIR}/escape-via-symlink.lock.json");
            let link = bench_dir.join(&relative);
            // `#[cfg(unix)]` gates the MATERIALIZER, never the table: a table that silently
            // shrinks on some host is a coverage loss no test-count floor can see.
            #[cfg(unix)]
            std::os::unix::fs::symlink(outside_file, &link).expect("symlink into the scratch dir");
            #[cfg(not(unix))]
            let _ = &link;
            relative
        }
    }
}

#[test]
fn bench_gate_refuses_every_escaping_lock_path_shape_at_the_active_scope() {
    let mut observations: Vec<(&str, String)> = Vec::new();

    for (label, shape, why) in ACTIVE_SCOPE_ESCAPE_SHAPES {
        let dir = write_valid_run(RunSpec::default());
        let outside = TempDir::new().expect("a scratch dir OUTSIDE the bench dir");
        let outside_file = outside.path().join("anywhere.json");

        // 1. The escape target holds the very bytes the row attests.
        fs::write(&outside_file, synthetic_lock_bytes(TARGET_SETFIT)).expect("scratch lock write");
        // 2. The IN-TREE lock record is deleted, so nothing legitimate can satisfy the row.
        fs::remove_file(dir.path().join(format!(
            "{LOCKS_DIR}/{}-s{}-seed{}.lock.json",
            TARGET_SETFIT.method.tag(),
            TARGET_SETFIT.shots,
            TARGET_SETFIT.seed
        )))
        .expect("delete the in-tree lock record");
        // 3. The row points at the materialized escape.
        let declared = materialize(shape, dir.path(), &outside_file);
        reseal_row(dir.path(), TARGET_SETFIT, |payload| {
            if let MethodEvidence::Setfit(evidence) = &mut payload.evidence {
                evidence.lock.lock_record_path = declared.clone();
            }
        });
        // 4. The manifest re-records the resealed row, so the run is otherwise self-consistent.
        let manifest = manifest_for(dir.path());

        let (observed, rendered) = match verify_run(&manifest, dir.path()) {
            Ok(set) => (format!("Ok({} rows verified)", set.len()), String::new()),
            Err(error) => (error.variant_tag().to_string(), error.to_string()),
        };
        // PRINTED PER SHAPE. The sweep is one test function but three bounds, and a single
        // combined verdict would hide a row that was never red.
        println!(
            "[bench_gate] ACTIVE_SCOPE_ESCAPE shape={label} declared={declared} \
             observed={observed} ({why})"
        );
        if observed == "evidence_path_escape" {
            assert!(
                rendered.contains(&declared),
                "the `{label}` refusal must quote the declared string verbatim: {rendered}"
            );
        }
        observations.push((label, observed));
    }

    let failures: Vec<&(&str, String)> =
        observations.iter().filter(|(_, observed)| observed != "evidence_path_escape").collect();
    assert!(
        failures.is_empty(),
        "every bound gap 1 names must be refused as `evidence_path_escape` through verify_run \
         at the ACTIVE 40-cell scope; these were not: {failures:?}"
    );
}

// ===========================================================================================
// WR-06 — a missing LOCK RECORD names its own kind, not the row's (plan 05-15 task 2)
// ===========================================================================================

#[test]
fn bench_gate_refuses_a_missing_lock_record_as_its_own_kind_not_as_a_missing_row() {
    let dir = write_valid_run(RunSpec::default());
    let lock_path = dir.path().join(format!(
        "{LOCKS_DIR}/{}-s{}-seed{}.lock.json",
        TARGET_SETFIT.method.tag(),
        TARGET_SETFIT.shots,
        TARGET_SETFIT.seed
    ));
    // The row is untouched and still names the committed spelling. ONLY the lock file is gone,
    // which is the honest shape of this defect: the row is fine, the evidence is not.
    fs::remove_file(&lock_path).expect("delete the in-tree lock record");
    let manifest = manifest_for(dir.path());

    let error = refuse(&manifest, dir.path(), "a missing lock record is a refusal");
    assert_eq!(error.variant_tag(), "evidence_file_missing");
    assert_eq!(error.cell(), Some(TARGET_SETFIT.render().as_str()));
    let rendered = error.to_string();
    assert!(rendered.contains("lock record"), "the refusal must name the KIND: {rendered}");
    assert!(
        rendered.contains(&lock_path.display().to_string()),
        "the refusal must name the FILE: {rendered}"
    );
    assert!(
        !rendered.contains("restore the row file"),
        "a missing lock record must not carry the ROW remedy — that is WR-06: {rendered}"
    );

    // And the ROW kind is UNCHANGED, message and all. This is a typing correction, not a
    // behaviour change for the row path: spot-check A's output must still read as it did.
    let dir = write_valid_run(RunSpec::default());
    let row_path = dir.path().join(ROWS_DIR).join(row_file_name(TARGET_SETFIT));
    let error = read_evidence(TARGET_SETFIT, EvidenceKind::Row, &row_path.with_extension("gone"))
        .expect_err("a missing row file is a refusal");
    assert_eq!(error.variant_tag(), "row_file_missing");
    assert!(
        error.to_string().contains("restore the row file"),
        "the ROW remedy is kept VERBATIM: {error}"
    );
}

// ===========================================================================================
// THE THIRD ENUMERATED FIELD — a foreign contract id, on both sides (plan 05-15 task 2)
// ===========================================================================================

#[test]
fn bench_gate_refuses_a_row_declaring_a_foreign_contract_id() {
    // Found by `05-15-gate-input-surface.md`, not by a probe: nothing anywhere compared a
    // row's declared contract against the constant, while `aggregate` stamped the published
    // payload with that constant regardless.
    let dir = write_valid_run(RunSpec::default());
    reseal_row(dir.path(), TARGET_SETFIT, |payload| {
        payload.contract_id = "setfit-benchmark-claims-v99".to_string();
    });
    let manifest = manifest_for(dir.path());

    let error = refuse(&manifest, dir.path(), "a foreign contract id is a refusal");
    assert_eq!(error.variant_tag(), "row_schema_refused");
    assert_eq!(error.cell(), Some(TARGET_SETFIT.render().as_str()));
    let rendered = error.to_string();
    assert!(rendered.contains("setfit-benchmark-claims-v99"), "names the DECLARED id: {rendered}");
    assert!(rendered.contains(CLAIMS_CONTRACT_ID), "names the EXPECTED id: {rendered}");
}

#[test]
fn bench_gate_refuses_a_manifest_declaring_a_foreign_contract_id() {
    let dir = write_valid_run(RunSpec::default());
    let mut manifest = manifest_for(dir.path());
    manifest.payload.contract_id = "setfit-benchmark-claims-v99".to_string();
    // Resealed by hand so the ENVELOPE digest does not fire first and mask the comparison.
    manifest.semantic_hash =
        sha256_hex(&manifest.payload.to_canonical_bytes().expect("payload serializes"));

    let error = refuse(&manifest, dir.path(), "a foreign manifest contract id is a refusal");
    assert_eq!(error.variant_tag(), "row_schema_refused");
    let rendered = error.to_string();
    assert!(rendered.contains("setfit-benchmark-claims-v99"), "names the DECLARED id: {rendered}");
    assert!(rendered.contains(CLAIMS_CONTRACT_ID), "names the EXPECTED id: {rendered}");
}

#[test]
fn bench_gate_refuses_each_contract_pinned_constant_a_row_may_not_choose() {
    // The other three (iii) entries the enumeration found: fields the contract pins to one
    // value each, checked only at EMISSION time and never when the gate READ a committed row.
    // Swept as a table rather than as three functions (CLAUDE.md Verification Discipline
    // rule 7). `throughput_batch_size` is deliberately absent — the contract pins no value for
    // it, so there is nothing contract-derived to compare against.
    let mutations: Vec<(&str, fn(&mut BenchRowPayload))> = vec![
        ("calibration_split", |p| p.quality.calibration_split = "test".to_string()),
        ("warmup_count", |p| p.resource.warmup_count = WARMUP_COUNT + 7),
        ("cold_measured_in_child_process", |p| {
            p.resource.cold_measured_in_child_process = false;
        }),
    ];
    for (field, mutate) in mutations {
        let dir = write_valid_run(RunSpec::default());
        reseal_row(dir.path(), TARGET_SETFIT, mutate);
        let manifest = manifest_for(dir.path());

        let error = refuse(&manifest, dir.path(), field);
        assert_eq!(
            error.variant_tag(),
            "row_schema_refused",
            "`{field}` must be refused as a schema-domain violation, got: {error}"
        );
        assert!(
            error.to_string().contains(field),
            "the refusal must name the offending field `{field}`: {error}"
        );
    }
}

// ===========================================================================================
// THE PATH-SHAPE CASE TABLE (plan 05-15 task 3)
//
// Scope: this table drives `resolve_committed_evidence_path` DIRECTLY, over BOTH evidence
// kinds. The ACTIVE-scope sweep above drives three of these same shapes through `verify_run`
// over 40 cells. NEITHER PROOF TRANSFERS TO THE OTHER (CLAUDE.md Verification Discipline rule
// 4): this one holds the resolver's two stages across every refusal shape, three acceptance
// rows and two behaviour-preserving rows; that one holds the wiring that makes the resolver
// reachable from a row at all. Deleting a row here because "the sweep covers it" is exactly
// the scope transfer rule 4 forbids.
//
// Ship the TABLE and re-run it, rather than re-reading the pattern (rule 7). A fourth shape is
// a new row, never a new test function.
// ===========================================================================================

/// How a case's declared string is built for a given [`EvidenceKind`].
///
/// A shape rather than a literal, because five of the thirteen cases need a filesystem object
/// created first and three need this kind's own directory spelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Declared {
    /// Used verbatim. For shapes stage 1 refuses without touching the filesystem.
    Literal(&'static str),
    /// The real committed file for this kind, under this kind's own directory.
    CommittedForKind,
    /// The same file, prefixed `./` — [`std::path::Component::CurDir`], which names the SAME
    /// file and must be ACCEPTED. This row goes red if the syntactic stage over-refuses.
    CommittedForKindCurDir,
    /// A well-formed relative path under this kind's directory naming a file that is absent.
    AbsentUnderKindDir,
    /// A well-formed relative path naming this kind's DIRECTORY itself.
    KindDirItself,
    /// A path under this kind's directory whose LAST component is a symlink out of the tree.
    LastComponentSymlink,
    /// A path whose FIRST component is a symlinked DIRECTORY pointing out of the tree.
    FirstComponentSymlinkDir,
    /// A path resolving, via a symlink, into a SIBLING directory whose name has the bench
    /// directory's name as a STRING PREFIX. This row goes red if containment is implemented
    /// with `str::starts_with` instead of `Path::starts_with`, which is component-wise.
    PrefixSiblingSymlink,
}

/// What the resolver-then-read pair must do with a case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Expect {
    /// Refused with this variant tag.
    Refused(&'static str),
    /// Resolved AND read. Without these rows the table would prove only that something is
    /// refused, which every broken implementation also achieves.
    Accepted,
}

/// One row of the table.
struct PathCase {
    label: &'static str,
    declared: Declared,
    expect: Expect,
    /// A substring the rendered refusal must carry. Empty means no additional check.
    detail_contains: &'static str,
    why: &'static str,
}

const EVIDENCE_PATH_CASES: [PathCase; 13] = [
    // ---- MUST NOT MATCH: eight refusal shapes ------------------------------------------
    PathCase {
        label: "absolute",
        declared: Declared::Literal("/tmp/outside/anywhere.json"),
        expect: Expect::Refused("evidence_path_escape"),
        detail_contains: "ABSOLUTE",
        why: "`Path::join` DISCARDS its base when the argument is absolute",
    },
    PathCase {
        label: "leading_parent_traversal",
        declared: Declared::Literal("../../../etc/passwd"),
        expect: Expect::Refused("evidence_path_escape"),
        detail_contains: "`..`",
        why: "`Path::join` never resolves `..`; the kernel does, at open time",
    },
    PathCase {
        label: "interior_parent_traversal",
        declared: Declared::Literal("locks/../../outside.json"),
        expect: Expect::Refused("evidence_path_escape"),
        detail_contains: "`..`",
        why: "a `..` that is not the FIRST component climbs out just as effectively",
    },
    PathCase {
        label: "empty",
        declared: Declared::Literal(""),
        expect: Expect::Refused("evidence_path_escape"),
        detail_contains: "empty",
        why: "an empty join yields the benchmark directory itself, which would be READ as a \
              directory and reported as an I/O accident rather than as the nonsense it is",
    },
    PathCase {
        label: "whitespace_only",
        declared: Declared::Literal("   "),
        expect: Expect::Refused("evidence_path_escape"),
        detail_contains: "empty",
        why: "same as the empty case, and a trim is the only thing that separates them",
    },
    PathCase {
        label: "last_component_symlink",
        declared: Declared::LastComponentSymlink,
        expect: Expect::Refused("evidence_path_escape"),
        detail_contains: "outside the benchmark directory",
        why: "the declared string is well-formed; only canonicalization can see this",
    },
    PathCase {
        label: "first_component_symlink_dir",
        declared: Declared::FirstComponentSymlinkDir,
        expect: Expect::Refused("evidence_path_escape"),
        detail_contains: "outside the benchmark directory",
        why: "the escape is in the FIRST component, so a last-component-only check misses it",
    },
    PathCase {
        label: "prefix_sibling_symlink",
        declared: Declared::PrefixSiblingSymlink,
        expect: Expect::Refused("evidence_path_escape"),
        detail_contains: "outside the benchmark directory",
        why: "THE ROW THAT GOES RED IF CONTAINMENT USES `str::starts_with`: a bench dir of \
              `<tmp>/bench` is a string prefix of `<tmp>/bench-evil/x.json` and a COMPONENT \
              prefix of nothing in it. `Path::starts_with` is component-wise, and is why this \
              row passes",
    },
    // ---- MUST MATCH: three acceptance rows ---------------------------------------------
    PathCase {
        label: "committed_spelling",
        declared: Declared::CommittedForKind,
        expect: Expect::Accepted,
        detail_contains: "",
        why: "the real spelling every committed row carries; a gate that refuses it cannot pass",
    },
    PathCase {
        label: "committed_spelling_cur_dir",
        declared: Declared::CommittedForKindCurDir,
        expect: Expect::Accepted,
        detail_contains: "",
        why: "`./x` names the same file as `x`; THE ROW THAT GOES RED IF STAGE 1 OVER-REFUSES",
    },
    PathCase {
        label: "committed_spelling_other_kind_dir",
        declared: Declared::CommittedForKind,
        expect: Expect::Accepted,
        detail_contains: "",
        why: "swept over BOTH kinds by the loop, so the LEDGER_DIR spelling is proven to be \
              accepted under `EvidenceKind::Ledger` and not only the LOCKS_DIR one",
    },
    // ---- BEHAVIOUR-PRESERVING: the new refusal must not swallow a distinct diagnosis ----
    PathCase {
        label: "absent_file_under_kind_dir",
        declared: Declared::AbsentUnderKindDir,
        expect: Expect::Refused("evidence_file_missing"),
        detail_contains: "does not exist",
        why: "AN ABSENT FILE IS NOT AN ESCAPE. Reporting it as one would tell an operator to \
              repoint a field that is already correct",
    },
    PathCase {
        label: "directory_in_a_files_position",
        declared: Declared::KindDirItself,
        expect: Expect::Refused("evidence_read_failed"),
        detail_contains: "not a regular file",
        why: "a contained, existing DIRECTORY passes containment and is refused by the bounded \
              read — the pre-existing diagnosis, unchanged",
    },
];

/// This kind's own directory under the benchmark directory.
fn kind_dir(kind: EvidenceKind) -> &'static str {
    match kind {
        EvidenceKind::Row => ROWS_DIR,
        EvidenceKind::Lock => LOCKS_DIR,
        EvidenceKind::Ledger => LEDGER_DIR,
        // Never swept by `EVIDENCE_PATH_CASES`, and that is the point: the selection
        // manifest's path is GATE-DERIVED and never reaches `resolve_committed_evidence_path`,
        // so a row shape for it would be a guard over a path production cannot take.
        EvidenceKind::SelectionManifest => SELECTIONS_DIR,
    }
}

/// The committed filename this kind carries for the s8/seed13 cell.
fn kind_committed_file(kind: EvidenceKind) -> &'static str {
    match kind {
        EvidenceKind::Row => "setfit-s8-seed13.json",
        EvidenceKind::Lock => "setfit-s8-seed13.lock.json",
        EvidenceKind::Ledger => "setfit-s8-seed13.jsonl",
        // See `kind_dir`: this kind is never swept by the path-shape table, because its path
        // is derived and never row-supplied. The arm exists so that adding the kind forces
        // this file to be revisited rather than falling into a `_` arm.
        EvidenceKind::SelectionManifest => SELECTION_MANIFEST_FILE,
    }
}

/// A purpose-built tree whose BENCH DIRECTORY HAS A CHOSEN NAME.
///
/// `write_valid_run`'s temp dir has a random name, and the prefix-sibling row needs a sibling
/// whose name has the bench directory's name as a string prefix — which cannot be arranged
/// without naming the bench directory. Returns `(root, bench_dir)`; the root must outlive the
/// bench dir, so it is handed back rather than dropped.
fn path_case_fixture(kind: EvidenceKind) -> (TempDir, PathBuf) {
    let root = TempDir::new().expect("a temp root");
    let bench = root.path().join("bench");
    // `bench-evil` has `bench` as a STRING prefix and is not under it by any component.
    let evil = root.path().join("bench-evil");
    let outside = root.path().join("outside");
    fs::create_dir_all(bench.join(LOCKS_DIR)).expect("locks dir");
    fs::create_dir_all(bench.join(LEDGER_DIR)).expect("ledger dir");
    fs::create_dir_all(&evil).expect("the prefix-sibling dir");
    fs::create_dir_all(&outside).expect("the outside dir");
    fs::write(evil.join("x.json"), b"{}\n").expect("prefix-sibling file");
    fs::write(outside.join("anywhere.json"), b"{}\n").expect("outside file");
    fs::write(bench.join(kind_dir(kind)).join(kind_committed_file(kind)), b"{}\n")
        .expect("the committed evidence file");
    (root, bench)
}

/// Build one case's declared string against a fixture, creating any symlink it needs.
fn declared_string(declared: Declared, root: &Path, bench: &Path, kind: EvidenceKind) -> String {
    let dir = kind_dir(kind);
    match declared {
        Declared::Literal(literal) => literal.to_string(),
        Declared::CommittedForKind => format!("{dir}/{}", kind_committed_file(kind)),
        Declared::CommittedForKindCurDir => format!("./{dir}/{}", kind_committed_file(kind)),
        Declared::AbsentUnderKindDir => format!("{dir}/never-written.json"),
        Declared::KindDirItself => dir.to_string(),
        Declared::LastComponentSymlink => {
            let relative = format!("{dir}/via-last-link.json");
            symlink_for_test(&root.join("outside").join("anywhere.json"), &bench.join(&relative));
            relative
        }
        Declared::FirstComponentSymlinkDir => {
            symlink_for_test(&root.join("outside"), &bench.join("linked-dir"));
            "linked-dir/anywhere.json".to_string()
        }
        Declared::PrefixSiblingSymlink => {
            let relative = format!("{dir}/to-prefix-sibling.json");
            symlink_for_test(&root.join("bench-evil").join("x.json"), &bench.join(&relative));
            relative
        }
    }
}

/// `#[cfg(unix)]` gates the MATERIALIZER, never a table row: a table that silently shrinks on
/// some host is a coverage loss no test-count floor can see.
fn symlink_for_test(target: &Path, link: &Path) {
    #[cfg(unix)]
    std::os::unix::fs::symlink(target, link).expect("symlink");
    #[cfg(not(unix))]
    {
        let _ = (target, link);
    }
}

#[test]
fn bench_gate_evidence_path_case_table_over_both_evidence_kinds() {
    let mut checked = 0_usize;
    for kind in [EvidenceKind::Lock, EvidenceKind::Ledger] {
        for case in &EVIDENCE_PATH_CASES {
            // A FRESH fixture per (case, kind), so no case can see another's symlinks.
            let (root, bench) = path_case_fixture(kind);
            let declared = declared_string(case.declared, root.path(), &bench, kind);
            let cell = CellKey::new(Method::Setfit, 8, 13);

            // The resolver, then the bounded read — exactly the pair `verify_provenance`
            // composes, so `evidence_read_failed` can still be reached for a contained
            // directory and is not swallowed by the new refusal.
            let outcome = resolve_committed_evidence_path(cell, &bench, kind, &declared)
                .and_then(|path| read_evidence(cell, kind, &path));

            match (case.expect, outcome) {
                (Expect::Accepted, Ok(_)) => {}
                (Expect::Accepted, Err(error)) => panic!(
                    "[{kind:?}/{}] `{declared}` must be ACCEPTED ({}), but was refused: {error}",
                    case.label, case.why
                ),
                (Expect::Refused(tag), Err(error)) => {
                    assert_eq!(
                        error.variant_tag(),
                        tag,
                        "[{kind:?}/{}] `{declared}` must be refused as `{tag}` ({}), got: {error}",
                        case.label,
                        case.why
                    );
                    if !case.detail_contains.is_empty() {
                        assert!(
                            error.to_string().contains(case.detail_contains),
                            "[{kind:?}/{}] the refusal must carry `{}`: {error}",
                            case.label,
                            case.detail_contains
                        );
                    }
                }
                (Expect::Refused(tag), Ok(_)) => panic!(
                    "[{kind:?}/{}] `{declared}` was ACCEPTED but must be refused as `{tag}`. {}",
                    case.label, case.why
                ),
            }
            checked += 1;
        }
    }
    // ROWS x KINDS, asserted, so neither field can silently lose coverage and a row deleted
    // to make something pass goes red here rather than quietly.
    assert_eq!(checked, EVIDENCE_PATH_CASES.len() * 2, "every row is swept over BOTH kinds");
    let refused = EVIDENCE_PATH_CASES
        .iter()
        .filter(|case| matches!(case.expect, Expect::Refused("evidence_path_escape")))
        .count();
    let accepted =
        EVIDENCE_PATH_CASES.iter().filter(|case| case.expect == Expect::Accepted).count();
    assert!(refused >= 8, "the table must carry at least eight escape shapes, has {refused}");
    assert!(accepted >= 3, "the table must carry at least three acceptance rows, has {accepted}");
    println!(
        "[bench_gate] EVIDENCE_PATH_CASES rows={} kinds=2 assertions={checked} escapes={refused} \
         accepted={accepted}",
        EVIDENCE_PATH_CASES.len()
    );
}

// ===========================================================================================
// DETERMINISTIC REFUSAL ORDER (plan 05-15 task 3)
// ===========================================================================================

#[test]
fn bench_gate_reports_the_first_offending_cell_in_contract_order_across_runs() {
    // TWO cells doctored, so there is a CHOICE to make. The contract order is (method, shots
    // ascending, seed ascending), so s8/seed17 precedes s32/seed41 and must be the one named
    // on every invocation. Repeated, because an assertion taken once is about a run rather
    // than about an order.
    let first = CellKey::new(Method::Setfit, 8, 17);
    let second = CellKey::new(Method::Setfit, 32, 41);
    let mut named: Vec<String> = Vec::new();

    for _ in 0..3 {
        let dir = write_valid_run(RunSpec::default());
        for cell in [second, first] {
            // Doctored in REVERSE contract order, so a gate that reported "whichever was
            // edited first" would name the wrong one.
            reseal_row(dir.path(), cell, |payload| {
                if let MethodEvidence::Setfit(evidence) = &mut payload.evidence {
                    evidence.lock.lock_record_path = "../escaped.json".to_string();
                }
            });
        }
        let manifest = manifest_for(dir.path());
        let error = refuse(&manifest, dir.path(), "two escaping cells are a refusal");
        assert_eq!(error.variant_tag(), "evidence_path_escape");
        named.push(error.cell().unwrap_or("<none>").to_string());
    }

    assert_eq!(
        named,
        vec![first.render(), first.render(), first.render()],
        "the refusal must name the FIRST offending cell in contract order, on every run"
    );
}

// ===========================================================================================
// GAP 2 — THE SELECTION-MANIFEST BINDING CASE TABLE (plan 05-16 task 2)
//
// Verifier gap 2 (EVAL-02, graded PARTIAL) measured that the RECORDING half is complete —
// 40/40 committed rows carry a distinct `selection_manifest_hash` equal to the `semantic_hash`
// of the committed manifest for their own cell — while the ENFORCING half did not exist:
// `verify_run` opened no selection manifest, so the report exited 0 with the whole
// `selections/` directory deleted (spot-check G) and exited 0 with a row's hash doctored to 64
// zeros (spot-check F). The forty committed manifests were inert files.
//
// SCOPE: every row below is mutated on a freshly built VALID ACTIVE 40-cell run and driven
// through the PUBLIC `verify_run` door. That is deliberate and it is not interchangeable with a
// helper-level proof — CLAUDE.md Verification Discipline rule 4, the same rule that forced
// 05-11 to re-mutate four of the six original negatives.
//
// Ship the TABLE and re-run it rather than re-reading the check (rule 7). A further shape is a
// new row here, never another test function.
// ===========================================================================================

/// One mutation of a valid run's selection-manifest binding.
///
/// Every variant carries the cell it breaks, so the table can assert WHICH cell the refusal
/// names as well as which variant it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SelectionMutation {
    /// The control. Nothing is touched.
    None,
    /// The ENTIRE `selections/` directory is removed — spot-check G, and the cheapest possible
    /// disproof of "the forty manifests are inert", because it edits no row byte at all.
    DeleteSelectionsDir,
    /// One cell's own `selections/s{shots}-seed{seed}/` directory is removed.
    DeleteCellDir(CellKey),
    /// One manifest is truncated to zero bytes: a PARSE failure, not a hash disagreement.
    TruncateManifestToZeroBytes(CellKey),
    /// Spot-check F: one row's `selection_manifest_hash` doctored to 64 zeros, resealed.
    RowHashToZeros(CellKey),
    /// The same, but doctored to another cell's REAL hash — so the value is well-formed and
    /// even meaningful, just not this cell's.
    RowHashToAnotherCellsHash {
        /// The cell whose row is doctored.
        cell: CellKey,
        /// The cell whose real hash is written into it.
        other: CellKey,
    },
    /// TRANSPLANT: the donor's manifest is copied over this cell's file AND this cell's row
    /// hash is doctored to the transplanted manifest's hash, so the hash check AGREES and only
    /// the cell-key comparison can refuse.
    TransplantWithRowDoctored {
        /// The cell whose manifest file is overwritten.
        cell: CellKey,
        /// The cell whose manifest is transplanted into it.
        donor: CellKey,
    },
    /// THE INVERTING CASE: the same transplant WITHOUT the row-hash doctoring, which must be
    /// refused by the HASH check instead. Without this row the transplant rows could pass while
    /// the cell-key check they exist to prove was never reached.
    TransplantWithoutRowDoctored {
        /// The cell whose manifest file is overwritten.
        cell: CellKey,
        /// The cell whose manifest is transplanted into it.
        donor: CellKey,
    },
    /// One manifest's payload is edited and its envelope digest is NOT resealed.
    EditManifestPayloadWithoutResealing(CellKey),
}

/// What the swept run must do with a mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SelectionExpect {
    /// `verify_run` returns `Ok`. Without this row the table could pass by refusing everything.
    Accepted,
    /// Refused with this variant tag, naming this cell.
    Refused(&'static str, CellKey),
}

/// One row of the table.
struct SelectionCase {
    label: &'static str,
    mutation: SelectionMutation,
    expect: SelectionExpect,
    why: &'static str,
}

/// The first cell in contract order: `(setfit, 8, 13)`. Whatever breaks EVERY cell is reported
/// here, and the low end of the shot axis has no contracted step below it.
const FIRST_CELL: CellKey = CellKey::new(Method::Setfit, 8, 13);
/// Its seed-adjacent neighbour, whose manifest is byte-DIFFERENT despite the adjacency.
const ADJACENT_SEED: CellKey = CellKey::new(Method::Setfit, 8, 17);
/// One contracted step UP the shot axis from [`FIRST_CELL`].
const ONE_SHOT_STEP_UP: CellKey = CellKey::new(Method::Setfit, 16, 13);
/// The high end of the shot axis: no contracted step above 64.
const TOP_OF_SHOT_AXIS: CellKey = CellKey::new(Method::Setfit, 64, 13);
/// One contracted step DOWN from the top of the shot axis.
const ONE_SHOT_STEP_DOWN: CellKey = CellKey::new(Method::Setfit, 32, 13);

const SELECTION_BINDING_CASES: [SelectionCase; 11] = [
    // ---- THE CONTROL --------------------------------------------------------------------
    SelectionCase {
        label: "control",
        mutation: SelectionMutation::None,
        expect: SelectionExpect::Accepted,
        why: "a valid run with forty real sealed manifests must still verify; without this row \
              a table that refused everything would pass",
    },
    // ---- ABSENT EVIDENCE ----------------------------------------------------------------
    SelectionCase {
        label: "selections_dir_deleted",
        mutation: SelectionMutation::DeleteSelectionsDir,
        expect: SelectionExpect::Refused("evidence_file_missing", FIRST_CELL),
        why: "SPOT-CHECK G. Every cell loses its manifest, so the FIRST in contract order is \
              the one named. No row byte changes, so nothing else can be what refused",
    },
    SelectionCase {
        label: "one_cell_dir_deleted",
        mutation: SelectionMutation::DeleteCellDir(TARGET_SETFIT),
        expect: SelectionExpect::Refused("evidence_file_missing", TARGET_SETFIT),
        why: "a missing PER-CELL directory is as absent as a missing file, and must name the \
              cell and the path rather than producing an Ok over an absent input",
    },
    SelectionCase {
        label: "manifest_truncated_to_zero_bytes",
        mutation: SelectionMutation::TruncateManifestToZeroBytes(TARGET_SETFIT),
        expect: SelectionExpect::Refused("evidence_read_failed", TARGET_SETFIT),
        why: "OBSERVED, not assumed: a zero-byte file is a regular file within the cap, so the \
              bounded read SUCCEEDS and `SelectionManifest::from_bytes` fails at serde — a \
              PARSE failure, which the implementation maps to `evidence_read_failed` and NOT \
              to the digest-sourced mismatch. A zero-byte file is not a hash disagreement",
    },
    // ---- A DOCTORED ROW HASH ------------------------------------------------------------
    SelectionCase {
        label: "row_hash_doctored_to_64_zeros",
        mutation: SelectionMutation::RowHashToZeros(TARGET_SETFIT),
        expect: SelectionExpect::Refused("selection_manifest_mismatch", TARGET_SETFIT),
        why: "SPOT-CHECK F. The row is resealed and its manifest digest re-recorded, so every \
              other check still passes and ONLY the recomputation from the committed manifest \
              can catch it",
    },
    SelectionCase {
        label: "row_hash_doctored_to_another_cells_real_hash",
        mutation: SelectionMutation::RowHashToAnotherCellsHash {
            cell: FIRST_CELL,
            other: ADJACENT_SEED,
        },
        expect: SelectionExpect::Refused("selection_manifest_mismatch", FIRST_CELL),
        why: "ADJACENCY: two cells one seed apart do not merge. The value written is a REAL, \
              well-formed manifest digest — just not this cell's — so a check that only \
              validated the SHAPE of the hash would accept it",
    },
    // ---- TRANSPLANTS: the rows that carry the weight ------------------------------------
    SelectionCase {
        label: "transplant_adjacent_seed_with_row_doctored",
        mutation: SelectionMutation::TransplantWithRowDoctored {
            cell: FIRST_CELL,
            donor: ADJACENT_SEED,
        },
        expect: SelectionExpect::Refused("selection_manifest_cell_mismatch", FIRST_CELL),
        why: "THE ROW-HASH DOCTORING IS WHAT MAKES THIS ROW MEAN ANYTHING. The transplanted \
              manifest is internally valid and seals correctly, and the row now claims its \
              hash — so step 1 AGREES and the cell-key comparison is the only thing left that \
              can refuse. Without the doctoring this row would report the hash mismatch and \
              prove nothing about the second check",
    },
    SelectionCase {
        label: "transplant_one_shot_step_down_with_row_doctored",
        mutation: SelectionMutation::TransplantWithRowDoctored {
            cell: ONE_SHOT_STEP_UP,
            donor: FIRST_CELL,
        },
        expect: SelectionExpect::Refused("selection_manifest_cell_mismatch", ONE_SHOT_STEP_UP),
        why: "8 against a 16 cell — one contracted step DOWN the shot axis, in the interior. \
              Row doctored, as above",
    },
    SelectionCase {
        label: "transplant_one_shot_step_up_with_row_doctored",
        mutation: SelectionMutation::TransplantWithRowDoctored {
            cell: ONE_SHOT_STEP_DOWN,
            donor: TOP_OF_SHOT_AXIS,
        },
        expect: SelectionExpect::Refused("selection_manifest_cell_mismatch", ONE_SHOT_STEP_DOWN),
        why: "64 against a 32 cell — one contracted step UP, in the interior. Row doctored",
    },
    // ---- THE INVERTING ROW --------------------------------------------------------------
    SelectionCase {
        label: "transplant_WITHOUT_row_doctored",
        mutation: SelectionMutation::TransplantWithoutRowDoctored {
            cell: FIRST_CELL,
            donor: ADJACENT_SEED,
        },
        expect: SelectionExpect::Refused("selection_manifest_mismatch", FIRST_CELL),
        why: "THE INVERSION. The same transplant with the row left alone must report the HASH \
              mismatch, which proves the two checks are DISTINGUISHABLE and that the rows \
              above really reach the second one rather than passing on the first",
    },
    // ---- AN UNSEALED MANIFEST -----------------------------------------------------------
    SelectionCase {
        label: "manifest_payload_edited_without_resealing",
        mutation: SelectionMutation::EditManifestPayloadWithoutResealing(TARGET_SETFIT),
        expect: SelectionExpect::Refused("selection_manifest_mismatch", TARGET_SETFIT),
        why: "sourced from `SelectionManifest::from_bytes`'s OWN digest refusal, never from a \
              second recomputation in `bench_gate` — one definition of a manifest's identity \
              (OPS-03)",
    },
];

/// The shot-axis transplants proven at BOTH ENDS of the axis, where there is no contracted
/// step below 8 or above 64. Kept beside the interior rows rather than folded into them,
/// because "the comparison works one step away" and "the comparison works where there is no
/// step" are two different statements.
const SELECTION_BOUNDARY_TRANSPLANTS: [(&str, CellKey, CellKey); 2] = [
    ("s16_manifest_under_an_s8_cell", FIRST_CELL, ONE_SHOT_STEP_UP),
    ("s32_manifest_under_an_s64_cell", TOP_OF_SHOT_AXIS, ONE_SHOT_STEP_DOWN),
];

/// Overwrite a row's `selection_manifest_hash` and reseal it.
fn doctor_selection_hash(root: &Path, cell: CellKey, hash: &str) {
    let hash = hash.to_string();
    reseal_row(root, cell, |payload| {
        payload.selection_manifest_hash = hash;
    });
}

/// Copy `donor`'s committed manifest over `cell`'s own file, returning the donor's digest.
fn transplant_manifest(root: &Path, cell: CellKey, donor: CellKey) -> String {
    let donor_manifest = synthetic_selection_manifest(donor.shots, donor.seed);
    fs::write(
        selection_manifest_at(root, cell.shots, cell.seed),
        donor_manifest.to_file_bytes().expect("the donor manifest serializes"),
    )
    .expect("transplant write");
    donor_manifest.semantic_hash
}

/// Apply one mutation to a freshly built valid run, BEFORE the manifest is derived.
fn apply_selection_mutation(root: &Path, mutation: SelectionMutation) {
    match mutation {
        SelectionMutation::None => {}
        SelectionMutation::DeleteSelectionsDir => {
            fs::remove_dir_all(root.join("selections")).expect("remove the selections dir");
        }
        SelectionMutation::DeleteCellDir(cell) => {
            let path = selection_manifest_at(root, cell.shots, cell.seed);
            fs::remove_dir_all(path.parent().expect("the manifest has a parent"))
                .expect("remove the per-cell selection dir");
        }
        SelectionMutation::TruncateManifestToZeroBytes(cell) => {
            fs::write(selection_manifest_at(root, cell.shots, cell.seed), b"")
                .expect("truncate the manifest");
        }
        SelectionMutation::RowHashToZeros(cell) => {
            doctor_selection_hash(root, cell, &"0".repeat(64));
        }
        SelectionMutation::RowHashToAnotherCellsHash { cell, other } => {
            let hash = synthetic_selection_hash(other.shots, other.seed);
            doctor_selection_hash(root, cell, &hash);
        }
        SelectionMutation::TransplantWithRowDoctored { cell, donor } => {
            let hash = transplant_manifest(root, cell, donor);
            doctor_selection_hash(root, cell, &hash);
        }
        SelectionMutation::TransplantWithoutRowDoctored { cell, donor } => {
            let _ = transplant_manifest(root, cell, donor);
        }
        SelectionMutation::EditManifestPayloadWithoutResealing(cell) => {
            let path = selection_manifest_at(root, cell.shots, cell.seed);
            let bytes = fs::read(&path).expect("manifest read");
            let mut value: serde_json::Value =
                serde_json::from_slice(&bytes).expect("the manifest is JSON");
            // An EXISTING field's value, because the envelope is `deny_unknown_fields` and a
            // new key would fail to PARSE rather than fail to hash.
            *value
                .get_mut("payload")
                .and_then(|p| p.get_mut("normalization_version"))
                .expect("the payload carries a normalization version") =
                serde_json::Value::String("nfc-trim-ws-v99".to_string());
            let mut edited = serde_json::to_vec_pretty(&value).expect("manifest serializes");
            edited.push(b'\n');
            fs::write(&path, edited).expect("manifest write");
        }
    }
}

#[test]
fn bench_gate_selection_binding_case_table_at_the_active_scope() {
    let mut observations: Vec<(&str, String)> = Vec::new();

    for case in &SELECTION_BINDING_CASES {
        let dir = write_valid_run(RunSpec::default());
        apply_selection_mutation(dir.path(), case.mutation);
        // AFTER the mutation, so a resealed row's new digest is what the manifest records and
        // the defect under test is the binding rather than a row-digest disagreement.
        let manifest = manifest_for(dir.path());

        let (observed, named_cell, rendered) = match verify_run(&manifest, dir.path()) {
            Ok(set) => (format!("Ok({} rows verified)", set.len()), String::new(), String::new()),
            Err(error) => (
                error.variant_tag().to_string(),
                error.cell().unwrap_or("<none>").to_string(),
                error.to_string(),
            ),
        };
        // PRINTED PER ROW, and asserted only at the end, so one run reports every row rather
        // than aborting on the first — the same shape 05-15's escape sweep used.
        println!(
            "[bench_gate] SELECTION_BINDING case={} observed={observed} cell={named_cell} ({})",
            case.label, case.why
        );

        let expected = match case.expect {
            SelectionExpect::Accepted => "Ok(40 rows verified)".to_string(),
            SelectionExpect::Refused(tag, cell) => {
                if observed == tag {
                    // The refusal must NAME the cell and the manifest path, or an operator
                    // cannot tell which of forty cells to look at.
                    assert_eq!(
                        named_cell,
                        cell.render(),
                        "[{}] the refusal must name {}: {rendered}",
                        case.label,
                        cell.render()
                    );
                    assert!(
                        rendered.contains(&selection_manifest_rel(cell.shots, cell.seed))
                            || rendered.contains("selections"),
                        "[{}] the refusal must name the manifest path: {rendered}",
                        case.label
                    );
                }
                tag.to_string()
            }
        };
        observations.push((case.label, format!("{observed}|expected={expected}")));
    }

    let failures: Vec<&(&str, String)> = observations
        .iter()
        .filter(|(_, line)| {
            let (observed, expected) = line.split_once("|expected=").unwrap_or((line, ""));
            observed != expected
        })
        .collect();
    assert!(
        failures.is_empty(),
        "every selection-binding shape must be refused with its own variant through verify_run \
         at the ACTIVE 40-cell scope, and the control must be ACCEPTED; these were not: \
         {failures:#?}"
    );

    // NON-VACUITY over the table itself: a row deleted to make something pass goes red here.
    let accepted = SELECTION_BINDING_CASES
        .iter()
        .filter(|case| case.expect == SelectionExpect::Accepted)
        .count();
    let cell_mismatches = SELECTION_BINDING_CASES
        .iter()
        .filter(|case| {
            matches!(case.expect, SelectionExpect::Refused("selection_manifest_cell_mismatch", _))
        })
        .count();
    // THE SWEPT-ROW COUNT IS PINNED HERE, and it has to be, because the Make floor cannot see
    // it. `assert_tests_ran` counts test FUNCTIONS, and this table is deliberately ONE
    // function sweeping eleven rows — so eleven rows silently becoming three would not move
    // the floor by a single test. A row deleted to make something pass goes red on this line.
    assert_eq!(
        observations.len(),
        SELECTION_BINDING_CASES.len(),
        "every row of the table must be swept"
    );
    assert_eq!(SELECTION_BINDING_CASES.len(), 11, "the table carries eleven rows");
    assert_eq!(accepted, 1, "the table must carry exactly one ACCEPTANCE row");
    assert!(
        cell_mismatches >= 3,
        "the table must carry at least three transplant rows, has {cell_mismatches}"
    );

    // THE SCOPE FENCE, asserted rather than asserted-in-prose: this proof is single-method and
    // intra-cell, so it must not lean on the DEFERRED two-method scope for any of its rows.
    assert!(
        SELECTION_BINDING_CASES
            .iter()
            .all(|case| matches!(case.expect, SelectionExpect::Accepted)
                || matches!(case.expect, SelectionExpect::Refused(_, cell) if cell.method == Method::Setfit)),
        "every selection-binding row is an ACTIVE-scope, single-method cell; a LoRA cell here \
         would be D-ITEM-05-15's cross-method check wearing this one's name"
    );
}

#[test]
fn bench_gate_refuses_a_transplanted_manifest_at_both_ends_of_the_shot_axis() {
    // The interior transplants above prove the cell-key comparison one contracted step away.
    // These two prove it where there IS no step — below 8 and above 64 — because "works for a
    // neighbour" and "works at the end of the axis" are different statements and the second is
    // where an off-by-one in an axis walk would hide.
    for (label, cell, donor) in SELECTION_BOUNDARY_TRANSPLANTS {
        let dir = write_valid_run(RunSpec::default());
        let hash = transplant_manifest(dir.path(), cell, donor);
        doctor_selection_hash(dir.path(), cell, &hash);
        let manifest = manifest_for(dir.path());

        let error = refuse(&manifest, dir.path(), label);
        assert_eq!(
            error.variant_tag(),
            "selection_manifest_cell_mismatch",
            "[{label}] a transplanted manifest at the end of the shot axis must be refused by \
             the CELL-KEY comparison, got: {error}"
        );
        assert_eq!(error.cell(), Some(cell.render().as_str()), "[{label}] {error}");
        let rendered = error.to_string();
        assert!(
            rendered.contains(&donor.shots.to_string()),
            "[{label}] the refusal must name the manifest's own declared shots: {rendered}"
        );
    }
}

#[test]
fn bench_gate_the_forty_synthetic_selection_manifests_are_forty_distinct_digests() {
    // The committed set carries 40 DISTINCT `semantic_hash` values (measured on the tree, not
    // assumed). A fixture whose manifests collided would make every transplant negative pass
    // for the wrong reason: a donor identical to the target satisfies the hash check by
    // accident and the cell-key comparison is never what refused.
    let digests: BTreeSet<String> = RunManifest::expectation()
        .into_iter()
        .map(|cell| synthetic_selection_hash(cell.shots, cell.seed))
        .collect();
    assert_eq!(digests.len(), EXPECTED_CELLS, "forty cells, forty distinct selection digests");
    for digest in &digests {
        assert_eq!(digest.len(), 64, "a sealed digest is 64 lowercase hex characters");
        assert!(digest.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    }
}

#[test]
fn bench_gate_a_hash_agreeing_in_its_first_characters_is_still_a_refusal() {
    // EXACT BYTE EQUALITY, stated as a negative. A prefix match, a truncation to 8 or 16
    // characters, or an `eq_ignore_ascii_case` would each accept one of the three rows below,
    // and each is a plausible implementation slip that no other test in this file would see.
    let honest = synthetic_selection_hash(FIRST_CELL.shots, FIRST_CELL.seed);
    let mut flipped_last: String = honest.clone();
    flipped_last.pop();
    flipped_last.push(if honest.ends_with('0') { '1' } else { '0' });

    let doctored: Vec<(&str, String)> = vec![
        ("last_character_flipped", flipped_last),
        ("truncated_to_sixteen", honest[..16].to_string()),
        ("upper_cased", honest.to_uppercase()),
    ];
    for (label, hash) in doctored {
        assert_ne!(hash, honest, "[{label}] the doctored value must differ from the honest one");
        let dir = write_valid_run(RunSpec::default());
        doctor_selection_hash(dir.path(), FIRST_CELL, &hash);
        let manifest = manifest_for(dir.path());

        let error = refuse(&manifest, dir.path(), label);
        assert_eq!(
            error.variant_tag(),
            "selection_manifest_mismatch",
            "[{label}] a hash that agrees in its first characters is still a refusal, got: \
             {error}"
        );
    }
}

#[test]
fn bench_gate_reports_the_first_offending_selection_binding_in_contract_order_across_runs() {
    // TWO cells broken, so there is a CHOICE. Contract order is (method, shots ascending, seed
    // ascending), so (8, 17) precedes (32, 41) and must be named on every invocation. The two
    // are broken in REVERSE contract order, so a gate reporting "whichever was edited first"
    // names the wrong one.
    let first = ADJACENT_SEED;
    let second = CellKey::new(Method::Setfit, 32, 41);
    let mut named: Vec<String> = Vec::new();

    for _ in 0..3 {
        let dir = write_valid_run(RunSpec::default());
        for cell in [second, first] {
            doctor_selection_hash(dir.path(), cell, &"0".repeat(64));
        }
        let manifest = manifest_for(dir.path());
        let error = refuse(&manifest, dir.path(), "two broken selection bindings are a refusal");
        assert_eq!(error.variant_tag(), "selection_manifest_mismatch");
        named.push(error.cell().unwrap_or("<none>").to_string());
    }

    assert_eq!(
        named,
        vec![first.render(), first.render(), first.render()],
        "the refusal must name the FIRST offending cell in contract order, on every run"
    );
}

// ===========================================================================================
// THE CLOSED-FORM QUALITY CROSS-CHECK (05-17, verifier advisory 2)
// ===========================================================================================
//
// Spot-check D doctored `quality.f_avg` from 0.4579 to 0.99, repaired the row envelope digest,
// the manifest's `row_sha256` and the manifest envelope digest, and got rc=0 with the published
// mean moving 0.4746 -> 0.5278. Every mutation below repairs the same three digests — that is
// what `reseal_row` followed by `manifest_for` does — so nothing other than the cross-check can
// be the thing that refuses, and the PRE-FIX observation for every row is `Ok(40 rows verified)`.

/// One doctoring of a published quality figure, applied to a VALID run.
#[derive(Debug, Clone, Copy)]
enum QualityMutation {
    /// No mutation. The ACCEPTANCE row.
    None,
    /// SPOT-CHECK D REPLAYED: `f_avg` -> 0.99, every digest repaired.
    FAvgTo099(CellKey),
    /// `macro_f1` moved, with its bits sibling repaired so only the matrix can catch it.
    MacroF1Doctored(CellKey),
    /// `mcc` moved, bits repaired.
    MccDoctored(CellKey),
    /// ONE element of ONE per-class vector moved — the smallest edit the check must see.
    PerClassF1ElementDoctored(CellKey),
    /// `n_test_rows` inflated, so the cell looks better powered than its matrix says.
    NTestRowsDoctored(CellKey),
    /// `f_avg_bits` moved while `f_avg` is left alone: the row contradicts ITSELF.
    FAvgBitsOnlyDoctored(CellKey),
    /// The reverse — `f_avg` moved while `f_avg_bits` is left alone.
    FAvgDecimalOnlyDoctored(CellKey),
    /// `ece_top_label_validation_bits` moved. Its VALUE is not recomputable from any committed
    /// file, but a bits field is a claim the row makes about ITSELF and costs nothing to hold.
    EceBitsOnlyDoctored(CellKey),
    /// A confusion matrix that is not square.
    NonSquareMatrix(CellKey),
    /// A square 4x4 matrix against three `ordered_labels`.
    DimensionMismatch(CellKey),
    /// A square, correctly-dimensioned matrix whose counts total zero.
    AllZeroMatrix(CellKey),
}

/// What the swept run must do with a quality mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QualityExpect {
    /// `verify_run` returns `Ok`. Without this row the table could pass by refusing everything.
    Accepted,
    /// Refused with this variant tag, naming this cell, and naming this field in its message.
    Refused(&'static str, CellKey, &'static str),
}

/// One row of the quality cross-check table.
struct QualityCase {
    label: &'static str,
    mutation: QualityMutation,
    expect: QualityExpect,
    why: &'static str,
}

const QUALITY_CROSS_CHECK_CASES: [QualityCase; 12] = [
    // ---- THE CONTROL --------------------------------------------------------------------
    QualityCase {
        label: "control",
        mutation: QualityMutation::None,
        expect: QualityExpect::Accepted,
        why: "the synthetic quality blocks are closed forms over their own confusion matrices \
              (05-17 task 1), so an untouched run must still verify; without this row a table \
              that refused everything would pass",
    },
    // ---- THE RECOMPUTED FIGURES ---------------------------------------------------------
    QualityCase {
        label: "spot_check_D_f_avg_to_0_99",
        mutation: QualityMutation::FAvgTo099(FIRST_CELL),
        expect: QualityExpect::Refused("quality_cross_check_mismatch", FIRST_CELL, "f_avg"),
        why: "SPOT-CHECK D, replayed. The verifier measured rc=0 for exactly this with the row \
              envelope digest, the manifest's row_sha256 and the manifest envelope digest all \
              repaired — which this mutation also repairs — so the headline is recomputable or \
              nothing refuses it",
    },
    QualityCase {
        label: "macro_f1_doctored",
        mutation: QualityMutation::MacroF1Doctored(TARGET_SETFIT),
        expect: QualityExpect::Refused("quality_cross_check_mismatch", TARGET_SETFIT, "macro_f1"),
        why: "the second published accuracy figure. It is averaged over a DIFFERENT class set \
              than f_avg, so a check that only held the headline would leave it free",
    },
    QualityCase {
        label: "mcc_doctored",
        mutation: QualityMutation::MccDoctored(TARGET_SETFIT),
        expect: QualityExpect::Refused("quality_cross_check_mismatch", TARGET_SETFIT, "mcc"),
        why: "MCC narrows to f32 before the row widens it back, so its recomputation has to \
              reproduce the NARROWING as well as the arithmetic — a row that matched a pure-f64 \
              recomputation would be refused here for the wrong reason",
    },
    QualityCase {
        label: "one_per_class_f1_element_doctored",
        mutation: QualityMutation::PerClassF1ElementDoctored(TARGET_SETFIT),
        expect: QualityExpect::Refused(
            "quality_cross_check_mismatch",
            TARGET_SETFIT,
            "per_class_f1",
        ),
        why: "ONE element of ONE vector: the smallest edit a reader could publish, and the one \
              a check comparing only vector LENGTHS would miss",
    },
    QualityCase {
        label: "n_test_rows_doctored",
        mutation: QualityMutation::NTestRowsDoctored(TARGET_SETFIT),
        expect: QualityExpect::Refused(
            "quality_cross_check_mismatch",
            TARGET_SETFIT,
            "n_test_rows",
        ),
        why: "the denominator every metric divides by, and the field that makes a cell look \
              better powered than it was. The matrix's own total is the answer",
    },
    // ---- THE ROW CONTRADICTING ITSELF ---------------------------------------------------
    QualityCase {
        label: "f_avg_bits_doctored_value_untouched",
        mutation: QualityMutation::FAvgBitsOnlyDoctored(FIRST_CELL),
        expect: QualityExpect::Refused("quality_cross_check_mismatch", FIRST_CELL, "f_avg_bits"),
        why: "the row states the same number twice in two encodings. The lock hashes BITS and a \
              decimal is a rendering, so a bits field a consumer trusts while the decimal it \
              reads says otherwise is the worst of the two to leave unheld",
    },
    QualityCase {
        label: "f_avg_decimal_doctored_bits_untouched",
        mutation: QualityMutation::FAvgDecimalOnlyDoctored(FIRST_CELL),
        expect: QualityExpect::Refused("quality_cross_check_mismatch", FIRST_CELL, "f_avg"),
        why: "THE REVERSE, and it must report `f_avg` rather than `f_avg_bits`: the recomputation \
              runs FIRST in the fixed field order, so the report names the figure a reader saw \
              rather than its encoding. Without this row the previous one could pass while the \
              value check was never reached",
    },
    QualityCase {
        label: "ece_bits_doctored",
        mutation: QualityMutation::EceBitsOnlyDoctored(TARGET_SETFIT),
        expect: QualityExpect::Refused(
            "quality_cross_check_mismatch",
            TARGET_SETFIT,
            "ece_top_label_validation_bits",
        ),
        why: "THE CALIBRATION DIAGNOSTICS ARE NOT RECOMPUTABLE — no committed file carries the \
              per-row probability vectors — but their BITS siblings are an internal consistency \
              claim, and holding what can be held is not the same as claiming the value is \
              proven. The residual disclosure says which of the two this is",
    },
    // ---- DEGENERATE MATRICES: a typed refusal, never a row of NaNs ----------------------
    QualityCase {
        label: "non_square_matrix",
        mutation: QualityMutation::NonSquareMatrix(TARGET_SETFIT),
        expect: QualityExpect::Refused("row_schema_refused", TARGET_SETFIT, "confusion_matrix"),
        why: "serde accepts any Vec<Vec<u64>>, so the shape is not a schema property the parse \
              can hold. The alternative to this refusal is a metric computed over a ragged \
              tally, or a panic on an index",
    },
    QualityCase {
        label: "four_by_four_against_three_labels",
        mutation: QualityMutation::DimensionMismatch(TARGET_SETFIT),
        expect: QualityExpect::Refused("row_schema_refused", TARGET_SETFIT, "confusion_matrix"),
        why: "a square matrix can still disagree with the LABEL MAP, and then index i of one is \
              not index i of the other — every per-class number would be published under another \
              class's name",
    },
    QualityCase {
        label: "all_zero_matrix",
        mutation: QualityMutation::AllZeroMatrix(TARGET_SETFIT),
        expect: QualityExpect::Refused("row_schema_refused", TARGET_SETFIT, "confusion_matrix"),
        why: "EVERY metric divides by this total. The alternative to refusing is a row of NaNs, \
              and serde_json renders a NaN as `null` — which a reader takes for a MISSING cell \
              rather than a visible failure",
    },
];

/// Apply one quality mutation to a freshly built valid run, resealing the row.
///
/// `reseal_row` recomputes the row's own envelope digest and `manifest_for` then records the new
/// digest, so all three of spot-check D's repairs happen for every row of the table. That is
/// deliberate: a mutation that left any digest stale would be refused at step 4 as
/// `row_digest_mismatch` and would prove nothing about step 6.
fn apply_quality_mutation(root: &Path, mutation: QualityMutation) {
    match mutation {
        QualityMutation::None => {}
        QualityMutation::FAvgTo099(cell) => reseal_row(root, cell, |payload| {
            payload.quality.f_avg = 0.99;
            payload.quality.f_avg_bits = 0.99_f64.to_bits();
        }),
        QualityMutation::MacroF1Doctored(cell) => reseal_row(root, cell, |payload| {
            payload.quality.macro_f1 = 0.91;
            payload.quality.macro_f1_bits = 0.91_f64.to_bits();
        }),
        QualityMutation::MccDoctored(cell) => reseal_row(root, cell, |payload| {
            payload.quality.mcc = 0.87;
            payload.quality.mcc_bits = 0.87_f64.to_bits();
        }),
        QualityMutation::PerClassF1ElementDoctored(cell) => reseal_row(root, cell, |payload| {
            payload.quality.per_class_f1[2] = 0.95;
        }),
        QualityMutation::NTestRowsDoctored(cell) => reseal_row(root, cell, |payload| {
            payload.quality.n_test_rows += 1_000;
        }),
        QualityMutation::FAvgBitsOnlyDoctored(cell) => reseal_row(root, cell, |payload| {
            payload.quality.f_avg_bits ^= 1;
        }),
        QualityMutation::FAvgDecimalOnlyDoctored(cell) => reseal_row(root, cell, |payload| {
            payload.quality.f_avg = f64::from_bits(payload.quality.f_avg_bits ^ 1);
        }),
        QualityMutation::EceBitsOnlyDoctored(cell) => reseal_row(root, cell, |payload| {
            payload.quality.ece_top_label_validation_bits ^= 1;
        }),
        QualityMutation::NonSquareMatrix(cell) => reseal_row(root, cell, |payload| {
            payload.quality.confusion_matrix = vec![vec![1, 2, 3], vec![4, 5, 6]];
        }),
        QualityMutation::DimensionMismatch(cell) => reseal_row(root, cell, |payload| {
            payload.quality.confusion_matrix =
                vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8], vec![9, 10, 11, 12], vec![13, 14, 15, 16]];
        }),
        QualityMutation::AllZeroMatrix(cell) => reseal_row(root, cell, |payload| {
            payload.quality.confusion_matrix = vec![vec![0; 3]; 3];
        }),
    }
}

#[test]
fn bench_gate_quality_cross_check_case_table_at_the_active_scope() {
    let mut observations: Vec<(&str, String)> = Vec::new();

    for case in &QUALITY_CROSS_CHECK_CASES {
        let dir = write_valid_run(RunSpec::default());
        apply_quality_mutation(dir.path(), case.mutation);
        // AFTER the mutation, so the resealed row's NEW digest is what the manifest records —
        // spot-check D's third repair, without which every row would be refused at step 4.
        let manifest = manifest_for(dir.path());

        let (observed, named_cell, rendered) = match verify_run(&manifest, dir.path()) {
            Ok(set) => (format!("Ok({} rows verified)", set.len()), String::new(), String::new()),
            Err(error) => (
                error.variant_tag().to_string(),
                error.cell().unwrap_or("<none>").to_string(),
                error.to_string(),
            ),
        };
        println!(
            "[bench_gate] QUALITY_CROSS_CHECK case={} observed={observed} cell={named_cell} ({})",
            case.label, case.why
        );

        let expected = match case.expect {
            QualityExpect::Accepted => "Ok(40 rows verified)".to_string(),
            QualityExpect::Refused(tag, cell, field) => {
                if observed == tag {
                    assert_eq!(
                        named_cell,
                        cell.render(),
                        "[{}] the refusal must name {}: {rendered}",
                        case.label,
                        cell.render()
                    );
                    // The FIELD is what tells an operator which published figure to look at.
                    // A refusal that named only the cell would leave seventeen candidates.
                    assert!(
                        rendered.contains(field),
                        "[{}] the refusal must name the field `{field}`: {rendered}",
                        case.label
                    );
                }
                tag.to_string()
            }
        };
        observations.push((case.label, format!("{observed}|expected={expected}")));
    }

    let failures: Vec<&(&str, String)> = observations
        .iter()
        .filter(|(_, line)| {
            let (observed, expected) = line.split_once("|expected=").unwrap_or((line, ""));
            observed != expected
        })
        .collect();
    assert!(
        failures.is_empty(),
        "every published quality figure must be recomputable from the row's own confusion \
         matrix through verify_run at the ACTIVE 40-cell scope, and the control must be \
         ACCEPTED; these were not: {failures:#?}"
    );

    // THE SWEPT-ROW COUNT IS PINNED HERE, as 05-16 established, because `assert_tests_ran`
    // counts test FUNCTIONS and this table is ONE function sweeping twelve rows — twelve rows
    // silently becoming three would not move the Make floor by a single test.
    assert_eq!(
        observations.len(),
        QUALITY_CROSS_CHECK_CASES.len(),
        "every row of the table must have been swept"
    );
    assert_eq!(QUALITY_CROSS_CHECK_CASES.len(), 12, "eleven mutations plus one acceptance row");
    assert_eq!(
        QUALITY_CROSS_CHECK_CASES
            .iter()
            .filter(|case| case.expect == QualityExpect::Accepted)
            .count(),
        1,
        "a refusal table with no acceptance row can pass by refusing everything"
    );
}

#[test]
fn bench_gate_the_single_cell_door_also_applies_the_quality_cross_check() {
    // SCOPE IS PART OF WHAT A NEGATIVE PROVES (CLAUDE.md rule 4). The table above holds the
    // cross-check at `verify_run` / ACTIVE 40; this holds it at the OTHER shipped door, which
    // `apr setfit bench verify-cell` drives and whose own printed enumeration must say so.
    let dir = write_valid_run(RunSpec::default());
    apply_quality_mutation(dir.path(), QualityMutation::FAvgTo099(TARGET_SETFIT));
    let manifest = manifest_for(dir.path());

    let error = verify_cell(&manifest, dir.path(), TARGET_SETFIT)
        .expect_err("spot-check D through the single-cell door");
    assert_eq!(error.variant_tag(), "quality_cross_check_mismatch", "{error}");
    let expected_cell = TARGET_SETFIT.render();
    assert_eq!(error.cell(), Some(expected_cell.as_str()), "{error}");

    // NON-VACUITY: the door must still PASS on the untouched cell, or the assertion above
    // would hold for a door that refused everything.
    let clean = write_valid_run(RunSpec::default());
    let clean_manifest = manifest_for(clean.path());
    verify_cell(&clean_manifest, clean.path(), TARGET_SETFIT)
        .expect("the untouched cell verifies through the single-cell door");
}

#[test]
fn bench_gate_the_deferred_scope_pairing_negative_still_reports_unpaired_selection() {
    // THE FENCE BETWEEN THIS PLAN AND D-ITEM-05-15, asserted rather than reasoned about.
    // The selection binding runs in STEP 6; `verify_pairing` is step 5 and is byte-unchanged.
    // Had the binding been wired earlier it would PRE-EMPT this negative and silently change
    // which variant the cross-method check is observed through — the negative would still be
    // green and would have stopped testing what it names.
    let dir = write_valid_run_deferred(RunSpec::default());
    reseal_row(dir.path(), TARGET_LORA, |payload| {
        payload.selection_manifest_hash = "a-different-draw-entirely".to_string();
    });
    let manifest = manifest_for(dir.path());

    let error = refuse(&manifest, dir.path(), "an unpaired pair is still an unpaired pair");
    assert_eq!(
        error.variant_tag(),
        "unpaired_selection",
        "step 5 runs before step 6, so the cross-method rule is what must fire here — not one \
         of 05-16's intra-cell tags: {error}"
    );
}

// ===========================================================================================
// The six negatives are SIX DISTINCT variants
// ===========================================================================================

/// Every doctored shape, each returning its own refusal, collected in one place.
///
/// The assertion is that six mutations produce six DISTINCT variant tags. Six refusals that all
/// came back as one catch-all variant would pass six individual `expect_err`s while telling a
/// reader nothing about WHICH dishonesty was found — and a missing cell, a tampered row and an
/// unpaired comparison are different defects with the same symptom.
#[test]
fn bench_gate_the_six_doctored_negatives_are_six_distinct_variants() {
    let mut tags: Vec<&'static str> = Vec::new();

    // 1. omission
    {
        let dir = write_valid_run_deferred(RunSpec::default());
        fs::remove_file(dir.path().join(ROWS_DIR).join(row_file_name(TARGET_SETFIT)))
            .expect("remove");
        let manifest = manifest_for(dir.path());
        tags.push(refuse(&manifest, dir.path(), "1").variant_tag());
    }
    // 2. trimmed block
    {
        let dir = write_valid_run_deferred(RunSpec::default());
        let mut value = read_row_value(dir.path(), TARGET_SETFIT);
        value
            .get_mut("payload")
            .and_then(|p| p.get_mut("evidence"))
            .and_then(|e| e.get_mut("setfit"))
            .and_then(serde_json::Value::as_object_mut)
            .expect("setfit block")
            .remove("lock");
        write_row_value(dir.path(), TARGET_SETFIT, &value);
        let manifest = manifest_for(dir.path());
        tags.push(refuse(&manifest, dir.path(), "2").variant_tag());
    }
    // 3. unpaired
    {
        let dir = write_valid_run_deferred(RunSpec::default());
        reseal_row(dir.path(), TARGET_LORA, |p| {
            p.selection_manifest_hash = "another-draw".to_string();
        });
        let manifest = manifest_for(dir.path());
        tags.push(refuse(&manifest, dir.path(), "3").variant_tag());
    }
    // 4. edited payload bytes
    {
        let dir = write_valid_run_deferred(RunSpec::default());
        let mut value = read_row_value(dir.path(), TARGET_SETFIT);
        *value
            .get_mut("payload")
            .and_then(|p| p.get_mut("dataset_fingerprint"))
            .expect("fingerprint") = serde_json::Value::String("0".repeat(64));
        write_row_value(dir.path(), TARGET_SETFIT, &value);
        let manifest = manifest_for(dir.path());
        tags.push(refuse(&manifest, dir.path(), "4").variant_tag());
    }
    // 5. post-test selection
    {
        let dir = write_valid_run_deferred(RunSpec::default());
        reseal_row(dir.path(), TARGET_LORA, |p| {
            if let MethodEvidence::Lora(e) = &mut p.evidence {
                e.epochs_completed = 1;
            }
        });
        let manifest = manifest_for(dir.path());
        tags.push(refuse(&manifest, dir.path(), "5").variant_tag());
    }
    // 6. forged provenance
    {
        let dir = write_valid_run_deferred(RunSpec::default());
        let lock_path = dir.path().join(format!(
            "{LOCKS_DIR}/{}-s{}-seed{}.lock.json",
            TARGET_SETFIT.method.tag(),
            TARGET_SETFIT.shots,
            TARGET_SETFIT.seed
        ));
        fs::write(&lock_path, b"{\"schema\":\"selection-lock-v1\",\"edited\":true}")
            .expect("lock write");
        let manifest = manifest_for(dir.path());
        tags.push(refuse(&manifest, dir.path(), "6").variant_tag());
    }

    assert_eq!(tags.len(), 6, "six doctored shapes");
    let distinct: BTreeSet<&&str> = tags.iter().collect();
    assert_eq!(
        distinct.len(),
        6,
        "six doctored shapes must produce SIX DISTINCT refusals, got {tags:?}"
    );

    // PRINTED SO A CLOSING AUDIT CAN READ THE COUNT OFF A LOG RATHER THAN QUOTE IT FROM A
    // PLAN. A number an auditor copies out of a document is a number nobody measured; under
    // `--nocapture` this line makes the tally an observation. Visible with
    // `cargo test -p aprender-train --lib --features setfit bench_gate -- --nocapture`.
    println!(
        "[bench_gate] DOCTORED_NEGATIVES={} DISTINCT_REFUSAL_VARIANTS={} TAGS={:?}",
        tags.len(),
        distinct.len(),
        tags
    );
}

// ===========================================================================================
// The two vacuity backstops, and the manifest's own digest
// ===========================================================================================

#[test]
fn bench_gate_refuses_a_zero_cell_manifest_before_reading_any_row() {
    let dir = write_valid_run(RunSpec::default());
    let mut manifest = manifest_for(dir.path());
    manifest.payload.cells.clear();
    // Reseal by hand so the digest check does not fire first and mask the backstop.
    manifest.semantic_hash =
        sha256_hex(&manifest.payload.to_canonical_bytes().expect("payload serializes"));

    let error = refuse(&manifest, dir.path(), "zero cells is a failure");
    assert_eq!(error.variant_tag(), "empty_expectation_set");
}

#[test]
fn bench_gate_refuses_a_manifest_whose_expectation_set_is_not_the_contracted_forty() {
    // THE CONTRACT'S OWN NAMED ADVERSARY: a producer declaring a twelve-cell expectation,
    // satisfying it completely, and publishing a "complete" run. RE-MUTATED at the ACTIVE
    // scope — the 80-cell proof does NOT transfer (CLAUDE.md Verification Discipline rule 4),
    // because the set this backstop compares against is the thing that changed.
    let dir = write_valid_run(RunSpec::default());
    let mut manifest = manifest_for(dir.path());
    manifest.payload.cells.truncate(12);
    manifest.semantic_hash =
        sha256_hex(&manifest.payload.to_canonical_bytes().expect("payload serializes"));

    let error = refuse(&manifest, dir.path(), "a 12-cell expectation is a failure");
    assert_eq!(error.variant_tag(), "expectation_set_mismatch");
    let rendered = error.to_string();
    assert!(rendered.contains("12"), "{rendered}");
    assert!(rendered.contains("40"), "{rendered}");
}

#[test]
fn bench_gate_refuses_a_manifest_declaring_a_second_methods_cell_before_reading_a_row() {
    // ACTIVE-SCOPE OUT-OF-SCOPE NEGATIVE #1, path one of two.
    //
    // A manifest that DECLARES a cell outside the active scope is refused at STEP 2 — before
    // any row byte is read — by the expectation-set backstop that already existed. No new
    // variant is minted: a third variant would be reachable only from a test-only constructor,
    // which is a guard over a path production cannot take.
    let dir = write_valid_run(RunSpec::default());
    let mut manifest = manifest_for(dir.path());
    manifest.payload.cells.push(crate::train::setfit::bench_row::CellEntry {
        method: Method::Lora,
        shots: 16,
        seed: 29,
        status: CellStatus::Complete,
        row_sha256: Some("c".repeat(64)),
    });
    manifest.semantic_hash =
        sha256_hex(&manifest.payload.to_canonical_bytes().expect("payload serializes"));

    let error = refuse(&manifest, dir.path(), "a second method's declared cell is a refusal");
    assert_eq!(
        error.variant_tag(),
        "expectation_set_mismatch",
        "the EXISTING step-2 variant, reused rather than replaced: {error}",
    );
    // And it fired BEFORE any row was read — there is no row file for that cell at all, so a
    // gate that got as far as the row loop would have reported a missing file instead.
    assert!(
        !dir.path().join(ROWS_DIR).join(row_file_name(TARGET_LORA)).exists(),
        "the negative only proves step 2 ran first if no such row exists to be read",
    );
}

#[test]
fn bench_gate_refuses_a_second_methods_row_placed_in_a_declared_setfit_slot() {
    // ACTIVE-SCOPE OUT-OF-SCOPE NEGATIVE #2, path two of two.
    //
    // The other route an out-of-scope cell can take through a production door: the manifest is
    // untouched and correct, but a second method's ROW is filed in a declared SetFit slot. That
    // is refused at STEP 4 by the slot-agreement check — again an EXISTING variant.
    //
    // Building the row at all is what BENCH_METHODS keeps possible: it is still the
    // ROW-VALIDITY domain and still carries both methods. Had the narrowing landed on it
    // instead of on ACTIVE_METHODS, this negative could not be constructed.
    let deferred = write_valid_run_deferred(RunSpec::default());
    let lora_bytes =
        fs::read(deferred.path().join(ROWS_DIR).join(row_file_name(TARGET_LORA))).expect("read");

    let dir = write_valid_run(RunSpec::default());
    fs::write(dir.path().join(ROWS_DIR).join(row_file_name(TARGET_SETFIT)), &lora_bytes)
        .expect("write");
    let manifest = manifest_for(dir.path());

    let error = refuse(&manifest, dir.path(), "a second method's row in a SetFit slot");
    assert_eq!(
        error.variant_tag(),
        "row_slot_mismatch",
        "the EXISTING step-4 variant, reused rather than replaced: {error}",
    );
    assert!(error.to_string().contains(&TARGET_LORA.render()), "{error}");
}

#[test]
fn bench_gate_refuses_a_manifest_whose_own_digest_disagrees_with_its_payload() {
    let dir = write_valid_run(RunSpec::default());
    let mut manifest = manifest_for(dir.path());
    manifest.semantic_hash = "0".repeat(64);

    let error = refuse(&manifest, dir.path(), "a doctored manifest is a refusal");
    assert_eq!(error.variant_tag(), "manifest_digest_mismatch");
}

#[test]
fn bench_gate_refuses_a_row_substituted_for_the_one_the_manifest_recorded() {
    let dir = write_valid_run(RunSpec::default());
    // Manifest FIRST, then a resealed replacement: the row is internally perfect and its cell
    // is right, but it is not the run that produced the recorded number.
    let manifest = manifest_for(dir.path());
    reseal_row(dir.path(), TARGET_SETFIT, |payload| {
        payload.quality.f_avg = 0.99;
        payload.quality.f_avg_bits = 0.99_f64.to_bits();
    });

    let error = refuse(&manifest, dir.path(), "a substitution is a refusal");
    assert_eq!(error.variant_tag(), "row_manifest_digest_mismatch");
}

#[test]
fn bench_gate_refuses_a_row_filed_under_the_wrong_slot() {
    let dir = write_valid_run(RunSpec::default());
    // Put cell (setfit, 16, 31)'s payload in (setfit, 16, 29)'s file, resealed so it is
    // internally consistent — a relabelling that only the slot comparison can see.
    let other = CellKey::new(Method::Setfit, 16, 31);
    let bytes = fs::read(dir.path().join(ROWS_DIR).join(row_file_name(other))).expect("read");
    fs::write(dir.path().join(ROWS_DIR).join(row_file_name(TARGET_SETFIT)), &bytes).expect("write");
    let manifest = manifest_for(dir.path());

    let error = refuse(&manifest, dir.path(), "a mis-slotted row is a refusal");
    assert_eq!(error.variant_tag(), "row_slot_mismatch");
    assert!(error.to_string().contains(&other.render()), "{error}");
}

// ===========================================================================================
// Aggregation: determinism, ordering, degeneracy
// ===========================================================================================

/// Every `f64` reachable in a serialized value, in document order.
fn collect_f64s(value: &serde_json::Value, out: &mut Vec<f64>) {
    match value {
        serde_json::Value::Number(number) => {
            if let Some(f) = number.as_f64() {
                out.push(f);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                collect_f64s(item, out);
            }
        }
        serde_json::Value::Object(map) => {
            for (_, item) in map {
                collect_f64s(item, out);
            }
        }
        _ => {}
    }
}

/// Whether any `null` occurs anywhere in a serialized value.
///
/// This is the CHECK for "no non-finite f64 was serialized", and it is a structural walk rather
/// than a substring grep. `serde_json` renders `NaN` and `±inf` as `null`, and a grep for the
/// strings "NaN"/"Infinity" would find neither of them while colliding with the `null_reason`
/// KEY this module deliberately emits.
fn contains_null(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Null => true,
        serde_json::Value::Array(items) => items.iter().any(contains_null),
        serde_json::Value::Object(map) => map.values().any(contains_null),
        _ => false,
    }
}

#[test]
fn bench_gate_two_aggregations_of_one_run_are_bit_identical() {
    let dir = write_valid_run(RunSpec::default());
    let manifest = manifest_for(dir.path());
    let verified = verify_run(&manifest, dir.path()).expect("verifies");

    let first = serde_json::to_value(aggregate(&verified)).expect("serializes");
    let second = serde_json::to_value(aggregate(&verified)).expect("serializes");

    let (mut a, mut b) = (Vec::new(), Vec::new());
    collect_f64s(&first, &mut a);
    collect_f64s(&second, &mut b);
    assert!(!a.is_empty(), "the determinism check must have f64s to compare");
    assert_eq!(a.len(), b.len());
    for (index, (left, right)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            left.to_bits(),
            right.to_bits(),
            "f64 #{index} differs by BITS between two aggregations of the same rows"
        );
    }
    assert_eq!(first, second, "the whole document must be identical");
}

#[test]
fn bench_gate_aggregate_emits_the_pinned_key_sequence() {
    let dir = write_valid_run(RunSpec::default());
    let manifest = manifest_for(dir.path());
    let verified = verify_run(&manifest, dir.path()).expect("verifies");
    let report = aggregate(&verified);

    assert_eq!(report.key_sequence.len(), EXPECTED_CELLS);
    assert_eq!(
        &report.key_sequence[..5],
        &[
            "setfit/s8/seed13",
            "setfit/s8/seed17",
            "setfit/s8/seed23",
            "setfit/s8/seed29",
            "setfit/s8/seed31",
        ]
    );
    // THE TAIL MOVED WITH THE SCOPE. It read `lora/s64/*` under the 80-cell expectation; under
    // the ACTIVE 40-cell one the last group is SetFit's s64. A tail still reading `lora/...`
    // here after the narrowing would mean the aggregate had not followed the contract.
    assert_eq!(
        &report.key_sequence[EXPECTED_CELLS - 5..],
        &[
            "setfit/s64/seed37",
            "setfit/s64/seed41",
            "setfit/s64/seed43",
            "setfit/s64/seed47",
            "setfit/s64/seed53",
        ]
    );
    // Uniqueness, so the "deterministic order" claim cannot be satisfied by a sequence with a
    // repeated key that happens to sort stably.
    let distinct: BTreeSet<&String> = report.key_sequence.iter().collect();
    assert_eq!(distinct.len(), EXPECTED_CELLS);
}

#[test]
fn bench_gate_aggregate_recomputes_the_closed_form_summary_from_the_rows() {
    let dir = write_valid_run(RunSpec::default());
    let manifest = manifest_for(dir.path());
    let verified = verify_run(&manifest, dir.path()).expect("verifies");
    let report = aggregate(&verified);

    assert_eq!(report.quality.len(), 4, "ONE active method x four shot levels");
    assert_eq!(report.methods, vec!["setfit".to_string()], "the active scope measured one method");
    assert!(
        report.deltas.is_empty(),
        "a delta needs two arms; under the active scope there is nothing to difference, and an \
         EMPTY delta list is the point — a delta computed over an absent arm and reported with \
         a null reason would be a comparison section wearing a caveat",
    );
    assert_eq!(report.n_seeds, PAIRED_DESIGN_N);
    assert_eq!(report.degrees_of_freedom, 9);

    // UNCERTAINTY SURVIVES THE DESCOPE. Dropping the paired delta must not drop the second
    // half of EVAL-04: every active group carries a seed-dispersion interval on the frozen t.
    for group in &report.quality {
        assert!(
            group.f_avg_seed_ci95.is_present(),
            "group {:?}/s{} has no seed-dispersion interval; a single-method report that \
             quoted only a mean and a std would have silently dropped EVAL-04's uncertainty \
             clause under cover of a scope amendment",
            group.method,
            group.shots,
        );
    }

    let group = report
        .quality
        .iter()
        .find(|g| g.method == Method::Setfit && g.shots == 16)
        .expect("the group exists");
    assert_eq!(group.f_avg.n, PAIRED_DESIGN_N);
    assert_eq!(group.per_seed.len(), PAIRED_DESIGN_N);

    // RECOMPUTED BY HAND from the fixture's own generator, which is the property EVAL-04 asks
    // for: the published number must be derivable from the stored rows alone.
    let expected: Vec<f64> = BENCH_SEEDS
        .iter()
        .map(|seed| synthetic_f_avg(RunSpec::default(), CellKey::new(Method::Setfit, 16, *seed)))
        .collect();
    let mean = aprender::stats::hypothesis::mean_f64(&expected).expect("ten values");
    let std = aprender::stats::hypothesis::sample_std_f64(&expected).expect("ten values");
    assert_eq!(group.f_avg.mean.to_bits(), mean.to_bits());
    assert_eq!(group.f_avg.std.to_bits(), std.to_bits());
    assert_eq!(group.f_avg.min.to_bits(), expected[0].to_bits());
    assert_eq!(group.f_avg.max.to_bits(), expected[9].to_bits());

    // THE ACTIVE-SCOPE INTERVAL, recomputed by hand from the same ten stored values — so the
    // uncertainty a reader sees is derivable from the rows alone, exactly as the mean is.
    let ci = aprender::stats::hypothesis::ci95_one_sample_df9(&expected).expect("ten values");
    assert_eq!(group.f_avg_seed_ci95.low.expect("low").to_bits(), ci.low.to_bits());
    assert_eq!(group.f_avg_seed_ci95.high.expect("high").to_bits(), ci.high.to_bits());
    assert_eq!(
        group.f_avg_seed_ci95.half_width.expect("half width").to_bits(),
        (T_CRIT_975_DF9 * std / (PAIRED_DESIGN_N as f64).sqrt()).to_bits(),
        "the half width is the FROZEN t times the standard error, bit for bit",
    );
}

#[test]
fn bench_gate_deferred_scope_still_computes_the_paired_delta() {
    // THE PAIRED MACHINERY IS RETAINED AND UNEXERCISED, not deleted (D-19, D-ITEM-05-15).
    // These assertions were the ACTIVE-scope ones before the narrowing; they are RE-SITED here
    // rather than dropped, so the delta path keeps a running proof and D-ITEM-05-15 restores
    // an arm that still works instead of one nobody has executed since.
    let dir = write_valid_run_deferred(RunSpec::default());
    let manifest = manifest_for(dir.path());
    let verified = verify_scoped(&manifest, dir.path()).expect("verifies");
    let report = aggregate(&verified);

    assert_eq!(report.quality.len(), 8, "two methods x four shot levels");
    assert_eq!(report.deltas.len(), 4, "one paired comparison per shot level");
    assert_eq!(report.methods, vec!["setfit".to_string(), "lora".to_string()]);

    let delta = report.deltas.iter().find(|d| d.shots == 16).expect("the shot level exists");
    assert!(delta.ci95.is_present(), "a varying delta set has an interval");
    assert!(delta.ci95.null_reason.is_none());
    assert!(delta.p_value.is_some(), "p-values live in the detail (D-08)");
    assert_eq!(delta.per_seed_deltas.len(), PAIRED_DESIGN_N);
}

#[test]
fn bench_gate_a_zero_variance_delta_set_reports_a_point_estimate_and_no_interval() {
    let dir = write_valid_run_deferred(RunSpec { zero_variance_shots: Some(8) });
    let manifest = manifest_for(dir.path());
    let verified = verify_scoped(&manifest, dir.path()).expect("verifies");
    let report = aggregate(&verified);

    let degenerate = report.deltas.iter().find(|d| d.shots == 8).expect("the shot level exists");
    assert!(
        !degenerate.ci95.is_present(),
        "an interval over identical differences has no finite width"
    );
    assert_eq!(degenerate.ci95.null_reason.as_deref(), Some(ZERO_VARIANCE_NULL_REASON));
    assert!(degenerate.std_delta.is_none());
    // THE POINT ESTIMATE SURVIVES. It is well defined and it is what a reader wants; only the
    // interval is absent, and it is absent WITH A REASON.
    //
    // RE-DERIVED FROM THE FIXTURE, not a literal. Before 05-17 this line read
    // `0.25_f64.to_bits()`, which was the difference between the two hand-written constants the
    // degenerate branch of the old `synthetic_f_avg` returned. The quality block is now a closed
    // form over its own confusion matrix, so the degenerate delta is whatever those two matrices
    // produce — and pinning a stale literal here would have been the same defect this plan is
    // closing elsewhere: a published number that no longer follows from the evidence beside it.
    // The PROPERTY under test is unchanged and is asserted below: identical across all ten
    // seeds, bit for bit.
    let degenerate_spec = RunSpec { zero_variance_shots: Some(8) };
    let expected_delta = synthetic_f_avg(degenerate_spec, CellKey::new(Method::Setfit, 8, 13))
        - synthetic_f_avg(degenerate_spec, CellKey::new(Method::Lora, 8, 13));
    assert_eq!(degenerate.mean_delta.to_bits(), expected_delta.to_bits());
    for seed in BENCH_SEEDS {
        let per_seed = synthetic_f_avg(degenerate_spec, CellKey::new(Method::Setfit, 8, seed))
            - synthetic_f_avg(degenerate_spec, CellKey::new(Method::Lora, 8, seed));
        assert_eq!(
            per_seed.to_bits(),
            expected_delta.to_bits(),
            "seed {seed}'s paired delta at the degenerate shot level must be BIT-identical to \
             every other seed's, or the branch under test is not reached for the reason its \
             name claims"
        );
    }
    // The non-degenerate levels are unaffected, so the branch is not a global switch.
    let ordinary = report.deltas.iter().find(|d| d.shots == 16).expect("the shot level exists");
    assert!(ordinary.ci95.is_present());

    // THE SERIALIZED SHAPE. Parsed, never grepped: a substring search for "NaN"/"Infinity"
    // would find neither (serde renders them as `null`) and would collide with the
    // `null_reason` key this module deliberately emits.
    let value = serde_json::to_value(&report).expect("serializes");
    assert!(
        !contains_null(&value),
        "no `null` may appear anywhere in the aggregate — a non-finite f64 serializes as null \
         and reads as a MISSING measurement rather than a degenerate one"
    );
    let mut floats = Vec::new();
    collect_f64s(&value, &mut floats);
    assert!(!floats.is_empty());
    for (index, float) in floats.iter().enumerate() {
        assert!(float.is_finite(), "f64 #{index} is not finite: {float}");
    }
    let rendered = serde_json::to_string(&value).expect("renders");
    assert!(rendered.contains("\"null_reason\":\"zero_variance\""), "{rendered}");
}

#[test]
fn bench_gate_the_synthetic_fixture_keeps_its_two_dispersion_properties() {
    // 05-17. `synthetic_quality` became a closed form over `synthetic_confusion`, and two
    // properties of the OLD arithmetic generator are load-bearing for tests that predate this
    // plan. They are asserted here rather than argued in a comment, because a matrix edited in
    // a later round could break either of them while every existing test stayed green for the
    // wrong reason: the interval tests would silently exercise the zero-variance branch, and
    // the zero-variance test would silently exercise the interval one.

    // PROPERTY 1 — a non-degenerate shot level gives ten PAIRWISE DISTINCT f_avg values, per
    // method, so `seed_dispersion_ci95` has a non-degenerate sample and the interval arm runs.
    for method in BENCH_METHODS {
        for shots in BENCH_SHOTS {
            let values: Vec<u64> = BENCH_SEEDS
                .iter()
                .map(|seed| {
                    synthetic_f_avg(RunSpec::default(), CellKey::new(method, shots, *seed))
                        .to_bits()
                })
                .collect();
            let mut unique = values.clone();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(
                unique.len(),
                BENCH_SEEDS.len(),
                "{method:?}/s{shots}: the ten synthetic f_avg values must be pairwise distinct; \
                 got {values:?}"
            );
            // And MONOTONE INCREASING, which is the stronger statement the aggregate test
            // depends on when it asserts `min == expected[0]` and `max == expected[9]`.
            let mut ascending = values.clone();
            ascending.sort_unstable();
            assert_eq!(
                values, ascending,
                "{method:?}/s{shots}: the ten values must ascend with the seed index"
            );
        }
    }

    // PROPERTY 2 — at `zero_variance_shots` the ten seeds are BIT-IDENTICAL per method, which
    // is what `paired_ci95_df9` refuses with `ZeroVarianceDifferences`.
    let spec = RunSpec { zero_variance_shots: Some(8) };
    for method in BENCH_METHODS {
        let first = synthetic_f_avg(spec, CellKey::new(method, 8, BENCH_SEEDS[0])).to_bits();
        for seed in BENCH_SEEDS {
            assert_eq!(
                synthetic_f_avg(spec, CellKey::new(method, 8, seed)).to_bits(),
                first,
                "{method:?}/s8/seed{seed} must be bit-identical to seed {} at the degenerate \
                 shot level",
                BENCH_SEEDS[0],
            );
        }
        // NON-VACUITY: the degenerate pin must apply to THAT shot level only, or the fixture
        // would be globally degenerate and property 1 above would be testing nothing.
        let other: Vec<u64> = BENCH_SEEDS
            .iter()
            .map(|seed| synthetic_f_avg(spec, CellKey::new(method, 16, *seed)).to_bits())
            .collect();
        assert!(
            other.iter().any(|bits| *bits != other[0]),
            "{method:?}: s16 must still vary across seeds when s8 is pinned"
        );
    }

    // PROPERTY 3 — every synthetic block's `n_test_rows` is its own matrix's total. The
    // pre-05-17 fixture said 35 against a matrix totalling 36; the cross-check this plan adds
    // would have refused every synthetic row for the fixture's defect rather than the gate's.
    for method in BENCH_METHODS {
        for shots in BENCH_SHOTS {
            for seed in BENCH_SEEDS {
                let cell = CellKey::new(method, shots, seed);
                let quality = synthetic_quality(RunSpec::default(), cell);
                let total: u64 = quality.confusion_matrix.iter().flatten().sum();
                assert_eq!(
                    quality.n_test_rows,
                    total,
                    "{} publishes n_test_rows={} beside a matrix totalling {total}",
                    cell.render(),
                    quality.n_test_rows,
                );
            }
        }
    }
}

#[test]
fn bench_gate_resource_groups_carry_their_hosts_and_mechanism_classes() {
    let dir = write_valid_run_deferred(RunSpec::default());
    let manifest = manifest_for(dir.path());
    let verified = verify_scoped(&manifest, dir.path()).expect("verifies");
    let report = aggregate(&verified);

    let setfit = report
        .resource
        .iter()
        .find(|r| r.method == Method::Setfit && r.shots == 16)
        .expect("group");
    let lora =
        report.resource.iter().find(|r| r.method == Method::Lora && r.shots == 16).expect("group");

    assert_eq!(setfit.hosts, vec!["local-cpu (macos/aarch64)".to_string()]);
    assert_eq!(lora.hosts, vec!["lambda-vector (linux/x86_64)".to_string()]);
    assert_ne!(
        setfit.hosts, lora.hosts,
        "D-09 puts the two methods on two hosts; the aggregate must keep them apart"
    );

    // The mixed-mechanism case, which is what the renderer has to label.
    assert_eq!(
        setfit.train_peak_rss_mechanism_classes,
        vec![MechanismClass::ExactKernelHighWaterMark]
    );
    assert_eq!(lora.train_peak_rss_mechanism_classes, vec![MechanismClass::SampledLowerBound]);
    assert!(!mechanisms_are_comparable(
        &setfit.train_peak_rss_mechanisms[0],
        &lora.train_peak_rss_mechanisms[0]
    ));
    // ... and the SAME-mechanism control, so the predicate is not "always false".
    assert!(mechanisms_are_comparable(
        &setfit.inference_peak_rss_mechanisms[0],
        &lora.inference_peak_rss_mechanisms[0]
    ));

    // The size split: adapter-only bytes and deployable bytes differ by the base model.
    assert!(lora.deployable_total_bytes.mean > lora.artifact_bytes.mean);
    assert_eq!(
        setfit.deployable_total_bytes.mean.to_bits(),
        setfit.artifact_bytes.mean.to_bits(),
        "SetFit ships ONE standalone file, so the two figures are equal by construction"
    );
}

#[test]
fn bench_gate_mechanism_class_case_table() {
    // MUST-MATCH.
    assert_eq!(
        mechanism_class(MECHANISM_CHILD_MAX_RSS_TIME_L),
        MechanismClass::ExactKernelHighWaterMark
    );
    assert_eq!(
        mechanism_class(MECHANISM_CHILD_MAX_RSS_VM_HWM),
        MechanismClass::ExactKernelHighWaterMark
    );
    assert_eq!(mechanism_class("vm_hwm"), MechanismClass::ExactKernelHighWaterMark);
    assert_eq!(mechanism_class("sysinfo_sampled_10hz"), MechanismClass::SampledLowerBound);
    assert_eq!(mechanism_class("sysinfo_sampled_2hz"), MechanismClass::SampledLowerBound);
    // MUST-NOT-MATCH — the NEIGHBOURS of the wanted strings, which is where a loose pattern
    // actually goes wrong, rather than obviously unrelated text.
    assert_eq!(mechanism_class("vm_hwm_child"), MechanismClass::Unrecognised);
    assert_eq!(mechanism_class("sysinfo"), MechanismClass::Unrecognised);
    assert_eq!(mechanism_class("max_rss"), MechanismClass::Unrecognised);
    assert_eq!(mechanism_class(""), MechanismClass::Unrecognised);
    // An unrecognised mechanism is never comparable, INCLUDING to itself: an unknown boundary
    // is not evidence that two numbers mean the same thing.
    assert!(!mechanisms_are_comparable("mystery", "mystery"));
}

// ===========================================================================================
// Contract cross-pins and structural guards
// ===========================================================================================

/// Just enough of the claims contract to reach the frozen statistics constants.
#[derive(Debug, Deserialize)]
struct ContractFile {
    equations: ContractEquations,
}

#[derive(Debug, Deserialize)]
struct ContractEquations {
    claims_statistics: ContractClaimsStatistics,
}

#[derive(Debug, Deserialize)]
struct ContractClaimsStatistics {
    n_seeds: usize,
    degrees_of_freedom: usize,
    t_crit_975_df9: f64,
}

#[test]
fn bench_gate_frozen_t_constant_matches_the_contract_by_bits() {
    let parsed: ContractFile =
        serde_yaml::from_str(CLAIMS_CONTRACT_YAML).expect("the claims contract parses");
    let stats = parsed.equations.claims_statistics;

    // BIT equality, not a tolerance. A constant that can drift within a tolerance is not frozen,
    // and every published interval's width is this number times a standard error.
    assert_eq!(
        stats.t_crit_975_df9.to_bits(),
        T_CRIT_975_DF9.to_bits(),
        "the contract's frozen t literal and aprender_core::stats::hypothesis::T_CRIT_975_DF9 \
         are two pinned copies of one value and they have drifted"
    );
    assert_eq!(stats.n_seeds, PAIRED_DESIGN_N);
    assert_eq!(stats.degrees_of_freedom, PAIRED_DESIGN_N - 1);

    // NON-VACUITY: a parse that silently produced 0.0 would satisfy nothing useful, and the
    // assertion above would still hold if the constant were also 0.0.
    assert!(stats.t_crit_975_df9 > 2.0 && stats.t_crit_975_df9 < 3.0);
}

#[test]
fn bench_gate_aggregate_publishes_the_frozen_t_constant_it_used() {
    let dir = write_valid_run(RunSpec::default());
    let manifest = manifest_for(dir.path());
    let verified = verify_run(&manifest, dir.path()).expect("verifies");
    let report = aggregate(&verified);
    assert_eq!(report.t_crit_975_df9.to_bits(), T_CRIT_975_DF9.to_bits());
    assert_eq!(report.contract_id, CLAIMS_CONTRACT_ID);
}

/// Non-comment source lines only, so prose can neither satisfy nor trip a structural guard.
fn non_comment_source(src: &str) -> String {
    src.lines()
        .filter(|line| {
            let trimmed = line.trim_start();
            !trimmed.starts_with("//")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn bench_gate_source_carries_no_rng_or_resampling_vocabulary() {
    let source = non_comment_source(BENCH_GATE_SOURCE);
    // NON-VACUITY FIRST: a guard whose haystack is empty passes for the wrong reason.
    assert!(
        source.len() > 5_000,
        "the no-RNG guard has nothing to scan; the comment filter has eaten the module"
    );
    for forbidden in [
        "rand::",
        "rand_chacha",
        "thread_rng",
        "StdRng",
        "bootstrap",
        "resample",
        "permutation_test",
        "shuffle",
    ] {
        assert!(
            !source.contains(forbidden),
            "`{forbidden}` appears in bench_gate's non-comment source. EVAL-04 requires the \
             published numbers to be EXACTLY recomputable from the rows; a resampling stream \
             downgrades `exactly` to `up to an RNG seed` (D-06)"
        );
    }
}

#[test]
fn bench_gate_verified_run_set_has_no_public_constructor() {
    let source = non_comment_source(BENCH_GATE_SOURCE);
    assert!(
        source.contains("pub struct VerifiedRunSet"),
        "the type must exist for this guard to mean anything"
    );
    for door in ["pub fn new(", "pub const fn new(", "pub fn from_rows", "pub rows:"] {
        assert!(
            !source.contains(door),
            "`{door}` would let a caller mint a VerifiedRunSet without verify_run, which is the \
             whole guarantee aggregate's signature rests on"
        );
    }
}

#[test]
fn bench_gate_verify_run_reads_the_lock_and_ledger_bytes_from_disk() {
    let source = non_comment_source(BENCH_GATE_SOURCE);
    // The recomputation must READ FILES. A gate that compared the row's `lock_hash` to itself
    // would pass every test above while proving nothing.
    assert!(source.contains("lock_record_path"), "the lock path is resolved");
    assert!(source.contains("candidate_ledger_path"), "the ledger path is resolved");
    assert!(
        source.matches("read_evidence(cell").count() >= 2,
        "both provenance branches must read bytes from disk"
    );
    assert!(
        source.contains("sha256_hex(&bytes)"),
        "the digests must be RECOMPUTED over the bytes that were read"
    );
}

#[test]
fn bench_gate_the_selection_manifest_path_cannot_be_influenced_by_a_row() {
    let source = non_comment_source(BENCH_GATE_SOURCE);

    // THE SIGNATURE IS THE PROOF, and it is asserted rather than described. A reviewer must be
    // able to settle "can a row choose this path?" from the declaration alone — a strictly
    // stronger guarantee than a body that happens to be correct today.
    assert!(
        source
            .contains("pub fn selection_manifest_path(bench_dir: &Path, cell: CellKey) -> PathBuf"),
        "the derived-path helper must take only a benchmark directory and a cell key; a \
         `&BenchRow` parameter would re-open exactly the surface gap 1 closed"
    );

    // NO SECOND DEFINITION OF A MANIFEST'S IDENTITY (OPS-03). `SelectionManifest::from_bytes`
    // verifies `sha256(payload.to_canonical_bytes()) == semantic_hash` BEFORE returning, so a
    // recomputation here would be a second definition that drifts. This guard goes red on the
    // obvious way to write one.
    assert!(
        !source.contains("to_canonical_bytes()).map_or_else"),
        "non-vacuity: the guard's haystack must still hold the manifest's own digest check"
    );
    for forbidden in
        ["sha256_hex(&manifest.payload", "manifest.payload.to_canonical_bytes", "verify_digest()"]
    {
        assert!(
            !source.contains(forbidden),
            "`{forbidden}` would recompute a selection manifest's semantic_hash inside \
             bench_gate. `SelectionManifest::from_bytes` already verifies it before returning; \
             two definitions of one value disagree eventually and invisibly (OPS-03)"
        );
    }

    // The comparison must be EXACT. Each of these is a plausible relaxation that would accept
    // a doctored key, and `bench_gate_a_hash_agreeing_in_its_first_characters_is_still_a_refusal`
    // is the behavioural half of the same statement.
    for forbidden in ["eq_ignore_ascii_case", "starts_with(&row.payload.selection", "[..16]"] {
        assert!(
            !source.contains(forbidden),
            "`{forbidden}` would weaken the pairing-key comparison below exact byte equality"
        );
    }

    // And the cell key is WIDENED, never the manifest narrowed: a `try_from` that failed would
    // silently change the question from "the same seed" to "the same seed, if it fits".
    assert!(
        source.contains("u64::from(cell.seed)"),
        "the seed comparison must widen the cell key with u64::from"
    );
}

#[test]
fn bench_gate_layout_constants_agree_with_the_shipped_row_filename_grammar() {
    // The gate resolves rows by NAME. Two spellings of a filename are two filenames, and a
    // drift would report `row_file_missing` for a path the writer never used.
    assert_eq!(row_file_name(CellKey::new(Method::Setfit, 16, 29)), "setfit-s16-seed29.json");
    assert_eq!(row_file_name(CellKey::new(Method::Lora, 64, 53)), "lora-s64-seed53.json");
    assert_eq!(ROWS_DIR, "rows");
    assert_eq!(LOCKS_DIR, "locks");
    assert_eq!(LEDGER_DIR, "ledger");
    assert_eq!(RUN_MANIFEST_FILE, "run-manifest.json");
    assert_eq!(CONTRACTED_CANDIDATES_TRAINED, 1);

    // THE SELECTION-MANIFEST SPELLING, cross-pinned the same way. The fixture builds this path
    // from a literal and the gate builds it from `selection_manifest_path`; a drift between
    // them would make the fixture write forty manifests the gate never finds and report
    // `evidence_file_missing` for a file that is on disk.
    assert_eq!(SELECTIONS_DIR, "selections");
    assert_eq!(SELECTION_MANIFEST_FILE, "selection-manifest.json");
    let bench = Path::new("/tmp/bench");
    assert_eq!(
        selection_manifest_path(bench, CellKey::new(Method::Setfit, 16, 29)),
        bench.join(selection_manifest_rel(16, 29))
    );
    // BOTH METHODS OF A PAIR RESOLVE THE SAME FILE — the pairing design, asserted rather than
    // described. A path that took the method into account would give the two halves of a pair
    // two different manifests and make `pairing_rule` unsatisfiable by construction.
    assert_eq!(
        selection_manifest_path(bench, CellKey::new(Method::Setfit, 8, 13)),
        selection_manifest_path(bench, CellKey::new(Method::Lora, 8, 13)),
    );
}

#[test]
fn bench_gate_evidence_reads_are_bounded_from_the_declared_length() {
    let dir = TempDir::new().expect("temp dir");
    let path: PathBuf = dir.path().join("rows").join("setfit-s8-seed13.json");
    let error = read_evidence(CellKey::new(Method::Setfit, 8, 13), EvidenceKind::Row, &path)
        .expect_err("a missing file is a refusal");
    assert_eq!(error.variant_tag(), "row_file_missing");

    // A directory in a file's position is refused rather than read.
    let error = read_evidence(CellKey::new(Method::Setfit, 8, 13), EvidenceKind::Row, dir.path())
        .expect_err("a directory is not a row");
    assert_eq!(error.variant_tag(), "evidence_read_failed");

    // WR-06, option (a): the SAME absence under a non-ROW kind names its own artifact. One
    // unconditional mapping told the operator to restore the wrong thing.
    for kind in [EvidenceKind::Lock, EvidenceKind::Ledger] {
        let error = read_evidence(CellKey::new(Method::Setfit, 8, 13), kind, &path)
            .expect_err("a missing evidence file is a refusal");
        assert_eq!(error.variant_tag(), "evidence_file_missing");
        let rendered = error.to_string();
        assert!(rendered.contains(kind.tag()), "the refusal must name its own kind: {rendered}");
        assert!(
            !rendered.contains("restore the row file"),
            "a missing {} must not carry the ROW remedy: {rendered}",
            kind.tag()
        );
    }
}

// ===========================================================================================
// THE SINGLE-CELL VERIFICATION DOOR (plan 05-11 task 2)
//
// `verify_cell` is `verify_run`'s steps 1 + 4 + 6 over ONE declared cell. The two tests below
// carry it. The first is a BEHAVIOURAL equivalence table, not a structural restatement: a test
// asserting "the door calls verify_row_evidence and verify_provenance" would restate the
// door's own definition and could never go red, which is the exact defect class this phase
// exists to prevent.
// ===========================================================================================

/// One per-row defect: a name, the mutation that introduces it, and nothing else.
///
/// The mutation runs against a fresh directory for EACH entry point, so neither run can see
/// the other's side effects and the comparison is of two verdicts on the same defect rather
/// than of one verdict on a directory the other already touched.
struct RowDefect {
    name: &'static str,
    apply: fn(&Path),
}

/// Every per-row defect the door and `verify_run` both have to see, applied to `TARGET_SETFIT`.
fn per_row_defect_table() -> Vec<RowDefect> {
    vec![
        RowDefect {
            name: "the row file is absent",
            apply: |root| {
                fs::remove_file(root.join(ROWS_DIR).join(row_file_name(TARGET_SETFIT)))
                    .expect("remove");
            },
        },
        RowDefect {
            name: "the envelope digest no longer covers the payload",
            apply: |root| {
                let mut value = read_row_value(root, TARGET_SETFIT);
                *value
                    .get_mut("payload")
                    .and_then(|p| p.get_mut("quality"))
                    .and_then(|q| q.get_mut("f_avg"))
                    .expect("f_avg") = serde_json::json!(0.123_456_789);
                write_row_value(root, TARGET_SETFIT, &value);
            },
        },
        RowDefect {
            name: "the schema no longer parses (a required block was trimmed)",
            apply: |root| {
                let mut value = read_row_value(root, TARGET_SETFIT);
                value
                    .get_mut("payload")
                    .and_then(|p| p.get_mut("evidence"))
                    .and_then(|e| e.get_mut("setfit"))
                    .and_then(serde_json::Value::as_object_mut)
                    .expect("setfit block")
                    .remove("lock");
                write_row_value(root, TARGET_SETFIT, &value);
            },
        },
        RowDefect {
            name: "the row is filed under the wrong slot",
            apply: |root| {
                let other = CellKey::new(Method::Setfit, 16, 31);
                let bytes = fs::read(root.join(ROWS_DIR).join(row_file_name(other))).expect("read");
                fs::write(root.join(ROWS_DIR).join(row_file_name(TARGET_SETFIT)), &bytes)
                    .expect("write");
            },
        },
        RowDefect {
            name: "the committed lock bytes were tampered with",
            apply: |root| {
                let lock_path = root.join(format!(
                    "{LOCKS_DIR}/{}-s{}-seed{}.lock.json",
                    TARGET_SETFIT.method.tag(),
                    TARGET_SETFIT.shots,
                    TARGET_SETFIT.seed
                ));
                let mut bytes = fs::read(&lock_path).expect("lock read");
                let last = bytes.len() - 2;
                bytes[last] = b'9';
                fs::write(&lock_path, &bytes).expect("lock write");
            },
        },
        RowDefect {
            name: "the lock role is not one the vocabulary admits",
            apply: |root| {
                reseal_row(root, TARGET_SETFIT, |payload| {
                    if let MethodEvidence::Setfit(evidence) = &mut payload.evidence {
                        evidence.lock.role = "chosen_after_the_fact".to_string();
                    }
                });
            },
        },
        RowDefect {
            name: "the lock rule is not the committed one",
            apply: |root| {
                reseal_row(root, TARGET_SETFIT, |payload| {
                    if let MethodEvidence::Setfit(evidence) = &mut payload.evidence {
                        evidence.lock.rule = "best_observed_on_test".to_string();
                    }
                });
            },
        },
    ]
}

#[test]
fn bench_gate_the_single_cell_door_yields_the_same_variant_as_verify_run_for_every_row_defect() {
    // THE TEST THAT CARRIES THE WHOLE NO-DIVERGENCE CLAIM. Each defect is built ONCE per entry
    // point, fed to both, and the two variant TAGS are compared. If the door ever grows its own
    // copy of a check, or drops one, or reorders two, a row of this table goes red — which a
    // "the door calls the extracted function" assertion could never do.
    let mut compared = 0_usize;

    for defect in per_row_defect_table() {
        // verify_run's verdict.
        let run_dir = write_valid_run(RunSpec::default());
        let run_manifest =
            if defect.name.contains("wrong slot") || defect.name.contains("envelope digest") {
                // These two must be doctored AFTER the manifest records the honest digest,
                // otherwise the manifest would simply record the doctored bytes and the defect
                // would present as something else. Same ordering on both sides below.
                let m = manifest_for(run_dir.path());
                (defect.apply)(run_dir.path());
                m
            } else {
                (defect.apply)(run_dir.path());
                manifest_for(run_dir.path())
            };
        let run_tag = verify_run(&run_manifest, run_dir.path())
            .err()
            .unwrap_or_else(|| panic!("verify_run ACCEPTED `{}`", defect.name))
            .variant_tag();

        // The door's verdict, on the same defect in a fresh directory.
        let door_dir = write_valid_run(RunSpec::default());
        let door_manifest =
            if defect.name.contains("wrong slot") || defect.name.contains("envelope digest") {
                let m = manifest_for(door_dir.path());
                (defect.apply)(door_dir.path());
                m
            } else {
                (defect.apply)(door_dir.path());
                manifest_for(door_dir.path())
            };
        let door_tag = verify_cell(&door_manifest, door_dir.path(), TARGET_SETFIT)
            .err()
            .unwrap_or_else(|| panic!("verify_cell ACCEPTED `{}`", defect.name))
            .variant_tag();

        assert_eq!(
            door_tag, run_tag,
            "`{}`: the door said `{door_tag}` and verify_run said `{run_tag}`. The door is \
             steps 1 + 4 + 6 of verify_run and must not diagnose a row defect differently",
            defect.name,
        );
        compared += 1;
    }

    // NON-VACUITY. A table that silently shrank to zero rows would pass every assertion above.
    assert_eq!(compared, 7, "the per-row defect table must cover all seven shapes");
}

#[test]
fn bench_gate_the_single_cell_door_passes_on_one_complete_cell_among_thirty_nine_pending() {
    // THE PILOT STATE, WHICH IS THE DOOR'S REASON TO EXIST. The manifest declares 40 cells with
    // 39 still `pending` — precisely the state step 3's SWEEP refuses on. A door that inherited
    // the set-level checks could never pass on the cell it exists to check, and `bench report`
    // over a copy holding one row refuses at completeness BEFORE the row loop, so the pilot
    // row's own bytes are never read at all. This door reads them.
    let dir = write_valid_run(RunSpec::default());

    // Delete every row but the pilot, so the other 39 entries are genuinely pending rather
    // than hand-edited into looking that way.
    for cell in RunManifest::expectation() {
        if cell != TARGET_SETFIT {
            fs::remove_file(dir.path().join(ROWS_DIR).join(row_file_name(cell))).expect("remove");
        }
    }
    let manifest = manifest_for(dir.path());

    assert_eq!(manifest.completed(), 1, "exactly one cell is complete");
    assert_eq!(
        manifest.payload.cells.len(),
        EXPECTED_CELLS,
        "the expectation set is still fully DECLARED — that is what makes the other 39 \
         visible as pending rather than absent",
    );

    // The set-level door refuses, and must: 39 pending cells is not a publishable run.
    let run_error =
        verify_run(&manifest, dir.path()).expect_err("a 39-pending run is not complete");
    assert_eq!(run_error.variant_tag(), "incomplete_cell");

    // The single-cell door passes on the one complete cell. This is the whole point.
    verify_cell(&manifest, dir.path(), TARGET_SETFIT)
        .expect("the pilot cell's own evidence is valid and the door must say so");

    // ... and still refuses a cell that IS pending, so it is not simply permissive.
    let pending = CellKey::new(Method::Setfit, 16, 31);
    let pending_error = verify_cell(&manifest, dir.path(), pending)
        .expect_err("a pending cell has no evidence to verify");
    assert_eq!(pending_error.variant_tag(), "incomplete_cell");
}

#[test]
fn bench_gate_the_single_cell_door_refuses_a_cell_outside_the_active_scope() {
    // Asking the door for a second method's cell is an expectation-set disagreement, and it is
    // reported with the EXISTING variant — no new one is minted for the door either.
    let dir = write_valid_run(RunSpec::default());
    let manifest = manifest_for(dir.path());

    let error = verify_cell(&manifest, dir.path(), TARGET_LORA)
        .expect_err("a cell outside the active scope is not verifiable");
    assert_eq!(error.variant_tag(), "expectation_set_mismatch");
}

#[test]
fn bench_gate_the_variant_tag_table_gained_exactly_the_one_arm_this_round_authorised_05_17() {
    // Counted over the SHIPPED SOURCE rather than eyeballed in a diff, because a diff review is
    // exactly what missed this class of change before.
    //
    // 05-11 pinned this at 13 and its own two out-of-scope refusals reused EXISTING variants,
    // which is still the default answer. 05-15 mints exactly TWO, and both are reachable from
    // production input rather than from a test-only constructor — which is what 05-11's rule
    // actually forbade:
    //   * `evidence_path_escape`  — gap 1. A producer-written path leaving the benchmark tree
    //     had no refusal at all; reusing `evidence_read_failed` would have reported an I/O
    //     accident for a deliberate escape.
    //   * `evidence_file_missing` — WR-06. An absent LOCK or LEDGER was reported as a missing
    //     ROW file with a remedy naming the wrong artifact.
    //
    // The four contract-derived comparisons 05-15 also added (contract_id on both the row and
    // the manifest, calibration_split, warmup_count, cold_measured_in_child_process) mint
    // NOTHING: they reuse `row_schema_refused`. That asymmetry is the point of this guard — a
    // new arm has to be argued for, one at a time.
    //
    // 05-16 mints exactly TWO more, 15 -> 17, and each is argued separately because they are
    // NOT reachable by the same attack:
    //   * `selection_manifest_mismatch`      — gap 2. The pairing key was never recomputed
    //     from the committed manifest, so `apr setfit bench report` returned 0 with the whole
    //     `selections/` tree deleted and with a row's key doctored to 64 zeros. Reusing
    //     `provenance_mismatch` would have filed a PAIRING-KEY forgery under the lock/ledger
    //     recomputation, which names a different file and carries a different remedy.
    //   * `selection_manifest_cell_mismatch` — the transplant. A manifest from another cell
    //     seals correctly and hashes correctly, so a producer who also doctors the row's key
    //     satisfies the hash check completely; this is the second, independent statement of
    //     which cell drew the selection, and folding it into the tag above would tell an
    //     operator to look for a digest disagreement that does not exist.
    // 05-16's missing-manifest case mints NOTHING — it reuses `evidence_file_missing` via a
    // new `EvidenceKind`, which is the default answer and is why only two arms appear here.
    //
    // 05-17 mints exactly ONE, 17 -> 18, and the argument is that no existing tag says what it
    // says:
    //   * `quality_cross_check_mismatch` — verifier advisory 2 / spot-check D. The published
    //     metrics were numbers a producer typed, even though the row records the counts that
    //     determine them in closed form. Reusing `row_schema_refused` would report a PARSE
    //     problem for a row that parses perfectly and whose every digest is intact; reusing
    //     `provenance_mismatch` would name a committed FILE this refusal never opens — the
    //     evidence is inside the row itself.
    // Its THREE degenerate-matrix cases mint NOTHING: a ragged, mis-dimensioned, oversized or
    // all-zero `confusion_matrix` reuses `row_schema_refused`, because `Vec<Vec<u64>>` is the
    // wire type and a shape the schema cannot hold IS a schema refusal. The five `_bits`
    // sibling checks mint nothing either — they are the same claim about the same row.
    const GATE_SOURCE: &str = include_str!("bench_gate.rs");
    let table = GATE_SOURCE
        .split_once("pub const fn variant_tag(&self) -> &'static str {")
        .expect("the variant_tag table exists")
        .1
        .split_once("\n    }")
        .expect("the table ends")
        .0;
    let arms = table.matches("=> \"").count();
    assert_eq!(
        arms, 18,
        "BenchGateError::variant_tag has {arms} arms; it had 13 before 05-15, 15 after it, 17 \
         after 05-16, and 05-17 authorised exactly one more. A nineteenth would be a variant \
         nobody argued for",
    );
    for minted in [
        "evidence_path_escape",
        "evidence_file_missing",
        "selection_manifest_mismatch",
        "selection_manifest_cell_mismatch",
        "quality_cross_check_mismatch",
    ] {
        assert!(table.contains(minted), "the `{minted}` arm must be one of those argued for");
    }
}
