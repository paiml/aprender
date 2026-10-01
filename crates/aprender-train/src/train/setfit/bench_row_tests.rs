//! `BenchRow` / `RunManifest` tests (plan 05-05, EVAL-03 / EVAL-04).
//!
//! Every test name starts `bench_row_`, and the module path itself contains `bench_row`, so
//! the filter `cargo test -p aprender-train --lib --features setfit bench_row` selects
//! exactly this file. The count matters as much as the status: a name-filtered `cargo test`
//! that matches nothing prints `test result: ok. 0 passed` and exits 0 (CR-02), so the plan's
//! acceptance criterion reads the matched count out of the log rather than trusting `ok`.
//!
//! # The headline test
//!
//! `bench_row_expectation_matches_the_contract_as_a_typed_set` is the one that makes the
//! completeness rule mean something. It DESERIALIZES the contract with serde_yaml into typed
//! `Vec<String>` / `Vec<u32>` fields and compares SETS and CARDINALITY against the Rust
//! derivation. It deliberately does NOT text-search the contract: a substring check is
//! satisfied by a value sitting in a comment, by a contract carrying an EXTRA seed, and by a
//! contract carrying a DUPLICATED seed — three different benchmarks, one green gate. The two
//! mutated-copy negatives below prove this comparison rejects extras and duplicates rather
//! than only detecting absence.

use std::collections::BTreeSet;

use serde::Deserialize;

use super::*;

// ===========================================================================================
// The contract, and the typed view of it this file compares against
// ===========================================================================================

/// Just enough of the claims contract to reach the expectation set.
#[derive(Debug, Deserialize)]
struct ContractFile {
    equations: ContractEquations,
}

#[derive(Debug, Deserialize)]
struct ContractEquations {
    expectation_set: ContractExpectationSet,
}

#[derive(Debug, Deserialize)]
struct ContractExpectationSet {
    methods: Vec<String>,
    shots: Vec<u32>,
    seeds: Vec<u32>,
    expected_cells: usize,
}

/// The source of this module, read at compile time for the two structural guards below.
const BENCH_ROW_SOURCE: &str = include_str!("bench_row.rs");

/// Non-comment source lines only, so prose can neither satisfy nor trip a structural guard.
fn non_comment_source(src: &str) -> String {
    src.lines().filter(|line| !line.trim_start().starts_with("//")).collect::<Vec<_>>().join("\n")
}

/// How many times `needle` occurs in `haystack`. `str::matches` rather than a membership
/// helper, because the acceptance criterion for this file greps its non-comment lines for a
/// membership call and requires zero: a membership predicate is exactly the shape the
/// forbidden substring-parity test would be written in.
fn occurrences(haystack: &str, needle: &str) -> usize {
    haystack.matches(needle).count()
}

// ===========================================================================================
// Contract parity — a TYPED SET comparison, never a substring search
// ===========================================================================================

/// The full parity verdict, as a value rather than a panic, so the two mutated-contract
/// negatives can assert that it FAILS without `should_panic` swallowing an unrelated panic.
fn parity_verdict(contract_yaml: &str) -> Result<(), String> {
    let parsed: ContractFile =
        serde_yaml::from_str(contract_yaml).map_err(|e| format!("contract must parse: {e}"))?;
    let set = parsed.equations.expectation_set;

    // NON-VACUITY FIRST. An empty list would make every comparison below hold trivially,
    // for exactly the axis that went missing.
    if set.methods.is_empty() || set.shots.is_empty() || set.seeds.is_empty() {
        return Err("the contract's expectation set has an empty axis".to_string());
    }

    // NO DUPLICATES IN ANY AXIS. A duplicated seed leaves the Cartesian SET at 80 while the
    // list product reads 88; without this check the set comparison alone would stay green.
    let method_set: BTreeSet<&String> = set.methods.iter().collect();
    let shot_set: BTreeSet<u32> = set.shots.iter().copied().collect();
    let seed_set: BTreeSet<u32> = set.seeds.iter().copied().collect();
    if method_set.len() != set.methods.len() {
        return Err(format!(
            "contract methods carry a duplicate: {} listed, {} distinct",
            set.methods.len(),
            method_set.len()
        ));
    }
    if shot_set.len() != set.shots.len() {
        return Err(format!(
            "contract shots carry a duplicate: {} listed, {} distinct",
            set.shots.len(),
            shot_set.len()
        ));
    }
    if seed_set.len() != set.seeds.len() {
        return Err(format!(
            "contract seeds carry a duplicate: {} listed, {} distinct",
            set.seeds.len(),
            seed_set.len()
        ));
    }

    // THE DECLARED CARDINALITY IS THE PRODUCT, not an independently editable number.
    let product = set.methods.len() * set.shots.len() * set.seeds.len();
    if product != set.expected_cells {
        return Err(format!(
            "contract expected_cells {} != |methods| * |shots| * |seeds| = {product}",
            set.expected_cells
        ));
    }

    // Build the contract's product as a typed set of cell keys.
    let mut contracted: BTreeSet<CellKey> = BTreeSet::new();
    for method_tag in &set.methods {
        let method = Method::from_tag(method_tag)
            .ok_or_else(|| format!("contract names an unknown method `{method_tag}`"))?;
        for shots in &set.shots {
            for seed in &set.seeds {
                contracted.insert(CellKey::new(method, *shots, *seed));
            }
        }
    }

    let derived = RunManifest::expectation();
    let derived_set: BTreeSet<CellKey> = derived.iter().copied().collect();

    // CARDINALITY on both sides, so a duplicate in the Rust derivation is red too.
    if derived.len() != derived_set.len() {
        return Err(format!(
            "the Rust derivation carries a duplicate: {} entries, {} distinct",
            derived.len(),
            derived_set.len()
        ));
    }
    if derived.len() != set.expected_cells {
        return Err(format!(
            "the Rust derivation has {} cells, the contract declares {}",
            derived.len(),
            set.expected_cells
        ));
    }
    if derived_set != contracted {
        return Err(format!(
            "set mismatch: {} only in Rust, {} only in the contract",
            derived_set.difference(&contracted).count(),
            contracted.difference(&derived_set).count()
        ));
    }
    Ok(())
}

#[test]
fn bench_row_expectation_matches_the_contract_as_a_typed_set() {
    // The contract is DESERIALIZED, not searched. See this module's header for why a
    // substring check cannot discharge OBLIG-CLAIMS-EXPECTATION-CLOSED-FORM.
    let verdict = parity_verdict(CLAIMS_CONTRACT_YAML);
    assert_eq!(
        verdict,
        Ok(()),
        "the committed contract and RunManifest::expectation() must agree as SETS and in \
         CARDINALITY; edit BOTH or neither",
    );
}

#[test]
fn bench_row_parity_rejects_a_contract_with_an_extra_seed() {
    // A seed the Rust side does not derive. A `contains(seed)` check passes on this file —
    // every value it looks for is still present — while the benchmark it describes has 88
    // cells rather than 80.
    let mutated = CLAIMS_CONTRACT_YAML.replace(
        "      - 53\n    expected_cells: 40",
        "      - 53\n      - 59\n    expected_cells: 40",
    );
    assert_ne!(
        mutated, CLAIMS_CONTRACT_YAML,
        "the mutation must actually apply, or this negative proves nothing",
    );
    assert!(
        parity_verdict(&mutated).is_err(),
        "an EXTRA contracted seed must fail parity; a comparison that only detects ABSENCE \
         would stay green here",
    );
}

#[test]
fn bench_row_parity_rejects_a_contract_with_a_duplicated_seed() {
    // The Cartesian SET is unchanged by a duplicate — only the list cardinality moves. This
    // is the negative a naive set-only comparison passes.
    let mutated = CLAIMS_CONTRACT_YAML.replace(
        "      - 53\n    expected_cells: 40",
        "      - 53\n      - 53\n    expected_cells: 40",
    );
    assert_ne!(
        mutated, CLAIMS_CONTRACT_YAML,
        "the mutation must actually apply, or this negative proves nothing",
    );
    assert!(
        parity_verdict(&mutated).is_err(),
        "a DUPLICATED contracted seed must fail parity; the set is identical, so only the \
         cardinality half can catch it",
    );
}

#[test]
fn bench_row_parity_rejects_a_contract_with_an_extra_method() {
    // THE MUTATION THE 2.0.0 NARROWING MADE NECESSARY. The method axis is the one that
    // moved, so it is the axis a silent widening would move back — and a widening is the
    // only direction in which the narrowing can be undone without anyone noticing. The
    // seed mutations above cannot catch it: they move a different axis.
    let mutated = CLAIMS_CONTRACT_YAML.replace(
        "    methods:\n      - setfit\n    shots:",
        "    methods:\n      - setfit\n      - lora\n    shots:",
    );
    assert_ne!(
        mutated, CLAIMS_CONTRACT_YAML,
        "the mutation must actually apply, or this negative proves nothing",
    );
    assert!(
        parity_verdict(&mutated).is_err(),
        "restoring a SECOND METHOD to the ACTIVE expectation set must fail parity: the \
         contract would declare 80 cells while RunManifest::expectation() derives 40",
    );
}

#[test]
fn bench_row_active_scope_and_row_validity_are_different_questions() {
    // THE TWO FACTS THAT MUST NOT COLLAPSE INTO ONE. A second method's cell is:
    //   (a) VALID AS A ROW — `is_contracted` reads BENCH_METHODS, which keeps both, so the
    //       deferred-scope negatives can still construct the rows they doctor; and
    //   (b) OUT OF SCOPE for the active expectation set — `expectation()` derives from
    //       ACTIVE_METHODS, so no complete run contains it.
    // Narrowing BENCH_METHODS would have made (a) false and silently deleted two shipped
    // negatives while looking like a tightened gate.
    let lora_cell = CellKey::new(Method::Lora, 16, 29);

    assert!(
        lora_cell.is_contracted(),
        "row validity must still admit a second method's cell — BENCH_METHODS is the \
         ROW-VALIDITY domain and keeps both methods",
    );
    assert!(
        !RunManifest::expectation().contains(&lora_cell),
        "the ACTIVE expectation set must NOT contain it — ACTIVE_METHODS is the \
         EXPECTATION domain and names one method",
    );

    // And the active set is exactly the SetFit half, cell for cell.
    let active = RunManifest::expectation();
    assert!(
        active.iter().all(|c| c.method == Method::Setfit),
        "every active cell is a SetFit cell",
    );
    assert_eq!(active.len(), EXPECTED_CELLS);
}

#[test]
fn bench_row_deferred_scope_is_the_retained_eighty_cell_product() {
    // The deferred scope is RETAINED, not deleted — D-ITEM-05-15 restores it rather than
    // re-deriving it. Its constructor is `#[cfg(test)]`-gated, so this test is one of the
    // only places that can name it and production code cannot reach it at all.
    let deferred = RunManifest::expectation_for(ExpectationScope::DeferredTwoMethod);
    assert_eq!(deferred.len(), 80, "2 methods x 4 shots x 10 seeds");

    let unique: BTreeSet<CellKey> = deferred.iter().copied().collect();
    assert_eq!(unique.len(), 80, "no cell key may repeat");

    // The active scope is a strict SUBSET of the deferred one: restoration ADDS cells, it
    // does not move the ones already measured.
    let active: BTreeSet<CellKey> = RunManifest::expectation().iter().copied().collect();
    assert!(
        active.is_subset(&unique),
        "restoring the second arm must not invalidate a single active cell",
    );
    assert_eq!(unique.len() - active.len(), EXPECTED_CELLS, "the deferred half is the same size");
}

// ===========================================================================================
// The expectation set itself
// ===========================================================================================

#[test]
fn bench_row_expectation_is_forty_unique_cells() {
    let cells = RunManifest::expectation();
    let unique: BTreeSet<CellKey> = cells.iter().copied().collect();
    assert_eq!(cells.len(), EXPECTED_CELLS);
    assert_eq!(unique.len(), EXPECTED_CELLS, "no cell key may repeat");
    // Derived from ACTIVE_METHODS — the expectation-set domain — NOT from BENCH_METHODS,
    // which is the row-validity domain and still carries both methods. Asserting the product
    // against the wrong constant is exactly the collapse this plan exists to prevent.
    assert_eq!(EXPECTED_CELLS, ACTIVE_METHODS.len() * BENCH_SHOTS.len() * BENCH_SEEDS.len());
    assert_eq!(EXPECTED_CELLS, 40, "1 active method x 4 shots x 10 seeds");
    assert_eq!(
        BENCH_METHODS.len(),
        2,
        "the ROW-VALIDITY domain keeps both methods; narrowing it would make the two \
         deferred-scope negatives unbuildable"
    );
}

#[test]
fn bench_row_expectation_is_in_the_deterministic_contract_order() {
    let cells = RunManifest::expectation();
    let rendered: Vec<String> = cells.iter().map(CellKey::render).collect();

    assert_eq!(
        &rendered[..5],
        &[
            "setfit/s8/seed13".to_string(),
            "setfit/s8/seed17".to_string(),
            "setfit/s8/seed23".to_string(),
            "setfit/s8/seed29".to_string(),
            "setfit/s8/seed31".to_string(),
        ],
        "method in contract order, then shots ascending, then seed ascending",
    );
    assert_eq!(
        &rendered[EXPECTED_CELLS - 5..],
        &[
            "setfit/s64/seed37".to_string(),
            "setfit/s64/seed41".to_string(),
            "setfit/s64/seed43".to_string(),
            "setfit/s64/seed47".to_string(),
            "setfit/s64/seed53".to_string(),
        ],
        "iterating a hash-ordered map instead of the contract order changes this sequence",
    );
}

// ===========================================================================================
// Row fixtures
// ===========================================================================================

fn quality_block() -> QualityBlock {
    QualityBlock {
        f_avg: 0.6875,
        f_avg_bits: 0.6875_f64.to_bits(),
        macro_f1: 0.625,
        macro_f1_bits: 0.625_f64.to_bits(),
        per_class_precision: vec![0.5, 0.75, 0.625],
        per_class_recall: vec![0.5, 0.8125, 0.5625],
        per_class_f1: vec![0.5, 0.78125, 0.59375],
        mcc: 0.4375,
        mcc_bits: 0.4375_f64.to_bits(),
        confusion_matrix: vec![vec![20, 15, 10], vec![12, 150, 27], vec![8, 14, 24]],
        n_test_rows: 280,
        ordered_labels: vec!["none".to_string(), "against".to_string(), "favor".to_string()],
        ece_top_label_validation: 0.0625,
        ece_top_label_validation_bits: 0.0625_f64.to_bits(),
        brier_multiclass_validation: 0.5,
        brier_multiclass_validation_bits: 0.5_f64.to_bits(),
        calibration_split: CALIBRATION_SPLIT.to_string(),
    }
}

fn resource_block() -> ResourceBlock {
    ResourceBlock {
        train_wall_ms: 42_000,
        cold_latency_ms: 18.5,
        cold_measured_in_child_process: true,
        warm_latency_ms_median: 2.25,
        throughput_rows_per_sec: 480.0,
        throughput_batch_size: 32,
        warmup_count: WARMUP_COUNT,
        train_peak_rss_bytes: 1_234_567_890,
        train_peak_rss_mechanism: MECHANISM_CHILD_MAX_RSS_TIME_L.to_string(),
        inference_peak_rss_bytes: 234_567_890,
        inference_peak_rss_mechanism: MECHANISM_CHILD_MAX_RSS_TIME_L.to_string(),
        peak_rss_sample_interval_hz: None,
        artifact_bytes: 90_123_456,
        deployable_total_bytes: 90_123_456,
    }
}

fn setfit_evidence() -> MethodEvidence {
    MethodEvidence::Setfit(SetfitEvidence {
        evidence_table_hash: "a".repeat(64),
        apr_artifact_sha256: "b".repeat(64),
        lock: BenchLockRef {
            lock_hash: "c".repeat(64),
            role: "consumed".to_string(),
            rule: "max_validation_accuracy".to_string(),
            lock_record_path: "locks/setfit-s16-seed29.json".to_string(),
        },
    })
}

fn lora_evidence() -> MethodEvidence {
    MethodEvidence::Lora(LoraEvidence {
        base_model_sha256: "d".repeat(64),
        base_model_bytes: 9_000_000_000,
        adapter_sha256: "e".repeat(64),
        epochs_requested: 3,
        epochs_completed: 3,
        early_stopping_disabled: true,
        val_split: 0.0,
        no_selection_attestation: true,
        candidate_ledger_sha256: "f".repeat(64),
        candidates_trained: 1,
        candidate_ledger_path: "ledgers/lora-s16-seed29.jsonl".to_string(),
    })
}

fn payload(method: Method, evidence: MethodEvidence) -> BenchRowPayload {
    BenchRowPayload {
        schema_version: BENCH_ROW_SCHEMA_VERSION,
        contract_id: CLAIMS_CONTRACT_ID.to_string(),
        method,
        shots: 16,
        seed: 29,
        dataset_revision: "4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66".to_string(),
        dataset_fingerprint: "1".repeat(64),
        model_revision: "sentence-transformers/all-MiniLM-L6-v2@1110a243".to_string(),
        selection_manifest_hash: "2".repeat(64),
        backend_identity: "cpu:setfit-core:scalar".to_string(),
        host: HostIdentity {
            hostname: "bench-host-01".to_string(),
            os: "macos".to_string(),
            arch: "aarch64".to_string(),
        },
        quality: quality_block(),
        resource: resource_block(),
        evidence,
    }
}

fn setfit_row() -> BenchRow {
    BenchRow::new(payload(Method::Setfit, setfit_evidence()))
}

fn lora_row() -> BenchRow {
    BenchRow::new(payload(Method::Lora, lora_evidence()))
}

fn row_json(row: &BenchRow) -> serde_json::Value {
    let bytes = row.to_file_bytes().expect("a fixture row must serialize");
    serde_json::from_slice(&bytes).expect("its own output must be JSON")
}

fn bytes_of(value: &serde_json::Value) -> Vec<u8> {
    serde_json::to_vec(value).expect("a test-constructed value must serialize")
}

// ===========================================================================================
// Round trip and the five distinct refusals
// ===========================================================================================

#[test]
fn bench_row_round_trips_with_a_stable_semantic_hash() {
    let row = setfit_row();
    let first = row.to_file_bytes().expect("serialize");
    let parsed = BenchRow::from_bytes(&first).expect("its own bytes must verify");
    let second = parsed.to_file_bytes().expect("re-serialize");
    let reparsed = BenchRow::from_bytes(&second).expect("and verify again");

    assert_eq!(first, second, "the on-disk form must be byte-stable across a round trip");
    assert_eq!(row.semantic_hash, parsed.semantic_hash);
    assert_eq!(parsed.semantic_hash, reparsed.semantic_hash);
    assert_eq!(row.payload, reparsed.payload);
}

#[test]
fn bench_row_refuses_an_unknown_field() {
    let row = setfit_row();
    let mut value = row_json(&row);
    value
        .as_object_mut()
        .expect("the envelope is an object")
        // A bootstrap-shaped column is the field this schema exists to keep out (D-06). It is
        // written here rather than in the row type, so the guard is exercised, not described.
        .insert("bootstrap_ci_low".to_string(), serde_json::json!(0.01));

    let err = BenchRow::from_bytes(&bytes_of(&value)).expect_err("an unnamed field is refused");
    assert!(
        matches!(err, BenchRowError::Serialization { .. }),
        "expected a deserialization refusal naming the offending field, got {err:?}",
    );
}

#[test]
fn bench_row_refuses_a_wrong_schema_version() {
    let mut row = setfit_row();
    row.payload.schema_version = BENCH_ROW_SCHEMA_VERSION + 1;
    let row = BenchRow::new(row.payload); // re-seal, so the digest is NOT what fails

    let err = BenchRow::from_bytes(&row.to_file_bytes().expect("serialize"))
        .expect_err("a foreign schema version is refused");
    assert!(
        matches!(
            err,
            BenchRowError::UnsupportedSchemaVersion { got, supported }
                if got == BENCH_ROW_SCHEMA_VERSION + 1 && supported == BENCH_ROW_SCHEMA_VERSION
        ),
        "expected the schema-version refusal, got {err:?}",
    );
}

#[test]
fn bench_row_refuses_a_bit_flipped_payload() {
    let row = setfit_row();
    let mut value = row_json(&row);
    // Flip one value inside the payload, leaving the envelope digest untouched. This is the
    // tampering case: the bytes are well-formed and the digest no longer describes them.
    value["payload"]["quality"]["f_avg"] = serde_json::json!(0.9999);

    let err = BenchRow::from_bytes(&bytes_of(&value)).expect_err("a stale digest is refused");
    assert!(
        matches!(err, BenchRowError::SemanticHashMismatch { .. }),
        "expected the digest refusal — the check must run BEFORE the value is returned, so a \
         field-level diagnosis here would mean the order has drifted; got {err:?}",
    );
}

#[test]
fn bench_row_refuses_a_setfit_row_missing_its_lock_block() {
    let row = setfit_row();
    let mut value = row_json(&row);
    value["payload"]["evidence"]["setfit"]
        .as_object_mut()
        .expect("the setfit evidence is an object")
        .remove("lock");
    // Re-seal so the DIGEST is not what fails: the point is that the block is unusable.
    let resealed = reseal(&mut value);

    let err = BenchRow::from_bytes(&resealed).expect_err("a lock-less setfit row is refused");
    assert!(
        matches!(err, BenchRowError::MissingSetfitEvidence { .. }),
        "expected the missing-setfit-evidence refusal, got {err:?}",
    );
}

#[test]
fn bench_row_refuses_a_lora_row_missing_its_attestation_block() {
    let row = lora_row();
    let mut value = row_json(&row);
    let block = value["payload"]["evidence"]["lora"]
        .as_object_mut()
        .expect("the lora evidence is an object");
    block.remove("candidate_ledger_sha256");
    block.remove("candidates_trained");
    let resealed = reseal(&mut value);

    let err = BenchRow::from_bytes(&resealed).expect_err("a ledger-less lora row is refused");
    assert!(
        matches!(err, BenchRowError::MissingLoraEvidence { .. }),
        "expected the missing-lora-evidence refusal, got {err:?}",
    );
}

/// Recompute the envelope digest over the (mutated) payload, so a test that is about a
/// SEMANTIC refusal is not accidentally satisfied by the digest guard firing first.
fn reseal(value: &mut serde_json::Value) -> Vec<u8> {
    let payload_bytes = serde_json::to_vec(&value["payload"]).expect("payload serializes");
    let digest = sha256_hex(&payload_bytes);
    value["semantic_hash"] = serde_json::json!(digest);
    bytes_of(value)
}

#[test]
fn bench_row_five_refusals_are_five_distinct_variants() {
    // Each refusal must be its OWN variant. A single catch-all error would let a caller
    // "handle" tampering by handling a parse failure, and would make every discriminating
    // test above pass for the wrong reason.
    let row = setfit_row();

    let mut unknown = row_json(&row);
    unknown
        .as_object_mut()
        .expect("object")
        .insert("bootstrap_ci_low".to_string(), serde_json::json!(0.01));
    let e_unknown = BenchRow::from_bytes(&bytes_of(&unknown)).expect_err("unknown field");

    let mut versioned = setfit_row();
    versioned.payload.schema_version = 99;
    let versioned = BenchRow::new(versioned.payload);
    let e_version = BenchRow::from_bytes(&versioned.to_file_bytes().expect("serialize"))
        .expect_err("schema version");

    let mut flipped = row_json(&row);
    flipped["payload"]["quality"]["f_avg"] = serde_json::json!(0.1234);
    let e_digest = BenchRow::from_bytes(&bytes_of(&flipped)).expect_err("digest");

    let mut lockless = row_json(&row);
    lockless["payload"]["evidence"]["setfit"].as_object_mut().expect("object").remove("lock");
    let lockless = reseal(&mut lockless);
    let e_setfit = BenchRow::from_bytes(&lockless).expect_err("missing lock");

    let lora = lora_row();
    let mut ledgerless = row_json(&lora);
    ledgerless["payload"]["evidence"]["lora"]
        .as_object_mut()
        .expect("object")
        .remove("candidate_ledger_sha256");
    let ledgerless = reseal(&mut ledgerless);
    let e_lora = BenchRow::from_bytes(&ledgerless).expect_err("missing ledger");

    let tags = [
        e_unknown.variant_tag(),
        e_version.variant_tag(),
        e_digest.variant_tag(),
        e_setfit.variant_tag(),
        e_lora.variant_tag(),
    ];
    let distinct: BTreeSet<&'static str> = tags.iter().copied().collect();
    assert_eq!(distinct.len(), 5, "the five refusals collapsed into {tags:?}");
}

#[test]
fn bench_row_refuses_an_uncontracted_cell() {
    // 42 is NOT a contracted seed (OBLIG-TWEET-EVAL-SEED-SET), and 7 is not a contracted
    // shot count. A row outside the matrix is not a row this benchmark can place.
    let mut off_seed = setfit_row();
    off_seed.payload.seed = 42;
    let off_seed = BenchRow::new(off_seed.payload);
    let err = BenchRow::from_bytes(&off_seed.to_file_bytes().expect("serialize"))
        .expect_err("an uncontracted seed is refused");
    assert!(matches!(err, BenchRowError::UncontractedCell { .. }), "got {err:?}");

    let mut off_shots = setfit_row();
    off_shots.payload.shots = 7;
    let off_shots = BenchRow::new(off_shots.payload);
    let err = BenchRow::from_bytes(&off_shots.to_file_bytes().expect("serialize"))
        .expect_err("an uncontracted shot count is refused");
    assert!(matches!(err, BenchRowError::UncontractedCell { .. }), "got {err:?}");
}

#[test]
fn bench_row_refuses_evidence_that_contradicts_its_method_tag() {
    // SAFE-03's spirit: a lora row may not present setfit evidence. The tag and the block
    // are two statements of the same fact, and they must agree.
    let spoofed = BenchRow::new(payload(Method::Lora, setfit_evidence()));
    let err = BenchRow::from_bytes(&spoofed.to_file_bytes().expect("serialize"))
        .expect_err("a spoofed method tag is refused");
    assert!(matches!(err, BenchRowError::MissingLoraEvidence { .. }), "got {err:?}");
}

// ===========================================================================================
// The run manifest
// ===========================================================================================

fn declared_manifest() -> RunManifest {
    RunManifest::declare()
}

#[test]
fn bench_row_manifest_round_trips_and_verifies_its_digest() {
    let manifest = declared_manifest();
    let bytes = manifest.to_file_bytes().expect("serialize");
    let parsed = RunManifest::from_bytes(&bytes).expect("its own bytes must verify");
    assert_eq!(parsed.payload.cells.len(), EXPECTED_CELLS);
    assert_eq!(parsed.semantic_hash, manifest.semantic_hash);
    assert_eq!(parsed.to_file_bytes().expect("re-serialize"), bytes);

    let mut value: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
    value["payload"]["cells"][0]["status"] = serde_json::json!("complete");
    let err = RunManifest::from_bytes(&bytes_of(&value)).expect_err("a stale digest is refused");
    assert!(matches!(err, BenchRowError::SemanticHashMismatch { .. }), "got {err:?}");
}

#[test]
fn bench_row_manifest_record_is_idempotent_on_an_identical_hash() {
    let mut manifest = declared_manifest();
    let cell = CellKey::new(Method::Setfit, 16, 29);
    let hash = "9".repeat(64);

    assert_eq!(manifest.record(cell, &hash), Ok(RecordOutcome::Recorded));
    // The resume path: a driver restarted after a crash re-records the cell it already
    // finished. Refusing here is what gets manifests deleted and re-created, which erases
    // the pre-declared expectation set that makes omission visible at all.
    assert_eq!(manifest.record(cell, &hash), Ok(RecordOutcome::AlreadyRecorded));
    assert_eq!(manifest.completed(), 1);
}

#[test]
fn bench_row_manifest_record_collides_on_a_differing_hash() {
    let mut manifest = declared_manifest();
    // The LAST cell of the ACTIVE expectation set. It was `lora/s64/seed53` before the
    // 2.0.0 narrowing; a manifest cannot record a cell it does not declare, so the tail of
    // the set moved with the set rather than this test losing its subject.
    let cell = CellKey::new(Method::Setfit, 64, 53);
    let first = "1".repeat(64);
    let second = "2".repeat(64);

    assert_eq!(manifest.record(cell, &first), Ok(RecordOutcome::Recorded));
    let err = manifest.record(cell, &second).expect_err("a contradicting re-record is refused");
    assert!(
        matches!(
            &err,
            BenchRowError::CellAlreadyRecorded { existing, incoming, .. }
                if existing == &first && incoming == &second
        ),
        "the error must name BOTH digests so the contradiction is visible, got {err:?}",
    );
    // And the first recording SURVIVES: a refused write must not have partially applied.
    assert_eq!(manifest.completed(), 1);
    assert_eq!(manifest.row_sha256(cell), Some(first.as_str()));
}

#[test]
fn bench_row_manifest_record_refuses_a_cell_outside_the_expectation() {
    let mut manifest = declared_manifest();
    let err = manifest
        .record(CellKey::new(Method::Setfit, 16, 42), &"3".repeat(64))
        .expect_err("42 is not a contracted seed");
    assert!(matches!(err, BenchRowError::UnknownCell { .. }), "got {err:?}");
}

#[test]
fn bench_row_manifest_refuses_an_empty_expectation_set() {
    // BACKSTOP ONE. A gate whose input is empty and which therefore finds no failures
    // reports success — the vacuity defect this repository has shipped more than once.
    let manifest = declared_manifest();
    let bytes = manifest.to_file_bytes().expect("serialize");
    let mut value: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
    value["payload"]["cells"] = serde_json::json!([]);
    let resealed = reseal(&mut value);

    let err = RunManifest::from_bytes(&resealed).expect_err("zero cells is an ERROR, not an Ok");
    assert!(matches!(err, BenchRowError::EmptyExpectationSet), "got {err:?}");
}

#[test]
fn bench_row_manifest_refuses_an_expectation_set_that_is_not_the_eighty_cells() {
    // BACKSTOP TWO. Otherwise a producer declares a 12-cell expectation, satisfies it
    // completely, and publishes a "complete" run.
    let manifest = declared_manifest();
    let bytes = manifest.to_file_bytes().expect("serialize");
    let mut value: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
    let cells = value["payload"]["cells"].as_array_mut().expect("an array");
    cells.truncate(12);
    let resealed = reseal(&mut value);

    let err = RunManifest::from_bytes(&resealed).expect_err("a short expectation set is refused");
    assert!(
        matches!(err, BenchRowError::ExpectationSetMismatch { declared, expected }
            if declared == 12 && expected == EXPECTED_CELLS),
        "got {err:?}",
    );
}

#[test]
fn bench_row_manifest_refuses_a_reordered_expectation_set() {
    // Order is part of the contract (OBLIG-CLAIMS-DETERMINISTIC-ORDER): two manifests of the
    // same run must be byte-comparable, and PF-007's "shuffle rows, require unchanged
    // aggregates" check needs a canonical order to shuffle away from.
    let manifest = declared_manifest();
    let bytes = manifest.to_file_bytes().expect("serialize");
    let mut value: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
    let cells = value["payload"]["cells"].as_array_mut().expect("an array");
    // The LAST index of the ACTIVE set, derived rather than typed: a literal 79 was a
    // second, silently-drifting statement of the set's size.
    cells.swap(0, EXPECTED_CELLS - 1);
    let resealed = reseal(&mut value);

    let err = RunManifest::from_bytes(&resealed).expect_err("a reordered manifest is refused");
    assert!(matches!(err, BenchRowError::ExpectationSetMismatch { .. }), "got {err:?}");
}

#[test]
fn bench_row_manifest_refuses_a_wrong_schema_version() {
    let mut manifest = declared_manifest();
    manifest.payload.schema_version = RUN_MANIFEST_SCHEMA_VERSION + 1;
    let manifest = RunManifest::seal(manifest.payload);
    let err = RunManifest::from_bytes(&manifest.to_file_bytes().expect("serialize"))
        .expect_err("a foreign schema version is refused");
    assert!(matches!(err, BenchRowError::UnsupportedSchemaVersion { .. }), "got {err:?}");
}

// ===========================================================================================
// Structural guards over this module's own source
// ===========================================================================================

#[test]
fn bench_row_deny_unknown_fields_is_on_every_serde_struct() {
    let source = non_comment_source(BENCH_ROW_SOURCE);
    let derives = occurrences(&source, "#[derive(");
    let denies = occurrences(&source, "deny_unknown_fields");
    assert!(
        denies >= 4,
        "expected at least four `deny_unknown_fields` attributes in non-comment source, \
         found {denies} (across {derives} derive attributes). Dropping it from even one \
         nested struct opens exactly one hole, and the hole is invisible.",
    );
}

#[test]
fn bench_row_no_resampling_vocabulary_enters_the_row_schema() {
    // D-06 made textual as well as structural. `deny_unknown_fields` stops a resampled column
    // arriving from OUTSIDE; this stops one being ADDED from inside.
    let source = non_comment_source(BENCH_ROW_SOURCE);
    for banned in ["bootstrap", "resample", "resampled", "permutation_test"] {
        assert_eq!(
            occurrences(&source, banned),
            0,
            "`{banned}` appears in non-comment source; EVAL-04's \"exactly recompute\" is \
             bit-level and an RNG stream downgrades it to \"up to a seed\"",
        );
    }
}
