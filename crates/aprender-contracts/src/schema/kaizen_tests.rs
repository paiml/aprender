use super::*;
use crate::schema::{parse_contract_str, validate_contract, ContractKind};

/// The metadata block every kaizen record carries, as a prefix for the record
/// bodies below.
const KAIZEN_METADATA: &str = r#"metadata:
  kind: kaizen
  version: "1.0.0"
  created: "2026-03-04"
  author: "PAIML Engineering"
  description: "Pre-allocated CPU staging buffer for the backward mean-pool gradient"
  references:
    - "KAIZEN-061"
"#;

fn errors_of(yaml: &str) -> Vec<String> {
    let contract = parse_contract_str(yaml).expect("kaizen fixture parses");
    validate_contract(&contract)
        .into_iter()
        .filter(|v| v.severity == crate::error::Severity::Error)
        .map(|v| format!("{}: {}", v.rule, v.message))
        .collect()
}

fn rules_of(yaml: &str) -> Vec<String> {
    let contract = parse_contract_str(yaml).expect("kaizen fixture parses");
    validate_contract(&contract)
        .into_iter()
        .filter(|v| v.severity == crate::error::Severity::Error)
        .map(|v| v.rule)
        .collect()
}

/// The real `backward-cpu-staging-v1` delta: three cost metrics, all falling.
const HEALTHY_RECORD: &str = r#"contract: C-BWDSTG-001
title: Pre-allocated CPU Staging Buffer
kaizen: KAIZEN-061
parent: C-GPUTRAINIT-002
status: implemented
date: 2026-03-04
baseline:
  alloc_per_backward: 1
  alloc_size_bytes: 1310720
  per_epoch_heap_churn_bytes: 18350080000
target:
  alloc_per_backward: 0
  alloc_size_bytes: 0
  per_epoch_heap_churn_bytes: 0
"#;

fn with_metadata(body: &str) -> String {
    format!("{KAIZEN_METADATA}{body}")
}

#[test]
fn kaizen_kind_round_trips() {
    assert_eq!(ContractKind::Kaizen.to_string(), "kaizen");
    let k: ContractKind = serde_yaml::from_str("kaizen").expect("kebab-case parses");
    assert_eq!(k, ContractKind::Kaizen);
}

/// KAIZEN_STATUSES is the corpus vocabulary, exactly — not a wish-list.
/// Widening it is a deliberate edit here, which is the point.
#[test]
fn kaizen_status_vocabulary_is_the_measured_corpus() {
    assert_eq!(
        KAIZEN_STATUSES.to_vec(),
        vec![
            "draft",
            "implemented",
            "implementing",
            "pending",
            "planned"
        ],
    );
}

/// A well-formed kaizen record validates clean — and, critically, is NOT asked
/// for Kani harnesses. That exemption is the whole reason the kind exists.
#[test]
fn healthy_kaizen_record_validates_and_is_exempt_from_provability() {
    let yaml = with_metadata(HEALTHY_RECORD);
    let errors = errors_of(&yaml);
    assert!(errors.is_empty(), "healthy record rejected: {errors:?}");

    let contract = parse_contract_str(&yaml).expect("parses");
    assert_eq!(contract.kind(), ContractKind::Kaizen);
    assert!(!contract.requires_proofs());
    assert!(contract.kani_harnesses.is_empty());
}

/// THE regression this kind is built to catch: a record whose `baseline:` and
/// `target:` have been written the wrong way round. Every cost metric rises,
/// none falls, and the record claims that as an improvement.
#[test]
fn reversed_baseline_and_target_is_rejected() {
    let reversed = r#"contract: C-BWDSTG-001
kaizen: KAIZEN-061
status: implemented
baseline:
  alloc_per_backward: 0
  alloc_size_bytes: 0
  per_epoch_heap_churn_bytes: 0
target:
  alloc_per_backward: 1
  alloc_size_bytes: 1310720
  per_epoch_heap_churn_bytes: 18350080000
"#;
    let rules = rules_of(&with_metadata(reversed));
    assert!(
        rules.contains(&"KAIZEN-006".to_string()),
        "a reversed kaizen delta must be rejected, got: {rules:?}"
    );
}

/// Mutation proof that KAIZEN-006 is load-bearing rather than a rule that
/// refuses everything: the SAME numbers in the right order pass.
#[test]
fn kaizen_006_accepts_the_unreversed_record() {
    assert!(
        !rules_of(&with_metadata(HEALTHY_RECORD)).contains(&"KAIZEN-006".to_string()),
        "KAIZEN-006 must not fire on a record whose costs fall"
    );
}

/// A kaizen may buy a win with a cost — `gpu-workspace-clip-v1` restores
/// per-block gradient clipping and RAISES `d2h_per_block` 0 → 9. Rejecting
/// that would have made the rule false of the corpus it was written for.
#[test]
fn a_cost_bought_win_is_accepted() {
    let body = r#"contract: C-WCLIP-001
kaizen: KAIZEN-054
status: implemented
baseline:
  per_block_clipping: false
  d2h_per_block: 0
target:
  per_block_clipping: true
  d2h_per_block: 9
"#;
    let errors = errors_of(&with_metadata(body));
    assert!(errors.is_empty(), "cost-bought win rejected: {errors:?}");
}

/// `gradient-accumulation-canary-v1` lowers `batch_size` 4 → 1 and raises
/// `gradient_accumulation_steps` 1 → 4 to hold `effective_batch` constant.
/// One improvement, two opposite movements.
#[test]
fn a_mixed_direction_record_is_accepted() {
    let body = r#"contract: gradient-accumulation-canary-v1
kaizen: KAIZEN-069
status: planned
baseline:
  batch_size: 4
  gradient_accumulation_steps: 1
  effective_batch: 4
target:
  batch_size: 1
  gradient_accumulation_steps: 4
  effective_batch: 4
"#;
    let errors = errors_of(&with_metadata(body));
    assert!(errors.is_empty(), "mixed-direction record rejected: {errors:?}");
}

/// KAIZEN-005: a target that restates the baseline claims nothing.
#[test]
fn target_restating_baseline_unchanged_is_rejected() {
    let body = r#"contract: C-NOOP-001
kaizen: KAIZEN-000
status: implemented
baseline:
  alloc_per_step: 3
target:
  alloc_per_step: 3
"#;
    assert!(rules_of(&with_metadata(body)).contains(&"KAIZEN-005".to_string()));
}

/// KAIZEN-005: prose on one side of every shared key pins no number.
#[test]
fn target_with_no_comparable_metric_is_rejected() {
    let body = r#"contract: C-PROSE-001
kaizen: KAIZEN-000
status: implemented
baseline:
  approach: "allocate per call"
target:
  approach: "reuse the buffer"
"#;
    assert!(rules_of(&with_metadata(body)).contains(&"KAIZEN-005".to_string()));
}

/// KAIZEN-004: a target with no baseline, and a target sharing no key with it.
#[test]
fn incoherent_baseline_target_pairs_are_rejected() {
    let no_baseline = r#"contract: C-NOBASE-001
kaizen: KAIZEN-000
status: implemented
target:
  alloc_per_step: 0
"#;
    assert!(rules_of(&with_metadata(no_baseline)).contains(&"KAIZEN-004".to_string()));

    let disjoint = r#"contract: C-DISJOINT-001
kaizen: KAIZEN-000
status: implemented
baseline:
  alloc_per_step: 3
target:
  latency_ms: 4
"#;
    assert!(rules_of(&with_metadata(disjoint)).contains(&"KAIZEN-004".to_string()));
}

/// KAIZEN-001 / KAIZEN-002: identity and a status from the closed vocabulary.
#[test]
fn missing_identity_and_invented_status_are_rejected() {
    let anonymous = r#"kaizen: KAIZEN-000
status: implemented
invariants:
  - "the buffer is never reallocated"
"#;
    assert!(rules_of(&with_metadata(anonymous)).contains(&"KAIZEN-001".to_string()));

    let invented = r#"contract: C-STATUS-001
kaizen: KAIZEN-000
status: mostly-done
invariants:
  - "the buffer is never reallocated"
"#;
    assert!(rules_of(&with_metadata(invented)).contains(&"KAIZEN-002".to_string()));

    let absent = r#"contract: C-STATUS-002
kaizen: KAIZEN-000
invariants:
  - "the buffer is never reallocated"
"#;
    assert!(rules_of(&with_metadata(absent)).contains(&"KAIZEN-002".to_string()));
}

/// KAIZEN-003: a record with no delta, no invariants, no obligations and no
/// falsification tests asserts nothing a later measurement could contradict.
#[test]
fn a_record_that_states_nothing_is_rejected() {
    let body = r#"contract: C-EMPTY-001
kaizen: KAIZEN-000
status: draft
"#;
    assert!(rules_of(&with_metadata(body)).contains(&"KAIZEN-003".to_string()));
}

/// KAIZEN-003 is discharged by `invariants:` alone — 29 of the 46 corpus
/// records make their claim that way and carry no baseline/target at all.
#[test]
fn invariants_alone_discharge_non_vacuity() {
    let body = r#"contract: blis-safe-alloc-v1
kaizen: '027'
status: implemented
invariants:
  - id: C-BLIS-ALLOC-001
    description: BLIS alloc functions return fully initialized Vec<f32>
"#;
    let errors = errors_of(&with_metadata(body));
    assert!(errors.is_empty(), "invariants-only record rejected: {errors:?}");
}

/// A case table, run rather than read (CLAUDE.md Verification Discipline #7).
/// The must-NOT-parse rows are the ones that matter: a permissive numeric
/// prefix reads `2026-03-04` as the year 2026 and invents a comparison out of
/// a date, and `CUBLAS_COMPUTE_32F` as the number 32.
#[test]
fn quantity_parse_case_table() {
    let must_parse: &[(&str, f64)] = &[
        ("40.34", 40.34),
        ("0", 0.0),
        ("<5", 5.0),
        ("<= 12", 12.0),
        ("1000+", 1000.0),
        ("100%", 100.0),
        ("6 ms", 6.0),
        ("1_310_720", 1_310_720.0),
        ("-3", -3.0),
    ];
    for (input, expected) in must_parse {
        let got = parse_quantity(&serde_yaml::Value::String((*input).to_string()));
        assert_eq!(got, Some(*expected), "{input:?} should read as a quantity");
    }

    let must_not_parse = [
        "2026-03-04",
        "CUBLAS_COMPUTE_32F",
        "fp32",
        "OOM",
        "pytorch",
        "131,072x fewer bytes transferred",
        "",
        "cuMemcpyDtoH full gradient buffer",
    ];
    for input in must_not_parse {
        let got = parse_quantity(&serde_yaml::Value::String(input.to_string()));
        assert_eq!(got, None, "{input:?} must NOT read as a quantity");
    }

    // Booleans are a state change, not a movement along an axis.
    assert_eq!(parse_quantity(&serde_yaml::Value::Bool(true)), None);
    assert_eq!(
        parse_quantity(&serde_yaml::from_str::<serde_yaml::Value>("9").expect("number")),
        Some(9.0)
    );
}

/// The cost-metric table, both arms. The must-NOT-flag rows are the keys that
/// falsified the two rules this module documents as rejected.
#[test]
fn cost_metric_case_table() {
    for name in [
        "alloc_size_bytes",
        "per_epoch_heap_churn_bytes",
        "per_forward_overhead_us",
        "total_launch_overhead_ms",
        "syncs_per_step_36_blocks",
        "wasted_alloc_per_step_bytes",
        "heap_allocs",
        "sync_points",
        "cuMemAlloc_per_forward",
    ] {
        assert!(is_cost_metric(name), "{name} should read as a cost metric");
    }
    for name in [
        "d2h_per_block",
        "batch_size",
        "gradient_accumulation_steps",
        "effective_batch",
        "per_block_clipping",
        "cache_hit_rate",
        "weight_dtype",
    ] {
        assert!(
            !is_cost_metric(name),
            "{name} must NOT read as a cost metric — a kaizen may raise it"
        );
    }
}

/// The `baseline: {before:, after:}` shape used by `gpu-l2-norm-reduction-v1`
/// is a delta pair too; treating it as "no claim" would have exempted the one
/// record that states its claim most explicitly.
#[test]
fn before_after_nested_in_baseline_is_a_delta_pair() {
    let body = r#"contract: C-SQSUM-002
kaizen: '049'
status: implemented
baseline:
  before:
    transfer_bytes: 134217728
  after:
    transfer_bytes: 1024
"#;
    let errors = errors_of(&with_metadata(body));
    assert!(errors.is_empty(), "before/after record rejected: {errors:?}");

    let reversed = r#"contract: C-SQSUM-002
kaizen: '049'
status: implemented
baseline:
  before:
    transfer_bytes: 1024
  after:
    transfer_bytes: 134217728
"#;
    assert!(rules_of(&with_metadata(reversed)).contains(&"KAIZEN-006".to_string()));
}

/// The kaizen second parse pass runs ONLY for `kind: kaizen`. A non-kaizen
/// contract carrying a top-level `status:` of an incompatible shape must still
/// parse — the pass must not become a new way for unrelated contracts to fail.
#[test]
fn kaizen_pass_does_not_run_on_other_kinds() {
    let yaml = r#"metadata:
  kind: registry
  version: "1.0.0"
  description: "a registry that happens to carry a top-level status list"
  references:
    - "internal"
status:
  - alpha
  - beta
"#;
    let contract = parse_contract_str(yaml).expect("non-kaizen contract still parses");
    assert!(contract.kaizen_record.is_none());
}

/// Two real corpus records, read from the tree, validate clean. If the
/// metadata blocks added by this change ever drift out of the kind, these fail.
#[test]
fn real_kaizen_records_validate() {
    for yaml in [
        include_str!("../../../../contracts/entrenar/kaizen/backward-cpu-staging-v1.yaml"),
        include_str!("../../../../contracts/trueno/kaizen/blis-safe-alloc-v1.yaml"),
        include_str!("../../../../contracts/entrenar/kaizen/gpu-workspace-clip-v1.yaml"),
        include_str!(
            "../../../../contracts/entrenar/kaizen/gradient-accumulation-canary-v1.yaml"
        ),
        include_str!("../../../../contracts/entrenar/kaizen/vram-guard-v1.yaml"),
    ] {
        let contract = parse_contract_str(yaml).expect("real kaizen record parses");
        assert_eq!(contract.kind(), ContractKind::Kaizen);
        let errors: Vec<_> = validate_contract(&contract)
            .into_iter()
            .filter(|v| v.severity == crate::error::Severity::Error)
            .map(|v| format!("{}: {}", v.rule, v.message))
            .collect();
        assert!(errors.is_empty(), "real record rejected: {errors:?}");
    }
}
