// CRUX competitive-research metadata domains — aprender#2555.
//
// Before this suite, a crux-shaped contract carrying
//
//     competitor: THIS-COMPETITOR-DOES-NOT-EXIST
//     demand_score: 99999
//     intake_status: banana
//
// passed `pv validate` with 0 errors and exit 0. Root cause: `Metadata`
// declared none of the three fields, so serde dropped them silently and no
// validator could check a field it never parsed.
//
// Included from `validator_tests.rs`, so `super::*` is `schema::validator`.

use super::*;
use crate::schema::types::IntakeStatus;
use crate::schema::{parse_contract_str, validate_contract};

/// A minimal crux-shaped contract with the three metadata fields templated in.
fn crux_contract(competitor: &str, demand_score: &str, intake_status: &str) -> String {
    format!(
        r#"
metadata:
  version: "1.0.0"
  description: "CRUX intake domain test"
  kind: registry
  registry: true
  references:
    - "contracts/crux-competitive-research-ux-v1.yaml"
  competitor: {competitor}
  demand_score: {demand_score}
  intake_status: {intake_status}
equations: {{}}
proof_obligations: []
falsification_tests: []
"#
    )
}

fn crux_errors(yaml: &str) -> Vec<Violation> {
    let contract = parse_contract_str(yaml).expect("contract parses");
    validate_contract(&contract)
        .into_iter()
        .filter(|v| v.severity == Severity::Error && v.rule.starts_with("CRUX-"))
        .collect()
}

// ── The exact reproducer from aprender#2555 ──────────────────────────────────

/// THE falsifier. On 4bbfeb07f this contract validated with 0 errors, exit 0.
#[test]
fn the_2555_reproducer_is_rejected() {
    let yaml = crux_contract("THIS-COMPETITOR-DOES-NOT-EXIST", "99999", "banana");
    // `intake_status: banana` is rejected first, at PARSE time — the contract
    // never becomes a `Contract` at all.
    let err = parse_contract_str(&yaml).expect_err("nonsense must not parse");
    let msg = err.to_string();
    assert!(
        msg.contains("intake_status") && msg.contains("banana"),
        "parse error must name the offending field and value, got: {msg}"
    );

    // With only the parse-level offender repaired, the other two still fail.
    let yaml = crux_contract("THIS-COMPETITOR-DOES-NOT-EXIST", "99999", "missing");
    let rules: Vec<_> = crux_errors(&yaml).iter().map(|v| v.rule.clone()).collect();
    assert!(rules.contains(&"CRUX-001".to_string()), "got {rules:?}");
    assert!(rules.contains(&"CRUX-002".to_string()), "got {rules:?}");
}

// ── intake_status: a closed enum, enforced at PARSE time ─────────────────────

/// An invented `intake_status` must fail to PARSE, not merely lint. A lint is
/// advisory and a caller may ignore it; a parse failure cannot be ignored.
#[test]
fn invented_intake_status_fails_to_parse() {
    for bad in ["banana", "implemented", "done", "SUPPORTED", ""] {
        let yaml = crux_contract("ollama", "3", &format!("\"{bad}\""));
        let err = parse_contract_str(&yaml)
            .err()
            .unwrap_or_else(|| panic!("intake_status {bad:?} must not parse"));
        assert!(
            err.to_string().contains("intake_status"),
            "parse error for {bad:?} must name intake_status, got: {err}"
        );
    }
}

/// The closed vocabulary is exactly `STATUS_BADGE` in
/// `scripts/crux_scaffold_contracts.py`, the generator that emits all 275
/// `crux-*-v1.yaml` files.
#[test]
fn the_four_intake_statuses_parse_and_round_trip() {
    let expected = [
        ("missing", IntakeStatus::Missing),
        ("partial", IntakeStatus::Partial),
        ("supported", IntakeStatus::Supported),
        ("unclear", IntakeStatus::Unclear),
    ];
    for (word, variant) in expected {
        let yaml = crux_contract("ollama", "3", word);
        let contract = parse_contract_str(&yaml)
            .unwrap_or_else(|e| panic!("intake_status {word:?} must parse: {e}"));
        assert_eq!(contract.metadata.intake_status, Some(variant));
        assert_eq!(variant.to_string(), word);
        assert!(crux_errors(&yaml).is_empty());
    }
}

/// The field is optional: the ~1,450 contracts in `contracts/` that are not
/// crux stories carry none of the three and must be unaffected.
#[test]
fn absent_crux_fields_produce_no_violations() {
    let yaml = r#"
metadata:
  version: "1.0.0"
  description: "No crux fields"
  kind: registry
  registry: true
  references:
    - "Paper (2024)"
equations: {}
proof_obligations: []
falsification_tests: []
"#;
    let contract = parse_contract_str(yaml).expect("parses");
    assert_eq!(contract.metadata.intake_status, None);
    assert_eq!(contract.metadata.demand_score, None);
    assert_eq!(contract.metadata.competitor, None);
    assert!(crux_errors(yaml).is_empty());
}

// ── CRUX-001: demand_score ∈ 1..=5 ───────────────────────────────────────────

/// `demand_score` is the ranking signal the competitive-research programme
/// sorts by, so an out-of-range value silently dominates every ranking.
#[test]
fn out_of_range_demand_score_is_an_error() {
    for bad in ["99999", "0", "-1", "6", "1000000000000"] {
        let yaml = crux_contract("ollama", bad, "missing");
        let errors = crux_errors(&yaml);
        assert!(
            errors.iter().any(|v| v.rule == "CRUX-001"),
            "demand_score {bad} must raise CRUX-001, got {errors:?}"
        );
        assert!(
            errors.iter().any(|v| v.message.contains(bad)),
            "CRUX-001 message must quote the offending value {bad}, got {errors:?}"
        );
    }
}

/// Both bounds are INCLUSIVE — an off-by-one at the top would reject the 86
/// contracts carrying demand_score 5; one at the bottom would admit 0.
#[test]
fn demand_score_bounds_are_inclusive() {
    for good in ["1", "2", "3", "4", "5"] {
        let yaml = crux_contract("ollama", good, "missing");
        assert!(
            crux_errors(&yaml).is_empty(),
            "demand_score {good} is inside 1..=5 and must be accepted"
        );
    }
}

// ── CRUX-002: competitor ∈ CRUX_COMPETITORS ──────────────────────────────────

#[test]
fn unknown_competitor_is_an_error() {
    // The literal is the YAML scalar, so the empty string is written `''`.
    for bad in [
        "THIS-COMPETITOR-DOES-NOT-EXIST",
        "HuggingFace",
        "llama.cpp",
        "openai",
        "''",
    ] {
        let yaml = crux_contract(&format!("'{}'", bad.trim_matches('\'')), "3", "missing");
        let errors = crux_errors(&yaml);
        assert!(
            errors.iter().any(|v| v.rule == "CRUX-002"),
            "competitor {bad:?} must raise CRUX-002, got {errors:?}"
        );
    }
}

/// Every registry member must be accepted. This is the anti-typo direction: a
/// mangled entry in `CRUX_COMPETITORS` would red here rather than silently
/// rejecting real contracts.
#[test]
fn every_registered_competitor_is_accepted() {
    for competitor in CRUX_COMPETITORS {
        let yaml = crux_contract(&format!("\"{competitor}\""), "3", "missing");
        assert!(
            crux_errors(&yaml).is_empty(),
            "registered competitor {competitor:?} must validate"
        );
    }
}

/// The registry is the corpus vocabulary, so it must not silently shrink.
#[test]
fn competitor_registry_covers_the_corpus_vocabulary() {
    for required in [
        "huggingface",
        "pytorch",
        "llama_cpp",
        "vllm",
        "ecosystem",
        "ollama",
        "openclaw",
        "hf-kernels-community",
        "apr-qa-playbook",
        "openclip",
        "none",
    ] {
        assert!(
            CRUX_COMPETITORS.contains(&required),
            "{required} is used by contracts/ and must stay in CRUX_COMPETITORS"
        );
    }
}

// ── Why CRUX_COMPETITORS is NOT BEAT_INCUMBENTS ──────────────────────────────

/// Reusing `BEAT_INCUMBENTS` for `metadata.competitor` was considered and
/// REJECTED. This test pins the measurement so the decision cannot be undone by
/// a well-meaning "these two lists are redundant" refactor.
///
/// `BEAT_INCUMBENTS` answers "whom does aprender claim to BEAT on a pinned
/// benchmark"; `competitor` answers "whose UX was this story extracted from".
/// Under the BEAT matcher, the largest sources in the corpus are unnameable.
#[test]
fn beat_incumbents_cannot_name_the_crux_corpus() {
    let beat_accepts = |c: &str| BEAT_INCUMBENTS.iter().any(|p| c.to_lowercase().contains(p));

    // 88 contracts, the single largest source — and not a BEAT pillar at all.
    assert!(!beat_accepts("huggingface"));
    // 32 contracts.
    assert!(!beat_accepts("vllm"));
    // 37 contracts. The BEAT list spells this pillar "llama.cpp"; "llama.cpp"
    // is not a substring of "llama_cpp", so even a named pillar is missed
    // under the spelling the crux corpus actually uses.
    assert!(!beat_accepts("llama_cpp"));
    assert!(beat_accepts("llama.cpp"));
    // The remaining corpus sources.
    for c in [
        "ecosystem",
        "openclaw",
        "hf-kernels-community",
        "apr-qa-playbook",
        "openclip",
        "none",
    ] {
        assert!(!beat_accepts(c), "BEAT_INCUMBENTS unexpectedly accepts {c}");
    }
    // Only these two overlap.
    assert!(beat_accepts("pytorch") && beat_accepts("ollama"));

    // Conversely, the BEAT pillars are not automatically research sources: a
    // crux story naming one is a deliberate registry edit, not an accident.
    assert!(!CRUX_COMPETITORS.contains(&"scikit-learn"));
    assert!(!CRUX_COMPETITORS.contains(&"unsloth"));
}

// ── Corpus regression anchors ────────────────────────────────────────────────

/// A real scaffolded crux contract still parses and validates.
#[test]
fn real_crux_contract_still_validates() {
    let yaml = include_str!("../../../../contracts/crux-I-10-v1.yaml");
    let contract = parse_contract_str(yaml).expect("crux-I-10 parses");
    assert_eq!(contract.metadata.competitor.as_deref(), Some("ecosystem"));
    assert_eq!(contract.metadata.demand_score, Some(4));
    assert_eq!(contract.metadata.intake_status, Some(IntakeStatus::Partial));
    assert!(crux_errors(yaml).is_empty());
}

/// Regression anchor for the one real drift this gate found: on 4bbfeb07f,
/// `contracts/apr-lint-producers-v1.yaml` carried `intake_status: implemented`,
/// a word outside the generator's badge vocabulary. Nothing could see it
/// because nothing parsed the field.
#[test]
fn apr_lint_producers_carries_a_vocabulary_intake_status() {
    let yaml = include_str!("../../../../contracts/apr-lint-producers-v1.yaml");
    let contract = parse_contract_str(yaml).expect("apr-lint-producers parses");
    assert_eq!(
        contract.metadata.intake_status,
        Some(IntakeStatus::Supported)
    );
    assert!(crux_errors(yaml).is_empty());
}
