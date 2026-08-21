    // Validator rules PARITY-000..010 for `metadata.kind: competitive-parity`.
    // `include!`d from validator.rs.
    use super::*;
    use crate::schema::parse_contract_str;

    /// A minimal, fully-valid competitive-parity contract. Tests mutate ONE
    /// field of it and assert the specific rule fires -- so every rule below is
    /// mutation-verified in both directions by construction.
    fn good_yaml() -> String {
        r#"
metadata:
  version: "1.0.0"
  description: "competitive parity ledger fixture"
  kind: competitive-parity
  references:
    - "docs/BEATS.md"
proof_obligations:
  - type: invariant
    property: "every in-scope entry point has a dated verdict"
falsification_tests:
  - id: FALSIFY-PARITY-FIXTURE-001
    rule: "freshness"
    prediction: "an expired row degrades to UNMEASURED"
    if_fails: "the freshness bound is not evaluated at check time"
parity:
  scope: "fixture"
  rows:
    - entry_point: "apr run --gpu"
      competitor: "ollama"
      competitor_version: "0.12.4"
      invocation_apr: "apr run m.gguf --gpu"
      invocation_competitor: "ollama run qwen2.5-coder:1.5b-instruct-q4_K_M"
      dimension: "decode_tok_s"
      verdict: PARITY
      measured_on: "2026-07-31"
      valid_until: "2026-11-30"
      owner: "pillar-4"
      evidence: "contracts/beat-ollama-decode-throughput-speed-v1.yaml"
"#
        .to_string()
    }

    fn errors_of(yaml: &str) -> Vec<Violation> {
        let c = parse_contract_str(yaml).expect("fixture parses");
        validate_contract(&c)
            .into_iter()
            .filter(|v| v.severity == Severity::Error)
            .collect()
    }

    fn rules(yaml: &str) -> Vec<String> {
        errors_of(yaml).into_iter().map(|v| v.rule).collect()
    }

    // -- the GREEN direction ------------------------------------------------

    #[test]
    fn the_fixture_is_valid() {
        let c = parse_contract_str(&good_yaml()).expect("parses");
        assert_eq!(c.kind(), ContractKind::CompetitiveParity);
        assert!(errors_of(&good_yaml()).is_empty(), "fixture: {:?}", errors_of(&good_yaml()));
    }

    /// FATAL 1, half one: the kind CARRIES the provability obligation. Removing
    /// the obligations must turn it RED -- for every other non-kernel kind it
    /// would stay green.
    #[test]
    fn competitive_parity_requires_proofs() {
        let c = parse_contract_str(&good_yaml()).expect("parses");
        assert!(c.requires_proofs(), "the kind must carry PROVABILITY-001");

        let stripped = good_yaml().replace(
            r#"proof_obligations:
  - type: invariant
    property: "every in-scope entry point has a dated verdict""#,
            "proof_obligations: []",
        );
        let r = rules(&stripped);
        assert!(
            r.contains(&"PROVABILITY-001".to_string()),
            "a ledger with no obligations must be REJECTED, got {r:?}"
        );
    }

    /// A ledger that asserts more than it can falsify is rejected.
    #[test]
    fn falsifiers_may_not_be_fewer_than_obligations() {
        let y = good_yaml().replace(
            "falsification_tests:",
            r#"proof_obligations_extra_marker: true
falsification_tests:"#,
        );
        // Add a second obligation without a second falsifier.
        let y = y.replace(
            r#"    property: "every in-scope entry point has a dated verdict""#,
            r#"    property: "every in-scope entry point has a dated verdict"
  - type: invariant
    property: "every verdict class carries a freshness bound""#,
        );
        let r = rules(&y);
        assert!(
            r.contains(&"PROVABILITY-001".to_string()),
            "2 obligations / 1 falsifier must be REJECTED, got {r:?}"
        );
    }

    /// FATAL 1, half two: `registry: true` is the exemption 481 contracts use.
    /// On this kind it must be a hard ERROR, and the provability invariant must
    /// still run underneath it.
    #[test]
    fn registry_true_cannot_exempt_a_parity_contract() {
        let y = good_yaml().replace(
            "  kind: competitive-parity",
            "  kind: competitive-parity\n  registry: true",
        );
        let c = parse_contract_str(&y).expect("parses");

        // The kind is NOT rewritten to Registry the way `kernel` would be.
        assert_eq!(c.kind(), ContractKind::CompetitiveParity);
        assert!(c.requires_proofs(), "registry: true must not buy an exemption");

        let r = rules(&y);
        assert!(
            r.contains(&"PARITY-000".to_string()),
            "registry: true must be rejected outright, got {r:?}"
        );
    }

    /// The control: on a KERNEL contract, `registry: true` DOES exempt. This is
    /// the behaviour that made the naive design useless, asserted here so the
    /// contrast is a test rather than a claim in a comment.
    #[test]
    fn registry_true_still_exempts_a_kernel_contract() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "registry-flagged kernel"
  registry: true
  references:
    - "somewhere"
"#;
        let c = parse_contract_str(yaml).expect("parses");
        assert_eq!(c.kind(), ContractKind::Registry);
        assert!(
            !c.requires_proofs(),
            "the kernel+registry exemption is the defect being routed around"
        );
        assert!(c.provability_violations().is_empty());
    }

    // -- the RED direction, one rule at a time ------------------------------

    #[test]
    fn parity_001_missing_ledger() {
        let y = good_yaml();
        let cut = y.find("parity:").expect("has ledger");
        let r = rules(&y[..cut]);
        assert!(r.contains(&"PARITY-001".to_string()), "got {r:?}");
    }

    #[test]
    fn parity_001_empty_rows() {
        let y = good_yaml();
        let cut = y.find("parity:").expect("has ledger");
        let r = rules(&format!("{}parity:\n  rows: []\n", &y[..cut]));
        assert!(r.contains(&"PARITY-001".to_string()), "got {r:?}");
    }

    #[test]
    fn parity_002_missing_entry_point() {
        let r = rules(&good_yaml().replace(r#"entry_point: "apr run --gpu""#, r#"entry_point: """#));
        assert!(r.contains(&"PARITY-002".to_string()), "got {r:?}");
    }

    /// Two rows for one entry point let a favourable measurement mask an
    /// unfavourable one, which is deletion with extra steps.
    #[test]
    fn parity_002_duplicate_entry_point() {
        let y = good_yaml();
        let (head, row) = y.split_at(y.find("    - entry_point:").expect("has a row"));
        let r = rules(&format!("{head}{row}{row}"));
        assert!(r.contains(&"PARITY-002".to_string()), "got {r:?}");
    }

    #[test]
    fn parity_003_missing_competitor() {
        let r = rules(&good_yaml().replace(r#"competitor: "ollama""#, r#"competitor: """#));
        assert!(r.contains(&"PARITY-003".to_string()), "got {r:?}");
    }

    /// PARITY-004 is the rule the whole repo fails today: exactly ONE
    /// comparison in-tree pins an exact competitor version.
    #[test]
    fn parity_004_rejects_every_shape_of_unpinned_version() {
        for bad in [
            "",
            "latest",
            "unpinned",
            "whatever uv resolves",
            "llama.cpp@main",
            "HEAD",
            "current release",
            "scikit-learn",
        ] {
            let y = good_yaml().replace(
                r#"competitor_version: "0.12.4""#,
                &format!("competitor_version: {bad:?}"),
            );
            let r = rules(&y);
            assert!(
                r.contains(&"PARITY-004".to_string()),
                "{bad:?} must be rejected as unpinned, got {r:?}"
            );
        }
    }

    #[test]
    fn parity_004_accepts_real_pins() {
        for good in [
            "0.12.4",
            "bitsandbytes==0.49.2",
            "b4589 (0f1d51b2c9e4)",
            "sha256:6b3f2c1d9a0e4f5b8c7d",
            "1.9.0",
        ] {
            let y = good_yaml().replace(
                r#"competitor_version: "0.12.4""#,
                &format!("competitor_version: {good:?}"),
            );
            assert!(errors_of(&y).is_empty(), "{good:?} is a real pin: {:?}", errors_of(&y));
        }
    }

    #[test]
    fn parity_005_missing_either_invocation() {
        let a = rules(&good_yaml().replace(
            r#"invocation_apr: "apr run m.gguf --gpu""#,
            r#"invocation_apr: """#,
        ));
        assert!(a.contains(&"PARITY-005".to_string()), "got {a:?}");
        let b = rules(&good_yaml().replace(
            r#"invocation_competitor: "ollama run qwen2.5-coder:1.5b-instruct-q4_K_M""#,
            r#"invocation_competitor: """#,
        ));
        assert!(b.contains(&"PARITY-005".to_string()), "got {b:?}");
    }

    #[test]
    fn parity_006_missing_verdict() {
        let r = rules(&good_yaml().replace("      verdict: PARITY\n", ""));
        assert!(r.contains(&"PARITY-006".to_string()), "got {r:?}");
    }

    /// A verdict outside the closed vocabulary never reaches the validator --
    /// serde refuses the document. Asserted here so the guarantee is tested at
    /// the surface a contract author actually touches.
    #[test]
    fn a_verdict_outside_the_vocabulary_does_not_parse() {
        let y = good_yaml().replace("verdict: PARITY", "verdict: MOSTLY_BETTER");
        assert!(
            parse_contract_str(&y).is_err(),
            "an invented verdict must not parse"
        );
    }

    /// Every verdict class is accepted. WORSE and UNMEASURED are first-class:
    /// if the schema rejected them, deletion would again be the cheapest way to
    /// comply.
    #[test]
    fn every_verdict_class_is_recordable() {
        for v in ["BETTER", "PARITY", "WORSE", "NOT_COMPARABLE", "UNMEASURED"] {
            let y = good_yaml().replace("verdict: PARITY", &format!("verdict: {v}"));
            assert!(errors_of(&y).is_empty(), "{v} must be recordable: {:?}", errors_of(&y));
        }
    }

    /// FATAL 3: the freshness bound is required on EVERY verdict class, not
    /// only UNMEASURED. A BETTER row with no `valid_until` is exactly the shape
    /// the withdrawn 1.371x claim had.
    #[test]
    fn parity_007_requires_valid_until_on_every_verdict_class() {
        for v in ["BETTER", "PARITY", "WORSE", "NOT_COMPARABLE", "UNMEASURED"] {
            let y = good_yaml()
                .replace("verdict: PARITY", &format!("verdict: {v}"))
                .replace("      valid_until: \"2026-11-30\"\n", "");
            let r = rules(&y);
            assert!(
                r.contains(&"PARITY-007".to_string()),
                "{v} row with no valid_until must be REJECTED, got {r:?}"
            );
        }
    }

    #[test]
    fn parity_007_rejects_non_dates_and_impossible_dates() {
        for bad in ["soon", "2026-13-01", "2026-02-30", "26-11-30", "2026-11-31"] {
            let y = good_yaml().replace(
                r#"valid_until: "2026-11-30""#,
                &format!("valid_until: {bad:?}"),
            );
            let r = rules(&y);
            assert!(
                r.contains(&"PARITY-007".to_string()),
                "{bad:?} must be rejected, got {r:?}"
            );
        }
    }

    #[test]
    fn parity_007_requires_measured_on() {
        let r = rules(&good_yaml().replace("      measured_on: \"2026-07-31\"\n", ""));
        assert!(r.contains(&"PARITY-007".to_string()), "got {r:?}");
    }

    /// A row that expires on or before the day it was taken is permanently
    /// stale -- a way to write a compliant row that can never count.
    #[test]
    fn parity_007_valid_until_must_follow_measured_on() {
        for until in ["2026-07-31", "2026-07-30"] {
            let y = good_yaml().replace(
                r#"valid_until: "2026-11-30""#,
                &format!("valid_until: {until:?}"),
            );
            let r = rules(&y);
            assert!(
                r.contains(&"PARITY-007".to_string()),
                "valid_until {until:?} <= measured_on must be rejected, got {r:?}"
            );
        }
    }

    #[test]
    fn parity_008_missing_owner() {
        let r = rules(&good_yaml().replace(r#"owner: "pillar-4""#, r#"owner: """#));
        assert!(r.contains(&"PARITY-008".to_string()), "got {r:?}");
    }

    #[test]
    fn parity_009_missing_evidence() {
        let r = rules(&good_yaml().replace(
            r#"evidence: "contracts/beat-ollama-decode-throughput-speed-v1.yaml""#,
            r#"evidence: """#,
        ));
        assert!(r.contains(&"PARITY-009".to_string()), "got {r:?}");
    }

    #[test]
    fn parity_010_missing_dimension() {
        let r = rules(&good_yaml().replace(r#"dimension: "decode_tok_s""#, r#"dimension: """#));
        assert!(r.contains(&"PARITY-010".to_string()), "got {r:?}");
    }

    // -- the shipped ledger --------------------------------------------------

    /// The real ledger validates, uses the new kind, and -- the point of the
    /// whole exercise -- carries at least three NON-WIN rows. A mechanism whose
    /// first five rows are all wins is untested in the direction that matters.
    #[test]
    fn shipped_ledger_validates_and_records_losses() {
        let yaml = include_str!("../../../../contracts/apr-competitive-parity-v1.yaml");
        let c = parse_contract_str(yaml).expect("shipped ledger parses");
        assert_eq!(c.kind(), ContractKind::CompetitiveParity);
        let errs: Vec<_> = validate_contract(&c)
            .into_iter()
            .filter(|v| v.severity == Severity::Error)
            .collect();
        assert!(errs.is_empty(), "shipped ledger has errors: {errs:?}");

        let ledger = c.parity.as_ref().expect("has a ledger");
        assert!(ledger.rows.len() >= 5, "seed at least five rows");

        let losses = ledger
            .rows
            .iter()
            .filter(|r| {
                matches!(
                    r.verdict,
                    Some(crate::schema::Verdict::Worse)
                        | Some(crate::schema::Verdict::Unmeasured)
                        | Some(crate::schema::Verdict::NotComparable)
                )
            })
            .count();
        assert!(
            losses >= 3,
            "the ledger must seed at least three NON-WINS; found {losses}"
        );
    }
