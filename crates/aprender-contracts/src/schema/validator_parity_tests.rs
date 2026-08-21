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
    test: "cargo test -p aprender-contracts --lib parity_"
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

    // ======================================================================
    // DEFECT 2 -- `valid_until` had NO CEILING, so 2099-12-31 was the new
    // exemption. Check-time freshness is only as strong as the dates it reads.
    // ======================================================================

    /// The mutation the reviewer actually ran: rewrite `valid_until` to
    /// 2099-12-31. It used to leave `pv validate` green, which made "staleness
    /// blocks" voluntary. It must now be an ERROR.
    #[test]
    fn parity_011_rejects_the_2099_exemption() {
        let y = good_yaml().replace(
            r#"valid_until: "2026-11-30""#,
            r#"valid_until: "2099-12-31""#,
        );
        assert_ne!(y, good_yaml(), "the mutation must actually apply");
        assert!(
            rules(&y).contains(&"PARITY-011".to_string()),
            "2099-12-31 must be refused: {:?}",
            rules(&y)
        );
    }

    /// The boundary is checked from BOTH sides, so the ceiling cannot be
    /// off-by-one or silently disabled. measured_on 2026-07-31 + 180d =
    /// 2027-01-27 (accepted); +181d = 2027-01-28 (refused).
    #[test]
    fn parity_011_boundary_is_exact_on_both_sides() {
        let at = |d: &str| {
            good_yaml().replace(
                r#"valid_until: "2026-11-30""#,
                &format!(r#"valid_until: "{d}""#),
            )
        };
        assert_eq!(
            crate::schema::parity::days_between("2026-07-31", "2027-01-27"),
            Some(180)
        );
        assert!(
            !rules(&at("2027-01-27")).contains(&"PARITY-011".to_string()),
            "exactly 180 days must be ACCEPTED: {:?}",
            rules(&at("2027-01-27"))
        );
        assert!(
            rules(&at("2027-01-28")).contains(&"PARITY-011".to_string()),
            "181 days must be REFUSED: {:?}",
            rules(&at("2027-01-28"))
        );
    }

    /// The ceiling is anchored to `measured_on`, not to today: an OLD but
    /// honest measurement must stay writable. It is `is_expired` that then
    /// degrades it, not this rule.
    #[test]
    fn parity_011_anchors_to_measured_on_not_today() {
        let y = good_yaml()
            .replace(r#"measured_on: "2026-07-31""#, r#"measured_on: "2019-01-01""#)
            .replace(r#"valid_until: "2026-11-30""#, r#"valid_until: "2019-03-01""#);
        assert!(
            !rules(&y).contains(&"PARITY-011".to_string()),
            "a short window on an ancient measurement is honest, not a violation: {:?}",
            rules(&y)
        );
        // ... and it is EXPIRED as of today, which is where it must be caught.
        let c = parse_contract_str(&y).expect("parses");
        let l = c.parity.as_ref().expect("ledger");
        assert_eq!(l.measured_count("2026-08-21"), 0);
    }

    // ======================================================================
    // DEFECT 3 -- the honest DOWNGRADE. `MEASURED_MIN` had no give, so filing
    // the `apr code` row as UNMEASURED (which its own note says it should be)
    // was mechanically forbidden. PARITY-012..014 make the correction possible
    // WITHOUT making silent regression free.
    // ======================================================================

    /// Turn the fixture's single row UNMEASURED and record a downgrade for it.
    fn downgraded_yaml(extra: &str) -> String {
        good_yaml().replace("      verdict: PARITY\n", "      verdict: UNMEASURED\n")
            + &format!("  downgrades:\n    - entry_point: \"apr run --gpu\"\n{extra}")
    }

    const GOOD_DOWNGRADE: &str = concat!(
        "      reason: RECEIPT_MISSING\n",
        "      owner: \"pillar-4\"\n",
        "      recorded_on: \"2026-08-21\"\n",
        "      recheck_by: \"2026-10-31\"\n",
    );

    /// (a) DOWNGRADING WITH A RECORDED REASON PASSES.
    #[test]
    fn downgrade_with_a_recorded_reason_is_accepted() {
        let y = downgraded_yaml(GOOD_DOWNGRADE);
        assert!(
            errors_of(&y).is_empty(),
            "an owned, dated, reasoned downgrade must be legal: {:?}",
            errors_of(&y)
        );
        let c = parse_contract_str(&y).expect("parses");
        let l = c.parity.as_ref().expect("ledger");
        assert_eq!(l.downgrades.len(), 1);
        assert_eq!(
            l.measured_count("2026-08-21"),
            0,
            "it really is UNMEASURED now"
        );
        assert!(l.downgrade_for("apr run --gpu").is_some());
    }

    /// (b1) DOWNGRADING WITH NO REASON FAILS.
    #[test]
    fn parity_013_downgrade_without_a_reason_is_refused() {
        let y = downgraded_yaml(concat!(
            "      owner: \"pillar-4\"\n",
            "      recorded_on: \"2026-08-21\"\n",
            "      recheck_by: \"2026-10-31\"\n",
        ));
        assert!(
            rules(&y).contains(&"PARITY-013".to_string()),
            "a reasonless downgrade must be refused: {:?}",
            rules(&y)
        );
    }

    /// The reason vocabulary is CLOSED -- prose does not parse, so "recorded a
    /// reason" cannot be discharged by writing a sentence.
    #[test]
    fn downgrade_reason_vocabulary_is_closed() {
        let y = downgraded_yaml(concat!(
            "      reason: \"we were busy\"\n",
            "      owner: \"pillar-4\"\n",
            "      recorded_on: \"2026-08-21\"\n",
            "      recheck_by: \"2026-10-31\"\n",
        ));
        assert!(
            parse_contract_str(&y).is_err(),
            "an invented reason must fail to PARSE, not merely lint"
        );
    }

    /// (b2) A DOWNGRADE CANNOT LAUNDER A DELETION. This is the hinge: without
    /// it, "delete the row, record a downgrade" would be cheaper than measuring
    /// and PMAT-733 would be back.
    #[test]
    fn parity_012_downgrade_naming_no_live_row_is_refused() {
        let y = good_yaml().replace("      verdict: PARITY\n", "      verdict: UNMEASURED\n")
            + &format!(
                "  downgrades:\n    - entry_point: \
                 \"lib:aprender-core::StandardScaler::fit_transform\"\n{GOOD_DOWNGRADE}"
            );
        assert!(
            rules(&y).contains(&"PARITY-012".to_string()),
            "a downgrade for a row that is not in the ledger must be refused: {:?}",
            rules(&y)
        );
    }

    /// A downgrade recorded against a row still CLAIMING a measurement is a
    /// pre-authorisation nobody reviewed.
    #[test]
    fn parity_014_downgrade_for_a_still_measured_row_is_refused() {
        let y = good_yaml()
            + &format!("  downgrades:\n    - entry_point: \"apr run --gpu\"\n{GOOD_DOWNGRADE}");
        assert!(
            rules(&y).contains(&"PARITY-014".to_string()),
            "a downgrade beside a live PARITY verdict must be refused: {:?}",
            rules(&y)
        );
    }

    #[test]
    fn parity_012_duplicate_downgrade_is_refused() {
        let y = downgraded_yaml(GOOD_DOWNGRADE)
            + &format!("    - entry_point: \"apr run --gpu\"\n{GOOD_DOWNGRADE}");
        assert!(
            rules(&y).contains(&"PARITY-012".to_string()),
            "the same row cannot be downgraded twice: {:?}",
            rules(&y)
        );
    }

    #[test]
    fn parity_013_downgrade_without_an_owner_is_refused() {
        let y = downgraded_yaml(concat!(
            "      reason: RECEIPT_MISSING\n",
            "      recorded_on: \"2026-08-21\"\n",
            "      recheck_by: \"2026-10-31\"\n",
        ));
        assert!(
            rules(&y).contains(&"PARITY-013".to_string()),
            "an unowned downgrade is a permanent one: {:?}",
            rules(&y)
        );
    }

    /// A downgrade dated far enough ahead is a deletion that kept its
    /// paperwork. The same ceiling as PARITY-011, for the same reason.
    #[test]
    fn parity_013_downgrade_cannot_be_dated_to_2099() {
        let y = downgraded_yaml(concat!(
            "      reason: RECEIPT_MISSING\n",
            "      owner: \"pillar-4\"\n",
            "      recorded_on: \"2026-08-21\"\n",
            "      recheck_by: \"2099-12-31\"\n",
        ));
        assert!(
            rules(&y).contains(&"PARITY-013".to_string()),
            "an unbounded recheck_by must be refused: {:?}",
            rules(&y)
        );
    }

    #[test]
    fn parity_013_recheck_by_must_follow_recorded_on() {
        let y = downgraded_yaml(concat!(
            "      reason: RECEIPT_MISSING\n",
            "      owner: \"pillar-4\"\n",
            "      recorded_on: \"2026-08-21\"\n",
            "      recheck_by: \"2026-08-01\"\n",
        ));
        assert!(
            rules(&y).contains(&"PARITY-013".to_string()),
            "a backdated recheck is not a recheck: {:?}",
            rules(&y)
        );
    }

    // ======================================================================
    // The provability invariant was discharged by PROSE.
    // ======================================================================

    /// `falsification_tests.len() >= proof_obligations.len()` is satisfiable by
    /// writing more YAML unless each entry names something RUNNABLE.
    #[test]
    fn parity_015_prose_only_falsification_entry_is_refused() {
        let y = good_yaml().replace(
            "    test: \"cargo test -p aprender-contracts --lib parity_\"\n",
            "",
        );
        assert_ne!(y, good_yaml(), "the mutation must actually apply");
        assert!(
            rules(&y).contains(&"PARITY-015".to_string()),
            "an unbound falsification entry must be refused: {:?}",
            rules(&y)
        );
    }

    /// `test_harness:` is the other accepted spelling (619 in-tree entries use
    /// it), so the rule must not force a rewrite of the whole corpus.
    #[test]
    fn parity_015_accepts_test_harness_spelling() {
        let y = good_yaml().replace(
            "    test: \"cargo test -p aprender-contracts --lib parity_\"",
            "    test_harness: \"bash scripts/check_competitive_parity.sh --self-test\"",
        );
        assert!(
            !rules(&y).contains(&"PARITY-015".to_string()),
            "test_harness must count as a binding: {:?}",
            rules(&y)
        );
    }

    /// An EMPTY binding is not a binding.
    #[test]
    fn parity_015_empty_binding_is_not_a_binding() {
        let y = good_yaml().replace(
            "    test: \"cargo test -p aprender-contracts --lib parity_\"",
            "    test: \"   \"",
        );
        assert!(
            rules(&y).contains(&"PARITY-015".to_string()),
            "whitespace is not a command: {:?}",
            rules(&y)
        );
    }
