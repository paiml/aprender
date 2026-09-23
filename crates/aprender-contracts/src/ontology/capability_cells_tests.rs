//! ONT-4c5 case table: every row names the defect it would let through if it went green.

use super::*;
use crate::ontology::verdict::Reason;

const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

fn rung(id: &str, sha: &str, hosts: &[&str], required: bool) -> Rung {
    Rung {
        id: id.to_string(),
        sha256: sha.to_string(),
        arch: "qwen2".into(),
        gguf: format!("{id}.gguf"),
        backends: vec!["cpu".into(), "cuda".into()],
        hosts: hosts.iter().map(|h| (*h).to_string()).collect(),
        required,
        contract: "model-capability-ladder-v1".into(),
    }
}

/// One rung row: `green` drives the ladder-green predicate; `extra` is spliced in (labels, a wrong sha, …).
fn row(id: &str, sha: &str, green: bool, extra: &str) -> String {
    format!(
        r#"{{"id":"{id}","present":true,"sha_ok":true,"sha256":"{sha}","required":true,"green":{green},"capability_match":{{"passed":true,"skipped":false}},"backends":{{"cpu":{{"ran":true,"fallback":false}},"cuda":{{"ran":true,"fallback":false}}}}{extra}}}"#
    )
}

fn receipt(version: &str, host: &str, rows: &[String]) -> Receipt {
    let file = format!("evidence/dogfood/models/{version}/{host}.json");
    let text = format!(
        r#"{{"schema":"apr-model-ladder-receipt/v2","host":"{host}","version":"{version}","sha":"x","cc":"8.9","gpu":"g","rungs":[{}]}}"#,
        rows.join(",")
    );
    receipts::parse(&file, &text).expect("fixture receipt parses")
}

fn ids(set: &BTreeSet<String>) -> Vec<&str> {
    set.iter().map(String::as_str).collect()
}

const NOT_RUN: Verdict = Verdict::Unknown(Reason::NotRun);

#[test]
fn clean_ladder_every_required_cell_measured_and_fail_is_admitted() {
    let rungs = [rung("a", SHA_A, &[], true), rung("b", SHA_B, &[], true)];
    let recs = [
        receipt(
            "0.69.1",
            "lambda",
            &[row("a", SHA_A, true, ""), row("b", SHA_B, true, "")],
        ),
        receipt(
            "0.69.1",
            "gx10",
            &[row("a", SHA_A, true, ""), row("b", SHA_B, false, "")],
        ),
    ];
    let cc = compute(&rungs, &recs).expect("computes");
    assert_eq!(cc.v_star.as_deref(), Some("0.69.1"));
    assert_eq!(
        ids(&cc.domain),
        ["a@gx10", "a@lambda", "b@gx10", "b@lambda"]
    );
    assert!(cc.not_run.is_empty(), "{:?}", cc.not_run);
    // this row ADMITS Fail: a not-green witness is a verdict, and ladder-green is what refuses it
    assert_eq!(cc.cells["b@gx10"], Verdict::Fail);
    assert_eq!(cc.cells["a@lambda"], Verdict::Pass);
}

#[test]
fn a_missing_cell_is_not_run_never_folded_into_fail() {
    let rungs = [rung("a", SHA_A, &[], true)];
    let recs = [
        receipt("0.69.1", "lambda", &[row("a", SHA_A, true, "")]),
        receipt("0.69.1", "gx10", &[]),
    ];
    let cc = compute(&rungs, &recs).expect("computes");
    assert_eq!(ids(&cc.not_run), ["a@gx10"]);
    assert!(
        !cc.cells.contains_key("a@gx10"),
        "absent is not a Fail cell"
    );
}

#[test]
fn every_not_run_label_beats_a_green_measurement() {
    for label in ["DEFER", "MANUAL", "NO-VERDICT", "SKIP", "REPORT", "WARN"] {
        for key in ["verdict", "label"] {
            let rungs = [rung("a", SHA_A, &[], true)];
            let recs = [receipt(
                "0.69.1",
                "lambda",
                &[row("a", SHA_A, true, &format!(r#","{key}":"{label}""#))],
            )];
            let cc = compute(&rungs, &recs).expect("computes");
            assert_eq!(ids(&cc.not_run), ["a@lambda"], "{key}={label}");
            // the row's three labels are exactly Unknown(NotRun); the other Unknowns keep their own reason,
            // and every one of them is in not_run — no third state reaches the admitted set
            if ["DEFER", "MANUAL", "NO-VERDICT"].contains(&label) {
                assert_eq!(cc.cells["a@lambda"], NOT_RUN, "{key}={label}");
            } else {
                assert!(
                    matches!(cc.cells["a@lambda"], Verdict::Unknown(_)),
                    "{key}={label}"
                );
            }
        }
    }
}

#[test]
fn a_pass_label_never_upgrades_and_a_fail_label_downgrades() {
    let rungs = [rung("a", SHA_A, &[], true)];
    let pass_on_red = [receipt(
        "0.69.1",
        "lambda",
        &[row("a", SHA_A, false, r#","verdict":"PASS""#)],
    )];
    assert_eq!(
        compute(&rungs, &pass_on_red).expect("computes").cells["a@lambda"],
        Verdict::Fail
    );
    let fail_on_green = [receipt(
        "0.69.1",
        "lambda",
        &[row("a", SHA_A, true, r#","verdict":"FAIL""#)],
    )];
    assert_eq!(
        compute(&rungs, &fail_on_green).expect("computes").cells["a@lambda"],
        Verdict::Fail
    );
}

#[test]
fn any_not_run_label_among_several_wins() {
    let rungs = [rung("a", SHA_A, &[], true)];
    let recs = [receipt(
        "0.69.1",
        "lambda",
        &[row(
            "a",
            SHA_A,
            true,
            r#","verdict":"PASS","label":"MANUAL""#,
        )],
    )];
    assert_eq!(
        ids(&compute(&rungs, &recs).expect("computes").not_run),
        ["a@lambda"]
    );
}

#[test]
fn an_unknown_label_is_refused_by_name_and_is_not_run() {
    let rungs = [rung("a", SHA_A, &[], true)];
    let recs = [receipt(
        "0.69.1",
        "lambda",
        &[row("a", SHA_A, true, r#","verdict":"DEFERRED""#)],
    )];
    let cc = compute(&rungs, &recs).expect("computes");
    assert_eq!(ids(&cc.not_run), ["a@lambda"]);
    assert_eq!(
        cc.refused_labels,
        ["evidence/dogfood/models/0.69.1/lambda.json: a: DEFERRED"]
    );
}

/// The round-1 grill hole: an older green row must not hide a DEFER (or a missing row) at V*.
#[test]
fn an_older_green_row_never_masks_the_current_release() {
    let rungs = [rung("a", SHA_A, &[], true)];
    let deferred_now = [
        receipt("0.68.2", "lambda", &[row("a", SHA_A, true, "")]),
        receipt(
            "0.69.1",
            "lambda",
            &[row("a", SHA_A, true, r#","verdict":"DEFER""#)],
        ),
    ];
    assert_eq!(
        ids(&compute(&rungs, &deferred_now).expect("computes").not_run),
        ["a@lambda"]
    );
    let missing_now = [
        receipt("0.68.2", "lambda", &[row("a", SHA_A, true, "")]),
        receipt("0.69.1", "lambda", &[]),
    ];
    assert_eq!(
        ids(&compute(&rungs, &missing_now).expect("computes").not_run),
        ["a@lambda"]
    );
}

#[test]
fn a_host_with_only_older_receipts_is_in_the_domain_and_every_cell_on_it_is_not_run() {
    let rungs = [rung("a", SHA_A, &[], true)];
    let recs = [
        receipt("0.69.1", "lambda", &[row("a", SHA_A, true, "")]),
        receipt("0.68.2", "gx10", &[row("a", SHA_A, true, "")]),
    ];
    let cc = compute(&rungs, &recs).expect("computes");
    assert!(cc.domain.contains("a@gx10"));
    assert_eq!(ids(&cc.not_run), ["a@gx10"]);
}

/// V* is not gameable toward green: a stray higher receipt with no rows turns cells NotRun.
#[test]
fn a_stray_higher_version_can_only_turn_cells_not_run() {
    let rungs = [rung("a", SHA_A, &[], true)];
    let recs = [
        receipt("0.69.1", "lambda", &[row("a", SHA_A, true, "")]),
        receipt("0.70.0", "lambda", &[]),
    ];
    let cc = compute(&rungs, &recs).expect("computes");
    assert_eq!(cc.v_star.as_deref(), Some("0.70.0"));
    assert_eq!(ids(&cc.not_run), ["a@lambda"]);
}

#[test]
fn v_star_orders_numerically_not_lexically() {
    let rungs = [rung("a", SHA_A, &[], true)];
    let recs = [
        receipt("0.9.0", "lambda", &[]),
        receipt("0.10.0", "lambda", &[row("a", SHA_A, true, "")]),
    ];
    let cc = compute(&rungs, &recs).expect("computes");
    assert_eq!(cc.v_star.as_deref(), Some("0.10.0"));
    assert!(cc.not_run.is_empty());
}

#[test]
fn a_version_that_is_not_dotted_numerals_is_refused_by_name() {
    let rungs = [rung("a", SHA_A, &[], true)];
    for bad in ["0.69.1-rc1", "", "v0.69", "0..1"] {
        let recs = [receipt(bad, "lambda", &[row("a", SHA_A, true, "")])];
        let err = compute(&rungs, &recs).expect_err(bad);
        assert_eq!(err.version, bad);
        assert!(err.to_string().contains("refused by name"), "{err}");
    }
}

#[test]
fn duplicate_rows_combine_not_run_then_fail_then_pass() {
    let rungs = [rung("a", SHA_A, &[], true)];
    let green_and_defer = [receipt(
        "0.69.1",
        "lambda",
        &[
            row("a", SHA_A, true, ""),
            row("a", SHA_A, true, r#","verdict":"DEFER""#),
        ],
    )];
    assert_eq!(
        ids(&compute(&rungs, &green_and_defer).expect("computes").not_run),
        ["a@lambda"]
    );
    let green_and_red = [receipt(
        "0.69.1",
        "lambda",
        &[row("a", SHA_A, true, ""), row("a", SHA_A, false, "")],
    )];
    let cc = compute(&rungs, &green_and_red).expect("computes");
    assert_eq!(cc.cells["a@lambda"], Verdict::Fail);
    assert!(cc.not_run.is_empty());
    // NOT the lattice meet: a DEFER beside a failing row must stay NotRun, never be admitted as Fail
    let red_and_defer = [receipt(
        "0.69.1",
        "lambda",
        &[
            row("a", SHA_A, false, ""),
            row("a", SHA_A, false, r#","verdict":"DEFER""#),
        ],
    )];
    assert_eq!(
        ids(&compute(&rungs, &red_and_defer).expect("computes").not_run),
        ["a@lambda"]
    );
}

#[test]
fn a_row_without_sha_or_with_a_wrong_hex_is_not_a_cell() {
    let rungs = [rung("a", SHA_A, &[], true)];
    let wrong_hex = [receipt("0.69.1", "lambda", &[row("a", SHA_B, true, "")])];
    assert_eq!(
        ids(&compute(&rungs, &wrong_hex).expect("computes").not_run),
        ["a@lambda"]
    );
    let no_sha_text = r#"{"schema":"apr-model-ladder-receipt/v2","host":"lambda","version":"0.69.1","sha":"x","cc":"8.9","gpu":"g","rungs":[{"id":"a","green":true,"capability_match":{"passed":true,"skipped":false},"backends":{"cpu":{"ran":true,"fallback":false},"cuda":{"ran":true,"fallback":false}}}]}"#;
    let no_sha = [receipts::parse("r.json", no_sha_text).expect("parses")];
    assert_eq!(
        ids(&compute(&rungs, &no_sha).expect("computes").not_run),
        ["a@lambda"]
    );
}

#[test]
fn an_optional_rung_is_outside_the_domain() {
    let rungs = [rung("a", SHA_A, &[], true), rung("opt", SHA_B, &[], false)];
    let recs = [receipt("0.69.1", "lambda", &[row("a", SHA_A, true, "")])];
    let cc = compute(&rungs, &recs).expect("computes");
    assert_eq!(ids(&cc.domain), ["a@lambda"]);
    assert!(cc.not_run.is_empty());
}

#[test]
fn listed_hosts_scope_the_domain_exactly_as_ladder_green_does() {
    let rungs = [rung("a", SHA_A, &["lambda", "yoga"], true)];
    let recs = [
        receipt("0.69.1", "lambda", &[row("a", SHA_A, true, "")]),
        receipt("0.69.1", "gx10", &[row("a", SHA_A, true, "")]),
    ];
    let cc = compute(&rungs, &recs).expect("computes");
    // one definition: receipts::expected_hosts — gx10 is not listed, yoga is and has no receipt
    assert_eq!(ids(&cc.domain), ["a@lambda", "a@yoga"]);
    assert_eq!(ids(&cc.not_run), ["a@yoga"]);
}

#[test]
fn an_empty_domain_is_empty_not_a_pass() {
    // no receipts, no listed hosts → D = ∅; the GATE turns this into a decline (exit 2), never Pass
    let cc = compute(&[rung("a", SHA_A, &[], true)], &[]).expect("computes");
    assert!(cc.domain.is_empty());
    assert!(cc.v_star.is_none());
}

#[test]
fn apply_writes_one_edge_per_not_run_cell_and_per_found_cell() {
    let rungs = [rung("a", SHA_A, &[], true), rung("b", SHA_B, &[], true)];
    let recs = [
        receipt(
            "0.69.1",
            "lambda",
            &[row("a", SHA_A, true, ""), row("b", SHA_B, false, "")],
        ),
        receipt(
            "0.69.1",
            "gx10",
            &[row("a", SHA_A, true, r#","verdict":"NO-VERDICT""#)],
        ),
    ];
    let mut g = Graph::new();
    let cc = apply(&mut g, &rungs, &recs).expect("applies");
    assert_eq!(ids(&cc.not_run), ["a@gx10", "b@gx10"]);
    assert_eq!(not_run_edges(&g), cc.not_run.len());
    let a = iri("model", SHA_A);
    let edges: Vec<String> = g
        .objects(&a, &model("capabilityCell"))
        .iter()
        .filter_map(|t| t.as_literal().map(|(v, _)| v.to_string()))
        .collect();
    assert_eq!(edges, ["a@gx10=NotRun", "a@lambda=Pass"]);
    let b = iri("model", SHA_B);
    assert_eq!(g.objects(&b, &model("notRunCell")).len(), 1);
}

#[test]
fn a_refused_label_lands_on_its_own_rung_only() {
    let rungs = [rung("a", SHA_A, &[], true), rung("b", SHA_B, &[], true)];
    let recs = [receipt(
        "0.69.1",
        "lambda",
        &[
            row("a", SHA_A, true, r#","label":"maybe""#),
            row("b", SHA_B, true, ""),
        ],
    )];
    let mut g = Graph::new();
    apply(&mut g, &rungs, &recs).expect("applies");
    assert_eq!(
        g.objects(&iri("model", SHA_A), &model("refusedLabel"))
            .len(),
        1
    );
    assert!(g
        .objects(&iri("model", SHA_B), &model("refusedLabel"))
        .is_empty());
}
