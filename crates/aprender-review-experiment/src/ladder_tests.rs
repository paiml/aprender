// `json!` expands to an `unwrap` of an infallible `to_value`.
#![allow(clippy::disallowed_methods)]

use super::*;
use serde_json::{json, Value};

const PREREG: &str = "ef51087dc79bab0ad160e8a14f5b13e2ea43986b30c05dafc63c84c2dc21cdc0";

fn h(id: &str, p: f64) -> Value {
    json!({"id": id, "ci": [0.01, 0.2], "p_holm": p, "verdict": "holds"})
}

/// A `prm-001-report-v1` report on which every gate up to vote passes.
fn full() -> Value {
    json!({
        "spec": "PRM-001",
        "version": 3,
        "prereg_sha": PREREG,
        "lane": {"rung": "tie-breaker", "availability_7d": 0.97},
        "speed": {"within_budget": true},
        "results": {"hypotheses": [
            {"id": "H1", "verdict": "holds"},
            {"id": "H2", "verdict": "holds"},
            h("H4", 0.01),
            h("H5", 0.04),
            {"id": "H7", "verdict": "holds"},
        ]},
        "hardware_ruling": {"primary": "gx10-cpu", "rung_eligible": "vote"},
        "promotion": {
            "tie_breaks_decided": 20,
            "tie_break_outcome_agreement": 0.9,
            "voter_outcome_agreement": 0.9
        }
    })
}

fn set(mut v: Value, ptr: &str, x: Value) -> Value {
    *v.pointer_mut(ptr).unwrap_or_else(|| panic!("{ptr}")) = x;
    v
}

fn drop_h(mut v: Value, id: &str) -> Value {
    let hs = v["results"]["hypotheses"]
        .as_array_mut()
        .expect("hypotheses");
    hs.retain(|h| h["id"] != id);
    v
}

fn replace_h(v: Value, x: Value) -> Value {
    let id = x["id"].as_str().expect("id").to_string();
    let mut v = drop_h(v, &id);
    v["results"]["hypotheses"]
        .as_array_mut()
        .expect("hypotheses")
        .push(x);
    v
}

/// The rung the evidence grants, before the one-rung step.
fn rung(v: &Value) -> Mode {
    evidence(&v.to_string(), PREREG).mode
}

#[test]
fn every_gate_passed_is_a_full_vote() {
    let d = evidence(&full().to_string(), PREREG);
    assert_eq!(d.mode, Mode::Vote, "{:?}", d.reasons);
    assert!(d.reasons.is_empty());
    assert_eq!(
        serde_json::to_value(Mode::TieBreaker).expect("mode"),
        json!("tie-breaker")
    );
}

/// FALSIFY-RXG-001: a failing H4 (the planted report) keeps the lane in shadow.
#[test]
fn falsify_rxg_001_a_failing_h4_keeps_the_lane_in_shadow() {
    let fail = json!({"id": "H4", "ci": [-0.05, 0.1], "p_holm": 0.3, "verdict": "fails"});
    assert_eq!(rung(&replace_h(full(), fail)), Mode::Shadow);
    for bad in [
        json!({"id": "H4", "p_holm": 0.051, "verdict": "holds"}),
        json!({"id": "H4", "p_holm": -0.1, "verdict": "holds"}),
        json!({"id": "H4", "verdict": "holds"}),
    ] {
        assert_eq!(rung(&replace_h(full(), bad.clone())), Mode::Shadow, "{bad}");
    }
    assert_eq!(rung(&drop_h(full(), "H4")), Mode::Shadow);
    let nan = full()
        .to_string()
        .replace("\"p_holm\":0.01", "\"p_holm\":NaN");
    assert_eq!(evidence(&nan, PREREG).mode, Mode::Shadow);
}

/// FALSIFY-RXG-002: H4 alone is tripwire, never above; the speed gate and the
/// hardware ruling cap the rung.
#[test]
fn falsify_rxg_002_h4_alone_is_tripwire_never_vote() {
    let h5_fail = json!({"id": "H5", "p_holm": 0.6, "verdict": "fails"});
    let d = evidence(&replace_h(full(), h5_fail).to_string(), PREREG);
    assert_eq!(d.mode, Mode::Tripwire, "{:?}", d.reasons);
    assert!(
        d.reasons.iter().any(|r| r.starts_with("H5")),
        "{:?}",
        d.reasons
    );
    // the planted report: H4 only
    let h4_only = drop_h(drop_h(drop_h(drop_h(full(), "H5"), "H7"), "H1"), "H2");
    assert_eq!(rung(&h4_only), Mode::Tripwire);
    // speed: replay p95 over the queue budget, or unmeasured, is shadow
    assert_eq!(
        rung(&set(full(), "/speed/within_budget", json!(false))),
        Mode::Shadow
    );
    let mut no_speed = full();
    no_speed.as_object_mut().expect("obj").remove("speed");
    assert_eq!(rung(&no_speed), Mode::Shadow);
    // the hardware ruling caps; none is shadow
    for (cap, want) in [
        ("shadow", Mode::Shadow),
        ("tripwire", Mode::Tripwire),
        ("tie-breaker", Mode::TieBreaker),
    ] {
        let v = set(full(), "/hardware_ruling/rung_eligible", json!(cap));
        assert_eq!(rung(&v), want, "{cap}");
    }
    // a ruling that names no eligible rung grants none
    let mut no_rung = full();
    no_rung["hardware_ruling"]
        .as_object_mut()
        .expect("ruling")
        .remove("rung_eligible");
    assert_eq!(rung(&no_rung), Mode::Shadow);
    let mut no_ruling = full();
    no_ruling
        .as_object_mut()
        .expect("obj")
        .remove("hardware_ruling");
    assert_eq!(rung(&no_ruling), Mode::Shadow);
}

/// FALSIFY-RXG-003: only a locked, confirmatory prm-001-report-v1 is read.
#[test]
fn falsify_rxg_003_only_a_locked_confirmatory_report_is_read() {
    let old = full()
        .to_string()
        .replace("\"spec\":\"PRM-001\"", "\"schema\":\"rex-001-report-v2\"");
    let cases = [
        ("rex-001-report-v2", old),
        ("v2 spec", set(full(), "/version", json!(2)).to_string()),
        (
            "stale prereg",
            set(full(), "/prereg_sha", json!("0".repeat(64))).to_string(),
        ),
        ("exploratory", {
            let mut v = full();
            v["exploratory"] = json!(true);
            v.to_string()
        }),
        ("not json", "{".to_string()),
    ];
    for (name, text) in cases {
        assert_eq!(evidence(&text, PREREG).mode, Mode::Shadow, "{name}");
    }
}

/// FALSIFY-RXG-004: the CI is report-only; the Holm p decides H4/H5.
#[test]
fn falsify_rxg_004_the_ci_is_report_only() {
    let crossing = json!({"id": "H4", "ci": [-0.01, 0.2], "p_holm": 0.02, "verdict": "holds"});
    assert_eq!(rung(&replace_h(full(), crossing)), Mode::Vote);
    let clean = json!({"id": "H4", "ci": [0.05, 0.2], "p_holm": 0.2, "verdict": "holds"});
    assert_eq!(rung(&replace_h(full(), clean)), Mode::Shadow);
}

/// FALSIFY-RXG-005: tie-breaker needs H4 ∧ H5 ∧ H7 ∧ H1 ∧ H2 and 7-day
/// availability ≥ 95%; without any one of them the lane stays tripwire.
#[test]
fn falsify_rxg_005_tie_breaker_needs_h4_h5_h7_h1_h2_and_availability() {
    let tb = set(full(), "/promotion/tie_breaks_decided", json!(0));
    assert_eq!(rung(&tb), Mode::TieBreaker);
    for id in ["H5", "H7", "H1", "H2"] {
        let d = evidence(&drop_h(tb.clone(), id).to_string(), PREREG);
        assert_eq!(d.mode, Mode::Tripwire, "without {id}");
        assert!(
            d.reasons.iter().any(|r| r.starts_with(id)),
            "{id}: {:?}",
            d.reasons
        );
        let fails = json!({"id": id, "p_holm": 0.01, "verdict": "fails"});
        assert_eq!(
            rung(&replace_h(tb.clone(), fails)),
            Mode::Tripwire,
            "{id} fails"
        );
    }
    // H5 is in the Holm family: a holds verdict over α does not hold
    let h5_p = json!({"id": "H5", "p_holm": 0.6, "verdict": "holds"});
    assert_eq!(rung(&replace_h(tb.clone(), h5_p)), Mode::Tripwire);
    // H7 is outside the Holm family: its verdict decides, a p is not needed
    assert!(full()["results"]["hypotheses"][4].get("p_holm").is_none());
    // availability: inclusive at 0.95, below or absent is tripwire
    assert_eq!(
        rung(&set(tb.clone(), "/lane/availability_7d", json!(0.95))),
        Mode::TieBreaker
    );
    assert_eq!(
        rung(&set(tb.clone(), "/lane/availability_7d", json!(0.9499))),
        Mode::Tripwire
    );
    let mut no_avail = tb.clone();
    no_avail["lane"]
        .as_object_mut()
        .expect("lane")
        .remove("availability_7d");
    assert_eq!(rung(&no_avail), Mode::Tripwire);
}

/// FALSIFY-RXG-006: a lambda primary cell is refused at tie-breaker and above
/// (R-9, S-4), whatever the evidence.
#[test]
fn falsify_rxg_006_lambda_primary_is_refused_above_tripwire() {
    let v = set(full(), "/hardware_ruling/primary", json!("lambda-cuda"));
    let d = evidence(&v.to_string(), PREREG);
    assert_eq!(d.mode, Mode::Tripwire, "{:?}", d.reasons);
    assert!(
        d.reasons.iter().any(|r| r.contains("S-4")),
        "{:?}",
        d.reasons
    );
    // the reason names the cell as written, never a Rust Debug rendering
    assert!(
        d.reasons
            .iter()
            .any(|r| r.contains("primary cell lambda-cuda is") && !r.contains("Some(")),
        "{:?}",
        d.reasons
    );
    // a missing primary is not a non-lambda cell
    let mut none = full();
    none["hardware_ruling"]
        .as_object_mut()
        .expect("ruling")
        .remove("primary");
    assert_eq!(rung(&none), Mode::Tripwire);
    let n = evidence(&none.to_string(), PREREG);
    assert!(
        n.reasons.iter().any(|r| r.contains("primary cell (unnamed) is")),
        "{:?}",
        n.reasons
    );
}

/// FALSIFY-RXG-007: vote needs ≥ 20 decided tie-breaks whose 14-day outcome
/// agreement is at least the voters' on the same rounds.
#[test]
fn falsify_rxg_007_vote_needs_twenty_agreeing_tie_breaks() {
    assert_eq!(
        rung(&set(full(), "/promotion/tie_breaks_decided", json!(19))),
        Mode::TieBreaker
    );
    assert_eq!(
        rung(&set(
            full(),
            "/promotion/tie_break_outcome_agreement",
            json!(0.89)
        )),
        Mode::TieBreaker
    );
    let mut none = full();
    none.as_object_mut().expect("obj").remove("promotion");
    assert_eq!(rung(&none), Mode::TieBreaker);
    let mut no_voters = full();
    no_voters["promotion"]
        .as_object_mut()
        .expect("promotion")
        .remove("voter_outcome_agreement");
    assert_eq!(rung(&no_voters), Mode::TieBreaker);
}

/// FALSIFY-RXG-008: the ladder never skips a rung, and a demotion drops
/// exactly one rung; S-4 is the exception and caps at once.
#[test]
fn falsify_rxg_008_never_skips_a_rung_and_demotes_one() {
    let all = full().to_string();
    let step = |cur| decide(&all, PREREG, cur).mode;
    assert_eq!(step(Mode::Shadow), Mode::Tripwire);
    assert_eq!(step(Mode::Tripwire), Mode::TieBreaker);
    assert_eq!(step(Mode::TieBreaker), Mode::Vote);
    let d = decide("{", PREREG, Mode::Vote);
    assert_eq!(d.mode, Mode::TieBreaker, "{:?}", d.reasons);
    assert_eq!(decide("{", PREREG, Mode::Tripwire).mode, Mode::Shadow);
    assert_eq!(decide("{", PREREG, Mode::Shadow).mode, Mode::Shadow);
    let lambda = set(full(), "/hardware_ruling/primary", json!("lambda-cuda")).to_string();
    // S-4 caps a vote lane at tripwire at once, but never promotes past one rung
    assert_eq!(decide(&lambda, PREREG, Mode::Vote).mode, Mode::Tripwire);
    assert_eq!(decide(&lambda, PREREG, Mode::Shadow).mode, Mode::Tripwire);
    assert_eq!(
        decide(&lambda, PREREG, Mode::TieBreaker).mode,
        Mode::Tripwire
    );
    // S-4 never promotes: a lambda lane with no evidence stays in shadow
    let lambda_h4_fail = drop_h(
        set(full(), "/hardware_ruling/primary", json!("lambda-cuda")),
        "H4",
    );
    assert_eq!(
        decide(&lambda_h4_fail.to_string(), PREREG, Mode::Shadow).mode,
        Mode::Shadow
    );
    // a tripwire-only report demotes a vote lane one rung, not two
    let h4_only = drop_h(drop_h(full(), "H5"), "H7").to_string();
    assert_eq!(decide(&h4_only, PREREG, Mode::Vote).mode, Mode::TieBreaker);
}
