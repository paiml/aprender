use super::*;

const PREREG: &str = "ef51087dc79bab0ad160e8a14f5b13e2ea43986b30c05dafc63c84c2dc21cdc0";

fn report(h4: &str, h5: &str, ruling: &str) -> String {
    format!(
        r#"{{"schema":"{SCHEME}","prereg_sha":"{PREREG}","results":{{"hypotheses":[
            {{"id":"H1","ci":[0.0,0.1],"verdict":"holds"}},{h4},{h5}]}}{ruling}}}"#
    )
}

const H4_OK: &str = r#"{"id":"H4","ci":[0.01,0.2],"verdict":"holds"}"#;
const H5_OK: &str = r#"{"id":"H5","ci":[0.0,0.1],"verdict":"holds"}"#;
const H4_FAIL: &str = r#"{"id":"H4","ci":[-0.05,0.1],"verdict":"fails"}"#;
const VOTE: &str = r#","hardware_ruling":{"mode":"vote"}"#;

#[test]
fn both_gates_passed_is_a_full_vote() {
    let d = decide(&report(H4_OK, H5_OK, VOTE), PREREG);
    assert_eq!(d.mode, Mode::Vote, "{:?}", d.reasons);
    assert!(d.reasons.is_empty());
}

#[test]
fn falsify_rxg_001_a_failing_h4_keeps_the_lane_in_shadow() {
    // The §7 REX-09 planted report: H4 failing, everything else passing.
    let d = decide(&report(H4_FAIL, H5_OK, VOTE), PREREG);
    assert_eq!(d.mode, Mode::Shadow, "{:?}", d.reasons);
    // A "holds" verdict whose CI crosses 0 is not a pass either.
    let crossing = r#"{"id":"H4","ci":[-0.01,0.2],"verdict":"holds"}"#;
    assert_eq!(
        decide(&report(crossing, H5_OK, VOTE), PREREG).mode,
        Mode::Shadow
    );
    let nan = r#"{"id":"H4","ci":[NaN,0.2],"verdict":"holds"}"#;
    assert_eq!(decide(&report(nan, H5_OK, VOTE), PREREG).mode, Mode::Shadow);
    let absent = r#"{"id":"H3","ci":[0.1,0.2],"verdict":"holds"}"#;
    assert_eq!(
        decide(&report(absent, H5_OK, VOTE), PREREG).mode,
        Mode::Shadow
    );
}

#[test]
fn falsify_rxg_002_h4_alone_is_tripwire_never_vote() {
    let h5_fail = r#"{"id":"H5","ci":[-0.1,0.0],"verdict":"fails"}"#;
    let d = decide(&report(H4_OK, h5_fail, VOTE), PREREG);
    assert_eq!(d.mode, Mode::Tripwire, "{:?}", d.reasons);
    assert!(d.reasons.iter().any(|r| r.starts_with("H5")));
    // The queue budget caps the rung, whatever H4/H5 say.
    let capped = decide(
        &report(H4_OK, H5_OK, r#","hardware_ruling":{"mode":"shadow"}"#),
        PREREG,
    );
    assert_eq!(capped.mode, Mode::Shadow);
    assert_eq!(decide(&report(H4_OK, H5_OK, ""), PREREG).mode, Mode::Shadow);
}

#[test]
fn falsify_rxg_003_only_a_locked_confirmatory_report_is_read() {
    let ok = report(H4_OK, H5_OK, VOTE);
    let cases = [
        ("foreign schema", ok.replace(SCHEME, "rex-001-report-v0")),
        ("stale prereg", ok.replace(PREREG, &"0".repeat(64))),
        (
            "exploratory",
            ok.replacen('{', r#"{"exploratory":true,"#, 1),
        ),
        ("not json", "{".to_string()),
    ];
    for (name, text) in cases {
        assert_eq!(decide(&text, PREREG).mode, Mode::Shadow, "{name}");
    }
}
