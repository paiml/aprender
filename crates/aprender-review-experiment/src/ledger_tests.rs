use super::*;

fn receipt(advisory: Option<&str>, width: u64) -> String {
    let lanes = r#"[{"family":"claude","model":"claude-sonnet-5","verdict":"PASS","findings":[]},
        {"family":"gemini","model":"gemini-3.1-pro-high","verdict":"PASS","findings":[{"f":1}]},
        {"family":"claude","model":"claude-haiku-4-5","verdict":"PASS","findings":[]}]"#;
    let adv = advisory.map_or(String::new(), |a| format!(r#","advisory_lane":{a}"#));
    format!(
        r#"{{"ticket":"PMAT-1","head":"abc","diff_sha256":"d","agreed":true,"width":{width},"lanes":{lanes}{adv}}}"#
    )
}

const ANSWERED: &str = r#"{"state":"answered","row":{"Verdict":{"verdict":"PASS","cell":"gx10-cuda","backend":"cuda"}},"apr":"0.69.3","weights_sha256":"w4b","counted":"PASS","counts":false}"#;
const DOWN: &str = r#"{"state":"unavailable","row":{"NotRun":"NoExecutor"},"apr":"0.69.3","weights_sha256":"w4b","counted":"PASS","counts":false}"#;

fn one(text: String) -> (Vec<Row>, Coverage) {
    build(&[("quorum-PMAT-1.json".into(), text)], "paiml/aprender")
}

#[test]
fn an_answered_shadow_becomes_a_ledger_row() {
    let (rows, c) = one(receipt(Some(ANSWERED), 3));
    assert!(c.holds(), "{c:?}");
    let r = &rows[0];
    assert_eq!(r.schema, SCHEME);
    assert_eq!(r.width, 3);
    assert_eq!(r.lanes[1].findings, 1);
    assert_eq!(
        r.shadow,
        Shadow::Verdict {
            verdict: Verdict::Pass,
            cell: "gx10-cuda".into(),
            backend: "cuda".into()
        }
    );
    assert_eq!(r.weights_sha256.as_deref(), Some("w4b"));
    assert_eq!(r.outcome, "pending");
    assert_eq!(c.identity_gaps, 0);
}

#[test]
fn falsify_rxl_001_a_quorum_without_a_shadow_row_is_missing() {
    let recs = vec![
        ("a.json".to_string(), receipt(Some(ANSWERED), 3)),
        ("b.json".to_string(), receipt(None, 3)),
    ];
    let (rows, c) = build(&recs, "paiml/aprender");
    assert_eq!(rows.len(), 1);
    assert_eq!((c.quorums, c.carried), (2, 1));
    assert_eq!(c.missing, ["b.json"]);
    assert!(!c.holds(), "one missing shadow row fails REX-07");
    assert!(!build(&[], "r").1.holds(), "no quorum is not 100 %");
    let off = r#"{"state":"off","counts":false,"why":"advisory_lane disabled"}"#;
    assert!(
        !one(receipt(Some(off), 3)).1.holds(),
        "a lane that was off is not a shadow row"
    );
    let (_, bad) = build(&[("x".into(), "{".into())], "r");
    assert!(!bad.holds(), "an unparsable receipt is not dropped");
}

#[test]
fn falsify_rxl_002_gx10_down_is_unknown_and_keeps_the_width() {
    let (rows, c) = one(receipt(Some(DOWN), 3));
    assert!(c.holds(), "{c:?}");
    assert_eq!(rows[0].width, 3);
    assert_eq!(rows[0].shadow, Shadow::NotRun(NotRun::NoExecutor));
    // A verdict without its cell is not a verdict.
    let half = ANSWERED.replace(r#","cell":"gx10-cuda""#, "");
    assert!(!one(receipt(Some(&half), 3)).1.holds(), "no cell");
}

#[test]
fn falsify_rxl_003_a_counted_or_width_changing_shadow_is_a_violation() {
    let counted = ANSWERED.replace(r#""counts":false"#, r#""counts":true"#);
    assert!(!one(receipt(Some(&counted), 3)).1.holds(), "counted");
    let silent = ANSWERED.replace(r#","counts":false"#, "");
    assert!(
        !one(receipt(Some(&silent), 3)).1.holds(),
        "uncounted unrecorded"
    );
    assert!(!one(receipt(Some(ANSWERED), 4)).1.holds(), "width 4");
    assert!(!one(receipt(Some(DOWN), 2)).1.holds(), "width 2");
}

#[test]
fn every_wire_form_round_trips() {
    for row in [
        r#"{"Verdict":{"verdict":"FAIL","cell":"intel-wgpu","backend":"wgpu"}}"#,
        r#"{"NotRun":"NoExecutor"}"#,
        r#"{"NotRun":"Busy"}"#,
        r#"{"NotRun":"Timeout"}"#,
        r#"{"NotRun":"ContextOverflow"}"#,
        r#"{"NotRun":"TrainActive"}"#,
        r#"{"Refused":{"cell":"gx10-cuda","removed_by":"ladder"}}"#,
        r#"{"Refused":{"cell":"gx10-cuda","removed_by":"gpu-proof"}}"#,
        r#"{"Refused":{"cell":"gx10-cuda","removed_by":"parse"}}"#,
    ] {
        let s: Shadow = serde_json::from_str(row).expect(row);
        assert_eq!(serde_json::to_string(&s).expect("ser"), row);
    }
}

#[test]
fn falsify_rxl_004_a_receipt_without_a_weights_sha_does_not_hold() {
    let bare = ANSWERED.replace(r#","weights_sha256":"w4b""#, "");
    let (rows, c) = one(receipt(Some(&bare), 3));
    assert_eq!(rows.len(), 1, "the row is kept");
    assert_eq!(c.identity_gaps, 1);
    assert!(!c.holds(), "an identity gap fails holds()");
    let untagged = ANSWERED.replace(r#""apr":"0.69.3""#, r#""apr":null"#);
    assert!(!one(receipt(Some(&untagged), 3)).1.holds(), "no apr tag");
}

#[test]
fn falsify_rxl_005_a_free_text_reason_fails_to_parse() {
    for row in [
        r#"{"NotRun":"gx10 connect refused"}"#,
        r#"{"LaneUnavailable":{"why":"gx10 connect refused"}}"#,
        r#"{"Refused":{"cell":"gx10-cuda","removed_by":"aprender#9999: refuses qwen35"}}"#,
        r#"{"Verdict":{"verdict":"LGTM","cell":"gx10-cuda","backend":"cuda"}}"#,
        r#"{"Verdict":{"verdict":"PASS","cell":"gx10-cuda","backend":"cuda","why":"x"}}"#,
    ] {
        assert!(serde_json::from_str::<Shadow>(row).is_err(), "{row}");
        let adv = ANSWERED.replace(
            r#"{"Verdict":{"verdict":"PASS","cell":"gx10-cuda","backend":"cuda"}}"#,
            row,
        );
        let (rows, c) = one(receipt(Some(&adv), 3));
        assert!(rows.is_empty() && !c.holds(), "{row} is a violation");
    }
    let untyped = r#"{"state":"answered","apr":"0.69.3","weights_sha256":"w","counts":false}"#;
    assert!(!one(receipt(Some(untyped), 3)).1.holds(), "no typed row");
}
