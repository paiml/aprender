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

const ANSWERED: &str = r#"{"state":"answered","verdict":"PASS","served_by":"gx10-cuda","apr":"0.69.3","counted":"PASS","counts":false}"#;
const DOWN: &str = r#"{"state":"unavailable","verdict":null,"served_by":null,"apr":null,"counted":"PASS","counts":false,"why":"gx10 connect refused"}"#;

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
        Shadow::Answered {
            verdict: "PASS".into(),
            served_by: "gx10-cuda".into()
        }
    );
    assert_eq!(r.outcome, "pending");
    // The rail records no weights sha yet: counted, not hidden.
    assert_eq!(c.identity_gaps, 1);
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
    assert_eq!(
        rows[0].shadow,
        Shadow::Unknown {
            reason: Unknown::LaneUnavailable {
                why: Some("gx10 connect refused".into())
            }
        }
    );
    // An "answered" state without a verdict is not a verdict.
    let half = r#"{"state":"answered","verdict":null,"served_by":"gx10-cuda","counted":"PASS","counts":false}"#;
    assert!(matches!(
        one(receipt(Some(half), 3)).0[0].shadow,
        Shadow::Unknown { .. }
    ));
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
