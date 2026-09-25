use super::*;

fn receipt(advisory: Option<&str>, width: u64) -> String {
    let lanes = r#"[{"family":"claude","model":"claude-sonnet-5","verdict":"PASS","findings":[],"input_sha256":"i1","output_sha256":"o1"},
        {"family":"gemini","model":"gemini-3.1-pro-high","verdict":"PASS","findings":["ledger.rs:205 drops the weights sha",{"f":1}],"input_sha256":"i2","output_sha256":"o2"},
        {"family":"claude","model":"claude-haiku-4-5","verdict":"PASS","findings":[],"input_sha256":"i3","output_sha256":"o3"}]"#;
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
    assert_eq!(
        r.lanes[1].findings,
        Some(vec![
            "ledger.rs:205 drops the weights sha".to_string(),
            r#"{"f":1}"#.to_string()
        ]),
        "findings are kept as text, verbatim"
    );
    assert_eq!(r.lanes[1].output_sha256.as_deref(), Some("o2"));
    assert_eq!(c.trace_gaps, 0);
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

#[test]
fn falsify_rxl_006_a_counted_lane_without_trace_shas_does_not_hold() {
    for gap in [
        r#","output_sha256":"o2""#,
        r#","input_sha256":"i2""#,
        r#""findings":["ledger.rs:205 drops the weights sha",{"f":1}],"#,
    ] {
        let (rows, c) = one(receipt(Some(ANSWERED), 3).replace(gap, ""));
        assert_eq!(rows.len(), 1, "the row is kept");
        assert_eq!(c.trace_gaps, 1, "{gap}");
        assert!(!c.holds(), "a lane without its trace is RED: {gap}");
    }
}

#[test]
fn falsify_rxl_007_an_outcome_join_appends_and_never_rewrites() {
    let (rows, _) = one(receipt(Some(ANSWERED), 3));
    let before = rows.clone();
    let j = |head: &str, outcome: &str, at: &str| OutcomeJoin {
        repo: "paiml/aprender".into(),
        head: head.into(),
        outcome: outcome.into(),
        at: at.into(),
    };
    assert_eq!(outcome(&rows[0], &[]), "pending");
    let joins = [
        j("abc", "reverted", "2026-09-30T00:00:00Z"),
        j("abc", "merged", "2026-09-25T00:00:00Z"),
        j("other", "escape", "2026-10-01T00:00:00Z"),
    ];
    assert_eq!(outcome(&rows[0], &joins), "reverted", "latest join wins");
    assert_eq!(rows, before, "the row itself is never rewritten");
}

// v1 rows exactly as the two v1 writers serialized them.
const V1_LEGACY: &str = r#"{"schema":"review-ledger-v1","repo":"paiml/aprender","ticket":"PMAT-1","head":"abc","pr":null,"diff_sha256":"d","agreed":true,"width":3,"lanes":[{"family":"claude","model":"claude-sonnet-5","verdict":"PASS","findings":2}],"shadow":{"state":"Unknown","reason":{"LaneUnavailable":{"why":"gx10 down"}}},"apr_tag":null,"weights_sha256":null,"outcome":"pending"}"#;
const V1_TYPED: &str = r#"{"schema":"review-ledger-v1","repo":"paiml/aprender","ticket":"PMAT-1","head":"abc","pr":4354,"diff_sha256":"d","agreed":true,"width":3,"lanes":[{"family":"gemini","model":"gemini-3.1-pro-high","verdict":"FAIL","findings":2}],"shadow":{"Verdict":{"verdict":"PASS","cell":"gx10-cuda","backend":"cuda"}},"apr_tag":"0.69.3","weights_sha256":"w4b","outcome":"merged"}"#;

#[test]
fn falsify_rxl_008_v1_rows_parse_and_migrate_without_invented_text() {
    for line in [V1_LEGACY, V1_TYPED] {
        assert!(
            matches!(read_row(line), Ok(AnyRow::V1(_))),
            "{:?}",
            read_row(line)
        );
    }
    let Ok(AnyRow::V1(typed)) = read_row(V1_TYPED) else {
        panic!("v1 typed row")
    };
    let m = typed.migrate().expect("a typed v1 row migrates");
    assert_eq!(m.schema, SCHEME);
    assert_eq!(m.lanes[0].findings_count, 2, "the count carries over");
    assert_eq!(m.lanes[0].findings, None, "text never captured stays None");
    assert_eq!((m.pr, m.outcome.as_str()), (Some(4354), "merged"));
    // A migrated row round-trips as v2 and still reads as a trace gap.
    let line = serde_json::to_string(&m).expect("ser");
    assert_eq!(read_row(&line), Ok(AnyRow::V2(m.clone())));
    assert!(trace_gap(&m.lanes[0]));
    // A legacy shadow has no cell/backend: refused, not guessed.
    let Ok(AnyRow::V1(legacy)) = read_row(V1_LEGACY) else {
        panic!("v1 legacy row")
    };
    assert!(legacy.migrate().is_err());
    // A v1 body under the v2 name, and an unknown schema, are errors.
    assert!(read_row(&V1_TYPED.replace(SCHEME_V1, SCHEME)).is_err());
    assert!(read_row(&V1_TYPED.replace(SCHEME_V1, "review-ledger-v9")).is_err());
}
