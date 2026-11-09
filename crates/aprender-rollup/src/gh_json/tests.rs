use super::*;

const OK: &str = r#"[{"number":7,"additions":20,"deletions":3,
 "headRefOid":"abc123","labels":[{"name":"rollup-ok"},{"name":"x"}],
 "files":[{"path":"a.rs"},{"path":"b.rs"}],
 "reviewDecision":"APPROVED",
 "statusCheckRollup":[{"conclusion":"SUCCESS"},{"conclusion":"SKIPPED"},{"state":"SUCCESS"}]}]"#;

#[test]
fn parses_a_green_approved_pr() {
    let c = &parse_prs(OK).expect("parse")[0];
    assert_eq!((c.number, c.additions, c.deletions), (7, 20, 3));
    assert!(c.green && c.approved);
    assert_eq!(c.labels, ["rollup-ok", "x"]);
    assert_eq!(c.files, ["a.rs", "b.rs"]);
    assert_eq!(c.head, "abc123");
}

#[test]
fn one_failed_or_pending_check_is_not_green() {
    for bad in [
        r#"{"conclusion":"FAILURE"}"#,
        r#"{"status":"IN_PROGRESS"}"#,
        r#"{"state":"PENDING"}"#,
    ] {
        let j = OK.replace(r#"{"conclusion":"SKIPPED"}"#, bad);
        assert!(!parse_prs(&j).expect("parse")[0].green, "{bad}");
    }
}

#[test]
fn no_checks_is_not_green() {
    let j = OK.replace(r#""statusCheckRollup":[{"conclusion":"SUCCESS"},{"conclusion":"SKIPPED"},{"state":"SUCCESS"}]"#, r#""statusCheckRollup":[]"#);
    assert!(!parse_prs(&j).expect("parse")[0].green);
}

#[test]
fn review_decision_other_than_approved_is_not_approved() {
    for d in ["CHANGES_REQUESTED", "REVIEW_REQUIRED", ""] {
        let j = OK.replace("APPROVED", d);
        assert!(!parse_prs(&j).expect("parse")[0].approved, "{d}");
    }
}

#[test]
fn missing_line_counts_fail_closed_as_too_large() {
    let j = r#"[{"number":1,"labels":[],"files":[],"statusCheckRollup":[]}]"#;
    let c = &parse_prs(j).expect("parse")[0];
    assert_eq!(c.additions, u32::MAX);
}

#[test]
fn malformed_input_is_an_error_not_an_empty_list() {
    assert!(parse_prs("not json").is_err());
    assert!(parse_prs("{}").is_err());
    assert!(parse_prs(r#"[{"additions":1}]"#).is_err());
    assert_eq!(parse_prs("[]"), Ok(vec![]));
}

#[test]
fn only_skipped_checks_are_not_green() {
    let j = r#"[{"number":1,"additions":1,"deletions":1,"labels":[],"files":[],"headRefOid":"h",
     "reviewDecision":"APPROVED","statusCheckRollup":[{"conclusion":"SKIPPED"},{"conclusion":"NEUTRAL"}]}]"#;
    assert!(!parse_prs(j).expect("parse")[0].green);
}
