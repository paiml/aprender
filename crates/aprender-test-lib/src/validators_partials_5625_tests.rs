//! Coverage for the post-processing inside `collect_partials_cdp` (#5625).
//!
//! The CDP call itself needs a live chromium; the row filtering and the
//! "last row is final" marking are the branches, and they live in
//! `partials_from_raw`.

use crate::validators::partials_from_raw;
use serde_json::json;

#[test]
fn empty_input_yields_no_partials() {
    assert!(partials_from_raw(vec![]).is_empty());
}

#[test]
fn only_the_last_row_is_final() {
    let out = partials_from_raw(vec![
        vec![json!(10.0), json!("he")],
        vec![json!(20.0), json!("hel")],
        vec![json!(30.0), json!("hello")],
    ]);
    let finals: Vec<bool> = out.iter().map(|p| p.is_final).collect();
    assert_eq!(finals, vec![false, false, true]);
    assert_eq!(out[2].text, "hello");
    assert!((out[1].timestamp_ms - 20.0).abs() < f64::EPSILON);
}

#[test]
fn single_row_is_final() {
    let out = partials_from_raw(vec![vec![json!(1.5), json!("a")]]);
    assert_eq!(out.len(), 1);
    assert!(out[0].is_final);
}

#[test]
fn malformed_rows_are_dropped_and_finality_goes_to_last_survivor() {
    let out = partials_from_raw(vec![
        vec![json!(1.0), json!("ok")],
        vec![],                         // no timestamp
        vec![json!("x"), json!("bad")], // non-numeric timestamp
        vec![json!(2.0)],               // no text
        vec![json!(3.0), json!(7)],     // non-string text
        vec![json!(4.0), json!("last")],
        vec![json!(5.0), json!(null)], // trailing malformed row
    ]);
    let texts: Vec<&str> = out.iter().map(|p| p.text.as_str()).collect();
    assert_eq!(texts, vec!["ok", "last"]);
    assert!(!out[0].is_final);
    assert!(out[1].is_final);
}

#[test]
fn all_malformed_yields_empty_not_final() {
    assert!(partials_from_raw(vec![vec![json!(null), json!(null)], vec![]]).is_empty());
}
