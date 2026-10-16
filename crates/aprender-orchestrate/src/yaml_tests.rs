//! Pins the reader/writer behaviour the serde-saphyr port chose (#5209), so a later
//! option change shows up as a red test and not as a silent change in playbooks.

use super::from_str;
use serde_json::{json, Value};

#[test]
fn yes_and_no_stay_strings() {
    let v: Value = from_str("a: yes\nb: no\nc: true\n").expect("parses");
    assert_eq!(v["a"], json!("yes"));
    assert_eq!(v["b"], json!("no"));
    assert_eq!(v["c"], json!(true));
}

#[test]
fn merge_key_is_an_ordinary_key() {
    let v: Value = from_str("base: &b\n  x: 1\nd:\n  <<: *b\n  y: 2\n").expect("parses");
    assert_eq!(v["d"]["y"], json!(2));
    assert_eq!(v["d"]["<<"]["x"], json!(1), "<< is kept as a key, not merged");
    assert!(v["d"].get("x").is_none(), "no merge into the parent map");
}

#[test]
fn non_finite_floats_load_as_plain_strings() {
    let v: Value = from_str("p: .inf\nq: .nan\nr: 1.5\n").expect("loads, not a budget refusal");
    assert_eq!(v["p"], json!(".inf"), "kept as text, not a float");
    assert_eq!(v["q"], json!(".nan"), "kept as text, not a float");
    assert_eq!(v["r"], json!(1.5));
}

#[test]
fn writer_layout_for_scalars_and_nesting() {
    let v = json!({"a": "x", "n": 3, "t": {"f": true, "l": [1, 2]}});
    let text = serde_saphyr::to_string(&v).expect("renders");
    let back: Value = from_str(&text).expect("reads back what it wrote");
    assert_eq!(back, v, "round trip through the writer is lossless: {text}");
    assert!(text.contains("a: x\n"), "plain scalars stay unquoted: {text}");
    assert!(text.contains("\"n\": 3\n"), "integers stay bare, the key n is quoted: {text}");
}

#[test]
fn writer_key_order_follows_the_value_map_order() {
    let v = json!({"zeta": 1, "alpha": 2, "mid": 3});
    let text = serde_saphyr::to_string(&v).expect("renders");
    let keys: Vec<&String> = v.as_object().expect("object").keys().collect();
    let at: Vec<usize> = keys.iter().map(|k| text.find(k.as_str()).expect("key written")).collect();
    assert!(at.windows(2).all(|w| w[0] < w[1]), "written order == Map order {keys:?}: {text}");
}
