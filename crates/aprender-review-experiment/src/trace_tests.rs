// `json!` expands to an `unwrap` of an infallible `to_value`.
#![allow(clippy::disallowed_methods)]

use serde_json::{json, Value};

use super::*;

const WEIGHTS: &str = "3f1a9c0b5e2d4f6a8b7c9d0e1f2a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c";

fn sha(c: char) -> String {
    c.to_string().repeat(64)
}

/// A GREEN hosted (counted, silver) row.
fn hosted() -> Value {
    json!({
        "schema": "agent-trace-v1",
        "trace_id": "01K62ZQ4Y7N3V8E5R1T6W9XB2C",
        "quorum_id": "q-4400-1",
        "round": 1,
        "lane": "sonnet",
        "counted": true,
        "provider": "anthropic",
        "model_id": "claude-sonnet-5",
        "access_channel": "claude-code-cli",
        "terms_ref": {"url": "https://www.anthropic.com/legal/commercial-terms",
                      "effective_date": "2025-06-17", "fetched_at": "2026-09-25T17:00:00Z"},
        "served_by": "anthropic",
        "repo": "paiml/aprender",
        "pr": 4400,
        "head_sha": "a".repeat(40),
        "base_sha": "b".repeat(40),
        "diff_sha256": sha('d'),
        "input_sha": sha('e'),
        "input_parts": [{"role": "system", "blob_sha": sha('1'), "bytes": 812},
                        {"role": "user", "blob_sha": sha('2'), "bytes": 4096}],
        "prompt_version": "1.0.0",
        "output_sha": sha('f'),
        "decoding": {"temperature": 0.0, "top_p": null, "top_k": null, "max_tokens": 4096},
        "tokens": {"input": 4900, "output": 310},
        "latency_ms": {"ttft": 900, "total": 14_000},
        "lane_state": "Verdict",
        "verdict": "request_changes",
        "findings": [{"file": "src/a.rs", "line_start": 3, "line_end": 7,
                      "severity": "high", "category": "correctness", "text": "off by one"}],
        "parse_status": "ok",
        "logits_sha": null,
        "label_tier": "silver",
        "outcome": "pending",
        "split_guard": {"group_key": "paiml/aprender#4400", "split": "train",
                        "dedup_cluster": sha('d')},
        "secret_scan": {"scanner_versions": {"builtin": "1"}, "hits": 0, "status": "clean"},
        "trace_status": "ok",
        "agreed": true,
        "lane_disagreement": 1,
        "producer": {"binary": "quorum-rail", "version": "0.4.0", "git_sha": "c".repeat(40)},
    })
}

/// A GREEN local (shadow, uncounted) row with its logits.
fn local() -> Value {
    let mut v = hosted();
    v["trace_id"] = json!("01K62ZQ4Y7N3V8E5R1T6W9XB2D");
    v["lane"] = json!("qwen-shadow");
    v["counted"] = json!(false);
    v["provider"] = json!("local");
    v["model_id"] = json!(WEIGHTS);
    v["access_channel"] = json!("apr-serve");
    v["apr_tag"] = json!("v0.70.0");
    v["served_by"] = json!("gx10-cuda");
    v["logits_sha"] = json!(sha('9'));
    v
}

fn row(v: &Value) -> TraceRow {
    TraceRow::deserialize(v).expect("fixture parses")
}

fn red(what: &str, base: fn() -> Value, edit: impl FnOnce(&mut Value)) {
    let mut v = base();
    edit(&mut v);
    let reds = match TraceRow::deserialize(&v) {
        Err(_) => return, // refused at parse: RED
        Ok(r) => lint(&r, &v),
    };
    assert!(!reds.is_empty(), "{what} linted GREEN");
}

/// FALSIFY-ATR-001: a complete row is GREEN and survives a round trip byte
/// for byte of meaning; the index reader returns it.
#[test]
fn falsify_atr_001_a_complete_row_is_green_and_round_trips() {
    for v in [hosted(), local()] {
        let r = row(&v);
        assert_eq!(lint(&r, &v), Vec::<String>::new());
        let back = serde_json::to_value(&r).expect("ser");
        assert_eq!(row(&back), r);
        assert_eq!(lint(&r, &back), Vec::<String>::new());
    }
    let text = format!("{}\n\n{}\n", hosted(), local());
    let rows = parse_index(&text).expect("GREEN index");
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[1].lane, "qwen-shadow");
    assert_eq!(rows[0].findings[0].text, "off by one", "findings are text");
}

/// FALSIFY-ATR-002: any unknown identity field makes the row RED.
#[test]
fn falsify_atr_002_unknown_identity_is_red() {
    red("provider openai", hosted, |v| {
        v["provider"] = json!("openai")
    });
    red("a family alias", hosted, |v| {
        v["model_id"] = json!("sonnet")
    });
    red("model unknown", hosted, |v| {
        v["model_id"] = json!("unknown")
    });
    red("local model not a sha", local, |v| {
        v["model_id"] = json!("qwen3.5-4b")
    });
    red("channel/provider mismatch", hosted, |v| {
        v["access_channel"] = json!("antigravity")
    });
    red("no terms_ref", hosted, |v| {
        v.as_object_mut().expect("obj").remove("terms_ref");
    });
    red("served_by empty", hosted, |v| v["served_by"] = json!(" "));
    red("quorum_id empty", hosted, |v| v["quorum_id"] = json!(""));
    red("trace_id not a ULID", hosted, |v| {
        v["trace_id"] = json!("t-1")
    });
    red("input_sha short", hosted, |v| {
        v["input_sha"] = json!("e".repeat(63))
    });
    red("diff_sha256 upper", hosted, |v| {
        v["diff_sha256"] = json!("D".repeat(64))
    });
    red("head_sha short", hosted, |v| v["head_sha"] = json!("abc"));
    red("base_sha short", hosted, |v| v["base_sha"] = json!("abc"));
    red("no input parts", hosted, |v| v["input_parts"] = json!([]));
    red("bad part sha", hosted, |v| {
        v["input_parts"][0]["blob_sha"] = json!("x")
    });
    red("an unknown key", hosted, |v| v["temperature"] = json!(0.7));
    red("a missing key", hosted, |v| {
        v.as_object_mut().expect("obj").remove("producer");
    });
    red("wrong schema", hosted, |v| {
        v["schema"] = json!("agent-trace-v2")
    });
    red("unknown label tier", hosted, |v| {
        v["label_tier"] = json!("bronze")
    });
}

/// FALSIFY-ATR-003: `lane_state` and the output agree. A NotRun row is
/// still a row, with no output.
#[test]
fn falsify_atr_003_lane_state_and_output_agree() {
    let mut nr = hosted();
    nr["lane_state"] = json!({"NotRun": "Busy"});
    nr["output_sha"] = Value::Null;
    nr["verdict"] = Value::Null;
    nr["findings"] = json!([]);
    nr["agreed"] = Value::Null;
    assert_eq!(
        lint(&row(&nr), &nr),
        Vec::<String>::new(),
        "NotRun is a row"
    );
    red("NotRun with an output", hosted, |v| {
        v["lane_state"] = json!({"NotRun": "Timeout"})
    });
    red("Verdict without an output", hosted, |v| {
        v["output_sha"] = Value::Null
    });
    red("Refused with a verdict", hosted, |v| {
        v["lane_state"] = json!("Refused");
        v["output_sha"] = Value::Null;
        v["agreed"] = Value::Null;
    });
    red("parsed reply without verdict", hosted, |v| {
        v["verdict"] = Value::Null
    });
    red("an unknown NotRun reason", hosted, |v| {
        v["lane_state"] = json!({"NotRun": "Tired"})
    });
    red("agreed on a NotRun", hosted, |v| {
        v["lane_state"] = json!({"NotRun": "Busy"});
        v["output_sha"] = Value::Null;
        v["verdict"] = Value::Null;
        v["findings"] = json!([]);
    });
    red("empty finding text", hosted, |v| {
        v["findings"][0]["text"] = json!("")
    });
    red("inverted lines", hosted, |v| {
        v["findings"][0]["line_start"] = json!(9)
    });
    // A failed parse keeps its raw output and needs no verdict.
    let mut failed = hosted();
    failed["parse_status"] = json!("failed");
    failed["verdict"] = Value::Null;
    failed["findings"] = json!([]);
    assert_eq!(lint(&row(&failed), &failed), Vec::<String>::new());
}

/// FALSIFY-ATR-004: silver and gold stay separable, and local rows carry
/// their logits (top-k) while hosted ones never do.
#[test]
fn falsify_atr_004_silver_gold_and_logits_are_fenced() {
    red("hosted gold", hosted, |v| {
        v["label_tier"] = json!("gold");
        v["outcome"] = json!("merged");
    });
    red("gold pending", local, |v| v["label_tier"] = json!("gold"));
    let mut gold = local();
    gold["label_tier"] = json!("gold");
    gold["outcome"] = json!("merged");
    assert_eq!(lint(&row(&gold), &gold), Vec::<String>::new());
    red("local counted", local, |v| v["counted"] = json!(true));
    red("local without logits", local, |v| {
        v["logits_sha"] = Value::Null
    });
    red("local logits not a sha", local, |v| {
        v["logits_sha"] = json!("zz")
    });
    red("local without apr_tag", local, |v| {
        v.as_object_mut().expect("obj").remove("apr_tag");
    });
    red("hosted with logits", hosted, |v| {
        v["logits_sha"] = json!(sha('9'))
    });
}

/// FALSIFY-ATR-005: splits are keyed by repo#PR; the index reader names
/// every RED line and a repeated trace id; the tally counts unique diffs and
/// lane × tier × outcome.
#[test]
fn falsify_atr_005_split_key_index_and_tally() {
    red("group_key by row", hosted, |v| {
        v["split_guard"]["group_key"] = json!("01K62ZQ4Y7N3V8E5R1T6W9XB2C")
    });
    red("group_key other PR", hosted, |v| {
        v["split_guard"]["group_key"] = json!("paiml/aprender#4401")
    });
    let mut bad = hosted();
    bad["served_by"] = json!("");
    let text = format!("{}\n{}\nnot json\n{}\n", hosted(), bad, hosted());
    let errs = parse_index(&text).expect_err("RED index");
    assert!(
        errs.iter().any(|e| e.starts_with("line 2: served_by")),
        "{errs:?}"
    );
    assert!(errs.iter().any(|e| e.starts_with("line 3: ")), "{errs:?}");
    assert!(
        errs.iter()
            .any(|e| e.starts_with("line 4: trace_id") && e.ends_with("repeated")),
        "{errs:?}"
    );

    let mut other = local();
    other["trace_id"] = json!("01K62ZQ4Y7N3V8E5R1T6W9XB2E");
    let rows = parse_index(&format!("{}\n{}\n{}\n", hosted(), local(), other)).expect("GREEN");
    let t = tally(&rows);
    assert_eq!(t.rows, 3);
    assert_eq!(t.unique_diffs, 1, "three lanes on one diff");
    assert_eq!(t.by_lane_tier_outcome["qwen-shadow/silver/pending"], 2);
    assert_eq!(t.by_lane_tier_outcome["sonnet/silver/pending"], 1);
    assert_eq!(t.by_lane_tier_outcome.len(), 2);
}

#[test]
fn ulid_time_is_decoded() {
    assert_eq!(ulid_ms("00000000000000000000000000"), Some(0));
    assert_eq!(ulid_ms("0000000001ZZZZZZZZZZZZZZZZ"), Some(1));
    assert_eq!(ulid_ms("7ZZZZZZZZZZZZZZZZZZZZZZZZZ"), Some((1 << 48) - 1));
    assert_eq!(
        ulid_ms("8ZZZZZZZZZZZZZZZZZZZZZZZZZ"),
        None,
        "overflows 48 bits"
    );
    assert_eq!(
        ulid_ms("0000000000000000000000000U"),
        None,
        "U is not Crockford"
    );
    assert_eq!(ulid_ms("000"), None);
    let a = ulid_ms("01K62ZQ4Y7N3V8E5R1T6W9XB2C").expect("ulid");
    assert!(a > 1_700_000_000_000 && a < 1_900_000_000_000, "{a}");
}
