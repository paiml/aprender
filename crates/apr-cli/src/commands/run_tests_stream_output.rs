
/// `--stream` emits one JSON token line per generated token id, then one
/// `event:final` blob — N tokens → N+1 NDJSON lines.
#[test]
fn stream_output_emits_n_plus_one_json_lines() {
    let result = RunResult {
        text: "Hello world".to_string(),
        duration_secs: 0.25,
        cached: false,
        tokens_generated: Some(3),
        tok_per_sec: Some(12.0),
        used_gpu: Some(false),
        generated_tokens: Some(vec![100, 200, 300]),
    };

    let mut buf: Vec<u8> = Vec::new();
    write_stream_output(&mut buf, &result, "model.gguf", 32).expect("write must succeed");
    let s = String::from_utf8(buf).expect("utf-8");
    let lines: Vec<&str> = s.lines().collect();
    assert_eq!(
        lines.len(),
        4,
        "3 tokens + 1 final = 4 NDJSON lines, got: {s}"
    );

    // Each token line has expected shape and ascending index.
    for (i, line) in lines[..3].iter().enumerate() {
        let v: serde_json::Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("token line {i} must be valid JSON: {e} | {line}"));
        assert_eq!(v["event"], "token", "line {i}: event must be 'token'");
        assert_eq!(v["index"], i as i64, "line {i}: index field");
        assert!(v["token_id"].is_u64(), "line {i}: token_id is u64");
        assert!(v["text"].is_string(), "line {i}: text field present");
    }
    let token_ids: Vec<u64> = lines[..3]
        .iter()
        .map(|l| {
            serde_json::from_str::<serde_json::Value>(l).expect("json")["token_id"]
                .as_u64()
                .expect("u64")
        })
        .collect();
    assert_eq!(token_ids, vec![100, 200, 300], "token ids in order");

    // Final line must be event=final with full payload.
    let final_v: serde_json::Value = serde_json::from_str(lines[3]).expect("final json");
    assert_eq!(final_v["event"], "final");
    assert_eq!(final_v["model"], "model.gguf");
    assert_eq!(final_v["text"], "Hello world");
    assert_eq!(final_v["tokens_generated"], 3);
    assert_eq!(final_v["max_tokens"], 32);
    assert_eq!(final_v["tok_per_sec"], 12.0);
    assert_eq!(final_v["used_gpu"], false);
    assert_eq!(final_v["cached"], false);
}

/// Empty token list still emits exactly one `final` blob (no tokens, no
/// orphan lines).
#[test]
fn stream_output_no_tokens_emits_only_final() {
    let result = RunResult {
        text: String::new(),
        duration_secs: 0.0,
        cached: true,
        tokens_generated: Some(0),
        tok_per_sec: Some(0.0),
        used_gpu: Some(false),
        generated_tokens: Some(Vec::new()),
    };

    let mut buf: Vec<u8> = Vec::new();
    write_stream_output(&mut buf, &result, "noprompt.apr", 1).expect("write must succeed");
    let s = String::from_utf8(buf).expect("utf-8");
    let lines: Vec<&str> = s.lines().collect();
    assert_eq!(lines.len(), 1, "0 tokens + 1 final = 1 line, got: {s}");
    let v: serde_json::Value = serde_json::from_str(lines[0]).expect("final json");
    assert_eq!(v["event"], "final");
    assert_eq!(v["tokens_generated"], 0);
}

/// `generated_tokens: None` is treated identically to an empty vec (no
/// token lines, just the final blob).
#[test]
fn stream_output_none_tokens_emits_only_final() {
    let result = RunResult {
        text: String::new(),
        duration_secs: 0.0,
        cached: false,
        tokens_generated: None,
        tok_per_sec: None,
        used_gpu: None,
        generated_tokens: None,
    };

    let mut buf: Vec<u8> = Vec::new();
    write_stream_output(&mut buf, &result, "x.apr", 1).expect("write");
    let s = String::from_utf8(buf).expect("utf-8");
    assert_eq!(s.lines().count(), 1);
    let v: serde_json::Value =
        serde_json::from_str(s.lines().next().expect("line")).expect("final json");
    assert_eq!(v["event"], "final");
}

/// `build_final_json` rounds tok_per_sec to one decimal and rebuilds the
/// inference_time_ms field — match the legacy `--json` output shape.
#[test]
fn build_final_json_matches_legacy_json_shape() {
    let result = RunResult {
        text: "abc".to_string(),
        duration_secs: 1.0,
        cached: true,
        tokens_generated: Some(10),
        tok_per_sec: Some(99.99),
        used_gpu: Some(true),
        generated_tokens: Some(vec![1, 2, 3]),
    };
    let v = build_final_json(&result, "src.apr", 100);
    assert_eq!(v["model"], "src.apr");
    assert_eq!(v["text"], "abc");
    assert_eq!(v["tokens"], serde_json::json!([1, 2, 3]));
    assert_eq!(v["tokens_generated"], 10);
    assert_eq!(v["max_tokens"], 100);
    // 99.99 → rounded to 100.0 (one decimal)
    assert_eq!(v["tok_per_sec"], 100.0);
    assert_eq!(v["used_gpu"], true);
    assert_eq!(v["cached"], true);
    assert_eq!(v["inference_time_ms"], 1000.0);
}
