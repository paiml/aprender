// #3718: `apr run --json` says how many tokens the model read and why it stopped.

fn result_with(usage: RunUsage) -> RunResult {
    RunResult {
        text: "4".to_string(),
        duration_secs: 1.0,
        cached: true,
        // The word-count stand-in `run_model` substitutes when the engine
        // reports nothing. `completion_tokens` must never echo it.
        tokens_generated: Some(1),
        tok_per_sec: None,
        used_gpu: Some(false),
        generated_tokens: None,
        token_texts: None,
        usage,
        constraint_refusal: None,
    }
}

#[test]
fn json_carries_the_engine_counts_and_the_finish() {
    let usage = RunUsage {
        prompt_tokens: Some(79),
        completion_tokens: Some(32),
        finish_reason: Some("length"),
        context_length: Some(262_144),
    };
    let v = build_final_json(&result_with(usage), "m.gguf", 32, false);
    assert_eq!(v["prompt_tokens"], 79);
    assert_eq!(v["completion_tokens"], 32);
    assert_eq!(v["finish_reason"], "length");
    assert_eq!(v["context_length"], 262_144);
}

/// Not reported is `null`, never `0` or `"stop"`: a zero reads as measured and
/// a default "stop" is the silent cut this row removes.
#[test]
fn unreported_fields_are_null_not_zero() {
    let v = build_final_json(&result_with(RunUsage::default()), "m.apr", 32, false);
    for key in ["prompt_tokens", "completion_tokens", "finish_reason", "context_length"] {
        assert!(v.get(key).is_some(), "{key} must be present so consumers can key on it");
        assert!(v[key].is_null(), "{key} must be null when unreported, got {}", v[key]);
    }
    assert_eq!(v["tokens_generated"], 1, "the legacy field keeps its stand-in");
}

/// `--stream`'s terminal event is built by the same function, so it carries
/// the same fields.
#[test]
fn stream_final_event_carries_them_too() {
    let usage = RunUsage {
        prompt_tokens: Some(12),
        completion_tokens: Some(3),
        finish_reason: Some("stop"),
        context_length: Some(4096),
    };
    let mut buf: Vec<u8> = Vec::new();
    write_stream_output(&mut buf, &result_with(usage), "m.gguf", 8, false).expect("write");
    let text = String::from_utf8(buf).expect("utf-8");
    let last = text.lines().last().expect("final line");
    let v: serde_json::Value = serde_json::from_str(last).expect("json");
    assert_eq!(v["event"], "final");
    assert_eq!(v["prompt_tokens"], 12);
    assert_eq!(v["finish_reason"], "stop");
}
