// #3718: `apr run --json` says how many tokens the model read and why it stopped.

fn result_with(usage: RunUsage) -> RunResult {
    RunResult {
        logprobs: None,
        text: "4".to_string(),
        duration_secs: 1.0,
        cached: true,
        // The word-count stand-in `run_model` substitutes when the engine
        // reports nothing. `completion_tokens` must never echo it.
        tokens_generated: Some(1),
        tok_per_sec: None,
        used_gpu: Some(false),
        gpu_attempted: None,
        generated_tokens: None,
        token_texts: None,
        usage,
    }
}

#[test]
fn json_carries_the_engine_counts_and_the_finish() {
    let usage = RunUsage {
        prompt_tokens: Some(79),
        completion_tokens: Some(32),
        finish_reason: Some("length"),
        context_length: Some(262_144),
        generation_ms: None,
        setup_ms: None,
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
        generation_ms: None,
        setup_ms: None,
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

/// #3981: the JSON says how much of the window was generation and how much setup,
/// so a consumer can tell a generation rate from a whole-window rate.
#[test]
fn json_carries_generation_and_setup_ms() {
    let usage = RunUsage { generation_ms: Some(1_000), setup_ms: Some(5_000), ..RunUsage::default() };
    let v = build_final_json(&result_with(usage), "m.gguf", 32, false);
    assert_eq!(v["generation_ms"], 1_000);
    assert_eq!(v["setup_ms"], 5_000);
    let v = build_final_json(&result_with(RunUsage::default()), "m.gguf", 32, false);
    assert!(v.get("generation_ms").is_some() && v["generation_ms"].is_null(), "present, null when unmeasured");
    assert!(v.get("setup_ms").is_some() && v["setup_ms"].is_null(), "present, null when unmeasured");
}

