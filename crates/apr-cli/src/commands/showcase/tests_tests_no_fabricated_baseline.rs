// #3773: `apr showcase` printed a llama.cpp baseline that was never measured —
// `35.0 + generate_jitter() * 1.5` tok/s whenever any llama-server was on PATH —
// and a "speedup vs llama.cpp" from it. A baseline is now measured or named
// UNMEASURED with its reason; it is never a constant.

/// The llama.cpp baseline is UNMEASURED, every time, with the reason. A constant
/// with clock jitter (the #3773 mutant) returns Ok and turns this RED.
#[test]
fn llama_cpp_baseline_is_unmeasured_never_a_number() {
    let config = ShowcaseConfig::default();
    for _ in 0..3 {
        match super::benchmark::run_llama_cpp_bench(&config) {
            Ok((tps, ttft)) => panic!(
                "run_llama_cpp_bench returned a number ({tps} tok/s, {ttft} ms) it never \
                 measured — #3773's fabricated baseline is back"
            ),
            Err(e) => {
                let why = e.to_string();
                assert!(why.contains("UNMEASURED:"), "the refusal must say UNMEASURED: {why}");
                assert!(why.contains("llama_bin.sh"), "the reason names the pinned lane: {why}");
            }
        }
    }
}

/// A requested baseline that does not measure is RECORDED with its reason and
/// yields no speedup; the table and the JSON export say UNMEASURED.
#[test]
fn an_unmeasured_baseline_is_recorded_and_yields_no_speedup() {
    let config = ShowcaseConfig {
        baselines: vec![Baseline::LlamaCpp],
        ..Default::default()
    };
    let bench = super::benchmark::build_comparison(44.0, 78.0, 1.0, 30, &config);
    assert!(bench.llama_cpp_tps.is_none(), "no llama.cpp number may be recorded");
    assert!(bench.llama_cpp_ttft_ms.is_none());
    assert!(bench.speedup_vs_llama.is_none(), "no speedup against an unmeasured baseline");
    let why = bench.unmeasured.get("llama.cpp").expect("the reason is recorded");
    assert!(why.starts_with("UNMEASURED:"), "{why}");

    let json = serde_json::to_string(&bench).expect("serialize");
    assert!(json.contains("\"unmeasured\""), "the export carries the reason: {json}");
    assert!(json.contains("UNMEASURED:"), "{json}");
}

/// A baseline that was not requested is neither measured nor listed.
#[test]
fn an_unrequested_baseline_is_absent_not_unmeasured() {
    let config = ShowcaseConfig {
        baselines: vec![],
        ..Default::default()
    };
    let bench = super::benchmark::build_comparison(44.0, 78.0, 1.0, 30, &config);
    assert!(bench.unmeasured.is_empty());
    assert!(!serde_json::to_string(&bench).expect("serialize").contains("unmeasured"));
}

/// Ollama IS measured (its own eval counters), but a response that lacks them
/// used to fall back to hard-coded throughput and TTFT constants. Each gap is
/// UNMEASURED now.
#[test]
fn ollama_counters_or_unmeasured_never_a_fallback_constant() {
    let ok = r#"{"eval_count":100,"eval_duration":2000000000,"prompt_eval_duration":250000000}"#;
    let (tps, ttft) = super::benchmark::ollama_tps_ttft(ok).expect("complete response");
    assert!((tps - 50.0).abs() < 1e-9, "100 tokens in 2 s is 50 tok/s, got {tps}");
    assert!((ttft - 250.0).abs() < 1e-9, "250 ms prompt eval, got {ttft}");

    for (body, missing) in [
        (r#"{"eval_duration":2000000000,"prompt_eval_duration":1}"#, "eval_count"),
        (r#"{"eval_count":100,"prompt_eval_duration":1}"#, "eval_duration"),
        (r#"{"eval_count":100,"eval_duration":0,"prompt_eval_duration":1}"#, "eval_duration 0"),
        (r#"{"eval_count":100,"eval_duration":2000000000}"#, "prompt_eval_duration"),
        (r#"{"error":"model not found"}"#, "eval_count"),
    ] {
        let err = super::benchmark::ollama_tps_ttft(body)
            .expect_err("an incomplete response is not a measurement")
            .to_string();
        assert!(err.contains("UNMEASURED:") && err.contains(missing), "{body}: {err}");
    }
}
