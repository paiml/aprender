// #3720: `apr run --json` is a response contract: status, error, weights digest.

fn contract_result(text: &str) -> RunResult {
    RunResult {
        logprobs: None,
        text: text.to_string(),
        duration_secs: 1.0,
        cached: true,
        tokens_generated: Some(1),
        tok_per_sec: None,
        used_gpu: Some(false),
        gpu_attempted: None,
        generated_tokens: None,
        token_texts: None,
        usage: RunUsage::default(),
        constraint_refusal: None,
    }
}

#[test]
fn ok_run_has_status_ok_and_null_error() {
    let v = build_final_json(&contract_result("4"), "m.gguf", 8, false);
    assert_eq!(v["status"], "ok");
    assert!(v["error"].is_null());
    assert_eq!(v["apr_version"], env!("CARGO_PKG_VERSION"));
    assert!(v["apr_commit"].is_string());
}

#[test]
fn empty_completion_is_failed_with_the_inference_exit_code() {
    for text in ["", "  \n\t"] {
        let r = contract_result(text);
        let v = build_final_json(&r, "m.gguf", 8, false);
        assert_eq!(v["status"], "failed", "{text:?}");
        assert_eq!(v["error"]["kind"], "empty_completion");
        assert_eq!(v["error"]["exit_code"], 8);
        let err = empty_completion_exit(&r, false).expect_err("must be an error");
        assert_eq!(err.exit_code_value(), 8, "document and process must agree");
    }
    assert!(empty_completion_exit(&contract_result("4"), false).is_ok());
}

#[test]
fn forced_accelerator_that_did_not_run_is_refused_not_failed() {
    let v = build_final_json(&contract_result("4"), "m.gguf", 8, true);
    assert_eq!(v["status"], "refused");
    assert_eq!(v["error"]["kind"], "backend_unavailable");
    assert_eq!(v["error"]["exit_code"], 14);
    // an empty forced-GPU fallback is still a refusal: it was declined, it did not "fail to run"
    let e = build_final_json(&contract_result(""), "m.gguf", 8, true);
    assert_eq!(e["status"], "refused");
}

#[test]
fn model_digest_is_the_sha256_of_the_file_and_null_for_a_hub_id() {
    let dir = std::env::temp_dir().join(format!("apr-3720-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let p = dir.join("w.bin");
    std::fs::write(&p, b"abc").expect("write");
    let v = with_model_digest(serde_json::json!({}), p.to_str().expect("utf8"));
    assert_eq!(
        v["model_digest"],
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    let hub = with_model_digest(serde_json::json!({}), "hf://org/model");
    assert!(hub["model_digest"].is_null());
    std::fs::remove_dir_all(&dir).expect("cleanup");
}
