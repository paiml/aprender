// #7331: `apr run` on the Qwen3.8-2.4T-A95B checkpoint refuses UP FRONT, with a
// named reason, before any download, cache-dir creation, or allocation.
// Offline: nothing here touches a network or a weights file.

const QWEN38_2_4T: &str = "Qwen/Qwen3.8-2.4T-A95B";

#[test]
fn oversized_checkpoint_refusal_names_size_and_missing_forward() {
    for spelling in [
        QWEN38_2_4T,
        "hf://Qwen/Qwen3.8-2.4T-A95B",
        "qwen/qwen3.8-2.4t-a95b",
        "hf://Qwen/Qwen3.8-2.4T-A95B/",
        // Lane-2 finding 2 (PR #7356): a file-qualified hf:// URI and the HF web URLs name
        // the same repo, and a run on them reaches the same download.
        "hf://Qwen/Qwen3.8-2.4T-A95B/model-00001-of-00500.safetensors",
        "hf://qwen/qwen3.8-2.4t-a95b/config.json",
        "https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B",
        "https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/resolve/main/model.safetensors",
        "https://hf.co/Qwen/Qwen3.8-2.4T-A95B/",
        // Round 2 (lane-2 finding 1): one normaliser for every form. A bare file-qualified
        // ref is promoted to hf:// by run.rs when it is not on disk, then downloaded.
        "Qwen/Qwen3.8-2.4T-A95B/model.safetensors",
        "Qwen/Qwen3.8-2.4T-A95B/model.safetensors.index.json",
        "https://www.hf.co/Qwen/Qwen3.8-2.4T-A95B/resolve/main/model.safetensors",
        "http://www.huggingface.co/Qwen/Qwen3.8-2.4T-A95B",
        "HTTPS://HuggingFace.co/Qwen/Qwen3.8-2.4T-A95B",
        "https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B?download=true",
        "https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B#model-card",
        "https://huggingface.co/api/models/Qwen/Qwen3.8-2.4T-A95B",
        "huggingface.co/Qwen/Qwen3.8-2.4T-A95B/blob/main/config.json",
        "hf://Qwen/Qwen3.8-2.4T-A95B@main/config.json",
    ] {
        let reason = known_checkpoint_refusal(spelling)
            .unwrap_or_else(|| panic!("{spelling} must be refused up front"));
        assert!(reason.contains("Qwen3.8-2.4T-A95B"), "names the model: {reason}");
        assert!(reason.contains("TB"), "names the size: {reason}");
        assert!(reason.contains("qwen3_5_moe_text"), "names the arch: {reason}");
        assert!(reason.contains("no CUDA forward"), "names the missing forward: {reason}");
        assert!(reason.contains("#7331"), "names the ticket: {reason}");
        assert!(
            reason.contains("nothing was downloaded"),
            "says nothing was fetched: {reason}"
        );
    }
}

#[test]
fn other_checkpoints_are_not_refused_by_the_oversized_table() {
    for ok in [
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3.8-Flash-Next",
        "models/tiny.gguf",
        "hf://Qwen/Qwen3.8-2.4T-A95B-GGUF-not-this-one",
        "hf://Qwen/Qwen3.8-2.4T-A95B-GGUF-not-this-one/model.gguf",
        "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors",
        "https://example.com/Qwen/Qwen3.8-2.4T-A95B",
        "s3://Qwen/Qwen3.8-2.4T-A95B",
        "https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B-GGUF-not-this-one?x=Qwen/Qwen3.8-2.4T-A95B",
        "Qwen",
        "",
    ] {
        assert!(
            known_checkpoint_refusal(ok).is_none(),
            "{ok:?} is not the 2.4T checkpoint and must not be refused here"
        );
    }
}

/// The real entry point: refuses with exit 12 in well under 5 s, and the cache dir
/// a download would create (`~/.apr/cache/hf/<org>/<repo>`) never appears. If the
/// refusal ran after `resolve_model_source`/`resolve_model`, either the network
/// probe would be attempted (error text would differ) or `hf_cache_dir` would
/// create that directory.
#[test]
fn apr_run_refuses_before_download_or_allocation() {
    let cache = dirs::home_dir()
        .expect("home")
        .join(".apr/cache/hf/Qwen/Qwen3.8-2.4T-A95B");
    let existed_before = cache.exists();
    let start = std::time::Instant::now();
    let err = run(
        QWEN38_2_4T,
        None,
        Some("hi"),
        8,
        false,
        None,
        None,
        "text",
        false,
        false,
        false, // offline = false: the network is NOT blocked, the refusal alone stops it
        false,
        false,
        false,
        None,
        false,
        None,
        "basic",
        false,
        0.0,
        40,
        None,
        0,
        1.0,
        64,
        false,
        false,
        None,
        0,
        ConstraintArgs { json_schema: None, grammar: None },
    )
    .expect_err("the 2.4T checkpoint must be refused");
    assert!(start.elapsed() < std::time::Duration::from_secs(5), "refusal is instant");
    assert_eq!(err.exit_code_value(), 12, "NotImplemented: this build cannot do that");
    let text = err.to_string();
    assert!(text.contains("Qwen3.8-2.4T-A95B") && text.contains("TB"), "{text}");
    assert!(
        existed_before || !cache.exists(),
        "a download cache dir was created: the refusal ran too late"
    );
}

/// Lane-2 finding 1 (PR #7356): `--batch-jsonl` went to `run_batch`, which resolved (and
/// downloaded) the model without the refusal. Network blocked via a dead proxy, so a late
/// refusal shows as a different error, not a download.
#[cfg(feature = "inference")]
#[test]
fn apr_run_batch_refuses_before_download_or_allocation() {
    let cache = dirs::home_dir()
        .expect("home")
        .join(".apr/cache/hf/Qwen/Qwen3.8-2.4T-A95B");
    let existed_before = cache.exists();
    let err = run_batch(
        "hf://Qwen/Qwen3.8-2.4T-A95B/model-00001-of-00500.safetensors",
        Path::new("/nonexistent/prompts.jsonl"),
        8,
        0.0,
        40,
        true,
        false,
        &ConstraintArgs { json_schema: None, grammar: None },
    )
    .expect_err("the 2.4T checkpoint must be refused on the batch path");
    assert_eq!(err.exit_code_value(), 12, "{err}");
    assert!(err.to_string().contains("#7331"), "{err}");
    assert!(
        existed_before || !cache.exists(),
        "a download cache dir was created: the refusal ran too late"
    );
}
