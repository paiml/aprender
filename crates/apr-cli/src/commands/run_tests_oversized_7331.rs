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
        // Round 3 (lane 2 finding 1): the receipt's six forms, parsed with a real URL
        // parser and canonicalised.
        "https://huggingface.co:443/Qwen/Qwen3.8-2.4T-A95B/resolve/main/model.safetensors",
        "https://huggingface.co./Qwen/Qwen3.8-2.4T-A95B",
        "https://user@huggingface.co/Qwen/Qwen3.8-2.4T-A95B",
        "hf://Qwen/Qwen3.8-2.4T-A95B:Q4_K_M",
        "https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B.git",
        "hf://Qwen%2FQwen3.8-2.4T-A95B",
        // ... and their neighbours.
        "https://user:pw@HF.CO.:8443/Qwen/Qwen3.8-2.4T-A95B.GIT",
        "huggingface.co:443/Qwen/Qwen3.8-2.4T-A95B",
        "user@huggingface.co/Qwen/Qwen3.8-2.4T-A95B",
        "https://huggingface.co/Qwen%2FQwen3.8-2.4T-A95B",
        "https://huggingface.co/%51wen/Qwen3.8-2.4T-A95B",
        "Qwen/Qwen3.8-2.4T-A95B:Q4_K_M",
        "Qwen/Qwen3.8-2.4T-A95B.git",
        "hf://Qwen/Qwen3.8-2.4T-A95B.git@main/config.json",
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
        // Round 3 negative controls: the canonicaliser must not widen the match.
        "https://example.com:443/Qwen/Qwen3.8-2.4T-A95B",
        "https://huggingface.co.evil.example/Qwen/Qwen3.8-2.4T-A95B",
        "https://huggingface.co@evil.example/Qwen/Qwen3.8-2.4T-A95B",
        "https://huggingface.co..//Qwen/Qwen3.8-2.4T-A95B",
        "ftp://huggingface.co/Qwen/Qwen3.8-2.4T-A95B",
        "Qwen/Qwen3.8-2.4T-A95B-Other.git",
        "hf://Qwen/Qwen3.8-2.4T-A95B-GGUF:Q4_K_M",
        "hf://Qwen%2FQwen3.8-2.4T-A95B-Other",
        "hf://Qwen/Qwen3.8-2.4T-A95Bgit",
        "hf://Qwen/.git",
        "hf://Qwen/:Q4_K_M",
    ] {
        assert!(
            known_checkpoint_refusal(ok).is_none(),
            "{ok:?} is not the 2.4T checkpoint and must not be refused here"
        );
    }
}

/// The canonical id itself, row by row (round 3): the normaliser returns exactly the
/// `org/repo`, so a refusal is never an accident of a prefix match.
#[test]
fn checkpoint_repo_id_canonicalises_every_url_form() {
    for (source, want) in [
        ("https://huggingface.co:443/Qwen/Qwen3.8-2.4T-A95B/resolve/main/x", Some("Qwen/Qwen3.8-2.4T-A95B")),
        ("https://huggingface.co./Qwen/Qwen3.8-2.4T-A95B", Some("Qwen/Qwen3.8-2.4T-A95B")),
        ("https://user@huggingface.co/Qwen/Qwen3.8-2.4T-A95B", Some("Qwen/Qwen3.8-2.4T-A95B")),
        ("hf://Qwen/Qwen3.8-2.4T-A95B:Q4_K_M", Some("Qwen/Qwen3.8-2.4T-A95B")),
        ("https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B.git", Some("Qwen/Qwen3.8-2.4T-A95B")),
        ("hf://Qwen%2FQwen3.8-2.4T-A95B", Some("Qwen/Qwen3.8-2.4T-A95B")),
        ("hf://Qwen/Qwen3.8-2.4T-A95B@v1:Q4.git", Some("Qwen/Qwen3.8-2.4T-A95B")),
        ("Qwen/Qwen3-0.6B", Some("Qwen/Qwen3-0.6B")),
        ("https://hf.co/api/models/Qwen/Qwen3-0.6B?x=1", Some("Qwen/Qwen3-0.6B")),
        ("https://example.com/Qwen/Qwen3.8-2.4T-A95B", None),
        ("https://huggingface.co.evil.example/Qwen/Qwen3.8-2.4T-A95B", None),
        ("s3://Qwen/Qwen3.8-2.4T-A95B", None),
        ("hf://Qwen/.git", None),
        ("hf://Qwen", None),
    ] {
        assert_eq!(checkpoint_repo_id(source).as_deref(), want, "{source}");
    }
}

/// The exact refusal, measured through the real default path (PR #7356 round 4, lane 2
/// F1): `known_checkpoint_refusal` with the build's own capability table, not a synthetic
/// reason. In the default build (`inference` on) realizar's Qwen3.5-MoE reason exists, so
/// the no-CUDA-forward clause is stated once and the reason's CPU advice and second refusal are not quoted.
const QWEN38_REFUSAL: &str = "refusing to run Qwen/Qwen3.8-2.4T-A95B: the checkpoint is \
     about 4.9 TB in bf16 (model_type qwen3_5_moe_text), which exceeds every supported host, \
     and there is no CUDA forward for it: Qwen3.5 MoE is a hybrid (Gated DeltaNet / SSM \
     layers + mixture-of-experts), and the qwen3moe CUDA forward (#3714) does not run SSM \
     layers. Refused before any download or allocation: \
     nothing was downloaded and nothing was loaded (#7331).";

#[test]
fn oversized_refusal_text_is_exact_on_the_default_path() {
    #[cfg(feature = "inference")]
    assert!(
        realizar::capability::no_cuda_forward_reason("qwen3_5_moe_text").is_some(),
        "the default build's capability table must still refuse qwen3_5_moe_text"
    );
    // Synthetic reasons: bare generic, real-shaped, and one with no full stop.
    for (reason, want) in [
        ("no CUDA forward for architecture 'm'", "for it. Refused"),
        ("this build has no CUDA forward for architecture 'm'.", "for it. Refused"),
        ("this build has no CUDA forward for architecture 'm': SSM has no kernel. Re-run without --gpu. This is a refusal, x.", "for it: SSM has no kernel. Refused"),
        ("SSM has no kernel", "for it: SSM has no kernel. Refused"),
    ] {
        let text = refusal_text("o/r", "m", 1_000_000_000_000, reason);
        assert!(text.contains(want), "{text}");
        assert!(!text.contains("..") && !text.contains("--gpu"), "{text}");
    }
    let real = known_checkpoint_refusal(QWEN38_2_4T).expect("refused");
    #[cfg(feature = "inference")]
    assert_eq!(real, QWEN38_REFUSAL);
    assert_eq!(real.matches("no CUDA forward").count(), 1, "{real}");
    assert_eq!(real.matches("nothing was loaded").count(), 1, "{real}");
    assert!(!real.contains("--gpu") && !real.contains("CPU"), "{real}");
}


/// Every path under `dir`, sorted, or `None` when it does not exist. Lane 2 finding 5
/// (round 3): the old `existed_before || !cache.exists()` passed for any refusal order
/// once the dir existed; the listing must be unchanged instead.
fn cache_listing(dir: &Path) -> Option<Vec<std::path::PathBuf>> {
    fn walk(d: &Path, out: &mut Vec<std::path::PathBuf>) {
        if let Ok(rd) = std::fs::read_dir(d) {
            for e in rd.flatten() {
                let p = e.path();
                out.push(p.clone());
                if p.is_dir() {
                    walk(&p, out);
                }
            }
        }
    }
    if !dir.exists() {
        return None;
    }
    let mut out = Vec::new();
    walk(dir, &mut out);
    out.sort();
    Some(out)
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
    let before = cache_listing(&cache);
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
    assert_eq!(
        cache_listing(&cache),
        before,
        "the download cache changed: the refusal ran too late"
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
    let before = cache_listing(&cache);
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
    assert_eq!(
        cache_listing(&cache),
        before,
        "the download cache changed: the refusal ran too late"
    );
}
