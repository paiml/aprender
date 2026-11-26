//! #7330: `apr inspect --no-weights` names every tensor family from config + index only.

use super::*;
use std::path::PathBuf;

fn fixture(name: &str) -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/qwen38_2_4t")
        .join(name);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

fn sha256_hex(s: &str) -> String {
    use sha2::Digest;
    sha2::Sha256::digest(s.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn count_of(r: &FamilyReport, family: &str) -> usize {
    r.families
        .iter()
        .find(|f| f.family == family)
        .map_or(0, |f| f.count)
}

/// #7330: the PUBLISHED Qwen3.8-2.4T-A95B config and index (verbatim, hashes in the fixture
/// README), not a synthetic stand-in: every one of the 1609 tensors has a family name.
#[test]
fn every_tensor_in_the_published_2_4t_index_is_named() {
    let (cfg, idx) = (
        fixture("config.json"),
        fixture("model.safetensors.index.json"),
    );
    assert_eq!(
        sha256_hex(&cfg),
        "4e3819548967e319ab435d044a3a331dbe3b078590ce822e9d74b79430533987"
    );
    assert_eq!(
        sha256_hex(&idx),
        "e36c40d4e99b2714fff821218a0433bda2dec46afdb1ebee8ce96ced997928ee"
    );
    let r = report(&cfg, &idx).expect("report");
    assert!(r.unnamed.is_empty(), "unnamed tensors: {:?}", r.unnamed);
    assert_eq!(r.tensor_count, 1609);
    assert_eq!(r.model_type, "qwen3_5_moe_text");
    assert_eq!(r.num_hidden_layers, Some(92));
    assert_eq!(r.num_experts, Some(512));
    // The index writes total_size as a float (4892365451008.0): still a byte count.
    assert_eq!(r.total_size_bytes, Some(4_892_365_451_008));
    // All 92 layers are in the index, each with its packed 512-expert pair.
    assert_eq!(r.layers_in_index, 92);
    assert_eq!(count_of(&r, "language.mlp.experts"), 2 * 92);
    assert_eq!(count_of(&r, "language.linear_attn"), 9 * 69);
    assert_eq!(count_of(&r, "language.self_attn"), 6 * 23);
    assert_eq!(count_of(&r, "mtp.fc"), 1);
    let names: Vec<&str> = r.families.iter().map(|f| f.family.as_str()).collect();
    assert_eq!(names.len(), 22, "{names:?}");
    for want in [
        "language.embed_tokens",
        "language.final_norm",
        "head.lm_head",
        "language.linear_attn",
        "language.self_attn",
        "language.mlp.gate",
        "language.mlp.experts",
        "language.mlp.shared_expert",
        "language.mlp.shared_expert_gate",
        "language.input_layernorm",
        "language.post_attention_layernorm",
        "mtp.fc",
        "mtp.final_norm",
        "mtp.pre_fc_norm_hidden",
        "mtp.pre_fc_norm_embedding",
        "mtp.mlp.experts",
        "mtp.self_attn",
    ] {
        assert!(
            names.contains(&want),
            "family {want} missing from {names:?}"
        );
    }
    let total: usize = r.families.iter().map(|f| f.count).sum();
    assert_eq!(
        total, r.tensor_count,
        "every tensor is counted in exactly one family"
    );
    assert_eq!(render_json(&r)["layers_in_index"], 92);
}

const REPO: &str = "Qwen/Qwen3.8-2.4T-A95B";

fn seed(dir: &std::path::Path, files: &[&str]) {
    std::fs::create_dir_all(dir).expect("mkdir");
    for f in files {
        std::fs::write(dir.join(f), fixture(f)).expect("seed");
    }
}

/// #7330: a bare repo id is answered from the `apr pull` cache with no network call.
#[test]
fn a_bare_repo_id_is_read_from_the_cache_offline() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let cache = tmp.path().join("Qwen").join("Qwen3.8-2.4T-A95B");
    seed(&cache, &["config.json", "model.safetensors.index.json"]);
    let asked = std::cell::RefCell::new(Vec::new());
    let dir = no_weights_dir(
        std::path::Path::new(REPO),
        &|r| {
            asked.borrow_mut().push(r.to_string());
            Ok(cache.clone())
        },
        &|url, _| panic!("a cached repo needs no network, asked for {url}"),
    )
    .expect("resolved from the cache");
    assert_eq!(dir, cache);
    assert_eq!(*asked.borrow(), [REPO]);
    let read = |f: &str| std::fs::read_to_string(dir.join(f)).expect("read");
    let r = report(&read("config.json"), &read("model.safetensors.index.json")).expect("report");
    assert_eq!((r.tensor_count, r.unnamed.len()), (1609, 0));
    // `hf://org/repo` is the same repo.
    let dir = no_weights_dir(
        std::path::Path::new("hf://Qwen/Qwen3.8-2.4T-A95B"),
        &|_| Ok(cache.clone()),
        &|url, _| panic!("{url}"),
    )
    .expect("hf:// form");
    assert_eq!(dir, cache);
}

/// Nothing cached and the network forbidden: a named error through the real `apr pull`
/// download path (its `--offline` guard), never an empty report, and no file left behind.
#[test]
fn an_uncached_repo_id_offline_is_a_named_error() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let cache = tmp.path().join("Qwen").join("Qwen3.8-2.4T-A95B");
    let _offline = crate::commands::offline::scope(true);
    let err = no_weights_dir(
        std::path::Path::new(REPO),
        &|_| Ok(cache.clone()),
        &|url, path| crate::commands::pull::download_file(url, path),
    )
    .expect_err("no cache and no network is an error");
    assert!(matches!(err, CliError::NetworkError(_)), "{err:?}");
    let msg = err.to_string();
    for want in [REPO, "config.json", "not in the cache", "--offline"] {
        assert!(msg.contains(want), "{want:?} not in {msg}");
    }
    assert!(!cache.exists(), "a failed fetch leaves no cache directory");
}

/// Only the missing file is fetched, from the repo's `resolve/main`, and lands in the cache.
#[test]
fn a_partly_cached_repo_id_fetches_only_the_missing_file() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let cache = tmp.path().join("Qwen").join("Qwen3.8-2.4T-A95B");
    seed(&cache, &["config.json"]);
    let urls = std::cell::RefCell::new(Vec::new());
    let dir = no_weights_dir(
        std::path::Path::new(REPO),
        &|_| Ok(cache.clone()),
        &|url, path| {
            urls.borrow_mut().push(url.to_string());
            std::fs::write(path, fixture("model.safetensors.index.json")).map_err(CliError::from)
        },
    )
    .expect("fetched");
    assert_eq!(
        *urls.borrow(),
        ["https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/resolve/main/model.safetensors.index.json"]
    );
    assert_eq!(
        sha256_hex(
            &std::fs::read_to_string(dir.join("model.safetensors.index.json")).expect("index")
        ),
        "e36c40d4e99b2714fff821218a0433bda2dec46afdb1ebee8ce96ced997928ee"
    );
    let left: Vec<_> = std::fs::read_dir(&dir)
        .expect("ls")
        .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".part"))
        .collect();
    assert!(left.is_empty(), "{left:?}");
}

/// A 404 keeps its own class (exit 11), and names the repo and the file.
#[test]
fn a_repo_without_an_index_is_a_named_not_found() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let cache = tmp.path().join("o").join("r");
    let err = no_weights_dir(
        std::path::Path::new("o/r"),
        &|_| Ok(cache.clone()),
        &|url, _| Err(CliError::HttpNotFound(format!("HTTP 404: {url}"))),
    )
    .expect_err("404");
    assert!(matches!(err, CliError::HttpNotFound(_)), "{err:?}");
    assert!(err.to_string().contains("o/r") && err.to_string().contains("config.json"));
}

/// What is neither an existing path nor `org/repo` is refused by name before any lookup.
#[test]
fn a_path_that_is_neither_a_directory_nor_a_repo_id_is_refused() {
    for arg in [
        "/no/such/dir/at/all",
        "../x/y",
        "justonename",
        "a/b/c",
        "a//b",
        "a/b c",
    ] {
        let err = no_weights_dir(
            std::path::Path::new(arg),
            &|r| panic!("no cache lookup for {r}"),
            &|u, _| panic!("no fetch of {u}"),
        )
        .expect_err(arg);
        assert!(
            matches!(err, CliError::ValidationFailed(_)),
            "{arg}: {err:?}"
        );
        assert!(err.to_string().contains(arg), "{err}");
    }
}

/// `apr inspect ORG/REPO --no-weights --json` reaches the repo-id path through the real CLI
/// (dispatch unchanged): with the network forbidden and nothing cached it fails by name.
#[test]
fn the_cli_takes_a_bare_repo_id_and_fails_by_name_when_offline_and_uncached() {
    use clap::Parser;
    let args: Vec<String> = [
        "apr",
        "inspect",
        "apr-test-7330/never-cached",
        "--no-weights",
        "--json",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    let res = std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(move || {
            let _offline = crate::commands::offline::scope(true);
            let cli = crate::Cli::try_parse_from(args).expect("parse");
            crate::execute_command(&cli)
        })
        .expect("spawn")
        .join()
        .expect("join");
    let err = res.expect_err("uncached and offline");
    assert!(
        err.to_string().contains("apr-test-7330/never-cached"),
        "{err}"
    );
    assert!(err.to_string().contains("not in the cache"), "{err}");
}

#[test]
fn an_unknown_tensor_is_listed_as_unnamed_and_fails_the_run() {
    let r = report(&fixture("config.json"), &fixture("unnamed.index.json")).expect("report");
    assert_eq!(r.unnamed.len(), 1, "{:?}", r.unnamed);
    assert!(r.unnamed[0].ends_with("mystery_block.w"));
    assert!(render_text(&r).contains("UNNAMED"));
    // Round 3 (lane 2 finding 6): the name says the run fails, so assert it here too.
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("config.json"), fixture("config.json")).expect("config");
    std::fs::write(
        dir.path().join("model.safetensors.index.json"),
        fixture("unnamed.index.json"),
    )
    .expect("index");
    let err = run_no_weights(dir.path(), false).expect_err("the run fails");
    assert!(matches!(err, CliError::ValidationFailed(_)), "{err}");
}

#[test]
fn json_carries_families_and_an_empty_unnamed_list() {
    let r = report(
        &fixture("config.json"),
        &fixture("model.safetensors.index.json"),
    )
    .expect("report");
    let v = render_json(&r);
    assert_eq!(v["model_type"], "qwen3_5_moe_text");
    assert_eq!(v["unnamed"].as_array().expect("array").len(), 0);
    assert!(v["families"].as_array().expect("array").len() >= 15);
    assert_eq!(v["weights_read"], false);
}

/// PR #7356 lane-2 finding 3: the run itself, not only the report, fails on an unnamed
/// tensor (the old test above never called `run_no_weights`).
#[test]
fn run_no_weights_fails_on_a_directory_with_an_unnamed_tensor() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("config.json"), fixture("config.json")).expect("config");
    std::fs::write(
        dir.path().join("model.safetensors.index.json"),
        fixture("unnamed.index.json"),
    )
    .expect("index");
    let err = run_no_weights(dir.path(), true).expect_err("an unnamed tensor fails the run");
    assert!(matches!(err, CliError::ValidationFailed(_)), "{err}");
    assert!(
        err.to_string().contains("1 tensor(s) have no family name"),
        "{err}"
    );
}

/// PR #7356 lane-2 finding 2: `--no-weights` with a weight-reading flag is a parse error
/// (contracts/apr-inspect-flags-v1.yaml: never a silent ignore), not a run that drops it.
#[test]
fn no_weights_conflicts_with_every_weight_reading_flag() {
    use clap::Parser;
    for flag in ["--weights", "--vocab", "--filters", "--quality"] {
        let args = vec![
            "apr".to_string(),
            "inspect".to_string(),
            "some/dir".to_string(),
            flag.to_string(),
            "--no-weights".to_string(),
        ];
        let res = std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(move || {
                crate::Cli::try_parse_from(args)
                    .map(|_| ())
                    .map_err(|e| e.kind())
            })
            .expect("spawn")
            .join()
            .expect("join");
        assert_eq!(
            res,
            Err(clap::error::ErrorKind::ArgumentConflict),
            "{flag} --no-weights"
        );
    }
}

#[test]
fn a_real_published_index_has_zero_unnamed() {
    // Verbatim Qwen3.8-Flash-Next index (1658 tensors, hyper-connections, PLE,
    // indexer, MTP, vision): the naming rules are not fitted to the synthetic fixture.
    // From the published apr-cli tarball the sibling crate is absent, so there is nothing to
    // read (PR #7356 lane-2 finding 4); inside the workspace its manifest exists and the
    // fixture must be there (a missing fixture still fails, it is not skipped).
    let crates = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .to_path_buf();
    if !crates.join("aprender-serve/Cargo.toml").exists() {
        eprintln!("skip: aprender-serve is not beside apr-cli (packaged build)");
        return;
    }
    let dir = crates.join("aprender-serve/tests/fixtures/qwen4_exp");
    let cfg = std::fs::read_to_string(dir.join("config.json")).expect("config");
    let idx = std::fs::read_to_string(dir.join("model.safetensors.index.json")).expect("index");
    let r = report(&cfg, &idx).expect("report");
    assert_eq!(r.tensor_count, 1658);
    assert!(r.unnamed.is_empty(), "unnamed: {:?}", r.unnamed);
}

#[test]
fn malformed_inputs_are_errors_not_empty_reports() {
    assert!(report("not json", "{}").is_err());
    assert!(report("{}", "{\"weight_map\": 3}").is_err());
    assert!(
        report("{}", "{}").is_err(),
        "no weight_map is not an empty model"
    );
}

#[test]
fn run_no_weights_reads_a_directory_and_refuses_one_without_an_index() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen38_2_4t");
    assert!(run_no_weights(&dir, true).is_ok());
    let empty = std::env::temp_dir().join(format!("apr-7330-{}", std::process::id()));
    std::fs::create_dir_all(&empty).expect("mkdir");
    assert!(run_no_weights(&empty, true).is_err());
    let _ = std::fs::remove_dir_all(&empty);
}

#[test]
fn the_cli_flag_dispatches_to_the_no_weights_path_on_a_directory() {
    use clap::Parser;
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen38_2_4t");
    let args = vec![
        "apr".to_string(),
        "inspect".to_string(),
        dir.to_string_lossy().into_owned(),
        "--no-weights".to_string(),
        "--json".to_string(),
    ];
    // clap's parser for the full subcommand set needs a bigger stack than a test thread.
    let res = std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(move || {
            let cli = crate::Cli::try_parse_from(args).expect("parse");
            crate::execute_command(&cli)
        })
        .expect("spawn")
        .join()
        .expect("join");
    res.expect("a directory with config + index inspects with no weights");
}
