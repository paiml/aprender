//! aprender#7481 (U-02): a model downloaded by `apr run` or `apr pull hf://org/repo/file`
//! lives in the one pacha model store under its repo name, so `list` shows the name and
//! `rm` removes it by the same name. RED first: `record_model` does not exist yet.

use pacha::fetcher::{FetchConfig, ModelFetcher};

#[test]
fn hf_download_is_listed_and_removed_by_repo_name() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cache = dir.path().join("models");
    std::fs::create_dir_all(&cache).expect("mkdir");

    // The bytes as `apr run` / `apr pull` would leave them.
    let path = cache.join("3f9a1c0e7b2d4a58.gguf");
    std::fs::write(&path, b"GGUF-test-bytes").expect("write model");

    let mut fetcher =
        ModelFetcher::with_cache_dir(cache.clone(), FetchConfig::default()).expect("fetcher");
    fetcher.record_model("org/repo", &path).expect("record the download under its repo name");

    let names: Vec<String> = fetcher.list().into_iter().map(|m| m.name).collect();
    assert!(names.iter().any(|n| n == "org/repo"), "list must show the repo name, got {names:?}");

    assert!(fetcher.remove("org/repo").expect("rm by repo name"), "rm must find the repo name");
    assert!(!path.exists(), "rm must delete the model file");
    assert!(
        !fetcher.list().iter().any(|m| m.name == "org/repo"),
        "list must no longer show the repo after rm"
    );
}
