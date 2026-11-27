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

fn open(cache: &std::path::Path) -> ModelFetcher {
    ModelFetcher::with_cache_dir(cache.to_path_buf(), FetchConfig::default()).expect("fetcher")
}

/// Each CLI command is its own process, so every step reopens the store from disk.
#[test]
fn store_survives_reopen_and_keeps_every_file_of_every_repo() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cache = dir.path().join("models");
    std::fs::create_dir_all(&cache).expect("mkdir");

    let files = [("org/repo", "a.gguf"), ("org/repo", "b.gguf"), ("other/model", "c.gguf")];
    for (name, file) in files {
        let path = cache.join(file);
        std::fs::write(&path, b"GGUF-test-bytes").expect("write model");
        open(&cache).record_model(name, &path).expect("record");
    }

    let mut paths: Vec<_> = open(&cache).list().into_iter().map(|m| m.path).collect();
    paths.sort();
    let want: Vec<_> = ["a.gguf", "b.gguf", "c.gguf"].iter().map(|f| cache.join(f)).collect();
    assert_eq!(paths, want, "a later record must not drop an earlier entry");

    assert!(open(&cache).remove("hf://org/repo").expect("rm"), "rm must find the stored name");
    assert!(!cache.join("a.gguf").exists(), "rm must delete the first file of the repo");
    assert!(!cache.join("b.gguf").exists(), "rm must delete the second file of the repo");

    let names: Vec<String> = open(&cache).list().into_iter().map(|m| m.name).collect();
    assert_eq!(names, vec!["other/model".to_string()], "rm must persist and spare other repos");
}

/// `rm` and `list` need every name: the size cap must not drop entries from the manifest
/// while the files stay on disk (aprender#7481).
#[test]
fn store_keeps_every_name_past_the_cache_size_cap() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cache = dir.path().join("models");
    std::fs::create_dir_all(&cache).expect("mkdir");
    // Each file is 15 bytes, so the second one already passes 90% of this cap.
    let capped = || {
        let mut config = FetchConfig::default();
        config.cache = config.cache.with_max_size_bytes(16);
        ModelFetcher::with_cache_dir(cache.clone(), config).expect("fetcher")
    };

    for (name, file) in [("org/one", "a.gguf"), ("org/two", "b.gguf"), ("org/three", "c.gguf")] {
        let path = cache.join(file);
        std::fs::write(&path, b"GGUF-test-bytes").expect("write model");
        capped().record_model(name, &path).expect("record");
    }

    let mut names: Vec<String> = capped().list().into_iter().map(|m| m.name).collect();
    names.sort();
    assert_eq!(names, vec!["org/one", "org/three", "org/two"]);
}
