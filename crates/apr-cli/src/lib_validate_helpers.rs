
    /// B1: every `--step` name in `SHOWCASE_STEPS` resolves to its own step.
    #[test]
    fn test_parse_showcase_step_known_names_round_trip() {
        assert!(!SHOWCASE_STEPS.is_empty());
        for (name, expected) in SHOWCASE_STEPS {
            let got = parse_showcase_step(name).expect("known step must parse");
            assert_eq!(got, *expected, "step '{name}' resolved to the wrong variant");
        }
    }

    /// B1: an unknown `--step` is rejected, names the input and lists every valid name.
    #[test]
    fn test_parse_showcase_step_unknown_names_input_and_lists_all() {
        let err = parse_showcase_step("bogus").expect_err("unknown step must fail");
        let CliError::ValidationFailed(msg) = err else {
            panic!("expected ValidationFailed, got {err:?}");
        };
        assert!(msg.contains("unknown step 'bogus'"), "msg was: {msg}");
        for (name, _) in SHOWCASE_STEPS {
            assert!(msg.contains(name), "msg must list '{name}': {msg}");
        }
        // Matching is exact: case and whitespace variants are unknown too.
        assert!(parse_showcase_step("Import").is_err());
        assert!(parse_showcase_step("").is_err());
    }

    fn write_manifest(dir: &Path, files: &[(&str, u64)]) -> PathBuf {
        let map: serde_json::Map<String, serde_json::Value> = files
            .iter()
            .map(|(n, s)| {
                (
                    (*n).to_string(),
                    serde_json::json!({"size": s, "blake3": "00"}),
                )
            })
            .collect();
        let manifest = serde_json::json!({"version": 1, "repo": "org/repo", "files": map});
        let p = dir.join(".apr-manifest.json");
        std::fs::write(&p, manifest.to_string()).expect("write manifest");
        p
    }

    fn shard_err(r: Result<(), CliError>) -> String {
        match r {
            Err(CliError::ValidationFailed(m)) => m,
            other => panic!("expected ValidationFailed, got {other:?}"),
        }
    }

    /// B2: a manifest file that does not exist is refused.
    #[test]
    fn test_validate_shard_manifest_missing_manifest() {
        let dir = tempfile::tempdir().expect("tempdir");
        let m = shard_err(validate_shard_manifest(
            &dir.path().join(".apr-manifest.json"),
            dir.path(),
        ));
        assert!(m.contains("Failed to read manifest"), "msg: {m}");
    }

    /// B2: malformed manifest JSON is refused.
    #[test]
    fn test_validate_shard_manifest_bad_json() {
        let dir = tempfile::tempdir().expect("tempdir");
        let p = dir.path().join(".apr-manifest.json");
        std::fs::write(&p, "{not json").expect("write");
        let m = shard_err(validate_shard_manifest(&p, dir.path()));
        assert!(m.contains("Failed to parse manifest"), "msg: {m}");
    }

    /// B2: a shard listed in the manifest but absent on disk is refused.
    #[test]
    fn test_validate_shard_manifest_missing_shard() {
        let dir = tempfile::tempdir().expect("tempdir");
        let p = write_manifest(dir.path(), &[("model-00001.safetensors", 4)]);
        let m = shard_err(validate_shard_manifest(&p, dir.path()));
        assert!(m.contains("model-00001.safetensors"), "msg: {m}");
        assert!(m.contains("is missing"), "msg: {m}");
    }

    /// B2: a truncated shard (size differs from the manifest) is refused.
    #[test]
    fn test_validate_shard_manifest_size_mismatch() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("a.safetensors"), [0u8; 3]).expect("write");
        let p = write_manifest(dir.path(), &[("a.safetensors", 8)]);
        let m = shard_err(validate_shard_manifest(&p, dir.path()));
        assert!(m.contains("size mismatch"), "msg: {m}");
        assert!(m.contains("expected 8 bytes, got 3 bytes"), "msg: {m}");
    }

    /// B2: matching sizes pass.
    #[test]
    fn test_validate_shard_manifest_all_good() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("a.safetensors"), [0u8; 8]).expect("write");
        std::fs::write(dir.path().join("b.safetensors"), [0u8; 2]).expect("write");
        let p = write_manifest(dir.path(), &[("a.safetensors", 8), ("b.safetensors", 2)]);
        assert!(validate_shard_manifest(&p, dir.path()).is_ok());
    }

    /// B3: a path with no parent has no manifest to consult and passes.
    #[test]
    fn test_validate_shard_index_no_parent_ok() {
        assert!(validate_shard_index(Path::new("")).is_ok());
    }

    /// B3: a directory without `.apr-manifest.json` is not checked and passes.
    #[test]
    fn test_validate_shard_index_no_manifest_ok() {
        let dir = tempfile::tempdir().expect("tempdir");
        let idx = dir.path().join("model.safetensors.index.json");
        assert!(validate_shard_index(&idx).is_ok());
    }

    /// B3: a present manifest is delegated to `validate_shard_manifest`.
    #[test]
    fn test_validate_shard_index_delegates_to_manifest() {
        let dir = tempfile::tempdir().expect("tempdir");
        let idx = dir.path().join("model.safetensors.index.json");
        std::fs::write(dir.path().join(".apr-manifest.json"), "{not json").expect("write");
        let m = shard_err(validate_shard_index(&idx));
        assert!(m.contains("Failed to parse manifest"), "msg: {m}");

        write_manifest(dir.path(), &[("s.safetensors", 1)]);
        let m = shard_err(validate_shard_index(&idx));
        assert!(m.contains("is missing"), "msg: {m}");

        std::fs::write(dir.path().join("s.safetensors"), [0u8; 1]).expect("write");
        assert!(validate_shard_index(&idx).is_ok());
    }
