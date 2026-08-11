
    // ════════════════════════════════════════════════════════════════════
    // #2392 finding 4 — export / merge / shard / unshard silently destroyed
    // pre-existing output artifacts and had no --force flag at all, while
    // convert and quantize in the same CLI refused the identical situation
    // with exit 5 and "Use --force to overwrite".
    // ════════════════════════════════════════════════════════════════════

    /// Is this the overwrite refusal (as opposed to some later failure)?
    fn is_overwrite_refusal(e: &CliError) -> bool {
        matches!(e, CliError::ValidationFailed(m) if m.contains("already exists")
            && m.contains("--force"))
    }

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("apr-2392-f4-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir scratch");
        dir
    }

    /// `apr export -o precious.safetensors` overwrote a 9-byte file with
    /// 9717255 bytes of model and exited 0, with no way to opt out.
    #[test]
    fn export_refuses_to_clobber_an_existing_output() {
        let dir = scratch("export");
        let input = dir.join("in.apr");
        std::fs::write(&input, b"APR\0not-a-real-model").expect("write input");
        let out = dir.join("precious.safetensors");
        std::fs::write(&out, b"PRECIOUS\n").expect("write precious");

        let err = commands::export::run(
            Some(&input),
            "safetensors",
            Some(&out),
            None,
            false,
            None,
            true,
            false,
            false, // force
        )
        .expect_err("#2392 finding 4: export must refuse to clobber");
        assert!(
            is_overwrite_refusal(&err),
            "#2392 finding 4: expected the overwrite refusal, got: {err}"
        );
        assert_eq!(
            std::fs::read(&out).expect("precious still readable"),
            b"PRECIOUS\n",
            "#2392 finding 4: the pre-existing file must be untouched"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// …and `--force` must still let the user through. (A guard with no escape
    /// hatch is a different defect.) The export then fails for its own reasons
    /// — the input is deliberately not a real model — but it must not be the
    /// overwrite refusal.
    #[test]
    fn export_force_bypasses_the_overwrite_guard() {
        let dir = scratch("export-force");
        let input = dir.join("in.apr");
        std::fs::write(&input, b"APR\0not-a-real-model").expect("write input");
        let out = dir.join("existing.safetensors");
        std::fs::write(&out, b"OLD\n").expect("write existing");

        let result = commands::export::run(
            Some(&input),
            "safetensors",
            Some(&out),
            None,
            false,
            None,
            true,
            false,
            true, // force
        );
        if let Err(ref e) = result {
            assert!(
                !is_overwrite_refusal(e),
                "#2392 finding 4: --force must not hit the overwrite guard, got: {e}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `apr merge a b -o precious2.apr` did the same thing: a 10-byte file came
    /// back as 9717255 bytes of merged model, rc=0.
    #[test]
    fn merge_refuses_to_clobber_an_existing_output() {
        let dir = scratch("merge");
        let a = dir.join("a.apr");
        let b = dir.join("b.apr");
        std::fs::write(&a, b"APR\0a").expect("write a");
        std::fs::write(&b, b"APR\0b").expect("write b");
        let out = dir.join("precious2.apr");
        std::fs::write(&out, b"PRECIOUS2\n").expect("write precious");

        let err = commands::merge::run(
            &[a, b],
            "average",
            Some(&out),
            None,
            None,
            0.9,
            0.2,
            42,
            true,
            false,
            false, // force
        )
        .expect_err("#2392 finding 4: merge must refuse to clobber");
        assert!(
            is_overwrite_refusal(&err),
            "#2392 finding 4: expected the overwrite refusal, got: {err}"
        );
        assert_eq!(
            std::fs::read(&out).expect("precious still readable"),
            b"PRECIOUS2\n",
            "#2392 finding 4: the pre-existing file must be untouched"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `apr shard -o dir/` wrote a second shard set into a directory that
    /// already held one, replacing the weight-map index that names the shards.
    /// The index is what identifies a shard set, so that is what we guard.
    #[test]
    fn shard_refuses_to_replace_an_existing_shard_set() {
        let dir = scratch("shard");
        let input = dir.join("model.safetensors");
        std::fs::write(&input, b"not-a-real-safetensors").expect("write input");
        let out_dir = dir.join("shards");
        std::fs::create_dir_all(&out_dir).expect("mkdir out");
        let index = out_dir.join("model.safetensors.index.json");
        std::fs::write(&index, b"{\"existing\":true}").expect("write index");

        let err = dispatch_shard(&input, "1MB", &out_dir, false, false)
            .expect_err("#2392 finding 4: shard must refuse to replace a shard set");
        assert!(
            is_overwrite_refusal(&err),
            "#2392 finding 4: expected the overwrite refusal, got: {err}"
        );
        assert_eq!(
            std::fs::read(&index).expect("index still readable"),
            b"{\"existing\":true}",
            "#2392 finding 4: the pre-existing index must be untouched"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Sharding into a fresh directory must NOT be blocked — the guard is about
    /// destroying an existing shard set, not about the directory existing.
    #[test]
    fn shard_into_a_directory_without_an_index_is_not_blocked() {
        let dir = scratch("shard-clean");
        let input = dir.join("model.safetensors");
        std::fs::write(&input, b"not-a-real-safetensors").expect("write input");
        let out_dir = dir.join("shards");
        std::fs::create_dir_all(&out_dir).expect("mkdir out");
        std::fs::write(out_dir.join("unrelated.txt"), b"keep me").expect("write unrelated");

        let result = dispatch_shard(&input, "1MB", &out_dir, false, false);
        if let Err(ref e) = result {
            assert!(
                !is_overwrite_refusal(e),
                "#2392 finding 4: a directory with no shard index must not trip the guard, \
                 got: {e}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `apr unshard -o merged.safetensors` writes a single file and clobbered it
    /// the same way. Leaving one command in the family unguarded would make the
    /// "consistent overwrite policy" claim false.
    #[test]
    fn unshard_refuses_to_clobber_an_existing_output() {
        let dir = scratch("unshard");
        let in_dir = dir.join("shards");
        std::fs::create_dir_all(&in_dir).expect("mkdir in");
        let out = dir.join("merged.safetensors");
        std::fs::write(&out, b"PRECIOUS3\n").expect("write precious");

        let err = dispatch_unshard(&in_dir, &out, false, false)
            .expect_err("#2392 finding 4: unshard must refuse to clobber");
        assert!(
            is_overwrite_refusal(&err),
            "#2392 finding 4: expected the overwrite refusal, got: {err}"
        );
        assert_eq!(
            std::fs::read(&out).expect("precious still readable"),
            b"PRECIOUS3\n",
            "#2392 finding 4: the pre-existing file must be untouched"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The shared helper itself: absent output is always allowed, present
    /// output is refused unless forced. This is the single decision point every
    /// write-a-file command now routes through.
    #[test]
    fn refuse_overwrite_decision_table() {
        let dir = scratch("helper");
        let missing = dir.join("missing.apr");
        let present = dir.join("present.apr");
        std::fs::write(&present, b"x").expect("write present");

        assert!(crate::error::refuse_overwrite(&missing, false).is_ok());
        assert!(crate::error::refuse_overwrite(&missing, true).is_ok());
        assert!(crate::error::refuse_overwrite(&present, true).is_ok());
        let err = crate::error::refuse_overwrite(&present, false)
            .expect_err("existing output without --force must be refused");
        assert!(is_overwrite_refusal(&err), "got: {err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
