
    // --- `apr chat <local path>` must stay local (dogfood 0.63.0 regression) ---
    //
    // `apr chat /home/noah/models/qwen2.5-coder-0.5b-instruct.apr --offline`
    // printed `Downloading hf:///home...` and died on a 404 because chat
    // rewrote every argument containing a slash to `hf://<arg>` and passed
    // `offline = false` to the resolver.

    /// Unique scratch directory name for one of these tests.
    fn chat_local_scratch(tag: &str) -> String {
        format!(
            "apr-chat-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        )
    }

    /// An absolute path to an existing file resolves to THAT file, not `hf://`.
    #[test]
    fn test_chat_resolves_absolute_local_path_to_itself() {
        let dir = std::env::temp_dir().join(chat_local_scratch("abs"));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let model = dir.join("qwen2.5-coder-0.5b-instruct.apr");
        std::fs::write(&model, b"APR\0not-a-real-model").expect("write fixture");
        assert!(model.is_absolute(), "fixture must be an absolute path");

        let resolved = resolve_chat_model(&model, false)
            .expect("an existing absolute path must resolve without touching the network");
        assert_eq!(
            resolved, model,
            "chat must resolve a local path to itself, not rewrite it to hf://"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Same for a *relative* path containing a slash — a slash is not an hf:// signal.
    #[test]
    fn test_chat_resolves_relative_local_path_with_slash() {
        let cwd = std::env::current_dir().expect("cwd");
        let root = cwd.join(chat_local_scratch("rel"));
        let nested = root.join("models");
        std::fs::create_dir_all(&nested).expect("create scratch dir");
        std::fs::write(nested.join("tiny.gguf"), b"GGUF").expect("write fixture");

        let relative = PathBuf::from(chat_local_scratch("rel"))
            .join("models")
            .join("tiny.gguf");
        assert!(
            relative.to_string_lossy().contains('/'),
            "the point of this test is a slash-bearing relative path"
        );
        assert!(relative.exists(), "relative fixture must exist from cwd");

        let resolved = resolve_chat_model(&relative, false);
        std::fs::remove_dir_all(&root).ok();

        let resolved =
            resolved.expect("an existing relative path must resolve without touching the network");
        assert_eq!(
            resolved, relative,
            "a slash in a local path must not turn it into an hf:// repo id"
        );
    }

    /// `--offline` must stop chat before any network access for an uncached repo.
    ///
    /// If offline is not threaded through, this reaches huggingface.co and fails
    /// with a network/404 error instead of the OFFLINE MODE refusal.
    #[test]
    fn test_chat_offline_refuses_uncached_hf_repo_without_network() {
        let err = resolve_chat_model(
            Path::new("apr-offline-falsifier-org/apr-offline-falsifier-repo"),
            true,
        )
        .expect_err("an uncached hf repo must not resolve in offline mode");
        let msg = err.to_string();
        assert!(
            msg.contains("OFFLINE MODE"),
            "offline chat must refuse locally, got: {msg}"
        );
        assert!(
            !msg.contains("huggingface.co"),
            "offline chat must not have contacted the network, got: {msg}"
        );
    }
