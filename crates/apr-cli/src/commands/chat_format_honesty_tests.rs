// #3022 / #3024 — `apr chat` must never substitute its built-in demo model for a model
// file the user named. Both polarities, because the defect was invisible in exactly one
// direction: every "does a .gguf resolve to Gguf?" test passed while the bug shipped.
//
// The mutation this table is written against: delete the `.safetensors.index.json` arm
// from `detect_format` and `sharded_index_resolves_to_the_sharded_format` goes RED; make
// `resolve_chat_format` fall back to `Ok(ModelFormat::Demo)` instead of refusing and
// `an_existing_unrecognised_file_is_refused_not_demoted` goes RED.

/// A file on disk with the given name and leading bytes. `apr chat` only ever resolves
/// paths that exist (`resolve_chat_model` proves it first), so a table of non-existent
/// paths would test a state the command cannot be in.
fn touch(dir: &std::path::Path, name: &str, bytes: &[u8]) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, bytes).expect("fixture write");
    p
}

#[test]
fn sharded_index_resolves_to_the_sharded_format_not_demo() {
    let td = tempfile::tempdir().expect("tempdir");
    let p = touch(
        td.path(),
        "model.safetensors.index.json",
        br#"{"metadata":{"total_size":988065536},"weight_map":{"a":"model-00001-of-00002.safetensors"}}"#,
    );
    // Path::extension() is Some("json") here — the last dot-segment — which is exactly
    // why the pre-#3022 `match` fell through to Demo.
    assert_eq!(p.extension().and_then(|e| e.to_str()), Some("json"));
    assert_eq!(detect_format(&p), ModelFormat::ShardedSafeTensors);
    assert_eq!(
        resolve_chat_format(&p).expect("a sharded index is loadable"),
        ModelFormat::ShardedSafeTensors
    );
}

#[test]
fn the_three_single_file_formats_still_resolve_by_name() {
    let td = tempfile::tempdir().expect("tempdir");
    for (name, want) in [
        ("model.apr", ModelFormat::Apr),
        ("model.gguf", ModelFormat::Gguf),
        ("model.safetensors", ModelFormat::SafeTensors),
    ] {
        let p = touch(td.path(), name, b"\x00\x00\x00\x00\x00\x00\x00\x00");
        assert_eq!(detect_format(&p), want, "{name}");
        assert_eq!(resolve_chat_format(&p).expect("name resolves"), want, "{name}");
    }
}

#[test]
fn an_odd_extension_over_real_magic_bytes_is_still_a_model() {
    // A model is a model whatever it is called: the bytes decide when the name does not.
    // Without this, the refusal below would reject `qwen.bin` — a real GGUF — and the fix
    // for a silent substitution would become a false refusal.
    let td = tempfile::tempdir().expect("tempdir");
    let p = touch(td.path(), "qwen.bin", b"GGUF\x03\x00\x00\x00rest");
    assert_eq!(detect_format(&p), ModelFormat::Demo, "the NAME says nothing");
    assert_eq!(
        resolve_chat_format(&p).expect("the bytes say GGUF"),
        ModelFormat::Gguf
    );
}

#[test]
fn an_existing_unrecognised_file_is_refused_not_demoted() {
    let td = tempfile::tempdir().expect("tempdir");
    let p = touch(td.path(), "notes.txt", b"this is not a model at all, it is prose\n");
    let err = resolve_chat_format(&p).expect_err("prose is not a model");
    // The code is READ from error.rs, never typed here (D-7).
    assert_eq!(
        err.exit_code_value(),
        CliError::ModelLoadFailed(String::new()).exit_code_value(),
        "the refusal must carry error.rs's model-load code, not a bespoke one"
    );
    let msg = err.to_string();
    assert!(msg.contains("notes.txt"), "the refusal names the file: {msg}");
    assert!(
        msg.contains("#3022"),
        "the refusal cites the defect it exists to prevent: {msg}"
    );
}

#[test]
fn resolve_never_answers_demo_for_a_file_the_user_named() {
    // The general form of #3022 (#3024 ask 4). Whatever the input, a path that exists
    // either resolves to a real format or is refused — `Demo` is not an outcome.
    let td = tempfile::tempdir().expect("tempdir");
    let cases: [(&str, &[u8]); 6] = [
        ("model.safetensors.index.json", b"{\"weight_map\":{}}"),
        ("model.gguf", b"GGUF\x03\x00\x00\x00"),
        ("model.apr", b"APR2\x00\x00\x00\x00"),
        ("model.safetensors", b"\x40\x00\x00\x00\x00\x00\x00\x00"),
        ("weights.bin", b"GGUF\x03\x00\x00\x00"),
        ("readme.md", b"# not a model\n"),
    ];
    for (name, bytes) in cases {
        let p = touch(td.path(), name, bytes);
        match resolve_chat_format(&p) {
            Ok(ModelFormat::Demo) => panic!("{name} resolved to the demo model (#3022)"),
            Ok(_) | Err(_) => {}
        }
    }
}

#[test]
fn a_truncated_file_is_refused_rather_than_read_as_a_format() {
    // `detect_format_from_bytes` returns Demo for fewer than 8 bytes; that must reach the
    // user as a refusal, not as a demo session that prints their path.
    let td = tempfile::tempdir().expect("tempdir");
    let p = touch(td.path(), "truncated.model", b"GG");
    assert!(resolve_chat_format(&p).is_err());
}
