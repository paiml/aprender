// FALSIFY-EG2L-007 at the call site: `ChatSession::new` refuses an embedding GGUF on every host,
// before a tokenizer or backend is built. Delete the call in `ChatSession::new` and this goes RED.

#[test]
fn falsify_eg2l_007_chat_session_new_refuses_embedding_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("embed.gguf");
    std::fs::write(
        &path,
        crate::commands::model_kind_gate::tests::eg1_gguf("gemma-embedding2", Some(false)),
    )
    .expect("write");
    let Err(err) = ChatSession::new(&path, true) else {
        panic!("ChatSession::new must refuse an embedding file");
    };
    assert_eq!(err.exit_code_value(), 6, "{err}");
    assert!(err.to_string().contains("apr embed"), "{err}");
}
