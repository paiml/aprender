// #4263: `apr chat`'s row of the engine-identity guard (the run and serve rows
// are realizar's `api::tests_engine_identity`). Every turn of a Qwen3.5 chat
// enters the resident session: one witness entry per turn, on THIS session.

fn model_path() -> String {
    format!(
        "{}/models/Qwen3.5-0.8B-Q4_K_M.gguf",
        std::env::var("HOME").unwrap_or_default()
    )
}

#[test]
fn every_chat_turn_enters_the_one_engine() {
    let model = model_path();
    let path = std::path::Path::new(&model);
    if !path.exists() {
        eprintln!("SKIP: {model} is absent");
        return;
    }
    let mut chat = ChatSession::new(path, true).expect("load the hybrid");
    let id = chat
        .qwen35_session
        .as_ref()
        .expect("a Qwen3.5 chat holds a resident session")
        .id();
    let config = ChatConfig {
        max_tokens: 2,
        temperature: 0.0,
        force_cpu: true,
        ..Default::default()
    };
    for turn in 1..=2 {
        chat.generate("Engine row chat turn.", &config);
        let entries = realizar::session::entries_of_session(id);
        assert_eq!(
            entries.len(),
            turn,
            "turn {turn}: the session saw {entries:?} — apr chat decoded outside the one engine"
        );
        assert!(entries
            .iter()
            .all(|e| e.arch == "qwen35" && e.kind == realizar::session::EntryKind::Generate));
    }
}
