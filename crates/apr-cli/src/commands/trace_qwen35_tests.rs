//! #4270: `apr trace` on a Qwen3.5 hybrid generates through the one engine.

use super::*;

const MODEL: &str = "/home/noah/models/Qwen3.5-0.8B-Q4_K_M.gguf";

/// Before #4270 the GGUF trace built a dense `OwnedQuantizedModel` first, which
/// refuses the hybrid, so `apr trace` on any Qwen3.5 model was an error.
#[test]
fn trace_gguf_accepts_qwen35() {
    if !Path::new(MODEL).exists() {
        eprintln!("SKIP: {MODEL} is absent");
        return;
    }
    run_traced_inference_gguf(Path::new(MODEL)).expect("apr trace runs on qwen35");
}

/// The trace's generation is a `Session::generate` entry for its prompt.
#[test]
fn trace_qwen35_generates_through_session() {
    if !Path::new(MODEL).exists() {
        eprintln!("SKIP: {MODEL} is absent");
        return;
    }
    let mapped = realizar::gguf::MappedGGUFModel::from_path(MODEL).expect("map");
    let prompt = "trace_qwen35 witness prompt #4270";
    run_traced_inference_qwen35(&mapped, prompt).expect("trace runs");
    let tokens = mapped.model.encode(prompt).expect("tokenizer");
    let entries = realizar::session::entries_for(&tokens);
    assert!(
        entries
            .iter()
            .any(|e| e.arch == "qwen35" && e.kind == realizar::session::EntryKind::Generate),
        "no qwen35 Generate entry: {entries:?}"
    );
}
