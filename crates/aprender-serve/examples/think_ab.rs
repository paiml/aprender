//! think_ab — the `apr qa` thinking-ON golden leg, reproduced with an explicit backend
//! and with the think BODY measured (#3951).
//!
//! The gate (`run_golden_output_gate_runtime`) cannot answer the question #3951 asks:
//!   * its ON leg discards the backend (`let (on_text, _used_gpu) = …`), so a CPU-vs-GPU
//!     A/B cannot be read from it;
//!   * a pass hides how long the think block was, and `<think></think>` with nothing in
//!     it passes — which is how Qwen3.5-0.8B-Q4_K_M looked like a positive control
//!     while not thinking at all (#3948).
//!
//! This builds the ON prompt the way the gate does — `format_messages([user(q)], arch)`
//! with the empty `<think>…</think>` prefill cut off — generates greedily (temperature
//! 0, top_k 1) through `run_inference`, and prints one JSON line.
//!
//!   cargo run --release --features cuda -p aprender-serve --example think_ab -- \
//!       <model.gguf> <budget> <cpu|gpu>

use realizar::chat_template::{format_messages, ChatMessage};
use realizar::gguf::MappedGGUFModel;
use realizar::{run_inference, InferenceConfig};
use std::path::PathBuf;

const QUESTION: &str = "What is 2+2?";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 || !matches!(args[3].as_str(), "cpu" | "gpu") {
        eprintln!("usage: think_ab <model.gguf> <budget> <cpu|gpu>");
        std::process::exit(2);
    }
    let path = PathBuf::from(&args[1]);
    let budget: usize = args[2].parse().unwrap_or_else(|_| {
        eprintln!("budget must be a token count");
        std::process::exit(2)
    });
    let want_gpu = args[3] == "gpu";

    let mapped = MappedGGUFModel::from_path(&path).unwrap_or_else(|e| {
        eprintln!("map failed: {e}");
        std::process::exit(2)
    });
    let arch = mapped.model.architecture().map(String::from);
    let production = format_messages(&[ChatMessage::user(QUESTION)], arch.as_deref())
        .unwrap_or_else(|_| {
            format!("<|im_start|>user\n{QUESTION}<|im_end|>\n<|im_start|>assistant\n")
        });
    // The gate's `without_thinking_prefill`: drop a trailing EMPTY <think></think>.
    let on_prompt = match production.rfind("<think>") {
        Some(i) => production[..i].to_string(),
        None => production.clone(),
    };
    eprintln!("ON_PROMPT={on_prompt:?}");
    let tokens = mapped.model.encode(&on_prompt).unwrap_or_else(|| {
        eprintln!("tokenizer could not encode the prompt");
        std::process::exit(2)
    });
    eprintln!("ON_TOKENS={tokens:?}");
    drop(mapped);

    let mut cfg = InferenceConfig::new(&path)
        .with_input_tokens(tokens)
        .with_max_tokens(budget)
        .with_temperature(0.0)
        .with_top_k(1);
    cfg.no_gpu = !want_gpu;

    let r = run_inference(&cfg).unwrap_or_else(|e| {
        eprintln!("generation failed: {e}");
        std::process::exit(1)
    });
    let gen = r.text.strip_prefix(on_prompt.as_str()).unwrap_or(&r.text);

    let open = gen.find("<think>");
    let close = gen.find("</think>");
    let body = match (open, close) {
        (Some(o), Some(c)) if c > o => gen[o + "<think>".len()..c].trim().to_string(),
        (None, Some(c)) => gen[..c].trim().to_string(), // model began inside the block
        _ => String::new(),
    };
    let answer = close.map(|c| gen[c + "</think>".len()..].trim().to_string());
    let head: String = gen.chars().take(160).collect();
    let tail: String = {
        let v: Vec<char> = gen.chars().collect();
        v[v.len().saturating_sub(160)..].iter().collect()
    };

    let out = serde_json::json!({
        "model": path.file_name().map(|f| f.to_string_lossy().into_owned()),
        "arch": arch,
        "backend_requested": if want_gpu { "gpu" } else { "cpu" },
        "used_gpu": r.used_gpu,
        "gpu_attempted": r.gpu_attempted,
        "fell_back": want_gpu && r.gpu_attempted && !r.used_gpu,
        "budget": budget,
        "generated_tokens": r.generated_token_count,
        "chars": gen.len(),
        "has_open": open.is_some(),
        "closed": close.is_some(),
        "think_body_chars": body.len(),
        "answer_head": answer.map(|a| a.chars().take(120).collect::<String>()),
        "head": head,
        "tail": tail,
    });
    println!("{out}");
}
