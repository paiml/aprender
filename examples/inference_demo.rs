//! End-to-end LLM inference demo using trueno's sovereign compute stack.
//!
//! Usage:
//!   cargo run --example inference_demo --release -- <model.gguf> [prompt]
//!
//! Example:
//!   cargo run --example inference_demo --release -- \
//!     tiny-llama-1.1B-Q4_K_M.gguf "The capital of France is"
//!
//! The model must be a GGUF file with Q4_K quantized weights (llama.cpp compatible).

use std::path::Path;
use std::time::Instant;
use trueno::inference::{generate, GgufFile, LlamaModel, SampleParams};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf> [prompt] [--temp T] [--max-tokens N]", args[0]);
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --temp T         Temperature (default: 0.7, 0=greedy)");
        eprintln!("  --max-tokens N   Max tokens to generate (default: 128)");
        eprintln!("  --top-k K        Top-K sampling (default: 40)");
        eprintln!("  --top-p P        Top-P nucleus sampling (default: 0.9)");
        std::process::exit(1);
    }

    let model_path = &args[1];
    let mut prompt = "The".to_string();
    let mut params = SampleParams::default();
    let mut max_tokens = 128usize;

    // Parse args
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--temp" => {
                i += 1;
                params.temperature = args[i].parse().expect("invalid temperature");
            }
            "--max-tokens" => {
                i += 1;
                max_tokens = args[i].parse().expect("invalid max-tokens");
            }
            "--top-k" => {
                i += 1;
                params.top_k = args[i].parse().expect("invalid top-k");
            }
            "--top-p" => {
                i += 1;
                params.top_p = args[i].parse().expect("invalid top-p");
            }
            s if !s.starts_with("--") => {
                prompt = s.to_string();
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    // Load GGUF
    eprintln!("Loading model: {model_path}");
    let load_start = Instant::now();
    let gguf = GgufFile::load(Path::new(model_path)).expect("Failed to load GGUF");
    eprintln!(
        "  GGUF: {} tensors, loaded in {:.1}s",
        gguf.tensor_count,
        load_start.elapsed().as_secs_f64()
    );

    // Print model info
    if let Some(arch) = gguf.meta_str("general.architecture") {
        eprintln!("  Architecture: {arch}");
    }
    if let Some(name) = gguf.meta_str("general.name") {
        eprintln!("  Name: {name}");
    }

    // Build model
    let build_start = Instant::now();
    let model = LlamaModel::from_gguf(&gguf).expect("Failed to build model");
    eprintln!("  Model ready in {:.1}s", build_start.elapsed().as_secs_f64());

    // Tokenize prompt (simple: use GGUF vocab if available, else byte-level)
    let prompt_tokens = tokenize_simple(&gguf, &prompt);
    eprintln!("  Prompt: {:?} ({} tokens)", prompt, prompt_tokens.len());

    // Detect EOS token
    let eos_token = gguf.meta_u32("tokenizer.ggml.eos_token_id").unwrap_or(2);

    // Generate
    eprintln!("Generating ({max_tokens} tokens max, temp={:.1})...", params.temperature);
    eprintln!();

    let gen_start = Instant::now();
    let generated = generate(&model, &prompt_tokens, max_tokens, &params, eos_token)
        .expect("Generation failed");
    let gen_elapsed = gen_start.elapsed();

    // Decode and print
    let output_text = detokenize_simple(&gguf, &generated);
    let prompt_text_decoded = detokenize_simple(&gguf, &prompt_tokens);
    print!("{prompt_text_decoded}{output_text}");
    println!();

    // Stats
    let total_tokens = prompt_tokens.len() + generated.len();
    let tok_per_sec = generated.len() as f64 / gen_elapsed.as_secs_f64();
    eprintln!();
    eprintln!(
        "  Generated {} tokens in {:.2}s ({:.1} tok/s)",
        generated.len(),
        gen_elapsed.as_secs_f64(),
        tok_per_sec,
    );
    eprintln!("  Prefill: {} tokens, Decode: {} tokens", prompt_tokens.len(), generated.len());
    eprintln!("  Total: {total_tokens} tokens");
}

/// Simple tokenizer using GGUF vocab (token list).
/// Falls back to byte-level tokenization if vocab not found.
fn tokenize_simple(gguf: &GgufFile, text: &str) -> Vec<u32> {
    // Try to get vocab from GGUF metadata
    if let Some(trueno::inference::gguf::MetadataValue::Array(tokens)) =
        gguf.metadata.get("tokenizer.ggml.tokens")
    {
        // Build a simple greedy longest-match tokenizer
        let vocab: Vec<String> =
            tokens.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect();

        if !vocab.is_empty() {
            return greedy_tokenize(text, &vocab);
        }
    }

    // Fallback: byte-level (each byte = token ID)
    text.bytes().map(|b| b as u32).collect()
}

/// Greedy longest-match tokenization.
/// SentencePiece models encode spaces as the ▁ character (U+2581).
/// We normalise the input by replacing ASCII spaces with ▁ before matching.
fn greedy_tokenize(text: &str, vocab: &[String]) -> Vec<u32> {
    // SentencePiece space normalisation: replace ' ' with '▁' (U+2581, 3 UTF-8 bytes: E2 96 81)
    let normalised = text.replace(' ', "\u{2581}");
    // Also try the original text (tiktoken / byte-level models don't use ▁)
    let uses_sentencepiece = vocab.iter().any(|t| t.contains('\u{2581}'));
    let to_tokenize = if uses_sentencepiece { normalised.as_str() } else { text };

    let mut tokens = Vec::new();
    let bytes = to_tokenize.as_bytes();
    let mut pos = 0;

    while pos < bytes.len() {
        let mut best_len = 0;
        let mut best_id = 0u32;

        for (id, token) in vocab.iter().enumerate() {
            let tok_bytes = token.as_bytes();
            if tok_bytes.len() > best_len
                && pos + tok_bytes.len() <= bytes.len()
                && &bytes[pos..pos + tok_bytes.len()] == tok_bytes
            {
                best_len = tok_bytes.len();
                best_id = id as u32;
            }
        }

        if best_len == 0 {
            pos += 1; // skip unmatched byte
        } else {
            tokens.push(best_id);
            pos += best_len;
        }
    }

    // Prepend BOS (token 1 for llama/SentencePiece; Qwen2 uses 151643)
    if !tokens.is_empty() {
        let bos = 1u32;
        tokens.insert(0, bos);
    }

    tokens
}

/// Simple detokenizer using GGUF vocab.
fn detokenize_simple(gguf: &GgufFile, token_ids: &[u32]) -> String {
    if let Some(trueno::inference::gguf::MetadataValue::Array(tokens)) =
        gguf.metadata.get("tokenizer.ggml.tokens")
    {
        let vocab: Vec<String> =
            tokens.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect();

        if !vocab.is_empty() {
            return token_ids
                .iter()
                .filter_map(|&id| vocab.get(id as usize))
                .map(|s| s.replace('\u{2581}', " "))
                .collect();
        }
    }

    // Fallback: bytes
    String::from_utf8_lossy(&token_ids.iter().map(|&id| id as u8).collect::<Vec<u8>>()).to_string()
}
