//! SSC Chat Model Evaluation (C-CHAT-TRAIN-002)
//!
//! Loads base Qwen3-4B + trained LoRA adapter and evaluates on test samples.
//! Uses InstructPipeline::from_pretrained with auto-adapter loading (ENT-269).
//!
//! Usage:
//!   cargo run --release --example ssc_eval -- \
//!     --model-dir /tmp/qwen3-4b-ssc-eval \
//!     --data /home/noah/src/bashrs/training/conversations_v3.jsonl \
//!     --samples 50

use entrenar::finetune::{GenerateConfig, InstructConfig, InstructPipeline};
use entrenar::transformer::TransformerConfig;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let model_dir = get_arg(&args, "--model-dir").map(PathBuf::from).expect("--model-dir required");
    let data_path = get_arg(&args, "--data").map(PathBuf::from).expect("--data required");
    let num_samples: usize = get_arg(&args, "--samples").and_then(|s| s.parse().ok()).unwrap_or(50);

    // Load config.json from model directory
    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .unwrap_or_else(|e| panic!("Cannot read {}: {e}", config_path.display()));
    let json: serde_json::Value = serde_json::from_str(&config_str).expect("Invalid config.json");

    let model_config = TransformerConfig {
        hidden_size: json["hidden_size"].as_u64().unwrap_or(2560) as usize,
        num_hidden_layers: json["num_hidden_layers"].as_u64().unwrap_or(36) as usize,
        num_attention_heads: json["num_attention_heads"].as_u64().unwrap_or(32) as usize,
        num_kv_heads: json["num_key_value_heads"].as_u64().unwrap_or(8) as usize,
        intermediate_size: json["intermediate_size"].as_u64().unwrap_or(9728) as usize,
        vocab_size: json["vocab_size"].as_u64().unwrap_or(151936) as usize,
        max_position_embeddings: json["max_position_embeddings"].as_u64().unwrap_or(512) as usize,
        rms_norm_eps: json["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
        rope_theta: json["rope_theta"].as_f64().unwrap_or(1_000_000.0) as f32,
        use_bias: json["use_bias"].as_bool().unwrap_or(false),
        head_dim_override: json["head_dim"].as_u64().map(|v| v as usize),
        architecture: Default::default(),
        hf_architecture: json["architectures"]
            .as_array()
            .and_then(|a| a.first())
            .and_then(|v| v.as_str())
            .map(String::from),
        hf_model_type: json["model_type"].as_str().map(String::from),
        tie_word_embeddings: json["tie_word_embeddings"].as_bool().unwrap_or(false),
    };

    let instruct_config = InstructConfig {
        lora_rank: 16,
        lora_alpha: 32.0,
        max_seq_len: 512,
        ..InstructConfig::default()
    };

    println!("=== SSC Chat Model Evaluation (C-CHAT-TRAIN-002) ===");
    println!("Model dir: {}", model_dir.display());
    println!("Data: {}", data_path.display());
    println!("Samples: {num_samples}");
    println!();

    // Load model + adapter
    println!("Loading model...");
    let pipeline = InstructPipeline::from_pretrained(&model_dir, &model_config, instruct_config)
        .expect("Failed to load model");
    println!("Model loaded.");
    println!();

    // Load test data
    let data = std::fs::read_to_string(&data_path).expect("Cannot read data file");
    let lines: Vec<&str> = data.lines().collect();
    println!("Total entries: {}", lines.len());

    // Sample deterministically (every Nth entry)
    let step = lines.len() / num_samples;
    let samples: Vec<usize> = (0..num_samples).map(|i| i * step).collect();

    let gen_config = GenerateConfig {
        max_new_tokens: 60, // Need enough tokens for format + classification
        temperature: 0.0,   // Greedy for deterministic eval
        top_k: 1,
        stop_tokens: Vec::new(),
    };

    let system_prompt = "You are a shell script safety analyzer. You classify scripts as safe or unsafe and explain your reasoning.\n\nIMPORTANT: Always begin your response with exactly one of these on the first line:\n  Classification: safe\n  Classification: unsafe\n\nThen provide your analysis.";

    // Debug: generate a simple completion to check model output
    println!("=== Debug: simple completion test ===");
    let debug_config =
        GenerateConfig { max_new_tokens: 10, temperature: 0.0, top_k: 1, stop_tokens: Vec::new() };
    match pipeline.generate("Hello, my name is", &debug_config) {
        Ok(text) => {
            println!("Prompt: 'Hello, my name is'");
            println!("Response: '{text}'");
            println!("Response bytes: {:?}", text.as_bytes());
        }
        Err(e) => println!("Error: {e}"),
    }
    println!();

    let mut correct = 0usize;
    let mut total = 0usize;
    let mut safe_correct = 0usize;
    let mut safe_total = 0usize;
    let mut unsafe_correct = 0usize;
    let mut unsafe_total = 0usize;

    for (i, &idx) in samples.iter().enumerate() {
        let line = lines[idx];
        let entry: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let instruction = entry["instruction"].as_str().unwrap_or("");
        let expected_response = entry["response"].as_str().unwrap_or("");

        // Extract expected classification
        let expected_safe = expected_response.contains("Classification: safe")
            && !expected_response.contains("Classification: unsafe");
        let expected_unsafe = expected_response.contains("Classification: unsafe");

        if !expected_safe && !expected_unsafe {
            continue; // Skip entries without clear classification
        }

        total += 1;
        if expected_safe {
            safe_total += 1;
        } else {
            unsafe_total += 1;
        }

        // Run inference
        let start = std::time::Instant::now();
        let response = match pipeline.generate_chat(system_prompt, instruction, &gen_config) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[{}/{}] ERROR: {e}", i + 1, num_samples);
                continue;
            }
        };
        let elapsed = start.elapsed();

        // Extract predicted classification (strict format match)
        let predicted_safe = response.contains("Classification: safe")
            && !response.contains("Classification: unsafe");
        let predicted_unsafe = response.contains("Classification: unsafe");

        // Lenient fallback: look for "unsafe" or "safe" keywords
        let lenient_safe = !predicted_safe
            && !predicted_unsafe
            && response.to_lowercase().contains("safe")
            && !response.to_lowercase().contains("unsafe");
        let lenient_unsafe =
            !predicted_safe && !predicted_unsafe && response.to_lowercase().contains("unsafe");

        let final_safe = predicted_safe || lenient_safe;
        let final_unsafe = predicted_unsafe || lenient_unsafe;

        let strict_correct =
            (expected_safe && predicted_safe) || (expected_unsafe && predicted_unsafe);
        let is_correct = (expected_safe && final_safe) || (expected_unsafe && final_unsafe);
        if strict_correct {
            correct += 1;
            if expected_safe {
                safe_correct += 1;
            } else {
                unsafe_correct += 1;
            }
        }

        let label = if expected_safe { "safe" } else { "unsafe" };
        let pred = if final_safe {
            if predicted_safe {
                "safe"
            } else {
                "safe*"
            }
        } else if final_unsafe {
            if predicted_unsafe {
                "unsafe"
            } else {
                "unsafe*"
            }
        } else {
            "???"
        };
        let mark = if is_correct { "OK" } else { "WRONG" };

        println!(
            "[{:3}/{num_samples}] {mark:5} expected={label:6} predicted={pred:6} ({:.1}s) idx={idx}",
            i + 1,
            elapsed.as_secs_f64()
        );

        // Print response for debugging (first 200 chars + bytes)
        let snippet: String = response.chars().take(200).collect();
        println!("         Response[{}B]: '{snippet}'", response.len());
        if response.len() < 50 {
            println!("         Bytes: {:?}", response.as_bytes());
        }
    }

    println!();
    println!("=== Results ===");
    println!("Total evaluated: {total}");
    println!(
        "Strict correct: {correct}/{total} ({:.1}%)",
        100.0 * correct as f64 / total.max(1) as f64
    );
    println!(
        "Safe accuracy:   {safe_correct}/{safe_total} ({:.1}%)",
        100.0 * safe_correct as f64 / safe_total.max(1) as f64
    );
    println!(
        "Unsafe accuracy: {unsafe_correct}/{unsafe_total} ({:.1}%)",
        100.0 * unsafe_correct as f64 / unsafe_total.max(1) as f64
    );

    // Compute MCC
    let tp = unsafe_correct as f64;
    let tn = safe_correct as f64;
    let fp = (safe_total - safe_correct) as f64;
    let f_n = (unsafe_total - unsafe_correct) as f64;
    let mcc_num = tp * tn - fp * f_n;
    let mcc_den = ((tp + fp) * (tp + f_n) * (tn + fp) * (tn + f_n)).sqrt();
    let mcc = if mcc_den > 0.0 { mcc_num / mcc_den } else { 0.0 };

    println!("MCC: {mcc:.3}");
    println!();

    let accuracy = 100.0 * correct as f64 / total.max(1) as f64;
    if accuracy >= 85.0 {
        println!("PASS: C-CHAT-TRAIN-002 (>= 85% accuracy)");
    } else if accuracy < 50.0 {
        println!("KILL: KILL-QLORA-001 (< 50% accuracy) — model inadequate");
    } else {
        println!("MARGINAL: {accuracy:.1}% accuracy (need >= 85% for PASS, < 50% for KILL)");
    }
}

fn get_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned()
}
