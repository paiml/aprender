//! Test APR Q4K generation
use realizar::apr_transformer::AprTransformer;
use realizar::gguf::MappedGGUFModel;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Use the CORRECT Q4K APR file with actual Q4K tensors
    let apr_path = "/tmp/qwen2.5-coder-1.5b-q4k.apr";
    let gguf_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    // Load vocab from GGUF
    let mapped = MappedGGUFModel::from_path(gguf_path)?;
    let vocab = mapped.model.vocabulary().ok_or("No vocab")?;

    println!("Loading APR Q4K model from: {}", apr_path);
    let start = Instant::now();
    let transformer = AprTransformer::from_apr_file(apr_path)?;
    println!("Loaded in {:.2}s", start.elapsed().as_secs_f32());

    // Check Q4K layers
    if let Some(ref q4k_layers) = transformer.q4k_layers {
        println!("\nQ4K layers detected: {} layers", q4k_layers.len());
        if let Some(layer0) = q4k_layers.first() {
            println!("  Layer 0 Q4K weights:");
            println!(
                "    ffn_gate: {:?} bytes",
                layer0.ffn_gate_weight.as_ref().map(|v| v.len())
            );
            println!(
                "    ffn_up: {:?} bytes",
                layer0.ffn_up_weight.as_ref().map(|v| v.len())
            );
            println!(
                "    ffn_down: {:?} bytes",
                layer0.ffn_down_weight.as_ref().map(|v| v.len())
            );
            println!(
                "    ffn_down_q6k: {:?} bytes",
                layer0.ffn_down_weight_q6k.as_ref().map(|v| v.len())
            );
        }
    } else {
        println!("\nWARNING: No Q4K layers detected!");
    }

    // Encode prompt
    let prompt = "What is 2+2?";
    let prompt_tokens = mapped.model.encode(prompt).ok_or("Encoding failed")?;
    println!("\nPrompt: {:?}", prompt);
    println!("Tokens: {:?}", prompt_tokens);

    // Greedy generate 10 tokens
    let mut generated = prompt_tokens.clone();
    let start = Instant::now();

    for i in 0..10 {
        let logits = transformer.forward(&generated)?;
        let (argmax_idx, argmax_val) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        generated.push(argmax_idx as u32);

        let tok_str = vocab.get(argmax_idx).map_or("?", |s| s.as_str());
        println!(
            "  Token {}: {} ({:?}) logit={:.4}",
            i, argmax_idx, tok_str, argmax_val
        );

        // Stop on EOS
        if argmax_idx == 151643 || argmax_idx == 151645 {
            println!("  (EOS reached)");
            break;
        }
    }

    let gen_time = start.elapsed();
    let tok_per_sec = (generated.len() - prompt_tokens.len()) as f64 / gen_time.as_secs_f64();

    println!("\nGeneration: {:.1} tok/s", tok_per_sec);

    // Decode output
    let mut output = String::new();
    for &tok in &generated {
        if (tok as usize) < vocab.len() {
            let tok_str = &vocab[tok as usize];
            output.push_str(&tok_str.replace("▁", " ").replace('\u{0120}', " "));
        }
    }

    println!("\nGenerated text:");
    println!("─────────────────────────────────────────");
    println!("{}", output);
    println!("─────────────────────────────────────────");

    Ok(())
}
