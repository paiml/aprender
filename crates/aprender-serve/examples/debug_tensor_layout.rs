//! Debug script to compare tensor layouts between TinyLlama and Qwen2
//! Run with: cd /home/noah/src/realizar && cargo run --release --example debug_tensor_layout
//!
//! Five-Whys Analysis:
//! Why-1: Qwen2 produces "!" instead of correct output
//! Why-2: Token 0 consistently has highest logit
//! Why-3: LM head projection produces wrong values
//! Why-4: Tensor layout may differ between models
//! Why-5: Tied embedding weights may need transposition for LM head

use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Tensor Layout Debug ===\n");

    // Qwen2-0.5B model (Q4_0 format)
    let qwen2_path = PathBuf::from("/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf");

    // Qwen2.5 Coder model
    let qwen25_path = PathBuf::from(
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-0.5b-instruct-q4_0.gguf",
    );

    // DeepSeek model (for comparison - different architecture)
    let deepseek_path = PathBuf::from(
        "/home/noah/src/single-shot-eval/models/raw/deepseek-coder-1.3b-instruct-q4_k_m.gguf",
    );

    // Check if models exist and analyze
    for (name, path) in [
        ("Qwen2-0.5B", &qwen2_path),
        ("Qwen2.5-Coder-0.5B", &qwen25_path),
        ("DeepSeek-Coder-1.3B", &deepseek_path),
    ] {
        if path.exists() {
            println!("{} model: {}", name, path.display());
            analyze_model(path)?;
            println!();
        } else {
            println!("{} model not found at {}\n", name, path.display());
        }
    }

    Ok(())
}

fn analyze_model(path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    use realizar::gguf::{GGUFConfig, GGUFModel};

    // Read file
    let mut file = File::open(path)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;

    // Parse GGUF
    let model = GGUFModel::from_bytes(&data)?;
    let config = GGUFConfig::from_gguf(&model)?;

    println!("  Architecture: {}", config.architecture);
    println!("  Hidden dim: {}", config.hidden_dim);
    println!("  Vocab size: {}", config.vocab_size);
    println!("  Num layers: {}", config.num_layers);
    println!("  Num heads: {}/{}", config.num_heads, config.num_kv_heads);

    println!("\n  Key tensors:");

    // Find key tensors
    for tensor in &model.tensors {
        if tensor.name == "token_embd.weight"
            || tensor.name == "output.weight"
            || tensor.name.contains("blk.0.attn_q")
            || tensor.name.contains("blk.0.attn_k")
            || tensor.name.contains("blk.0.attn_v")
        {
            let qtype_name = match tensor.qtype {
                0 => "F32",
                1 => "F16",
                2 => "Q4_0",
                3 => "Q4_1",
                6 => "Q5_0",
                7 => "Q5_1",
                8 => "Q8_0",
                12 => "Q4_K",
                13 => "Q5_K",
                14 => "Q6_K",
                _ => "???",
            };
            let num_elements: u64 = tensor.dims.iter().product();
            println!(
                "    {}: dims={:?}, qtype={}, elements={}",
                tensor.name, tensor.dims, qtype_name, num_elements
            );

            // Check if this looks like [hidden, vocab] or [vocab, hidden]
            if (tensor.name == "token_embd.weight" || tensor.name == "output.weight")
                && tensor.dims.len() == 2
            {
                let (d0, d1) = (tensor.dims[0] as usize, tensor.dims[1] as usize);
                if d0 == config.hidden_dim {
                    println!(
                        "      -> Layout: [hidden_dim, vocab_size] = [{}×{}]",
                        d0, d1
                    );
                    println!("      -> This suggests TRANSPOSED for embedding lookup!");
                    println!("      -> Row 0 has {} elements (vocab tokens)", d1);
                } else if d1 == config.hidden_dim {
                    println!(
                        "      -> Layout: [vocab_size, hidden_dim] = [{}×{}]",
                        d0, d1
                    );
                    println!("      -> This is CORRECT for embedding lookup");
                    println!("      -> Row 0 has {} elements (hidden state)", d1);
                } else if d0 == config.vocab_size {
                    println!("      -> Layout: [vocab_size, ?] = [{}×{}]", d0, d1);
                } else if d1 == config.vocab_size {
                    println!("      -> Layout: [?, vocab_size] = [{}×{}]", d0, d1);
                }
            }
        }
    }

    // Test embedding lookup
    println!("\n  Embedding lookup test:");
    let token_embd = model.tensors.iter().find(|t| t.name == "token_embd.weight");
    if let Some(t) = token_embd {
        let num_elements: usize = t.dims.iter().map(|&d| d as usize).product();
        println!("    Total elements: {}", num_elements);
        println!(
            "    Expected for [vocab×hidden]: {}×{} = {}",
            config.vocab_size,
            config.hidden_dim,
            config.vocab_size * config.hidden_dim
        );

        // Check if embedding lookup would work
        let token_id = 1;
        let expected_start = token_id * config.hidden_dim;
        let expected_end = expected_start + config.hidden_dim;

        println!(
            "    Token {} lookup: [{}, {})",
            token_id, expected_start, expected_end
        );
        if expected_end <= num_elements {
            println!("    -> Within bounds: OK");
        } else {
            println!("    -> OUT OF BOUNDS!");
        }
    }

    Ok(())
}
