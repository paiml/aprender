//! Compare raw Q4K bytes between GGUF and APR
//! Debug tool for PMAT-103: Find where APR Q4K bytes diverge
#![allow(deprecated, clippy::manual_let_else)]
use realizar::apr_transformer::AprTransformer;
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let apr_path = "/tmp/qwen2.5-coder-1.5b-q4k.apr";
    let gguf_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    println!("Loading GGUF from: {}", gguf_path);
    let mapped = MappedGGUFModel::from_path(gguf_path)?;
    let gguf = OwnedQuantizedModel::from_mapped(&mapped)?;

    println!("Loading APR from: {}", apr_path);
    let apr = AprTransformer::from_apr_file(apr_path)?;

    let q4k_layers = if let Some(layers) = &apr.q4k_layers {
        layers
    } else {
        println!("No Q4K layers in APR!");
        return Ok(());
    };

    // Compare ffn_gate bytes
    println!("\n=== FFN Gate Weight Comparison (Layer 0) ===");

    // APR Q4K bytes
    let apr_gate = q4k_layers[0]
        .ffn_gate_weight
        .as_ref()
        .expect("no apr ffn_gate");
    println!("APR ffn_gate bytes: {}", apr_gate.len());
    println!("APR first 64 bytes: {:?}", &apr_gate[..64]);

    // GGUF ffn_gate bytes (Option<OwnedQuantizedTensor>)
    let gguf_layer0 = &gguf.layers()[0];
    let gguf_gate_tensor = gguf_layer0
        .ffn_gate_weight
        .as_ref()
        .expect("no gguf ffn_gate");
    println!("\nGGUF ffn_gate bytes: {}", gguf_gate_tensor.data.len());
    println!("GGUF first 64 bytes: {:?}", &gguf_gate_tensor.data[..64]);

    // Compare byte-by-byte
    let gguf_gate = &gguf_gate_tensor.data;
    if apr_gate.len() == gguf_gate.len() {
        let mut mismatches = 0;
        let mut first_mismatch_idx = None;
        for i in 0..apr_gate.len() {
            if apr_gate[i] != gguf_gate[i] {
                if first_mismatch_idx.is_none() {
                    first_mismatch_idx = Some(i);
                }
                mismatches += 1;
            }
        }
        println!("\nByte comparison:");
        println!("  Total mismatches: {} / {}", mismatches, apr_gate.len());
        if let Some(idx) = first_mismatch_idx {
            println!(
                "  First mismatch at byte {}: APR={}, GGUF={}",
                idx, apr_gate[idx], gguf_gate[idx]
            );
            // Show context around first mismatch
            let start = idx.saturating_sub(8);
            let end = std::cmp::min(idx + 8, apr_gate.len());
            println!("  APR around mismatch:  {:?}", &apr_gate[start..end]);
            println!("  GGUF around mismatch: {:?}", &gguf_gate[start..end]);
        }
    } else {
        println!(
            "\nLength mismatch: APR={}, GGUF={}",
            apr_gate.len(),
            gguf_gate.len()
        );
    }

    // Compare ffn_up bytes
    println!("\n=== FFN Up Weight Comparison (Layer 0) ===");
    let apr_up = q4k_layers[0].ffn_up_weight.as_ref().expect("no apr ffn_up");
    let gguf_up = &gguf_layer0.ffn_up_weight.data; // OwnedQuantizedTensor (not Option)
    println!("APR ffn_up bytes: {}", apr_up.len());
    println!("GGUF ffn_up bytes: {}", gguf_up.len());
    println!("APR first 64 bytes: {:?}", &apr_up[..64]);
    println!("GGUF first 64 bytes: {:?}", &gguf_up[..64]);

    if apr_up.len() == gguf_up.len() {
        let mismatches: usize = apr_up
            .iter()
            .zip(gguf_up.iter())
            .filter(|(&a, &b)| a != b)
            .count();
        println!("Total mismatches: {} / {}", mismatches, apr_up.len());
    }

    // Compare ffn_down (Q6K) bytes
    println!("\n=== FFN Down Weight Comparison (Layer 0) - Q6K ===");
    let apr_down = q4k_layers[0]
        .ffn_down_weight_q6k
        .as_ref()
        .expect("no apr ffn_down_q6k");
    let gguf_down = &gguf_layer0.ffn_down_weight.data; // OwnedQuantizedTensor (not Option)
    println!("APR ffn_down bytes: {}", apr_down.len());
    println!("GGUF ffn_down bytes: {}", gguf_down.len());
    println!("APR first 64 bytes: {:?}", &apr_down[..64]);
    println!("GGUF first 64 bytes: {:?}", &gguf_down[..64]);

    if apr_down.len() == gguf_down.len() {
        let mismatches: usize = apr_down
            .iter()
            .zip(gguf_down.iter())
            .filter(|(&a, &b)| a != b)
            .count();
        println!("Total mismatches: {} / {}", mismatches, apr_down.len());
    }

    // Test: direct GGUF Q4K matmul vs APR Q4K matmul
    println!("\n=== Direct Q4K Kernel Test ===");
    use trueno::backends::q4k::matmul_q4k_f32_colmajor_dispatch;

    let hidden_dim = 1536;
    let intermediate_dim = 8960;
    let test_input: Vec<f32> = (0..hidden_dim)
        .map(|i| (i as f32 * 0.01).sin() * 0.5)
        .collect();

    // Using GGUF bytes directly
    let gguf_result =
        matmul_q4k_f32_colmajor_dispatch(gguf_gate, &test_input, intermediate_dim, hidden_dim);
    println!("GGUF kernel result first 8: {:?}", &gguf_result[..8]);

    // Using APR bytes
    let apr_result =
        matmul_q4k_f32_colmajor_dispatch(apr_gate, &test_input, intermediate_dim, hidden_dim);
    println!("APR kernel result first 8:  {:?}", &apr_result[..8]);

    // Correlation
    let dot: f64 = gguf_result
        .iter()
        .zip(apr_result.iter())
        .map(|(&a, &b)| (a as f64) * (b as f64))
        .sum();
    let g_sq: f64 = gguf_result.iter().map(|&x| (x as f64).powi(2)).sum();
    let a_sq: f64 = apr_result.iter().map(|&x| (x as f64).powi(2)).sum();
    let corr = dot / (g_sq.sqrt() * a_sq.sqrt());
    println!("GGUF vs APR kernel correlation: {:.6}", corr);

    Ok(())
}
