//! Test FFN Q4K vs F32 paths
//! Debug tool for PMAT-103: Isolate where APR diverges from correct output
#![allow(deprecated)]
use realizar::apr_transformer::AprTransformer;
use trueno::backends::q4k::matmul_q4k_f32_colmajor_dispatch as matmul_q4k_f32;
use trueno::backends::q6k::matmul_q6k_f32_colmajor_dispatch as matmul_q6k_f32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let apr_path = "/tmp/qwen2.5-coder-1.5b-q4k.apr";

    println!("Loading APR from: {}", apr_path);
    let apr = AprTransformer::from_apr_file(apr_path)?;

    let hidden_dim = apr.config.hidden_dim;
    let intermediate_dim = apr.config.intermediate_dim;

    println!("Hidden dim: {}", hidden_dim);
    println!("Intermediate dim: {}", intermediate_dim);

    // Check if we have Q4K layers
    let Some(q4k_layers) = &apr.q4k_layers else {
        println!("No Q4K layers found!");
        return Ok(());
    };

    println!("\n=== Layer 0 Q4K Status ===");
    let q4k = &q4k_layers[0];
    println!(
        "ffn_gate_weight: {} bytes",
        q4k.ffn_gate_weight.as_ref().map_or(0, |v| v.len())
    );
    println!(
        "ffn_up_weight: {} bytes",
        q4k.ffn_up_weight.as_ref().map_or(0, |v| v.len())
    );
    println!(
        "ffn_down_weight: {} bytes",
        q4k.ffn_down_weight.as_ref().map_or(0, |v| v.len())
    );
    println!(
        "ffn_down_weight_q6k: {} bytes",
        q4k.ffn_down_weight_q6k.as_ref().map_or(0, |v| v.len())
    );

    // Create a test input vector (normalized random values to simulate RMSNorm output)
    let test_input: Vec<f32> = (0..hidden_dim)
        .map(|i| (i as f32 * 0.01).sin() * 0.5)
        .collect();

    println!("\n=== Test Input ===");
    println!("First 8: {:?}", &test_input[..8]);
    println!(
        "Input norm: {:.6}",
        test_input.iter().map(|x| x * x).sum::<f32>().sqrt()
    );

    // Test F32 path: FFN gate projection
    let f32_layer = &apr.layers[0];
    let f32_ffn_gate = f32_layer.ffn_gate_weight.as_ref().expect("no ffn_gate");

    println!("\n=== F32 FFN Gate ===");
    println!(
        "Weight len: {} (expected: {})",
        f32_ffn_gate.len(),
        hidden_dim * intermediate_dim
    );

    // F32 matmul: output[i] = sum_j(weight[i*in_dim + j] * input[j])
    // Weight is [out_dim, in_dim] = [intermediate_dim, hidden_dim]
    let mut f32_gate_out = vec![0.0f32; intermediate_dim];
    for o in 0..intermediate_dim {
        for i in 0..hidden_dim {
            f32_gate_out[o] += f32_ffn_gate[o * hidden_dim + i] * test_input[i];
        }
    }
    println!("F32 gate output first 8: {:?}", &f32_gate_out[..8]);
    println!(
        "F32 gate output norm: {:.6}",
        f32_gate_out.iter().map(|x| x * x).sum::<f32>().sqrt()
    );

    // Test Q4K path: FFN gate projection
    if let Some(ref q4k_gate) = q4k.ffn_gate_weight {
        println!("\n=== Q4K FFN Gate ===");
        println!("Q4K bytes: {}", q4k_gate.len());

        // Q4K matmul using GGML convention: matmul_q4k_f32(q4k_bytes, input, ne0, ne1)
        // ne0 = output dimension (intermediate_dim = 8960)
        // ne1 = input dimension (hidden_dim = 1536)
        // APR tensor dims: [8960, 1536] -> ne0=8960, ne1=1536
        let q4k_gate_out = matmul_q4k_f32(q4k_gate, &test_input, intermediate_dim, hidden_dim);

        println!("Q4K gate output first 8: {:?}", &q4k_gate_out[..8]);
        println!(
            "Q4K gate output norm: {:.6}",
            q4k_gate_out.iter().map(|x| x * x).sum::<f32>().sqrt()
        );

        // Compare F32 vs Q4K
        let diff: f32 = f32_gate_out
            .iter()
            .zip(q4k_gate_out.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum::<f32>()
            / intermediate_dim as f32;
        let max_diff: f32 = f32_gate_out
            .iter()
            .zip(q4k_gate_out.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        println!("\nF32 vs Q4K comparison:");
        println!("  Mean absolute diff: {:.6}", diff);
        println!("  Max absolute diff: {:.6}", max_diff);

        // Correlation
        let dot: f64 = f32_gate_out
            .iter()
            .zip(q4k_gate_out.iter())
            .map(|(&a, &b)| (a as f64) * (b as f64))
            .sum();
        let f32_sq: f64 = f32_gate_out.iter().map(|&x| (x as f64).powi(2)).sum();
        let q4k_sq: f64 = q4k_gate_out.iter().map(|&x| (x as f64).powi(2)).sum();
        let corr = dot / (f32_sq.sqrt() * q4k_sq.sqrt());
        println!("  Correlation: {:.6}", corr);
    }

    // Test FFN up
    if let Some(ref q4k_up) = q4k.ffn_up_weight {
        println!("\n=== Q4K FFN Up ===");
        let f32_ffn_up = &f32_layer.ffn_up_weight;

        // F32 path
        let mut f32_up_out = vec![0.0f32; intermediate_dim];
        for o in 0..intermediate_dim {
            for i in 0..hidden_dim {
                f32_up_out[o] += f32_ffn_up[o * hidden_dim + i] * test_input[i];
            }
        }

        // Q4K path
        let q4k_up_out = matmul_q4k_f32(q4k_up, &test_input, intermediate_dim, hidden_dim);

        println!("F32 up output first 8: {:?}", &f32_up_out[..8]);
        println!("Q4K up output first 8: {:?}", &q4k_up_out[..8]);

        let diff: f32 = f32_up_out
            .iter()
            .zip(q4k_up_out.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum::<f32>()
            / intermediate_dim as f32;
        println!("Mean absolute diff: {:.6}", diff);
    }

    // Test FFN down (Q6K)
    if let Some(ref q6k_down) = q4k.ffn_down_weight_q6k {
        println!("\n=== Q6K FFN Down ===");
        let f32_ffn_down = &f32_layer.ffn_down_weight;

        // Create intermediate input (simulating SwiGLU output)
        let intermediate_input: Vec<f32> = (0..intermediate_dim)
            .map(|i| (i as f32 * 0.001).cos() * 0.3)
            .collect();

        // F32 path: down projects [intermediate_dim] -> [hidden_dim]
        // Weight is [hidden_dim, intermediate_dim] = [1536, 8960]
        let mut f32_down_out = vec![0.0f32; hidden_dim];
        for o in 0..hidden_dim {
            for i in 0..intermediate_dim {
                f32_down_out[o] += f32_ffn_down[o * intermediate_dim + i] * intermediate_input[i];
            }
        }

        // Q6K path: ne0=hidden_dim (1536), ne1=intermediate_dim (8960)
        let q6k_down_out =
            matmul_q6k_f32(q6k_down, &intermediate_input, hidden_dim, intermediate_dim);

        println!("F32 down output first 8: {:?}", &f32_down_out[..8]);
        println!("Q6K down output first 8: {:?}", &q6k_down_out[..8]);

        let diff: f32 = f32_down_out
            .iter()
            .zip(q6k_down_out.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum::<f32>()
            / hidden_dim as f32;
        let max_diff: f32 = f32_down_out
            .iter()
            .zip(q6k_down_out.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("Mean absolute diff: {:.6}", diff);
        println!("Max absolute diff: {:.6}", max_diff);

        // Correlation
        let dot: f64 = f32_down_out
            .iter()
            .zip(q6k_down_out.iter())
            .map(|(&a, &b)| (a as f64) * (b as f64))
            .sum();
        let f32_sq: f64 = f32_down_out.iter().map(|&x| (x as f64).powi(2)).sum();
        let q6k_sq: f64 = q6k_down_out.iter().map(|&x| (x as f64).powi(2)).sum();
        let corr = dot / (f32_sq.sqrt() * q6k_sq.sqrt());
        println!("Correlation: {:.6}", corr);
    }

    Ok(())
}
