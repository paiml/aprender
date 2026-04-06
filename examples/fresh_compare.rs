//! Fresh CPU vs GPU comparison
#[cfg(feature = "cuda")]
use realizar::gguf::OwnedQuantizedModelCuda;
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or("/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf".to_string());
    let mapped = MappedGGUFModel::from_path(&path).expect("Failed to load");

    // Test token sequence
    let tokens: Vec<u32> = vec![791]; // Single token

    println!("Running forward pass on CPU (batch)...");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("Failed to parse");
    let cpu_logits = model.forward(&tokens).expect("CPU forward failed");
    let cpu_argmax = cpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    println!(
        "CPU argmax: {} (logit: {:.4})",
        cpu_argmax, cpu_logits[cpu_argmax]
    );

    println!("\nRunning forward pass on CPU (with cache)...");
    let config = QuantizedGenerateConfig {
        max_tokens: 1,
        temperature: 0.0,
        top_k: 1,
        ..Default::default()
    };
    let output = model
        .generate_with_cache(&tokens, &config)
        .expect("CPU generate failed");
    println!("CPU generated token: {:?}", output.last());

    #[cfg(feature = "cuda")]
    {
        println!("\nRunning forward pass on GPU...");
        let model = OwnedQuantizedModel::from_mapped(&mapped).expect("Failed to parse");
        let mut cuda_model = OwnedQuantizedModelCuda::new(model, 0).expect("CUDA init failed");

        let gpu_output = cuda_model
            .generate_gpu_resident(&tokens, &config)
            .expect("GPU forward failed");
        println!("GPU generated token: {:?}", gpu_output.last());

        let cpu_last = *output.last().unwrap_or(&0);
        let gpu_last = *gpu_output.last().unwrap_or(&0);

        println!("\nComparison:");
        if cpu_last == gpu_last {
            println!("✓ MATCH: both generated {}", cpu_last);
        } else {
            println!("✗ DIVERGE: CPU={}, GPU={}", cpu_last, gpu_last);
        }
    }
}
