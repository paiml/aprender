//! Compare CPU vs GPU generate
use realizar::gguf::{
    MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda, QuantizedGenerateConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    // Test with a few prompt tokens
    let prompt_tokens: Vec<u32> = vec![791]; // "Hello" token
    let gen_config = QuantizedGenerateConfig::deterministic(1);

    // CPU generate
    let cpu_tokens = model.generate(&prompt_tokens, &gen_config)?;
    println!("CPU generated tokens: {:?}", cpu_tokens);

    // GPU generate
    let mut cuda_model = OwnedQuantizedModelCuda::new(model, 0)?;
    let gpu_tokens = cuda_model.generate_gpu_resident(&prompt_tokens, &gen_config)?;
    println!("GPU generated tokens: {:?}", gpu_tokens);

    if cpu_tokens == gpu_tokens {
        println!("\n✓ CPU and GPU match!");
    } else {
        println!("\n✗ CPU and GPU differ!");
        println!("CPU: {:?}", cpu_tokens);
        println!("GPU: {:?}", gpu_tokens);
    }

    Ok(())
}
