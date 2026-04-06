use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "../aprender/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    // Token 13048 = "Hi"
    let token = 13048u32;

    // Get embedding
    let embedding = model.embed(&[token]);

    println!("Embedding for token {} (Hi):", token);
    println!("  Shape: [{}]", embedding.len());
    println!("  First 10 values: {:?}", &embedding[..10]);
    println!("  Last 10 values: {:?}", &embedding[embedding.len() - 10..]);
    println!("  Sum: {:.4}", embedding.iter().sum::<f32>());
    println!(
        "  Mean: {:.6}",
        embedding.iter().sum::<f32>() / embedding.len() as f32
    );
    println!(
        "  Max: {:.4}",
        embedding.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    );
    println!(
        "  Min: {:.4}",
        embedding.iter().copied().fold(f32::INFINITY, f32::min)
    );

    // Check for NaN/Inf
    let nan_count = embedding.iter().filter(|x| x.is_nan()).count();
    let inf_count = embedding.iter().filter(|x| x.is_infinite()).count();
    println!("  NaN count: {}, Inf count: {}", nan_count, inf_count);

    Ok(())
}
