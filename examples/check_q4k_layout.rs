//! Check Q4K data layout between APR and GGUF

use realizar::apr_transformer::AprTransformer;
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let apr_path = "/home/noah/models/qwen2.5-coder-1.5b-q4k.apr";
    let gguf_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    println!("Loading models...");
    let apr_model = AprTransformer::from_apr_file(apr_path)?;
    let mapped = MappedGGUFModel::from_path(gguf_path)?;
    let gguf_model = OwnedQuantizedModel::from_mapped(&mapped)?;

    // Check attn_output Q4K bytes
    let q4k_layers = apr_model.q4k_layers.as_ref().expect("No Q4K layers");
    let apr_attn_out = q4k_layers[0]
        .attn_output_weight
        .as_ref()
        .expect("No APR Q4K attn_output");
    let gguf_attn_out = &gguf_model.layers()[0].attn_output_weight;

    println!("\n=== attn_output Q4K bytes ===");
    println!("APR bytes: {}", apr_attn_out.len());
    println!("GGUF bytes: {}", gguf_attn_out.data.len());

    // Compare bytes
    let mismatches: usize = apr_attn_out
        .iter()
        .zip(gguf_attn_out.data.iter())
        .filter(|(&a, &b)| a != b)
        .count();
    println!(
        "Byte mismatches: {} / {}",
        mismatches,
        apr_attn_out.len().min(gguf_attn_out.data.len())
    );

    // First 32 bytes
    println!("\nAPR first 32 bytes: {:?}", &apr_attn_out[..32]);
    println!("GGUF first 32 bytes: {:?}", &gguf_attn_out.data[..32]);

    // Compare in_dim and out_dim
    println!(
        "\nGGUF weight dims: in_dim={}, out_dim={}",
        gguf_attn_out.in_dim, gguf_attn_out.out_dim
    );

    Ok(())
}
