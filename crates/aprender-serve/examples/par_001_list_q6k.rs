//! PAR-001: List all Q6_K tensors and their dimensions
//!
//! Check if any Q6_K tensors have out_dim != 256, which would break
//! the column-major assumption.

use realizar::gguf::MappedGGUFModel;

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";

    println!("=== PAR-001: List Q6_K Tensors ===\n");

    let mapped = MappedGGUFModel::from_path(path).expect("Failed to load model");

    println!("Total tensors: {}", mapped.model.tensors.len());

    let q6k_type = 14; // Q6_K
    let q6k_tensors: Vec<_> = mapped
        .model
        .tensors
        .iter()
        .filter(|t| t.qtype == q6k_type)
        .collect();

    println!("Q6_K tensors: {}\n", q6k_tensors.len());

    for tensor in &q6k_tensors {
        let dims = &tensor.dims;
        let in_dim = dims.first().copied().unwrap_or(0) as usize;
        let out_dim = dims.get(1).copied().unwrap_or(0) as usize;

        // Check if this tensor would work with column-major
        let works_colmajor = out_dim == 256;

        println!("{}", tensor.name);
        println!("  dims: {:?}", dims);
        println!("  in_dim={}, out_dim={}", in_dim, out_dim);
        println!(
            "  column-major compatible: {}",
            if works_colmajor { "✓ yes" } else { "✗ NO!" }
        );
        println!();
    }

    // Also list Q4_K tensors for comparison
    let q4k_type = 12;
    let q4k_count = mapped
        .model
        .tensors
        .iter()
        .filter(|t| t.qtype == q4k_type)
        .count();
    println!("Q4_K tensors: {}", q4k_count);

    println!("\n=== Complete ===");
}
