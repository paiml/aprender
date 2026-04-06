//! PAR-001c: Dump GGUF tensor dimensions to verify layout
//!
//! GGUF stores dimensions in "GGML order" which is different from row-major.
//! This script dumps raw dimensions before and after reversal to verify
//! we're interpreting them correctly.

use realizar::gguf::MappedGGUFModel;

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";

    println!("=== PAR-001c: GGUF Tensor Dimension Analysis ===\n");

    let mapped = MappedGGUFModel::from_path(path).expect("Failed to load model");

    // Print dimensions for key tensors
    let key_tensors = [
        "token_embd.weight",
        "output.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "blk.0.ffn_gate.weight",
        "output_norm.weight",
    ];

    println!("Tensor dimensions (after dims.reverse() in GGUF parser):");
    println!("  Format: [dim0, dim1, ...] where dim0=out_dim, dim1=in_dim for matmul weights\n");

    for name in key_tensors {
        if let Some(tensor) = mapped.model.tensors.iter().find(|t| t.name == name) {
            let num_elements = tensor.dims.iter().product::<u64>();
            let qtype = match tensor.qtype {
                0 => "F32",
                1 => "F16",
                2 => "Q4_0",
                3 => "Q4_1",
                12 => "Q4_K",
                14 => "Q6_K",
                _ => "unknown",
            };
            println!(
                "{:30} dims={:?} ({}x elements) qtype={}",
                name, tensor.dims, num_elements, qtype
            );
        } else {
            println!("{:30} NOT FOUND", name);
        }
    }

    // Check specific dimension interpretation for output.weight
    if let Some(output_tensor) = mapped
        .model
        .tensors
        .iter()
        .find(|t| t.name == "output.weight")
    {
        println!("\n\nDetailed analysis of output.weight (LM head):");
        println!("  Raw dims after reverse: {:?}", output_tensor.dims);

        if output_tensor.dims.len() >= 2 {
            let dim0 = output_tensor.dims[0] as usize;
            let dim1 = output_tensor.dims[1] as usize;

            println!("  dim0 = {} (should be vocab_size=32000)", dim0);
            println!("  dim1 = {} (should be hidden_dim=2048)", dim1);

            // What we set in from_ref_with_dims:
            println!("\n  Our interpretation:");
            println!("    in_dim = hidden_dim = 2048");
            println!("    out_dim = vocab_size = 32000");

            // For matmul Y = X @ W, we need:
            // - X: [1, in_dim] = [1, 2048]
            // - W: [in_dim, out_dim] = [2048, 32000] (column-major) or
            // - W: [out_dim, in_dim] = [32000, 2048] (row-major for W^T)
            // - Y: [1, out_dim] = [1, 32000]
            println!("\n  For Y = X @ W^T where W is stored row-major:");
            println!("    X: [1, 2048]");
            println!("    W: [{}, {}] stored row-major", dim0, dim1);
            println!(
                "    We iterate over {} rows, each of size {} elements",
                dim0, dim1
            );

            // Calculate expected byte size for Q6_K
            let superblocks_per_row = dim1.div_ceil(256);
            let bytes_per_row_q6k = superblocks_per_row * 210;
            let total_bytes_q6k = dim0 * bytes_per_row_q6k;

            println!("\n  For Q6_K storage:");
            println!("    Superblocks per row: {}", superblocks_per_row);
            println!("    Bytes per row: {}", bytes_per_row_q6k);
            println!("    Expected total bytes: {}", total_bytes_q6k);

            // Our matmul does:
            // for o in 0..out_dim:
            //     row_start = o * bytes_per_row
            //     dot(row[o], input)
            //
            // This is correct if W is stored as [out_dim, in_dim] row-major
            // i.e., row o has in_dim elements
            println!("\n  Our fused_q4k_parallel_matvec:");
            println!("    out_dim = 32000 (number of rows = vocab_size)");
            println!("    in_dim = 2048 (elements per row = hidden_dim)");

            // Check if dims match
            if dim0 == 32000 && dim1 == 2048 {
                println!("\n  ✓ Dimensions match expected [vocab_size, hidden_dim]");
            } else {
                println!(
                    "\n  ✗ MISMATCH! dim0={} vs vocab=32000, dim1={} vs hidden=2048",
                    dim0, dim1
                );
            }
        }
    }

    // Check token_embd for comparison
    if let Some(embd_tensor) = mapped
        .model
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight")
    {
        println!("\n\nDetailed analysis of token_embd.weight:");
        println!("  Raw dims after reverse: {:?}", embd_tensor.dims);

        if embd_tensor.dims.len() >= 2 {
            println!(
                "  Should be [vocab_size, hidden_dim] = [32000, 2048] = {} elements",
                32000 * 2048
            );
            println!(
                "  Actual elements: {}",
                embd_tensor.dims.iter().product::<u64>()
            );
        }
    }

    println!("\n=== Analysis complete ===");
}
