//! APR Q4K Verification Example
//!
//! Verifies that APR Q4K files can be loaded and parsed correctly.
//!
//! Run with: cargo run --release --example verify_apr_q4k

use realizar::apr::{AprHeader, TensorEntry};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::Instant;

fn main() {
    println!("APR Q4K Verification");
    println!("====================");
    println!();

    let apr_path =
        std::env::var("APR_PATH").unwrap_or_else(|_| "/tmp/qwen2.5-coder-1.5b-q4k.apr".to_string());

    if !Path::new(&apr_path).exists() {
        eprintln!("APR file not found: {}", apr_path);
        eprintln!("Run convert_apr_q4k example first to create the file.");
        return;
    }

    println!("File: {}", apr_path);

    let start = Instant::now();

    // Read entire file
    let mut file = File::open(&apr_path).expect("open file");
    let file_len = file.metadata().expect("metadata").len() as usize;
    let mut data = vec![0u8; file_len];
    file.read_exact(&mut data).expect("read file");

    println!("File size: {:.2} MB", file_len as f64 / 1_000_000.0);
    println!("Load time: {:.3}s", start.elapsed().as_secs_f64());
    println!();

    // Parse header
    let header = AprHeader::from_bytes(&data).expect("parse header");
    println!("Header:");
    println!("  Magic: {:?}", &header.magic);
    println!("  Version: {}.{}", header.version.0, header.version.1);
    println!("  Tensor count: {}", header.tensor_count);
    println!("  Flags: quantized={}", header.flags.is_quantized());
    println!("  Metadata offset: {}", header.metadata_offset);
    println!("  Metadata size: {}", header.metadata_size);
    println!("  Tensor index offset: {}", header.tensor_index_offset);
    println!("  Data offset: {}", header.data_offset);
    println!();

    // Parse metadata
    let metadata_start = header.metadata_offset as usize;
    let metadata_end = metadata_start + header.metadata_size as usize;
    let metadata_json: serde_json::Value =
        serde_json::from_slice(&data[metadata_start..metadata_end]).expect("parse metadata");

    println!("Metadata:");
    println!(
        "  Model type: {}",
        metadata_json
            .get("model_type")
            .unwrap_or(&serde_json::Value::Null)
    );
    println!(
        "  Architecture: {}",
        metadata_json
            .get("architecture")
            .unwrap_or(&serde_json::Value::Null)
    );
    println!(
        "  Hidden size: {}",
        metadata_json
            .get("hidden_size")
            .unwrap_or(&serde_json::Value::Null)
    );
    println!(
        "  Num layers: {}",
        metadata_json
            .get("num_layers")
            .unwrap_or(&serde_json::Value::Null)
    );
    println!(
        "  Quantization: {}",
        metadata_json
            .get("quantization")
            .unwrap_or(&serde_json::Value::Null)
    );
    println!();

    // Parse tensor index
    let index_start = header.tensor_index_offset as usize;
    let data_start = header.data_offset as usize;
    let mut pos = index_start;
    let mut tensor_count = 0usize;
    let mut q4k_count = 0usize;
    let mut total_bytes = 0usize;

    println!("Tensors (first 10):");
    while pos < data_start && tensor_count < header.tensor_count as usize {
        // Parse binary tensor entry
        let (entry, bytes_read) = TensorEntry::from_binary(&data[pos..]).expect("parse tensor");
        pos += bytes_read;

        if tensor_count < 10 {
            println!(
                "  {}: dtype={} shape={:?} size={} bytes",
                entry.name, entry.dtype, entry.shape, entry.size
            );
        }

        // Count Q4K tensors (APR dtype 8)
        if entry.dtype == "Q4_K" {
            q4k_count += 1;
        }
        total_bytes += entry.size as usize;
        tensor_count += 1;
    }
    println!("  ... ({} more tensors)", tensor_count.saturating_sub(10));
    println!();

    println!("Summary:");
    println!("  Total tensors: {}", tensor_count);
    println!(
        "  Q4K tensors: {} ({:.1}%)",
        q4k_count,
        q4k_count as f64 / tensor_count as f64 * 100.0
    );
    println!(
        "  Total tensor bytes: {:.2} MB",
        total_bytes as f64 / 1_000_000.0
    );
    println!();

    // Verify we can access tensor data
    println!("Verifying tensor data access...");
    pos = index_start;
    let mut verified = 0usize;
    while pos < data_start && verified < tensor_count {
        let (entry, bytes_read) = TensorEntry::from_binary(&data[pos..]).expect("parse tensor");
        pos += bytes_read;

        let tensor_start = data_start + entry.offset as usize;
        let tensor_end = tensor_start + entry.size as usize;

        if tensor_end > data.len() {
            eprintln!("ERROR: Tensor {} exceeds file bounds!", entry.name);
            return;
        }

        // Verify data is accessible (just check first byte)
        let _ = data[tensor_start];
        verified += 1;
    }

    println!("All {} tensors verified successfully!", verified);
    println!();
    println!("Total time: {:.3}s", start.elapsed().as_secs_f64());
}
