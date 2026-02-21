//! PTX Map: Model-to-PTX Source Mapping Tool
//!
//! Toyota Way: Mieruka (見える化) — Make the invisible visible.
//! Maps from model architecture → layers → kernels → PTX analysis → source locations.
//!
//! Answers: "What PTX runs for layer 5's attention projection?"
//! and "Which model layers use Q4KGemv?"

use crate::error::{CliError, Result};
use std::path::Path;

/// Decode-path kernel step within a transformer layer
struct KernelStep {
    /// Step number within the layer
    index: u32,
    /// Human-readable kernel name
    name: &'static str,
    /// Role within the layer (e.g., "QKV", "gate", "down")
    role: &'static str,
    /// Input → output shape description (populated from model dims)
    shape: String,
    /// Source file location in trueno-gpu
    source: &'static str,
    /// Whether this is a batched prefill variant
    is_batched: bool,
}

/// PTX analysis stats extracted from emitted PTX (used in tests)
#[cfg(test)]
struct PtxStats {
    registers: u32,
    shared_bytes: u32,
    global_loads: u32,
    global_stores: u32,
}

/// Model dimensions extracted from GGUF config
struct ModelInfo {
    name: String,
    quant: String,
    num_layers: usize,
    hidden_dim: u32,
    intermediate_dim: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
}

/// Analyze PTX source to count registers, shared memory, and memory ops
#[cfg(test)]
fn analyze_ptx(ptx: &str) -> PtxStats {
    let mut registers = 0u32;
    let mut shared_bytes = 0u32;
    let mut global_loads = 0u32;
    let mut global_stores = 0u32;

    for line in ptx.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with(".reg") {
            registers += parse_angle_bracket_count(trimmed);
        } else if trimmed.contains(".shared") && trimmed.contains(".align") {
            shared_bytes += parse_bracket_count(trimmed);
        } else if trimmed.starts_with("ld.global") {
            global_loads += 1;
        } else if trimmed.starts_with("st.global") {
            global_stores += 1;
        }
    }

    PtxStats {
        registers,
        shared_bytes,
        global_loads,
        global_stores,
    }
}

/// Parse register count from `.reg .f32 %f<24>;` → 24.
#[cfg(test)]
fn parse_angle_bracket_count(line: &str) -> u32 {
    let Some(start) = line.rfind('<') else {
        return 0;
    };
    let Some(end) = line.rfind('>') else { return 0 };
    line[start + 1..end].parse().unwrap_or(0)
}

/// Parse byte count from `.shared .align 4 .b8 shmem[256];` → 256.
#[cfg(test)]
fn parse_bracket_count(line: &str) -> u32 {
    let Some(start) = line.rfind('[') else {
        return 0;
    };
    let Some(end) = line.rfind(']') else { return 0 };
    line[start + 1..end].parse().unwrap_or(0)
}

/// Source location table for kernel types
fn source_location(kernel_name: &str) -> &'static str {
    match kernel_name {
        "VectorizedRmsNormKernel" | "BatchedVectorizedRmsNormKernel" => {
            "trueno-gpu/src/kernels/layernorm.rs"
        }
        "Q4KGemvKernel" | "BatchedQ4KGemvKernel" | "TensorCoreQ4KGemmKernel" => {
            "trueno-gpu/src/kernels/quantize/q4k/"
        }
        "Q6KGemvKernel" | "BatchedQ6KGemvKernel" => "trueno-gpu/src/kernels/quantize/q6k.rs",
        "RopeKernel" | "BatchedRopeKernel" => "trueno-gpu/src/kernels/rope.rs",
        "IncrementalAttentionKernel" | "AttentionKernel" => {
            "trueno-gpu/src/kernels/attention/mod.rs"
        }
        "ResidualAddKernel" | "BatchedResidualAddKernel" => {
            "trueno-gpu/src/kernels/elementwise/residual.rs"
        }
        "SwigluKernel" | "FusedSwigluKernel" | "BatchedSwigluKernel" => {
            "trueno-gpu/src/kernels/activation.rs"
        }
        "KvCacheScatterKernel" => "trueno-gpu/src/kernels/kv_cache.rs",
        "ArgMaxKernel" => "trueno-gpu/src/kernels/argmax.rs",
        _ => "unknown",
    }
}

/// Build the 12-step decode kernel sequence for a transformer layer
fn build_decode_sequence(info: &ModelInfo) -> Vec<KernelStep> {
    let h = info.hidden_dim;
    let inter = info.intermediate_dim;
    let heads = info.num_heads;
    let head_dim = info.head_dim;
    let kv_heads = info.num_kv_heads;
    let qkv_out = heads * head_dim + 2 * kv_heads * head_dim;

    vec![
        KernelStep {
            index: 1,
            name: "VectorizedRmsNormKernel",
            role: "pre-attn norm",
            shape: format!("{h} -> {h}"),
            source: source_location("VectorizedRmsNormKernel"),
            is_batched: false,
        },
        KernelStep {
            index: 2,
            name: "Q4KGemvKernel",
            role: "QKV proj",
            shape: format!("{h} -> {qkv_out}"),
            source: source_location("Q4KGemvKernel"),
            is_batched: false,
        },
        KernelStep {
            index: 3,
            name: "RopeKernel",
            role: "RoPE",
            shape: format!("{head_dim}x{heads} -> same"),
            source: source_location("RopeKernel"),
            is_batched: false,
        },
        KernelStep {
            index: 4,
            name: "IncrementalAttentionKernel",
            role: "GQA attention",
            shape: format!("Q[{heads}]xK[{kv_heads}] -> V"),
            source: source_location("IncrementalAttentionKernel"),
            is_batched: false,
        },
        KernelStep {
            index: 5,
            name: "Q4KGemvKernel",
            role: "out proj",
            shape: format!("{h} -> {h}"),
            source: source_location("Q4KGemvKernel"),
            is_batched: false,
        },
        KernelStep {
            index: 6,
            name: "ResidualAddKernel",
            role: "post-attn residual",
            shape: format!("{h} + {h}"),
            source: source_location("ResidualAddKernel"),
            is_batched: false,
        },
        KernelStep {
            index: 7,
            name: "VectorizedRmsNormKernel",
            role: "pre-FFN norm",
            shape: format!("{h} -> {h}"),
            source: source_location("VectorizedRmsNormKernel"),
            is_batched: false,
        },
        KernelStep {
            index: 8,
            name: "Q4KGemvKernel",
            role: "gate proj",
            shape: format!("{h} -> {inter}"),
            source: source_location("Q4KGemvKernel"),
            is_batched: false,
        },
        KernelStep {
            index: 9,
            name: "Q4KGemvKernel",
            role: "up proj",
            shape: format!("{h} -> {inter}"),
            source: source_location("Q4KGemvKernel"),
            is_batched: false,
        },
        KernelStep {
            index: 10,
            name: "SwigluKernel",
            role: "SwiGLU",
            shape: format!("{inter} -> {inter}"),
            source: source_location("SwigluKernel"),
            is_batched: false,
        },
        KernelStep {
            index: 11,
            name: "Q4KGemvKernel",
            role: "down proj",
            shape: format!("{inter} -> {h}"),
            source: source_location("Q4KGemvKernel"),
            is_batched: false,
        },
        KernelStep {
            index: 12,
            name: "ResidualAddKernel",
            role: "post-FFN residual",
            shape: format!("{h} + {h}"),
            source: source_location("ResidualAddKernel"),
            is_batched: false,
        },
    ]
}

/// Build the batched prefill kernel sequence
fn build_prefill_sequence(info: &ModelInfo) -> Vec<KernelStep> {
    let h = info.hidden_dim;
    let inter = info.intermediate_dim;
    let heads = info.num_heads;
    let head_dim = info.head_dim;
    let kv_heads = info.num_kv_heads;
    let qkv_out = heads * head_dim + 2 * kv_heads * head_dim;

    vec![
        KernelStep {
            index: 1,
            name: "BatchedVectorizedRmsNormKernel",
            role: "pre-attn norm",
            shape: format!("[S,{h}] -> [S,{h}]"),
            source: source_location("BatchedVectorizedRmsNormKernel"),
            is_batched: true,
        },
        KernelStep {
            index: 2,
            name: "BatchedQ4KGemvKernel",
            role: "QKV proj",
            shape: format!("[S,{h}] -> [S,{qkv_out}]"),
            source: source_location("BatchedQ4KGemvKernel"),
            is_batched: true,
        },
        KernelStep {
            index: 3,
            name: "BatchedRopeKernel",
            role: "RoPE",
            shape: format!("[S,{head_dim}x{heads}] -> same"),
            source: source_location("BatchedRopeKernel"),
            is_batched: true,
        },
        KernelStep {
            index: 4,
            name: "AttentionKernel",
            role: "causal attention",
            shape: format!("Q[S,{heads}]xK[S,{kv_heads}] -> V"),
            source: source_location("AttentionKernel"),
            is_batched: false,
        },
        KernelStep {
            index: 5,
            name: "BatchedQ4KGemvKernel",
            role: "out proj",
            shape: format!("[S,{h}] -> [S,{h}]"),
            source: source_location("BatchedQ4KGemvKernel"),
            is_batched: true,
        },
        KernelStep {
            index: 6,
            name: "BatchedResidualAddKernel",
            role: "post-attn residual",
            shape: format!("[S,{h}] + [S,{h}]"),
            source: source_location("BatchedResidualAddKernel"),
            is_batched: true,
        },
        KernelStep {
            index: 7,
            name: "BatchedVectorizedRmsNormKernel",
            role: "pre-FFN norm",
            shape: format!("[S,{h}] -> [S,{h}]"),
            source: source_location("BatchedVectorizedRmsNormKernel"),
            is_batched: true,
        },
        KernelStep {
            index: 8,
            name: "BatchedQ4KGemvKernel",
            role: "gate proj",
            shape: format!("[S,{h}] -> [S,{inter}]"),
            source: source_location("BatchedQ4KGemvKernel"),
            is_batched: true,
        },
        KernelStep {
            index: 9,
            name: "BatchedQ4KGemvKernel",
            role: "up proj",
            shape: format!("[S,{h}] -> [S,{inter}]"),
            source: source_location("BatchedQ4KGemvKernel"),
            is_batched: true,
        },
        KernelStep {
            index: 10,
            name: "BatchedSwigluKernel",
            role: "SwiGLU",
            shape: format!("[S,{inter}] -> [S,{inter}]"),
            source: source_location("BatchedSwigluKernel"),
            is_batched: true,
        },
        KernelStep {
            index: 11,
            name: "BatchedQ4KGemvKernel",
            role: "down proj",
            shape: format!("[S,{inter}] -> [S,{h}]"),
            source: source_location("BatchedQ4KGemvKernel"),
            is_batched: true,
        },
        KernelStep {
            index: 12,
            name: "BatchedResidualAddKernel",
            role: "post-FFN residual",
            shape: format!("[S,{h}] + [S,{h}]"),
            source: source_location("BatchedResidualAddKernel"),
            is_batched: true,
        },
    ]
}

/// Extract model info from GGUF file
#[cfg(feature = "inference")]
fn extract_model_info(model_path: &Path) -> Result<ModelInfo> {
    use realizar::format::{detect_format, ModelFormat};

    // Verify GGUF format
    let magic = std::fs::File::open(model_path)
        .ok()
        .and_then(|mut f| {
            use std::io::Read;
            let mut buf = [0u8; 8];
            f.read_exact(&mut buf).ok()?;
            Some(buf.to_vec())
        })
        .ok_or_else(|| CliError::FileNotFound(model_path.to_path_buf()))?;

    let fmt = detect_format(&magic)
        .map_err(|e| CliError::InvalidFormat(format!("Cannot detect format: {e}")))?;
    if fmt != ModelFormat::Gguf {
        return Err(CliError::InvalidFormat(
            "ptx-map requires a GGUF model (PTX kernels are for quantized inference)".to_string(),
        ));
    }

    let mapped =
        realizar::gguf::MappedGGUFModel::from_path(model_path.to_str().unwrap_or_default())
            .map_err(|e| CliError::ValidationFailed(format!("Failed to load GGUF: {e}")))?;

    let config = realizar::gguf::GGUFConfig::from_gguf(&mapped.model)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to read config: {e}")))?;

    // Extract model name and quantization from filename
    let filename = model_path
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or("unknown");
    let quant = if filename.contains("Q4_K") {
        "Q4_K".to_string()
    } else if filename.contains("Q6_K") {
        "Q6_K".to_string()
    } else if filename.contains("Q5_K") {
        "Q5_K".to_string()
    } else if filename.contains("Q8_0") {
        "Q8_0".to_string()
    } else {
        "Q4_K".to_string()
    };

    let num_kv_heads = if config.num_heads == 28 {
        4 // Qwen2 7B: 28 Q / 4 KV
    } else if config.num_heads == 12 {
        2 // Qwen2 1.5B: 12 Q / 2 KV
    } else {
        config.num_heads // fallback: MHA
    };

    Ok(ModelInfo {
        name: filename.to_string(),
        quant,
        num_layers: config.num_layers,
        hidden_dim: config.hidden_dim as u32,
        intermediate_dim: config.intermediate_dim as u32,
        num_heads: config.num_heads as u32,
        num_kv_heads: num_kv_heads as u32,
        head_dim: (config.hidden_dim / config.num_heads) as u32,
    })
}

/// Print table header
fn print_table_header() {
    println!(
        "  #   Kernel                             Role             Shape                  Source"
    );
    println!("  --- ---------------------------------- ---------------- ---------------------- --------------------------------------------");
}

/// Format shared memory bytes
#[cfg(test)]
fn format_shared(bytes: u32) -> String {
    if bytes == 0 {
        "0".to_string()
    } else if bytes >= 1024 {
        format!("{}KB", bytes / 1024)
    } else {
        format!("{}B", bytes)
    }
}

include!("ptx_map_print_kernel.rs");
