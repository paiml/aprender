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
            // .reg .f32 %f<24>; → 24 registers
            if let Some(angle) = trimmed.rfind('<') {
                if let Some(end) = trimmed.rfind('>') {
                    if let Ok(n) = trimmed[angle + 1..end].parse::<u32>() {
                        registers += n;
                    }
                }
            }
        } else if trimmed.contains(".shared") && trimmed.contains(".align") {
            // .shared .align 4 .b8 shmem[256]; → 256 bytes
            if let Some(bracket) = trimmed.rfind('[') {
                if let Some(end) = trimmed.rfind(']') {
                    if let Ok(n) = trimmed[bracket + 1..end].parse::<u32>() {
                        shared_bytes += n;
                    }
                }
            }
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

/// Source location table for kernel types
fn source_location(kernel_name: &str) -> &'static str {
    match kernel_name {
        "VectorizedRmsNormKernel" | "BatchedVectorizedRmsNormKernel" => {
            "trueno-gpu/src/kernels/layernorm.rs"
        }
        "Q4KGemvKernel" | "BatchedQ4KGemvKernel" | "TensorCoreQ4KGemmKernel" => {
            "trueno-gpu/src/kernels/quantize/q4k/"
        }
        "Q6KGemvKernel" | "BatchedQ6KGemvKernel" => {
            "trueno-gpu/src/kernels/quantize/q6k.rs"
        }
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
        .ok_or_else(|| {
            CliError::FileNotFound(model_path.to_path_buf())
        })?;

    let fmt = detect_format(&magic)
        .map_err(|e| CliError::InvalidFormat(format!("Cannot detect format: {e}")))?;
    if fmt != ModelFormat::Gguf {
        return Err(CliError::InvalidFormat(
            "ptx-map requires a GGUF model (PTX kernels are for quantized inference)".to_string(),
        ));
    }

    let mapped = realizar::gguf::MappedGGUFModel::from_path(
        model_path.to_str().unwrap_or_default(),
    )
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
    println!("  #   Kernel                             Role             Shape                  Source");
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

/// Print the kernel table
fn print_kernel_table(
    steps: &[KernelStep],
    kernel_filter: Option<&str>,
) {
    print_table_header();

    for step in steps {
        // Apply kernel filter
        if let Some(filter) = kernel_filter {
            let filter_lower = filter.to_lowercase();
            if !step.name.to_lowercase().contains(&filter_lower)
                && !step.role.to_lowercase().contains(&filter_lower)
            {
                continue;
            }
        }

        let batched_marker = if step.is_batched { " [B]" } else { "" };
        println!(
            "  {:<3} {:<34} {:<16} {:<22} {}{}",
            step.index,
            step.name,
            step.role,
            truncate_shape(&step.shape, 22),
            step.source,
            batched_marker,
        );
    }
}

/// Truncate a shape string to fit column width
fn truncate_shape(shape: &str, max_len: usize) -> String {
    if shape.len() <= max_len {
        shape.to_string()
    } else {
        format!("{}...", &shape[..max_len - 3])
    }
}

/// Print reverse lookup: kernel name → which steps use it
fn print_reverse_lookup(steps: &[KernelStep], kernel_name: &str, info: &ModelInfo) {
    let filter_lower = kernel_name.to_lowercase();
    let matching: Vec<&KernelStep> = steps
        .iter()
        .filter(|s| s.name.to_lowercase().contains(&filter_lower))
        .collect();

    if matching.is_empty() {
        println!("  No kernel matching '{}' found in the forward pass.", kernel_name);
        return;
    }

    println!("  Reverse lookup: '{}'\n", kernel_name);
    println!(
        "  {:<3} {:<8} {:<20}",
        "#", "Role", "Shape"
    );
    println!("  {:-<3} {:-<8} {:-<20}", "", "", "");

    for step in &matching {
        println!(
            "  {:<3} {:<8} {:<20}",
            step.index, step.role, truncate_shape(&step.shape, 20)
        );
    }

    let launches_per_layer = matching.len();
    let total = launches_per_layer * info.num_layers;
    println!(
        "\n  {} launches/layer x {} layers = {} total launches",
        launches_per_layer, info.num_layers, total
    );
}

/// Print JSON output
fn print_json(steps: &[KernelStep], info: &ModelInfo, prefill: bool) {
    println!("{{");
    println!("  \"model\": \"{}\",", info.name);
    println!("  \"quantization\": \"{}\",", info.quant);
    println!("  \"num_layers\": {},", info.num_layers);
    println!("  \"hidden_dim\": {},", info.hidden_dim);
    println!("  \"intermediate_dim\": {},", info.intermediate_dim);
    println!("  \"num_heads\": {},", info.num_heads);
    println!("  \"num_kv_heads\": {},", info.num_kv_heads);
    println!("  \"head_dim\": {},", info.head_dim);
    println!("  \"mode\": \"{}\",", if prefill { "prefill" } else { "decode" });
    println!("  \"kernels_per_layer\": {},", steps.len());
    let total = steps.len() * info.num_layers + 2; // +2 for final norm + lm_head
    println!("  \"total_launches\": {},", total);
    println!("  \"steps\": [");
    for (i, step) in steps.iter().enumerate() {
        let comma = if i + 1 < steps.len() { "," } else { "" };
        println!("    {{");
        println!("      \"index\": {},", step.index);
        println!("      \"kernel\": \"{}\",", step.name);
        println!("      \"role\": \"{}\",", step.role);
        println!("      \"shape\": \"{}\",", step.shape);
        println!("      \"source\": \"{}\",", step.source);
        println!("      \"batched\": {}", step.is_batched);
        println!("    }}{}", comma);
    }
    println!("  ]");
    println!("}}");
}

/// Main entry point for ptx-map command
#[allow(clippy::fn_params_excessive_bools)]
pub fn run(
    model_path: &Path,
    kernel_filter: Option<&str>,
    reverse: Option<&str>,
    json: bool,
    verbose: bool,
    prefill: bool,
) -> Result<()> {
    #[cfg(feature = "inference")]
    {
        let _verbose = verbose; // reserved for future PTX snippet output
        let info = extract_model_info(model_path)?;

        let steps = if prefill {
            build_prefill_sequence(&info)
        } else {
            build_decode_sequence(&info)
        };

        // JSON output
        if json {
            print_json(&steps, &info, prefill);
            return Ok(());
        }

        // Reverse lookup mode
        if let Some(kernel_name) = reverse {
            println!(
                "\nModel: {} ({})\n  {} layers, hidden={}, intermediate={}, heads={}, head_dim={}\n",
                info.name, info.quant, info.num_layers, info.hidden_dim,
                info.intermediate_dim, info.num_heads, info.head_dim
            );
            print_reverse_lookup(&steps, kernel_name, &info);
            return Ok(());
        }

        // Default: full table
        let mode = if prefill { "Prefill" } else { "Decode" };
        println!(
            "\nModel: {} ({})\n  {} layers, hidden={}, intermediate={}, heads={}, head_dim={}\n",
            info.name, info.quant, info.num_layers, info.hidden_dim,
            info.intermediate_dim, info.num_heads, info.head_dim
        );
        println!("{} Kernel Sequence (per transformer layer, {} launches):\n", mode, steps.len());

        print_kernel_table(&steps, kernel_filter);

        // Summary
        let total = steps.len() * info.num_layers + 2; // +2 for final norm + lm_head
        println!(
            "\n  Total: {} kernels/layer x {} layers + 2 (final norm, lm_head) = {} launches",
            steps.len(),
            info.num_layers,
            total
        );

        // PTX parity summary (uses realizar's validate_all_kernel_pairs)
        {
            use realizar::ptx_parity::{validate_all_kernel_pairs, KernelDimensions};
            let dims = KernelDimensions {
                hidden_dim: info.hidden_dim,
                intermediate_dim: info.intermediate_dim,
                num_heads: info.num_heads,
                head_dim: info.head_dim,
                rope_theta: 1_000_000.0,
                epsilon: 1e-6,
            };
            let report = validate_all_kernel_pairs(&dims);
            if report.total > 0 {
                println!(
                    "  PTX Parity: {}/{} kernel pairs {}",
                    report.passed,
                    report.total,
                    if report.all_passed() { "PASS" } else { "FAIL" }
                );
            }
        }

        println!();
        Ok(())
    }

    #[cfg(not(feature = "inference"))]
    {
        let _ = (model_path, kernel_filter, reverse, json, verbose, prefill);
        Err(CliError::FeatureDisabled(
            "ptx-map requires the 'inference' feature (--features inference)".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_ptx_counts_registers() {
        let ptx = r#"
.version 7.0
.target sm_80
.reg .f32 %f<24>;
.reg .b32 %r<8>;
ld.global.f32 %f0, [%r1];
ld.global.f32 %f1, [%r2];
st.global.f32 [%r3], %f2;
"#;
        let stats = analyze_ptx(ptx);
        assert_eq!(stats.registers, 32); // 24 + 8
        assert_eq!(stats.global_loads, 2);
        assert_eq!(stats.global_stores, 1);
        assert_eq!(stats.shared_bytes, 0);
    }

    #[test]
    fn test_analyze_ptx_counts_shared_memory() {
        let ptx = ".shared .align 4 .b8 shmem[256];";
        let stats = analyze_ptx(ptx);
        assert_eq!(stats.shared_bytes, 256);
    }

    #[test]
    fn test_analyze_ptx_empty() {
        let stats = analyze_ptx("");
        assert_eq!(stats.registers, 0);
        assert_eq!(stats.shared_bytes, 0);
        assert_eq!(stats.global_loads, 0);
        assert_eq!(stats.global_stores, 0);
    }

    #[test]
    fn test_source_location_known_kernels() {
        assert_eq!(
            source_location("VectorizedRmsNormKernel"),
            "trueno-gpu/src/kernels/layernorm.rs"
        );
        assert_eq!(
            source_location("Q4KGemvKernel"),
            "trueno-gpu/src/kernels/quantize/q4k/"
        );
        assert_eq!(
            source_location("RopeKernel"),
            "trueno-gpu/src/kernels/rope.rs"
        );
        assert_eq!(
            source_location("IncrementalAttentionKernel"),
            "trueno-gpu/src/kernels/attention/mod.rs"
        );
        assert_eq!(
            source_location("ResidualAddKernel"),
            "trueno-gpu/src/kernels/elementwise/residual.rs"
        );
        assert_eq!(
            source_location("SwigluKernel"),
            "trueno-gpu/src/kernels/activation.rs"
        );
    }

    #[test]
    fn test_source_location_batched_variants() {
        assert_eq!(
            source_location("BatchedVectorizedRmsNormKernel"),
            source_location("VectorizedRmsNormKernel")
        );
        assert_eq!(
            source_location("BatchedQ4KGemvKernel"),
            source_location("Q4KGemvKernel")
        );
        assert_eq!(
            source_location("BatchedRopeKernel"),
            source_location("RopeKernel")
        );
        assert_eq!(
            source_location("BatchedResidualAddKernel"),
            source_location("ResidualAddKernel")
        );
        assert_eq!(
            source_location("BatchedSwigluKernel"),
            source_location("SwigluKernel")
        );
    }

    #[test]
    fn test_source_location_unknown() {
        assert_eq!(source_location("FakeKernel"), "unknown");
    }

    #[test]
    fn test_build_decode_sequence_7b() {
        let info = ModelInfo {
            name: "test-7b".to_string(),
            quant: "Q4_K".to_string(),
            num_layers: 28,
            hidden_dim: 3584,
            intermediate_dim: 18944,
            num_heads: 28,
            num_kv_heads: 4,
            head_dim: 128,
        };
        let steps = build_decode_sequence(&info);
        assert_eq!(steps.len(), 12);
        assert_eq!(steps[0].name, "VectorizedRmsNormKernel");
        assert_eq!(steps[1].name, "Q4KGemvKernel");
        assert_eq!(steps[1].role, "QKV proj");
        assert_eq!(steps[11].name, "ResidualAddKernel");
        assert_eq!(steps[11].role, "post-FFN residual");
    }

    #[test]
    fn test_build_prefill_sequence_7b() {
        let info = ModelInfo {
            name: "test-7b".to_string(),
            quant: "Q4_K".to_string(),
            num_layers: 28,
            hidden_dim: 3584,
            intermediate_dim: 18944,
            num_heads: 28,
            num_kv_heads: 4,
            head_dim: 128,
        };
        let steps = build_prefill_sequence(&info);
        assert_eq!(steps.len(), 12);
        assert_eq!(steps[0].name, "BatchedVectorizedRmsNormKernel");
        assert!(steps[0].is_batched);
        assert_eq!(steps[1].name, "BatchedQ4KGemvKernel");
        // AttentionKernel is not batched (uses causal mask directly)
        assert!(!steps[3].is_batched);
    }

    #[test]
    fn test_format_shared() {
        assert_eq!(format_shared(0), "0");
        assert_eq!(format_shared(256), "256B");
        assert_eq!(format_shared(1024), "1KB");
        assert_eq!(format_shared(8192), "8KB");
    }

    #[test]
    fn test_truncate_shape() {
        assert_eq!(truncate_shape("3584 -> 3584", 20), "3584 -> 3584");
        assert_eq!(
            truncate_shape("this is a very long shape string", 20),
            "this is a very lo..."
        );
    }

    #[test]
    fn test_decode_sequence_shapes_use_model_dims() {
        let info = ModelInfo {
            name: "test".to_string(),
            quant: "Q4_K".to_string(),
            num_layers: 28,
            hidden_dim: 3584,
            intermediate_dim: 18944,
            num_heads: 28,
            num_kv_heads: 4,
            head_dim: 128,
        };
        let steps = build_decode_sequence(&info);
        // Gate proj: hidden -> intermediate
        assert!(steps[7].shape.contains("3584"));
        assert!(steps[7].shape.contains("18944"));
        // Down proj: intermediate -> hidden
        assert!(steps[10].shape.contains("18944"));
        assert!(steps[10].shape.contains("3584"));
    }

    #[test]
    fn test_reverse_lookup_finds_multiple_steps() {
        let info = ModelInfo {
            name: "test".to_string(),
            quant: "Q4_K".to_string(),
            num_layers: 28,
            hidden_dim: 3584,
            intermediate_dim: 18944,
            num_heads: 28,
            num_kv_heads: 4,
            head_dim: 128,
        };
        let steps = build_decode_sequence(&info);
        let matching: Vec<&KernelStep> = steps
            .iter()
            .filter(|s| s.name.to_lowercase().contains("q4kgemv"))
            .collect();
        // Q4KGemv appears 5 times: QKV, out, gate, up, down
        assert_eq!(matching.len(), 5);
    }

    #[test]
    fn test_1_5b_model_dimensions() {
        let info = ModelInfo {
            name: "test-1.5b".to_string(),
            quant: "Q4_K".to_string(),
            num_layers: 28,
            hidden_dim: 1536,
            intermediate_dim: 8960,
            num_heads: 12,
            num_kv_heads: 2,
            head_dim: 128,
        };
        let steps = build_decode_sequence(&info);
        assert_eq!(steps.len(), 12);
        // QKV output: 12*128 + 2*2*128 = 1536 + 512 = 2048
        assert!(steps[1].shape.contains("2048"));
    }
}
