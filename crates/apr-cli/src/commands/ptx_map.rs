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

/// A quantization label that was **measured**, never guessed.
///
/// dogfood-0.63.0, issue #2444: `apr ptx-map` read the quantization out of the
/// FILE NAME. Renaming a Q4_K file to `totally-not-Q8_0.gguf` made it report
/// `Q8_0` for the same bytes, and a name carrying no quant token fell through
/// to a hardcoded `"Q4_K"` default — which is what the shipped fixture
/// (lowercase `q4_k_m`, matched case-sensitively) always hit.
///
/// The field is now a type whose only constructor takes a GGUF qtype id read
/// out of a tensor header, so a `String` derived from a path cannot reach it.
mod measured_quant {
    /// Quantization named from the qtype stored in the model's own tensors.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) struct MeasuredQuant(&'static str);

    impl MeasuredQuant {
        /// Reported when no tensor carries a qtype this build can name.
        /// Honest absence — never a stand-in for a plausible default.
        pub(crate) const UNKNOWN: Self = Self("unknown");

        /// The only way to name a quantization: from a qtype id read out of
        /// the file. Delegates to realizar's table (ground truth for the
        /// kernel that runs) rather than GGUF's advisory `general.file_type`.
        #[allow(unused_variables)]
        pub(crate) fn from_qtype(qtype: u32) -> Option<Self> {
            #[cfg(feature = "inference")]
            {
                realizar::api::gguf_qtype_name(qtype).map(Self)
            }
            #[cfg(not(feature = "inference"))]
            {
                None
            }
        }

        pub(crate) fn as_str(self) -> &'static str {
            self.0
        }
    }

    impl std::fmt::Display for MeasuredQuant {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(self.0)
        }
    }
}

use measured_quant::MeasuredQuant;

/// Model dimensions extracted from GGUF config
struct ModelInfo {
    name: String,
    quant: MeasuredQuant,
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

/// Source location table for kernel types.
///
/// Every entry is the file that actually defines `pub struct <kernel_name>`, and
/// `source_paths_resolve_to_the_defining_file` proves it against the working
/// tree. Before that test existed the whole table pointed into a `trueno-gpu/`
/// directory that the APR-MONO consolidation deleted, and three of the six leaf
/// paths were wrong on top of that — the Source column named files a reader
/// could not open (dogfood-0.63.0, issue #2399 finding 3).
fn source_location(kernel_name: &str) -> &'static str {
    match kernel_name {
        "VectorizedRmsNormKernel" => "crates/aprender-gpu/src/kernels/layernorm/rmsnorm.rs",
        "BatchedVectorizedRmsNormKernel" => "crates/aprender-gpu/src/kernels/layernorm/batched.rs",
        "Q4KGemvKernel" => "crates/aprender-gpu/src/kernels/quantize/q4k/basic.rs",
        "BatchedQ4KGemvKernel" => "crates/aprender-gpu/src/kernels/quantize/q4k/batched.rs",
        "TensorCoreQ4KGemmKernel" => {
            "crates/aprender-gpu/src/kernels/quantize/fp16_tensor/tensor_core_gemm.rs"
        }
        "Q6KGemvKernel" => "crates/aprender-gpu/src/kernels/quantize/q6k/gemv.rs",
        "BatchedQ6KGemvKernel" => "crates/aprender-gpu/src/kernels/quantize/q6k/batched.rs",
        "RopeKernel" => "crates/aprender-gpu/src/kernels/elementwise/rope/standard.rs",
        "BatchedRopeKernel" => "crates/aprender-gpu/src/kernels/elementwise/rope/batched.rs",
        "IncrementalAttentionKernel" => {
            "crates/aprender-gpu/src/kernels/attention/paged/incremental.rs"
        }
        "AttentionKernel" => "crates/aprender-gpu/src/kernels/attention/flash/mod.rs",
        "ResidualAddKernel" | "BatchedResidualAddKernel" => {
            "crates/aprender-gpu/src/kernels/elementwise/residual.rs"
        }
        "FusedSwigluKernel" | "BatchedSwigluKernel" => {
            "crates/aprender-gpu/src/kernels/elementwise/swiglu.rs"
        }
        "KvCacheScatterKernel" => "crates/aprender-gpu/src/kernels/elementwise/kv_cache.rs",
        "ArgMaxKernel" => "crates/aprender-gpu/src/kernels/argmax.rs",
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
            // "SwigluKernel" was printed here for the decode path, but no such
            // kernel exists — the decode SwiGLU is `FusedSwigluKernel`
            // (crates/aprender-gpu/src/kernels/elementwise/swiglu.rs). #2399.
            name: "FusedSwigluKernel",
            role: "SwiGLU",
            shape: format!("{inter} -> {inter}"),
            source: source_location("FusedSwigluKernel"),
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

    // #2399 finding 2: ptx-map claims to show what will actually run. For an
    // architecture the inference path refuses to instantiate, nothing runs — so
    // printing a dense-transformer sequence (RoPE + GQA + SwiGLU) with an exact
    // launch count is fabrication. Ask realizar the same question `apr check`
    // asks, and refuse with the same words.
    if let Some(reason) = realizar::gguf::unsupported_architecture_reason(
        &config.architecture,
        mapped.model.tensors.iter().map(|t| t.name.as_str()),
    ) {
        return Err(CliError::ValidationFailed(format!(
            "{reason} (ptx-map maps the kernels that would run; this model has no runnable kernel path)"
        )));
    }

    // The file name is a LABEL, never a measurement: it names the row, and
    // nothing else (issue #2444).
    let filename = model_path
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or("unknown");

    let quant = dominant_weight_quant(mapped.model.tensors.iter().map(|t| (t.dims.len(), t.qtype)))
        .unwrap_or(MeasuredQuant::UNKNOWN);

    // Every attention number below is COPIED from the config realizar parsed
    // out of the file. `num_kv_heads` used to be reconstructed from a table
    // keyed on `num_heads` (28 → 4, 12 → 2, else → num_heads), which reported
    // 14 KV heads for Qwen2.5-0.5B (2 in metadata) and 32 for Qwen3-8B (8) —
    // and propagated into the printed QKV shape. `head_dim` used to be
    // recomputed as hidden_dim/num_heads, which ignores an explicit
    // `attention.key_length` (Qwen3-0.6B: 1024/16 = 64, actual 128).
    Ok(ModelInfo {
        name: filename.to_string(),
        quant,
        num_layers: config.num_layers,
        hidden_dim: config.hidden_dim as u32,
        intermediate_dim: config.intermediate_dim as u32,
        num_heads: config.num_heads as u32,
        num_kv_heads: config.num_kv_heads as u32,
        head_dim: config.head_dim() as u32,
    })
}

/// The quantization the model ACTUALLY stores, from the modal qtype across its
/// matmul weights (`n_dims >= 2` — the tensors a GEMV kernel is launched for).
///
/// Takes `(n_dims, qtype)` pairs and nothing else: the file name is not an
/// argument, so it cannot influence the answer. Ties break to the lowest qtype
/// id so the label is deterministic for a given file.
fn dominant_weight_quant<I: Iterator<Item = (usize, u32)>>(tensors: I) -> Option<MeasuredQuant> {
    let mut counts: std::collections::BTreeMap<u32, usize> = std::collections::BTreeMap::new();
    for (n_dims, qtype) in tensors {
        if n_dims >= 2 {
            *counts.entry(qtype).or_insert(0) += 1;
        }
    }
    counts
        .into_iter()
        .filter_map(|(qtype, n)| MeasuredQuant::from_qtype(qtype).map(|q| (q, n)))
        // max_by_key returns the LAST maximum; iterate descending so the
        // lowest qtype id wins a tie.
        .rev()
        .max_by_key(|&(_, n)| n)
        .map(|(quant, _)| quant)
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
