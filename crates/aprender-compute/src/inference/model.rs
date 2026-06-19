//! Llama-family transformer model for inference.
//!
//! Composes trueno primitives (rms_norm, Q4K matmul, fused attention)
//! into a complete transformer that loads GGUF weights and generates text.

use crate::backends::q4k::matmul_q4k_f32_dispatch;
use crate::blis::attention::fused_attention_decode;
use crate::blis::norms::rms_norm;
use crate::error::TruenoError;
use crate::inference::gguf::{GgmlType, GgufFile};

/// Model hyperparameters extracted from GGUF metadata.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_seq_len: usize,
    pub arch: String,
}

impl ModelConfig {
    /// Extract config from GGUF metadata.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, TruenoError> {
        let arch = gguf.meta_str("general.architecture").unwrap_or("llama").to_string();
        let prefix = &arch; // e.g., "llama" or "qwen2"

        let hidden_size = gguf
            .meta_u32(&format!("{prefix}.embedding_length"))
            .ok_or_else(|| TruenoError::InvalidInput("Missing embedding_length in GGUF".into()))?
            as usize;

        let num_heads = gguf
            .meta_u32(&format!("{prefix}.attention.head_count"))
            .ok_or_else(|| TruenoError::InvalidInput("Missing head_count in GGUF".into()))?
            as usize;

        let num_kv_heads = gguf
            .meta_u32(&format!("{prefix}.attention.head_count_kv"))
            .unwrap_or(num_heads as u32) as usize;

        let num_layers = gguf
            .meta_u32(&format!("{prefix}.block_count"))
            .ok_or_else(|| TruenoError::InvalidInput("Missing block_count in GGUF".into()))?
            as usize;

        let intermediate_size =
            gguf.meta_u32(&format!("{prefix}.feed_forward_length")).ok_or_else(|| {
                TruenoError::InvalidInput("Missing feed_forward_length in GGUF".into())
            })? as usize;

        let head_dim = hidden_size / num_heads;

        let vocab_size = gguf
            .meta_u32("tokenizer.ggml.vocab_size")
            .or_else(|| {
                // Fallback: count tokens array
                gguf.metadata.get("tokenizer.ggml.tokens").and_then(|v| {
                    if let crate::inference::gguf::MetadataValue::Array(arr) = v {
                        Some(arr.len() as u32)
                    } else {
                        None
                    }
                })
            })
            .unwrap_or(32000) as usize;

        let rms_norm_eps =
            gguf.meta_f32(&format!("{prefix}.attention.layer_norm_rms_epsilon")).unwrap_or(1e-5);

        let rope_theta = gguf.meta_f32(&format!("{prefix}.rope.freq_base")).unwrap_or(10000.0);

        let max_seq_len =
            gguf.meta_u32(&format!("{prefix}.context_length")).unwrap_or(2048) as usize;

        Ok(Self {
            hidden_size,
            intermediate_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            vocab_size,
            rms_norm_eps,
            rope_theta,
            max_seq_len,
            arch,
        })
    }
}

/// A weight matrix that may be Q4K (bytes) or any-other-quant dequantized to F32.
pub enum WeightMatrix {
    /// Raw Q4K bytes — use matmul_q4k_f32_dispatch
    Q4K { data: Vec<u8>, rows: usize },
    /// Dequantized F32 — use scalar dot-product
    F32 { data: Vec<f32>, rows: usize },
}

impl WeightMatrix {
    pub fn rows(&self) -> usize {
        match self {
            WeightMatrix::Q4K { rows, .. } => *rows,
            WeightMatrix::F32 { rows, .. } => *rows,
        }
    }
}

/// Weights for a single transformer layer.
pub struct LayerWeights {
    // Attention
    pub attn_norm: Vec<f32>,
    pub q_weight: WeightMatrix,
    pub k_weight: WeightMatrix,
    pub v_weight: WeightMatrix,
    pub o_weight: WeightMatrix,
    // Qwen2/Qwen3 biases (None for LLaMA)
    pub q_bias: Option<Vec<f32>>,
    pub k_bias: Option<Vec<f32>>,
    pub v_bias: Option<Vec<f32>>,

    // FFN
    pub ffn_norm: Vec<f32>,
    pub gate_weight: WeightMatrix,
    pub up_weight: WeightMatrix,
    pub down_weight: WeightMatrix,
}

/// Full model weights.
pub struct ModelWeights {
    pub token_embd: Vec<f32>,  // [vocab_size, hidden_size]
    pub output_norm: Vec<f32>, // [hidden_size]
    pub output_weight: WeightMatrix,
    pub layers: Vec<LayerWeights>,
}

/// Pre-allocated scratch buffers for the forward pass.
/// Eliminates all per-token heap allocations (FALSIFY-ARENA-001).
/// Contract: contracts/cgp/cgp-inference-arena-v1.yaml
pub struct ForwardArena {
    pub attn_input: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub attn_proj: Vec<f32>,
    pub ffn_input: Vec<f32>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub swiglu: Vec<f32>,
    pub ffn_out: Vec<f32>,
    pub hidden: Vec<f32>,
    pub residual: Vec<f32>,
    pub normed: Vec<f32>,
    pub k_cache_head: Vec<f32>,
    pub v_cache_head: Vec<f32>,
}

impl ForwardArena {
    pub fn new(config: &ModelConfig) -> Self {
        let kv_dim = config.num_kv_heads * config.head_dim;
        let head_cache_size = config.max_seq_len * config.head_dim;
        Self {
            attn_input: vec![0.0f32; config.hidden_size],
            q: vec![0.0f32; config.hidden_size],
            k: vec![0.0f32; kv_dim],
            v: vec![0.0f32; kv_dim],
            attn_out: vec![0.0f32; config.hidden_size],
            attn_proj: vec![0.0f32; config.hidden_size],
            ffn_input: vec![0.0f32; config.hidden_size],
            gate: vec![0.0f32; config.intermediate_size],
            up: vec![0.0f32; config.intermediate_size],
            swiglu: vec![0.0f32; config.intermediate_size],
            ffn_out: vec![0.0f32; config.hidden_size],
            hidden: vec![0.0f32; config.hidden_size],
            residual: vec![0.0f32; config.hidden_size],
            normed: vec![0.0f32; config.hidden_size],
            k_cache_head: vec![0.0f32; head_cache_size],
            v_cache_head: vec![0.0f32; head_cache_size],
        }
    }
}

/// KV cache for incremental decoding.
pub struct KvCache {
    /// k_cache[layer][pos * head_dim * num_kv_heads .. ] — flat per-layer
    pub k: Vec<Vec<f32>>,
    /// v_cache[layer][pos * head_dim * num_kv_heads .. ]
    pub v: Vec<Vec<f32>>,
    pub seq_len: usize,
}

impl KvCache {
    pub fn new(config: &ModelConfig) -> Self {
        let kv_dim = config.num_kv_heads * config.head_dim;
        let layer_size = config.max_seq_len * kv_dim;
        Self {
            k: (0..config.num_layers).map(|_| vec![0.0f32; layer_size]).collect(),
            v: (0..config.num_layers).map(|_| vec![0.0f32; layer_size]).collect(),
            seq_len: 0,
        }
    }
}

/// Complete transformer model ready for inference.
pub struct LlamaModel {
    pub config: ModelConfig,
    pub weights: ModelWeights,
}

impl LlamaModel {
    /// Load model from a GGUF file.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, TruenoError> {
        let config = ModelConfig::from_gguf(gguf)?;

        eprintln!(
            "Loading {} model: {}L × {}H ({}h {}kv) × {}I, vocab={}",
            config.arch,
            config.num_layers,
            config.hidden_size,
            config.num_heads,
            config.num_kv_heads,
            config.intermediate_size,
            config.vocab_size,
        );

        let weights = load_weights(gguf, &config)?;

        Ok(Self { config, weights })
    }

    /// Run one forward pass for a single token at the given position.
    /// Returns logits [vocab_size].
    /// Uses ForwardArena for zero per-token allocations (FALSIFY-ARENA-001).
    pub fn forward(
        &self,
        token_id: u32,
        pos: usize,
        kv_cache: &mut KvCache,
        arena: &mut ForwardArena,
    ) -> Result<Vec<f32>, TruenoError> {
        let cfg = &self.config;
        let w = &self.weights;

        // Token embedding lookup (copy into arena.hidden)
        let embd_start = token_id as usize * cfg.hidden_size;
        let embd_end = embd_start + cfg.hidden_size;
        if embd_end > w.token_embd.len() {
            return Err(TruenoError::InvalidInput(format!(
                "Token ID {token_id} out of range (vocab={})",
                cfg.vocab_size
            )));
        }
        arena.hidden[..cfg.hidden_size].copy_from_slice(&w.token_embd[embd_start..embd_end]);

        // Transformer layers
        for (layer_idx, lw) in w.layers.iter().enumerate() {
            self.forward_layer(layer_idx, lw, pos, kv_cache, arena)?;
        }

        // Final RMS norm
        rms_norm(
            &arena.hidden[..cfg.hidden_size],
            &w.output_norm,
            cfg.rms_norm_eps,
            &mut arena.normed[..cfg.hidden_size],
        )?;

        // Output projection → logits (this one allocation is unavoidable — vocab_size is large)
        let logits =
            matmul_weight(&w.output_weight, &arena.normed[..cfg.hidden_size], cfg.hidden_size);

        Ok(logits)
    }

    /// Forward one layer, reading/writing arena.hidden in-place.
    fn forward_layer(
        &self,
        layer_idx: usize,
        lw: &LayerWeights,
        pos: usize,
        kv_cache: &mut KvCache,
        arena: &mut ForwardArena,
    ) -> Result<(), TruenoError> {
        let cfg = &self.config;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let h_sz = cfg.hidden_size;

        // === Attention block ===
        rms_norm(
            &arena.hidden[..h_sz],
            &lw.attn_norm,
            cfg.rms_norm_eps,
            &mut arena.attn_input[..h_sz],
        )?;

        // QKV projections into arena buffers
        matmul_weight_into(&lw.q_weight, &arena.attn_input[..h_sz], h_sz, &mut arena.q[..h_sz]);
        matmul_weight_into(&lw.k_weight, &arena.attn_input[..h_sz], h_sz, &mut arena.k[..kv_dim]);
        matmul_weight_into(&lw.v_weight, &arena.attn_input[..h_sz], h_sz, &mut arena.v[..kv_dim]);

        // Optional bias (Qwen2/Qwen3)
        if let Some(bias) = &lw.q_bias {
            for (v, b) in arena.q[..h_sz].iter_mut().zip(bias.iter()) {
                *v += b;
            }
        }
        if let Some(bias) = &lw.k_bias {
            for (v, b) in arena.k[..kv_dim].iter_mut().zip(bias.iter()) {
                *v += b;
            }
        }
        if let Some(bias) = &lw.v_bias {
            for (v, b) in arena.v[..kv_dim].iter_mut().zip(bias.iter()) {
                *v += b;
            }
        }

        // RoPE in-place
        apply_rope(&mut arena.q[..h_sz], cfg.num_heads, cfg.head_dim, pos, cfg.rope_theta);
        apply_rope(&mut arena.k[..kv_dim], cfg.num_kv_heads, cfg.head_dim, pos, cfg.rope_theta);

        // Store K,V in cache
        let kv_off = pos * kv_dim;
        kv_cache.k[layer_idx][kv_off..kv_off + kv_dim].copy_from_slice(&arena.k[..kv_dim]);
        kv_cache.v[layer_idx][kv_off..kv_off + kv_dim].copy_from_slice(&arena.v[..kv_dim]);

        let seq_len = pos + 1;

        // Multi-head attention
        arena.attn_out[..h_sz].fill(0.0);
        let heads_per_kv = cfg.num_heads / cfg.num_kv_heads;

        for h in 0..cfg.num_heads {
            let kv_h = h / heads_per_kv;
            let q_head = &arena.q[h * cfg.head_dim..(h + 1) * cfg.head_dim];

            // Build contiguous K/V view for this head (reuse arena buffers)
            let view_len = seq_len * cfg.head_dim;
            for s in 0..seq_len {
                let src_off = s * kv_dim + kv_h * cfg.head_dim;
                let dst_off = s * cfg.head_dim;
                arena.k_cache_head[dst_off..dst_off + cfg.head_dim]
                    .copy_from_slice(&kv_cache.k[layer_idx][src_off..src_off + cfg.head_dim]);
                arena.v_cache_head[dst_off..dst_off + cfg.head_dim]
                    .copy_from_slice(&kv_cache.v[layer_idx][src_off..src_off + cfg.head_dim]);
            }

            let out_head = &mut arena.attn_out[h * cfg.head_dim..(h + 1) * cfg.head_dim];
            fused_attention_decode(
                q_head,
                &arena.k_cache_head[..view_len],
                &arena.v_cache_head[..view_len],
                cfg.head_dim,
                seq_len,
                out_head,
            );
        }

        // Output projection
        matmul_weight_into(
            &lw.o_weight,
            &arena.attn_out[..h_sz],
            h_sz,
            &mut arena.attn_proj[..h_sz],
        );

        // Residual: residual = hidden + attn_proj
        for i in 0..h_sz {
            arena.residual[i] = arena.hidden[i] + arena.attn_proj[i];
        }

        // === FFN block ===
        rms_norm(
            &arena.residual[..h_sz],
            &lw.ffn_norm,
            cfg.rms_norm_eps,
            &mut arena.ffn_input[..h_sz],
        )?;

        let i_sz = cfg.intermediate_size;
        matmul_weight_into(
            &lw.gate_weight,
            &arena.ffn_input[..h_sz],
            h_sz,
            &mut arena.gate[..i_sz],
        );
        matmul_weight_into(&lw.up_weight, &arena.ffn_input[..h_sz], h_sz, &mut arena.up[..i_sz]);

        // SiLU(gate) * up → swiglu
        for i in 0..i_sz {
            let g = arena.gate[i];
            let silu_g = g / (1.0 + (-g).exp());
            arena.swiglu[i] = silu_g * arena.up[i];
        }

        // Down projection
        matmul_weight_into(
            &lw.down_weight,
            &arena.swiglu[..i_sz],
            i_sz,
            &mut arena.ffn_out[..h_sz],
        );

        // Residual → hidden (for next layer)
        for i in 0..h_sz {
            arena.hidden[i] = arena.residual[i] + arena.ffn_out[i];
        }

        Ok(())
    }
}

/// Apply Rotary Position Embedding (RoPE) in-place.
fn apply_rope(x: &mut [f32], num_heads: usize, head_dim: usize, pos: usize, theta: f32) {
    for h in 0..num_heads {
        let head = &mut x[h * head_dim..(h + 1) * head_dim];
        for i in (0..head_dim).step_by(2) {
            let freq = 1.0 / theta.powf(i as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            let (sin_a, cos_a) = angle.sin_cos();
            let x0 = head[i];
            let x1 = head[i + 1];
            head[i] = x0 * cos_a - x1 * sin_a;
            head[i + 1] = x0 * sin_a + x1 * cos_a;
        }
    }
}

/// Load all weights from GGUF into model weight structs.
fn load_weights(gguf: &GgufFile, config: &ModelConfig) -> Result<ModelWeights, TruenoError> {
    // Token embeddings — may be F32, F16, or quantized (Q4K/Q6K).
    // For quantized embeddings, dequantize the full table at load time
    // since we need random-access per-token lookup.
    let token_embd = load_f32_or_dequant_tensor(
        gguf,
        "token_embd.weight",
        config.vocab_size * config.hidden_size,
    )?;

    // Output norm
    let output_norm = load_f32_tensor(gguf, "output_norm.weight", config.hidden_size)?;

    // Output projection — Q4K kept as bytes; everything else dequantized to F32.
    // Falls back to tied embeddings if output.weight not present.
    let output_weight = if gguf.tensor_info("output.weight").is_some() {
        load_weight_matrix(gguf, "output.weight", config.hidden_size)?
    } else {
        // Tied embeddings
        WeightMatrix::F32 { data: token_embd.clone(), rows: config.vocab_size }
    };

    // Layers
    let mut layers = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let prefix = format!("blk.{i}");

        let attn_norm =
            load_f32_tensor(gguf, &format!("{prefix}.attn_norm.weight"), config.hidden_size)?;
        let ffn_norm =
            load_f32_tensor(gguf, &format!("{prefix}.ffn_norm.weight"), config.hidden_size)?;

        let q_weight =
            load_weight_matrix(gguf, &format!("{prefix}.attn_q.weight"), config.hidden_size)?;
        let k_weight =
            load_weight_matrix(gguf, &format!("{prefix}.attn_k.weight"), config.hidden_size)?;
        let v_weight =
            load_weight_matrix(gguf, &format!("{prefix}.attn_v.weight"), config.hidden_size)?;
        let o_weight =
            load_weight_matrix(gguf, &format!("{prefix}.attn_output.weight"), config.hidden_size)?;

        // Qwen2/Qwen3 attention biases (optional — LLaMA has none)
        let kv_dim = config.num_kv_heads * config.head_dim;
        let q_bias = load_optional_f32(gguf, &format!("{prefix}.attn_q.bias"), config.hidden_size);
        let k_bias = load_optional_f32(gguf, &format!("{prefix}.attn_k.bias"), kv_dim);
        let v_bias = load_optional_f32(gguf, &format!("{prefix}.attn_v.bias"), kv_dim);

        let gate_weight =
            load_weight_matrix(gguf, &format!("{prefix}.ffn_gate.weight"), config.hidden_size)?;
        let up_weight =
            load_weight_matrix(gguf, &format!("{prefix}.ffn_up.weight"), config.hidden_size)?;
        let down_weight = load_weight_matrix(
            gguf,
            &format!("{prefix}.ffn_down.weight"),
            config.intermediate_size,
        )?;

        if i == 0 {
            eprintln!(
                "  Layer 0: Q[{}×{}] K[{}×{}] V[{}×{}] Gate[{}×{}]",
                q_weight.rows(),
                config.hidden_size,
                k_weight.rows(),
                config.hidden_size,
                v_weight.rows(),
                config.hidden_size,
                gate_weight.rows(),
                config.hidden_size,
            );
        }

        layers.push(LayerWeights {
            attn_norm,
            q_weight,
            k_weight,
            v_weight,
            o_weight,
            q_bias,
            k_bias,
            v_bias,
            ffn_norm,
            gate_weight,
            up_weight,
            down_weight,
        });
    }

    eprintln!("  Loaded {} layers", layers.len());

    Ok(ModelWeights { token_embd, output_norm, output_weight, layers })
}

/// Load a tensor as F32, dequantizing if quantized.
/// For Q4K weights, uses trueno's dequantize_q4k_to_f32.
fn load_f32_or_dequant_tensor(
    gguf: &GgufFile,
    name: &str,
    expected_elements: usize,
) -> Result<Vec<f32>, TruenoError> {
    let info = gguf
        .tensor_info(name)
        .ok_or_else(|| TruenoError::InvalidInput(format!("Missing tensor: {name}")))?;
    let data = gguf
        .tensor_data(name)
        .ok_or_else(|| TruenoError::InvalidInput(format!("Missing tensor data: {name}")))?;

    match info.dtype {
        GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16 => {
            Ok(to_f32_from_any(data, info.dtype, expected_elements))
        }
        GgmlType::Q4K => {
            let n_elements = info.n_elements() as usize;
            Ok(crate::backends::q4k::dequantize_q4k_to_f32(data, n_elements))
        }
        GgmlType::Q6K => Ok(dequantize_q6k_to_f32(data, info.n_elements() as usize)),
        GgmlType::Q5K => Ok(dequantize_q5k_to_f32(data, info.n_elements() as usize)),
        GgmlType::Q8_0 => Ok(dequantize_q8_0_to_f32(data, info.n_elements() as usize)),
        GgmlType::Q4_0 => Ok(dequantize_q4_0_to_f32(data, info.n_elements() as usize)),
        GgmlType::Q4_1 => Ok(dequantize_q4_1_to_f32(data, info.n_elements() as usize)),
        _ => {
            eprintln!(
                "  WARNING: tensor '{name}' has unsupported dtype {:?}, using zeros",
                info.dtype
            );
            Ok(vec![0.0f32; expected_elements])
        }
    }
}

/// Load an optional F32 tensor (returns None if tensor doesn't exist in GGUF).
fn load_optional_f32(gguf: &GgufFile, name: &str, expected_elements: usize) -> Option<Vec<f32>> {
    let info = gguf.tensor_info(name)?;
    let data = gguf.tensor_data(name)?;
    Some(to_f32_from_any(data, info.dtype, expected_elements))
}

/// Load a tensor as F32 (dequantizing F16 if needed).
fn load_f32_tensor(
    gguf: &GgufFile,
    name: &str,
    expected_elements: usize,
) -> Result<Vec<f32>, TruenoError> {
    let info = gguf
        .tensor_info(name)
        .ok_or_else(|| TruenoError::InvalidInput(format!("Missing tensor: {name}")))?;
    let data = gguf
        .tensor_data(name)
        .ok_or_else(|| TruenoError::InvalidInput(format!("Missing tensor data: {name}")))?;

    Ok(to_f32_from_any(data, info.dtype, expected_elements))
}

/// Load a weight tensor as a `WeightMatrix`.
/// Q4K weights are kept as raw bytes for the fused matmul kernel.
/// All other quantization types are dequantized to F32 at load time.
fn load_weight_matrix(
    gguf: &GgufFile,
    name: &str,
    in_dim: usize,
) -> Result<WeightMatrix, TruenoError> {
    let info = gguf
        .tensor_info(name)
        .ok_or_else(|| TruenoError::InvalidInput(format!("Missing tensor: {name}")))?;
    let data = gguf
        .tensor_data(name)
        .ok_or_else(|| TruenoError::InvalidInput(format!("Missing tensor data: {name}")))?;

    let n_elements = info.n_elements() as usize;
    let out_dim = n_elements / in_dim;

    match info.dtype {
        GgmlType::Q4K => Ok(WeightMatrix::Q4K { data: data.to_vec(), rows: out_dim }),
        GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16 => {
            let f32_data = to_f32_from_any(data, info.dtype, n_elements);
            Ok(WeightMatrix::F32 { data: f32_data, rows: out_dim })
        }
        GgmlType::Q6K => {
            let f32_data = dequantize_q6k_to_f32(data, n_elements);
            Ok(WeightMatrix::F32 { data: f32_data, rows: out_dim })
        }
        GgmlType::Q5K => {
            let f32_data = dequantize_q5k_to_f32(data, n_elements);
            Ok(WeightMatrix::F32 { data: f32_data, rows: out_dim })
        }
        GgmlType::Q8_0 => {
            let f32_data = dequantize_q8_0_to_f32(data, n_elements);
            Ok(WeightMatrix::F32 { data: f32_data, rows: out_dim })
        }
        GgmlType::Q4_0 => {
            let f32_data = dequantize_q4_0_to_f32(data, n_elements);
            Ok(WeightMatrix::F32 { data: f32_data, rows: out_dim })
        }
        GgmlType::Q4_1 => {
            let f32_data = dequantize_q4_1_to_f32(data, n_elements);
            Ok(WeightMatrix::F32 { data: f32_data, rows: out_dim })
        }
        _ => {
            eprintln!("  WARNING: tensor '{name}' dtype {:?} unsupported, using zeros", info.dtype);
            Ok(WeightMatrix::F32 { data: vec![0.0f32; n_elements], rows: out_dim })
        }
    }
}

/// Dispatch matrix-vector multiply based on weight type (allocating).
/// Used only for output projection (vocab_size too large for arena).
fn matmul_weight(weight: &WeightMatrix, input: &[f32], in_dim: usize) -> Vec<f32> {
    match weight {
        WeightMatrix::Q4K { data, rows } => matmul_q4k_f32_dispatch(data, input, *rows, in_dim),
        WeightMatrix::F32 { data, rows } => {
            let mut out = vec![0.0f32; *rows];
            for i in 0..*rows {
                let row = &data[i * in_dim..(i + 1) * in_dim];
                out[i] = row.iter().zip(input.iter()).map(|(a, b)| a * b).sum();
            }
            out
        }
    }
}

/// Dispatch matrix-vector multiply into a pre-allocated output buffer (zero-alloc).
fn matmul_weight_into(weight: &WeightMatrix, input: &[f32], in_dim: usize, out: &mut [f32]) {
    match weight {
        WeightMatrix::Q4K { data, rows } => {
            let result = matmul_q4k_f32_dispatch(data, input, *rows, in_dim);
            out[..*rows].copy_from_slice(&result);
        }
        WeightMatrix::F32 { data, rows } => {
            for i in 0..*rows {
                let row = &data[i * in_dim..(i + 1) * in_dim];
                out[i] = row.iter().zip(input.iter()).map(|(a, b)| a * b).sum();
            }
        }
    }
}

/// Convert IEEE 754 half-precision (FP16) bits to f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign); // ±0
        }
        // Denormalized: convert to normalized f32
        let mut m = mant;
        let mut e: i32 = -14;
        while m & 0x400 == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        let f32_exp = ((e + 127) as u32) << 23;
        return f32::from_bits(sign | f32_exp | (m << 13));
    }
    if exp == 31 {
        // Inf/NaN
        return f32::from_bits(sign | 0x7F80_0000 | (mant << 13));
    }
    let f32_exp = (exp + 112) << 23; // rebias: -15 + 127 = 112
    f32::from_bits(sign | f32_exp | (mant << 13))
}

/// Convert tensor bytes to f32, handling F32, F16, BF16.
fn to_f32_from_any(data: &[u8], dtype: GgmlType, n_elements: usize) -> Vec<f32> {
    match dtype {
        GgmlType::F32 => {
            // Safe: read f32 values from aligned-or-unaligned bytes
            let count = n_elements.min(data.len() / 4);
            (0..count)
                .map(|i| {
                    let off = i * 4;
                    f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
                })
                .collect()
        }
        GgmlType::F16 => {
            let count = n_elements.min(data.len() / 2);
            (0..count)
                .map(|i| {
                    let off = i * 2;
                    let bits = u16::from_le_bytes([data[off], data[off + 1]]);
                    f16_to_f32(bits)
                })
                .collect()
        }
        GgmlType::Bf16 => {
            let count = n_elements.min(data.len() / 2);
            (0..count)
                .map(|i| {
                    let off = i * 2;
                    let bits = u16::from_le_bytes([data[off], data[off + 1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect()
        }
        _ => {
            // For quantized norms (shouldn't happen), return zeros
            vec![0.0f32; n_elements]
        }
    }
}

/// Dequantize Q6_K to F32.
///
/// Q6_K layout per 256-element super-block (210 bytes):
/// - ql[128]: lower 4 bits of each 6-bit value (2 values per byte)
/// - qh[64]:  upper 2 bits (4 values per byte, 2 bits each)
/// - scales[16]: signed 8-bit scales for 16 groups of 16
/// - d[2]: f16 global scale
///
/// Value = d * scale[group] * q6  where q6 = (low4 | high2<<4) as i8 - 32
fn dequantize_q6k_to_f32(data: &[u8], num_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 210;

    let num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let mut result = vec![0.0f32; num_blocks * BLOCK_SIZE];

    for sb in 0..num_blocks {
        let sb_start = sb * BLOCK_BYTES;
        if sb_start + BLOCK_BYTES > data.len() {
            break;
        }
        let block = &data[sb_start..sb_start + BLOCK_BYTES];
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));

        let out_base = sb * BLOCK_SIZE;
        for group in 0..16usize {
            let scale = (scales[group] as i8) as f32;
            let group_off = group * 16;
            for j in 0..16usize {
                let idx = group_off + j;
                let ql_byte = ql[idx / 2];
                let low4 = if idx % 2 == 0 { ql_byte & 0x0F } else { ql_byte >> 4 };
                let qh_byte = qh[idx / 4];
                let high2 = (qh_byte >> ((idx % 4) * 2)) & 0x03;
                let q6 = ((low4 | (high2 << 4)) as i8).wrapping_sub(32) as f32;
                result[out_base + idx] = d * scale * q6;
            }
        }
    }

    result.truncate(num_elements);
    result
}

/// Dequantize Q5_K to F32.
///
/// Q5_K layout per 256-element super-block (176 bytes):
/// - d[2]:       f16 super-block scale
/// - dmin[2]:    f16 super-block min scale
/// - scales[12]: packed 6-bit scales and mins (8 sub-blocks × 2 values)
/// - qh[32]:     high bit of each 5-bit value (1 bit per element = 32 bytes)
/// - qs[128]:    lower 4 bits per element (2 per byte)
///
/// Value = d * scale * q5 - dmin * min
fn dequantize_q5k_to_f32(data: &[u8], num_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 176;

    let num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let mut result = vec![0.0f32; num_blocks * BLOCK_SIZE];

    for sb in 0..num_blocks {
        let sb_start = sb * BLOCK_BYTES;
        if sb_start + BLOCK_BYTES > data.len() {
            break;
        }
        let block = &data[sb_start..sb_start + BLOCK_BYTES];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));

        // Unpack 6-bit scales and mins (same layout as Q4K)
        let sc = &block[4..16];
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];
        for i in 0..4 {
            scales[i] = sc[i] & 0x3F;
            mins[i] = sc[i + 4] & 0x3F;
            scales[i + 4] = (sc[i + 8] & 0x0F) | ((sc[i] >> 6) << 4);
            mins[i + 4] = (sc[i + 8] >> 4) | ((sc[i + 4] >> 6) << 4);
        }

        let qh = &block[16..48];
        let qs = &block[48..176];

        let out_base = sb * BLOCK_SIZE;
        // llama.cpp `dequantize_row_q5_K` layout: process the 256-element
        // super-block in four 64-element chunks. For each chunk `it`, the low
        // and high nibble of the SAME `qs` byte land a STRIDE OF 32 apart, and
        // `qh[l]` holds the 5th bit across the four chunks (u1 = 1<<(2*it),
        // u2 = 2<<(2*it)). See contracts/q5k-dequant-correctness-v1.yaml.
        for it in 0..4usize {
            let ql_base = it * 32;
            let scale_lo = d * scales[2 * it] as f32;
            let min_lo = dmin * mins[2 * it] as f32;
            let scale_hi = d * scales[2 * it + 1] as f32;
            let min_hi = dmin * mins[2 * it + 1] as f32;
            let u1 = 1u8 << (2 * it);
            let u2 = 2u8 << (2 * it);
            for l in 0..32usize {
                let qsb = qs[ql_base + l];
                let qhb = qh[l];
                let lo = (qsb & 0x0F) + if qhb & u1 != 0 { 16 } else { 0 };
                result[out_base + it * 64 + l] = scale_lo * (lo as f32) - min_lo;
                let hi = (qsb >> 4) + if qhb & u2 != 0 { 16 } else { 0 };
                result[out_base + it * 64 + 32 + l] = scale_hi * (hi as f32) - min_hi;
            }
        }
    }

    result.truncate(num_elements);
    result
}

/// Dequantize Q8_0 to F32.
///
/// Q8_0 layout per 32-element block (34 bytes):
/// - d[2]:   f16 block scale
/// - qs[32]: signed 8-bit quantized values
///
/// Value = d * qs[i]
fn dequantize_q8_0_to_f32(data: &[u8], num_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 34;

    let num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let mut result = vec![0.0f32; num_blocks * BLOCK_SIZE];

    for b in 0..num_blocks {
        let b_start = b * BLOCK_BYTES;
        if b_start + BLOCK_BYTES > data.len() {
            break;
        }
        let block = &data[b_start..b_start + BLOCK_BYTES];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let out_base = b * BLOCK_SIZE;
        for j in 0..BLOCK_SIZE {
            result[out_base + j] = d * (block[2 + j] as i8) as f32;
        }
    }

    result.truncate(num_elements);
    result
}

/// Dequantize Q4_0 to F32.
///
/// Q4_0 layout per 32-element block (18 bytes):
/// - d[2]:   f16 block scale
/// - qs[16]: 4-bit quantized values, 2 per byte
///
/// Value = d * (q4 - 8)  where q4 ∈ 0..15 (centered: subtract 8)
fn dequantize_q4_0_to_f32(data: &[u8], num_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 18;

    let num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let mut result = vec![0.0f32; num_blocks * BLOCK_SIZE];

    for b in 0..num_blocks {
        let b_start = b * BLOCK_BYTES;
        if b_start + BLOCK_BYTES > data.len() {
            break;
        }
        let block = &data[b_start..b_start + BLOCK_BYTES];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let out_base = b * BLOCK_SIZE;
        for j in 0..16 {
            let byte = block[2 + j];
            let lo = (byte & 0x0F) as i32 - 8;
            let hi = ((byte >> 4) & 0x0F) as i32 - 8;
            result[out_base + j * 2] = d * lo as f32;
            result[out_base + j * 2 + 1] = d * hi as f32;
        }
    }

    result.truncate(num_elements);
    result
}

/// Dequantize Q4_1 to F32.
///
/// Q4_1 layout per 32-element block (20 bytes):
/// - d[2]:   f16 scale
/// - m[2]:   f16 min (additive offset)
/// - qs[16]: 4-bit quantized values, 2 per byte
///
/// Value = d * q4 + m  where q4 ∈ 0..15
fn dequantize_q4_1_to_f32(data: &[u8], num_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 20;

    let num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let mut result = vec![0.0f32; num_blocks * BLOCK_SIZE];

    for b in 0..num_blocks {
        let b_start = b * BLOCK_BYTES;
        if b_start + BLOCK_BYTES > data.len() {
            break;
        }
        let block = &data[b_start..b_start + BLOCK_BYTES];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let m = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let out_base = b * BLOCK_SIZE;
        for j in 0..16 {
            let byte = block[4 + j];
            let lo = (byte & 0x0F) as f32;
            let hi = ((byte >> 4) & 0x0F) as f32;
            result[out_base + j * 2] = d * lo + m;
            result[out_base + j * 2 + 1] = d * hi + m;
        }
    }

    result.truncate(num_elements);
    result
}

#[cfg(test)]
mod q5k_dequant_tests {
    use super::dequantize_q5k_to_f32;

    /// Build a single 176-byte Q5_K super-block.
    ///
    /// `d` / `dmin` are f16 bit patterns. `scale_bytes` is the 12-byte packed
    /// scales+mins. `qh` is 32 bytes (one bit per element). `qs` is 128 bytes
    /// (two 4-bit values per byte).
    fn build_q5k_block(
        d: u16,
        dmin: u16,
        scale_bytes: [u8; 12],
        qh: [u8; 32],
        qs: [u8; 128],
    ) -> Vec<u8> {
        let mut block = Vec::with_capacity(176);
        block.extend_from_slice(&d.to_le_bytes());
        block.extend_from_slice(&dmin.to_le_bytes());
        block.extend_from_slice(&scale_bytes);
        block.extend_from_slice(&qh);
        block.extend_from_slice(&qs);
        assert_eq!(block.len(), 176, "Q5_K super-block must be 176 bytes");
        block
    }

    /// Scale bytes that decode (via the Q4_K-style 6-bit unpack) to all eight
    /// scales == 1 and all eight mins == 0.
    fn scales_all_one_mins_zero() -> [u8; 12] {
        // scales[0..4] low6 = bytes[0..4]; mins[0..4] low6 = bytes[4..8];
        // scales[4..8] = (bytes[8..12] & 0x0F) | ((bytes[0..4] >> 6) << 4);
        // mins[4..8]   = (bytes[8..12] >> 4)   | ((bytes[4..8] >> 6) << 4).
        // bytes[0..4]=1 -> scales[0..4]=1, high-2 contribution 0.
        // bytes[4..8]=0 -> mins[0..4]=0.
        // bytes[8..12]=1 -> scales[4..8]=1, mins[4..8]=0.
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
    }

    /// FALSIFIER (PMAT-842): the GGUF/llama.cpp Q5_K layout places the low and
    /// high nibble of the SAME `qs` byte a STRIDE OF 32 apart, NOT at adjacent
    /// output indices. With qs[0] = 0x21, low nibble (1) -> out[0], high nibble
    /// (2) -> out[32]; out[1] comes from qs[1] (0). The buggy sequential-index
    /// implementation produced out[1] == 2.0 and out[32] == 0.0.
    #[test]
    fn test_q5k_dequant_stride32_nibble_layout() {
        let d = 0x3C00; // f16 1.0
        let dmin = 0x0000; // f16 0.0
        let mut qs = [0u8; 128];
        qs[0] = 0x21; // low nibble = 1, high nibble = 2
        let block = build_q5k_block(d, dmin, scales_all_one_mins_zero(), [0u8; 32], qs);

        let out = dequantize_q5k_to_f32(&block, 256);

        // low nibble of qs[0] -> output index 0
        assert_eq!(out[0], 1.0, "out[0] must come from qs[0] low nibble (=1)");
        // output index 1 comes from qs[1] (=0), NOT qs[0] high nibble
        assert_eq!(out[1], 0.0, "out[1] must come from qs[1] low nibble (=0)");
        // high nibble of qs[0] -> output index 32 (stride of 32)
        assert_eq!(out[32], 2.0, "out[32] must come from qs[0] high nibble (=2)");
    }

    /// FALSIFIER (PMAT-842): the 5th bit lives in `qh[l]`, with the low nibble
    /// using bit (2*it) and the high nibble using bit (2*it+1). For the first
    /// chunk (it=0): u1 = 0b01, u2 = 0b10. With qh[0] = 0x03 both bits are set,
    /// so the low value gains +16 (out[0]=17) and the high value gains +16
    /// (out[32]=18).
    #[test]
    fn test_q5k_dequant_fifth_bit_placement() {
        let d = 0x3C00; // f16 1.0
        let dmin = 0x0000; // f16 0.0
        let mut qs = [0u8; 128];
        qs[0] = 0x21; // low nibble = 1, high nibble = 2
        let mut qh = [0u8; 32];
        qh[0] = 0x03; // set bits 0 (u1) and 1 (u2) for chunk it=0
        let block = build_q5k_block(d, dmin, scales_all_one_mins_zero(), qh, qs);

        let out = dequantize_q5k_to_f32(&block, 256);

        // low nibble: 1 + 16 (bit 0 set) = 17
        assert_eq!(out[0], 17.0, "out[0] low nibble must add 16 for qh bit 0");
        // high nibble: 2 + 16 (bit 1 set) = 18, at stride-32 index
        assert_eq!(out[32], 18.0, "out[32] high nibble must add 16 for qh bit 1");
    }
}
