
// =============================================================================
// Helper Functions for Creating Quantized Data
// =============================================================================

/// Create valid Q4_0 data for a tensor with given dimensions
/// Q4_0: 18 bytes per 32 elements (2 f16 scale + 16 bytes quants)
#[must_use]
pub fn create_q4_0_data(num_elements: usize) -> Vec<u8> {
    let num_blocks = num_elements.div_ceil(32);
    let mut data = Vec::with_capacity(num_blocks * 18);

    for _ in 0..num_blocks {
        // f16 scale = 0.1
        let scale = half::f16::from_f32(0.1);
        data.extend_from_slice(&scale.to_le_bytes());
        // 16 bytes of quants (mid-range values)
        data.extend([0x88u8; 16]);
    }

    data
}

/// Create valid Q8_0 data for a tensor with given dimensions
/// Q8_0: 34 bytes per 32 elements (2 f16 scale + 32 i8 quants)
#[must_use]
pub fn create_q8_0_data(num_elements: usize) -> Vec<u8> {
    let num_blocks = num_elements.div_ceil(32);
    let mut data = Vec::with_capacity(num_blocks * 34);

    for _ in 0..num_blocks {
        // f16 scale = 0.1
        let scale = half::f16::from_f32(0.1);
        data.extend_from_slice(&scale.to_le_bytes());
        // 32 i8 quants (zeros)
        data.extend([0i8 as u8; 32]);
    }

    data
}

/// Create valid Q4_K data for a tensor with given dimensions
/// Q4_K: 144 bytes per 256 elements (QK_K)
///
/// For 2D tensors, uses row-padded layout to match transformer loader expectations.
/// Each row is padded independently to a multiple of QK_K=256 elements.
#[must_use]
pub fn create_q4_k_data(num_elements: usize) -> Vec<u8> {
    let num_super_blocks = num_elements.div_ceil(256);
    vec![0u8; num_super_blocks * 144]
}

/// Create valid Q4_K data for a 2D tensor with row-padded layout
/// This matches the layout expected by `QuantizedGGUFTransformer::get_tensor_ref`
#[must_use]
pub fn create_q4_k_data_2d(rows: usize, cols: usize) -> Vec<u8> {
    const QK_K: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 144;
    let super_blocks_per_row = cols.div_ceil(QK_K);
    vec![0u8; rows * super_blocks_per_row * SUPER_BLOCK_BYTES]
}

/// Create valid Q5_K data for a tensor with given dimensions
/// Q5_K: 176 bytes per 256 elements
#[must_use]
pub fn create_q5_k_data(num_elements: usize) -> Vec<u8> {
    let num_super_blocks = num_elements.div_ceil(256);
    vec![0u8; num_super_blocks * 176]
}

/// Create valid Q6_K data for a tensor with given dimensions
/// Q6_K: 210 bytes per 256 elements
#[must_use]
pub fn create_q6_k_data(num_elements: usize) -> Vec<u8> {
    let num_super_blocks = num_elements.div_ceil(256);
    vec![0u8; num_super_blocks * 210]
}

/// Create valid Q2_K data for a tensor with given dimensions
/// Q2_K: 84 bytes per 256 elements
#[must_use]
pub fn create_q2_k_data(num_elements: usize) -> Vec<u8> {
    let num_super_blocks = num_elements.div_ceil(256);
    vec![0u8; num_super_blocks * 84]
}

/// Create valid F16 data for a tensor with given dimensions
/// F16: 2 bytes per element
#[must_use]
pub fn create_f16_data(num_elements: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(num_elements * 2);
    for i in 0..num_elements {
        let val = half::f16::from_f32((i as f32) * 0.01);
        data.extend_from_slice(&val.to_le_bytes());
    }
    data
}

/// Create valid Q4_1 data for a tensor with given dimensions
/// Q4_1: 20 bytes per 32 elements (2 scale + 2 min + 16 quants)
#[must_use]
pub fn create_q4_1_data(num_elements: usize) -> Vec<u8> {
    let num_blocks = num_elements.div_ceil(32);
    let mut data = Vec::with_capacity(num_blocks * 20);
    for _ in 0..num_blocks {
        let scale = half::f16::from_f32(0.1);
        data.extend_from_slice(&scale.to_le_bytes());
        let min = half::f16::from_f32(0.0);
        data.extend_from_slice(&min.to_le_bytes());
        data.extend([0x88u8; 16]);
    }
    data
}

/// Create valid Q5_0 data for a tensor with given dimensions
/// Q5_0: 22 bytes per 32 elements (2 scale + 4 high bits + 16 quants)
#[must_use]
pub fn create_q5_0_data(num_elements: usize) -> Vec<u8> {
    let num_blocks = num_elements.div_ceil(32);
    let mut data = Vec::with_capacity(num_blocks * 22);
    for _ in 0..num_blocks {
        let scale = half::f16::from_f32(0.1);
        data.extend_from_slice(&scale.to_le_bytes());
        data.extend([0u8; 4]); // high bits
        data.extend([0x88u8; 16]); // quants
    }
    data
}

/// Create valid Q5_1 data for a tensor with given dimensions
/// Q5_1: 24 bytes per 32 elements (2 scale + 2 min + 4 high bits + 16 quants)
#[must_use]
pub fn create_q5_1_data(num_elements: usize) -> Vec<u8> {
    let num_blocks = num_elements.div_ceil(32);
    let mut data = Vec::with_capacity(num_blocks * 24);
    for _ in 0..num_blocks {
        let scale = half::f16::from_f32(0.1);
        data.extend_from_slice(&scale.to_le_bytes());
        let min = half::f16::from_f32(0.0);
        data.extend_from_slice(&min.to_le_bytes());
        data.extend([0u8; 4]); // high bits
        data.extend([0x88u8; 16]); // quants
    }
    data
}

/// Create F32 embedding data (small random-ish values)
#[must_use]
pub fn create_f32_embedding_data(vocab_size: usize, hidden_dim: usize) -> Vec<f32> {
    let mut data = Vec::with_capacity(vocab_size * hidden_dim);
    for i in 0..(vocab_size * hidden_dim) {
        // Pseudo-random but deterministic values
        let val = ((i % 1000) as f32 - 500.0) / 5000.0;
        data.push(val);
    }
    data
}

/// Create F32 norm weights (typically ~1.0)
#[must_use]
pub fn create_f32_norm_weights(dim: usize) -> Vec<f32> {
    vec![1.0f32; dim]
}

// =============================================================================
// Complete Model Builder
// =============================================================================

/// Build a minimal valid LLaMA-style GGUF model
///
/// This creates a complete model with:
/// - Token embeddings (F32)
/// - One transformer layer with Q4_K weights
/// - Output norm (F32)
/// - LM head (tied to token embeddings)
#[must_use]
pub fn build_minimal_llama_gguf(
    vocab_size: usize,
    hidden_dim: usize,
    intermediate_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
) -> Vec<u8> {
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;

    // Create tensor data
    let embed_data = create_f32_embedding_data(vocab_size, hidden_dim);
    let norm_data = create_f32_norm_weights(hidden_dim);

    // Q4_K weights for layer 0 - use row-padded layout for 2D tensors
    let q_data = create_q4_k_data_2d(hidden_dim, hidden_dim);
    let k_data = create_q4_k_data_2d(hidden_dim, kv_dim);
    let v_data = create_q4_k_data_2d(hidden_dim, kv_dim);
    let attn_out_data = create_q4_k_data_2d(hidden_dim, hidden_dim);
    let ffn_up_data = create_q4_k_data_2d(hidden_dim, intermediate_dim);
    let ffn_down_data = create_q4_k_data_2d(intermediate_dim, hidden_dim);
    let ffn_gate_data = create_q4_k_data_2d(hidden_dim, intermediate_dim);

    GGUFBuilder::new()
        // Metadata
        .architecture("llama")
        .hidden_dim("llama", hidden_dim as u32)
        .num_layers("llama", 1)
        .num_heads("llama", num_heads as u32)
        .num_kv_heads("llama", num_kv_heads as u32)
        .context_length("llama", 256)
        .rope_freq_base("llama", 10000.0)
        .rms_epsilon("llama", 1e-5)
        .ffn_hidden_dim("llama", intermediate_dim as u32)
        // Token embedding
        .add_f32_tensor(
            "token_embd.weight",
            &[vocab_size as u64, hidden_dim as u64],
            &embed_data,
        )
        // Layer 0 attention
        .add_f32_tensor("blk.0.attn_norm.weight", &[hidden_dim as u64], &norm_data)
        .add_q4_k_tensor(
            "blk.0.attn_q.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q_data,
        )
        .add_q4_k_tensor(
            "blk.0.attn_k.weight",
            &[hidden_dim as u64, kv_dim as u64],
            &k_data,
        )
        .add_q4_k_tensor(
            "blk.0.attn_v.weight",
            &[hidden_dim as u64, kv_dim as u64],
            &v_data,
        )
        .add_q4_k_tensor(
            "blk.0.attn_output.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &attn_out_data,
        )
        // Layer 0 FFN
        .add_f32_tensor("blk.0.ffn_norm.weight", &[hidden_dim as u64], &norm_data)
        .add_q4_k_tensor(
            "blk.0.ffn_up.weight",
            &[hidden_dim as u64, intermediate_dim as u64],
            &ffn_up_data,
        )
        .add_q4_k_tensor(
            "blk.0.ffn_down.weight",
            &[intermediate_dim as u64, hidden_dim as u64],
            &ffn_down_data,
        )
        .add_q4_k_tensor(
            "blk.0.ffn_gate.weight",
            &[hidden_dim as u64, intermediate_dim as u64],
            &ffn_gate_data,
        )
        // Output norm and head
        .add_f32_tensor("output_norm.weight", &[hidden_dim as u64], &norm_data)
        // Note: LM head often tied to token_embd, so we don't add output.weight
        // The loader will fallback to token_embd.weight
        .build()
}

/// Build an **executable** Pygmy GGUF model (Q4_0 quantized, minimal dimensions)
///
/// This function creates a GGUF model with Q4_0 quantized tensors that can
/// survive a full `forward_cached()` pass without crashing. The model produces
/// garbage output but exercises all code paths in the inference pipeline.
///
/// # Dr. Popper's "Active Pygmy" (T-COV-95)
///
/// This implements the "Minimum Viable Predictor" - a tiny model that:
/// 1. Has valid tensor layout (Q4_0 for weights, F32 for embeddings/norms)
/// 2. Has valid quantized weights (proper block structure)
/// 3. Has sufficient metadata for forward() graph
///
/// # Why Q4_0?
///
/// The fused matmul operations in OwnedQuantizedModel only support quantized
/// types (Q4_0/Q8_0/Q4_K/Q5_K/Q6_K), not F32. Q4_0 uses 32-element blocks,
/// which aligns perfectly with hidden_dim=32.
///
/// # Dimensions
///
/// - hidden_dim: 32 (aligns with Q4_0 block size)
/// - num_heads: 4
/// - num_kv_heads: 4 (MHA, no GQA for simplicity)
/// - vocab_size: 32 (small vocabulary, must be multiple of 32 for Q4_0)
/// - intermediate_dim: 64 (2x hidden for standard FFN ratio)
/// - context_length: 32
/// - num_layers: 1
#[must_use]
pub fn build_executable_pygmy_gguf() -> Vec<u8> {
    // Active Pygmy dimensions (T-COV-95)
    // All dimensions chosen to align with Q4_0's 32-element block size
    const VOCAB_SIZE: usize = 32;
    const HIDDEN_DIM: usize = 32;
    const INTERMEDIATE_DIM: usize = 64;
    const NUM_HEADS: usize = 4;
    const NUM_KV_HEADS: usize = 4;
    const CONTEXT_LENGTH: usize = 32;

    // KV dimension = num_kv_heads * head_dim = 4 * 8 = 32
    let kv_dim = NUM_KV_HEADS * (HIDDEN_DIM / NUM_HEADS);

    // Create F32 tensor data for embeddings and norms
    // Embedding: small values that won't cause overflow
    let embed_data: Vec<f32> = (0..VOCAB_SIZE * HIDDEN_DIM)
        .map(|i| ((i % 100) as f32 - 50.0) / 1000.0)
        .collect();

    // Norm weights: 1.0 for identity-like behavior
    let norm_data: Vec<f32> = vec![1.0; HIDDEN_DIM];

    // Create Q4_0 quantized data for weight tensors
    // Q4_0: 18 bytes per 32 elements (2 byte f16 scale + 16 bytes quants)
    let q_data = create_q4_0_data(HIDDEN_DIM * HIDDEN_DIM); // 32x32 = 1024 elements
    let k_data = create_q4_0_data(HIDDEN_DIM * kv_dim); // 32x32 = 1024 elements
    let v_data = create_q4_0_data(HIDDEN_DIM * kv_dim); // 32x32 = 1024 elements
    let attn_out_data = create_q4_0_data(HIDDEN_DIM * HIDDEN_DIM); // 32x32 = 1024 elements

    // FFN weights: Q4_0 quantized
    let ffn_gate_data = create_q4_0_data(HIDDEN_DIM * INTERMEDIATE_DIM); // 32x64 = 2048 elements
    let ffn_up_data = create_q4_0_data(HIDDEN_DIM * INTERMEDIATE_DIM); // 32x64 = 2048 elements
    let ffn_down_data = create_q4_0_data(INTERMEDIATE_DIM * HIDDEN_DIM); // 64x32 = 2048 elements

    // LM head weight (output projection): Q4_0 quantized
    // This is separate from token_embd to avoid the "type 0" fallback
    let lm_head_data = create_q4_0_data(HIDDEN_DIM * VOCAB_SIZE); // 32x32 = 1024 elements

    GGUFBuilder::new()
        // Metadata - full LLaMA config
        .architecture("llama")
        .hidden_dim("llama", HIDDEN_DIM as u32)
        .num_layers("llama", 1)
        .num_heads("llama", NUM_HEADS as u32)
        .num_kv_heads("llama", NUM_KV_HEADS as u32)
        .context_length("llama", CONTEXT_LENGTH as u32)
        .rope_freq_base("llama", 10000.0)
        .rms_epsilon("llama", 1e-5)
        .ffn_hidden_dim("llama", INTERMEDIATE_DIM as u32)
        // Token embedding - F32 (lookup table, not matmul)
        .add_f32_tensor(
            "token_embd.weight",
            &[VOCAB_SIZE as u64, HIDDEN_DIM as u64],
            &embed_data,
        )
        // Layer 0 attention norm - F32 (small, read once per token)
        .add_f32_tensor("blk.0.attn_norm.weight", &[HIDDEN_DIM as u64], &norm_data)
        // Layer 0 QKV projections - Q4_0 (large, fused matmul)
        .add_q4_0_tensor(
            "blk.0.attn_q.weight",
            &[HIDDEN_DIM as u64, HIDDEN_DIM as u64],
            &q_data,
        )
        .add_q4_0_tensor(
            "blk.0.attn_k.weight",
            &[HIDDEN_DIM as u64, kv_dim as u64],
            &k_data,
        )
        .add_q4_0_tensor(
            "blk.0.attn_v.weight",
            &[HIDDEN_DIM as u64, kv_dim as u64],
            &v_data,
        )
        .add_q4_0_tensor(
            "blk.0.attn_output.weight",
            &[HIDDEN_DIM as u64, HIDDEN_DIM as u64],
            &attn_out_data,
        )
        // Layer 0 FFN norm - F32
        .add_f32_tensor("blk.0.ffn_norm.weight", &[HIDDEN_DIM as u64], &norm_data)
        // Layer 0 FFN projections - Q4_0
        .add_q4_0_tensor(
            "blk.0.ffn_gate.weight",
            &[HIDDEN_DIM as u64, INTERMEDIATE_DIM as u64],
            &ffn_gate_data,
        )
        .add_q4_0_tensor(
            "blk.0.ffn_up.weight",
            &[HIDDEN_DIM as u64, INTERMEDIATE_DIM as u64],
            &ffn_up_data,
        )
        .add_q4_0_tensor(
            "blk.0.ffn_down.weight",
            &[INTERMEDIATE_DIM as u64, HIDDEN_DIM as u64],
            &ffn_down_data,
        )
        // Output norm - F32
        .add_f32_tensor("output_norm.weight", &[HIDDEN_DIM as u64], &norm_data)
        // LM head (output projection) - Q4_0 quantized
        // Explicit output.weight to avoid fallback to F32 token_embd.weight
        .add_q4_0_tensor(
            "output.weight",
            &[HIDDEN_DIM as u64, VOCAB_SIZE as u64],
            &lm_head_data,
        )
        .build()
}
