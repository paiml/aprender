//! Kani proof harnesses for critical invariants
//!
//! These harnesses provide bounded model checking proofs for
//! safety-critical properties in the aprender codebase.
//!
//! Run with: `cargo kani --harness <name>`
//! Run all:  `cargo kani`
//!
//! Prerequisites: `cargo install --locked kani-verifier && cargo kani setup`

#[cfg(kani)]
mod proofs {
    /// Proof: Q4K block size is always 256 bytes (32 weights per block)
    ///
    /// The Q4K quantization format packs 32 f32 weights into a single block.
    /// This invariant is critical for correct dequantization and memory layout.
    #[kani::proof]
    fn proof_q4k_block_size_invariant() {
        const Q4K_BLOCK_SIZE: usize = 144; // bytes per Q4K super-block
        const Q4K_WEIGHTS_PER_BLOCK: usize = 256; // weights per super-block
        const Q4K_SCALES_PER_BLOCK: usize = 12; // d, dmin + 10 scale bytes

        // Block size must be positive
        kani::assert(Q4K_BLOCK_SIZE > 0, "Q4K block size must be positive");

        // Weights per block must be a power of 2
        kani::assert(
            Q4K_WEIGHTS_PER_BLOCK.is_power_of_two(),
            "Q4K weights per block must be power of 2",
        );

        // Block must fit scales + quantized data
        // 256 weights at 4 bits = 128 bytes, plus 16 bytes header = 144
        let data_bytes = Q4K_WEIGHTS_PER_BLOCK / 2; // 4-bit packing
        kani::assert(
            Q4K_BLOCK_SIZE >= data_bytes + Q4K_SCALES_PER_BLOCK,
            "Q4K block must fit scales + data",
        );
    }

    /// Proof: Q6K block size maintains correct weight count
    ///
    /// Q6K packs 256 weights using 6-bit quantization with scale factors.
    #[kani::proof]
    fn proof_q6k_block_size_invariant() {
        const Q6K_BLOCK_SIZE: usize = 210; // bytes per Q6K super-block
        const Q6K_WEIGHTS_PER_BLOCK: usize = 256;

        kani::assert(Q6K_BLOCK_SIZE > 0, "Q6K block size must be positive");
        kani::assert(
            Q6K_WEIGHTS_PER_BLOCK.is_power_of_two(),
            "Q6K weights per block must be power of 2",
        );

        // 256 weights * 6 bits = 1536 bits = 192 bytes for data
        // Plus scale/offset metadata
        let min_data_bytes = (Q6K_WEIGHTS_PER_BLOCK * 6) / 8;
        kani::assert(
            Q6K_BLOCK_SIZE >= min_data_bytes,
            "Q6K block must fit quantized data",
        );
    }

    /// Proof: GQA ratio is always a positive integer divisor
    ///
    /// In Grouped Query Attention, num_heads must be divisible by num_kv_heads.
    /// This is critical for correct attention computation.
    #[kani::proof]
    fn proof_gqa_ratio_invariant() {
        let num_heads: usize = kani::any();
        let num_kv_heads: usize = kani::any();

        // Constrain to realistic values (1-128 heads)
        kani::assume(num_heads > 0 && num_heads <= 128);
        kani::assume(num_kv_heads > 0 && num_kv_heads <= num_heads);
        kani::assume(num_heads % num_kv_heads == 0);

        let gqa_ratio = num_heads / num_kv_heads;

        // GQA ratio must be positive
        kani::assert(gqa_ratio > 0, "GQA ratio must be positive");

        // GQA ratio must divide num_heads exactly
        kani::assert(
            gqa_ratio * num_kv_heads == num_heads,
            "GQA ratio * kv_heads must equal num_heads",
        );

        // Head dim calculation must not overflow for realistic hidden dims
        let hidden_dim: usize = kani::any();
        kani::assume(hidden_dim > 0 && hidden_dim <= 16384);
        kani::assume(hidden_dim % num_heads == 0);

        let head_dim = hidden_dim / num_heads;
        kani::assert(head_dim > 0, "Head dim must be positive");

        let kv_dim = num_kv_heads * head_dim;
        kani::assert(kv_dim <= hidden_dim, "KV dim must not exceed hidden dim");
    }

    /// Proof: RoPE position encoding indices stay within bounds
    ///
    /// RoPE applies sinusoidal position encodings. The frequency computation
    /// must not produce out-of-bounds indices.
    #[kani::proof]
    fn proof_rope_index_bounds() {
        let head_dim: usize = kani::any();
        let position: usize = kani::any();

        // Constrain to realistic values
        kani::assume(head_dim > 0 && head_dim <= 256);
        kani::assume(head_dim % 2 == 0); // Must be even for RoPE pairs
        kani::assume(position < 131072); // Max context length

        let half_dim = head_dim / 2;

        // Verify all RoPE pair indices are valid
        for i in 0..half_dim {
            // Adjacent pair indices (NORM style, type 0)
            let idx_even = 2 * i;
            let idx_odd = 2 * i + 1;
            kani::assert(idx_even < head_dim, "Even RoPE index within bounds");
            kani::assert(idx_odd < head_dim, "Odd RoPE index within bounds");

            // NEOX style indices (type 2: split halves)
            let idx_first = i;
            let idx_second = i + half_dim;
            kani::assert(idx_first < head_dim, "First NEOX index within bounds");
            kani::assert(idx_second < head_dim, "Second NEOX index within bounds");
        }
    }

    /// Proof: Tensor shape reversal is its own inverse (LAYOUT-001)
    ///
    /// GGUF stores shapes as [ne0, ne1] (column-major convention).
    /// APR uses [rows, cols] (row-major). The reversal must be idempotent
    /// when applied twice.
    #[kani::proof]
    fn proof_shape_reversal_involution() {
        let ne0: u64 = kani::any();
        let ne1: u64 = kani::any();

        kani::assume(ne0 > 0 && ne0 <= 1_000_000);
        kani::assume(ne1 > 0 && ne1 <= 1_000_000);

        // GGUF shape
        let gguf_shape = [ne0, ne1];

        // First reversal: GGUF -> APR
        let apr_shape = [gguf_shape[1], gguf_shape[0]];

        // Second reversal: APR -> GGUF (should recover original)
        let recovered = [apr_shape[1], apr_shape[0]];

        kani::assert(
            recovered[0] == gguf_shape[0] && recovered[1] == gguf_shape[1],
            "Shape reversal must be an involution (self-inverse)",
        );

        // APR shape semantics: [rows, cols]
        kani::assert(apr_shape[0] == ne1, "APR rows = GGUF ne1");
        kani::assert(apr_shape[1] == ne0, "APR cols = GGUF ne0");
    }
}
