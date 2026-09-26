//! CUDA Type Definitions for Weight Loading and Workspace Management
//!
//! This module contains types used for GPU weight management:
//! - `IndexedLayerWeights`: Pre-computed layer weight indices for O(1) lookup
//! - `WeightQuantType`: Quantization type detection and size calculation
//! - `TransformerWorkspace`: Pre-allocated workspace buffers

use trueno_gpu::driver::GpuBuffer;

/// PAR-043: Pre-computed layer weight indices for O(1) lookup
///
/// Eliminates per-layer string formatting and HashMap lookups during decode.
/// Each layer's weights are stored as raw device pointers for direct access.
///
/// Performance impact:
/// - Before: ~10-12ms overhead per token (string formatting + HashMap)
/// - After: ~0.1ms overhead per token (direct indexed access)
/// PMAT-232 CONTRACT: No Default — every field must be explicitly set from GGUF metadata.
#[derive(Debug, Clone)]
pub struct IndexedLayerWeights {
    /// Q projection weights device pointer (may be Q4K or Q5_0 quantized)
    pub attn_q_ptr: u64,
    /// Q projection weights size in bytes
    pub attn_q_len: usize,
    /// Q projection quantization type (Qwen 0.5B uses Q5_0)
    pub attn_q_qtype: WeightQuantType,
    /// K projection weights device pointer (may be Q4K or Q5_0 quantized)
    pub attn_k_ptr: u64,
    /// K projection weights size in bytes
    pub attn_k_len: usize,
    /// K projection quantization type (Qwen 0.5B uses Q5_0)
    pub attn_k_qtype: WeightQuantType,
    /// V projection weights device pointer (may be Q4K, Q6K, or Q8_0 quantized)
    pub attn_v_ptr: u64,
    /// V projection weights size in bytes
    pub attn_v_len: usize,
    /// V projection quantization type (needed because some models use Q6K/Q8_0 for V)
    pub attn_v_qtype: WeightQuantType,
    /// O projection weights device pointer (may be Q4K or Q4_0 quantized)
    pub attn_output_ptr: u64,
    /// O projection weights size in bytes
    pub attn_output_len: usize,
    /// O projection quantization type (PAR-058: Q4_0 models were broken)
    pub attn_output_qtype: WeightQuantType,
    /// FFN gate projection device pointer (may be Q4K or Q4_0 quantized)
    pub ffn_gate_ptr: u64,
    /// FFN gate projection size in bytes
    pub ffn_gate_len: usize,
    /// FFN gate projection quantization type (PAR-058: Q4_0 models were broken)
    pub ffn_gate_qtype: WeightQuantType,
    /// FFN up projection device pointer (may be Q4K or Q4_0 quantized)
    pub ffn_up_ptr: u64,
    /// FFN up projection size in bytes
    pub ffn_up_len: usize,
    /// FFN up projection quantization type (PAR-058: Q4_0 models were broken)
    pub ffn_up_qtype: WeightQuantType,
    /// FFN down projection device pointer (Q4K, Q6K, or Q4_0 quantized)
    pub ffn_down_ptr: u64,
    /// FFN down projection size in bytes
    pub ffn_down_len: usize,
    /// FFN down projection quantization type (some models use Q6K)
    pub ffn_down_qtype: WeightQuantType,
    /// Attention RMSNorm gamma device pointer (FP32)
    pub attn_norm_ptr: u64,
    /// Attention RMSNorm gamma size in elements
    pub attn_norm_len: usize,
    /// FFN RMSNorm gamma device pointer (FP32)
    pub ffn_norm_ptr: u64,
    /// FFN RMSNorm gamma size in elements
    pub ffn_norm_len: usize,
    /// Q projection bias device pointer (FP32, optional - 0 if no bias)
    pub attn_q_bias_ptr: u64,
    /// Q projection bias size in elements (0 if no bias)
    pub attn_q_bias_len: usize,
    /// K projection bias device pointer (FP32, optional - 0 if no bias)
    pub attn_k_bias_ptr: u64,
    /// K projection bias size in elements (0 if no bias)
    pub attn_k_bias_len: usize,
    /// V projection bias device pointer (FP32, optional - 0 if no bias)
    pub attn_v_bias_ptr: u64,
    /// V projection bias size in elements (0 if no bias)
    pub attn_v_bias_len: usize,
    /// GH-279: Per-head Q RMSNorm gamma device pointer (FP32, optional - 0 if no QkNorm)
    pub attn_q_norm_ptr: u64,
    /// Per-head Q RMSNorm gamma size in elements (0 if no QkNorm)
    pub attn_q_norm_len: usize,
    /// GH-279: Per-head K RMSNorm gamma device pointer (FP32, optional - 0 if no QkNorm)
    pub attn_k_norm_ptr: u64,
    /// Per-head K RMSNorm gamma size in elements (0 if no QkNorm)
    pub attn_k_norm_len: usize,
}

/// Weight quantization type for GGUF tensors
///
/// PMAT-232 CONTRACT: This enum MUST NOT derive Default. Every construction
/// must be explicit. Match statements MUST be exhaustive (no `_ =>` catch-all).
/// See contracts/tensor-layout-v1.yaml quant_dispatch section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightQuantType {
    /// Q4_K quantization (type 12) - 144 bytes per 256 elements
    Q4K,
    /// Q5_K quantization (type 13) - 176 bytes per 256 elements
    Q5K,
    /// Q6_K quantization (type 14) - 210 bytes per 256 elements
    Q6K,
    /// Q8_0 quantization (type 8) - 34 bytes per 32 elements
    Q8_0,
    /// Q5_0 quantization (type 6) - 22 bytes per 32 elements
    Q5_0,
    /// Q4_0 quantization (type 2) - 18 bytes per 32 elements
    Q4_0,
    /// Q4_1 quantization (type 3) - 20 bytes per 32 elements (2 f16 scale + 2 f16 min + 16 quants)
    /// PAR-058: Added to handle Qwen 0.5B which has FFN down in Q4_1 despite metadata
    Q4_1,
    /// F32 unquantized (type 0) - 4 bytes per element
    /// GH-374: APR checkpoints may have F32 LM head when source model was not quantized.
    /// Without this variant, F32 weights silently default to Q4K GEMV → garbage logits.
    F32,
    /// IQ4_XS (type 23) - 136 bytes per 256 elements, 4.25 bits/weight.
    ///
    /// #3477: `Qwen3.5-4B-UD-Q4_K_XL` stores `ffn_gate`/`ffn_up` in blk.12, 13,
    /// 16 … as IQ4_XS. Codebook-based: 4-bit indices into 16 non-linear IQ4_NL
    /// levels, with 6-bit per-sub-block scales split across `scales_l`/`scales_h`.
    IQ4XS,
    /// F16 unquantized (type 1) - 2 bytes per element.
    ///
    /// #3477 "no model left behind": `Qwen3.5-4B-UD-Q4_K_XL` stores `ssm_alpha`
    /// and `ssm_beta` as F16. Before this variant `from_ggml_type(1)` returned
    /// `None`, so the hybrid refused the whole model to CPU. Like F32 this is
    /// unquantized — no blocks, no scales, no codebook.
    F16,
    /// IQ4_NL (type 20) - 18 bytes per **32** elements, 4 bits/weight.
    ///
    /// #3869: the same 16 non-linear levels as [`Self::IQ4XS`], but a plain f16
    /// scale per 32-element block instead of split 6-bit sub-block scales.
    /// `Qwen2.5-0.5B-Instruct-IQ3_M` stores 96 tensors this way, `-IQ4_XS` 120.
    ///
    /// CAUTION: `block_iq4_nl` is the IDENTICAL C struct to `block_q4_0`
    /// (`{ ggml_half d; uint8_t qs[16]; }`), so a tensor's size can never tell
    /// them apart, and at 256-element granularity both also collide with Q4_K at
    /// 144 bytes. This type is therefore deliberately ABSENT from `from_size`'s
    /// inference ladder and must come from the declared ggml type id.
    IQ4NL,
    /// IQ3_S (type 21) - 110 bytes per 256 elements, 3.4375 bits/weight.
    ///
    /// #3884: 9-bit indices into a 512-entry grid (8 bits in `qs`, the 9th in
    /// `qh`), explicit sign bytes, and 4-bit scales each shared by TWO
    /// sub-blocks. `Qwen2.5-0.5B-Instruct-IQ3_M` carries 21 of these alongside
    /// the 96 IQ4_NL tensors, and they were the remaining GPU blocker after
    /// #3869 landed.
    ///
    /// Unlike [`Self::IQ4NL`] this size is UNAMBIGUOUS: 110 bytes per 256
    /// elements collides with nothing (144 is Q4_K/Q4_0/IQ4_NL, 176 is
    /// Q5_K/Q5_0), so it is safe in `from_size`'s inference ladder.
    IQ3S,
    /// IQ2_XXS (type 16) - 66 bytes per 256 elements, 2.0625 bits/weight.
    ///
    /// #3931: 95 of the 320 tensors in `Qwen3.5-0.8B-UD-IQ2_XXS`. Codebook-based:
    /// per 32-element sub-block, four 8-bit indices into 256 eight-magnitude
    /// grid entries, four 7-bit sign codes, and one 4-bit scale.
    ///
    /// 66 bytes per 256 elements collides with nothing in `from_size`'s ladder
    /// (210, 176, 144, 136, 110 per super-block; 18/20/22/24/34 per 32), so it is
    /// safe to infer from size.
    IQ2XXS,
    /// IQ2_S (type 22) - 82 bytes per 256 elements, 2.5625 bits/weight. #3953:
    /// the 5 ffn_down tensors of `Qwen3.5-0.8B-UD-IQ2_XXS`. 82 B/256 collides with
    /// no other format, so it is safe in `from_size`'s ladder.
    IQ2S,
    /// IQ3_XXS (type 18) - 98 bytes per 256 elements, 3.0625 bits/weight. #3963:
    /// the 24 attn tensors of `Qwen3.5-0.8B-UD-IQ2_XXS`. 98 B/256 collides with no
    /// other format, so it is safe in `from_size`'s ladder.
    IQ3XXS,
    /// Q2_K (type 10) - 84 bytes per 256 elements, affine 2-bit K-quant. #3960:
    /// the 3 ffn_down tensors of `Qwen3.5-0.8B-UD-IQ2_XXS`. 84 B/256 collides with
    /// no other format (checked by aprender-f5), so it is safe in `from_size`.
    Q2K,
    /// Q5_1 (type 7) - 24 bytes per 32 elements, AFFINE: `w = q * d + m`.
    ///
    /// #3885: the last blocker on `Qwen2.5-0.5B-Instruct-IQ4_XS`, which carries
    /// 24 Q5_1 tensors among 290. Legacy format, no codebook: a per-block scale
    /// AND a per-block min, with the quant's 5th bit living in `qh`.
    ///
    /// It is the only type here whose dequantization has an OFFSET term, so a
    /// kernel that drops `m` gives the right spread around the wrong centre.
    /// 192 bytes per 256 elements is unique, so it is safe in `from_size`.
    Q5_1,
    /// BF16 (type 30) - 2 bytes per element, the TOP HALF of an f32.
    ///
    /// #3908: `qwen2.5-coder-0.5b-instruct.apr` is 290 BF16 tensors + 1 F32, and
    /// the .apr CUDA loader had no type-30 path, so CUDA declined, wgpu failed,
    /// CPU ran, and the forced-GPU refusal (R-0b, #3002) returned rc=14. The
    /// refusal is correct; the missing kernel was the defect.
    ///
    /// Unquantized like [`Self::F32`] and [`Self::F16`] — no blocks, no scales,
    /// no codebook. It is the SIMPLEST type here: decoding is `f32::from_bits(
    /// bits << 16)`, an EXACT widening with no rounding, because a bfloat16 is
    /// literally the top 16 bits of an f32. So unlike every quantized variant
    /// above, its GPU kernel can be held to BIT-EXACT agreement with the CPU
    /// decoder rather than to a tolerance.
    ///
    /// Shares F16's size rule (2 bytes/element), which is why size alone can
    /// never distinguish BF16 from F16 — the declared ggml type id is the only
    /// discriminator, so it is deliberately ABSENT from `from_size`'s ladder.
    BF16,
}

impl WeightQuantType {
    /// Bytes per 256 elements for super-block quantization types
    pub const fn bytes_per_superblock(&self) -> usize {
        match self {
            // Q4K/Q4_0/IQ4_NL all land on 144 bytes per 256-element super-block
            // (Q4_0/IQ4_NL: 18 bytes/32-element block * 8 blocks = 144).
            Self::Q4K | Self::Q4_0 | Self::IQ4NL => 144,
            // Q5K/Q5_0 both land on 176 bytes per 256-element super-block
            // (Q5_0: 22 bytes/32-element block * 8 blocks = 176).
            Self::Q5K | Self::Q5_0 => 176,
            Self::Q6K => 210,
            Self::Q8_0 => 34 * 8, // Q8_0 uses 32-element blocks, so 8 blocks for 256 elements
            Self::Q4_1 => 20 * 8, // Q4_1 uses 32-element blocks, so 8 blocks for 256 elements
            Self::F32 => 256 * 4, // F32: 4 bytes per element, 256 elements
            // F16/BF16 both land on 512 bytes per 256-element super-block
            // (2 bytes per element, 256 elements).
            Self::F16 | Self::BF16 => 256 * 2,
            Self::IQ4XS => 136,   // IQ4_XS: 136 bytes per 256-element super-block
            Self::IQ3S => 110,    // IQ3_S: 110 bytes per 256-element super-block
            Self::IQ2XXS => 66,   // IQ2_XXS: 66 bytes per 256-element super-block
            Self::IQ2S => 82,     // IQ2_S: 82 bytes per 256-element super-block
            Self::IQ3XXS => 98,   // IQ3_XXS: 98 bytes per 256-element super-block
            Self::Q2K => 84,      // Q2_K: 84 bytes per 256-element super-block
            Self::Q5_1 => 24 * 8, // Q5_1 uses 32-element blocks, so 8 blocks for 256 elements
        }
    }

    /// Bytes per 32 elements (for block-based quantization types)
    pub const fn bytes_per_block(&self) -> usize {
        match self {
            // Q4K (super-block, treated as 18 per 32), Q4_0 and IQ4_NL (both
            // natively 32-element blocks) all land on 18 bytes per block.
            Self::Q4K | Self::Q4_0 | Self::IQ4NL => 18,
            // Q5K (super-block) and Q5_0 (natively 32-element) both land on 22.
            Self::Q5K | Self::Q5_0 => 22,
            Self::Q6K => 26, // Q6K is super-block (210/8 = 26.25, round to 26)
            Self::Q8_0 => 34,
            Self::Q4_1 => 20,
            Self::F32 => 128, // F32: 4 bytes per element, 32 elements
            // F16/BF16 both land on 64 (2 bytes per element, 32 elements).
            Self::F16 | Self::BF16 => 64,
            Self::IQ4XS => 17,  // IQ4_XS super-block: 136/8 = 17 per 32
            Self::IQ3S => 13,   // IQ3_S super-block: 110/8 = 13.75, truncated as Q6K's 210/8 is
            Self::IQ2XXS => 8,  // IQ2_XXS super-block: 66/8 = 8.25, truncated as IQ3S's is
            Self::IQ2S => 10,   // IQ2_S super-block: 82/8 = 10.25, truncated likewise
            Self::IQ3XXS => 12, // IQ3_XXS super-block: 98/8 = 12.25, truncated likewise
            Self::Q2K => 10,    // Q2_K super-block: 84/8 = 10.5, truncated likewise
            Self::Q5_1 => 24,   // Q5_1 is NATIVELY a 32-element block: exact
        }
    }

    /// Create from GGML type ID
    pub fn from_ggml_type(type_id: u32) -> Option<Self> {
        match type_id {
            0 => Some(Self::F32),     // GH-374: F32 LM head in APR checkpoints
            1 => Some(Self::F16),     // #3477: F16 ssm_alpha/ssm_beta in UD dynamic quants
            23 => Some(Self::IQ4XS),  // #3477: IQ4_XS ffn_gate/ffn_up in UD dynamic quants
            20 => Some(Self::IQ4NL),  // #3869: declared-type-only, see the variant docs
            21 => Some(Self::IQ3S),   // #3884: IQ3_S, the remaining IQ3_M blocker
            16 => Some(Self::IQ2XXS), // #3950: IQ2_XXS, 95/320 of Qwen3.5-0.8B-UD-IQ2_XXS
            22 => Some(Self::IQ2S),   // #3953: IQ2_S, 5/320 of Qwen3.5-0.8B-UD-IQ2_XXS
            18 => Some(Self::IQ3XXS), // #3963: IQ3_XXS, 24/320 of Qwen3.5-0.8B-UD-IQ2_XXS
            10 => Some(Self::Q2K),    // #3960: Q2_K, 3/320 of Qwen3.5-0.8B-UD-IQ2_XXS
            7 => Some(Self::Q5_1),    // #3885: Q5_1, the remaining IQ4_XS blocker
            30 => Some(Self::BF16),   // #3908: BF16, measured BIT-EXACT (0 ULP)
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1), // PAR-058: Q4_1 support
            6 => Some(Self::Q5_0),
            8 => Some(Self::Q8_0),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            _ => None,
        }
    }

    /// PAR-105-FIX: Check if a qtype matches the expected size for given dimensions
    /// Returns true if the qtype would produce the given byte size
    pub fn matches_size(&self, size_bytes: usize, n_rows: usize, n_cols: usize) -> bool {
        match self {
            // F32: 4 bytes per element
            Self::F32 => size_bytes == n_rows * n_cols * 4,
            // F16: 2 bytes per element
            // #3908: BF16 shares F16's size rule exactly, which is why size can never
            // tell them apart -- the declared ggml type id is the only discriminator,
            // and BF16 therefore stays out of `from_size`'s inference ladder.
            Self::F16 | Self::BF16 => size_bytes == n_rows * n_cols * 2,
            // Super-block formats (256 elements per super-block)
            Self::Q4K
            | Self::Q5K
            | Self::Q6K
            | Self::IQ4XS
            | Self::IQ3S
            | Self::IQ2XXS
            | Self::IQ2S
            | Self::IQ3XXS
            | Self::Q2K => {
                let n_superblocks = n_rows * ((n_cols + 255) / 256);
                size_bytes == n_superblocks * self.bytes_per_superblock()
            },
            // Block formats (32 elements per block). #3869: IQ4_NL belongs
            // here, not with the super-blocks - it is natively 18 B / 32 elems.
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q8_0 | Self::IQ4NL | Self::Q5_1 => {
                let n_blocks = n_rows * ((n_cols + 31) / 32);
                size_bytes == n_blocks * self.bytes_per_block()
            },
        }
    }

    /// #3908: the quant type of a weight, from its DECLARED type and its byte size.
    ///
    /// The declared type wins whenever it is CONSISTENT with the size; the size is
    /// consulted only when the declaration cannot be right (PAR-058: "GGUF metadata
    /// can lie" - Qwen 0.5B declared Q4_0 for a tensor whose bytes are Q4_1, a
    /// different size, which this still catches).
    ///
    /// Size-first resolution silently relabelled a tied BF16 LM head as F16 on
    /// `qwen2.5-coder-0.5b-instruct.apr`: both are 2 bytes/element, `from_size`
    /// returned F16, and the bf16 bytes were decoded as IEEE half. The F2 parity
    /// gate caught it (CPU argmax 319, GPU argmax 14880). A size can only ever
    /// agree with a declaration or contradict it; it cannot out-rank one it agrees
    /// with. This is the policy `indexed_ffn.rs` already applied to `ffn_down`, now
    /// in one place so the LM-head sites cannot drift from it again.
    #[must_use]
    pub fn resolve_declared_or_sized(
        declared: Option<Self>,
        size_bytes: usize,
        n_rows: usize,
        n_cols: usize,
    ) -> Option<Self> {
        if let Some(d) = declared {
            if d.matches_size(size_bytes, n_rows, n_cols) {
                return Some(d);
            }
        }
        Self::from_size(size_bytes, n_rows, n_cols).or(declared)
    }

    /// PAR-058: Detect quantization type from actual weight size
    /// Some GGUF files have incorrect type metadata, so we verify by size
    ///
    /// CORRECTNESS-002 FIX: For certain dimension combinations, Q4_0 and Q4K have
    /// the SAME byte size (e.g., 1536×8960: 1536×280×18 = 1536×35×144 = 7,741,440).
    /// Check super-block formats FIRST since they have more distinctive layouts.
    pub fn from_size(size_bytes: usize, n_rows: usize, n_cols: usize) -> Option<Self> {
        // GH-374: Check F32 first — unambiguous (no block alignment rounding)
        // APR checkpoints may have F32 LM head from SafeTensors import
        if size_bytes == n_rows * n_cols * 4 {
            return Some(Self::F32);
        }
        // #3477: 2 bytes/element is unambiguous against every BLOCK format here
        // (Q8_0 is 34/32 = 1.0625 B/elem, Q4_1 0.625). #3908: it is NOT
        // unambiguous against BF16, which has exactly F16's size - so this arm is
        // only a guess, and callers that HAVE a declared type must go through
        // `resolve_declared_or_sized`, which lets a consistent declaration win.
        if size_bytes == n_rows * n_cols * 2 {
            return Some(Self::F16);
        }

        // CORRECTNESS-002: Check super-block formats FIRST
        // Super-block formats (256 elements per super-block)
        let n_superblocks = n_rows * ((n_cols + 255) / 256);
        let superblock_formats = [
            (Self::Q6K, 210),
            (Self::Q5K, 176),
            (Self::Q4K, 144),
            (Self::IQ4XS, 136),
            (Self::IQ3S, 110),
            (Self::IQ2XXS, 66),
            (Self::IQ2S, 82),
            (Self::IQ3XXS, 98),
            (Self::Q2K, 84),
        ];

        for (fmt, bytes_per_sb) in superblock_formats {
            if size_bytes == n_superblocks * bytes_per_sb {
                return Some(fmt);
            }
        }

        // Then check block formats (32 elements per block)
        let n_blocks = n_rows * ((n_cols + 31) / 32);
        let formats = [
            (Self::Q4_0, 18),
            (Self::Q4_1, 20),
            (Self::Q5_0, 22),
            (Self::Q5_1, 24),
            (Self::Q8_0, 34),
        ];

        for (fmt, bytes_per_block) in formats {
            if size_bytes == n_blocks * bytes_per_block {
                return Some(fmt);
            }
        }

        None
    }
}

// =============================================================================
// PMAT-232: Bound Weight — kernel resolved at model load, not at inference
// =============================================================================
//
// Architecture: The model format defines the kernel. The kernel is bound at
// load time. The forward pass has ZERO dispatch.
//
// Before (7+ match sites per forward call):
//   match layer_weights.attn_q_qtype {
//       Q4K => q4k_gemv_into(...),
//       Q6K => q6k_gemv_into(...),
//       _ => q4k_gemv_into(...),  // catch-all silently selects the wrong kernel
//   }
//
// After (0 match sites per forward call):
//   layer.q_proj.gemv(executor, &input, &output)?;  // kernel pre-bound
//
// The match happens ONCE in BoundWeight::bind(). Adding a new WeightQuantType
// variant produces a compile error in exactly ONE place.

/// A GPU weight with its GEMV kernel pre-bound at model load time.
///
/// Construction validates the quant type → kernel mapping. The forward pass
/// calls `.gemv()` which CANNOT dispatch the wrong kernel because the kernel
/// was resolved at bind time.
///
/// This is Poka-Yoke applied to kernel dispatch: the mistake (wrong kernel)
/// is structurally impossible after construction.
#[derive(Debug, Clone)]
pub struct BoundWeight {
    /// Device pointer to quantized weight data
    pub ptr: u64,
    /// Size in bytes
    pub len: usize,
    /// Output dimension (rows in weight matrix)
    pub out_dim: u32,
    /// Input dimension (cols in weight matrix)
    pub in_dim: u32,
    /// The kernel that was bound at construction — private, cannot be changed
    kernel: GemvKernel,
}

/// The GEMV kernel to use. Resolved ONCE at model load time.
/// This is the SINGLE source of truth for quant type → kernel mapping.
/// See contracts/tensor-layout-v1.yaml quant_dispatch section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemvKernel {
    /// Q4_K super-block kernel (144 bytes / 256 elements)
    Q4K,
    /// Q5_K super-block kernel (176 bytes / 256 elements)
    Q5K,
    /// Q6_K super-block kernel (210 bytes / 256 elements)
    Q6K,
    /// Q8_0 block kernel (34 bytes / 32 elements)
    Q8_0,
    /// Q4_0 block kernel (18 bytes / 32 elements)
    Q4_0,
    /// Q5_0 block kernel (22 bytes / 32 elements)
    Q5_0,
    /// Q4_1 block kernel (20 bytes / 32 elements)
    Q4_1,
    /// F32 GEMV kernel (4 bytes per element, no dequantization)
    /// GH-374: For F32 LM head weights in APR checkpoints
    F32,
    /// F16 GEMV kernel (2 bytes per element, converting load, no dequantization)
    /// #3477: for F16 projections in UD dynamic quants
    F16,
    /// IQ4_XS GEMV kernel (136 bytes / 256 elements, codebook + split scales)
    /// #3477: for IQ4_XS ffn_gate/ffn_up in UD dynamic quants
    IQ4XS,
    /// IQ4_NL GEMV kernel (18 bytes / 32 elements, same codebook, f16 scale)
    /// #3869: for the IQ4_NL tensors that dominate IQ3_M and IQ4_XS files
    IQ4NL,
    /// IQ3_S GEMV kernel (110 bytes / 256 elements, 9-bit grid + sign bytes)
    /// #3884: the 21 IQ3_S tensors in Qwen2.5-0.5B-Instruct-IQ3_M
    IQ3S,
    /// IQ2_XXS GEMV kernel (66 bytes / 256 elements, 8-bit grid + 7-bit signs)
    /// #3931: the 95 IQ2_XXS tensors in Qwen3.5-0.8B-UD-IQ2_XXS
    IQ2XXS,
    /// IQ2_S GEMV kernel (82 bytes / 256 elements) - #3953
    IQ2S,
    /// IQ3_XXS GEMV kernel (98 bytes / 256 elements) - #3963
    IQ3XXS,
    /// Q2_K GEMV kernel (84 bytes / 256 elements, affine 2-bit) - #3960
    Q2K,
    /// Q5_1 GEMV kernel (24 bytes / 32 elements, affine w = q*d + m)
    /// #3885: the 24 Q5_1 tensors in Qwen2.5-0.5B-Instruct-IQ4_XS
    Q5_1,
    /// BF16 GEMV kernel (2 bytes per element, exact widening load)
    /// #3908: the 290 BF16 tensors in qwen2.5-coder-0.5b-instruct.apr
    BF16,
}

impl BoundWeight {
    /// Bind a weight to its correct GEMV kernel based on quantization type.
    ///
    /// This is the ONE place where WeightQuantType → GemvKernel mapping happens.
    /// The match is exhaustive — adding a new variant is a compile error here.
    pub fn bind(ptr: u64, len: usize, qtype: WeightQuantType, out_dim: u32, in_dim: u32) -> Self {
        // PMAT-232: Exhaustive mapping — no catch-all, no default.
        // If you add a WeightQuantType variant, this MUST be updated.
        let kernel = match qtype {
            WeightQuantType::Q4K => GemvKernel::Q4K,
            WeightQuantType::Q5K => GemvKernel::Q5K,
            WeightQuantType::Q6K => GemvKernel::Q6K,
            WeightQuantType::Q8_0 => GemvKernel::Q8_0,
            WeightQuantType::Q4_0 => GemvKernel::Q4_0,
            WeightQuantType::Q5_0 => GemvKernel::Q5_0,
            WeightQuantType::Q4_1 => GemvKernel::Q4_1,
            WeightQuantType::F32 => GemvKernel::F32,
            WeightQuantType::F16 => GemvKernel::F16,
            WeightQuantType::IQ4XS => GemvKernel::IQ4XS,
            WeightQuantType::IQ4NL => GemvKernel::IQ4NL,
            WeightQuantType::IQ3S => GemvKernel::IQ3S,
            WeightQuantType::IQ2XXS => GemvKernel::IQ2XXS,
            WeightQuantType::IQ2S => GemvKernel::IQ2S,
            WeightQuantType::IQ3XXS => GemvKernel::IQ3XXS,
            WeightQuantType::Q2K => GemvKernel::Q2K,
            WeightQuantType::Q5_1 => GemvKernel::Q5_1,
            WeightQuantType::BF16 => GemvKernel::BF16,
        };
        Self {
            ptr,
            len,
            out_dim,
            in_dim,
            kernel,
        }
    }

    /// The bound kernel (read-only).
    pub fn kernel(&self) -> GemvKernel {
        self.kernel
    }
}

/// A complete transformer layer with all kernels pre-bound.
///
/// Constructed from `IndexedLayerWeights` at model load time.
/// The forward pass uses this — ZERO dispatch, ZERO match statements.
#[derive(Debug, Clone)]
pub struct BoundLayerWeights {
    /// Q projection (hidden → q_dim)
    pub q_proj: BoundWeight,
    /// K projection (hidden → kv_dim)
    pub k_proj: BoundWeight,
    /// V projection (hidden → kv_dim)
    pub v_proj: BoundWeight,
    /// Output projection (q_dim → hidden)
    pub o_proj: BoundWeight,
    /// FFN gate projection (hidden → intermediate)
    pub ffn_gate: BoundWeight,
    /// FFN up projection (hidden → intermediate)
    pub ffn_up: BoundWeight,
    /// FFN down projection (intermediate → hidden)
    pub ffn_down: BoundWeight,
    /// Attention norm weight pointer
    pub attn_norm_ptr: u64,
    /// Attention norm weight length
    pub attn_norm_len: usize,
    /// FFN norm weight pointer
    pub ffn_norm_ptr: u64,
    /// FFN norm weight length
    pub ffn_norm_len: usize,
    /// Q bias pointer (0 if no bias)
    pub attn_q_bias_ptr: u64,
    /// Q bias length in elements (0 if no bias)
    pub attn_q_bias_len: usize,
    /// K bias pointer (0 if no bias)
    pub attn_k_bias_ptr: u64,
    /// K bias length in elements (0 if no bias)
    pub attn_k_bias_len: usize,
    /// V bias pointer (0 if no bias)
    pub attn_v_bias_ptr: u64,
    /// V bias length in elements (0 if no bias)
    pub attn_v_bias_len: usize,
    /// GH-279: Per-head Q RMSNorm gamma pointer (0 if no QkNorm)
    pub attn_q_norm_ptr: u64,
    /// Per-head Q RMSNorm gamma length in elements (0 if no QkNorm)
    pub attn_q_norm_len: usize,
    /// GH-279: Per-head K RMSNorm gamma pointer (0 if no QkNorm)
    pub attn_k_norm_ptr: u64,
    /// Per-head K RMSNorm gamma length in elements (0 if no QkNorm)
    pub attn_k_norm_len: usize,
}

impl BoundLayerWeights {
    /// Bind all layer weights from ValidatedLayerWeights.
    ///
    /// GH-279: Takes `&ValidatedLayerWeights` to ensure only validated weights
    /// can be bound to kernels. This is the compilation step: quant types are
    /// resolved to kernels ONCE. After this, the forward pass has zero dispatch.
    pub fn bind(
        src: &ValidatedLayerWeights,
        hidden_dim: u32,
        q_dim: u32,
        kv_dim: u32,
        intermediate_dim: u32,
    ) -> Self {
        Self {
            q_proj: BoundWeight::bind(
                src.attn_q_ptr,
                src.attn_q_len,
                src.attn_q_qtype,
                q_dim,
                hidden_dim,
            ),
            k_proj: BoundWeight::bind(
                src.attn_k_ptr,
                src.attn_k_len,
                src.attn_k_qtype,
                kv_dim,
                hidden_dim,
            ),
            v_proj: BoundWeight::bind(
                src.attn_v_ptr,
                src.attn_v_len,
                src.attn_v_qtype,
                kv_dim,
                hidden_dim,
            ),
            o_proj: BoundWeight::bind(
                src.attn_output_ptr,
                src.attn_output_len,
                src.attn_output_qtype,
                hidden_dim,
                q_dim,
            ),
            ffn_gate: BoundWeight::bind(
                src.ffn_gate_ptr,
                src.ffn_gate_len,
                src.ffn_gate_qtype,
                intermediate_dim,
                hidden_dim,
            ),
            ffn_up: BoundWeight::bind(
                src.ffn_up_ptr,
                src.ffn_up_len,
                src.ffn_up_qtype,
                intermediate_dim,
                hidden_dim,
            ),
            ffn_down: BoundWeight::bind(
                src.ffn_down_ptr,
                src.ffn_down_len,
                src.ffn_down_qtype,
                hidden_dim,
                intermediate_dim,
            ),
            attn_norm_ptr: src.attn_norm_ptr,
            attn_norm_len: src.attn_norm_len,
            ffn_norm_ptr: src.ffn_norm_ptr,
            ffn_norm_len: src.ffn_norm_len,
            attn_q_bias_ptr: src.attn_q_bias_ptr,
            attn_q_bias_len: src.attn_q_bias_len,
            attn_k_bias_ptr: src.attn_k_bias_ptr,
            attn_k_bias_len: src.attn_k_bias_len,
            attn_v_bias_ptr: src.attn_v_bias_ptr,
            attn_v_bias_len: src.attn_v_bias_len,
            attn_q_norm_ptr: src.attn_q_norm_ptr,
            attn_q_norm_len: src.attn_q_norm_len,
            attn_k_norm_ptr: src.attn_k_norm_ptr,
            attn_k_norm_len: src.attn_k_norm_len,
        }
    }
}

// =============================================================================
// GH-279: ValidatedLayerWeights — Poka-Yoke sealed constructor
// =============================================================================
//
// The problem: IndexedLayerWeights uses (0, 0) as both "optional field" and
// "missing required field". The type system cannot distinguish them.
//
// The solution: ValidatedLayerWeights wraps IndexedLayerWeights with a private
// inner field. The ONLY constructor (`validate()`) checks every architecture-
// required field against `required_roles()`. If any required field is (0, 0),
// construction FAILS with a descriptive error — not a silent garbage inference.
//
// The forward pass ONLY accepts ValidatedLayerWeights. Passing unvalidated
// weights is a compile error.

use crate::arch_requirements::{required_roles, WeightRole};
use crate::gguf::ArchConstraints;
use std::fmt;

/// Error returned when `ValidatedLayerWeights::validate()` finds a missing required field.
#[derive(Debug, Clone)]
pub struct WeightValidationError {
    /// The weight role that is missing
    pub role: WeightRole,
    /// Human-readable field name
    pub field: &'static str,
    /// Architecture name (for error messages)
    pub arch_name: String,
    /// Layer index
    pub layer_idx: usize,
}

impl fmt::Display for WeightValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GH-279: Missing required weight '{}' for architecture '{}' at layer {} \
             (ValidatedLayerWeights Poka-Yoke: (0, 0) is not allowed for required roles)",
            self.field, self.arch_name, self.layer_idx
        )
    }
}

impl std::error::Error for WeightValidationError {}

/// PMAT-235 Poka-Yoke: Validated layer weights.
///
/// Private inner field = IMPOSSIBLE to construct without passing validation.
/// The forward pass ONLY accepts this type — compile error if anyone passes
/// unvalidated `IndexedLayerWeights`.
///
/// # Invariant
///
/// Every field required by `required_roles(arch)` has non-zero ptr AND len.
/// This invariant is established by `validate()` and preserved by immutability.
#[derive(Debug, Clone)]
pub struct ValidatedLayerWeights {
    /// Private — construction only through `validate()`.
    inner: IndexedLayerWeights,
}

impl std::ops::Deref for ValidatedLayerWeights {
    type Target = IndexedLayerWeights;

    fn deref(&self) -> &IndexedLayerWeights {
        &self.inner
    }
}

impl ValidatedLayerWeights {
    /// The ONLY constructor. Validates all architecture-required fields are non-zero.
    ///
    /// If any required field has ptr == 0 AND len == 0, returns
    /// `Err(WeightValidationError)` with the field name, architecture, and layer index.
    ///
    /// # Errors
    ///
    /// Returns `WeightValidationError` if a required weight role is missing (ptr=0, len=0).
    pub fn validate(
        raw: IndexedLayerWeights,
        arch: &ArchConstraints,
        layer_idx: usize,
    ) -> Result<Self, WeightValidationError> {
        let roles = required_roles(arch);

        for &role in roles {
            // M-GPU-MOE-1.3 (qwen3-moe-forward-gpu-v1 v1.3.0): for MoE
            // architectures, the per-layer dense FFN tensor names
            // (FfnGate, FfnUp, FfnDown) DO NOT EXIST — MoE has 128
            // expert tensors per layer loaded into the `moe_layers`
            // parameter at forward-time. Skip these role checks; the
            // MoE forward path uses moe_layers directly and never
            // reads FfnGate/FfnUp/FfnDown from the indexed weights.
            // See: evidence/m-gpu-moe-1-2-blocked-by-preload-bug-2026-05-04/findings.md
            if arch.is_moe
                && matches!(
                    role,
                    WeightRole::FfnGate | WeightRole::FfnUp | WeightRole::FfnDown
                )
            {
                continue;
            }

            let (ptr, len) = Self::get_field(&raw, role);
            if ptr == 0 && len == 0 {
                return Err(WeightValidationError {
                    role,
                    field: role.field_name(),
                    arch_name: Self::arch_display_name(arch),
                    layer_idx,
                });
            }
        }

        Ok(Self { inner: raw })
    }

    /// Access the validated inner weights (read-only).
    #[must_use]
    pub fn inner(&self) -> &IndexedLayerWeights {
        &self.inner
    }

    /// Bypass validation to wrap raw weights (test-only).
    ///
    /// Production code MUST NOT use this — the weights may violate
    /// architecture constraints. Only available in test builds.
    #[cfg(test)]
    #[must_use]
    pub fn new_unchecked(raw: IndexedLayerWeights) -> Self {
        Self { inner: raw }
    }

    /// Mutable access to inner weights (test-only).
    ///
    /// Production code MUST NOT use this — validation invariants are not
    /// re-checked after mutation. Only available in test builds.
    #[cfg(test)]
    #[must_use]
    pub fn inner_mut(&mut self) -> &mut IndexedLayerWeights {
        &mut self.inner
    }

    /// Extract (ptr, len) for a given weight role from raw `IndexedLayerWeights`.
    fn get_field(raw: &IndexedLayerWeights, role: WeightRole) -> (u64, usize) {
        match role {
            WeightRole::AttnNorm => (raw.attn_norm_ptr, raw.attn_norm_len),
            WeightRole::FfnNorm => (raw.ffn_norm_ptr, raw.ffn_norm_len),
            WeightRole::AttnQNorm => (raw.attn_q_norm_ptr, raw.attn_q_norm_len),
            WeightRole::AttnKNorm => (raw.attn_k_norm_ptr, raw.attn_k_norm_len),
            WeightRole::AttnQBias => (raw.attn_q_bias_ptr, raw.attn_q_bias_len),
            WeightRole::AttnKBias => (raw.attn_k_bias_ptr, raw.attn_k_bias_len),
            WeightRole::AttnVBias => (raw.attn_v_bias_ptr, raw.attn_v_bias_len),
            WeightRole::QProj => (raw.attn_q_ptr, raw.attn_q_len),
            WeightRole::KProj => (raw.attn_k_ptr, raw.attn_k_len),
            WeightRole::VProj => (raw.attn_v_ptr, raw.attn_v_len),
            WeightRole::OProj => (raw.attn_output_ptr, raw.attn_output_len),
            WeightRole::FfnGate => (raw.ffn_gate_ptr, raw.ffn_gate_len),
            WeightRole::FfnUp => (raw.ffn_up_ptr, raw.ffn_up_len),
            WeightRole::FfnDown => (raw.ffn_down_ptr, raw.ffn_down_len),
        }
    }

    /// Human-readable architecture name for error messages.
    fn arch_display_name(arch: &ArchConstraints) -> String {
        // Reconstruct name from constraints
        if arch.has_qk_norm && !arch.has_bias {
            "qwen3".to_string()
        } else if !arch.has_qk_norm && arch.has_bias {
            "qwen2/phi (has_bias)".to_string()
        } else if arch.has_qk_norm && arch.has_bias {
            "unknown (has_qk_norm + has_bias)".to_string()
        } else {
            "llama/mistral/gemma (base)".to_string()
        }
    }
}

include!("transformer_workspace.rs");
