//! M-FFN-GGUF-4 step (c) candidate H2d.2 falsifier — APR Q4K dequant byte-identity.
//!
//! Tests that APR's scalar `dequantize_q4_k` and SIMD-accelerated
//! `dequantize_q4_k_simd` produce **byte-identical** Vec<f32> for the
//! same Q4K super-block bytes. Analog of M92's
//! `falsify_ffn_gguf_006_simd_vs_scalar_reduction_order_byte_identity`
//! but for Q4K dequant (not f32 dot product).
//!
//! ## Why this matters for SHIP-007 §22 / H2d.2
//!
//! The §27 evidence pinned SHIP-007 layer-3 to APR-side at the
//! per-element f32 level. Two reduction-order hypotheses have been
//! falsified:
//! - §28 parallel-reduction non-determinism (M91): FALSIFIED
//! - H2a' SIMD-vs-scalar reduction-order (M92): FALSIFIED
//!
//! The refined H2d.2 hypothesis (post-second-falsification): APR's
//! F32 weights themselves differ at bit level from a true dequantization
//! of GGUF Q4K bytes — despite SHIP-003 PR #1059 cos≥0.9999999 weight
//! invariance.
//!
//! This test addresses H2d.2 at the APR-INTERNAL dequant level. If
//! APR's two own dequant paths (scalar + SIMD) differ at bit level on
//! the same Q4K bytes, then H2d.2 has a mechanically realizable cause:
//! whichever path APR's loader uses produces different f32 bits than
//! whichever path GGUF's matvec uses internally.
//!
//! ## Test design
//!
//! Synthetic Q4K super-block (144 bytes):
//! - bytes 0..2: f16 d (block scale)
//! - bytes 2..4: f16 dmin (block min)
//! - bytes 4..16: 12 packed scale/min sub-block bytes
//! - bytes 16..144: 128 quantized 4-bit values (256 elements packed)
//!
//! Reproducible (no randomness — same byte pattern across runs).
//!
//! ## Expected outcome
//!
//! Per the precedent from M91+M92 (both reduction-order hypotheses
//! produced byte-identical results across paths), this test EXPECTS
//! both dequant paths to also produce byte-identical output. Asserting
//! BYTE-IDENTITY as the regression-test invariant.
//!
//! If FALSIFIED (paths produce DIFFERENT bits): H2d.2 has confirmed
//! mechanism — APR's loader picks one path, GGUF's matvec picks the
//! other (or has its own inline dequant), and the f32 weights APR
//! uses differ from what GGUF effectively uses at the bit level.
//! SHIP-007 fix scope = align dequant paths.
//!
//! Per `contracts/trace-ffn-sub-block-gguf-v1.yaml` v1.3.0 amendment
//! H2d.2 hypothesis class.

use realizar::quantize::{dequantize_q4_k, dequantize_q4_k_simd};

/// Build one synthetic Q4K super-block with reproducible byte values.
/// Mimics the layout documented in `dequantize_q4_k` source (144 bytes total).
fn synthetic_q4k_super_block() -> Vec<u8> {
    let mut block = vec![0u8; 144];
    // f16 d = 0x3C00 (= 1.0)
    block[0] = 0x00;
    block[1] = 0x3C;
    // f16 dmin = 0xB400 (= -0.25)
    block[2] = 0x00;
    block[3] = 0xB4;
    // 12 scale/min bytes — set to a non-trivial fixed pattern
    for (i, b) in block[4..16].iter_mut().enumerate() {
        *b = ((i * 7 + 3) % 256) as u8;
    }
    // 128 quant bytes — set to a non-trivial fixed pattern
    for (i, b) in block[16..144].iter_mut().enumerate() {
        // Mix of low and high nibbles to exercise both extraction paths
        *b = ((i * 13 + 17) % 256) as u8;
    }
    block
}

#[test]
fn falsify_ffn_gguf_007_q4k_scalar_vs_simd_dequant_byte_identity() {
    let block = synthetic_q4k_super_block();
    assert_eq!(
        block.len(),
        144,
        "test setup: super-block must be 144 bytes"
    );

    let result_scalar =
        dequantize_q4_k(&block).expect("dequantize_q4_k (scalar) failed on synthetic block");
    let result_simd =
        dequantize_q4_k_simd(&block).expect("dequantize_q4_k_simd failed on synthetic block");

    assert_eq!(
        result_scalar.len(),
        256,
        "Q4K super-block must dequantize to exactly 256 elements (got {})",
        result_scalar.len()
    );
    assert_eq!(
        result_simd.len(),
        256,
        "SIMD path must produce same count (got {})",
        result_simd.len()
    );

    // EMPIRICAL EXPECTATION (per M91+M92 precedent): both paths produce
    // byte-identical f32 bits. Assert as regression-test invariant.
    let mut first_diff: Option<(usize, u32, u32)> = None;
    for (i, (&s, &v)) in result_scalar.iter().zip(result_simd.iter()).enumerate() {
        if s.to_bits() != v.to_bits() {
            first_diff = Some((i, s.to_bits(), v.to_bits()));
            break;
        }
    }

    if let Some((i, scalar_bits, simd_bits)) = first_diff {
        panic!(
            "FALSIFY-FFN-GGUF-007: scalar and SIMD Q4K dequant produced DIFFERENT bits at \
             element {i}: scalar={} ({scalar_bits:#x}) vs simd={} ({simd_bits:#x}). \
             H2d.2 hypothesis MECHANICALLY CONFIRMED — APR's two own dequant paths differ \
             at bit level on same Q4K bytes. SHIP-007 fix scope: align dequant paths AND \
             ensure APR's loader uses the same path GGUF's matvec uses internally. \
             Update contract trace-ffn-sub-block-gguf-v1 v1.3.0 with empirical evidence.",
            f32::from_bits(scalar_bits),
            f32::from_bits(simd_bits)
        );
    }

    // Document the empirical canonical first element so a future
    // engineer doesn't have to re-derive it.
    eprintln!(
        "FALSIFY-FFN-GGUF-007: scalar+SIMD Q4K dequant byte-identical across all 256 elements. \
         element[0]={} ({:#x}); element[255]={} ({:#x}). \
         H2d.2 hypothesis FALSIFIED at APR-internal dequant level — both APR dequant paths \
         agree byte-for-byte.",
        result_scalar[0],
        result_scalar[0].to_bits(),
        result_scalar[255],
        result_scalar[255].to_bits()
    );
}
