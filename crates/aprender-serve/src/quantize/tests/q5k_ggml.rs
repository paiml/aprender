//! FALSIFY-QDOT-009 (PMAT-1101): realizar's `Q5_K` reader decodes ggml's `block_q5_K`.
//!
//! The oracle is gguf-py, llama.cpp's own reader, applied to a super-block that llama.cpp
//! quantized. It is never another aprender reader: before PMAT-1101 `dequantize_q5_k` and
//! `fused_q5k_dot` shared an invented layout, and every test compared one with the other.

use crate::quantize::{dequantize_q5_k, fused_q5k_dot, fused_q5k_parallel_matvec_into};

include!("q5k_ggml_fixture.rs");

#[test]
fn test_q5k_ggml_dequantize_matches_gguf_py_bit_exact() {
    let got = dequantize_q5_k(&GGML_Q5K_BLOCK).expect("one super-block");
    assert_eq!(got.len(), GGML_Q5K_EXPECTED.len());
    for (i, (g, e)) in got.iter().zip(GGML_Q5K_EXPECTED).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "value {i}: realizar {g}, gguf-py {e}");
    }
}

#[test]
fn test_q5k_ggml_fused_dot_reads_the_same_layout() {
    // A one-hot activation turns the dot product into the one value it selects.
    for (i, e) in GGML_Q5K_EXPECTED.iter().enumerate() {
        let mut one_hot = [0.0f32; 256];
        one_hot[i] = 1.0;
        let got = fused_q5k_dot(&GGML_Q5K_BLOCK, &one_hot).expect("one super-block");
        assert_eq!(got, *e, "one-hot {i}: realizar {got}, gguf-py {e}");
    }
}

#[test]
fn test_q5k_ggml_parallel_matvec_matches_gguf_py_rows() {
    // Two rows of the golden block against a ramp: each output is the gguf-py dot product.
    let weights = [GGML_Q5K_BLOCK, GGML_Q5K_BLOCK].concat();
    let act: Vec<f32> = (0..256u16).map(|i| (f32::from(i) - 127.5) / 64.0).collect();
    let want: f64 = GGML_Q5K_EXPECTED
        .iter()
        .zip(&act)
        .map(|(w, a)| f64::from(*w) * f64::from(*a))
        .sum();
    let mut out = [0.0f32; 2];
    fused_q5k_parallel_matvec_into(&weights, &act, 256, 2, &mut out).expect("2 x 256");
    for (row, got) in out.iter().enumerate() {
        assert!(
            (f64::from(*got) - want).abs() <= 1e-4 * want.abs().max(1.0),
            "row {row}: realizar {got}, gguf-py {want}"
        );
    }
}
