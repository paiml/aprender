//! FALSIFY-QDOT-010 (#3111): the CUDA Q5_K GEMV decodes ggml's `block_q5_K`, judged against
//! gguf-py's values of a llama.cpp-quantized super-block (the FALSIFY-QDOT-009 fixture), never
//! against another aprender reader.

use super::CudaExecutor;
use serial_test::serial;

include!("../../quantize/tests/q5k_ggml_fixture.rs");

/// Run the Q5_K GEMV on `rows` copies of the golden block against `input`.
fn gemv_golden(executor: &mut CudaExecutor, rows: u32, input: &[f32]) -> Vec<f32> {
    let weights = GGML_Q5K_BLOCK.repeat(rows as usize);
    let mut out = vec![0.0f32; rows as usize];
    executor
        .q5k_gemv(&weights, input, &mut out, rows, 256)
        .expect("q5k_gemv on the golden block");
    out
}

#[test]
#[serial]
fn test_q5k_gemv_one_hot_returns_each_gguf_py_value() {
    let mut executor = crate::cuda_executor_or_skip!(0);
    for (i, &want) in GGML_Q5K_EXPECTED.iter().enumerate() {
        let mut one_hot = [0.0f32; 256];
        one_hot[i] = 1.0;
        for got in gemv_golden(&mut executor, 2, &one_hot) {
            assert!(
                (got - want).abs() <= 1e-6 + 1e-5 * want.abs(),
                "value {i}: cuda {got}, gguf-py {want}"
            );
        }
    }
}

#[test]
#[serial]
fn test_q5k_gemv_ramp_matches_gguf_py_dot() {
    let mut executor = crate::cuda_executor_or_skip!(0);
    let input: Vec<f32> = (0..256u16).map(|i| (f32::from(i) - 127.5) / 64.0).collect();
    let want: f64 = GGML_Q5K_EXPECTED
        .iter()
        .zip(&input)
        .map(|(w, a)| f64::from(*w) * f64::from(*a))
        .sum();
    for got in gemv_golden(&mut executor, 4, &input) {
        assert!(
            (f64::from(got) - want).abs() <= 1e-4 * want.abs().max(1.0),
            "cuda {got}, gguf-py {want}"
        );
    }
}
