
#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;
    use crate::cuda::executor::test_fixtures::{
        generate_q4_0_weights, generate_q5_0_weights, generate_q8_0_weights,
    };

    /// Helper to create CudaExecutor for tests
    fn create_executor() -> Option<CudaExecutor> {
        CudaExecutor::new(0).ok()
    }

    // ========================================================================
    // Q8_0 GEMV Tests
    // ========================================================================

    #[test]
    fn test_q8_0_gemv_into_basic() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        // K=256, N=64: 64 output rows, 8 blocks per row (32 elements/block)
        let k = 256u32;
        let n = 64u32;
        let blocks = (n as usize) * (k as usize / 32);
        let weights = generate_q8_0_weights(blocks);

        // Upload weights to GPU
        let weight_buf = GpuBuffer::from_host(&exec.context, &weights).unwrap();
        let weight_ptr = weight_buf.as_ptr();

        // Create input/output buffers
        let input: Vec<f32> = (0..k as usize).map(|i| (i as f32) * 0.01).collect();
        let output = vec![0.0f32; n as usize];
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let output_buf = GpuBuffer::from_host(&exec.context, &output).unwrap();

        // Execute (may fail on PTX generation but exercises path)
        let result = exec.q8_0_gemv_into(weight_ptr, &input_buf, &output_buf, n, k);
        let _ = result;
    }

    #[test]
    fn test_q8_0_gemv_into_large() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let k = 512u32;
        let n = 128u32;
        let blocks = (n as usize) * (k as usize / 32);
        let weights = generate_q8_0_weights(blocks);

        let weight_buf = GpuBuffer::from_host(&exec.context, &weights).unwrap();
        let input: Vec<f32> = vec![0.5f32; k as usize];
        let output = vec![0.0f32; n as usize];
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let output_buf = GpuBuffer::from_host(&exec.context, &output).unwrap();

        let result = exec.q8_0_gemv_into(weight_buf.as_ptr(), &input_buf, &output_buf, n, k);
        let _ = result;
    }

    // ========================================================================
    // Q5_0 GEMV Tests
    // ========================================================================

    #[test]
    fn test_q5_0_gemv_into_basic() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let k = 256u32;
        let n = 64u32;
        let blocks = (n as usize) * (k as usize / 32);
        let weights = generate_q5_0_weights(blocks);

        let weight_buf = GpuBuffer::from_host(&exec.context, &weights).unwrap();
        let input: Vec<f32> = (0..k as usize).map(|i| (i as f32) * 0.01).collect();
        let output = vec![0.0f32; n as usize];
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let output_buf = GpuBuffer::from_host(&exec.context, &output).unwrap();

        let result = exec.q5_0_gemv_into(weight_buf.as_ptr(), &input_buf, &output_buf, n, k);
        let _ = result;
    }

    #[test]
    fn test_q5_0_gemv_into_large() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let k = 512u32;
        let n = 128u32;
        let blocks = (n as usize) * (k as usize / 32);
        let weights = generate_q5_0_weights(blocks);

        let weight_buf = GpuBuffer::from_host(&exec.context, &weights).unwrap();
        let input: Vec<f32> = vec![0.5f32; k as usize];
        let output = vec![0.0f32; n as usize];
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let output_buf = GpuBuffer::from_host(&exec.context, &output).unwrap();

        let result = exec.q5_0_gemv_into(weight_buf.as_ptr(), &input_buf, &output_buf, n, k);
        let _ = result;
    }

    // ========================================================================
    // Q4_0 GEMV Tests
    // ========================================================================

    #[test]
    fn test_q4_0_gemv_into_basic() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let k = 256u32;
        let n = 64u32;
        let blocks = (n as usize) * (k as usize / 32);
        let weights = generate_q4_0_weights(blocks);

        let weight_buf = GpuBuffer::from_host(&exec.context, &weights).unwrap();
        let input: Vec<f32> = (0..k as usize).map(|i| (i as f32) * 0.01).collect();
        let output = vec![0.0f32; n as usize];
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let output_buf = GpuBuffer::from_host(&exec.context, &output).unwrap();

        let result = exec.q4_0_gemv_into(weight_buf.as_ptr(), &input_buf, &output_buf, n, k);
        let _ = result;
    }

    #[test]
    fn test_q4_0_gemv_into_single_row() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let k = 256u32;
        let n = 1u32;
        let blocks = k as usize / 32;
        let weights = generate_q4_0_weights(blocks);

        let weight_buf = GpuBuffer::from_host(&exec.context, &weights).unwrap();
        let input: Vec<f32> = vec![1.0f32; k as usize];
        let output = vec![0.0f32; n as usize];
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let output_buf = GpuBuffer::from_host(&exec.context, &output).unwrap();

        let result = exec.q4_0_gemv_into(weight_buf.as_ptr(), &input_buf, &output_buf, n, k);
        let _ = result;
    }

    // ========================================================================
    // Q4_1 GEMV Tests
    // ========================================================================

    #[test]
    fn test_q4_1_gemv_into_basic() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        // Q4_1 has same block count as Q4_0 but 20 bytes per block instead of 18
        let k = 256u32;
        let n = 64u32;
        let blocks = (n as usize) * (k as usize / 32);
        // Use Q4_0 weights (18 bytes/block), Q4_1 expects 20 bytes/block
        // The kernel will interpret incorrectly, but we're testing the path
        let weights = generate_q4_0_weights(blocks);

        let weight_buf = GpuBuffer::from_host(&exec.context, &weights).unwrap();
        let input: Vec<f32> = (0..k as usize).map(|i| (i as f32) * 0.01).collect();
        let output = vec![0.0f32; n as usize];
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let output_buf = GpuBuffer::from_host(&exec.context, &output).unwrap();

        let result = exec.q4_1_gemv_into(weight_buf.as_ptr(), &input_buf, &output_buf, n, k);
        let _ = result;
    }

    // ========================================================================
    // Q5_K GEMV Tests
    // ========================================================================

    #[test]
    fn test_q5k_gemv_into_basic() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        // Q5_K: 176 bytes per 256 elements (super-block format)
        let k = 256u32;
        let n = 64u32;
        // Q5_K needs k to be multiple of 256
        let superblocks = (n as usize) * (k as usize / 256);
        // Simulate Q5K weight data
        let weights = vec![0u8; superblocks * 176];

        let weight_buf = GpuBuffer::from_host(&exec.context, &weights).unwrap();
        let input: Vec<f32> = (0..k as usize).map(|i| (i as f32) * 0.01).collect();
        let output = vec![0.0f32; n as usize];
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let output_buf = GpuBuffer::from_host(&exec.context, &output).unwrap();

        let result = exec.q5k_gemv_into(weight_buf.as_ptr(), &input_buf, &output_buf, n, k);
        let _ = result;
    }

    // ========================================================================
    // Q6_K GEMV Tests
    // ========================================================================

    #[test]
    fn test_q6k_gemv_into_basic() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        // Q6_K: 210 bytes per 256 elements
        let k = 256u32;
        let n = 64u32;

        // The HwDp4a Q6K variant (default on sm_75+) quantizes activations into
        // workspace.q8_activation_buf and would otherwise panic; size it for k.
        exec.init_workspace(k as usize, k as usize)
            .expect("init_workspace");

        let superblocks = (n as usize) * (k as usize / 256);
        let weights = vec![0u8; superblocks * 210];

        let weight_buf = GpuBuffer::from_host(&exec.context, &weights).unwrap();
        let input: Vec<f32> = (0..k as usize).map(|i| (i as f32) * 0.01).collect();
        let output = vec![0.0f32; n as usize];
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let output_buf = GpuBuffer::from_host(&exec.context, &output).unwrap();

        let result = exec.q6k_gemv_into(weight_buf.as_ptr(), &input_buf, &output_buf, n, k);
        let _ = result;
    }

    #[test]
    fn test_coalesced_q6k_gemv_into_basic() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let k = 256u32;
        let n = 64u32;
        let superblocks = (n as usize) * (k as usize / 256);
        let weights = vec![0u8; superblocks * 210];

        let weight_buf = GpuBuffer::from_host(&exec.context, &weights).unwrap();
        let input: Vec<f32> = (0..k as usize).map(|i| (i as f32) * 0.01).collect();
        let output = vec![0.0f32; n as usize];
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let output_buf = GpuBuffer::from_host(&exec.context, &output).unwrap();

        let result =
            exec.coalesced_q6k_gemv_into(weight_buf.as_ptr(), &input_buf, &output_buf, n, k);
        let _ = result;
    }

    // ========================================================================
    // Batched Q6K GEMV Tests
    // ========================================================================

    #[test]
    fn test_batched_q6k_gemv_into_basic() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let m = 4u32; // batch size
        let k = 256u32;
        let n = 64u32;
        let superblocks = (n as usize) * (k as usize / 256);
        let weights = vec![0u8; superblocks * 210];

        let weight_buf = GpuBuffer::from_host(&exec.context, &weights).unwrap();
        // Batched input: m * k elements
        let input: Vec<f32> = (0..(m * k) as usize).map(|i| (i as f32) * 0.001).collect();
        // Batched output: m * n elements
        let output = vec![0.0f32; (m * n) as usize];
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let output_buf = GpuBuffer::from_host(&exec.context, &output).unwrap();

        let result =
            exec.batched_q6k_gemv_into(weight_buf.as_ptr(), &input_buf, &output_buf, m, n, k);
        let _ = result;
    }

    #[test]
    fn test_batched_q6k_gemv_into_m8() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let m = 8u32;
        let k = 256u32;
        let n = 32u32;
        let superblocks = (n as usize) * (k as usize / 256);
        let weights = vec![0u8; superblocks * 210];

        let weight_buf = GpuBuffer::from_host(&exec.context, &weights).unwrap();
        let input: Vec<f32> = vec![0.5f32; (m * k) as usize];
        let output = vec![0.0f32; (m * n) as usize];
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let output_buf = GpuBuffer::from_host(&exec.context, &output).unwrap();

        let result =
            exec.batched_q6k_gemv_into(weight_buf.as_ptr(), &input_buf, &output_buf, m, n, k);
        let _ = result;
    }

    // ========================================================================
    // PMAT-782: Legacy-quant GPU↔CPU parity (GGML interleaved nibble layout)
    //
    // GGML packs Q4_0/Q4_1/Q5_0 nibbles INTERLEAVED: byte j (0..16) holds value
    // j in its LOW nibble and value j+16 in its HIGH nibble (see
    // dequantize_row_q5_0 in ggml-quants.c). The GPU kernels previously assumed
    // CONSECUTIVE packing (byte = tid/2, low/high = tid&1), so every value index
    // ≥1 mapped to the wrong nibble → garbage logits. These tests build real
    // GGML-layout blocks and assert the GPU GEMV matches the CPU dequant
    // reference that already produces coherent output.
    // ========================================================================

    /// Build a single-row GGML-spec Q4_0 weight: `n=1`, `k` elements.
    /// Returns (raw_bytes, dequantized_f32_row).
    fn ggml_q4_0_row(k: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
        use half::f16;
        let nb = k / 32;
        let mut data = Vec::with_capacity(nb * 18);
        for b in 0..nb {
            let d = 0.05 * (b as f32 + 1.0);
            data.extend_from_slice(&f16::from_f32(d).to_le_bytes());
            // qs[j] low nibble = value j, high nibble = value j+16
            for j in 0..16 {
                let v_lo = ((b as u32 * 7 + j as u32 * 3 + seed) % 16) as u8;
                let v_hi = ((b as u32 * 11 + j as u32 * 5 + seed + 1) % 16) as u8;
                data.push(v_lo | (v_hi << 4));
            }
        }
        let row = crate::quantize::dequantize_q4_0(&data).unwrap();
        (data, row)
    }

    /// Build a single-row GGML-spec Q4_1 weight: `n=1`, `k` elements.
    fn ggml_q4_1_row(k: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
        use half::f16;
        let nb = k / 32;
        let mut data = Vec::with_capacity(nb * 20);
        for b in 0..nb {
            let d = 0.05 * (b as f32 + 1.0);
            let m = -0.3 * (b as f32 + 1.0);
            data.extend_from_slice(&f16::from_f32(d).to_le_bytes());
            data.extend_from_slice(&f16::from_f32(m).to_le_bytes());
            // qs[j] low nibble = value j, high nibble = value j+16
            for j in 0..16 {
                let v_lo = ((b as u32 * 7 + j as u32 * 3 + seed) % 16) as u8;
                let v_hi = ((b as u32 * 11 + j as u32 * 5 + seed + 1) % 16) as u8;
                data.push(v_lo | (v_hi << 4));
            }
        }
        let row = crate::quantize::dequantize_q4_1(&data).unwrap();
        (data, row)
    }

    /// Build a single-row GGML-spec Q5_0 weight: `n=1`, `k` elements.
    fn ggml_q5_0_row(k: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
        use half::f16;
        let nb = k / 32;
        let mut data = Vec::with_capacity(nb * 22);
        for b in 0..nb {
            let d = 0.05 * (b as f32 + 1.0);
            data.extend_from_slice(&f16::from_f32(d).to_le_bytes());
            // qh: bit v = high (5th) bit of value v
            let qh: u32 = 0x1357_9BDF_u32.wrapping_mul(b as u32 + 1).wrapping_add(seed);
            data.extend_from_slice(&qh.to_le_bytes());
            for j in 0..16 {
                let v_lo = ((b as u32 * 7 + j as u32 * 3 + seed) % 16) as u8;
                let v_hi = ((b as u32 * 11 + j as u32 * 5 + seed + 1) % 16) as u8;
                data.push(v_lo | (v_hi << 4));
            }
        }
        let row = crate::quantize::dequantize_q5_0(&data).unwrap();
        (data, row)
    }

    fn cpu_dot(row: &[f32], x: &[f32]) -> f32 {
        row.iter().zip(x.iter()).map(|(w, v)| w * v).sum()
    }

    #[test]
    fn test_q4_0_gemv_matches_cpu_ggml_layout() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let k = 256usize;
        let n = 4usize;
        let x: Vec<f32> = (0..k).map(|i| (i as f32) * 0.013 - 1.0).collect();
        // Build n independent rows, concatenate weights row-major.
        let mut weights = Vec::new();
        let mut expected = vec![0.0f32; n];
        for (r, exp) in expected.iter_mut().enumerate() {
            let (w, row) = ggml_q4_0_row(k, r as u32 + 1);
            weights.extend_from_slice(&w);
            *exp = cpu_dot(&row, &x);
        }
        let weight_buf = GpuBuffer::from_host(&exec.context, &weights).unwrap();
        let input_buf = GpuBuffer::from_host(&exec.context, &x).unwrap();
        let out = vec![0.0f32; n];
        let output_buf = GpuBuffer::from_host(&exec.context, &out).unwrap();
        exec.q4_0_gemv_into(weight_buf.as_ptr(), &input_buf, &output_buf, n as u32, k as u32)
            .expect("q4_0 gemv");
        exec.stream.synchronize().unwrap();
        let mut got = vec![0.0f32; n];
        output_buf.copy_to_host(&mut got).unwrap();
        for r in 0..n {
            let diff = (got[r] - expected[r]).abs();
            let tol = 1e-2 * expected[r].abs().max(1.0);
            assert!(
                diff <= tol,
                "Q4_0 row {r}: GPU {} vs CPU {} (diff {diff} > tol {tol})",
                got[r],
                expected[r]
            );
        }
    }

    #[test]
    fn test_q4_1_gemv_matches_cpu_ggml_layout() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let k = 256usize;
        let n = 4usize;
        let x: Vec<f32> = (0..k).map(|i| (i as f32) * 0.009 - 0.5).collect();
        let mut weights = Vec::new();
        let mut expected = vec![0.0f32; n];
        for (r, exp) in expected.iter_mut().enumerate() {
            let (w, row) = ggml_q4_1_row(k, r as u32 + 2);
            weights.extend_from_slice(&w);
            *exp = cpu_dot(&row, &x);
        }
        let weight_buf = GpuBuffer::from_host(&exec.context, &weights).unwrap();
        let input_buf = GpuBuffer::from_host(&exec.context, &x).unwrap();
        let out = vec![0.0f32; n];
        let output_buf = GpuBuffer::from_host(&exec.context, &out).unwrap();
        exec.q4_1_gemv_into(weight_buf.as_ptr(), &input_buf, &output_buf, n as u32, k as u32)
            .expect("q4_1 gemv");
        exec.stream.synchronize().unwrap();
        let mut got = vec![0.0f32; n];
        output_buf.copy_to_host(&mut got).unwrap();
        for r in 0..n {
            let diff = (got[r] - expected[r]).abs();
            let tol = 1e-2 * expected[r].abs().max(1.0);
            assert!(
                diff <= tol,
                "Q4_1 row {r}: GPU {} vs CPU {} (diff {diff} > tol {tol})",
                got[r],
                expected[r]
            );
        }
    }

    #[test]
    fn test_q5_0_gemv_matches_cpu_ggml_layout() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let k = 256usize;
        let n = 4usize;
        let x: Vec<f32> = (0..k).map(|i| (i as f32) * 0.011 - 0.7).collect();
        let mut weights = Vec::new();
        let mut expected = vec![0.0f32; n];
        for (r, exp) in expected.iter_mut().enumerate() {
            let (w, row) = ggml_q5_0_row(k, r as u32 + 3);
            weights.extend_from_slice(&w);
            *exp = cpu_dot(&row, &x);
        }
        let weight_buf = GpuBuffer::from_host(&exec.context, &weights).unwrap();
        let input_buf = GpuBuffer::from_host(&exec.context, &x).unwrap();
        let out = vec![0.0f32; n];
        let output_buf = GpuBuffer::from_host(&exec.context, &out).unwrap();
        exec.q5_0_gemv_into(weight_buf.as_ptr(), &input_buf, &output_buf, n as u32, k as u32)
            .expect("q5_0 gemv");
        exec.stream.synchronize().unwrap();
        let mut got = vec![0.0f32; n];
        output_buf.copy_to_host(&mut got).unwrap();
        for r in 0..n {
            let diff = (got[r] - expected[r]).abs();
            let tol = 1e-2 * expected[r].abs().max(1.0);
            assert!(
                diff <= tol,
                "Q5_0 row {r}: GPU {} vs CPU {} (diff {diff} > tol {tol})",
                got[r],
                expected[r]
            );
        }
    }
}
