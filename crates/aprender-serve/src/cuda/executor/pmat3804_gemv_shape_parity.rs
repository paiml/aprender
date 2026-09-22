// PMAT-3804: GPU↔CPU GEMV parity AT THE SHAPES A REAL MODEL USES.
//
// `q8_gemv_tests.rs` already asserts Q4_0/Q4_1/Q5_0 parity (PMAT-782), but only
// at `k=256, n=4`. qwen2.5-coder-0.5b-instruct-q4_k_m runs every projection at
// `k=896` — a width that is a multiple of the 32-element legacy block but NOT
// of the 256-element K-quant super-block, which is exactly why llama.cpp fell
// that file back to Q5_0/Q8_0 for 146 of its 291 tensors. A kernel correct at
// k=256 is not thereby correct at k=896, so the shape is part of the claim.
//
// Q8_0 had NO numerical parity test at any shape before this file: the
// `test_q8_0_gemv_into_*` cases discard the result (`let _ = result;`) and
// assert nothing, while `dtype.rs::gpu_unsupported_quant_qtype` whitelists
// Q8_0 onto the GPU as a type with a "verified GPU GEMV kernel".

#[cfg(test)]
#[cfg(feature = "cuda")]
mod pmat3804_shape_parity {
    use super::*;

    /// Build an `n`-row GGML-spec Q5_0 weight matrix plus its CPU dequant.
    fn q5_0_matrix(n: usize, k: usize) -> (Vec<u8>, Vec<Vec<f32>>) {
        use half::f16;
        let nb = k / 32;
        let mut data = Vec::with_capacity(n * nb * 22);
        let mut rows = Vec::with_capacity(n);
        for r in 0..n {
            let seed = r as u32 + 3;
            let mut row_bytes = Vec::with_capacity(nb * 22);
            for b in 0..nb {
                let d = 0.05 * ((b % 7) as f32 + 1.0);
                row_bytes.extend_from_slice(&f16::from_f32(d).to_le_bytes());
                let qh: u32 = 0x1357_9BDF_u32
                    .wrapping_mul(b as u32 + 1)
                    .wrapping_add(seed);
                row_bytes.extend_from_slice(&qh.to_le_bytes());
                for j in 0..16 {
                    let v_lo = ((b as u32 * 7 + j as u32 * 3 + seed) % 16) as u8;
                    let v_hi = ((b as u32 * 11 + j as u32 * 5 + seed + 1) % 16) as u8;
                    row_bytes.push(v_lo | (v_hi << 4));
                }
            }
            rows.push(crate::quantize::dequantize_q5_0(&row_bytes).unwrap());
            data.extend_from_slice(&row_bytes);
        }
        (data, rows)
    }

    /// Build an `n`-row GGML-spec Q8_0 weight matrix plus its CPU dequant.
    fn q8_0_matrix(n: usize, k: usize) -> (Vec<u8>, Vec<Vec<f32>>) {
        use half::f16;
        let nb = k / 32;
        let mut data = Vec::with_capacity(n * nb * 34);
        let mut rows = Vec::with_capacity(n);
        for r in 0..n {
            let seed = r as i32 + 3;
            let mut row_bytes = Vec::with_capacity(nb * 34);
            for b in 0..nb {
                let d = 0.01 * ((b % 5) as f32 + 1.0);
                row_bytes.extend_from_slice(&f16::from_f32(d).to_le_bytes());
                for j in 0..32 {
                    // Exercise the full signed range, including negatives: a
                    // kernel that reads qs as UNSIGNED passes an all-positive
                    // fixture and fails here.
                    let v = ((b as i32 * 13 + j as i32 * 7 + seed) % 255) - 127;
                    row_bytes.push(v as i8 as u8);
                }
            }
            rows.push(crate::quantize::dequantize_q8_0(&row_bytes).unwrap());
            data.extend_from_slice(&row_bytes);
        }
        (data, rows)
    }

    fn dot(row: &[f32], x: &[f32]) -> f32 {
        row.iter().zip(x.iter()).map(|(w, v)| w * v).sum()
    }

    /// Run one GEMV on the GPU and compare every output row to the CPU dequant.
    fn check(
        label: &str,
        weights: &[u8],
        rows: &[Vec<f32>],
        n: usize,
        k: usize,
        run: impl Fn(
            &mut CudaExecutor,
            u64,
            &GpuBuffer<f32>,
            &GpuBuffer<f32>,
        ) -> Result<(), GpuError>,
    ) {
        let Some(mut exec) = CudaExecutor::new(0).ok() else {
            eprintln!("{label}: no CUDA device — skipped");
            return;
        };
        let x: Vec<f32> = (0..k).map(|i| ((i % 97) as f32) * 0.011 - 0.5).collect();
        let expected: Vec<f32> = rows.iter().map(|r| dot(r, &x)).collect();

        let wbuf = GpuBuffer::from_host(&exec.context, weights).unwrap();
        let xbuf = GpuBuffer::from_host(&exec.context, &x).unwrap();
        let ybuf = GpuBuffer::from_host(&exec.context, &vec![0.0f32; n]).unwrap();

        run(&mut exec, wbuf.as_ptr(), &xbuf, &ybuf).expect("gemv launch");
        exec.stream.synchronize().unwrap();
        let mut got = vec![0.0f32; n];
        ybuf.copy_to_host(&mut got).unwrap();

        let mut worst = 0.0f32;
        let mut worst_row = 0usize;
        for r in 0..n {
            let tol = 1e-2 * expected[r].abs().max(1.0);
            let rel = (got[r] - expected[r]).abs() / tol;
            if rel > worst {
                worst = rel;
                worst_row = r;
            }
        }
        assert!(
            worst <= 1.0,
            "{label} (n={n}, k={k}): GPU diverges from CPU dequant at row {worst_row}: \
             GPU {} vs CPU {} ({worst}x tolerance)",
            got[worst_row],
            expected[worst_row],
        );
    }

    /// Q5_0 carries 133 of the 291 tensors in qwen2.5-coder-0.5b Q4_K_M —
    /// attn_q, attn_k, attn_output, ffn_gate, ffn_up and token_embd.
    #[test]
    fn q5_0_gemv_parity_at_qwen_0_5b_shapes() {
        for &(n, k, what) in &[
            (4usize, 256usize, "existing coverage (k=256)"),
            (896, 896, "attn_q / attn_output"),
            (128, 896, "attn_k"),
            (4864, 896, "ffn_gate / ffn_up"),
        ] {
            let (w, rows) = q5_0_matrix(n, k);
            check(&format!("Q5_0 {what}"), &w, &rows, n, k, |e, wp, x, y| {
                e.q5_0_gemv_into(wp, x, y, n as u32, k as u32)
            });
        }
    }

    /// Q8_0 carries the LM head and 12 attn_v tensors in the same file, and had
    /// no numerical parity test at any shape before PMAT-3804.
    #[test]
    fn q8_0_gemv_parity_at_qwen_0_5b_shapes() {
        for &(n, k, what) in &[
            (4usize, 256usize, "smoke"),
            (128, 896, "attn_v"),
            (896, 896, "square"),
            (2048, 896, "lm_head slice"),
        ] {
            let (w, rows) = q8_0_matrix(n, k);
            check(&format!("Q8_0 {what}"), &w, &rows, n, k, |e, wp, x, y| {
                e.q8_0_gemv_into(wp, x, y, n as u32, k as u32)
            });
        }
    }
}
