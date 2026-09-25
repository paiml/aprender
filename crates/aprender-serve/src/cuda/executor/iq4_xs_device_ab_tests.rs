/// #3951: the IQ4_XS GEMV kernel against the CPU decoder, ON THE DEVICE, at the
/// shapes the model that fails actually uses.
///
/// IQ4_XS (qtype 23) is on the GPU whitelist on this evidence
/// (`gemv_entry_name_tests_3477.rs`): "10/10 IQ4_XS tensors exact, cosine 1.00000000
/// … every IQ4_XS tensor in Qwen3.5-4B-UD-Q4_K_XL, `[2560, 9216]` each". All ten have
/// ONE shape. `Qwen3.5-0.8B-IQ4_XS` — the model whose thinking-ON leg never closes —
/// carries 126 IQ4_XS tensors at EIGHT shapes, none of them `[2560, 9216]`, including
/// `ffn_down` at k = 3584: fourteen super-blocks per row, the only non-power-of-two
/// count. A warp-reduce GEMV can be exact at one `(k, n)` and wrong at another, so the
/// admission licensed a shape and the whitelist admitted a type.
///
/// The IQ4_NL kernel has a device A/B (`iq4_nl_device_ab_tests.rs`); IQ4_XS had none.
///
/// Oracle: `iq_parallel_matvec(GGML_TYPE_IQ4_XS, …)`, whose decoder is bit-exact against
/// gguf-py over every IQ4 tensor of two real models (aprender-9b, #3947).
///
/// Tolerance is CONDITION-AWARE: each row's error is divided by Σ|w_ij|·|x_j|, the
/// largest value that row's dot product could take. fp32 accumulation order moves a
/// result by ~1e-7 of that; a wrong block, scale or nibble moves it by O(1e-1). A fixed
/// relative tolerance on the RESULT would either pass real bugs on rows that cancel to
/// near zero or fail correct kernels on them.
#[cfg(test)]
#[cfg(feature = "cuda")]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod iq4_xs_device_ab_tests {
    use super::*;
    use crate::quantize::iq4_xs::{dequantize_iq4_xs, GGML_TYPE_IQ4_XS, IQ4_XS_BLOCK_BYTES};

    const QK: usize = 256;
    /// Far above fp32 reordering (~1e-7 of Σ|w||x|), far below any decoding error.
    const TOL: f32 = 1e-5;

    fn create_executor() -> Option<CudaExecutor> {
        CudaExecutor::new(0).ok()
    }

    /// `n * k/256` blocks of 136 bytes. `d` is pinned to f16 1.0 so every value is
    /// finite; `scales_h`, `scales_l` and `qs` are free bytes, and every byte pattern
    /// is a valid IQ4_XS block, so the kernel sees the full range of sub-block scales.
    fn iq4_xs_weights(n: usize, k: usize, seed: u32) -> Vec<u8> {
        let blocks = n * (k / QK);
        let mut data = Vec::with_capacity(blocks * IQ4_XS_BLOCK_BYTES);
        let mut state = seed;
        for _ in 0..blocks {
            data.push(0x00);
            data.push(0x3c); // d = f16 1.0
            for _ in 0..(IQ4_XS_BLOCK_BYTES - 2) {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                data.push((state >> 24) as u8);
            }
        }
        data
    }

    /// One A/B at `(k, n)`. Returns the worst condition-scaled error.
    fn ab(exec: &mut CudaExecutor, label: &str, weights: &[u8], k: usize, n: usize) -> f32 {
        ab_with_device_bytes(exec, label, weights, weights, k, n)
    }

    /// The A/B with the GPU fed `device` and the CPU oracle fed `weights`. Equal slices
    /// for a real run; one corrupted block for the negative control.
    fn ab_with_device_bytes(
        exec: &mut CudaExecutor,
        label: &str,
        weights: &[u8],
        device: &[u8],
        k: usize,
        n: usize,
    ) -> f32 {
        assert_eq!(k % QK, 0, "{label}: k={k} is not a multiple of 256");
        assert_eq!(weights.len(), n * (k / QK) * IQ4_XS_BLOCK_BYTES, "{label}: byte count");

        let input: Vec<f32> = (0..k).map(|i| (((i * 7 + 3) % 17) as f32 - 8.0) / 4.0).collect();

        let expected = crate::quantize::iq_parallel_matvec(GGML_TYPE_IQ4_XS, weights, &input, k, n)
            .expect("CPU IQ4_XS matvec");
        let dense = dequantize_iq4_xs(weights).expect("CPU IQ4_XS dequant");
        assert_eq!(dense.len(), n * k, "{label}: dequant length");

        let w_buf = GpuBuffer::from_host(&exec.context, device).unwrap();
        let x_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let y_buf = GpuBuffer::from_host(&exec.context, &vec![f32::NAN; n]).unwrap();
        exec.iq4_xs_gemv_into(
            w_buf.as_ptr(),
            &x_buf,
            &y_buf,
            u32::try_from(n).unwrap(),
            u32::try_from(k).unwrap(),
        )
        .expect("IQ4_XS GEMV launch");
        exec.stream.synchronize().unwrap();
        let mut got = vec![0.0f32; n];
        y_buf.copy_to_host(&mut got).unwrap();

        let mut worst = 0.0f32;
        let mut worst_row = 0usize;
        let mut nonfinite = 0usize;
        for row in 0..n {
            let bound: f32 = dense[row * k..(row + 1) * k]
                .iter()
                .zip(&input)
                .map(|(w, x)| (w * x).abs())
                .sum();
            if !got[row].is_finite() {
                nonfinite += 1;
                continue;
            }
            let err = (got[row] - expected[row]).abs() / bound.max(1e-6);
            if err > worst {
                worst = err;
                worst_row = row;
            }
        }
        assert_eq!(
            nonfinite, 0,
            "{label} k={k} n={n}: {nonfinite} rows came back non-finite — the output buffer is \
             pre-filled with NaN, so these rows were never written"
        );
        let nonzero = expected.iter().filter(|v| v.abs() > 1e-6).count();
        assert!(nonzero >= n / 2, "{label}: only {nonzero}/{n} reference rows non-zero — vacuous");
        eprintln!(
            "#3951 {label:<22} k={k:5} n={n:5} blocks/row={:2}  worst={worst:.3e} (row {worst_row}: GPU {} CPU {})",
            k / QK,
            got[worst_row],
            expected[worst_row]
        );
        worst
    }

    /// The admitted shape first, as the control: it was measured exact, so if this is
    /// red the harness is wrong, not the kernel. Then every shape the 0.8B uses.
    #[test]
    fn the_iq4_xs_kernel_agrees_with_the_cpu_decoder_at_every_shape_the_failing_model_uses() {
        let Some(mut exec) = create_executor() else {
            eprintln!("SKIP: no CUDA device — this is the one check that needs one");
            return;
        };
        let shapes: [(&str, usize, usize); 9] = [
            ("ADMITTED 4B ffn", 2560, 9216),
            ("attn_gate", 1024, 2048),
            ("attn_qkv", 1024, 6144),
            ("ffn_down", 3584, 1024),
            ("ffn_gate/up", 1024, 3584),
            ("attn_k", 1024, 512),
            ("attn_output", 2048, 1024),
            ("attn_q", 1024, 4096),
            ("single block row", 256, 64),
        ];
        let mut failures = Vec::new();
        for (i, (label, k, n)) in shapes.iter().enumerate() {
            let w = iq4_xs_weights(*n, *k, 0x9E37_79B9 ^ (i as u32).wrapping_mul(0x85EB_CA6B));
            let worst = ab(&mut exec, label, &w, *k, *n);
            if worst > TOL {
                failures.push(format!("{label} k={k} n={n}: worst {worst:.3e}"));
            }
        }
        assert!(
            failures.is_empty(),
            "#3951: the IQ4_XS GPU kernel disagrees with the CPU decoder beyond {TOL:e} of the \
             row bound at: {failures:?}. The CPU decoder is bit-exact vs gguf-py (#3947)."
        );
    }

    /// The same A/B on the model's REAL bytes, when a dump directory is supplied:
    /// `APR_IQ4XS_DUMP_DIR` holding `<tensor>__k<k>__n<n>.bin` (raw block bytes, as in
    /// the GGUF). Synthetic bytes exercise every scale and nibble; real bytes are what
    /// the model actually multiplies. SKIPs without the variable.
    #[test]
    fn the_iq4_xs_kernel_agrees_with_the_cpu_decoder_on_real_tensors() {
        let Ok(dir) = std::env::var("APR_IQ4XS_DUMP_DIR") else {
            eprintln!("SKIP: APR_IQ4XS_DUMP_DIR unset");
            return;
        };
        let Some(mut exec) = create_executor() else {
            eprintln!("SKIP: no CUDA device");
            return;
        };
        let mut entries: Vec<_> = std::fs::read_dir(&dir)
            .expect("dump dir")
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "bin"))
            .collect();
        entries.sort();
        assert!(!entries.is_empty(), "no .bin tensors in {dir} — would pass vacuously");
        let mut failures = Vec::new();
        for p in &entries {
            let stem = p.file_stem().unwrap().to_string_lossy().into_owned();
            let parts: Vec<&str> = stem.split("__").collect();
            let k: usize = parts[1].trim_start_matches('k').parse().unwrap();
            let n: usize = parts[2].trim_start_matches('n').parse().unwrap();
            let bytes = std::fs::read(p).unwrap();
            let worst = ab(&mut exec, parts[0], &bytes, k, n);
            if worst > TOL {
                failures.push(format!("{} k={k} n={n}: worst {worst:.3e}", parts[0]));
            }
        }
        assert!(failures.is_empty(), "#3951 real tensors disagree: {failures:?}");
    }

    /// NEGATIVE CONTROL. A green comparison licenses nothing until it has been seen to go
    /// red. Corrupt ONE 136-byte block — one row, one super-block, 1/14th of that row — in
    /// the GPU's copy only, at the shape most likely to hide it (ffn_down, 14 blocks per
    /// row), and require the harness to report it. If this passes silently, the tolerance
    /// or the bound is too loose to have caught a real kernel defect.
    #[test]
    fn the_harness_goes_red_on_a_single_corrupted_block() {
        let Some(mut exec) = create_executor() else {
            eprintln!("SKIP: no CUDA device");
            return;
        };
        let (k, n) = (3584usize, 1024usize);
        let weights = iq4_xs_weights(n, k, 0x5EED_1234);
        let mut device = weights.clone();
        let row = 517usize;
        let block_in_row = 9usize;
        let at = (row * (k / QK) + block_in_row) * IQ4_XS_BLOCK_BYTES;
        for b in &mut device[at + 8..at + IQ4_XS_BLOCK_BYTES] {
            *b = !*b; // flip every nibble of that block's quants; scales untouched
        }
        let worst = ab_with_device_bytes(&mut exec, "PLANTED 1 block", &weights, &device, k, n);
        assert!(
            worst > TOL,
            "#3951 negative control: one corrupted block in {} (row {row}) produced worst \
             {worst:.3e} <= {TOL:e}. The harness cannot see a single-block defect, so its \
             greens license nothing.",
            n * (k / QK)
        );
    }
}
