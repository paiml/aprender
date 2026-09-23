/// #3869: the IQ4_NL GEMV kernel against the CPU decoder, ON THE DEVICE.
///
/// This is the measurement the whitelist waits for. Everything else about this
/// kernel is checkable without a GPU — `ptxas` assembles it, and
/// `the_iq4_nl_thread_mapping_reproduces_the_cpu_decoder` runs its index math in
/// Rust — but neither compares the PTX *text* against the CPU decoder. Those are
/// two transcriptions of one intent, and only running both on the same bytes
/// tells you they agree.
///
/// Until this passes, `gpu_unsupported_quant_qtype(20)` stays true.
#[cfg(test)]
#[cfg(feature = "cuda")]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod iq4_nl_device_ab_tests {
    use super::*;
    use crate::quantize::iq4_nl::{GGML_TYPE_IQ4_NL, IQ4_NL_BLOCK_BYTES};

    fn create_executor() -> Option<CudaExecutor> {
        CudaExecutor::new(0).ok()
    }

    /// `n * ceil(k/32)` blocks of 18 bytes, deterministic, with each block's f16
    /// scale pinned to an exactly representable value so any disagreement is
    /// about INDEXING rather than rounding.
    fn iq4_nl_weights(n: usize, k: usize) -> Vec<u8> {
        let blocks_per_row = k.div_ceil(32);
        let mut data = Vec::with_capacity(n * blocks_per_row * IQ4_NL_BLOCK_BYTES);
        let mut state: u32 = 0x9E37_79B9;
        for _ in 0..(n * blocks_per_row) {
            data.push(0x00);
            data.push(0x3c); // f16 1.0
            for _ in 0..16 {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                data.push((state >> 24) as u8);
            }
        }
        data
    }

    /// The A/B. Same bytes, same activation, both decoders, compared elementwise.
    #[test]
    fn the_iq4_nl_kernel_agrees_with_the_cpu_decoder_on_device() {
        let Some(mut exec) = create_executor() else {
            eprintln!("SKIP: no CUDA device — this is the one check that needs one");
            return;
        };

        // A shape with more than one block per row, so the row stride is
        // exercised: 256 elements is 8 IQ4_NL blocks, not 1.
        let (k, n) = (256usize, 64usize);
        let weights = iq4_nl_weights(n, k);
        assert_eq!(weights.len(), n * (k / 32) * IQ4_NL_BLOCK_BYTES);

        let input: Vec<f32> = (0..k).map(|i| ((i % 13) as f32) - 6.0).collect();

        // CPU reference: the decoder verified against llama.cpp and cross-checked
        // structurally against IQ4_XS.
        let expected =
            crate::quantize::iq_parallel_matvec(GGML_TYPE_IQ4_NL, &weights, &input, k, n)
                .expect("CPU IQ4_NL matvec");

        // GPU.
        let weight_buf = GpuBuffer::from_host(&exec.context, &weights).unwrap();
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let output_buf = GpuBuffer::from_host(&exec.context, &vec![0.0f32; n]).unwrap();

        exec.iq4_nl_gemv_into(
            weight_buf.as_ptr(),
            &input_buf,
            &output_buf,
            u32::try_from(n).unwrap(),
            u32::try_from(k).unwrap(),
        )
        .expect("IQ4_NL GEMV launch");
        exec.stream.synchronize().unwrap();

        let mut got = vec![0.0f32; n];
        output_buf.copy_to_host(&mut got).unwrap();

        // Not "close enough on average": every row, with a relative tolerance
        // that fp32 accumulation order can explain and a transposed nibble
        // cannot.
        let mut worst = 0.0f32;
        let mut worst_row = 0usize;
        for (row, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            let rel = (g - e).abs() / e.abs().max(1.0);
            if rel > worst {
                worst = rel;
                worst_row = row;
            }
        }
        assert!(
            worst <= 1e-4,
            "#3869: GPU and CPU IQ4_NL disagree. worst row {worst_row}: GPU {} vs CPU {} \
             (relative {worst:.3e}). A disagreement this large is an indexing bug, not \
             accumulation order — the CPU decoder is the oracle.",
            got[worst_row],
            expected[worst_row]
        );

        // A guard against the whole thing being zeros on both sides, which would
        // agree perfectly and prove nothing.
        let nonzero = expected.iter().filter(|v| v.abs() > 1e-6).count();
        assert!(
            nonzero >= n / 2,
            "only {nonzero} of {n} reference rows are non-zero; this comparison would pass \
             on an all-zero kernel"
        );
        eprintln!("#3869 A/B: {n} rows, worst relative disagreement {worst:.3e}");
    }

    /// #3885: the A/B for Q5_1, the last blocker on
    /// `Qwen2.5-0.5B-Instruct-IQ4_XS`.
    ///
    /// Q5_1 is not an IQ type, so the oracle is `dequantize_q5_1` plus an
    /// explicit row-major dot rather than `iq_parallel_matvec`. Written out
    /// because borrowing the IQ path would have compared the kernel against a
    /// function that does not handle this type.
    #[test]
    fn the_q5_1_kernel_agrees_with_the_cpu_decoder_on_device() {
        let Some(mut exec) = create_executor() else {
            eprintln!("SKIP: no CUDA device");
            return;
        };

        // 256 elements per row is 8 Q5_1 blocks, so the row stride is exercised.
        let (k, n) = (256usize, 64usize);
        let blocks_per_row = k / 32;
        let mut weights = Vec::with_capacity(n * blocks_per_row * 24);
        let mut state: u32 = 0x0BAD_F00D;
        for _ in 0..(n * blocks_per_row) {
            weights.extend_from_slice(&[0x00, 0x3e]); // d = 1.5
            weights.extend_from_slice(&[0x00, 0xb4]); // m = -0.25, the affine term
            for _ in 4..24 {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                weights.push((state >> 24) as u8);
            }
        }
        assert_eq!(weights.len(), n * blocks_per_row * 24);

        let input: Vec<f32> = (0..k).map(|i| ((i % 13) as f32) - 6.0).collect();

        // CPU oracle: dequantize the whole tensor, then dot each row.
        let flat = crate::quantize::dequantize_q5_1(&weights).expect("CPU Q5_1 dequant");
        assert_eq!(flat.len(), n * k);
        let expected: Vec<f32> = (0..n)
            .map(|row| {
                (0..k)
                    .map(|j| flat[row * k + j] * input[j])
                    .sum::<f32>()
            })
            .collect();

        let weight_buf = GpuBuffer::from_host(&exec.context, &weights).unwrap();
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let output_buf = GpuBuffer::from_host(&exec.context, &vec![0.0f32; n]).unwrap();
        exec.q5_1_gemv_into(
            weight_buf.as_ptr(),
            &input_buf,
            &output_buf,
            u32::try_from(n).unwrap(),
            u32::try_from(k).unwrap(),
        )
        .expect("Q5_1 GEMV launch");
        exec.stream.synchronize().unwrap();

        let mut got = vec![0.0f32; n];
        output_buf.copy_to_host(&mut got).unwrap();

        let mut worst = 0.0f32;
        let mut worst_row = 0usize;
        for (row, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            let rel = (g - e).abs() / e.abs().max(1.0);
            if rel > worst {
                worst = rel;
                worst_row = row;
            }
        }
        assert!(
            worst <= 1e-4,
            "#3885: GPU and CPU Q5_1 disagree. worst row {worst_row}: GPU {} vs CPU {} \
             (relative {worst:.3e}). The CPU decoder is the oracle.",
            got[worst_row],
            expected[worst_row]
        );

        let nonzero = expected.iter().filter(|v| v.abs() > 1e-6).count();
        assert!(
            nonzero >= n / 2,
            "only {nonzero} of {n} reference rows are non-zero; this would pass on an \
             all-zero kernel"
        );
        eprintln!("#3885 A/B: {n} rows, worst relative disagreement {worst:.3e}");
    }

    /// #3884: the same A/B for IQ3_S, the last IQ type without a kernel.
    ///
    /// 110-byte super-blocks, 256 elements. `in_dim = 512` gives two blocks per
    /// row so the stride is exercised, and the activation is position-dependent
    /// so a block read at the wrong offset cannot coincidentally sum the same.
    #[test]
    fn the_iq3_s_kernel_agrees_with_the_cpu_decoder_on_device() {
        use crate::quantize::iq3_s::{GGML_TYPE_IQ3_S, IQ3_S_BLOCK_BYTES};

        let Some(mut exec) = create_executor() else {
            eprintln!("SKIP: no CUDA device");
            return;
        };

        let (k, n) = (512usize, 48usize);
        let blocks_per_row = k / 256;
        let mut weights = Vec::with_capacity(n * blocks_per_row * IQ3_S_BLOCK_BYTES);
        let mut state: u32 = 0x5EED_1234;
        for _ in 0..(n * blocks_per_row) {
            weights.push(0x00);
            weights.push(0x3c); // f16 1.0, so a disagreement is about INDEXING
            for _ in 2..IQ3_S_BLOCK_BYTES {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                weights.push((state >> 24) as u8);
            }
        }
        assert_eq!(weights.len(), n * blocks_per_row * IQ3_S_BLOCK_BYTES);

        let input: Vec<f32> = (0..k).map(|i| ((i % 17) as f32) - 8.0).collect();
        let expected = crate::quantize::iq_parallel_matvec(GGML_TYPE_IQ3_S, &weights, &input, k, n)
            .expect("CPU IQ3_S matvec");

        let weight_buf = GpuBuffer::from_host(&exec.context, &weights).unwrap();
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let output_buf = GpuBuffer::from_host(&exec.context, &vec![0.0f32; n]).unwrap();
        exec.iq3_s_gemv_into(
            weight_buf.as_ptr(),
            &input_buf,
            &output_buf,
            u32::try_from(n).unwrap(),
            u32::try_from(k).unwrap(),
        )
        .expect("IQ3_S GEMV launch");
        exec.stream.synchronize().unwrap();

        let mut got = vec![0.0f32; n];
        output_buf.copy_to_host(&mut got).unwrap();

        let mut worst = 0.0f32;
        let mut worst_row = 0usize;
        for (row, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            let rel = (g - e).abs() / e.abs().max(1.0);
            if rel > worst {
                worst = rel;
                worst_row = row;
            }
        }
        assert!(
            worst <= 1e-4,
            "#3884: GPU and CPU IQ3_S disagree. worst row {worst_row}: GPU {} vs CPU {} \
             (relative {worst:.3e}). The CPU decoder is the oracle.",
            got[worst_row],
            expected[worst_row]
        );

        let nonzero = expected.iter().filter(|v| v.abs() > 1e-6).count();
        assert!(
            nonzero >= n / 2,
            "only {nonzero} of {n} reference rows are non-zero; this would pass on an \
             all-zero kernel"
        );
        eprintln!("#3884 A/B: {n} rows, worst relative disagreement {worst:.3e}");
    }

    /// #3931: IQ2_XXS on the device, against the CPU decoder, at the shapes that
    /// exercise the stride and the tail. Returns `(worst relative disagreement,
    /// worst row)` so the planted-fault runs and the real run report the same
    /// number.
    ///
    /// Weights come from `quantize::iq2_xxs_geometry_tests::iq2_xxs_weights`, the
    /// same builder the CPU-side oracle test uses, so the oracle and the A/B cannot
    /// drift. Every block's f16 scale is exactly 1.0: a disagreement is INDEXING.
    fn iq2_xxs_ab(exec: &mut CudaExecutor, k: usize, n: usize) -> (f32, usize, Vec<f32>, Vec<f32>) {
        use crate::quantize::iq2_xxs::{GGML_TYPE_IQ2_XXS, IQ2_XXS_BLOCK_BYTES};
        let weights = crate::quantize::iq2_xxs_geometry_tests::iq2_xxs_weights(n, k);
        assert_eq!(
            weights.len(),
            n * k.div_ceil(256) * IQ2_XXS_BLOCK_BYTES,
            "the buffer must be (n * ceil(k/256) * 66) bytes — the geometry precondition"
        );
        let input: Vec<f32> = (0..k).map(|i| ((i % 17) as f32) - 8.0).collect();
        let expected =
            crate::quantize::iq_parallel_matvec(GGML_TYPE_IQ2_XXS, &weights, &input, k, n)
                .expect("CPU IQ2_XXS matvec — the oracle");

        let weight_buf = GpuBuffer::from_host(&exec.context, &weights).unwrap();
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        // NaN-prefilled: an unwritten row compares as NaN and can never pass.
        let output_buf = GpuBuffer::from_host(&exec.context, &vec![f32::NAN; n]).unwrap();
        exec.iq2_xxs_gemv_into(
            weight_buf.as_ptr(),
            &input_buf,
            &output_buf,
            u32::try_from(n).unwrap(),
            u32::try_from(k).unwrap(),
        )
        .expect("IQ2_XXS GEMV launch");
        exec.stream.synchronize().unwrap();
        let mut got = vec![0.0f32; n];
        output_buf.copy_to_host(&mut got).unwrap();

        let mut worst = 0.0f32;
        let mut worst_row = 0usize;
        for (row, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            let rel = (g - e).abs() / e.abs().max(1.0);
            if !(rel <= worst) {
                worst = rel;
                worst_row = row;
            }
        }
        (worst, worst_row, got, expected)
    }

    /// #3931: THE measurement the IQ2_XXS whitelist waits for.
    ///
    /// Until this passes on a real device, with its planted faults proven RED
    /// first, `gpu_unsupported_quant_qtype(16)` stays true.
    #[test]
    fn the_iq2_xxs_kernel_agrees_with_the_cpu_decoder_on_device() {
        let Some(mut exec) = create_executor() else {
            eprintln!("SKIP: no CUDA device");
            return;
        };
        // 512: two super-blocks per row, so the row stride (nb * 66) is exercised.
        // 300: the second block is 44/256 used, so the k_dim guard is exercised.
        for (k, n) in [(512usize, 64usize), (300, 16)] {
            let (worst, row, got, expected) = iq2_xxs_ab(&mut exec, k, n);
            assert!(
                worst <= 1e-4,
                "#3931: GPU and CPU IQ2_XXS disagree at k={k}. worst row {row}: GPU {} vs \
                 CPU {} (relative {worst:.3e}). The CPU decoder is the oracle.",
                got[row],
                expected[row]
            );
            // Positive controls: an oracle of zeros agrees with a kernel of zeros,
            // and identical rows agree with a kernel that ignores the row index.
            let nonzero = expected.iter().filter(|v| v.abs() > 1e-6).count();
            assert!(nonzero >= n / 2, "k={k}: only {nonzero}/{n} oracle rows non-zero");
            let distinct: std::collections::BTreeSet<u32> =
                expected.iter().map(|v| v.to_bits()).collect();
            assert!(
                distinct.len() >= n / 2,
                "k={k}: only {} of {n} oracle rows distinct; a kernel ignoring the row \
                 index would still match",
                distinct.len()
            );
            // Capability cell, not a verdict (operator doctrine 2026-09-23).
            eprintln!(
                "CELL (synthetic IQ2_XXS, q=16, cuda sm_89, gemv k={k} n={n}) -> worst rel \
                 disagreement {worst:.3e} at row {row}; oracle = \
                 quantize::iq_parallel_matvec (llama.cpp df03399 + gguf-py validated); \
                 {nonzero}/{n} rows non-zero, {} distinct",
                distinct.len()
            );
        }
    }

    /// #3931: the A/B on a REAL MODEL'S BYTES — every IQ2_XXS tensor in the file,
    /// not a synthetic buffer.
    ///
    /// The F16 and IQ4_XS admissions quote per-tensor counts "enumerated from the
    /// files", but only the resulting sha was committed, not the harness, so the
    /// measurement could not be re-run. This is the harness, kept.
    ///
    /// Gated on `APR_IQ_AB_MODEL` (a GGUF path) because it needs a model file. It
    /// does NOT pass silently without one: unset, it prints SKIP naming the
    /// variable, and set to a file holding no IQ2_XXS tensor it FAILS — a run that
    /// measured nothing is not evidence.
    ///
    /// Positive control on the real bytes: for the first tensor, a copy whose
    /// every block scale is corrupted is uploaded and MUST disagree with the CPU
    /// decoder on the original. Otherwise agreement on real data could be a
    /// comparison that cannot fail.
    ///
    /// Run:  APR_IQ_AB_MODEL=/path/model.gguf cargo test -p aprender-serve --lib \
    ///         --features cuda every_iq2_xxs_tensor_in_a_real_model -- --ignored --nocapture
    #[test]
    #[ignore = "needs a GGUF via APR_IQ_AB_MODEL and a CUDA device"]
    fn every_iq2_xxs_tensor_in_a_real_model_agrees_on_device() {
        use crate::quantize::iq2_xxs::{dequantize_iq2_xxs, GGML_TYPE_IQ2_XXS, IQ2_XXS_BLOCK_BYTES};
        // Admission rule (cop, from aprender-70's IQ4_XS finding, 2026-09-23):
        // cover EVERY (k, n) shape the model uses for this type, with real bytes
        // per shape; per-row error normalised by the row's own magnitude,
        //   |gpu - cpu| / sum_c |w[r][c]| * |x[c]|  <= 1e-5,
        // and the output PRE-FILLED WITH NaN so a row the kernel never writes
        // cannot pass as a coincidental zero.
        const TOL: f64 = 1e-5;
        let Ok(path) = std::env::var("APR_IQ_AB_MODEL") else {
            eprintln!("SKIP: set APR_IQ_AB_MODEL to a GGUF holding IQ2_XXS tensors");
            return;
        };
        let Some(mut exec) = create_executor() else {
            panic!("APR_IQ_AB_MODEL is set but there is no CUDA device: nothing was measured");
        };
        let mapped = crate::gguf::MappedGGUFModel::from_path(&path).expect("map the GGUF");
        let base = mapped.model.tensor_data_start;
        let tensors: Vec<_> = mapped
            .model
            .tensors
            .iter()
            .filter(|t| t.qtype == GGML_TYPE_IQ2_XXS && t.dims.len() == 2)
            .collect();
        assert!(!tensors.is_empty(), "{path} holds no 2-D IQ2_XXS tensor: nothing would be measured");

        // (k, n) -> (tensor count, worst normalised error, the tensor that set it)
        let mut shapes: std::collections::BTreeMap<(usize, usize), (usize, f64, String)> =
            std::collections::BTreeMap::new();
        for (ti, t) in tensors.iter().enumerate() {
            // GGUF dims: ne0 is the contiguous (in) dimension, ne1 the rows.
            let k = usize::try_from(t.dims[0]).unwrap();
            let n = usize::try_from(t.dims[1]).unwrap();
            let nb = k.div_ceil(256);
            let bytes = n * nb * IQ2_XXS_BLOCK_BYTES;
            let start = base + usize::try_from(t.offset).unwrap();
            let weights = &mapped.mmap[start..start + bytes];

            let input: Vec<f32> =
                (0..k).map(|i| (((i * 7 + ti) % 23) as f32 - 11.0) * 0.125).collect();
            let expected =
                crate::quantize::iq_parallel_matvec(GGML_TYPE_IQ2_XXS, weights, &input, k, n)
                    .expect("CPU IQ2_XXS matvec");
            let dense = dequantize_iq2_xxs(weights).expect("dequantize the real tensor");
            let denom: Vec<f64> = (0..n)
                .map(|r| {
                    (0..k)
                        .map(|c| f64::from(dense[r * nb * 256 + c].abs()) * f64::from(input[c].abs()))
                        .sum()
                })
                .collect();

            let run = |exec: &mut CudaExecutor, w: &[u8]| -> Vec<f32> {
                let wb = GpuBuffer::from_host(&exec.context, w).unwrap();
                let ib = GpuBuffer::from_host(&exec.context, &input).unwrap();
                let ob = GpuBuffer::from_host(&exec.context, &vec![f32::NAN; n]).unwrap();
                exec.iq2_xxs_gemv_into(
                    wb.as_ptr(), &ib, &ob,
                    u32::try_from(n).unwrap(), u32::try_from(k).unwrap(),
                )
                .expect("IQ2_XXS GEMV launch");
                exec.stream.synchronize().unwrap();
                let mut got = vec![0.0f32; n];
                ob.copy_to_host(&mut got).unwrap();
                got
            };
            // Worst normalised error; NaN (an unwritten row) is +inf, never a pass.
            let norm_worst = |got: &[f32]| -> (f64, usize) {
                let mut worst = (0.0f64, 0usize);
                for (r, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
                    let err = if g.is_finite() { f64::from((g - e).abs()) } else { f64::INFINITY };
                    let m = if denom[r] > 0.0 { err / denom[r] } else if err == 0.0 { 0.0 } else { f64::INFINITY };
                    if m > worst.0 { worst = (m, r); }
                }
                worst
            };

            let got = run(&mut exec, weights);
            let unwritten = got.iter().filter(|v| v.is_nan()).count();
            assert_eq!(unwritten, 0, "{}: {unwritten} of {n} rows never written (still NaN)", t.name);
            let (worst, row) = norm_worst(&got);
            assert!(
                worst <= TOL,
                "{}: [{n} x {k}] row {row}: GPU {} vs CPU {}, |err|/sum|w||x| = {worst:.3e} > {TOL:e}",
                t.name, got[row], expected[row]
            );
            let e = shapes.entry((k, n)).or_insert((0, 0.0, t.name.clone()));
            e.0 += 1;
            if worst > e.1 { e.1 = worst; e.2 = t.name.clone(); }

            if ti == 0 {
                // Positive control on the real bytes: corrupt every block's f16 scale.
                let mut bad = weights.to_vec();
                for blk in bad.chunks_exact_mut(IQ2_XXS_BLOCK_BYTES) {
                    blk[1] ^= 0x40;
                }
                let (control, _) = norm_worst(&run(&mut exec, &bad));
                assert!(
                    control > 1e-2,
                    "POSITIVE CONTROL FAILED on {}: corrupted scales still agree ({control:.3e})",
                    t.name
                );
                eprintln!("CONTROL {}: corrupted block scales -> |err|/sum|w||x| {control:.3e} (RED, as required)", t.name);
            }
        }
        for ((k, n), (count, worst, name)) in &shapes {
            eprintln!(
                "CELL ({path}, q=16 IQ2_XXS, cuda, gemv k={k} n={n}) -> {count} tensor(s), \
                 worst |err|/sum|w||x| {worst:.3e} ({name}); NaN-prefilled, 0 unwritten; \
                 oracle = quantize::iq_parallel_matvec (llama.cpp df03399 + gguf-py validated)"
            );
        }
        eprintln!("SHAPES: {} distinct (k, n) over {} tensors", shapes.len(), tensors.len());
    }

    /// The same comparison at a shape whose rows do NOT divide evenly, so the
    /// kernel's `x_idx >= k_dim` guard is exercised rather than assumed.
    #[test]
    fn the_iq4_nl_kernel_handles_a_row_that_is_not_a_whole_number_of_blocks() {
        let Some(mut exec) = create_executor() else {
            eprintln!("SKIP: no CUDA device");
            return;
        };

        // 100 is not a multiple of 32: 4 blocks per row, the last one padding.
        let (k, n) = (100usize, 16usize);
        let weights = iq4_nl_weights(n, k);
        let input: Vec<f32> = (0..k).map(|i| ((i % 7) as f32) - 3.0).collect();

        let expected =
            crate::quantize::iq_parallel_matvec(GGML_TYPE_IQ4_NL, &weights, &input, k, n)
                .expect("CPU IQ4_NL matvec");

        let weight_buf = GpuBuffer::from_host(&exec.context, &weights).unwrap();
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let output_buf = GpuBuffer::from_host(&exec.context, &vec![0.0f32; n]).unwrap();
        exec.iq4_nl_gemv_into(
            weight_buf.as_ptr(),
            &input_buf,
            &output_buf,
            u32::try_from(n).unwrap(),
            u32::try_from(k).unwrap(),
        )
        .expect("IQ4_NL GEMV launch");
        exec.stream.synchronize().unwrap();

        let mut got = vec![0.0f32; n];
        output_buf.copy_to_host(&mut got).unwrap();

        for (row, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() <= 1e-4 * e.abs().max(1.0),
                "row {row} at k=100 (a partial last block): GPU {g} vs CPU {e}. The kernel \
                 must stop at k_dim and not read its padding."
            );
        }
    }
}
