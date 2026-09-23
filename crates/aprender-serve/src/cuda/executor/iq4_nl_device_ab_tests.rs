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
    /// A GEMV launcher for one IQ type: `(exec, weight_ptr, input, output, n, k)`.
    type IqLaunch =
        fn(&mut CudaExecutor, u64, &GpuBuffer<f32>, &GpuBuffer<f32>, u32, u32) -> Result<(), GpuError>;

    /// #3950/#3963: the real-bytes A/B for ANY IQ type this crate decodes on the
    /// CPU — one audited implementation instead of a copy per kernel.
    ///
    /// Admission rule (cop, after IQ4_XS was admitted on one shape): cover EVERY
    /// (k, n) shape the model uses for the type, with real bytes; per-row error
    /// normalised by the row's own magnitude,
    ///   |gpu - cpu| / sum_c |w[r][c]| * |x[c]|  <= 1e-5,
    /// and the output PRE-FILLED WITH NaN so an unwritten row cannot pass.
    ///
    /// Positive control on the real bytes of the first tensor: every block's f16
    /// scale corrupted on the GPU copy must disagree.
    ///
    /// Gated on `APR_IQ_AB_MODEL`. Unset: SKIP naming it. Set to a file with no
    /// tensor of the type: FAIL — a run that measured nothing is not evidence.
    fn real_model_iq_ab(qtype: u32, type_name: &str, launch: IqLaunch) {
        const TOL: f64 = 1e-5;
        let Ok(path) = std::env::var("APR_IQ_AB_MODEL") else {
            eprintln!("SKIP: set APR_IQ_AB_MODEL to a GGUF holding {type_name} tensors");
            return;
        };
        let Some(mut exec) = create_executor() else {
            panic!("APR_IQ_AB_MODEL is set but there is no CUDA device: nothing was measured");
        };
        let block_bytes = crate::quantize::iq_dispatch::iq_block_bytes(qtype).expect("an IQ type");
        let block_elems = crate::quantize::iq_dispatch::iq_block_elems(qtype).expect("an IQ type");
        let mapped = crate::gguf::MappedGGUFModel::from_path(&path).expect("map the GGUF");
        let base = mapped.model.tensor_data_start;
        let tensors: Vec<_> = mapped
            .model
            .tensors
            .iter()
            .filter(|t| t.qtype == qtype && t.dims.len() == 2)
            .collect();
        assert!(!tensors.is_empty(), "{path} holds no 2-D {type_name} tensor: nothing would be measured");

        // (k, n) -> (tensor count, worst normalised error, the tensor that set it)
        let mut shapes: std::collections::BTreeMap<(usize, usize), (usize, f64, String)> =
            std::collections::BTreeMap::new();
        for (ti, t) in tensors.iter().enumerate() {
            // GGUF dims: ne0 is the contiguous (in) dimension, ne1 the rows.
            let k = usize::try_from(t.dims[0]).unwrap();
            let n = usize::try_from(t.dims[1]).unwrap();
            let nb = k.div_ceil(block_elems);
            let start = base + usize::try_from(t.offset).unwrap();
            let weights = &mapped.mmap[start..start + n * nb * block_bytes];

            let input: Vec<f32> =
                (0..k).map(|i| (((i * 7 + ti) % 23) as f32 - 11.0) * 0.125).collect();
            let expected = crate::quantize::iq_parallel_matvec(qtype, weights, &input, k, n)
                .expect("CPU matvec — the oracle");
            let mut row = vec![0.0f32; nb * block_elems];
            let denom: Vec<f64> = (0..n)
                .map(|r| {
                    for b in 0..nb {
                        let off = (r * nb + b) * block_bytes;
                        crate::quantize::iq_dispatch::dequantize_iq_block(
                            qtype,
                            &weights[off..off + block_bytes],
                            &mut row[b * block_elems..(b + 1) * block_elems],
                        )
                        .expect("decode");
                    }
                    (0..k).map(|c| f64::from(row[c].abs()) * f64::from(input[c].abs())).sum()
                })
                .collect();

            let run = |exec: &mut CudaExecutor, w: &[u8]| -> Vec<f32> {
                let wb = GpuBuffer::from_host(&exec.context, w).unwrap();
                let ib = GpuBuffer::from_host(&exec.context, &input).unwrap();
                let ob = GpuBuffer::from_host(&exec.context, &vec![f32::NAN; n]).unwrap();
                launch(exec, wb.as_ptr(), &ib, &ob, u32::try_from(n).unwrap(), u32::try_from(k).unwrap())
                    .expect("GEMV launch");
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
                    if m > worst.0 {
                        worst = (m, r);
                    }
                }
                worst
            };

            let got = run(&mut exec, weights);
            let unwritten = got.iter().filter(|v| v.is_nan()).count();
            assert_eq!(unwritten, 0, "{}: {unwritten} of {n} rows never written (still NaN)", t.name);
            let (worst, r) = norm_worst(&got);
            assert!(
                worst <= TOL,
                "{}: [{n} x {k}] row {r}: GPU {} vs CPU {}, |err|/sum|w||x| = {worst:.3e} > {TOL:e}",
                t.name, got[r], expected[r]
            );
            let e = shapes.entry((k, n)).or_insert((0, 0.0, t.name.clone()));
            e.0 += 1;
            if worst > e.1 {
                e.1 = worst;
                e.2 = t.name.clone();
            }

            if ti == 0 {
                let mut bad = weights.to_vec();
                for blk in bad.chunks_exact_mut(block_bytes) {
                    blk[1] ^= 0x40; // the f16 scale's exponent
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
                "CELL ({path}, q={qtype} {type_name}, cuda, gemv k={k} n={n}) -> {count} tensor(s), \
                 worst |err|/sum|w||x| {worst:.3e} ({name}); NaN-prefilled, 0 unwritten; \
                 oracle = quantize::iq_parallel_matvec (bit-exact vs gguf-py, iq_gguf_py_parity_tests)"
            );
        }
        eprintln!("SHAPES: {} distinct (k, n) over {} tensors", shapes.len(), tensors.len());
    }

    /// #3950: every IQ2_XXS tensor of a real model, every shape. See `real_model_iq_ab`.
    ///
    /// Run:  APR_IQ_AB_MODEL=/path/model.gguf cargo test -p aprender-serve --lib \
    ///         --features cuda every_iq2_xxs_tensor_in_a_real_model -- --ignored --nocapture
    #[test]
    #[ignore = "needs a GGUF via APR_IQ_AB_MODEL and a CUDA device"]
    fn every_iq2_xxs_tensor_in_a_real_model_agrees_on_device() {
        real_model_iq_ab(crate::quantize::iq2_xxs::GGML_TYPE_IQ2_XXS, "IQ2_XXS", CudaExecutor::iq2_xxs_gemv_into);
    }

    /// #3963: every IQ3_XXS tensor of a real model, every shape. See `real_model_iq_ab`.
    #[test]
    #[ignore = "needs a GGUF via APR_IQ_AB_MODEL and a CUDA device"]
    fn every_iq3_xxs_tensor_in_a_real_model_agrees_on_device() {
        real_model_iq_ab(crate::quantize::iq3_xxs::GGML_TYPE_IQ3_XXS, "IQ3_XXS", CudaExecutor::iq3_xxs_gemv_into);
    }

    /// #3963: IQ3_XXS on synthetic blocks — f16 scale 1.0 so a disagreement is
    /// INDEXING — at k=512 (two super-blocks: the 98-byte stride) and k=300 (a
    /// padded tail: the k_dim guard). NaN-prefilled; >= n/2 rows non-zero and
    /// distinct, so neither a zero kernel nor a row-blind one can pass.
    #[test]
    fn the_iq3_xxs_kernel_agrees_with_the_cpu_decoder_on_device() {
        use crate::quantize::iq3_xxs::{GGML_TYPE_IQ3_XXS, IQ3_XXS_BLOCK_BYTES};
        let Some(mut exec) = create_executor() else {
            eprintln!("SKIP: no CUDA device");
            return;
        };
        for (k, n) in [(512usize, 64usize), (300, 16)] {
            let nb = k.div_ceil(256);
            let mut weights = Vec::with_capacity(n * nb * IQ3_XXS_BLOCK_BYTES);
            let mut state: u32 = 0x3C3C_5EED;
            for _ in 0..(n * nb) {
                weights.extend_from_slice(&[0x00, 0x3c]); // f16 1.0
                for _ in 2..IQ3_XXS_BLOCK_BYTES {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    weights.push((state >> 24) as u8);
                }
            }
            assert_eq!(weights.len(), n * nb * IQ3_XXS_BLOCK_BYTES, "geometry: n * ceil(k/256) * 98");
            let input: Vec<f32> = (0..k).map(|i| ((i % 17) as f32) - 8.0).collect();
            let expected = crate::quantize::iq_parallel_matvec(GGML_TYPE_IQ3_XXS, &weights, &input, k, n)
                .expect("CPU IQ3_XXS matvec");
            let wb = GpuBuffer::from_host(&exec.context, &weights).unwrap();
            let ib = GpuBuffer::from_host(&exec.context, &input).unwrap();
            let ob = GpuBuffer::from_host(&exec.context, &vec![f32::NAN; n]).unwrap();
            exec.iq3_xxs_gemv_into(wb.as_ptr(), &ib, &ob, u32::try_from(n).unwrap(), u32::try_from(k).unwrap())
                .expect("IQ3_XXS GEMV launch");
            exec.stream.synchronize().unwrap();
            let mut got = vec![0.0f32; n];
            ob.copy_to_host(&mut got).unwrap();
            let mut worst = (0.0f32, 0usize);
            for (row, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
                let rel = (g - e).abs() / e.abs().max(1.0);
                if !(rel <= worst.0) {
                    worst = (rel, row);
                }
            }
            assert!(
                worst.0 <= 1e-4,
                "#3963: GPU and CPU IQ3_XXS disagree at k={k}. worst row {}: GPU {} vs CPU {} (relative {:.3e})",
                worst.1, got[worst.1], expected[worst.1], worst.0
            );
            let nonzero = expected.iter().filter(|v| v.abs() > 1e-6).count();
            let distinct: std::collections::BTreeSet<u32> = expected.iter().map(|v| v.to_bits()).collect();
            assert!(nonzero >= n / 2 && distinct.len() >= n / 2, "k={k}: degenerate oracle");
            eprintln!(
                "CELL (synthetic IQ3_XXS, q=18, cuda sm_89, gemv k={k} n={n}) -> worst rel {:.3e} at row {}; \
                 {nonzero}/{n} non-zero, {} distinct",
                worst.0, worst.1, distinct.len()
            );
        }
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

    // ========================================================================
    // #3908: BF16 (ggml type 30).

    /// A bf16 whose value is the integer `v`, exactly.
    ///
    /// bf16 is the top 16 bits of an f32 and carries 7 explicit mantissa bits,
    /// so every integer up to 256 is exact and the low 16 bits of the f32 are
    /// zero -- the truncation loses nothing.
    fn bf16_int(v: i32) -> u16 {
        let bits = (v as f32).to_bits();
        debug_assert_eq!(bits & 0x0000_ffff, 0, "{v} is not exact in bf16");
        (bits >> 16) as u16
    }

    /// `n * k` bf16 weights whose decoded values are small integers.
    fn bf16_integer_weights(n: usize, k: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(n * k * 2);
        let mut state: u32 = 0x5EED_1234;
        for _ in 0..(n * k) {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let v = ((state >> 24) % 17) as i32 - 8; // [-8, 8]
            data.extend_from_slice(&bf16_int(v).to_le_bytes());
        }
        data
    }

    /// `n * k` bf16 weights spanning ordinary magnitudes, never NaN/Inf.
    ///
    /// The exponent is pinned into [0x70, 0x84] so the values sit in roughly
    /// [1e-4, 1e2]: a random 8-bit exponent would produce Inf and NaN, and a
    /// comparison against those tells you nothing about indexing.
    fn bf16_float_weights(n: usize, k: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(n * k * 2);
        let mut state: u32 = 0x0C0F_FEE1;
        for _ in 0..(n * k) {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let sign = (state >> 31) & 1;
            let exp = 0x70 + ((state >> 16) % 0x15);
            let mant = (state >> 8) & 0x7f;
            let bits = ((sign << 15) | (exp << 7) | mant) as u16;
            data.extend_from_slice(&bits.to_le_bytes());
        }
        data
    }

    /// Row-major dot of decoded weights with `x`, the same shape the CPU
    /// inference path computes (`float16_matmul`: `start = row * in_dim * 2`).
    fn bf16_reference(weights: &[u8], input: &[f32], k: usize, n: usize) -> Vec<f32> {
        let flat = crate::inference::simd_bf16_to_f32(weights);
        assert_eq!(flat.len(), n * k, "decoder returned the wrong element count");
        (0..n)
            .map(|row| (0..k).map(|j| flat[row * k + j] * input[j]).sum::<f32>())
            .collect()
    }

    /// #3908: the BF16 GEMV against the CPU decoder, ON THE DEVICE, BIT-EXACTLY.
    ///
    /// Every other kernel here is held to a relative tolerance because its
    /// dequantization rounds. BF16's does not: decoding is `bits << 16`
    /// reinterpreted as f32, which is exact. But the GEMV still SUMS, and the
    /// GPU sums in a different order from the CPU (stride-32 per lane, then a
    /// warp tree reduction) -- fp32 addition is not associative, so demanding
    /// bit-equality of an arbitrary dot product would fail a CORRECT kernel.
    ///
    /// So the exactness is made well-posed instead of asserted loosely: every
    /// decoded weight and every activation is a small integer, so each product
    /// and every partial sum is an exact integer in f32 (bounded by
    /// 256 * 8 * 6 = 12288, far under 2^24). Any summation order then yields the
    /// IDENTICAL f32, and bit-equality becomes a statement about the decode and
    /// the indexing alone -- which is exactly what is under test.
    ///
    /// A single wrong bit here is a defect, not accumulation order.
    #[test]
    fn the_bf16_kernel_agrees_with_the_cpu_decoder_bit_exactly_on_device() {
        let Some(mut exec) = create_executor() else {
            eprintln!("SKIP: no CUDA device -- this is the one check that needs one");
            return;
        };

        let (k, n) = (256usize, 64usize);
        let weights = bf16_integer_weights(n, k);
        assert_eq!(weights.len(), n * k * 2);

        let input: Vec<f32> = (0..k).map(|i| ((i % 13) as f32) - 6.0).collect();
        let expected = bf16_reference(&weights, &input, k, n);

        let weight_buf = GpuBuffer::from_host(&exec.context, &weights).unwrap();
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let output_buf = GpuBuffer::from_host(&exec.context, &vec![0.0f32; n]).unwrap();
        exec.bf16_gemv_into(
            weight_buf.as_ptr(),
            &input_buf,
            &output_buf,
            u32::try_from(n).unwrap(),
            u32::try_from(k).unwrap(),
        )
        .expect("BF16 GEMV launch");
        exec.stream.synchronize().unwrap();

        let mut got = vec![0.0f32; n];
        output_buf.copy_to_host(&mut got).unwrap();

        let mismatches: Vec<String> = got
            .iter()
            .zip(expected.iter())
            .enumerate()
            .filter(|(_, (g, e))| g.to_bits() != e.to_bits())
            .map(|(row, (g, e))| {
                format!("\n  row {row}: GPU {g} (0x{:08x}) vs CPU {e} (0x{:08x})", g.to_bits(), e.to_bits())
            })
            .collect();
        assert!(
            mismatches.is_empty(),
            "#3908: BF16 GPU and CPU disagree on integer-exact data, where every \
             partial sum is exact and summation order cannot matter. {} of {n} rows \
             differ:{}",
            mismatches.len(),
            mismatches.join("")
        );

        // A guard against both sides being zero, which would agree perfectly and
        // prove nothing.
        let nonzero = expected.iter().filter(|v| v.abs() > 1e-6).count();
        assert!(
            nonzero >= n / 2,
            "only {nonzero} of {n} reference rows are non-zero; this comparison would \
             pass on an all-zero kernel"
        );
        eprintln!("#3908 A/B: {n} rows, 0 ULP (bit-exact) on integer-exact data");
    }

    /// #3908: the same A/B on ordinary non-integer bf16 values.
    ///
    /// Here the sum's order DOES matter, so this is a relative tolerance like
    /// every sibling above. The bit-exact test proves the decode; this proves
    /// the kernel on values whose mantissas are fully populated.
    #[test]
    fn the_bf16_kernel_agrees_with_the_cpu_decoder_on_ordinary_values() {
        let Some(mut exec) = create_executor() else {
            eprintln!("SKIP: no CUDA device");
            return;
        };

        let (k, n) = (256usize, 64usize);
        let weights = bf16_float_weights(n, k);
        let input: Vec<f32> = (0..k).map(|i| ((i % 13) as f32) - 6.0).collect();
        let expected = bf16_reference(&weights, &input, k, n);

        let weight_buf = GpuBuffer::from_host(&exec.context, &weights).unwrap();
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let output_buf = GpuBuffer::from_host(&exec.context, &vec![0.0f32; n]).unwrap();
        exec.bf16_gemv_into(
            weight_buf.as_ptr(),
            &input_buf,
            &output_buf,
            u32::try_from(n).unwrap(),
            u32::try_from(k).unwrap(),
        )
        .expect("BF16 GEMV launch");
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
            "#3908: GPU and CPU BF16 disagree. worst row {worst_row}: GPU {} vs CPU {} \
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
        eprintln!("#3908 A/B: {n} rows, worst relative disagreement {worst:.3e}");
    }

    /// A row length that is not a multiple of the warp stride, so the kernel's
    /// `i >= k_dim` bound is exercised rather than assumed.
    #[test]
    fn the_bf16_kernel_handles_a_row_that_is_not_a_multiple_of_the_warp_stride() {
        let Some(mut exec) = create_executor() else {
            eprintln!("SKIP: no CUDA device");
            return;
        };

        // 100 is not a multiple of 32: the last lane iteration is partial.
        let (k, n) = (100usize, 16usize);
        let weights = bf16_integer_weights(n, k);
        let input: Vec<f32> = (0..k).map(|i| ((i % 7) as f32) - 3.0).collect();
        let expected = bf16_reference(&weights, &input, k, n);

        let weight_buf = GpuBuffer::from_host(&exec.context, &weights).unwrap();
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let output_buf = GpuBuffer::from_host(&exec.context, &vec![0.0f32; n]).unwrap();
        exec.bf16_gemv_into(
            weight_buf.as_ptr(),
            &input_buf,
            &output_buf,
            u32::try_from(n).unwrap(),
            u32::try_from(k).unwrap(),
        )
        .expect("BF16 GEMV launch");
        exec.stream.synchronize().unwrap();

        let mut got = vec![0.0f32; n];
        output_buf.copy_to_host(&mut got).unwrap();

        for (row, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                e.to_bits(),
                "row {row} at k=100 (a partial last lane step): GPU {g} vs CPU {e}. \
                 The kernel must stop at k_dim and not read past it."
            );
        }
    }
}
