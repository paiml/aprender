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

    /// #3953: run the IQ2_S kernel on `weights` and compare it with the CPU oracle.
    ///
    /// ADMISSION RULE (cop, from aprender-70's IQ4_XS finding): IQ4_XS was whitelisted
    /// after an A/B at ONE shape, and a real model calls it at eight, including k=3584 --
    /// 14 super-blocks, not a power of two. So this is run at EVERY (k, n) the real model
    /// uses for type 22, on synthetic AND real bytes, with:
    ///   * per-row tolerance |gpu - cpu| / sum_j |w_ij| * |x_j| <= 1e-5 -- scaled by the
    ///     row's own magnitude, so a large row cannot hide a small absolute error and a
    ///     near-zero row cannot turn rounding into a huge ratio;
    ///   * the output PRE-FILLED WITH NaN, so a row the kernel never writes fails outright
    ///     instead of reading as 0.0 and agreeing with a CPU value near 0.
    /// Returns (worst ratio, its row, gpu, cpu).
    fn iq2_s_device_ab(
        exec: &mut CudaExecutor,
        weights: &[u8],
        k: usize,
        n: usize,
    ) -> Result<(f64, usize, f32, f32), String> {
        use crate::quantize::iq2_s::{dequantize_iq2_s, GGML_TYPE_IQ2_S, IQ2_S_BLOCK_BYTES};
        let row_bytes = k.div_ceil(256) * IQ2_S_BLOCK_BYTES;
        assert_eq!(weights.len(), n * row_bytes, "buffer must be n * ceil(k/256) * 82 bytes");

        // Position-dependent, so a block read at the wrong offset cannot sum the same.
        let input: Vec<f32> = (0..k).map(|i| ((i % 17) as f32) - 8.0).collect();
        let expected = crate::quantize::iq_parallel_matvec(GGML_TYPE_IQ2_S, weights, &input, k, n)
            .map_err(|e| format!("CPU oracle failed: {e}"))?;

        let weight_buf = GpuBuffer::from_host(&exec.context, weights).unwrap();
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let output_buf = GpuBuffer::from_host(&exec.context, &vec![f32::NAN; n]).unwrap();
        exec.iq2_s_gemv_into(
            weight_buf.as_ptr(),
            &input_buf,
            &output_buf,
            u32::try_from(n).unwrap(),
            u32::try_from(k).unwrap(),
        )
        .map_err(|e| format!("IQ2_S GEMV launch: {e:?}"))?;
        exec.stream.synchronize().unwrap();
        let mut got = vec![0.0f32; n];
        output_buf.copy_to_host(&mut got).unwrap();

        let (mut worst, mut wrow) = (0.0f64, 0usize);
        for row in 0..n {
            if !got[row].is_finite() {
                return Err(format!(
                    "row {row} is {} -- the kernel never wrote it (output was pre-filled with NaN)",
                    got[row]
                ));
            }
            let w = dequantize_iq2_s(&weights[row * row_bytes..(row + 1) * row_bytes])
                .map_err(|e| format!("row {row} dequant: {e}"))?;
            let scale: f64 = w.iter().zip(&input).map(|(a, b)| f64::from(a.abs() * b.abs())).sum();
            let ratio = f64::from((got[row] - expected[row]).abs()) / scale.max(f64::MIN_POSITIVE);
            if ratio > worst {
                worst = ratio;
                wrow = row;
            }
        }
        let nonzero = expected.iter().filter(|v| v.abs() > 1e-6).count();
        if nonzero < n / 2 {
            return Err(format!("only {nonzero}/{n} oracle rows non-zero: would pass on an all-zero kernel"));
        }
        Ok((worst, wrow, got[wrow], expected[wrow]))
    }

    fn assert_within(label: &str, k: usize, n: usize, r: Result<(f64, usize, f32, f32), String>) {
        let (worst, row, g, e) = r.unwrap_or_else(|m| panic!("#3953 {label} k={k} n={n}: {m}"));
        assert!(
            worst <= 1e-5,
            "#3953 {label} k={k} n={n}: GPU and CPU IQ2_S disagree. worst row {row}: GPU {g} vs \
             CPU {e} (err/sum|w||x| {worst:.3e} > 1e-5). The CPU decoder is the oracle."
        );
        eprintln!("#3953 A/B {label} k={k} n={n}: worst err/sum|w||x| {worst:.3e} at row {row}");
    }

    /// The small shape the planted faults were first proven against: 2 super-blocks per
    /// row, so the stride is exercised.
    #[test]
    fn the_iq2_s_kernel_agrees_with_the_cpu_decoder_on_device() {
        use crate::quantize::iq2_s_geometry_tests::iq2_s_weights;
        let Some(mut exec) = create_executor() else {
            eprintln!("SKIP: no CUDA device");
            return;
        };
        let (k, n) = (512usize, 48usize);
        assert_within("synthetic", k, n, iq2_s_device_ab(&mut exec, &iq2_s_weights(n, k), k, n));
    }

    /// EVERY (k, n) Qwen3.5-0.8B-UD-IQ2_XXS uses for type 22. Measured from the file:
    /// five tensors, all `blk.{8,9,10,17,21}.ffn_down.weight`, all ne=[3584, 1024] --
    /// ONE shape, 14 super-blocks per row, which the 2-block test above never reaches.
    const IQ2_S_REAL_SHAPES: &[(usize, usize)] = &[(3584, 1024)];

    #[test]
    fn the_iq2_s_kernel_agrees_at_every_real_model_shape_on_device() {
        use crate::quantize::iq2_s_geometry_tests::iq2_s_weights;
        let Some(mut exec) = create_executor() else {
            eprintln!("SKIP: no CUDA device");
            return;
        };
        for &(k, n) in IQ2_S_REAL_SHAPES {
            assert_within("synthetic", k, n, iq2_s_device_ab(&mut exec, &iq2_s_weights(n, k), k, n));
        }
    }

    /// The same, on the REAL bytes of every type-22 tensor in the model.
    ///
    /// If the offset convention were wrong, GPU and CPU would read the SAME wrong bytes
    /// and agree perfectly -- an A/B that proves nothing. So before comparing, the bytes
    /// are shown to BE a weight tensor: every block's f16 scale `d` must be finite and
    /// small. Header or string bytes decoded as f16 fail that.
    #[test]
    fn the_iq2_s_kernel_agrees_on_every_real_type22_tensor_on_device() {
        use crate::gguf::MappedGGUFModel;
        use crate::quantize::iq2_s::IQ2_S_BLOCK_BYTES;
        const MODEL: &str = "/home/noah/models/Qwen3.5-0.8B-UD-IQ2_XXS.gguf";
        let Some(mut exec) = create_executor() else {
            eprintln!("SKIP: no CUDA device");
            return;
        };
        if !std::path::Path::new(MODEL).exists() {
            eprintln!("SKIP: {MODEL} is not on this host -- the real-bytes A/B did NOT run");
            return;
        }
        let mapped = MappedGGUFModel::from_path(MODEL).expect("map the model");
        let data = mapped.data();
        let base = mapped.model.tensor_data_start;
        let mut seen = 0usize;
        for t in mapped.model.tensors.iter().filter(|t| t.qtype == 22) {
            // The order `dims` is stored in is not what its doc says, and neither the
            // "multiple of 256" test nor the byte size can recover it here: 3584 and 1024
            // are BOTH multiples of 256, and n*(k/256)*82 is symmetric when both are. So
            // the order is taken from IQ2_S_REAL_SHAPES, which was read from the raw file's
            // ggml `ne` by an independent parser -- ground truth, not the API's convention.
            let (a, b) = (t.dims[0] as usize, t.dims[1] as usize);
            let (k, n) = if IQ2_S_REAL_SHAPES.contains(&(a, b)) {
                (a, b)
            } else if IQ2_S_REAL_SHAPES.contains(&(b, a)) {
                (b, a)
            } else {
                panic!("{}: dims {:?} match no shape in IQ2_S_REAL_SHAPES -- the census is stale", t.name, t.dims)
            };
            let len = n * k.div_ceil(256) * IQ2_S_BLOCK_BYTES;
            let start = base + usize::try_from(t.offset).unwrap();
            let bytes = &data[start..start + len];
            let mut bad_d = 0usize;
            for blk in bytes.chunks_exact(IQ2_S_BLOCK_BYTES) {
                let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                if !d.is_finite() || d.abs() > 1.0 {
                    bad_d += 1;
                }
            }
            assert_eq!(
                bad_d, 0,
                "{}: {bad_d} blocks have a non-finite or implausible scale -- these bytes are \
                 not this tensor (offset {start}), and an A/B on them would agree about garbage",
                t.name
            );
            assert_within(&t.name, k, n, iq2_s_device_ab(&mut exec, bytes, k, n));
            seen += 1;
        }
        assert_eq!(seen, 5, "expected the 5 type-22 tensors the census found, saw {seen}");
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
