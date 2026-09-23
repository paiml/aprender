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
