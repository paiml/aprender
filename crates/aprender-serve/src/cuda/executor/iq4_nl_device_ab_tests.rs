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
