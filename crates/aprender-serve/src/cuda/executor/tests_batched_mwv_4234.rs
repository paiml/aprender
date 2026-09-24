//! #4234: `batched_mwv_q4k_gemv_into` is `mwv_q4k_gemv_into` per vector, BITWISE.
//!
//! Continuous batching must not make a stream's tokens depend on its neighbours,
//! so a tolerance here would be the wrong test: the batched kernel claims the
//! single-vector kernel's exact arithmetic per vector, and bits are what it is held to.

use super::*;
use serial_test::serial;

/// Deterministic Q4_K rows: random scales and nibbles, sane `d` / `dmin`.
fn q4k_rows(n: usize, k: usize, seed: u64) -> Vec<u8> {
    const SB_BYTES: usize = 144;
    let sbs = k.div_ceil(256);
    let mut s = seed;
    let mut next = move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (s >> 33) as u8
    };
    let mut out = vec![0u8; n * sbs * SB_BYTES];
    for sb in out.chunks_exact_mut(SB_BYTES) {
        for b in sb.iter_mut() {
            *b = next();
        }
        // d = 0.0123, dmin = 0.0041 as f16 bit patterns (finite, small).
        sb[0..2].copy_from_slice(&0x224C_u16.to_le_bytes());
        sb[2..4].copy_from_slice(&0x1C33_u16.to_le_bytes());
    }
    out
}

fn activations(m: usize, k: usize) -> Vec<f32> {
    (0..m * k)
        .map(|i| ((i * 2_654_435_761) % 2_000) as f32 / 1_000.0 - 1.0)
        .collect()
}

#[test]
#[serial]
fn batched_mwv_q4k_gemv_is_bitwise_the_single_vector_kernel_per_vector() {
    let mut executor = crate::cuda_executor_or_skip!(0);
    // (k, n): 256-aligned and not (the last super-block's bounds checks).
    for (case, &(k, n)) in [(1024u32, 96u32), (1000, 40)].iter().enumerate() {
        let name = format!("bmwv_4234_{case}");
        executor
            .load_quantized_weights(&name, &q4k_rows(n as usize, k as usize, 7 + case as u64))
            .expect("load weights");
        let w = executor.get_quantized_weight_ptr(&name).expect("ptr");
        for m in 1u32..=12 {
            let x = activations(m as usize, k as usize);
            let input = GpuBuffer::from_host(executor.context(), &x).expect("input");
            let output = GpuBuffer::<f32>::new(executor.context(), (m * n) as usize).expect("out");
            executor
                .batched_mwv_q4k_gemv_into(w, &input, &output, m, n, k)
                .expect("batched");
            executor.synchronize().expect("sync");
            let mut got = vec![0.0f32; (m * n) as usize];
            output.copy_to_host(&mut got).expect("download");

            for r in 0..m as usize {
                let row = &x[r * k as usize..(r + 1) * k as usize];
                let one_in = GpuBuffer::from_host(executor.context(), row).expect("in");
                let one_out = GpuBuffer::<f32>::new(executor.context(), n as usize).expect("out");
                executor
                    .mwv_q4k_gemv_into(w, &one_in, &one_out, n, k)
                    .expect("single");
                executor.synchronize().expect("sync");
                let mut want = vec![0.0f32; n as usize];
                one_out.copy_to_host(&mut want).expect("download");
                let got_r = &got[r * n as usize..(r + 1) * n as usize];
                let differing = got_r
                    .iter()
                    .zip(&want)
                    .filter(|(g, w)| g.to_bits() != w.to_bits())
                    .count();
                assert_eq!(
                    differing, 0,
                    "k={k} n={n} m={m} vector {r}: {differing} of {n} outputs differ in their bits"
                );
                assert!(
                    want.iter().any(|v| *v != 0.0),
                    "vacuous: an all-zero reference row"
                );
            }
        }
    }
}

#[test]
#[serial]
fn batched_mwv_q4k_gemv_refuses_short_buffers_and_an_empty_batch() {
    let mut executor = crate::cuda_executor_or_skip!(0);
    let (k, n) = (256u32, 8u32);
    executor
        .load_quantized_weights("bmwv_4234_refuse", &q4k_rows(n as usize, k as usize, 1))
        .expect("load");
    let w = executor
        .get_quantized_weight_ptr("bmwv_4234_refuse")
        .expect("ptr");
    let input = GpuBuffer::<f32>::new(executor.context(), (2 * k) as usize).expect("in");
    let output = GpuBuffer::<f32>::new(executor.context(), (2 * n) as usize).expect("out");
    for (what, m) in [("an empty batch", 0u32), ("buffers for 2 vectors, m=3", 3)] {
        let err = executor
            .batched_mwv_q4k_gemv_into(w, &input, &output, m, n, k)
            .expect_err(what);
        assert!(
            err.to_string().contains("batched_mwv_q4k_gemv_into"),
            "{what}: {err}"
        );
    }
}
