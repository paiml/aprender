//! #4485: a DP4A GEMV replayed from the manual decode graph (trueno#243) must
//! produce what the same GEMV produces launched eagerly on the same input.

use super::CudaExecutor;
use crate::cuda::gpu_profile::{Q4kVariant, Q6kVariant};
use serial_test::serial;
use trueno_gpu::driver::GpuBuffer;

const N: u32 = 64;
const K: u32 = 2560;

fn q4k_weights() -> Vec<u8> {
    let mut seed = 0x9e37_79b9_u32;
    let mut out = Vec::new();
    for _ in 0..N * (K / 256) {
        out.extend_from_slice(&[0x00, 0x3C, 0x00, 0x20]);
        for _ in 0..12 {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            out.push(((seed >> 16) as u8) & 0x3f);
        }
        for _ in 0..128 {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            out.push((seed >> 16) as u8);
        }
    }
    out
}

fn q6k_weights() -> Vec<u8> {
    let mut seed = 0x7f4a_7c15_u32;
    let mut out = Vec::new();
    for _ in 0..N * (K / 256) {
        for _ in 0..(128 + 64 + 16) {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            out.push((seed >> 16) as u8);
        }
        out.extend_from_slice(&[0x00, 0x20]);
    }
    out
}

fn input(t: usize) -> Vec<f32> {
    (0..K)
        .map(|i| ((i as f32 + 1.0) * (0.13 + 0.07 * t as f32)).sin() * (1.0 + t as f32))
        .collect()
}

fn run(q6: bool) {
    let mut ex = crate::cuda_executor_or_skip!(0);
    ex.init_workspace(K as usize, K as usize)
        .expect("workspace");
    let name = if q6 { "g4485_q6k" } else { "g4485_q4k" };
    let bytes = if q6 { q6k_weights() } else { q4k_weights() };
    ex.load_quantized_weights(name, &bytes).expect("load");
    let w = ex.get_quantized_weight_ptr(name).expect("ptr");
    ex.gpu_profile.q4k = Q4kVariant::HwDp4a;
    ex.gpu_profile.q6k = Q6kVariant::HwDp4a;
    let x = GpuBuffer::<f32>::new(ex.context(), K as usize).expect("x");
    let mut xs = GpuBuffer::<f32>::new(ex.context(), K as usize).expect("xs");
    let out = GpuBuffer::<f32>::new(ex.context(), N as usize).expect("out");
    let gemv = |ex: &mut CudaExecutor, xin: &GpuBuffer<f32>| {
        if q6 {
            ex.q6k_gemv_into(w, xin, &out, N, K).expect("q6k");
        } else {
            ex.q4k_gemv_into(w, xin, &out, N, K).expect("q4k");
        }
    };
    let read = |ex: &mut CudaExecutor| {
        ex.synchronize().expect("sync");
        let mut h = vec![0f32; N as usize];
        out.copy_to_host(&mut h).expect("read");
        h
    };
    // x is copied from the staging buffer by a recorded kernel, the way a
    // layer's rmsnorm writes the GEMV input inside the graph.
    let step = |ex: &mut CudaExecutor, xs: &GpuBuffer<f32>| {
        ex.residual_add_into(xs, xs, &x, K).expect("x = 2*xs");
        gemv(ex, &x);
    };

    let mut eager = Vec::new();
    for t in 0..3 {
        ex.upload_on_stream(&mut xs, &input(t)).expect("up");
        step(&mut ex, &xs);
        eager.push(read(&mut ex));
    }

    ex.begin_graph_recording();
    ex.upload_on_stream(&mut xs, &input(0)).expect("up");
    step(&mut ex, &xs);
    let kernels = ex.end_graph_recording().expect("build");
    let exec = ex.take_decode_graph().expect("graph");
    assert_eq!(read(&mut ex), eager[0], "the capture launch itself");
    for t in 0..3 {
        ex.upload_on_stream(&mut xs, &input(t)).expect("up");
        ex.launch_graph_exec(&exec).expect("replay");
        let got = read(&mut ex);
        let d = got
            .iter()
            .zip(&eager[t])
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        eprintln!("[4485 graph] q6={q6} {kernels} kernels, input {t}: max|replay-eager| {d:e}");
        assert_eq!(
            got, eager[t],
            "q6={q6}: replay differs from eager on input {t}"
        );
    }
}

#[test]
#[serial]
fn a_replayed_dp4a_q4k_gemv_matches_the_eager_launch() {
    run(false);
}

#[test]
#[serial]
fn a_replayed_dp4a_q6k_gemv_matches_the_eager_launch() {
    run(true);
}
