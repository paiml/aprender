//! #4376: the tensor-core MMQ GEMM (`launch_mma_q4k_gemm`) on the real device.
//!
//! Two comparisons, on shapes that are not tile multiples (the kernel tiles 64 rows x 32
//! columns per block, 16 x 8 per mma):
//!
//! 1. against `launch_dp4a_q4k_gemm` on the same Q8_1 input. Both kernels do the same integer
//!    arithmetic, so they may differ only by f32 summation order;
//! 2. against a host f32 reference, `dequant(W) · x`, written here from the Q4K layout and
//!    independent of both kernels, so a bug they share still shows.
//!
//! It needs a CUDA device and skips, loudly, without one.

use crate::cuda::executor::CudaExecutor;
use rand::{Rng, SeedableRng};
use trueno_gpu::driver::GpuBuffer;

const SB_BYTES: usize = 144;

/// Random but valid Q4K super-blocks, `n` rows × `k/256` blocks.
fn random_q4k(rng: &mut rand::rngs::StdRng, n: usize, k: usize) -> Vec<u8> {
    let mut w = vec![0u8; n * (k / 256) * SB_BYTES];
    for blk in w.chunks_exact_mut(SB_BYTES) {
        let d = half::f16::from_f32(rng.random_range(0.002..0.02));
        let dmin = half::f16::from_f32(rng.random_range(0.0..0.01));
        blk[0..2].copy_from_slice(&d.to_le_bytes());
        blk[2..4].copy_from_slice(&dmin.to_le_bytes());
        rng.fill(&mut blk[4..]);
    }
    w
}

/// Host dequant of one Q4K row: value = d·sc·q − dmin·m per 32-value sub-block.
fn dequant_q4k_row(row: &[u8], k: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(k);
    for blk in row.chunks_exact(SB_BYTES) {
        let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let dmin = half::f16::from_le_bytes([blk[2], blk[3]]).to_f32();
        let s = &blk[4..16];
        for j in 0..8 {
            let (sc, m) = if j < 4 {
                (s[j] & 63, s[j + 4] & 63)
            } else {
                (
                    (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4),
                    (s[j + 4] >> 4) | ((s[j] >> 6) << 4),
                )
            };
            let qs = &blk[16 + 32 * (j / 2)..16 + 32 * (j / 2) + 32];
            for &b in qs {
                let q = if j % 2 == 0 { b & 0xF } else { b >> 4 };
                out.push(d * f32::from(sc) * f32::from(q) - dmin * f32::from(m));
            }
        }
    }
    out
}

fn run(
    exec: &mut CudaExecutor,
    mmq: bool,
    w: &GpuBuffer<u8>,
    x: &GpuBuffer<f32>,
    m: u32,
    n: u32,
    k: u32,
) -> Vec<f32> {
    let y = GpuBuffer::<f32>::new(&exec.context, (m * n) as usize).expect("alloc y");
    if mmq {
        exec.launch_mma_q4k_gemm(w.as_ptr(), x.as_ptr(), y.as_ptr(), m, n, k)
    } else {
        exec.launch_dp4a_q4k_gemm(w.as_ptr(), x.as_ptr(), y.as_ptr(), m, n, k)
    }
    .expect("launch");
    exec.stream.synchronize().expect("sync");
    let mut got = vec![0.0f32; (m * n) as usize];
    y.copy_to_host(&mut got).expect("download");
    got
}

#[test]
fn mma_q4k_gemm_matches_dp4a_and_the_host_reference() {
    let Ok(mut exec) = CudaExecutor::new(0) else {
        eprintln!("SKIP #4376 mma_q4k_gemm row: no CUDA device");
        return;
    };
    let mut rng = rand::rngs::StdRng::seed_from_u64(4376);
    // (m, n, k): a partial warp, a partial block in both axes, and several super-blocks.
    for (m, n, k) in [(5u32, 8u32, 256u32), (37, 72, 512), (130, 100, 1024)] {
        let (mu, nu, ku) = (m as usize, n as usize, k as usize);
        let w = random_q4k(&mut rng, nu, ku);
        let x: Vec<f32> = (0..mu * ku)
            .map(|_| rng.random_range(-1.0f32..1.0))
            .collect();
        let w_buf = GpuBuffer::from_host(&exec.context, &w).expect("upload w");
        let x_buf = GpuBuffer::from_host(&exec.context, &x).expect("upload x");

        let mma = run(&mut exec, true, &w_buf, &x_buf, m, n, k);
        let dp4a = run(&mut exec, false, &w_buf, &x_buf, m, n, k);

        let row_bytes = (ku / 256) * SB_BYTES;
        let wf: Vec<Vec<f32>> = (0..nu)
            .map(|c| dequant_q4k_row(&w[c * row_bytes..(c + 1) * row_bytes], ku))
            .collect();
        let reference: Vec<f32> = (0..mu)
            .flat_map(|r| {
                let xr = &x[r * ku..(r + 1) * ku];
                wf.iter()
                    .map(|wc| wc.iter().zip(xr).map(|(a, b)| a * b).sum::<f32>())
                    .collect::<Vec<_>>()
            })
            .collect();

        let scale = reference.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        assert!(
            scale > 0.1,
            "fixture must produce non-trivial outputs (max |y| {scale})"
        );
        let max_err = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(p, q)| (p - q).abs())
                .fold(0.0f32, f32::max)
                / scale
        };
        let vs_dp4a = max_err(&mma, &dp4a);
        let vs_ref = max_err(&mma, &reference);
        let dp4a_vs_ref = max_err(&dp4a, &reference);
        eprintln!(
            "#4376 mma m={m} n={n} k={k}: |mma-dp4a|={vs_dp4a:.2e} |mma-ref|={vs_ref:.2e} \
             |dp4a-ref|={dp4a_vs_ref:.2e} (relative to max |y| {scale:.3})"
        );
        // Same integer products; only f32 summation order differs.
        assert!(
            vs_dp4a < 1e-4,
            "mma vs dp4a at m={m} n={n} k={k}: {vs_dp4a:.2e}"
        );
        // Int8 activations: within the error dp4a itself has, plus slack.
        assert!(
            vs_ref < 2e-2 && vs_ref <= 1.5 * dp4a_vs_ref + 1e-4,
            "mma vs f32 reference at m={m} n={n} k={k}: {vs_ref:.2e} (dp4a {dp4a_vs_ref:.2e})"
        );
    }
}

/// Kernel time at Qwen3.5-4B prefill shapes, mma vs dp4a (each includes its Q8_1
/// quantize pass). `cargo test ... -- --ignored mma_q4k_gemm_bench --nocapture`.
#[test]
#[ignore = "benchmark: needs a quiet GPU"]
fn mma_q4k_gemm_bench_4376() {
    let Ok(mut exec) = CudaExecutor::new(0) else {
        eprintln!("SKIP #4376 bench: no CUDA device");
        return;
    };
    let mut rng = rand::rngs::StdRng::seed_from_u64(4376);
    for (m, n, k) in [(4096u32, 9216u32, 2560u32), (4096, 2560, 9216)] {
        let (mu, nu, ku) = (m as usize, n as usize, k as usize);
        let w = random_q4k(&mut rng, nu, ku);
        let x: Vec<f32> = (0..mu * ku)
            .map(|_| rng.random_range(-1.0f32..1.0))
            .collect();
        let w_buf = GpuBuffer::from_host(&exec.context, &w).expect("upload w");
        let x_buf = GpuBuffer::from_host(&exec.context, &x).expect("upload x");
        let y = GpuBuffer::<f32>::new(&exec.context, mu * nu).expect("alloc y");
        for mmq in [false, true] {
            let mut launch = |exec: &mut CudaExecutor| {
                if mmq {
                    exec.launch_mma_q4k_gemm(w_buf.as_ptr(), x_buf.as_ptr(), y.as_ptr(), m, n, k)
                } else {
                    exec.launch_dp4a_q4k_gemm(w_buf.as_ptr(), x_buf.as_ptr(), y.as_ptr(), m, n, k)
                }
                .expect("launch");
            };
            launch(&mut exec);
            exec.stream.synchronize().expect("sync");
            let iters = 10;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                launch(&mut exec);
            }
            exec.stream.synchronize().expect("sync");
            let ms = t0.elapsed().as_secs_f64() * 1e3 / f64::from(iters);
            let tops = 2.0 * f64::from(m) * f64::from(n) * f64::from(k) / (ms * 1e-3) / 1e12;
            let name = if mmq { "mma" } else { "dp4a" };
            eprintln!("#4376 bench {name} m={m} n={n} k={k}: {ms:.3} ms ({tops:.1} TOPS)");
        }
    }
}
