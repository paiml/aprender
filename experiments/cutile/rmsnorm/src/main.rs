//! T3 — cutile-rs vs the production hand-PTX RMSNorm, on GB10 (sm_121).
//!
//! A MEASUREMENT, not an adoption. The question this answers is narrow and falsifiable:
//! does a Tile-track kernel compute the same thing as the hand-PTX one, and how fast.
//!
//! ## Comparand discipline
//!
//! The hand-PTX side is the SAME committed artifact the cuda-oxide RMSNorm run used —
//! `experiments/cuda-oxide/rmsnorm/baseline-ptx/rmsnorm_h{N}.sm121.ptx`, emitted by
//! `RmsNormKernel::new(h).with_epsilon(1e-5).emit_ptx_for_target("sm_121")`. Included by
//! path rather than copied, so there is exactly one comparand and the cutile and oxide
//! numbers are directly comparable. Picking the wrong comparand is how the retracted
//! "oxide is 1.45x faster" claim happened (it raced `TiledQ4KGemv`, not the production
//! `HwDp4a`), so the comparand is named here rather than left implicit.
//!
//! ## Timing method, stated because it is not the oxide run's
//!
//! Wall-clock around a batch of `INNER` launches with one sync at the end, repeated
//! `OUTER` times, median reported. The oxide run used CUDA-event medians; cutile drives
//! its own stream through `sync_on`, so a single event pair cannot bracket both paths
//! identically. Batching amortises launch overhead, and the SAME method is applied to
//! both sides — an unfair timer is worse than no timer.

use cuda_core::{launch_kernel_on_stream, CudaContext, DeviceBuffer};
use cutile::api;
use cutile::error::Error;
use cutile::tensor::{IntoPartition, Partition, Reshape, Tensor, ToHostVec};
use cutile::tile_kernel::TileKernel;
use cuda_async::device_operation::DeviceOp;
use std::ffi::c_void;
use std::sync::Arc;
use std::time::Instant;

const EPS: f32 = 1e-5;
const WARMUP: usize = 10;
const INNER: usize = 50;
const OUTER: usize = 5;

// The hand-PTX comparand, one copy, shared with the cuda-oxide run.
// (a) The 32-thread single-warp `RmsNormKernel`. Shared with the cuda-oxide run so the
//     two experiments have one comparand in common and can be cross-checked.
const WARP_2048: &str = include_str!("../../../cuda-oxide/rmsnorm/baseline-ptx/rmsnorm_h2048.sm121.ptx");
const WARP_4096: &str = include_str!("../../../cuda-oxide/rmsnorm/baseline-ptx/rmsnorm_h4096.sm121.ptx");
const WARP_8192: &str = include_str!("../../../cuda-oxide/rmsnorm/baseline-ptx/rmsnorm_h8192.sm121.ptx");

// (b) The 256-thread `VectorizedRmsNormKernel` — the FAIR comparand, because it uses the
//     same 256 threads/row this cutile kernel does. Racing a 256-thread Tile kernel only
//     against the 32-thread kernel is the error that produced the retracted "oxide is
//     1.45x faster" claim (it raced TiledQ4KGemv, not the production HwDp4a).
const VEC_2048: &str = include_str!("../baseline-ptx/rmsnorm_vec_h2048.sm121.ptx");
const VEC_4096: &str = include_str!("../baseline-ptx/rmsnorm_vec_h4096.sm121.ptx");
const VEC_8192: &str = include_str!("../baseline-ptx/rmsnorm_vec_h8192.sm121.ptx");

// (c) `BatchedVectorizedRmsNormKernel` (batch=8) — grid (1, M, 1), 256 thr, %ctaid.y.
//     The ONLY honest comparand at rows=8: it does all 8 rows in ONE launch, exactly as
//     the cutile kernel does. Racing cutile's single batched launch against 8 sequential
//     single-row launches would be the comparand error a third time.
const BATCH_2048: &str = include_str!("../baseline-ptx/rmsnorm_batch8_h2048.sm121.ptx");
const BATCH_4096: &str = include_str!("../baseline-ptx/rmsnorm_batch8_h4096.sm121.ptx");
const BATCH_8192: &str = include_str!("../baseline-ptx/rmsnorm_batch8_h8192.sm121.ptx");

/// Which hand-PTX kernel to race against.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Baseline {
    /// `RmsNormKernel` — 32 threads (1 warp) per row.
    Warp,
    /// `VectorizedRmsNormKernel` — 256 threads (8 warps) per row.
    Vectorized,
    /// `BatchedVectorizedRmsNormKernel` (batch=8) — grid (1, M, 1), one launch for all rows.
    Batched8,
}

impl Baseline {
    fn ptx(self, hidden: usize) -> &'static str {
        match (self, hidden) {
            (Self::Warp, 2048) => WARP_2048,
            (Self::Warp, 4096) => WARP_4096,
            (Self::Warp, 8192) => WARP_8192,
            (Self::Vectorized, 2048) => VEC_2048,
            (Self::Vectorized, 4096) => VEC_4096,
            (Self::Vectorized, 8192) => VEC_8192,
            (Self::Batched8, 2048) => BATCH_2048,
            (Self::Batched8, 4096) => BATCH_4096,
            (Self::Batched8, 8192) => BATCH_8192,
            _ => panic!("no committed baseline PTX for hidden={hidden}"),
        }
    }
    fn entry(self) -> &'static str {
        match self {
            Self::Warp => "rmsnorm",
            Self::Vectorized => "rmsnorm_vectorized",
            Self::Batched8 => "batched_rmsnorm_vectorized",
        }
    }
    fn threads(self) -> u32 {
        match self {
            Self::Warp => 32,
            Self::Vectorized | Self::Batched8 => 256,
        }
    }
    /// True when the kernel dispatches rows itself (`%ctaid.y`), so ONE launch covers all.
    fn is_batched(self) -> bool {
        matches!(self, Self::Batched8)
    }
    fn label(self) -> &'static str {
        match self {
            Self::Warp => "warp32",
            Self::Vectorized => "vec256",
            Self::Batched8 => "batch8",
        }
    }
}

#[cutile::module]
mod kernel {
    use cutile::core::*;

    /// RMSNorm, adapted from the upstream `cutile-examples/examples/rms_norm.rs`.
    ///
    /// Computes, per row: `out[i] = x[i] * rsqrt(mean(x^2) + eps) * w[i]`.
    #[cutile::entry()]
    fn rms_norm<const N: i32, const BLOCK_SIZE: i32>(
        x: &Tensor<f32, { [-1, N] }>,
        w: &Tensor<f32, { [N] }>,
        out: &mut Tensor<f32, { [1, N] }>,
        eps: f32,
    ) {
        let tile_shape: Shape<{ [1, BLOCK_SIZE] }> = shape![1, BLOCK_SIZE];
        let num_tiles: i32 = N / BLOCK_SIZE;
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;

        let x_part: Partition<f32, { [1, BLOCK_SIZE] }> = x.partition(tile_shape);
        let mut acc: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, tile_shape);
        for j in 0i32..num_tiles {
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = x_part.load([row, j]);
            acc = acc + tx * tx;
        }
        let s: Tile<f32, { [1] }> = reduce_sum(acc, 1i32);
        let s: Tile<f32, { [] }> = s.reshape(shape![]);
        let s: f32 = tile_to_scalar(s);
        let n: f32 = convert_scalar(N);
        let s: f32 = 1.0f32 / (s / n + eps);
        let s: Tile<f32, { [] }> = sqrt(scalar_to_tile(s), rounding::NegativeInf, ftz::Disabled);
        let s: f32 = tile_to_scalar(s);
        let rms: Tile<f32, { [1, BLOCK_SIZE] }> = s.broadcast(tile_shape);

        let w_part: Partition<f32, { [BLOCK_SIZE] }> = w.partition(shape![BLOCK_SIZE]);
        let mut out_part: PartitionMut<f32, { [1, BLOCK_SIZE] }> = out.partition_mut(tile_shape);
        for j in 0i32..num_tiles {
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = x_part.load([row, j]);
            let tw: Tile<f32, { [1, BLOCK_SIZE] }> = w_part.load([j]).reshape(tile_shape);
            out_part.store(tx * rms * tw, [0i32, j]);
        }
    }
}

/// Deterministic inputs — no rand dependency, and the same bytes on every run and
/// every host, so a parity difference is never the data's fault.
fn make_data(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            // top 24 bits -> [-1, 1)
            ((s >> 40) as f32 / (1u32 << 23) as f32) - 1.0
        })
        .collect()
}

/// f64 reference. Deliberately a different precision from both GPU paths, so it is an
/// independent oracle rather than a re-run of one of them.
fn cpu_reference(x: &[f32], w: &[f32], rows: usize, hidden: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * hidden];
    for r in 0..rows {
        let row = &x[r * hidden..(r + 1) * hidden];
        let mean_sq: f64 = row.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / hidden as f64;
        let inv = 1.0 / (mean_sq + f64::from(EPS)).sqrt();
        for i in 0..hidden {
            out[r * hidden + i] = (f64::from(row[i]) * inv * f64::from(w[i])) as f32;
        }
    }
    out
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        dot += f64::from(*x) * f64::from(*y);
        na += f64::from(*x) * f64::from(*x);
        nb += f64::from(*y) * f64::from(*y);
    }
    if na == 0.0 || nb == 0.0 { return 0.0; }
    dot / (na.sqrt() * nb.sqrt())
}

fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN timings"));
    v[v.len() / 2]
}

/// Hand-PTX path: the production comparand.
///
/// The committed baseline uses `%tid.x` ONLY — no `%ctaid` — so it is a single-row,
/// single-block (one warp) kernel: exactly one row per launch, which is how the serve
/// executor calls it per decode token. Launching it with `grid=(rows,1,1)` does NOT
/// process `rows` rows; every block races on row 0. That was measured here first, as
/// `cos = 0.35 ~= 1/sqrt(8)` with 8 rows — the signature of one row written and seven
/// left zero, not of a wrong kernel. So multi-row means `rows` launches with offset
/// pointers, and the primary gate is rows=1 like-for-like.
fn run_hand_ptx(
    x: &[f32],
    w: &[f32],
    rows: usize,
    hidden: usize,
    base: Baseline,
) -> Result<(Vec<f32>, f64), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = CudaContext::load_module_from_ptx_src(&ctx, base.ptx(hidden))?;
    let func = module.load_function(base.entry())?;

    let d_in = DeviceBuffer::from_host(&stream, x)?;
    let d_w = DeviceBuffer::from_host(&stream, w)?;
    let d_out = DeviceBuffer::<f32>::zeroed(&stream, rows * hidden)?;
    let row_bytes = (hidden * std::mem::size_of::<f32>()) as u64;

    let launch = |reps: usize| -> Result<(), Box<dyn std::error::Error>> {
        // A batched kernel dispatches rows itself via %ctaid.y: ONE launch, grid (1, M, 1).
        let per_call = if base.is_batched() { 1 } else { rows };
        for _ in 0..reps {
            for r in 0..per_call {
                let off = row_bytes * r as u64;
                let mut p_in = d_in.cu_deviceptr() + off;
                let mut p_out = d_out.cu_deviceptr() + off;
                let mut p_w = d_w.cu_deviceptr();
                let mut params: [*mut c_void; 3] = [
                    std::ptr::addr_of_mut!(p_in).cast(),
                    std::ptr::addr_of_mut!(p_out).cast(),
                    std::ptr::addr_of_mut!(p_w).cast(),
                ];
                // SAFETY: `func` came from this module; the pointers are inside buffers of
                // rows*hidden / hidden elements; and the launch shape is the one-block,
                // one-warp contract baked into the committed PTX.
                let grid = if base.is_batched() {
                    (1, u32::try_from(rows)?, 1)
                } else {
                    (1, 1, 1)
                };
                unsafe {
                    launch_kernel_on_stream(&func, grid, (base.threads(), 1, 1), 0, &stream, &mut params)?;
                }
            }
        }
        Ok(())
    };

    launch(WARMUP)?;
    let out = d_out.to_host_vec(&stream)?; // syncs

    let mut samples = Vec::with_capacity(OUTER);
    for _ in 0..OUTER {
        let t0 = Instant::now();
        launch(INNER)?;
        let _ = d_out.to_host_vec(&stream)?; // sync point
        samples.push(t0.elapsed().as_secs_f64() * 1e6 / INNER as f64);
    }
    Ok((out, median(samples)))
}

/// cutile path.
fn run_cutile(
    x: &[f32],
    w: &[f32],
    rows: usize,
    hidden: usize,
    block: usize,
) -> Result<(Vec<f32>, f64), Error> {
    let device = cuda_core::Device::new(0)?;
    let stream = device.new_stream()?;
    let generics = vec![hidden.to_string(), block.to_string()];

    // Upload ONCE, outside the timing loop. cutile hands the inputs back out of every
    // launch (that is the ownership-across-the-launch-boundary design), so the loop can
    // rebind them instead of re-uploading — which makes this a kernel-time comparison
    // rather than a PCIe one, matching what the hand-PTX side measures.
    let hx = Arc::new(x.to_vec());
    let hw = Arc::new(w.to_vec());
    let xt: Tensor<f32> = api::copy_host_vec_to_device(&hx).sync_on(&stream)?;
    let mut xt: Arc<Tensor<f32>> = Arc::new(xt.reshape(&[rows, hidden])?);
    let mut wt: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&hw).sync_on(&stream)?.into();
    let mut ot: Partition<Tensor<f32>> =
        api::zeros::<f32>(&[rows, hidden]).sync_on(&stream)?.partition([1, hidden]);

    // The FIRST call JIT-compiles the embedded AST through Tile IR. Timing it would
    // measure the compiler, not the kernel.
    for _ in 0..WARMUP {
        let (x2, w2, o2, _) = kernel::rms_norm(xt, wt, ot, EPS)
            .generics(generics.clone())
            .sync_on(&stream)?;
        xt = x2;
        wt = w2;
        ot = o2;
    }

    let mut samples = Vec::with_capacity(OUTER);
    for _ in 0..OUTER {
        let t0 = Instant::now();
        for _ in 0..INNER {
            let (x2, w2, o2, _) = kernel::rms_norm(xt, wt, ot, EPS)
                .generics(generics.clone())
                .sync_on(&stream)?;
            xt = x2;
            wt = w2;
            ot = o2;
        }
        samples.push(t0.elapsed().as_secs_f64() * 1e6 / INNER as f64);
    }
    let out = ot.unpartition().to_host_vec().sync_on(&stream)?;
    Ok((out, median(samples)))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let cc = ctx.compute_capability()?;
    println!("device   : {} (sm_{}{})", ctx.device_name()?, cc.0, cc.1);
    println!("eps={EPS} warmup={WARMUP} inner={INNER} outer={OUTER} (median of {OUTER} batch means)");
    println!("baselines: vec256 = VectorizedRmsNormKernel (256 thr/row)  <-- FAIR comparand");
    println!("           warp32 = RmsNormKernel (32 thr/row), shared with the cuda-oxide run");
    println!("both hand-PTX kernels use %tid.x ONLY (no %ctaid) => one row per launch");
    println!();

    let mut parity_ok = true;

    println!("== PRIMARY  rows=1 — like-for-like, one row per launch on every path ==");
    println!("{:>7} {:>11} {:>12} {:>12} {:>10} {:>10} {:>9}",
        "hidden", "cos(cutile)", "cos(vec256)", "cos(warp32)", "cutile us", "vec256 us", "warp32 us");
    for &hidden in &[2048usize, 4096, 8192] {
        let (x, w) = (make_data(hidden, 0x51ED_5EED), make_data(hidden, 0xC0FF_EE11));
        let reference = cpu_reference(&x, &w, 1, hidden);
        let (a, a_us) = run_cutile(&x, &w, 1, hidden, 256)?;
        let (b, b_us) = run_hand_ptx(&x, &w, 1, hidden, Baseline::Vectorized)?;
        let (c, c_us) = run_hand_ptx(&x, &w, 1, hidden, Baseline::Warp)?;
        let ok = [(&a, ()), (&b, ()), (&c, ())].iter().all(|(o, ())| cosine(o, &reference) >= 0.9999 && maxdiff(o, &reference) < 1e-4);
        parity_ok &= ok;
        println!("{hidden:>7} {:>11.7} {:>12.7} {:>12.7} {a_us:>10.1} {b_us:>10.1} {c_us:>9.1}   cutile/vec256={:.2}x  cutile/warp32={:.2}x {}",
            cosine(&a, &reference), cosine(&b, &reference), cosine(&c, &reference),
            a_us / b_us, a_us / c_us, if ok { "OK" } else { "PARITY-FAIL" });
    }
    println!();

    println!("== SECONDARY rows=8 — batched: ONE launch on both sides ==");
    println!("{:>7} {:>11} {:>12} {:>10} {:>10}", "hidden", "cos(cutile)", "cos(batch8)", "cutile us", "batch8 us");
    for &hidden in &[2048usize, 4096, 8192] {
        let (x, w) = (make_data(8 * hidden, 0x51ED_5EED), make_data(hidden, 0xC0FF_EE11));
        let reference = cpu_reference(&x, &w, 8, hidden);
        let (a, a_us) = run_cutile(&x, &w, 8, hidden, 256)?;
        let (b, b_us) = run_hand_ptx(&x, &w, 8, hidden, Baseline::Batched8)?;
        let ok = [&a, &b].iter().all(|o| cosine(o, &reference) >= 0.9999 && maxdiff(o, &reference) < 1e-4);
        parity_ok &= ok;
        println!("{hidden:>7} {:>11.7} {:>12.7} {a_us:>10.1} {b_us:>10.1}   cutile/batch8={:.2}x {}",
            cosine(&a, &reference), cosine(&b, &reference), a_us / b_us,
            if ok { "OK" } else { "PARITY-FAIL" });
    }
    println!();

    println!("parity gate (cos >= 0.9999 AND maxdiff < 1e-4 vs f64 CPU, ALL THREE paths, BOTH shapes): {}",
        if parity_ok { "PASS" } else { "FAIL" });
    println!();
    println!("Reading the ratios: > 1 means cutile is SLOWER.");
    println!("  cutile/vec256 is THE result. Both use 256 threads per row, so it isolates the");
    println!("  Tile abstraction rather than an occupancy difference.");
    println!("  cutile/batch8 is THE rows=8 result: BatchedVectorizedRmsNormKernel does all 8");
    println!("  rows in one launch via %ctaid.y, exactly as the cutile kernel does.");
    println!("  cutile/warp32 is a CROSS-CHECK only: warp32 is a 32-thread kernel, so beating it");
    println!("  with 256 threads measures occupancy, not Tile IR. It is reported because the");
    println!("  cuda-oxide RMSNorm run used exactly this comparand, which makes the two");
    println!("  experiments comparable -- not because it is a fair race.");
    println!("  Both sides upload once outside the timing loop and re-launch in place, so this is");
    println!("  kernel + launch overhead, not PCIe. cutile's per-launch cost includes its own");
    println!("  host-side builder; the hand-PTX side is a bare cuLaunchKernel. Not corrected for.");
    if !parity_ok {
        std::process::exit(1);
    }
    Ok(())
}
