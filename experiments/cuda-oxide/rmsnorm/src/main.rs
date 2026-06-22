// PMAT-893 — pure-Rust cuda-oxide port of the RMSNorm kernel.
//
// Target = hand-PTX `RmsNormKernel`
// (crates/aprender-gpu/src/kernels/layernorm/rmsnorm.rs, entry `rmsnorm`), a
// single-warp (32-thread, 1-block) kernel: each lane strides hidden by 32,
// FMA-accumulates the sum-of-squares, warp-reduces via shfl.down, then computes
// rms_inv = rsqrt(mean(x^2) + eps) and writes out[i] = x[i] * rms_inv * gamma[i].
//
// CPU reference = the same math the serve `rmsnorm` path computes
// (crates/aprender-serve/src/cuda/executor/layers/rmsnorm.rs dispatch -> the
// RmsNormKernel above): out = x / sqrt(mean(x^2) + eps) * gamma.
//
// RMSNorm is pure f32 FMA + a warp-shuffle reduce + rsqrt — ZERO DP4A. Per the
// PMAT-882 verdict (FMA/softmax kernels are the GO class; DP4A-bound Q4K GEMV/FFN
// are NO-GO), RMSNorm is squarely the GO class. It is also strongly memory-
// bandwidth-bound (2 reads + 1 write per element, ~3 FLOP/element), so the
// expected outcome is parity (a tie) that RETIRES the hand-PTX + the GH-480
// Blackwell-JIT workaround — that is still a GO on the <=1.2x perf gate.
//
// Layout: one ROW per block. For B rows of width `hidden`:
//   x     : [B * hidden]  f32   (row r at r*hidden)
//   gamma : [hidden]      f32   (shared across rows, as in the live serve path)
//   out   : [B * hidden]  f32
//
// Two kernels are provided:
//   (A) rmsnorm_warp   — 1 warp (32 threads) per row, faithful hand-PTX analog.
//   (B) rmsnorm_block  — 256 threads (8 warps) per row, shared-mem cross-warp
//       reduce. The faster analog of the production VectorizedRmsNormKernel for
//       the larger hidden sizes (4096/8192) where a single warp is occupancy-
//       starved. (B) is the GO candidate at large hidden; (A) at small.
//
// TRUE hand-PTX A/B: the actual `RmsNormKernel::new(hidden).with_epsilon(eps)
// .emit_ptx_for_target("sm_121")` PTX is emitted (committed in baseline-ptx/),
// loaded via load_module_from_ptx_src, and launched on the same GB10 with the
// same data + GPU-event timing (median of 5x N launches). Both verified parity-
// correct vs the f64 CPU reference inside the harness.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::shared::SharedArray;
use cuda_device::{kernel, thread, warp};
use cuda_host::cuda_module;

const EPS: f32 = 1e-5;

// kernel (A): 1 warp per row.
const WARP: usize = 32;
// kernel (B): 256 threads (8 warps) per row; shared = 8 warp partial sums.
const TB: usize = 256;
const NWARP_B: usize = TB / 32; // 8
const SMEM_B: usize = NWARP_B; // 8 f32 partials

#[cuda_module]
mod kernels {
    use super::*;

    // ---- (A) single-warp RMSNorm — faithful hand-PTX `rmsnorm` analog ----
    // Grid = (rows,1,1). Block = (32,1,1). One warp per row.
    #[kernel]
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn rmsnorm_warp(x: &[f32], gamma: &[f32], out: &[f32], hidden: u32, eps: f32) {
        unsafe {
            let row = thread::blockIdx_x();
            let lane = thread::threadIdx_x();
            let h = hidden as usize;
            let base = (row * hidden) as usize;

            // sum of squares: lane strides by 32 (matches hand-PTX `idx += 32`).
            let mut sq = 0.0f32;
            let mut i = lane;
            while i < hidden {
                let v = x[base + i as usize];
                sq += v * v; // FMA
                i += 32;
            }
            // warp reduce (shfl.down butterfly, identical order to hand-PTX).
            sq += warp::shuffle_down_f32_sync(0xFFFF_FFFF, sq, 16);
            sq += warp::shuffle_down_f32_sync(0xFFFF_FFFF, sq, 8);
            sq += warp::shuffle_down_f32_sync(0xFFFF_FFFF, sq, 4);
            sq += warp::shuffle_down_f32_sync(0xFFFF_FFFF, sq, 2);
            sq += warp::shuffle_down_f32_sync(0xFFFF_FFFF, sq, 1);
            // broadcast lane-0 total to all lanes.
            let total = warp::shuffle_f32_sync(0xFFFF_FFFF, sq, 0);

            let mean_sq = total / (h as f32);
            let rms_inv = 1.0f32 / (mean_sq + eps).sqrt();

            // normalize + scale: out[i] = x[i] * rms_inv * gamma[i].
            let mut j = lane;
            while j < hidden {
                let g = gamma[j as usize];
                let v = x[base + j as usize];
                let r = v * rms_inv * g;
                let op = &mut *(out.as_ptr().add(base + j as usize) as *mut f32);
                *op = r;
                j += 32;
            }
        }
    }

    // ---- (B) block (256-thread / 8-warp) RMSNorm ----
    // Grid = (rows,1,1). Block = (256,1,1). 8 warps cooperate per row via SMEM.
    // The analog of the production VectorizedRmsNormKernel; better occupancy at
    // large hidden where a single warp under-utilizes memory bandwidth.
    #[kernel]
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn rmsnorm_block(x: &[f32], gamma: &[f32], out: &[f32], hidden: u32, eps: f32) {
        static mut SMEM: SharedArray<f32, SMEM_B> = SharedArray::UNINIT;
        unsafe {
            let row = thread::blockIdx_x();
            let tid = thread::threadIdx_x();
            let widx = (tid / 32) as usize;
            let lane = warp::lane_id();
            let h = hidden as usize;
            let base = (row * hidden) as usize;

            // pass 1: sum of squares, lane strides by 256 (block size).
            let mut sq = 0.0f32;
            let mut i = tid;
            while i < hidden {
                let v = x[base + i as usize];
                sq += v * v;
                i += 256;
            }
            // intra-warp reduce.
            sq += warp::shuffle_down_f32_sync(0xFFFF_FFFF, sq, 16);
            sq += warp::shuffle_down_f32_sync(0xFFFF_FFFF, sq, 8);
            sq += warp::shuffle_down_f32_sync(0xFFFF_FFFF, sq, 4);
            sq += warp::shuffle_down_f32_sync(0xFFFF_FFFF, sq, 2);
            sq += warp::shuffle_down_f32_sync(0xFFFF_FFFF, sq, 1);
            // each warp's lane 0 publishes its partial.
            if lane == 0 {
                SMEM[widx] = sq;
            }
            thread::sync_threads();

            // warp 0 reduces the 8 warp partials (lanes 0..8) and broadcasts.
            let mut total = 0.0f32;
            if widx == 0 {
                let mut wp = if (lane as usize) < NWARP_B {
                    SMEM[lane as usize]
                } else {
                    0.0f32
                };
                wp += warp::shuffle_down_f32_sync(0xFFFF_FFFF, wp, 4);
                wp += warp::shuffle_down_f32_sync(0xFFFF_FFFF, wp, 2);
                wp += warp::shuffle_down_f32_sync(0xFFFF_FFFF, wp, 1);
                if lane == 0 {
                    SMEM[0] = wp;
                }
            }
            thread::sync_threads();
            total = SMEM[0] + (total - total); // read broadcast slot

            let mean_sq = total / (h as f32);
            let rms_inv = 1.0f32 / (mean_sq + eps).sqrt();

            // pass 2: normalize + scale.
            let mut j = tid;
            while j < hidden {
                let g = gamma[j as usize];
                let v = x[base + j as usize];
                let r = v * rms_inv * g;
                let op = &mut *(out.as_ptr().add(base + j as usize) as *mut f32);
                *op = r;
                j += 256;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CPU reference — f64 accumulation, the serve `rmsnorm` math.
// out[r,i] = x[r,i] / sqrt(mean_i(x[r,:]^2) + eps) * gamma[i]
// ---------------------------------------------------------------------------
fn cpu_rmsnorm(x: &[f32], gamma: &[f32], rows: usize, hidden: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * hidden];
    for r in 0..rows {
        let base = r * hidden;
        let mut ss = 0.0f64;
        for i in 0..hidden {
            let v = x[base + i] as f64;
            ss += v * v;
        }
        let mean_sq = ss / hidden as f64;
        let rms_inv = 1.0f64 / (mean_sq + eps as f64).sqrt();
        for i in 0..hidden {
            out[base + i] = ((x[base + i] as f64) * rms_inv * (gamma[i] as f64)) as f32;
        }
    }
    out
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..a.len() {
        dot += a[i] as f64 * b[i] as f64;
        na += a[i] as f64 * a[i] as f64;
        nb += b[i] as f64 * b[i] as f64;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    let mut m = 0.0f32;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        if d > m {
            m = d;
        }
    }
    m
}

// Deterministic pseudo-random inputs in ~[-1,1] (same generator family as 882).
fn make_inputs(rows: usize, hidden: usize, seed: usize) -> (Vec<f32>, Vec<f32>) {
    let g = |i: usize| -> f32 {
        let x = (i.wrapping_mul(2654435761).wrapping_add(seed.wrapping_mul(40503))) & 0xFFFF;
        (x as f32 / 32768.0) - 1.0
    };
    let x: Vec<f32> = (0..rows * hidden).map(g).collect();
    // gamma centered near 1.0 (real RMSNorm weights), in ~[0.5, 1.5].
    let gamma: Vec<f32> = (0..hidden).map(|i| 1.0 + 0.5 * g(i + 7777)).collect();
    (x, gamma)
}

struct OxideResult {
    cos: f32,
    maxdiff: f32,
    us: f64,
}

fn run_oxide(
    ctx: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    rows: usize,
    hidden: usize,
    seed: usize,
    do_perf: bool,
    block_variant: bool, // false = (A) warp, true = (B) block
) -> OxideResult {
    let stream = ctx.default_stream();
    let (x, gamma) = make_inputs(rows, hidden, seed);

    let dx = DeviceBuffer::from_host(&stream, &x).unwrap();
    let dg = DeviceBuffer::from_host(&stream, &gamma).unwrap();
    let dout = DeviceBuffer::<f32>::zeroed(&stream, rows * hidden).unwrap();

    let block = if block_variant { TB } else { WARP };
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (block as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let launch = || {
        if block_variant {
            module
                .rmsnorm_block(&stream, cfg, &dx, &dg, &dout, hidden as u32, EPS)
                .expect("rmsnorm_block");
        } else {
            module
                .rmsnorm_warp(&stream, cfg, &dx, &dg, &dout, hidden as u32, EPS)
                .expect("rmsnorm_warp");
        }
    };

    launch();
    let got = dout.to_host_vec(&stream).unwrap();
    let want = cpu_rmsnorm(&x, &gamma, rows, hidden, EPS);
    let cos = cosine_similarity(&got, &want);
    let maxdiff = max_abs_diff(&got, &want);

    let mut med = 0.0f64;
    if do_perf {
        for _ in 0..20 {
            launch();
        }
        let _ = dout.to_host_vec(&stream).unwrap();
        let iters = 100usize;
        let reps = 5;
        let mut times = Vec::new();
        for _ in 0..reps {
            let start = stream
                .record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
                .unwrap();
            for _ in 0..iters {
                launch();
            }
            let end = stream
                .record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
                .unwrap();
            let ms = start.elapsed_ms(&end).unwrap() as f64;
            times.push(ms * 1000.0 / iters as f64); // us/launch
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        med = times[times.len() / 2];
    }
    OxideResult {
        cos,
        maxdiff,
        us: med,
    }
}

// ---------------------------------------------------------------------------
// TRUE hand-PTX A/B: load the emitted `rmsnorm` PTX (RmsNormKernel, single warp,
// hidden+eps baked in) and launch it on the same data. The hand-PTX processes
// ONE row per launch (input_ptr/output_ptr/gamma_ptr, 1 block of 32 threads), so
// for B rows we launch B times with per-row offset pointers — exactly how the
// serve executor invokes it per decode token. Timing sums all rows per "launch".
// ---------------------------------------------------------------------------
fn run_handptx(
    ctx: &std::sync::Arc<CudaContext>,
    ptx_path: &str,
    rows: usize,
    hidden: usize,
    seed: usize,
) -> Option<(f32, f32, f64)> {
    let stream = ctx.default_stream();
    let (x, gamma) = make_inputs(rows, hidden, seed);

    let ptx = std::fs::read_to_string(ptx_path).ok()?;
    let module = ctx.load_module_from_ptx_src(&ptx).ok()?;
    let func = module.load_function("rmsnorm").ok()?;

    let dx = DeviceBuffer::from_host(&stream, &x).ok()?;
    let dg = DeviceBuffer::from_host(&stream, &gamma).ok()?;
    let dout = DeviceBuffer::<f32>::zeroed(&stream, rows * hidden).ok()?;

    let x_base = dx.cu_deviceptr();
    let g_ptr = dg.cu_deviceptr();
    let o_base = dout.cu_deviceptr();
    let grid = (1u32, 1u32, 1u32); // hand-PTX = 1 block per row
    let block = (32u32, 1u32, 1u32); // single warp

    // launches all `rows` rows (one launch per row, offset pointers).
    let launch_all = || -> Result<(), ()> {
        for r in 0..rows {
            let mut in_ptr = x_base + (r * hidden * 4) as u64;
            let mut out_ptr = o_base + (r * hidden * 4) as u64;
            let mut gam_ptr = g_ptr;
            unsafe {
                let mut params: [*mut std::ffi::c_void; 3] = [
                    &mut in_ptr as *mut _ as *mut std::ffi::c_void,
                    &mut out_ptr as *mut _ as *mut std::ffi::c_void,
                    &mut gam_ptr as *mut _ as *mut std::ffi::c_void,
                ];
                cuda_core::launch_kernel_on_stream(&func, grid, block, 0, &stream, &mut params)
                    .map_err(|_| ())?;
            }
        }
        Ok(())
    };

    launch_all().ok()?;
    let got = dout.to_host_vec(&stream).ok()?;
    let want = cpu_rmsnorm(&x, &gamma, rows, hidden, EPS);
    let cos = cosine_similarity(&got, &want);
    let maxdiff = max_abs_diff(&got, &want);

    for _ in 0..20 {
        launch_all().ok()?;
    }
    let _ = dout.to_host_vec(&stream).ok()?;
    let iters = 100usize;
    let reps = 5;
    let mut times = Vec::new();
    for _ in 0..reps {
        let start = stream
            .record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
            .ok()?;
        for _ in 0..iters {
            launch_all().ok()?;
        }
        let end = stream
            .record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
            .ok()?;
        let ms = start.elapsed_ms(&end).ok()? as f64;
        times.push(ms * 1000.0 / iters as f64);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some((cos, maxdiff, times[times.len() / 2]))
}

fn main() {
    let ctx = CudaContext::new(0).expect("ctx");
    let module = kernels::load(&ctx).expect("load");

    println!("== PMAT-893 cuda-oxide RMSNorm — parity + perf (GB10 sm_121) ==");
    println!("   (A) warp=32/row   (B) block=256/row   eps={EPS:e}");

    let hiddens = [2048usize, 4096, 8192];
    let rows = 8usize; // a few rows (decode-batch-like)

    // PARITY for both kernels across all hidden sizes.
    for (variant, name) in [(false, "warp (A)"), (true, "block (B)")] {
        let mut all_pass = true;
        println!("\n-- PARITY {name} (rows={rows}) vs f64 CPU reference --");
        for &h in &hiddens {
            let r = run_oxide(&ctx, &module, rows, h, 4242 + h, false, variant);
            // generous tol: f64-ref vs f32-kernel, bandwidth-bound, magnitude ~1.
            let pass = r.cos >= 0.9999 && r.maxdiff < 1e-4;
            if !pass {
                all_pass = false;
            }
            println!(
                "  hidden={:>5} : cos={:.7} maxdiff={:.3e} -> {}",
                h,
                r.cos,
                r.maxdiff,
                if pass { "PASS" } else { "FAIL" }
            );
        }
        println!("PARITY {name}: {}", if all_pass { "PASS" } else { "FAIL" });
        if !all_pass {
            eprintln!("PMAT-893 PARITY FAILED ({name})");
            std::process::exit(1);
        }
    }

    let find_ptx = |names: &[&str]| -> Option<String> {
        for n in names {
            if std::path::Path::new(n).exists() {
                return Some((*n).to_string());
            }
        }
        None
    };
    let ptx_for = |h: usize| -> Option<String> {
        find_ptx(&[
            &format!("/tmp/rmsnorm_spike/rmsnorm_h{h}.ptx"),
            &format!("baseline-ptx/rmsnorm_h{h}.sm121.ptx"),
        ])
    };

    // =========================================================================
    // PRIMARY GATE — FAIR single-ROW A/B (F-OXIDE-RMSNORM-PARITY-001).
    //
    // METHODOLOGY HONESTY: the hand-PTX `rmsnorm` is a SINGLE-ROW, SINGLE-BLOCK
    // (1 warp) kernel — exactly one row per launch (how the serve executor calls
    // it per decode token). To compare like-for-like we run rows=1 for BOTH: each
    // side does exactly ONE grid launch of ONE block. This removes the launch-
    // count confound (a multi-row oxide launch vs N hand-PTX relaunches would
    // unfairly favor oxide on launch overhead, not kernel quality). The fair gate
    // is per-row throughput. Oxide (A) is the matched 1-warp/row analog; (B) is
    // the 256-thread/row analog (extra occupancy at large hidden).
    // =========================================================================
    println!("\n-- PRIMARY GATE: FAIR single-row A/B (oxide 1 block vs hand-PTX 1 block, GB10 sm_121) --");
    println!("   GPU-event median of 5x100, rows=1 (one block-launch each side)");
    println!(
        "   F-OXIDE-RMSNORM-PARITY-001: cos>=0.9999 AND maxdiff<1e-4 AND oxide_us/handptx_us<=1.2"
    );
    println!(
        "\n  {:>6} | {:>9} {:>9} | {:>11} | {:>8} {:>8} | {:>10} {:>9} | verdict",
        "hidden", "oxA us", "oxB us", "handPTX us", "ratioA", "ratioB", "cos(ox)", "maxdiff"
    );
    let mut overall_go = true;
    for &h in &hiddens {
        let ra = run_oxide(&ctx, &module, 1, h, 77, true, false);
        let rb = run_oxide(&ctx, &module, 1, h, 77, true, true);
        let (best_us, best_cos, best_md, best_name) = if ra.us <= rb.us {
            (ra.us, ra.cos, ra.maxdiff, "A")
        } else {
            (rb.us, rb.cos, rb.maxdiff, "B")
        };
        match ptx_for(h).as_deref().and_then(|p| run_handptx(&ctx, p, 1, h, 77)) {
            Some((hcos, hmd, hus)) => {
                let best_ratio = best_us / hus;
                let parity_ok = best_cos >= 0.9999 && best_md < 1e-4;
                let go = parity_ok && best_ratio <= 1.2;
                if !go {
                    overall_go = false;
                }
                println!(
                    "  {:>6} | {:>9.3} {:>9.3} | {:>11.3} | {:>8.3} {:>8.3} | {:>10.7} {:>9.2e} | {} (best={} {:.3}x; handPTX cos={:.5} md={:.2e})",
                    h, ra.us, rb.us, hus, ra.us / hus, rb.us / hus, best_cos, best_md,
                    if go { "GO" } else { "NO-GO" }, best_name, best_ratio, hcos, hmd
                );
            }
            None => {
                overall_go = false;
                println!("  {h:>6} | hand-PTX MISSING/launch-FAILED (regen baseline-ptx/rmsnorm_h{h}.sm121.ptx)");
            }
        }
    }

    // =========================================================================
    // SECONDARY — multi-row throughput (rows=8). Reported for context only, NOT
    // the gate. Here the oxide kernel does all 8 rows in ONE grid launch while the
    // hand-PTX must relaunch 8x (it has no blockIdx row dispatch). The large
    // ratio is therefore DOMINATED by hand-PTX launch overhead, not kernel speed
    // — an honest, expected artifact of the hand-PTX's single-row ABI. It does
    // show the real-world batched-decode win (one oxide launch replaces N), but
    // we do NOT claim it as a per-kernel speedup.
    // =========================================================================
    let rows8 = 8usize;
    println!("\n-- SECONDARY (context only, NOT the gate): {rows8}-row throughput --");
    println!("   oxide = 1 grid launch ({rows8} blocks); hand-PTX = {rows8} relaunches (single-row ABI)");
    println!("   ratio here is launch-overhead-dominated; reported for batched-decode context");
    println!("\n  {:>6} | {:>9} {:>9} | {:>11} | note", "hidden", "oxA us", "oxB us", "handPTX us");
    for &h in &hiddens {
        let ra = run_oxide(&ctx, &module, rows8, h, 77, true, false);
        let rb = run_oxide(&ctx, &module, rows8, h, 77, true, true);
        match ptx_for(h).as_deref().and_then(|p| run_handptx(&ctx, p, rows8, h, 77)) {
            Some((_hcos, _hmd, hus)) => {
                let best = ra.us.min(rb.us);
                println!(
                    "  {:>6} | {:>9.3} {:>9.3} | {:>11.3} | 1 oxide launch vs {rows8} hand-PTX launches ({:.1}x fewer-launch win)",
                    h, ra.us, rb.us, hus, hus / best
                );
            }
            None => println!("  {h:>6} | (hand-PTX missing)"),
        }
    }

    println!(
        "\nPMAT-893 VERDICT (F-OXIDE-RMSNORM-PARITY-001, fair single-row gate): {}",
        if overall_go { "GO" } else { "NO-GO" }
    );
    println!("PMAT-893 DONE");
}
