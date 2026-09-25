// #3522 OXIDE-001 — pure-Rust cuda-oxide port of the GDN per-head L2 norm.
//
// Target = hand-PTX `PerHeadL2NormKernel`
// (crates/aprender-gpu/src/kernels/gdn/l2_norm.rs, entry `gdn_per_head_l2_norm`):
// one warp per head, head_dim and eps baked into the PTX, one param (x), in place.
//
//   for each head chunk of head_dim:
//       scale = 1 / sqrt(sum(x^2) + eps)     -- eps on the SUM: L2 norm, not RMSNorm
//       x[i] *= scale
//
// CPU reference = `l2_norm_per_head` in crates/aprender-serve/src/gguf/inference/
// forward/forward_qwen35.rs, evaluated here in f64.
//
// SAFETY SHAPE (kernel-safety, #3522): both device kernels are safe Rust — no
// `unsafe` block, no raw pointer. Reads go through bounds-checked slice indexing;
// the one write goes through `DisjointSlice<f32, LinearTiles<TILE>>`.
//
// OUT OF PLACE. A safe kernel cannot read `x` as `&[f32]` and write it through a
// `DisjointSlice` over the same memory, so the port takes (x, out). The hand PTX
// is in place: it loads every element twice (pass 1 and pass 2) and stores once.
// The port loads once (TILE values stay in registers) and stores once, into a
// second buffer. Both move the same bytes on the store side; the timing A/B is
// therefore "the kernel as each authoring can safely express it", which is the
// question O-4 asks. A caller that needs in place gets it by ping-ponging buffers.
//
// Geometry: one warp per head, TILE = head_dim / 32 contiguous elements per lane.
// head_dim = 128 (Qwen3.5 head_k_dim, the only production shape) gives TILE = 4.
//
// Two variants, differing only in the reciprocal square root:
//   (A) l2_norm_sqrt  — `1.0 / sqrt(..)`, IEEE sqrt and divide
//   (B) l2_norm_rsqrt — `float::rsqrt_approx_f32`, the hand PTX's rsqrt.approx
//
// Output: a human table on stdout plus one `apr-kernel-receipt/v1` JSON line
// prefixed `RECEIPT ` per variant; `receipt.sh` adds host facts and writes
// evidence/kernels/gdn_l2_norm/<host>.json.

use cuda_core::{CudaContext, DeviceBuffer, IntoResult, LaunchConfig1D, sys};
use cuda_device::{
    DisjointSlice, LinearTiles, cuda_module, float, kernel, launch_bounds, launch_contract, thread,
    warp,
};
use std::sync::Arc;

const HEAD_DIM: usize = 128;
const WARP: usize = 32;
const TILE: usize = HEAD_DIM / WARP;
const EPS: f32 = 1e-6;
const FULL: u32 = 0xFFFF_FFFF;

// F-OXIDE-ROPE-PARITY-001 form, as #3522's kernel-parity shape states it.
const PARITY_COS: f64 = 0.9999;
const PARITY_MAXDIFF: f32 = 1e-3;
// kernel-timing shape: oxide_us / handptx_us <= 1.2.
const TIMING_RATIO_MAX: f64 = 1.2;

#[cuda_module]
mod kernels {
    use super::*;

    /// Sum of squares of this lane's TILE, reduced across the warp and
    /// broadcast: the same shfl.down butterfly and lane-0 broadcast as the
    /// hand PTX.
    #[inline(always)]
    fn warp_sum_sq(x: &[f32], base: usize) -> f32 {
        let mut sq = 0.0f32;
        for k in 0..TILE {
            let v = x[base + k];
            sq += v * v;
        }
        sq += warp::shuffle_down_f32_sync(FULL, sq, 16);
        sq += warp::shuffle_down_f32_sync(FULL, sq, 8);
        sq += warp::shuffle_down_f32_sync(FULL, sq, 4);
        sq += warp::shuffle_down_f32_sync(FULL, sq, 2);
        sq += warp::shuffle_down_f32_sync(FULL, sq, 1);
        warp::shuffle_f32_sync(FULL, sq, 0)
    }

    /// (A) scale via `1 / sqrt`.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(32)]
    #[launch_contract(domain = 1, coordinates = u32, block = (32, 1, 1))]
    pub fn l2_norm_sqrt(x: &[f32], mut out: DisjointSlice<f32, LinearTiles<TILE>>, eps: f32) {
        let t = thread::index_1d_u32(launch_context);
        let base = t.get() as usize * TILE;
        let scale = 1.0f32 / (warp_sum_sq(x, base) + eps).sqrt();
        let Some(mut run) = out.thread_run32(t) else {
            return;
        };
        for k in 0..run.len() {
            let v = x[base + k as usize] * scale;
            if let Some(mut slot) = run.at(k) {
                slot.write(v);
            }
        }
    }

    /// (B) scale via `rsqrt.approx`, the hand PTX's form.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(32)]
    #[launch_contract(domain = 1, coordinates = u32, block = (32, 1, 1))]
    pub fn l2_norm_rsqrt(x: &[f32], mut out: DisjointSlice<f32, LinearTiles<TILE>>, eps: f32) {
        let t = thread::index_1d_u32(launch_context);
        let base = t.get() as usize * TILE;
        let scale = float::rsqrt_approx_f32(warp_sum_sq(x, base) + eps);
        let Some(mut run) = out.thread_run32(t) else {
            return;
        };
        for k in 0..run.len() {
            let v = x[base + k as usize] * scale;
            if let Some(mut slot) = run.at(k) {
                slot.write(v);
            }
        }
    }
}

/// f64 evaluation of `l2_norm_per_head` (forward_qwen35.rs).
fn cpu_l2_norm_per_head(x: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for (h, chunk) in x.chunks_exact(HEAD_DIM).enumerate() {
        let ss: f64 = chunk.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        let scale = 1.0 / (ss + f64::from(eps)).sqrt();
        for i in 0..HEAD_DIM {
            let j = h * HEAD_DIM + i;
            out[j] = (f64::from(x[j]) * scale) as f32;
        }
    }
    out
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (f64::from(x), f64::from(y));
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Deterministic inputs. Per-head magnitudes differ (as in the aprender-gpu
/// device test), so one norm over the whole vector instead of per head misses.
/// Cosine is scale-blind, so the per-head unit-length check below is what
/// catches a global norm; maxdiff catches it too.
fn make_inputs(heads: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let mut next = move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let mut x = Vec::with_capacity(heads * HEAD_DIM);
    for h in 0..heads {
        let scale = 0.25 * (h as f32 + 1.0);
        for _ in 0..HEAD_DIM {
            x.push(next() * scale);
        }
    }
    x
}

/// Worst |‖head‖ - 1| over heads: a global norm leaves heads at different lengths.
fn worst_unit_error(got: &[f32]) -> f64 {
    got.chunks_exact(HEAD_DIM)
        .map(|h| {
            let n: f64 = h.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
            (n.sqrt() - 1.0).abs()
        })
        .fold(0.0, f64::max)
}

struct Measured {
    cos: f64,
    maxdiff: f32,
    unit_err: f64,
    /// Device time per launch, from a CUDA-graph replay (the gated number).
    us: f64,
    /// Eager per-launch time: back-to-back stream launches. At ~2 us per kernel
    /// this measures host submission as much as the kernel, and on yoga it was
    /// bimodal (ratio 0.92 or 1.38 run to run), so it is recorded, never gated.
    eager_us: f64,
}

impl Measured {
    fn ok(&self) -> bool {
        self.cos >= PARITY_COS && self.maxdiff < PARITY_MAXDIFF && self.unit_err < 1e-4
    }
}

/// GPU-event median of 5 x 100 warm eager launches, in microseconds per launch.
fn time_eager_us(stream: &Arc<cuda_core::CudaStream>, mut launch: impl FnMut()) -> f64 {
    for _ in 0..20 {
        launch();
    }
    stream.synchronize().expect("warmup sync");
    let flags = Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT);
    let iters = 100;
    let mut times: Vec<f64> = (0..5)
        .map(|_| {
            let start = stream.record_event(flags).expect("event");
            for _ in 0..iters {
                launch();
            }
            let end = stream.record_event(flags).expect("event");
            f64::from(start.elapsed_ms(&end).expect("elapsed")) * 1000.0 / f64::from(iters)
        })
        .collect();
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

/// GPU-event median of 5 replays of a CUDA graph holding 100 captured launches,
/// in microseconds per launch. The graph takes host submission out of the number,
/// so it is the kernel's device time plus the graph's per-node cost, the same for
/// both authorings. Needs a real (non-legacy) stream: capture refuses the null one.
fn time_graph_us(stream: &Arc<cuda_core::CudaStream>, mut launch: impl FnMut()) -> f64 {
    const NODES: u32 = 100;
    let s = stream.cu_stream();
    assert!(!s.is_null(), "graph capture needs a created stream, not the legacy default");
    // SAFETY: `s` is a live stream owned by `stream`; the graph and its executable
    // are created, launched on `s` and destroyed inside this function, and the
    // captured launches reference buffers the caller keeps alive across the call.
    let exec = unsafe {
        sys::cuStreamBeginCapture_v2(
            s,
            sys::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        )
        .result()
        .expect("begin capture");
        for _ in 0..NODES {
            launch();
        }
        let mut graph = std::ptr::null_mut();
        sys::cuStreamEndCapture(s, &mut graph)
            .result()
            .expect("end capture");
        let mut exec = std::ptr::null_mut();
        sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0)
            .result()
            .expect("instantiate");
        sys::cuGraphDestroy(graph).result().expect("destroy graph");
        exec
    };
    // SAFETY: `exec` is the executable instantiated above, `s` its stream.
    let replay = || unsafe { sys::cuGraphLaunch(exec, s) }.result().expect("graph launch");
    for _ in 0..3 {
        replay();
    }
    stream.synchronize().expect("warmup sync");
    let flags = Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT);
    let mut times: Vec<f64> = (0..5)
        .map(|_| {
            let start = stream.record_event(flags).expect("event");
            replay();
            let end = stream.record_event(flags).expect("event");
            f64::from(start.elapsed_ms(&end).expect("elapsed")) * 1000.0 / f64::from(NODES)
        })
        .collect();
    // SAFETY: nothing else holds `exec`; its replays completed in elapsed_ms.
    unsafe { sys::cuGraphExecDestroy(exec) }
        .result()
        .expect("destroy exec");
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

fn measure(got: &[f32], x: &[f32], us: f64, eager_us: f64) -> Measured {
    let want = cpu_l2_norm_per_head(x, EPS);
    Measured {
        cos: cosine(got, &want),
        maxdiff: max_abs_diff(got, &want),
        unit_err: worst_unit_error(got),
        us,
        eager_us,
    }
}

fn run_oxide(
    ctx: &Arc<CudaContext>,
    module: &kernels::LoadedModule,
    heads: usize,
    rsqrt: bool,
    perf: bool,
) -> Measured {
    let stream = ctx.new_stream().expect("stream");
    let x = make_inputs(heads, 0x3522_0002 + heads as u64);
    let d_x = DeviceBuffer::from_host(&stream, &x).expect("x");
    let mut d_out = DeviceBuffer::<f32>::zeroed(&stream, heads * HEAD_DIM).expect("out");

    let config = LaunchConfig1D::new(heads as u32, WARP as u32, 0);
    let launch = |d_out: &mut DeviceBuffer<f32>| {
        if rsqrt {
            let p = module.prepare_l2_norm_rsqrt(config).expect("prepare rsqrt");
            module
                .l2_norm_rsqrt(&stream, &p, &d_x, d_out, EPS)
                .expect("launch rsqrt");
        } else {
            let p = module.prepare_l2_norm_sqrt(config).expect("prepare sqrt");
            module
                .l2_norm_sqrt(&stream, &p, &d_x, d_out, EPS)
                .expect("launch sqrt");
        }
    };
    launch(&mut d_out);
    let got = d_out.to_host_vec(&stream).expect("download");
    let (us, eager_us) = if perf {
        (
            time_graph_us(&stream, || launch(&mut d_out)),
            time_eager_us(&stream, || launch(&mut d_out)),
        )
    } else {
        (0.0, 0.0)
    };
    measure(&got, &x, us, eager_us)
}

/// The hand PTX, loaded from the committed baseline and launched on the same data:
/// grid (heads), block (32), one pointer param, in place.
fn run_handptx(ctx: &Arc<CudaContext>, ptx: &str, heads: usize) -> (Measured, u32) {
    let stream = ctx.new_stream().expect("stream");
    let x = make_inputs(heads, 0x3522_0002 + heads as u64);
    let module = ctx.load_module_from_ptx_src(ptx).expect("load hand PTX");
    let func = module
        .load_function("gdn_per_head_l2_norm")
        .expect("gdn_per_head_l2_norm");
    let regs = func.num_registers().expect("regs");
    let d_x = DeviceBuffer::from_host(&stream, &x).expect("x");
    let mut ptrs = [d_x.cu_deviceptr()];
    let mut launch = || {
        let mut params: Vec<*mut std::ffi::c_void> =
            ptrs.iter_mut().map(|p| (p as *mut u64).cast()).collect();
        // SAFETY: one device pointer matching the entry's one .u64 param; the
        // buffer holds heads * HEAD_DIM f32, which is what a (heads, 32) launch
        // of this kernel touches.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &func,
                (heads as u32, 1, 1),
                (WARP as u32, 1, 1),
                0,
                &stream,
                &mut params,
            )
        }
        .expect("hand PTX launch");
    };
    // Parity from the first launch; timing then re-normalises in place, which
    // does the same work on already-unit heads.
    launch();
    let got = d_x.to_host_vec(&stream).expect("download");
    let us = time_graph_us(&stream, &mut launch);
    let eager_us = time_eager_us(&stream, &mut launch);
    (measure(&got, &x, us, eager_us), regs)
}

/// Register count of an oxide kernel, read back from the embedded module.
fn oxide_registers(ctx: &Arc<CudaContext>, name: &str) -> Option<u32> {
    let module = cuda_host::embedded::load_all_ptx_bundles_merged(ctx).ok()?;
    module.load_function(name).ok()?.num_registers().ok()
}

fn main() {
    let ctx = CudaContext::new(0).expect("ctx");
    let (major, minor) = ctx.compute_capability().expect("cc");
    let sm = format!("sm_{major}{minor}");
    // SAFETY: the embedded module is built from this crate's `kernels`.
    let module = unsafe { kernels::load(&ctx) }.expect("load oxide module");

    let ptx_path = format!("baseline-ptx/gdn_per_head_l2_norm_h{HEAD_DIM}.{sm}.ptx");
    let ptx = std::fs::read_to_string(&ptx_path).unwrap_or_else(|e| {
        eprintln!("missing hand-PTX baseline {ptx_path}: {e}");
        eprintln!("regenerate: APR_BLESS_PTX=1 cargo test -p aprender-gpu --features cuda --lib gdn_l2_norm_ptx_golden");
        std::process::exit(2);
    });

    println!("== #3522 cuda-oxide per-head L2 norm ({sm}) ==");
    println!("   head_dim={HEAD_DIM} eps={EPS:e} tile={TILE}/lane, 1 warp/head");

    let mut all_ok = true;
    // A timing NO-GO is its own exit code (4), not a pass: the receipts still
    // print, so receipt.sh records `"pass":false` and then fails.
    let mut timing_all_ok = true;
    for rsqrt in [false, true] {
        let name = if rsqrt { "rsqrt (B)" } else { "sqrt (A)" };
        for heads in [1usize, 16, 32, 48] {
            let r = run_oxide(&ctx, &module, heads, rsqrt, false);
            let ok = r.ok();
            all_ok &= ok;
            println!(
                "  parity {name} heads={heads:>2}: cos={:.7} maxdiff={:.3e} unit_err={:.2e} {}",
                r.cos,
                r.maxdiff,
                r.unit_err,
                if ok { "PASS" } else { "FAIL" }
            );
        }
    }

    // Timing: the production shape (16 heads) plus 32 and 48.
    let regs_hand;
    {
        let (h, r) = run_handptx(&ctx, &ptx, 16);
        regs_hand = r;
        let ok = h.ok();
        println!(
            "  hand PTX heads=16: cos={:.7} maxdiff={:.3e} unit_err={:.2e} regs={r} {}",
            h.cos,
            h.maxdiff,
            h.unit_err,
            if ok { "PASS" } else { "FAIL" }
        );
        all_ok &= ok;
    }
    println!("\n  heads | variant | oxide us | handPTX us | ratio | verdict | eager oxide/hand");
    for rsqrt in [false, true] {
        let (variant, entry) = if rsqrt {
            ("rsqrt", "l2_norm_rsqrt")
        } else {
            ("sqrt", "l2_norm_sqrt")
        };
        let regs = oxide_registers(&ctx, entry);
        let mut worst_ratio = 0.0f64;
        let mut worst = (0.0f64, 0.0f64, 0usize);
        let mut worst_eager = 0.0f64;
        let mut parity = (1.0f64, 0.0f32, 0.0f64);
        for heads in [16usize, 32, 48] {
            let o = run_oxide(&ctx, &module, heads, rsqrt, true);
            let (h, _) = run_handptx(&ctx, &ptx, heads);
            let ratio = o.us / h.us;
            let ok = ratio <= TIMING_RATIO_MAX;
            let eager_ratio = o.eager_us / h.eager_us;
            println!(
                "  {heads:>5} | {variant:>7} | {:>8.3} | {:>10.3} | {ratio:.3} | {:>7} | {:.3}/{:.3} = {eager_ratio:.3}",
                o.us,
                h.us,
                if ok { "GO" } else { "NO-GO" },
                o.eager_us,
                h.eager_us,
            );
            if ratio > worst_ratio {
                worst_ratio = ratio;
                worst = (o.us, h.us, heads);
            }
            worst_eager = worst_eager.max(eager_ratio);
            parity = (
                parity.0.min(o.cos),
                parity.1.max(o.maxdiff),
                parity.2.max(o.unit_err),
            );
        }
        let timing_ok = worst_ratio <= TIMING_RATIO_MAX;
        timing_all_ok &= timing_ok;
        let parity_ok = parity.0 >= PARITY_COS && parity.1 < PARITY_MAXDIFF && parity.2 < 1e-4;
        // One line per variant; receipt.sh adds host, sha, ptxas and writes the file.
        println!(
            "RECEIPT {{\"schema\":\"apr-kernel-receipt/v1\",\"kernel\":\"gdn_l2_norm\",\"variant\":\"{variant}\",\"entry\":\"{entry}\",\"authoring\":\"oxide\",\"cc\":\"{sm}\",\"head_dim\":{HEAD_DIM},\"parity\":{{\"cos_min\":{:.9},\"maxdiff_max\":{:.3e},\"unit_err_max\":{:.3e},\"cos_floor\":{PARITY_COS},\"maxdiff_ceiling\":{PARITY_MAXDIFF:e},\"unit_err_ceiling\":1e-4,\"pass\":{parity_ok}}},\"timing\":{{\"oxide_us\":{:.3},\"handptx_us\":{:.3},\"ratio\":{worst_ratio:.4},\"worst_heads\":{},\"ratio_max\":{TIMING_RATIO_MAX},\"method\":\"cuda-graph-100\",\"eager_ratio_max\":{worst_eager:.4},\"pass\":{timing_ok}}},\"register_budget\":{{\"oxide\":{},\"handptx\":{regs_hand}}}}}",
            parity.0,
            parity.1,
            parity.2,
            worst.0,
            worst.1,
            worst.2,
            regs.map_or("null".to_string(), |r| r.to_string()),
        );
    }

    if !all_ok {
        eprintln!("#3522 PARITY FAILED");
        std::process::exit(1);
    }
    if !timing_all_ok {
        eprintln!("#3522 TIMING NO-GO (oxide/hand > {TIMING_RATIO_MAX})");
        std::process::exit(4);
    }
    println!("#3522 L2 DONE");
}
