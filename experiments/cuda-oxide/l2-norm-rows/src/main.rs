// #3522 OXIDE-001 — pure-Rust cuda-oxide port of the row-batched GDN per-head L2 norm.
//
// Target = hand-PTX `PerHeadL2NormRowsKernel`
// (crates/aprender-gpu/src/kernels/gdn/rows.rs, entry `gdn_per_head_l2_norm_rows`):
// grid (heads, rows), one warp per (row, head), block 32; head_dim, eps and the row
// stride baked into the PTX; one param (x), in place.
//
//   for each row t, head h:  chunk = x[t*row_stride + h*head_dim ..][..head_dim]
//       scale = 1 / sqrt(sum(chunk^2) + eps)
//       chunk[i] *= scale
//
// Why rows are strided: in a prefill chunk `q` and `k` sit INSIDE each [conv_dim]
// conv-output row, conv_dim apart, so the heads of row t+1 do not follow those of
// row t. Everything between one row's heads and the next row's stays untouched.
//
// CPU reference = `l2_norm_per_head` (forward_qwen35.rs) applied to each row's head
// region, evaluated in f64.
//
// SAFETY SHAPE (kernel-safety, #3522): both device kernels are safe Rust — no
// `unsafe` block. The block reduction passes its shared scratch as `&raw mut SMEM`,
// which is safe to write in Rust 2024; `block_reduce` owns the unsafety internally.
// Reads go through bounds-checked slice indexing.
// The write goes through `DisjointSlice<f32, RuntimeRowMajorTiles<1, 1>>`: the
// row stride is a RUNTIME row width, bound once on the host (`RowWidth`) and read
// back on the device with `row_width()`, so every thread uses the same stride and
// the tiles of distinct (row, lane) coordinates are disjoint. `tile_2d32_rt` checks
// that the tile stays inside one row (last col < stride) and inside the slice.
// The read side indexes `x` with that same `row_width()`: one stride, one source.
//
// OUT OF PLACE, as in the per-token l2-norm port: a safe kernel cannot read `x` as
// `&[f32]` and write it through a `DisjointSlice` over the same memory. `out` starts
// as a copy of `x` and must end with only the head regions changed; the harness
// checks the padding between rows bit for bit.
//
// Geometry: one 128-thread block per (row, head), one element per thread, so the
// thread's 2-D coordinate is (col = head*128 + tid, row = t) and each warp access
// is 32 consecutive floats — coalesced, like the hand PTX's lane-strided loop.
// The sum of squares is a 4-warp `block_reduce` through 4 floats of shared memory.
//
// The first port (cb69551a0) kept the hand PTX's block of 32 but gave each lane
// TILE = 4 CONTIGUOUS floats (the only per-lane ownership the index spaces offer;
// there is no lane-strided tile). Parity was exact but every warp load and store
// then spanned 4 cache lines: timing NO-GO, worst 1.25x on sm_89 and 2.17x on
// sm_121. See GDN-DECISIONS.md, row 8.
//
// Two variants, differing only in the reciprocal square root:
//   (A) l2_norm_rows_sqrt  — `1.0 / sqrt(..)`, IEEE sqrt and divide
//   (B) l2_norm_rows_rsqrt — `float::rsqrt_approx_f32`, the hand PTX's rsqrt.approx
//
// Output: a human table on stdout plus one `apr-kernel-receipt/v1` JSON line
// prefixed `RECEIPT ` per variant; `receipt.sh` adds host facts and writes
// evidence/kernels/gdn_l2_norm_rows/<host>.json.

use cuda_core::{CudaContext, DeviceBuffer, IntoResult, LaunchConfig2D, sys};
use cuda_device::cooperative_groups::{block_reduce, ops::Sum, this_thread_block};
use cuda_device::{
    DisjointSlice, LocalIndex32, RuntimeRowMajorTiles, SharedArray, cuda_module, float, kernel,
    launch_bounds, launch_contract, thread,
};
use cuda_host::RowWidth;
use std::sync::Arc;

const HEAD_DIM: usize = 128;
const WARP: usize = 32;
const WARPS: usize = HEAD_DIM / WARP;
const EPS: f32 = 1e-6;

const PARITY_COS: f64 = 0.9999;
const PARITY_MAXDIFF: f32 = 1e-3;
const UNIT_ERR_MAX: f64 = 1e-4;
/// oxide / hand-PTX time ceiling, worst over the timed shapes.
const TIMING_RATIO_MAX: f64 = 1.2;

/// Parity cases (heads, rows, row_stride). (1, 1, HEAD_DIM) is the contiguous
/// degenerate case; 3*HEAD_DIM + 5 is a stride that is not a multiple of the tile,
/// so any row misaddressing lands off the 4-float grid; 37 rows is the aprender-gpu
/// device test's odd count.
const PARITY_CASES: [(usize, usize, usize); 4] = [
    (1, 1, HEAD_DIM),
    (3, 7, 3 * HEAD_DIM + 5),
    (16, 37, 3 * 16 * HEAD_DIM),
    (48, 64, 3 * 48 * HEAD_DIM),
];
/// Timed head counts, each with a committed hand-PTX baseline (head_dim, eps and the
/// stride are baked in), at ROWS rows of stride `3 * heads * HEAD_DIM`: Qwen3.5's
/// conv_dim (q | k | v, 16 heads -> 6144), one 64-token prefill chunk.
const TIMED_HEADS: [usize; 3] = [16, 32, 48];
const ROWS: usize = 64;

const fn timed_stride(heads: usize) -> usize {
    3 * heads * HEAD_DIM
}

#[cuda_module]
mod kernels {
    use super::*;

    /// Sum of squares over this (row, head)'s HEAD_DIM elements, one per thread,
    /// reduced across the block's 4 warps and broadcast to every thread.
    ///
    /// Every thread reaches the barrier inside `block_reduce`; the output tile
    /// bounds check comes after it, so no thread returns early past a barrier.
    #[inline(always)]
    fn block_sum_sq(v: f32) -> f32 {
        static mut SMEM: SharedArray<f32, WARPS> = SharedArray::UNINIT;
        let block = this_thread_block();
        block_reduce::<f32, Sum, WARPS>(&block, v * v, &raw mut SMEM)
    }

    /// (A) scale via `1 / sqrt`.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(128)]
    #[launch_contract(domain = 2, coordinates = u32, block = (128, 1, 1))]
    pub fn l2_norm_rows_sqrt(
        x: &[f32],
        mut out: DisjointSlice<f32, RuntimeRowMajorTiles<1, 1>>,
        eps: f32,
    ) {
        let c = thread::coord_2d_u32(launch_context);
        let v = x[c.row() as usize * out.row_width() as usize + c.col() as usize];
        let scale = 1.0f32 / (block_sum_sq(v) + eps).sqrt();
        let Some(mut o) = out.tile_2d32_rt(c) else {
            return;
        };
        o.at(LocalIndex32::constant::<0>(), LocalIndex32::constant::<0>())
            .write(v * scale);
    }

    /// (B) scale via `rsqrt.approx`, the hand PTX's form.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(128)]
    #[launch_contract(domain = 2, coordinates = u32, block = (128, 1, 1))]
    pub fn l2_norm_rows_rsqrt(
        x: &[f32],
        mut out: DisjointSlice<f32, RuntimeRowMajorTiles<1, 1>>,
        eps: f32,
    ) {
        let c = thread::coord_2d_u32(launch_context);
        let v = x[c.row() as usize * out.row_width() as usize + c.col() as usize];
        let scale = float::rsqrt_approx_f32(block_sum_sq(v) + eps);
        let Some(mut o) = out.tile_2d32_rt(c) else {
            return;
        };
        o.at(LocalIndex32::constant::<0>(), LocalIndex32::constant::<0>())
            .write(v * scale);
    }
}

/// f64 evaluation of `l2_norm_per_head` on each row's head region; the rest of
/// every row is copied through unchanged.
fn cpu_l2_norm_rows(x: &[f32], (heads, rows, stride): (usize, usize, usize)) -> Vec<f32> {
    let mut out = x.to_vec();
    for t in 0..rows {
        for h in 0..heads {
            let b = t * stride + h * HEAD_DIM;
            let chunk = &x[b..b + HEAD_DIM];
            let ss: f64 = chunk.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
            let scale = 1.0 / (ss + f64::from(EPS)).sqrt();
            for i in 0..HEAD_DIM {
                out[b + i] = (f64::from(x[b + i]) * scale) as f32;
            }
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
    dot / (na.sqrt() * nb.sqrt())
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// `rows * stride` floats: head h of every row scaled by 0.25*(h+1) (so a global or
/// per-row norm leaves heads at different lengths), padding left at unit scale.
fn make_inputs((heads, rows, stride): (usize, usize, usize), seed: u64) -> Vec<f32> {
    let mut s = seed;
    let mut next = move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    (0..rows * stride)
        .map(|j| {
            let col = j % stride;
            let scale = if col < heads * HEAD_DIM {
                0.25 * ((col / HEAD_DIM) as f32 + 1.0)
            } else {
                1.0
            };
            next() * scale
        })
        .collect()
}

/// Worst |‖head‖ - 1| over every (row, head).
fn worst_unit_error(got: &[f32], (heads, rows, stride): (usize, usize, usize)) -> f64 {
    let mut worst = 0.0f64;
    for t in 0..rows {
        for h in 0..heads {
            let b = t * stride + h * HEAD_DIM;
            let n: f64 = got[b..b + HEAD_DIM]
                .iter()
                .map(|&v| f64::from(v) * f64::from(v))
                .sum();
            worst = worst.max((n.sqrt() - 1.0).abs());
        }
    }
    worst
}

/// Every element outside the head regions is bit-identical to the input.
fn padding_intact(got: &[f32], x: &[f32], (heads, _, stride): (usize, usize, usize)) -> bool {
    got.iter()
        .zip(x)
        .enumerate()
        .all(|(j, (g, v))| j % stride < heads * HEAD_DIM || g.to_bits() == v.to_bits())
}

struct Measured {
    cos: f64,
    maxdiff: f32,
    unit_err: f64,
    padding_ok: bool,
    /// Device time per launch, from a CUDA-graph replay (the gated number).
    us: f64,
    /// Eager per-launch time: recorded, never gated (see GDN-DECISIONS.md).
    eager_us: f64,
}

impl Measured {
    fn ok(&self) -> bool {
        self.cos >= PARITY_COS
            && self.maxdiff < PARITY_MAXDIFF
            && self.unit_err < UNIT_ERR_MAX
            && self.padding_ok
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
/// in microseconds per launch. The graph takes host submission out of the number.
/// Needs a real (non-legacy) stream: capture refuses the null one.
fn time_graph_us(stream: &Arc<cuda_core::CudaStream>, mut launch: impl FnMut()) -> f64 {
    const NODES: u32 = 100;
    let s = stream.cu_stream();
    assert!(
        !s.is_null(),
        "graph capture needs a created stream, not the legacy default"
    );
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
    let replay = || {
        unsafe { sys::cuGraphLaunch(exec, s) }
            .result()
            .expect("graph launch")
    };
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

fn measure(
    got: &[f32],
    x: &[f32],
    dims: (usize, usize, usize),
    us: f64,
    eager_us: f64,
) -> Measured {
    let want = cpu_l2_norm_rows(x, dims);
    Measured {
        cos: cosine(got, &want),
        maxdiff: max_abs_diff(got, &want),
        unit_err: worst_unit_error(got, dims),
        padding_ok: padding_intact(got, x, dims),
        us,
        eager_us,
    }
}

fn seed((heads, rows, _): (usize, usize, usize)) -> u64 {
    0x3522_000A + (heads * 4096 + rows) as u64
}

fn run_oxide(
    ctx: &Arc<CudaContext>,
    module: &kernels::LoadedModule,
    dims: (usize, usize, usize),
    rsqrt: bool,
    perf: bool,
) -> Measured {
    let (heads, rows, stride) = dims;
    let stream = ctx.new_stream().expect("stream");
    let x = make_inputs(dims, seed(dims));
    let d_x = DeviceBuffer::from_host(&stream, &x).expect("x");
    // `out` starts as x: the padding the kernel must not touch is already right.
    let mut d_out = DeviceBuffer::from_host(&stream, &x).expect("out");
    let st = stride as u32;

    let config = LaunchConfig2D::new((heads as u32, rows as u32), (HEAD_DIM as u32, 1), 0);
    let launch = |d_out: &mut DeviceBuffer<f32>| {
        if rsqrt {
            let p = module
                .prepare_l2_norm_rows_rsqrt(config)
                .expect("prepare rsqrt");
            module
                .l2_norm_rows_rsqrt(&stream, &p, &d_x, RowWidth::new(d_out, st), EPS)
                .expect("launch rsqrt");
        } else {
            let p = module
                .prepare_l2_norm_rows_sqrt(config)
                .expect("prepare sqrt");
            module
                .l2_norm_rows_sqrt(&stream, &p, &d_x, RowWidth::new(d_out, st), EPS)
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
    measure(&got, &x, dims, us, eager_us)
}

/// The hand PTX for `heads` heads at the timed stride, loaded from its committed
/// baseline and launched on the same data: grid (heads, rows), block 32, one
/// pointer param, in place.
fn run_handptx(ctx: &Arc<CudaContext>, sm: &str, heads: usize, rows: usize) -> (Measured, u32) {
    let dims = (heads, rows, timed_stride(heads));
    let ptx_path = format!("baseline-ptx/gdn_per_head_l2_norm_rows_h{heads}.{sm}.ptx");
    let ptx = std::fs::read_to_string(&ptx_path).unwrap_or_else(|e| {
        eprintln!("missing hand-PTX baseline {ptx_path}: {e}");
        eprintln!("regenerate: APR_BLESS_PTX=1 cargo test -p aprender-gpu --features cuda --lib gdn_l2_norm_rows_ptx_golden");
        std::process::exit(2);
    });
    let stream = ctx.new_stream().expect("stream");
    let x = make_inputs(dims, seed(dims));
    let module = ctx.load_module_from_ptx_src(&ptx).expect("load hand PTX");
    let func = module
        .load_function("gdn_per_head_l2_norm_rows")
        .expect("gdn_per_head_l2_norm_rows");
    let regs = func.num_registers().expect("regs");
    let d_x = DeviceBuffer::from_host(&stream, &x).expect("x");
    let mut ptrs = [d_x.cu_deviceptr()];
    let mut launch = || {
        let mut params: Vec<*mut std::ffi::c_void> =
            ptrs.iter_mut().map(|p| (p as *mut u64).cast()).collect();
        // SAFETY: one device pointer matching the entry's one .u64 param; the
        // buffer holds rows * stride f32 with the stride this PTX was baked with,
        // and a (heads, rows) x 32 launch touches only the head regions of it.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &func,
                (heads as u32, rows as u32, 1),
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
    (measure(&got, &x, dims, us, eager_us), regs)
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

    println!("== #3522 cuda-oxide per-head L2 norm rows ({sm}) ==");
    println!(
        "   head_dim={HEAD_DIM} eps={EPS:e} 1 elem/thread, {WARPS} warps/(row, head), timed at {ROWS} rows"
    );

    let mut all_ok = true;
    // A timing NO-GO is its own exit code (4), not a pass: the receipts still
    // print, so receipt.sh records `"pass":false` and then fails.
    let mut timing_all_ok = true;
    for rsqrt in [false, true] {
        let name = if rsqrt { "rsqrt (B)" } else { "sqrt (A)" };
        for case in PARITY_CASES {
            let r = run_oxide(&ctx, &module, case, rsqrt, false);
            let ok = r.ok();
            all_ok &= ok;
            println!(
                "  parity {name} heads={} rows={} stride={}: cos={:.7} maxdiff={:.3e} unit_err={:.2e} padding={} {}",
                case.0,
                case.1,
                case.2,
                r.cos,
                r.maxdiff,
                r.unit_err,
                if r.padding_ok { "intact" } else { "CLOBBERED" },
                if ok { "PASS" } else { "FAIL" }
            );
        }
    }

    let regs_hand;
    {
        let (h, r) = run_handptx(&ctx, &sm, TIMED_HEADS[0], ROWS);
        regs_hand = r;
        let ok = h.ok();
        println!(
            "  hand PTX heads={}: cos={:.7} maxdiff={:.3e} unit_err={:.2e} regs={r} {}",
            TIMED_HEADS[0],
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
            ("rsqrt", "l2_norm_rows_rsqrt")
        } else {
            ("sqrt", "l2_norm_rows_sqrt")
        };
        let regs = oxide_registers(&ctx, entry);
        let mut worst_ratio = 0.0f64;
        let mut worst = (0.0f64, 0.0f64, 0usize);
        let mut worst_eager = 0.0f64;
        let mut parity = (1.0f64, 0.0f32, 0.0f64, true);
        for n in TIMED_HEADS {
            let dims = (n, ROWS, timed_stride(n));
            // Three rounds, alternating which kernel is timed first; the round with
            // the median ratio is the row. A host clock step mid-run (yoga, see
            // GDN-DECISIONS.md, causal conv1d seq) then lands in at most one round.
            let mut rounds: Vec<(Measured, Measured)> = (0..3)
                .map(|round| {
                    if round % 2 == 0 {
                        let o = run_oxide(&ctx, &module, dims, rsqrt, true);
                        (o, run_handptx(&ctx, &sm, n, ROWS).0)
                    } else {
                        let h = run_handptx(&ctx, &sm, n, ROWS).0;
                        (run_oxide(&ctx, &module, dims, rsqrt, true), h)
                    }
                })
                .collect();
            rounds.sort_by(|a, b| (a.0.us / a.1.us).total_cmp(&(b.0.us / b.1.us)));
            let (o, h) = rounds.swap_remove(1);
            let ratio = o.us / h.us;
            let ok = ratio <= TIMING_RATIO_MAX;
            let eager_ratio = o.eager_us / h.eager_us;
            println!(
                "  {n:>5} | {variant:>7} | {:>8.3} | {:>10.3} | {ratio:.3} | {:>7} | {:.3}/{:.3} = {eager_ratio:.3}",
                o.us,
                h.us,
                if ok { "GO" } else { "NO-GO" },
                o.eager_us,
                h.eager_us,
            );
            if ratio > worst_ratio {
                worst_ratio = ratio;
                worst = (o.us, h.us, n);
            }
            worst_eager = worst_eager.max(eager_ratio);
            parity = (
                parity.0.min(o.cos),
                parity.1.max(o.maxdiff),
                parity.2.max(o.unit_err),
                parity.3 && o.padding_ok,
            );
        }
        let timing_ok = worst_ratio <= TIMING_RATIO_MAX;
        timing_all_ok &= timing_ok;
        let parity_ok = parity.0 >= PARITY_COS
            && parity.1 < PARITY_MAXDIFF
            && parity.2 < UNIT_ERR_MAX
            && parity.3;
        // One line per variant; receipt.sh adds host, sha, ptxas and writes the file.
        println!(
            "RECEIPT {{\"schema\":\"apr-kernel-receipt/v1\",\"kernel\":\"gdn_l2_norm_rows\",\"variant\":\"{variant}\",\"entry\":\"{entry}\",\"authoring\":\"oxide\",\"cc\":\"{sm}\",\"head_dim\":{HEAD_DIM},\"rows\":{ROWS},\"row_stride\":\"3*heads*head_dim\",\"parity\":{{\"cos_min\":{:.9},\"maxdiff_max\":{:.3e},\"unit_err_max\":{:.3e},\"padding_intact\":{},\"cos_floor\":{PARITY_COS},\"maxdiff_ceiling\":{PARITY_MAXDIFF:e},\"unit_err_ceiling\":{UNIT_ERR_MAX:e},\"pass\":{parity_ok}}},\"timing\":{{\"oxide_us\":{:.3},\"handptx_us\":{:.3},\"ratio\":{worst_ratio:.4},\"worst_heads\":{},\"ratio_max\":{TIMING_RATIO_MAX},\"method\":\"cuda-graph-100\",\"eager_ratio_max\":{worst_eager:.4},\"pass\":{timing_ok}}},\"register_budget\":{{\"oxide\":{},\"handptx\":{regs_hand}}}}}",
            parity.0,
            parity.1,
            parity.2,
            parity.3,
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
    println!("#3522 L2 NORM ROWS DONE");
}
