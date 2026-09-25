// #3522 OXIDE-001 — pure-Rust cuda-oxide port of the GDN q|gate de-interleave.
//
// Target = hand-PTX `SplitInterleavedKernel`
// (crates/aprender-gpu/src/kernels/gdn/split_interleave.rs, entry
// `gdn_split_interleaved_q_gate`): grid (heads), block 256 striding over head_dim;
// head_dim baked into the PTX; params (src, q, gate).
//
//   q   [h*head_dim + i] = src[h*2*head_dim + i]
//   gate[h*head_dim + i] = src[h*2*head_dim + head_dim + i]
//
// CPU reference = `forward_attention`'s two `copy_from_slice`s. Pure data movement:
// the gate is BIT-EXACT, not a tolerance (cos and max|Δ| are recorded too, for the
// receipt schema).
//
// SAFETY SHAPE (kernel-safety, #3522): the device kernel is safe Rust — no `unsafe`
// block, no raw pointer. Reads go through `slice::get`. Each output is a
// `DisjointSlice<f32, LinearTiles<1>>` claimed as a one-element run at the thread's
// global index, so every thread owns exactly its own q and gate element and
// `thread_run32` checks both stay inside their slices.
//
// Geometry: the hand kernel's launch, grid (heads), block 256, at head_dim = 256 =
// block, so the hand kernel's stride loop runs once and the oxide kernel has none:
// global index i = h*head_dim + j, and the source index is i + (i / head_dim) *
// head_dim (a shift and an add; head_dim is a power of two).
//
// One variant: there is no transcendental to choose between.
//
// Output: a human table on stdout plus one `apr-kernel-receipt/v1` JSON line
// prefixed `RECEIPT `; `receipt.sh` adds host facts and writes
// evidence/kernels/gdn_split_interleave/<host>.json.

use cuda_core::{CudaContext, DeviceBuffer, IntoResult, LaunchConfig1D, sys};
use cuda_device::{
    DisjointSlice, LinearTiles, ThreadRunMut32, cuda_module, kernel, launch_bounds, launch_contract,
    thread,
};
use std::sync::Arc;

const HEAD_DIM: usize = 256;
const BLOCK: u32 = 256;
const _: () = assert!(HEAD_DIM == BLOCK as usize);

const TIMING_RATIO_MAX: f64 = 1.2;

/// Head counts: the identity-sized case, Qwen3.5-0.8B's 16 and the timed widths.
const PARITY_HEADS: [usize; 4] = [1, 16, 32, 48];
const TIMED_HEADS: [usize; 3] = [16, 32, 48];

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel(launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub fn split_interleave(
        src: &[f32],
        mut q: DisjointSlice<f32, LinearTiles<1>>,
        mut gate: DisjointSlice<f32, LinearTiles<1>>,
    ) {
        let i = thread::index_1d_u32(launch_context).get() as usize;
        let s = i + (i / HEAD_DIM) * HEAD_DIM;
        let (Some(&qv), Some(&gv)) = (src.get(s), src.get(s + HEAD_DIM)) else {
            return;
        };
        let Some(ThreadRunMut32::Full(mut qo)) = q.thread_run32(thread::index_1d_u32(launch_context)) else {
            return;
        };
        let Some(ThreadRunMut32::Full(mut go)) = gate.thread_run32(thread::index_1d_u32(launch_context)) else {
            return;
        };
        qo.at_const::<0>().write(qv);
        go.at_const::<0>().write(gv);
    }
}

/// `forward_attention`'s split: two `copy_from_slice`s per head.
fn cpu_split_interleave(src: &[f32], heads: usize) -> (Vec<f32>, Vec<f32>) {
    let mut q = vec![0.0f32; heads * HEAD_DIM];
    let mut gate = vec![0.0f32; heads * HEAD_DIM];
    for h in 0..heads {
        let base = h * HEAD_DIM * 2;
        q[h * HEAD_DIM..(h + 1) * HEAD_DIM].copy_from_slice(&src[base..base + HEAD_DIM]);
        gate[h * HEAD_DIM..(h + 1) * HEAD_DIM]
            .copy_from_slice(&src[base + HEAD_DIM..base + 2 * HEAD_DIM]);
    }
    (q, gate)
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

fn bit_exact(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Source values, with signed zeros and a NaN payload planted so a copy that goes
/// through float arithmetic (or canonicalises) cannot pass bit-exact.
fn make_inputs(heads: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let mut v: Vec<f32> = (0..heads * 2 * HEAD_DIM)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect();
    v[0] = -0.0;
    if v.len() > HEAD_DIM + 3 {
        v[HEAD_DIM + 3] = f32::from_bits(0x7fc0_1234);
    }
    v
}

fn seed(heads: usize) -> u64 {
    0x3522_000E + (heads as u64) * 65_536
}

struct Measured {
    cos: f64,
    maxdiff: f32,
    bit_exact: bool,
    /// Device time per launch, from a CUDA-graph replay (the gated number).
    us: f64,
    /// Eager per-launch time: recorded, never gated (see GDN-DECISIONS.md).
    eager_us: f64,
}

fn measure(q: &[f32], gate: &[f32], src: &[f32], heads: usize, us: f64, eager_us: f64) -> Measured {
    let (wq, wg) = cpu_split_interleave(src, heads);
    // NaN-free views for cos / max|Δ|: the planted NaN is judged by bit_exact alone.
    let finite = |v: &[f32]| -> Vec<f32> { v.iter().map(|x| if x.is_nan() { 0.0 } else { *x }).collect() };
    let got: Vec<f32> = finite(q).into_iter().chain(finite(gate)).collect();
    let want: Vec<f32> = finite(&wq).into_iter().chain(finite(&wg)).collect();
    Measured {
        cos: cosine(&got, &want),
        maxdiff: max_abs_diff(&got, &want),
        bit_exact: bit_exact(q, &wq) && bit_exact(gate, &wg),
        us,
        eager_us,
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


fn run_oxide(ctx: &Arc<CudaContext>, module: &kernels::LoadedModule, heads: usize, perf: bool) -> Measured {
    let stream = ctx.new_stream().expect("stream");
    let src = make_inputs(heads, seed(heads));
    let n = heads * HEAD_DIM;
    let d_src = DeviceBuffer::from_host(&stream, &src).expect("src");
    let mut d_q = DeviceBuffer::<f32>::zeroed(&stream, n).expect("q");
    let mut d_gate = DeviceBuffer::<f32>::zeroed(&stream, n).expect("gate");

    let config = LaunchConfig1D::new(heads as u32, BLOCK, 0);
    let launch = |d_q: &mut DeviceBuffer<f32>, d_gate: &mut DeviceBuffer<f32>| {
        let p = module.prepare_split_interleave(config).expect("prepare");
        module
            .split_interleave(&stream, &p, &d_src, d_q, d_gate)
            .expect("launch");
    };
    launch(&mut d_q, &mut d_gate);
    let q = d_q.to_host_vec(&stream).expect("download q");
    let gate = d_gate.to_host_vec(&stream).expect("download gate");
    let (us, eager_us) = if perf {
        (
            time_graph_us(&stream, || launch(&mut d_q, &mut d_gate)),
            time_eager_us(&stream, || launch(&mut d_q, &mut d_gate)),
        )
    } else {
        (0.0, 0.0)
    };
    measure(&q, &gate, &src, heads, us, eager_us)
}

/// The hand PTX, loaded from its committed baseline and launched on the same data:
/// grid (heads), block 256, params (src, q, gate).
fn run_handptx(ctx: &Arc<CudaContext>, sm: &str, heads: usize) -> (Measured, u32) {
    let ptx_path = format!("baseline-ptx/gdn_split_interleaved_q_gate.{sm}.ptx");
    let ptx = std::fs::read_to_string(&ptx_path).unwrap_or_else(|e| {
        eprintln!("missing hand-PTX baseline {ptx_path}: {e}");
        eprintln!("regenerate: APR_BLESS_PTX=1 cargo test -p aprender-gpu --lib gdn_split_interleave_ptx_golden");
        std::process::exit(2);
    });
    let stream = ctx.new_stream().expect("stream");
    let src = make_inputs(heads, seed(heads));
    let n = heads * HEAD_DIM;
    let module = ctx.load_module_from_ptx_src(&ptx).expect("load hand PTX");
    let func = module
        .load_function("gdn_split_interleaved_q_gate")
        .expect("gdn_split_interleaved_q_gate");
    let regs = func.num_registers().expect("regs");
    let d_src = DeviceBuffer::from_host(&stream, &src).expect("src");
    let d_q = DeviceBuffer::<f32>::zeroed(&stream, n).expect("q");
    let d_gate = DeviceBuffer::<f32>::zeroed(&stream, n).expect("gate");
    let mut src_ptr = d_src.cu_deviceptr();
    let mut q_ptr = d_q.cu_deviceptr();
    let mut gate_ptr = d_gate.cu_deviceptr();
    let mut launch = || {
        let mut params: [*mut std::ffi::c_void; 3] = [
            (&raw mut src_ptr).cast(),
            (&raw mut q_ptr).cast(),
            (&raw mut gate_ptr).cast(),
        ];
        // SAFETY: the three params match the entry's (.u64, .u64, .u64) in order;
        // src holds heads * 2 * HEAD_DIM f32 and q, gate heads * HEAD_DIM each, at
        // the head width this PTX was baked with, and a heads-block launch touches
        // only those.
        unsafe {
            cuda_core::launch_kernel_on_stream(&func, (heads as u32, 1, 1), (BLOCK, 1, 1), 0, &stream, &mut params)
        }
        .expect("hand PTX launch");
    };
    launch();
    let q = d_q.to_host_vec(&stream).expect("download q");
    let gate = d_gate.to_host_vec(&stream).expect("download gate");
    let us = time_graph_us(&stream, &mut launch);
    let eager_us = time_eager_us(&stream, &mut launch);
    (measure(&q, &gate, &src, heads, us, eager_us), regs)
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

    println!("== #3522 cuda-oxide q|gate de-interleave ({sm}) ==");
    println!("   head_dim={HEAD_DIM} 1 thread/(head, element), bit-exact gate");

    let mut all_ok = true;
    let mut timing_all_ok = true;
    for heads in PARITY_HEADS {
        let r = run_oxide(&ctx, &module, heads, false);
        all_ok &= r.bit_exact;
        println!(
            "  parity oxide heads={heads}: cos={:.7} maxdiff={:.3e} {}",
            r.cos,
            r.maxdiff,
            if r.bit_exact { "BIT-EXACT PASS" } else { "FAIL" }
        );
    }
    let mut regs_hand = 0;
    for heads in PARITY_HEADS {
        let (h, regs) = run_handptx(&ctx, &sm, heads);
        regs_hand = regs;
        all_ok &= h.bit_exact;
        println!(
            "  parity handPTX heads={heads}: {} regs={regs}",
            if h.bit_exact { "BIT-EXACT PASS" } else { "FAIL" }
        );
    }

    println!("\n  heads | oxide us | handPTX us | ratio | verdict | eager oxide/hand");
    let entry = "split_interleave";
    let regs = oxide_registers(&ctx, entry);
    let mut worst_ratio = 0.0f64;
    let mut worst = (0.0f64, 0.0f64, 0usize);
    let mut worst_eager = 0.0f64;
    let mut parity = (1.0f64, 0.0f32, true);
    for n in TIMED_HEADS {
        // Three rounds, alternating which kernel is timed first; the round with the
        // median ratio is the row (see GDN-DECISIONS.md, causal conv1d seq).
        let mut rounds: Vec<(Measured, Measured)> = (0..3)
            .map(|round| {
                if round % 2 == 0 {
                    let o = run_oxide(&ctx, &module, n, true);
                    (o, run_handptx(&ctx, &sm, n).0)
                } else {
                    let h = run_handptx(&ctx, &sm, n).0;
                    (run_oxide(&ctx, &module, n, true), h)
                }
            })
            .collect();
        rounds.sort_by(|a, b| (a.0.us / a.1.us).total_cmp(&(b.0.us / b.1.us)));
        let (o, h) = rounds.swap_remove(1);
        let ratio = o.us / h.us;
        let ok = ratio <= TIMING_RATIO_MAX;
        let eager_ratio = o.eager_us / h.eager_us;
        println!(
            "  {n:>5} | {:>8.3} | {:>10.3} | {ratio:.3} | {:>7} | {:.3}/{:.3} = {eager_ratio:.3}",
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
        parity = (parity.0.min(o.cos), parity.1.max(o.maxdiff), parity.2 && o.bit_exact);
    }
    let timing_ok = worst_ratio <= TIMING_RATIO_MAX;
    timing_all_ok &= timing_ok;
    let parity_ok = parity.2;
    println!(
        "RECEIPT {{\"schema\":\"apr-kernel-receipt/v1\",\"kernel\":\"gdn_split_interleave\",\"variant\":\"copy\",\"entry\":\"{entry}\",\"authoring\":\"oxide\",\"cc\":\"{sm}\",\"head_dim\":{HEAD_DIM},\"parity\":{{\"cos_min\":{:.9},\"maxdiff_max\":{:.3e},\"bit_exact\":{},\"pass\":{parity_ok}}},\"timing\":{{\"oxide_us\":{:.3},\"handptx_us\":{:.3},\"ratio\":{worst_ratio:.4},\"worst_heads\":{},\"ratio_max\":{TIMING_RATIO_MAX},\"method\":\"cuda-graph-100\",\"eager_ratio_max\":{worst_eager:.4},\"pass\":{timing_ok}}},\"register_budget\":{{\"oxide\":{},\"handptx\":{regs_hand}}}}}",
        parity.0,
        parity.1,
        parity.2,
        worst.0,
        worst.1,
        worst.2,
        regs.map_or("null".to_string(), |r| r.to_string()),
    );

    if !all_ok {
        eprintln!("#3522 PARITY FAILED");
        std::process::exit(1);
    }
    if !timing_all_ok {
        eprintln!("#3522 TIMING NO-GO (oxide/hand > {TIMING_RATIO_MAX})");
        std::process::exit(4);
    }
    println!("#3522 SPLIT INTERLEAVE DONE");
}
