// #3522 OXIDE-001 — pure-Rust cuda-oxide port of the GDN causal conv1d + SiLU.
//
// Target = hand-PTX `CausalConv1dSiluKernel`
// (crates/aprender-gpu/src/kernels/gdn/causal_conv1d.rs, entry `gdn_causal_conv1d_silu`):
// one thread per channel, grid ceil(channels/256), block 256, channels and K baked
// into the PTX, params (input, state, weight, output), state updated in place.
//
//   sum      = sum_{k<K-1} state[c][k] * w[c][k] + input[c] * w[c][K-1]
//   state[c] = state[c][1..] ++ [input[c]]
//   out[c]   = sum / (1 + exp(-sum))
//
// CPU reference = `causal_conv1d` + the SiLU loop of `forward_deltanet` in
// crates/aprender-serve/src/gguf/inference/forward/forward_qwen35.rs, in f64.
//
// SAFETY SHAPE (kernel-safety, #3522): both device kernels are safe Rust — no
// `unsafe` block, no raw pointer. Reads go through bounds-checked slice indexing;
// the writes go through two `DisjointSlice`s: `out` with `LinearTiles<1>` (one
// element per thread) and `state_out` with `LinearTiles<3>` (the thread's K-1
// window). Each slice is claimed with its own launch-checked thread index.
//
// OUT OF PLACE, for the reason l2-norm gives: a safe kernel cannot read `state` as
// `&[f32]` and write the same memory through a `DisjointSlice`, so the port takes
// (input, state, weight, out, state_out). Both authorings move the same bytes per
// channel: 1 + 3 + 4 loads and 1 + 3 stores.
//
// Two variants, differing only in the exponential of the SiLU:
//   (A) conv1d_silu_exp — `exp(-sum)` (libdevice), the more precise
//   (B) conv1d_silu_ex2 — `exp2(-sum * log2 e)`, mirroring the hand PTX's ex2.approx
// Both keep the CPU's separate multiply and add (no FMA contraction), as the hand
// PTX does, and divide with `/` (`div.rn.f32`).
//
// Output: a human table on stdout plus one `apr-kernel-receipt/v1` JSON line
// prefixed `RECEIPT ` per variant; `receipt.sh` adds host facts and writes
// evidence/kernels/gdn_causal_conv1d_silu/<host>.json.

use cuda_core::{CudaContext, DeviceBuffer, IntoResult, LaunchConfig1D, sys};
use cuda_device::{DisjointSlice, LinearTiles, cuda_module, kernel, launch_bounds, launch_contract, thread};
use std::sync::Arc;

const BLOCK: usize = 256;
/// Conv window (Qwen3.5's `conv_kernel`); the hand PTX bakes it in.
const K: usize = 4;
const S: usize = K - 1;

// F-OXIDE-ROPE-PARITY-001 form, as #3522's kernel-parity shape states it.
const PARITY_COS: f64 = 0.9999;
const PARITY_MAXDIFF: f32 = 1e-3;
// kernel-timing shape: oxide_us / handptx_us <= 1.2.
const TIMING_RATIO_MAX: f64 = 1.2;

/// Parity widths. 1000 is not a multiple of the block, so the tail threads of
/// the last block must write nothing and read nothing.
const PARITY_NS: [usize; 4] = [1, 1000, 2048, 6144];
/// Timed widths, each with a committed hand-PTX baseline (channels are baked in).
/// 6144 is Qwen3.5-0.8B's `conv_dim` (the device test's shape).
const TIMED_NS: [usize; 3] = [2048, 4096, 6144];

#[cuda_module]
mod kernels {
    use super::*;

    /// (A) SiLU via `exp`.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub fn conv1d_silu_exp(
        input: &[f32],
        state: &[f32],
        weight: &[f32],
        mut out: DisjointSlice<f32, LinearTiles<1>>,
        mut state_out: DisjointSlice<f32, LinearTiles<3>>,
    ) {
        let c = thread::index_1d_u32(launch_context).get() as usize;
        let Some(mut o) = out.thread_run32(thread::index_1d_u32(launch_context)) else {
            return;
        };
        let Some(mut so) = state_out.thread_run32(thread::index_1d_u32(launch_context)) else {
            return;
        };
        let x = input[c];
        let s0 = state[c * 3];
        let s1 = state[c * 3 + 1];
        let s2 = state[c * 3 + 2];
        let mut sum = 0.0f32;
        sum += s0 * weight[c * 4];
        sum += s1 * weight[c * 4 + 1];
        sum += s2 * weight[c * 4 + 2];
        sum += x * weight[c * 4 + 3];
        if let Some(mut slot) = so.at(0) {
            slot.write(s1);
        }
        if let Some(mut slot) = so.at(1) {
            slot.write(s2);
        }
        if let Some(mut slot) = so.at(2) {
            slot.write(x);
        }
        if let Some(mut slot) = o.at(0) {
            slot.write(sum / (1.0f32 + (-sum).exp()));
        }
    }

    /// (B) SiLU via `exp2(-sum * log2 e)`, the hand PTX's form.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub fn conv1d_silu_ex2(
        input: &[f32],
        state: &[f32],
        weight: &[f32],
        mut out: DisjointSlice<f32, LinearTiles<1>>,
        mut state_out: DisjointSlice<f32, LinearTiles<3>>,
    ) {
        let c = thread::index_1d_u32(launch_context).get() as usize;
        let Some(mut o) = out.thread_run32(thread::index_1d_u32(launch_context)) else {
            return;
        };
        let Some(mut so) = state_out.thread_run32(thread::index_1d_u32(launch_context)) else {
            return;
        };
        let x = input[c];
        let s0 = state[c * 3];
        let s1 = state[c * 3 + 1];
        let s2 = state[c * 3 + 2];
        let mut sum = 0.0f32;
        sum += s0 * weight[c * 4];
        sum += s1 * weight[c * 4 + 1];
        sum += s2 * weight[c * 4 + 2];
        sum += x * weight[c * 4 + 3];
        if let Some(mut slot) = so.at(0) {
            slot.write(s1);
        }
        if let Some(mut slot) = so.at(1) {
            slot.write(s2);
        }
        if let Some(mut slot) = so.at(2) {
            slot.write(x);
        }
        if let Some(mut slot) = o.at(0) {
            let e = ((-sum) * std::f32::consts::LOG2_E).exp2();
            slot.write(sum / (1.0f32 + e));
        }
    }
}

/// f64 evaluation of `causal_conv1d` + SiLU (forward_qwen35.rs). Returns
/// (output, new state).
fn cpu_conv1d_silu(input: &[f32], state: &[f32], weight: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = input.len();
    let mut out = vec![0.0f32; n];
    let mut st = state.to_vec();
    for c in 0..n {
        let mut sum = 0.0f64;
        for k in 0..S {
            sum += f64::from(state[c * S + k]) * f64::from(weight[c * K + k]);
        }
        sum += f64::from(input[c]) * f64::from(weight[c * K + S]);
        for k in 0..S - 1 {
            st[c * S + k] = state[c * S + k + 1];
        }
        st[c * S + S - 1] = input[c];
        out[c] = (sum / (1.0 + (-sum).exp())) as f32;
    }
    (out, st)
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

/// Deterministic inputs, scaled as the aprender-gpu device test scales them
/// (weight 0.5, state 1.0, input 1.0).
fn make_inputs(n: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut s = seed;
    let mut next = move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let input: Vec<f32> = (0..n).map(|_| next()).collect();
    let state: Vec<f32> = (0..n * S).map(|_| next()).collect();
    let weight: Vec<f32> = (0..n * K).map(|_| next() * 0.5).collect();
    (input, state, weight)
}

struct Measured {
    /// Cosine of the SiLU output against the f64 reference.
    cos: f64,
    /// Worst |Δ| over the output and the updated state.
    maxdiff: f32,
    /// Device time per launch, from a CUDA-graph replay (the gated number).
    us: f64,
    /// Eager per-launch time: recorded, never gated (see GDN-DECISIONS.md).
    eager_us: f64,
}

impl Measured {
    fn ok(&self) -> bool {
        self.cos >= PARITY_COS && self.maxdiff < PARITY_MAXDIFF
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

fn measure(got: &[f32], got_state: &[f32], inp: &(Vec<f32>, Vec<f32>, Vec<f32>), us: f64, eager_us: f64) -> Measured {
    let (want, want_state) = cpu_conv1d_silu(&inp.0, &inp.1, &inp.2);
    Measured {
        cos: cosine(got, &want),
        maxdiff: max_abs_diff(got, &want).max(max_abs_diff(got_state, &want_state)),
        us,
        eager_us,
    }
}

fn seed(n: usize) -> u64 {
    0x3522_0006 + n as u64
}

fn run_oxide(
    ctx: &Arc<CudaContext>,
    module: &kernels::LoadedModule,
    n: usize,
    ex2: bool,
    perf: bool,
) -> Measured {
    let stream = ctx.new_stream().expect("stream");
    let inp = make_inputs(n, seed(n));
    let d_in = DeviceBuffer::from_host(&stream, &inp.0).expect("input");
    let d_state = DeviceBuffer::from_host(&stream, &inp.1).expect("state");
    let d_w = DeviceBuffer::from_host(&stream, &inp.2).expect("weight");
    let mut d_out = DeviceBuffer::<f32>::zeroed(&stream, n).expect("out");
    let mut d_state_out = DeviceBuffer::<f32>::zeroed(&stream, n * S).expect("state_out");

    let config = LaunchConfig1D::new(n.div_ceil(BLOCK) as u32, BLOCK as u32, 0);
    let launch = |d_out: &mut DeviceBuffer<f32>, d_so: &mut DeviceBuffer<f32>| {
        if ex2 {
            let p = module.prepare_conv1d_silu_ex2(config).expect("prepare ex2");
            module
                .conv1d_silu_ex2(&stream, &p, &d_in, &d_state, &d_w, d_out, d_so)
                .expect("launch ex2");
        } else {
            let p = module.prepare_conv1d_silu_exp(config).expect("prepare exp");
            module
                .conv1d_silu_exp(&stream, &p, &d_in, &d_state, &d_w, d_out, d_so)
                .expect("launch exp");
        }
    };
    launch(&mut d_out, &mut d_state_out);
    let got = d_out.to_host_vec(&stream).expect("download");
    let got_state = d_state_out.to_host_vec(&stream).expect("download state");
    let (us, eager_us) = if perf {
        (
            time_graph_us(&stream, || launch(&mut d_out, &mut d_state_out)),
            time_eager_us(&stream, || launch(&mut d_out, &mut d_state_out)),
        )
    } else {
        (0.0, 0.0)
    };
    measure(&got, &got_state, &inp, us, eager_us)
}

/// The hand PTX for width `n`, loaded from its committed baseline and launched on
/// the same data: grid ceil(n/256), block 256, params (input, state, weight,
/// output), state in place.
fn run_handptx(ctx: &Arc<CudaContext>, sm: &str, n: usize) -> (Measured, u32) {
    let ptx_path = format!("baseline-ptx/gdn_causal_conv1d_silu_c{n}.{sm}.ptx");
    let ptx = std::fs::read_to_string(&ptx_path).unwrap_or_else(|e| {
        eprintln!("missing hand-PTX baseline {ptx_path}: {e}");
        eprintln!("regenerate: APR_BLESS_PTX=1 cargo test -p aprender-gpu --features cuda --lib gdn_conv1d_ptx_golden");
        std::process::exit(2);
    });
    let stream = ctx.new_stream().expect("stream");
    let inp = make_inputs(n, seed(n));
    let module = ctx.load_module_from_ptx_src(&ptx).expect("load hand PTX");
    let func = module
        .load_function("gdn_causal_conv1d_silu")
        .expect("gdn_causal_conv1d_silu");
    let regs = func.num_registers().expect("regs");
    let d_in = DeviceBuffer::from_host(&stream, &inp.0).expect("input");
    let d_state = DeviceBuffer::from_host(&stream, &inp.1).expect("state");
    let d_w = DeviceBuffer::from_host(&stream, &inp.2).expect("weight");
    let d_out = DeviceBuffer::<f32>::zeroed(&stream, n).expect("out");
    let mut ptrs = [
        d_in.cu_deviceptr(),
        d_state.cu_deviceptr(),
        d_w.cu_deviceptr(),
        d_out.cu_deviceptr(),
    ];
    let grid = n.div_ceil(BLOCK) as u32;
    let mut launch = || {
        let mut params: Vec<*mut std::ffi::c_void> =
            ptrs.iter_mut().map(|p| (p as *mut u64).cast()).collect();
        // SAFETY: four device pointers matching the entry's four .u64 params;
        // input/output hold n f32, state n*3, weight n*4, and the baked-in bound
        // stops every thread at n.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &func,
                (grid, 1, 1),
                (BLOCK as u32, 1, 1),
                0,
                &stream,
                &mut params,
            )
        }
        .expect("hand PTX launch");
    };
    // Parity from the first launch. Timing then keeps shifting the window in
    // place, which does the same loads, arithmetic and stores on shifted values.
    launch();
    let got = d_out.to_host_vec(&stream).expect("download");
    let got_state = d_state.to_host_vec(&stream).expect("download state");
    let us = time_graph_us(&stream, &mut launch);
    let eager_us = time_eager_us(&stream, &mut launch);
    (measure(&got, &got_state, &inp, us, eager_us), regs)
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

    println!("== #3522 cuda-oxide causal conv1d + SiLU ({sm}) ==");
    println!("   block={BLOCK}, 1 channel/thread, K={K}");

    let mut all_ok = true;
    // A timing NO-GO is its own exit code (4), not a pass: the receipts still
    // print, so receipt.sh records `"pass":false` and then fails.
    let mut timing_all_ok = true;
    for ex2 in [false, true] {
        let name = if ex2 { "ex2 (B)" } else { "exp (A)" };
        for n in PARITY_NS {
            let r = run_oxide(&ctx, &module, n, ex2, false);
            let ok = r.ok();
            all_ok &= ok;
            println!(
                "  parity {name} n={n:>5}: cos={:.7} maxdiff={:.3e} {}",
                r.cos,
                r.maxdiff,
                if ok { "PASS" } else { "FAIL" }
            );
        }
    }

    let regs_hand;
    {
        let (h, r) = run_handptx(&ctx, &sm, TIMED_NS[0]);
        regs_hand = r;
        let ok = h.ok();
        println!(
            "  hand PTX c={}: cos={:.7} maxdiff={:.3e} regs={r} {}",
            TIMED_NS[0],
            h.cos,
            h.maxdiff,
            if ok { "PASS" } else { "FAIL" }
        );
        all_ok &= ok;
    }
    println!("\n      n | variant | oxide us | handPTX us | ratio | verdict | eager oxide/hand");
    for ex2 in [false, true] {
        let (variant, entry) = if ex2 {
            ("ex2", "conv1d_silu_ex2")
        } else {
            ("exp", "conv1d_silu_exp")
        };
        let regs = oxide_registers(&ctx, entry);
        let mut worst_ratio = 0.0f64;
        let mut worst = (0.0f64, 0.0f64, 0usize);
        let mut worst_eager = 0.0f64;
        let mut parity = (1.0f64, 0.0f32);
        for n in TIMED_NS {
            let o = run_oxide(&ctx, &module, n, ex2, true);
            let (h, _) = run_handptx(&ctx, &sm, n);
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
            parity = (parity.0.min(o.cos), parity.1.max(o.maxdiff));
        }
        let timing_ok = worst_ratio <= TIMING_RATIO_MAX;
        timing_all_ok &= timing_ok;
        let parity_ok = parity.0 >= PARITY_COS && parity.1 < PARITY_MAXDIFF;
        // One line per variant; receipt.sh adds host, sha, ptxas and writes the file.
        println!(
            "RECEIPT {{\"schema\":\"apr-kernel-receipt/v1\",\"kernel\":\"gdn_causal_conv1d_silu\",\"variant\":\"{variant}\",\"entry\":\"{entry}\",\"authoring\":\"oxide\",\"cc\":\"{sm}\",\"block\":{BLOCK},\"conv_kernel\":{K},\"parity\":{{\"cos_min\":{:.9},\"maxdiff_max\":{:.3e},\"cos_floor\":{PARITY_COS},\"maxdiff_ceiling\":{PARITY_MAXDIFF:e},\"pass\":{parity_ok}}},\"timing\":{{\"oxide_us\":{:.3},\"handptx_us\":{:.3},\"ratio\":{worst_ratio:.4},\"worst_n\":{},\"ratio_max\":{TIMING_RATIO_MAX},\"method\":\"cuda-graph-100\",\"eager_ratio_max\":{worst_eager:.4},\"pass\":{timing_ok}}},\"register_budget\":{{\"oxide\":{},\"handptx\":{regs_hand}}}}}",
            parity.0,
            parity.1,
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
    println!("#3522 CAUSAL CONV1D DONE");
}
