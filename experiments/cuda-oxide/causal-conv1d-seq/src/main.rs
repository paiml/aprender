// #3522 OXIDE-001 — pure-Rust cuda-oxide port of the GDN causal conv1d + SiLU over a
// prefill chunk.
//
// Target = hand-PTX `CausalConv1dSiluSeqKernel`
// (crates/aprender-gpu/src/kernels/gdn/causal_conv1d_seq.rs, entry
// `gdn_causal_conv1d_silu_seq`): one thread per channel walks `t_count` rows in order,
// grid ceil(channels/256), block 256. `channels`, K and both row strides are baked into
// the PTX; `t_count` is a runtime param. The window is read once, kept in registers,
// and written back once:
//
//   for t in 0..T:
//       sum          = sum_{k<K-1} window[k] * w[c][k] + input[t][c] * w[c][K-1]
//       window       = window[1..] ++ [input[t][c]]
//       output[t][c] = sum / (1 + exp(-sum))
//   state[c] = window
//
// CPU reference = `causal_conv1d` + SiLU of `forward_deltanet`, stepped T times, in f64.
//
// SAFETY SHAPE (kernel-safety, #3522): both device kernels are safe Rust. The launch is
// 2-D (`domain = 2`, grid y = 1) because the safe strided-column write needs a 2-D
// tile: thread (row 0, col c) claims
//   * `out` as `RuntimeRowMajorTiles<T_TILE, 1>` — rows 0..T_TILE of column c, the row
//     width (the output row stride) bound on the host by `RowWidth`;
//   * `state_out` as `RuntimeRowMajorTiles<1, 3>` with row width 3·channels — the
//     thread's K-1 window.
// The input column is a `MatrixView32::col` (one checked band, stride = input row
// stride). The weight and state are one checked `first_chunk` each, so the interior of
// the loop carries no bounds branch.
//
// PORT CONSTRAINT: a safe tile's row count is a type parameter, so the output tile is
// T_TILE rows whatever `t_count` is. The kernel still takes `t_count` at runtime and
// walks min(t_count, T_TILE) rows, but the output buffer must hold T_TILE rows. The hand
// PTX has no such floor. A shipping port needs one monomorphisation per chunk size, or a
// clipped 2-D run type that cuda-oxide does not have at this rev.
//
// OUT OF PLACE for the state, as for the per-token port: (input, state, weight) are
// read, (out, state_out) written.
//
// FMA: LLVM contracts each `sum += a * b` into `fma.rn.f32`; the hand PTX emits a
// separate mul and add. Both are within 1e-7 of the f64 reference, and the per-token
// port does the same (its PTX has 13 `fma`s).
//
// Two variants, differing only in the exponential of the SiLU:
//   (A) conv1d_seq_exp — `exp(-sum)` (libdevice)
//   (B) conv1d_seq_ex2 — `exp2(-sum * log2 e)`, the hand PTX's ex2.approx form
//
// Output: a human table on stdout plus one `apr-kernel-receipt/v1` JSON line prefixed
// `RECEIPT ` per variant; `receipt.sh` adds host facts and writes
// evidence/kernels/gdn_causal_conv1d_silu_seq/<host>.json.

use cuda_core::{CudaContext, DeviceBuffer, IntoResult, LaunchConfig2D, sys};
use cuda_device::{
    DisjointSlice, LocalIndex32, MatrixView32, RuntimeRowMajorTiles, cuda_module, kernel,
    launch_bounds, launch_contract, thread,
};
use cuda_host::RowWidth;
use std::sync::Arc;

const BLOCK: usize = 256;
/// Conv window (Qwen3.5's `conv_kernel`); the hand PTX bakes it in.
const K: usize = 4;
const S: usize = K - 1;
/// Output tile rows: the chunk length the port is monomorphised for.
const T_TILE: usize = 64;

const PARITY_COS: f64 = 0.9999;
const PARITY_MAXDIFF: f32 = 1e-3;
const TIMING_RATIO_MAX: f64 = 1.2;

/// Parity cases (channels, row stride, t_count). 1000 is not a multiple of the block;
/// stride 1024 > 1000 leaves padding columns no thread may write; t_count 23 and 1
/// leave output rows no thread may write.
const PARITY_CASES: [(usize, usize, usize); 6] = [
    (1, 1, 64),
    (1000, 1000, 23),
    (1000, 1024, 64),
    (2048, 2048, 64),
    (6144, 6144, 1),
    (6144, 6144, 64),
];
/// Timed widths, each with a committed hand-PTX baseline; strides = channels and
/// t_count = T_TILE.
const TIMED_NS: [usize; 3] = [2048, 4096, 6144];

#[cuda_module]
mod kernels {
    use super::*;

    /// (A) SiLU via `exp`.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 2, coordinates = u32, block = (256, 1, 1))]
    pub fn conv1d_seq_exp(
        input: &[f32],
        in_stride: u32,
        t_count: u32,
        state: &[f32],
        weight: &[f32],
        mut out: DisjointSlice<f32, RuntimeRowMajorTiles<T_TILE, 1>>,
        mut state_out: DisjointSlice<f32, RuntimeRowMajorTiles<1, 3>>,
    ) {
        let c = thread::coord_2d_u32(launch_context).col();
        let ci = c as usize;
        let rows = if t_count < T_TILE as u32 {
            t_count
        } else {
            T_TILE as u32
        };
        // Every read is proved before any write: the state/weight chunks bound c to
        // the channel count, so a padding column (c >= channels, c < stride) returns
        // here.
        let (Some(st), Some(w), Some(col)) = (
            state.get(ci * 3..).and_then(|s| s.first_chunk::<3>()),
            weight.get(ci * 4..).and_then(|s| s.first_chunk::<4>()),
            MatrixView32::new(input, in_stride).col(c, rows),
        ) else {
            return;
        };
        let Some(mut o) = out.tile_2d32_rt(thread::coord_2d_u32(launch_context)) else {
            return;
        };
        let Some(mut so) = state_out.tile_2d32_rt(thread::coord_2d_u32(launch_context)) else {
            return;
        };
        let (mut w0, mut w1, mut w2) = (st[0], st[1], st[2]);
        // The weights into registers once. Read through `w[i]` in the loop, they are
        // reloaded every step: nothing tells LLVM the output stores cannot alias them.
        let (k0, k1, k2, k3) = (w[0], w[1], w[2], w[3]);
        let mut t = 0u32;
        for x in col.iter() {
            let Some(row) = LocalIndex32::<T_TILE>::new(t) else {
                break;
            };
            let mut sum = 0.0f32;
            sum += w0 * k0;
            sum += w1 * k1;
            sum += w2 * k2;
            sum += x * k3;
            w0 = w1;
            w1 = w2;
            w2 = x;
            let e = (-sum).exp();
            o.at(row, LocalIndex32::constant::<0>())
                .write(sum / (1.0f32 + e));
            t += 1;
        }
        so.at_const::<0, 0>().write(w0);
        so.at_const::<0, 1>().write(w1);
        so.at_const::<0, 2>().write(w2);
    }

    /// (B) SiLU via `exp2(-sum * log2 e)`, the hand PTX's form.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 2, coordinates = u32, block = (256, 1, 1))]
    pub fn conv1d_seq_ex2(
        input: &[f32],
        in_stride: u32,
        t_count: u32,
        state: &[f32],
        weight: &[f32],
        mut out: DisjointSlice<f32, RuntimeRowMajorTiles<T_TILE, 1>>,
        mut state_out: DisjointSlice<f32, RuntimeRowMajorTiles<1, 3>>,
    ) {
        let c = thread::coord_2d_u32(launch_context).col();
        let ci = c as usize;
        let rows = if t_count < T_TILE as u32 {
            t_count
        } else {
            T_TILE as u32
        };
        // Every read is proved before any write: the state/weight chunks bound c to
        // the channel count, so a padding column (c >= channels, c < stride) returns
        // here.
        let (Some(st), Some(w), Some(col)) = (
            state.get(ci * 3..).and_then(|s| s.first_chunk::<3>()),
            weight.get(ci * 4..).and_then(|s| s.first_chunk::<4>()),
            MatrixView32::new(input, in_stride).col(c, rows),
        ) else {
            return;
        };
        let Some(mut o) = out.tile_2d32_rt(thread::coord_2d_u32(launch_context)) else {
            return;
        };
        let Some(mut so) = state_out.tile_2d32_rt(thread::coord_2d_u32(launch_context)) else {
            return;
        };
        let (mut w0, mut w1, mut w2) = (st[0], st[1], st[2]);
        // The weights into registers once. Read through `w[i]` in the loop, they are
        // reloaded every step: nothing tells LLVM the output stores cannot alias them.
        let (k0, k1, k2, k3) = (w[0], w[1], w[2], w[3]);
        let mut t = 0u32;
        for x in col.iter() {
            let Some(row) = LocalIndex32::<T_TILE>::new(t) else {
                break;
            };
            let mut sum = 0.0f32;
            sum += w0 * k0;
            sum += w1 * k1;
            sum += w2 * k2;
            sum += x * k3;
            w0 = w1;
            w1 = w2;
            w2 = x;
            let e = ((-sum) * std::f32::consts::LOG2_E).exp2();
            o.at(row, LocalIndex32::constant::<0>())
                .write(sum / (1.0f32 + e));
            t += 1;
        }
        so.at_const::<0, 0>().write(w0);
        so.at_const::<0, 1>().write(w1);
        so.at_const::<0, 2>().write(w2);
    }
}

/// f64 evaluation of `causal_conv1d` + SiLU stepped over `t_count` rows. Returns
/// (output [T_TILE][stride], new state); rows >= t_count and padding columns stay 0.
fn cpu_conv1d_seq(case: (usize, usize, usize), inp: &Inputs) -> (Vec<f32>, Vec<f32>) {
    let (channels, stride, t_count) = case;
    let mut out = vec![0.0f32; T_TILE * stride];
    let mut st = inp.state.clone();
    for c in 0..channels {
        let win = &mut st[c * S..c * S + S];
        let w = &inp.weight[c * K..c * K + K];
        for t in 0..t_count.min(T_TILE) {
            let x = inp.input[t * stride + c];
            let mut sum = 0.0f64;
            for k in 0..S {
                sum += f64::from(win[k]) * f64::from(w[k]);
            }
            sum += f64::from(x) * f64::from(w[S]);
            win.rotate_left(1);
            win[S - 1] = x;
            out[t * stride + c] = (sum / (1.0 + (-sum).exp())) as f32;
        }
    }
    (out, st)
}

struct Inputs {
    input: Vec<f32>,
    state: Vec<f32>,
    weight: Vec<f32>,
}

/// Deterministic inputs, scaled as the aprender-gpu device test scales them
/// (weight 0.5, state 1.0, input 1.0). The input holds T_TILE rows of `stride`.
fn make_inputs(channels: usize, stride: usize, seed: u64) -> Inputs {
    let mut s = seed;
    let mut next = move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let input: Vec<f32> = (0..T_TILE * stride).map(|_| next()).collect();
    let state: Vec<f32> = (0..channels * S).map(|_| next()).collect();
    let weight: Vec<f32> = (0..channels * K).map(|_| next() * 0.5).collect();
    Inputs {
        input,
        state,
        weight,
    }
}

fn measure(
    got: &[f32],
    got_state: &[f32],
    case: (usize, usize, usize),
    inp: &Inputs,
    us: f64,
    eager_us: f64,
) -> Measured {
    let (want, want_state) = cpu_conv1d_seq(case, inp);
    Measured {
        cos: cosine(got, &want),
        maxdiff: max_abs_diff(got, &want).max(max_abs_diff(got_state, &want_state)),
        us,
        eager_us,
    }
}

fn seed(channels: usize, stride: usize, t_count: usize) -> u64 {
    0x3522_0007 + (channels * 131 + stride * 7 + t_count) as u64
}

fn run_oxide(
    ctx: &Arc<CudaContext>,
    module: &kernels::LoadedModule,
    case: (usize, usize, usize),
    ex2: bool,
    perf: bool,
) -> Measured {
    let (channels, stride, t_count) = case;
    let stream = ctx.new_stream().expect("stream");
    let inp = make_inputs(channels, stride, seed(channels, stride, t_count));
    let d_in = DeviceBuffer::from_host(&stream, &inp.input).expect("input");
    let d_state = DeviceBuffer::from_host(&stream, &inp.state).expect("state");
    let d_w = DeviceBuffer::from_host(&stream, &inp.weight).expect("weight");
    let mut d_out = DeviceBuffer::<f32>::zeroed(&stream, T_TILE * stride).expect("out");
    let mut d_state_out = DeviceBuffer::<f32>::zeroed(&stream, channels * S).expect("state_out");

    let config = LaunchConfig2D::new((channels.div_ceil(BLOCK) as u32, 1), (BLOCK as u32, 1), 0);
    let (st, tc, sw) = (stride as u32, t_count as u32, (channels * S) as u32);
    let launch = |d_out: &mut DeviceBuffer<f32>, d_so: &mut DeviceBuffer<f32>| {
        if ex2 {
            let p = module.prepare_conv1d_seq_ex2(config).expect("prepare ex2");
            module
                .conv1d_seq_ex2(
                    &stream,
                    &p,
                    &d_in,
                    st,
                    tc,
                    &d_state,
                    &d_w,
                    RowWidth::new(d_out, st),
                    RowWidth::new(d_so, sw),
                )
                .expect("launch ex2");
        } else {
            let p = module.prepare_conv1d_seq_exp(config).expect("prepare exp");
            module
                .conv1d_seq_exp(
                    &stream,
                    &p,
                    &d_in,
                    st,
                    tc,
                    &d_state,
                    &d_w,
                    RowWidth::new(d_out, st),
                    RowWidth::new(d_so, sw),
                )
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
    measure(&got, &got_state, case, &inp, us, eager_us)
}

/// The hand PTX for width `n` (strides = n), loaded from its committed baseline and
/// launched on the same data: grid ceil(n/256), block 256, params (input, state,
/// weight, output, t_count = T_TILE), state in place.
fn run_handptx(ctx: &Arc<CudaContext>, sm: &str, n: usize) -> (Measured, u32) {
    let ptx_path = format!("baseline-ptx/gdn_causal_conv1d_silu_seq_c{n}.{sm}.ptx");
    let ptx = std::fs::read_to_string(&ptx_path).unwrap_or_else(|e| {
        eprintln!("missing hand-PTX baseline {ptx_path}: {e}");
        eprintln!("regenerate: APR_BLESS_PTX=1 cargo test -p aprender-gpu --features cuda --lib gdn_conv1d_seq_ptx_golden");
        std::process::exit(2);
    });
    let case = (n, n, T_TILE);
    let stream = ctx.new_stream().expect("stream");
    let inp = make_inputs(n, n, seed(n, n, T_TILE));
    let module = ctx.load_module_from_ptx_src(&ptx).expect("load hand PTX");
    let func = module
        .load_function("gdn_causal_conv1d_silu_seq")
        .expect("gdn_causal_conv1d_silu_seq");
    let regs = func.num_registers().expect("regs");
    let d_in = DeviceBuffer::from_host(&stream, &inp.input).expect("input");
    let d_state = DeviceBuffer::from_host(&stream, &inp.state).expect("state");
    let d_w = DeviceBuffer::from_host(&stream, &inp.weight).expect("weight");
    let d_out = DeviceBuffer::<f32>::zeroed(&stream, T_TILE * n).expect("out");
    let mut ptrs = [
        d_in.cu_deviceptr(),
        d_state.cu_deviceptr(),
        d_w.cu_deviceptr(),
        d_out.cu_deviceptr(),
    ];
    let mut t_count = T_TILE as u32;
    let grid = n.div_ceil(BLOCK) as u32;
    let mut launch = || {
        let mut params: Vec<*mut std::ffi::c_void> =
            ptrs.iter_mut().map(|p| (p as *mut u64).cast()).collect();
        params.push((&mut t_count as *mut u32).cast());
        // SAFETY: four device pointers and one u32 matching the entry's params;
        // input/output hold T_TILE rows of n f32, state n*3, weight n*4, and the
        // baked-in bound stops every thread at n.
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
    // Parity from the first launch. Timing then keeps advancing the window in place,
    // which does the same loads, arithmetic and stores on shifted values.
    launch();
    let got = d_out.to_host_vec(&stream).expect("download");
    let got_state = d_state.to_host_vec(&stream).expect("download state");
    let us = time_graph_us(&stream, &mut launch);
    let eager_us = time_eager_us(&stream, &mut launch);
    (measure(&got, &got_state, case, &inp, us, eager_us), regs)
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

    println!("== #3522 cuda-oxide causal conv1d + SiLU, sequence ({sm}) ==");
    println!("   block={BLOCK}, 1 channel/thread, K={K}, T_TILE={T_TILE}");

    let mut all_ok = true;
    // A timing NO-GO is its own exit code (4), not a pass: the receipts still
    // print, so receipt.sh records `"pass":false` and then fails.
    let mut timing_all_ok = true;
    for ex2 in [false, true] {
        let name = if ex2 { "ex2 (B)" } else { "exp (A)" };
        for case in PARITY_CASES {
            let r = run_oxide(&ctx, &module, case, ex2, false);
            let ok = r.ok();
            all_ok &= ok;
            println!(
                "  parity {name} (c,stride,t)={case:?}: cos={:.7} maxdiff={:.3e} {}",
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
            ("ex2", "conv1d_seq_ex2")
        } else {
            ("exp", "conv1d_seq_exp")
        };
        let regs = oxide_registers(&ctx, entry);
        let mut worst_ratio = 0.0f64;
        let mut worst = (0.0f64, 0.0f64, 0usize);
        let mut worst_eager = 0.0f64;
        let mut parity = (1.0f64, 0.0f32);
        for n in TIMED_NS {
            // Three rounds, alternating which kernel is timed first, and the round
            // with the median ratio is the row. yoga's clocks step between ~15 µs and
            // ~11.5 µs per launch partway through a run; with a fixed order that step
            // always landed on the kernel timed first (oxide: 15.5 vs 11.5 = 1.35
            // NO-GO). A step now contaminates at most one round, and the median drops it.
            let mut rounds: Vec<(Measured, Measured)> = (0..3)
                .map(|round| {
                    if round % 2 == 0 {
                        let o = run_oxide(&ctx, &module, (n, n, T_TILE), ex2, true);
                        (o, run_handptx(&ctx, &sm, n).0)
                    } else {
                        let h = run_handptx(&ctx, &sm, n).0;
                        (run_oxide(&ctx, &module, (n, n, T_TILE), ex2, true), h)
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
            parity = (parity.0.min(o.cos), parity.1.max(o.maxdiff));
        }
        let timing_ok = worst_ratio <= TIMING_RATIO_MAX;
        timing_all_ok &= timing_ok;
        let parity_ok = parity.0 >= PARITY_COS && parity.1 < PARITY_MAXDIFF;
        // One line per variant; receipt.sh adds host, sha, ptxas and writes the file.
        println!(
            "RECEIPT {{\"schema\":\"apr-kernel-receipt/v1\",\"kernel\":\"gdn_causal_conv1d_silu_seq\",\"variant\":\"{variant}\",\"entry\":\"{entry}\",\"authoring\":\"oxide\",\"cc\":\"{sm}\",\"block\":{BLOCK},\"conv_kernel\":{K},\"t_count\":{T_TILE},\"parity\":{{\"cos_min\":{:.9},\"maxdiff_max\":{:.3e},\"cos_floor\":{PARITY_COS},\"maxdiff_ceiling\":{PARITY_MAXDIFF:e},\"pass\":{parity_ok}}},\"timing\":{{\"oxide_us\":{:.3},\"handptx_us\":{:.3},\"ratio\":{worst_ratio:.4},\"worst_n\":{},\"ratio_max\":{TIMING_RATIO_MAX},\"method\":\"cuda-graph-100\",\"eager_ratio_max\":{worst_eager:.4},\"pass\":{timing_ok}}},\"register_budget\":{{\"oxide\":{},\"handptx\":{regs_hand}}}}}",
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
    println!("#3522 CAUSAL CONV1D SEQ DONE");
}
