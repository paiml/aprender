// #3522 OXIDE-001 — pure-Rust cuda-oxide port of the row-batched Gated DeltaNet gates.
//
// Target = hand-PTX `GdnGatesRowsKernel`
// (crates/aprender-gpu/src/kernels/gdn/rows.rs, entry `gdn_gates_rows`): one thread
// per (row, head), grid ceil(rows*heads/256), block 256, heads baked into the PTX,
// params (alpha, dt_bias, a, beta_raw, dt, beta, count = rows*heads).
//
//   i = row * heads + h,  h = i % heads
//   dt[i]   = softplus(alpha[i] + dt_bias[h]) * a[h]
//   beta[i] = 1 / (1 + exp(-beta_raw[i]))
//   softplus(x) = if x > 20 { x } else { ln(1 + exp(x)) }
//
// `dt_bias` and `a` are per head and shared by every row; the rest are [rows][heads].
// The oxide port takes `heads` as a runtime u32. The hand PTX bakes it in, so its
// `rem.u32` has an immediate divisor. A zero `heads` returns via `checked_rem`
// instead of trapping.
//
// CPU reference = `forward_deltanet`'s gate arithmetic and `fn_softplus` in
// crates/aprender-serve/src/gguf/inference/forward/forward_qwen35.rs, in f64 (the
// branch is taken on the f32 pre-activation, as the CPU and both kernels take it).
//
// SAFETY SHAPE (kernel-safety, #3522): both device kernels are safe Rust — no
// `unsafe` block, no raw pointer. Reads are `slice.get(i)` with an early return;
// each output is a `DisjointSlice<f32, LinearTiles<1>>` claimed as a full run from
// the launch-checked thread index (the index token is !Copy, so one per claim).
//
// Two variants, differing only in the transcendentals, as in gdn-gates:
//   (A) gdn_gates_rows_exp — `exp` and `ln` (libdevice), the more precise
//   (B) gdn_gates_rows_ex2 — `exp2(x * log2 e)` and `lg2_approx_f32(y) * ln 2`, the
//       hand PTX's ex2.approx / lg2.approx form.
//
// Output: a human table on stdout plus one `apr-kernel-receipt/v1` JSON line
// prefixed `RECEIPT ` per variant; `receipt.sh` adds host facts and writes
// evidence/kernels/gdn_gates_rows/<host>.json.

use cuda_core::{CudaContext, DeviceBuffer, IntoResult, LaunchConfig1D, sys};
use cuda_device::{DisjointSlice, LinearTiles, ThreadRunMut32, cuda_module, kernel, launch_bounds, launch_contract, thread};
use std::sync::Arc;

const BLOCK: usize = 256;

/// Parity floor on cosine against the f64 reference.
const PARITY_COS: f64 = 0.9999;
const PARITY_MAXDIFF: f32 = 1e-3;
/// oxide / hand-PTX time ceiling, worst over the timed shapes.
const TIMING_RATIO_MAX: f64 = 1.2;
/// `fn_softplus`'s linear-branch threshold.
const SOFTPLUS_BIG: f32 = 20.0;

/// Parity cases (heads, rows). 1000 x 3 is not a multiple of the block, so the tail
/// threads of the last block must write nothing and read nothing; 37 rows is the
/// aprender-gpu device test's odd count.
const PARITY_CASES: [(usize, usize); 4] = [(1, 1), (16, 37), (48, 64), (1000, 3)];
/// Timed head counts, each with a committed hand-PTX baseline (heads are baked in),
/// at ROWS rows: one 64-token prefill chunk, as the sequence conv is timed.
const TIMED_HEADS: [usize; 3] = [16, 32, 48];
const ROWS: usize = 64;

#[cuda_module]
mod kernels {
    use super::*;

    /// (A) softplus and sigmoid via `exp` / `ln`.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub fn gdn_gates_rows_exp(
        alpha: &[f32],
        dt_bias: &[f32],
        a: &[f32],
        beta_raw: &[f32],
        heads: u32,
        mut dt: DisjointSlice<f32, LinearTiles<1>>,
        mut beta: DisjointSlice<f32, LinearTiles<1>>,
    ) {
        let i = thread::index_1d_u32(launch_context).get() as usize;
        let Some(h) = i.checked_rem(heads as usize) else {
            return;
        };
        let Some(ThreadRunMut32::Full(mut d)) = dt.thread_run32(thread::index_1d_u32(launch_context)) else {
            return;
        };
        let Some(ThreadRunMut32::Full(mut b)) = beta.thread_run32(thread::index_1d_u32(launch_context)) else {
            return;
        };
        let (Some(&al), Some(&bi), Some(&av), Some(&br)) = (alpha.get(i), dt_bias.get(h), a.get(h), beta_raw.get(i)) else {
            return;
        };
        let pre = al + bi;
        let sp = if pre > SOFTPLUS_BIG { pre } else { (1.0f32 + pre.exp()).ln() };
        d.at_const::<0>().write(sp * av);
        b.at_const::<0>().write(1.0f32 / (1.0f32 + (-br).exp()));
    }

    /// (B) softplus and sigmoid via `exp2` / `lg2.approx`, the hand PTX's form.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub fn gdn_gates_rows_ex2(
        alpha: &[f32],
        dt_bias: &[f32],
        a: &[f32],
        beta_raw: &[f32],
        heads: u32,
        mut dt: DisjointSlice<f32, LinearTiles<1>>,
        mut beta: DisjointSlice<f32, LinearTiles<1>>,
    ) {
        let i = thread::index_1d_u32(launch_context).get() as usize;
        let Some(h) = i.checked_rem(heads as usize) else {
            return;
        };
        let Some(ThreadRunMut32::Full(mut d)) = dt.thread_run32(thread::index_1d_u32(launch_context)) else {
            return;
        };
        let Some(ThreadRunMut32::Full(mut b)) = beta.thread_run32(thread::index_1d_u32(launch_context)) else {
            return;
        };
        let (Some(&al), Some(&bi), Some(&av), Some(&br)) = (alpha.get(i), dt_bias.get(h), a.get(h), beta_raw.get(i)) else {
            return;
        };
        let pre = al + bi;
        let sp = if pre > SOFTPLUS_BIG {
            pre
        } else {
            cuda_device::float::lg2_approx_f32(1.0f32 + (pre * std::f32::consts::LOG2_E).exp2()) * std::f32::consts::LN_2
        };
        d.at_const::<0>().write(sp * av);
        b.at_const::<0>().write(1.0f32 / (1.0f32 + ((-br) * std::f32::consts::LOG2_E).exp2()));
    }
}

/// Four input vectors: alpha and beta_raw are [rows][heads], dt_bias and a [heads].
struct Inputs {
    heads: usize,
    alpha: Vec<f32>,
    dt_bias: Vec<f32>,
    a: Vec<f32>,
    beta_raw: Vec<f32>,
}

/// f64 evaluation of `forward_deltanet`'s gates, row by row. Returns (dt, beta).
fn cpu_gates(inp: &Inputs) -> (Vec<f32>, Vec<f32>) {
    let mut dt = Vec::with_capacity(inp.alpha.len());
    let mut beta = Vec::with_capacity(inp.alpha.len());
    for i in 0..inp.alpha.len() {
        let h = i % inp.heads;
        let pre32 = inp.alpha[i] + inp.dt_bias[h];
        let pre = f64::from(pre32);
        let sp = if pre32 > SOFTPLUS_BIG { pre } else { (1.0 + pre.exp()).ln() };
        dt.push((sp * f64::from(inp.a[h])) as f32);
        beta.push((1.0 / (1.0 + (-f64::from(inp.beta_raw[i])).exp())) as f32);
    }
    (dt, beta)
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

/// Deterministic inputs, scaled as gdn-gates scales them. The pre-activation
/// `alpha + dt_bias` spans about ±32, so both softplus branches are taken (and the
/// threshold's neighbourhood); `a` is negative, as `ssm_a = -exp(A_log)` is in the
/// model; `beta_raw` spans ±8, into the sigmoid's saturating tails.
fn make_inputs(heads: usize, rows: usize, seed: u64) -> Inputs {
    let mut s = seed;
    let mut next = move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let n = heads * rows;
    Inputs {
        heads,
        alpha: (0..n).map(|_| next() * 24.0).collect(),
        dt_bias: (0..heads).map(|_| next() * 8.0).collect(),
        a: (0..heads).map(|_| -(next() * 2.0).exp()).collect(),
        beta_raw: (0..n).map(|_| next() * 8.0).collect(),
    }
}

struct Measured {
    /// Worst cosine of dt and beta against the f64 reference.
    cos: f64,
    /// Worst |Δ| over dt and beta.
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

fn measure(got_dt: &[f32], got_beta: &[f32], inp: &Inputs, us: f64, eager_us: f64) -> Measured {
    let (want_dt, want_beta) = cpu_gates(inp);
    Measured {
        cos: cosine(got_dt, &want_dt).min(cosine(got_beta, &want_beta)),
        maxdiff: max_abs_diff(got_dt, &want_dt).max(max_abs_diff(got_beta, &want_beta)),
        us,
        eager_us,
    }
}

fn seed(heads: usize, rows: usize) -> u64 {
    0x3522_0009 + (heads * 4096 + rows) as u64
}

fn run_oxide(
    ctx: &Arc<CudaContext>,
    module: &kernels::LoadedModule,
    (heads, rows): (usize, usize),
    ex2: bool,
    perf: bool,
) -> Measured {
    let n = heads * rows;
    let stream = ctx.new_stream().expect("stream");
    let inp = make_inputs(heads, rows, seed(heads, rows));
    let d_alpha = DeviceBuffer::from_host(&stream, &inp.alpha).expect("alpha");
    let d_bias = DeviceBuffer::from_host(&stream, &inp.dt_bias).expect("dt_bias");
    let d_a = DeviceBuffer::from_host(&stream, &inp.a).expect("a");
    let d_br = DeviceBuffer::from_host(&stream, &inp.beta_raw).expect("beta_raw");
    let mut d_dt = DeviceBuffer::<f32>::zeroed(&stream, n).expect("dt");
    let mut d_beta = DeviceBuffer::<f32>::zeroed(&stream, n).expect("beta");
    let heads_u32 = heads as u32;

    let config = LaunchConfig1D::new(n.div_ceil(BLOCK) as u32, BLOCK as u32, 0);
    let launch = |d_dt: &mut DeviceBuffer<f32>, d_beta: &mut DeviceBuffer<f32>| {
        if ex2 {
            let p = module.prepare_gdn_gates_rows_ex2(config).expect("prepare ex2");
            module
                .gdn_gates_rows_ex2(&stream, &p, &d_alpha, &d_bias, &d_a, &d_br, heads_u32, d_dt, d_beta)
                .expect("launch ex2");
        } else {
            let p = module.prepare_gdn_gates_rows_exp(config).expect("prepare exp");
            module
                .gdn_gates_rows_exp(&stream, &p, &d_alpha, &d_bias, &d_a, &d_br, heads_u32, d_dt, d_beta)
                .expect("launch exp");
        }
    };
    launch(&mut d_dt, &mut d_beta);
    let got_dt = d_dt.to_host_vec(&stream).expect("download dt");
    let got_beta = d_beta.to_host_vec(&stream).expect("download beta");
    let (us, eager_us) = if perf {
        (
            time_graph_us(&stream, || launch(&mut d_dt, &mut d_beta)),
            time_eager_us(&stream, || launch(&mut d_dt, &mut d_beta)),
        )
    } else {
        (0.0, 0.0)
    };
    measure(&got_dt, &got_beta, &inp, us, eager_us)
}

/// The hand PTX for `heads` heads, loaded from its committed baseline and launched
/// on the same data: grid ceil(rows*heads/256), block 256, params (alpha, dt_bias,
/// a, beta_raw, dt, beta, count).
fn run_handptx(ctx: &Arc<CudaContext>, sm: &str, (heads, rows): (usize, usize)) -> (Measured, u32) {
    let n = heads * rows;
    let ptx_path = format!("baseline-ptx/gdn_gates_rows_h{heads}.{sm}.ptx");
    let ptx = std::fs::read_to_string(&ptx_path).unwrap_or_else(|e| {
        eprintln!("missing hand-PTX baseline {ptx_path}: {e}");
        eprintln!("regenerate: APR_BLESS_PTX=1 cargo test -p aprender-gpu --features cuda --lib gdn_gates_rows_ptx_golden");
        std::process::exit(2);
    });
    let stream = ctx.new_stream().expect("stream");
    let inp = make_inputs(heads, rows, seed(heads, rows));
    let module = ctx.load_module_from_ptx_src(&ptx).expect("load hand PTX");
    let func = module.load_function("gdn_gates_rows").expect("gdn_gates_rows");
    let regs = func.num_registers().expect("regs");
    let d_alpha = DeviceBuffer::from_host(&stream, &inp.alpha).expect("alpha");
    let d_bias = DeviceBuffer::from_host(&stream, &inp.dt_bias).expect("dt_bias");
    let d_a = DeviceBuffer::from_host(&stream, &inp.a).expect("a");
    let d_br = DeviceBuffer::from_host(&stream, &inp.beta_raw).expect("beta_raw");
    let d_dt = DeviceBuffer::<f32>::zeroed(&stream, n).expect("dt");
    let d_beta = DeviceBuffer::<f32>::zeroed(&stream, n).expect("beta");
    let mut ptrs = [
        d_alpha.cu_deviceptr(),
        d_bias.cu_deviceptr(),
        d_a.cu_deviceptr(),
        d_br.cu_deviceptr(),
        d_dt.cu_deviceptr(),
        d_beta.cu_deviceptr(),
    ];
    let mut count = n as u32;
    let grid = n.div_ceil(BLOCK) as u32;
    let mut launch = || {
        let mut params: Vec<*mut std::ffi::c_void> =
            ptrs.iter_mut().map(|p| (p as *mut u64).cast()).collect();
        params.push((&mut count as *mut u32).cast());
        // SAFETY: six device pointers and one u32 matching the entry's params;
        // alpha/beta_raw/dt/beta hold rows*heads f32, dt_bias/a hold heads, and
        // `count` = rows*heads stops every thread past the end.
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
    launch();
    let got_dt = d_dt.to_host_vec(&stream).expect("download dt");
    let got_beta = d_beta.to_host_vec(&stream).expect("download beta");
    let us = time_graph_us(&stream, &mut launch);
    let eager_us = time_eager_us(&stream, &mut launch);
    (measure(&got_dt, &got_beta, &inp, us, eager_us), regs)
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

    println!("== #3522 cuda-oxide GDN gates rows ({sm}) ==");
    println!("   block={BLOCK}, 1 (row, head)/thread, timed at {ROWS} rows");

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
                "  parity {name} heads={} rows={}: cos={:.7} maxdiff={:.3e} {}",
                case.0,
                case.1,
                r.cos,
                r.maxdiff,
                if ok { "PASS" } else { "FAIL" }
            );
        }
    }

    let regs_hand;
    {
        let (h, r) = run_handptx(&ctx, &sm, (TIMED_HEADS[0], ROWS));
        regs_hand = r;
        let ok = h.ok();
        println!(
            "  hand PTX heads={}: cos={:.7} maxdiff={:.3e} regs={r} {}",
            TIMED_HEADS[0],
            h.cos,
            h.maxdiff,
            if ok { "PASS" } else { "FAIL" }
        );
        all_ok &= ok;
    }
    println!("\n  heads | variant | oxide us | handPTX us | ratio | verdict | eager oxide/hand");
    for ex2 in [false, true] {
        let (variant, entry) = if ex2 {
            ("ex2", "gdn_gates_rows_ex2")
        } else {
            ("exp", "gdn_gates_rows_exp")
        };
        let regs = oxide_registers(&ctx, entry);
        let mut worst_ratio = 0.0f64;
        let mut worst = (0.0f64, 0.0f64, 0usize);
        let mut worst_eager = 0.0f64;
        let mut parity = (1.0f64, 0.0f32);
        for n in TIMED_HEADS {
            // Three rounds, alternating which kernel is timed first; the round with
            // the median ratio is the row. A host clock step mid-run (yoga, see
            // GDN-DECISIONS.md, causal conv1d seq) then lands in at most one round.
            let mut rounds: Vec<(Measured, Measured)> = (0..3)
                .map(|round| {
                    if round % 2 == 0 {
                        let o = run_oxide(&ctx, &module, (n, ROWS), ex2, true);
                        (o, run_handptx(&ctx, &sm, (n, ROWS)).0)
                    } else {
                        let h = run_handptx(&ctx, &sm, (n, ROWS)).0;
                        (run_oxide(&ctx, &module, (n, ROWS), ex2, true), h)
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
            "RECEIPT {{\"schema\":\"apr-kernel-receipt/v1\",\"kernel\":\"gdn_gates_rows\",\"variant\":\"{variant}\",\"entry\":\"{entry}\",\"authoring\":\"oxide\",\"cc\":\"{sm}\",\"block\":{BLOCK},\"parity\":{{\"cos_min\":{:.9},\"maxdiff_max\":{:.3e},\"cos_floor\":{PARITY_COS},\"maxdiff_ceiling\":{PARITY_MAXDIFF:e},\"pass\":{parity_ok}}},\"timing\":{{\"oxide_us\":{:.3},\"handptx_us\":{:.3},\"ratio\":{worst_ratio:.4},\"worst_n\":{},\"ratio_max\":{TIMING_RATIO_MAX},\"method\":\"cuda-graph-100\",\"eager_ratio_max\":{worst_eager:.4},\"pass\":{timing_ok}}},\"register_budget\":{{\"oxide\":{},\"handptx\":{regs_hand}}}}}",
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
    println!("#3522 GDN GATES ROWS DONE");
}
