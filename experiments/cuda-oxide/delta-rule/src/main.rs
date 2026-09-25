// #3522 OXIDE-001 — pure-Rust cuda-oxide port of the GDN gated delta-rule
// recurrence, one decode token.
//
// Target = hand-PTX `DeltaRuleRecurrenceKernel`
// (crates/aprender-gpu/src/kernels/gdn/delta_rule.rs, entry
// `gdn_delta_rule_recurrence`): grid (num_v_heads), block (head_v_dim); every head
// dimension baked into the PTX; params (q, k, v, beta, gate, state, output). Per
// value head h, with the state stored transposed (row j = column j of S):
//
//   kh        = h % num_k_heads
//   s_h      *= exp(gate[h])
//   delta[j]  = (v[j] - sum_i s_h[j*Dk+i] * k[kh*Dk+i]) * beta[h]
//   s_h[j*Dk+i] += k[kh*Dk+i] * delta[j]
//   out[j]    = (sum_i s_h[j*Dk+i] * q[kh*Dk+i]) * Dk^-0.5
//
// CPU reference = a verbatim port of aprender-serve's `delta_rule_recurrence_gqa`
// (the same port the hand kernel's device test uses). Parity is checked over three
// CHAINED decode steps, on the output AND the in-place state: an in-place kernel
// that is right once and wrong on the state it leaves behind is caught on step 2.
//
// SAFETY SHAPE (kernel-safety, #3522): the device kernel is safe Rust — no `unsafe`
// block, no raw pointer.
//   * The state is a `DisjointSlice<f32, LinearTiles<DK>>`: thread t = h*Dv + j
//     claims the Dk-element run t*Dk.., which IS memory row j of head h's state.
//     That is the hand kernel's exclusive-row ownership, now a type: no other
//     thread can reach the row, so the in-place read-modify-write needs no barrier.
//   * The output is a `DisjointSlice<f32, LinearTiles<1>>` at t.
//   * q/k heads are read as one checked `&[f32; DK]` each (`first_chunk`), so the
//     inner loops index fixed arrays and carry no per-element bounds branch.
//
// Geometry: the hand kernel's launch, grid (num_v_heads), block Dv = 128. Both
// passes over i run ascending, as the CPU loop does (the hand kernel's comment: the
// fp32 accumulation order is part of the parity contract).
//
// Two decay variants, as for the other exp-bearing rows:
//   (A) delta_rule_exp — `gate.exp()` (libm), the CPU reference's form.
//   (B) delta_rule_ex2 — `exp2(gate * log2 e)`, the hand PTX's `ex2.approx` form.
//
// Output: a human table on stdout plus one `apr-kernel-receipt/v1` JSON line per
// variant prefixed `RECEIPT `; `receipt.sh` adds host facts and writes
// evidence/kernels/gdn_delta_rule/<host>.json.

use cuda_core::{CudaContext, DeviceBuffer, IntoResult, LaunchConfig1D, sys};
use cuda_device::{
    DisjointSlice, LinearTiles, LocalIndex32, StaticViewMut32, ThreadRunMut32, cuda_module, kernel, launch_bounds,
    launch_contract, thread,
};
use std::sync::Arc;

/// Qwen3.5 head widths (every size): Dk = Dv = 128. num_k_heads = 16.
const DK: usize = 128;
const DV: usize = 128;
const NK: usize = 16;
const BLOCK: u32 = DV as u32;

// F-OXIDE-ROPE-PARITY-001 form, as #3522's kernel-parity shape states it.
const PARITY_COS: f64 = 0.9999;
const PARITY_MAXDIFF: f32 = 1e-3;
const TIMING_RATIO_MAX: f64 = 1.2;

/// num_v_heads: 0.8B/2B (16), 4B/9B (32), 27B (48). Each has a committed hand-PTX
/// baseline, since the hand kernel bakes the head counts in.
const NV_SHAPES: [usize; 3] = [16, 32, 48];
const STEPS: usize = 3;

#[cuda_module]
mod kernels {
    use super::*;

    /// One thread per state row: decay + dot k (pass 1), update + dot q (pass 2).
    #[inline(always)]
    fn recur(
        t: usize,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        beta: &[f32],
        decay: f32,
        mut s: StaticViewMut32<'_, f32, DK>,
        mut o: StaticViewMut32<'_, f32, 1>,
    ) {
        let h = t / DV;
        let kh = h % NK;
        let (Some(qh), Some(kh_), Some(&vj), Some(&b)) = (
            q.get(kh * DK..).and_then(|s| s.first_chunk::<DK>()),
            k.get(kh * DK..).and_then(|s| s.first_chunk::<DK>()),
            v.get(t),
            beta.get(h),
        ) else {
            return;
        };
        let mut sum = 0.0f32;
        for i in 0..DK as u32 {
            let Some(li) = LocalIndex32::<DK>::new(i) else { return };
            let mut e = s.at(li);
            let v = e.read() * decay;
            e.write(v);
            sum += v * kh_[i as usize];
        }
        let delta = (vj - sum) * b;
        let mut acc = 0.0f32;
        for i in 0..DK as u32 {
            let Some(li) = LocalIndex32::<DK>::new(i) else { return };
            let mut e = s.at(li);
            let v = e.read() + kh_[i as usize] * delta;
            e.write(v);
            acc += v * qh[i as usize];
        }
        o.at_const::<0>().write(acc * (1.0 / 128.0f32.sqrt()));
    }

    /// (A) decay via `exp` (libm).
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
    pub fn delta_rule_exp(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        beta: &[f32],
        gate: &[f32],
        mut state: DisjointSlice<f32, LinearTiles<DK>>,
        mut out: DisjointSlice<f32, LinearTiles<1>>,
    ) {
        let h = thread::index_1d_u32(launch_context).get() as usize / DV;
        let Some(&g) = gate.get(h) else { return };
        let (Some(ThreadRunMut32::Full(s)), Some(ThreadRunMut32::Full(o))) = (
            state.thread_run32(thread::index_1d_u32(launch_context)),
            out.thread_run32(thread::index_1d_u32(launch_context)),
        ) else {
            return;
        };
        recur(thread::index_1d_u32(launch_context).get() as usize, q, k, v, beta, g.exp(), s, o);
    }

    /// (B) decay via `exp2(gate * log2 e)`, the hand PTX's form.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
    pub fn delta_rule_ex2(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        beta: &[f32],
        gate: &[f32],
        mut state: DisjointSlice<f32, LinearTiles<DK>>,
        mut out: DisjointSlice<f32, LinearTiles<1>>,
    ) {
        let h = thread::index_1d_u32(launch_context).get() as usize / DV;
        let Some(&g) = gate.get(h) else { return };
        let decay = (g * std::f32::consts::LOG2_E).exp2();
        let (Some(ThreadRunMut32::Full(s)), Some(ThreadRunMut32::Full(o))) = (
            state.thread_run32(thread::index_1d_u32(launch_context)),
            out.thread_run32(thread::index_1d_u32(launch_context)),
        ) else {
            return;
        };
        recur(thread::index_1d_u32(launch_context).get() as usize, q, k, v, beta, decay, s, o);
    }
}

/// Verbatim port of aprender-serve's `delta_rule_recurrence_gqa` (forward_qwen35.rs),
/// as crates/aprender-gpu/src/kernels/gdn/delta_rule.rs's device test carries it.
#[allow(clippy::too_many_arguments)]
fn cpu_delta_rule(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    beta: &[f32],
    gate: &[f32],
    state: &mut [f32],
    output: &mut [f32],
    nv: usize,
) {
    let scale = 1.0 / (DK as f32).sqrt();
    for h in 0..nv {
        let kh = h % NK;
        let q_h = &q[kh * DK..(kh + 1) * DK];
        let k_h = &k[kh * DK..(kh + 1) * DK];
        let v_h = &v[h * DV..(h + 1) * DV];
        let s_h = &mut state[h * DV * DK..(h + 1) * DV * DK];
        let exp_gate = gate[h].exp();
        for s in s_h.iter_mut() {
            *s *= exp_gate;
        }
        let mut delta = vec![0.0; DV];
        for j in 0..DV {
            let row_j = &s_h[j * DK..(j + 1) * DK];
            let mut sum = 0.0;
            for i in 0..DK {
                sum += row_j[i] * k_h[i];
            }
            delta[j] = (v_h[j] - sum) * beta[h];
        }
        for j in 0..DV {
            let row_j = &mut s_h[j * DK..(j + 1) * DK];
            for i in 0..DK {
                row_j[i] += k_h[i] * delta[j];
            }
        }
        for j in 0..DV {
            let row_j = &s_h[j * DK..(j + 1) * DK];
            let mut sum = 0.0;
            for i in 0..DK {
                sum += row_j[i] * q_h[i];
            }
            output[h * DV + j] = sum * scale;
        }
    }
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

fn max_abs(a: &[f32]) -> f32 {
    a.iter().map(|x| x.abs()).fold(0.0, f32::max)
}

struct Lcg(u64);
impl Lcg {
    /// Uniform in [-1, 1).
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * scale).collect()
    }
}

fn l2_norm_per_head(x: &mut [f32]) {
    for head in x.chunks_exact_mut(DK) {
        let sq: f32 = head.iter().map(|v| v * v).sum();
        let s = 1.0 / (sq + 1e-6).sqrt();
        for v in head.iter_mut() {
            *v *= s;
        }
    }
}

/// One decode step's inputs, shaped as the device test draws them: q/k
/// L2-normalised per head, beta in [0.25, 0.75], gate (dt) negative so the state
/// decays.
struct Step {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    beta: Vec<f32>,
    gate: Vec<f32>,
}

fn make_step(rng: &mut Lcg, nv: usize) -> Step {
    let mut q = rng.vec(NK * DK, 1.0);
    let mut k = rng.vec(NK * DK, 1.0);
    l2_norm_per_head(&mut q);
    l2_norm_per_head(&mut k);
    Step {
        q,
        k,
        v: rng.vec(nv * DV, 1.0),
        beta: (0..nv).map(|_| 0.5 + 0.25 * rng.next()).collect(),
        gate: (0..nv).map(|_| -0.1 - 0.3 * rng.next().abs()).collect(),
    }
}

fn initial_state(rng: &mut Lcg, nv: usize) -> Vec<f32> {
    rng.vec(nv * DV * DK, 0.05)
}

fn seed(nv: usize) -> u64 {
    0x3522_000C + (nv as u64) * 65_536
}

/// Worst over the chained steps: output and state, both.
#[derive(Clone, Copy)]
struct Parity {
    cos: f64,
    maxdiff: f32,
    /// maxdiff over the reference's own max |value| — the scale-honest number.
    rel: f32,
}

impl Parity {
    fn new() -> Self {
        Self { cos: 1.0, maxdiff: 0.0, rel: 0.0 }
    }
    fn fold(&mut self, got: &[f32], want: &[f32]) {
        let d = max_abs_diff(got, want);
        self.cos = self.cos.min(cosine(got, want));
        self.maxdiff = self.maxdiff.max(d);
        self.rel = self.rel.max(d / max_abs(want).max(f32::MIN_POSITIVE));
    }
    fn pass(&self) -> bool {
        self.cos >= PARITY_COS && self.maxdiff < PARITY_MAXDIFF
    }
}

struct Measured {
    parity: Parity,
    /// Device time per launch, from a CUDA-graph replay (the gated number).
    us: f64,
    /// Eager per-launch time: recorded, never gated (see GDN-DECISIONS.md).
    eager_us: f64,
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


/// Host-side copies of one step's inputs on the device.
struct DevStep {
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    beta: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
}

fn upload(stream: &Arc<cuda_core::CudaStream>, s: &Step) -> DevStep {
    DevStep {
        q: DeviceBuffer::from_host(stream, &s.q).expect("q"),
        k: DeviceBuffer::from_host(stream, &s.k).expect("k"),
        v: DeviceBuffer::from_host(stream, &s.v).expect("v"),
        beta: DeviceBuffer::from_host(stream, &s.beta).expect("beta"),
        gate: DeviceBuffer::from_host(stream, &s.gate).expect("gate"),
    }
}

/// Three chained steps against the CPU port (parity), then, when `perf`, the last
/// step's launch timed on the state it left behind.
fn run_oxide(ctx: &Arc<CudaContext>, module: &kernels::LoadedModule, nv: usize, ex2: bool, perf: bool) -> Measured {
    let stream = ctx.new_stream().expect("stream");
    let mut rng = Lcg(seed(nv));
    let mut host_state = initial_state(&mut rng, nv);
    let mut d_state = DeviceBuffer::from_host(&stream, &host_state).expect("state");
    let mut d_out = DeviceBuffer::<f32>::zeroed(&stream, nv * DV).expect("out");
    let config = LaunchConfig1D::new(nv as u32, BLOCK, 0);
    let launch = |d: &DevStep, d_state: &mut DeviceBuffer<f32>, d_out: &mut DeviceBuffer<f32>| {
        if ex2 {
            let p = module.prepare_delta_rule_ex2(config).expect("prepare ex2");
            module
                .delta_rule_ex2(&stream, &p, &d.q, &d.k, &d.v, &d.beta, &d.gate, d_state, d_out)
                .expect("launch ex2");
        } else {
            let p = module.prepare_delta_rule_exp(config).expect("prepare exp");
            module
                .delta_rule_exp(&stream, &p, &d.q, &d.k, &d.v, &d.beta, &d.gate, d_state, d_out)
                .expect("launch exp");
        }
    };
    let mut parity = Parity::new();
    let mut last = None;
    for _ in 0..STEPS {
        let s = make_step(&mut rng, nv);
        let d = upload(&stream, &s);
        launch(&d, &mut d_state, &mut d_out);
        let mut want = vec![0.0f32; nv * DV];
        cpu_delta_rule(&s.q, &s.k, &s.v, &s.beta, &s.gate, &mut host_state, &mut want, nv);
        parity.fold(&d_out.to_host_vec(&stream).expect("download out"), &want);
        parity.fold(&d_state.to_host_vec(&stream).expect("download state"), &host_state);
        last = Some(d);
    }
    let d = last.expect("steps");
    let (us, eager_us) = if perf {
        (
            time_graph_us(&stream, || launch(&d, &mut d_state, &mut d_out)),
            time_eager_us(&stream, || launch(&d, &mut d_state, &mut d_out)),
        )
    } else {
        (0.0, 0.0)
    };
    Measured { parity, us, eager_us }
}

/// The hand PTX, loaded from its committed baseline and run through the same three
/// chained steps: grid (nv), block Dv, params (q, k, v, beta, gate, state, out).
fn run_handptx(ctx: &Arc<CudaContext>, sm: &str, nv: usize, perf: bool) -> (Measured, u32) {
    let ptx_path = format!("baseline-ptx/gdn_delta_rule_recurrence.nv{nv}.{sm}.ptx");
    let ptx = std::fs::read_to_string(&ptx_path).unwrap_or_else(|e| {
        eprintln!("missing hand-PTX baseline {ptx_path}: {e}");
        eprintln!("regenerate: APR_BLESS_PTX=1 cargo test -p aprender-gpu --features cuda --lib gdn_delta_rule_ptx_golden");
        std::process::exit(2);
    });
    let stream = ctx.new_stream().expect("stream");
    let module = ctx.load_module_from_ptx_src(&ptx).expect("load hand PTX");
    let func = module
        .load_function("gdn_delta_rule_recurrence")
        .expect("gdn_delta_rule_recurrence");
    let regs = func.num_registers().expect("regs");
    let mut rng = Lcg(seed(nv));
    let mut host_state = initial_state(&mut rng, nv);
    let d_state = DeviceBuffer::from_host(&stream, &host_state).expect("state");
    let d_out = DeviceBuffer::<f32>::zeroed(&stream, nv * DV).expect("out");
    let launch = |d: &DevStep| {
        let mut ptrs = [
            d.q.cu_deviceptr(),
            d.k.cu_deviceptr(),
            d.v.cu_deviceptr(),
            d.beta.cu_deviceptr(),
            d.gate.cu_deviceptr(),
            d_state.cu_deviceptr(),
            d_out.cu_deviceptr(),
        ];
        let mut params: [*mut std::ffi::c_void; 7] =
            std::array::from_fn(|i| (&raw mut ptrs[i]).cast());
        // SAFETY: the seven params match the entry's seven .u64 in order; q and k
        // hold NK * DK f32, v and out nv * DV, beta and gate nv, state nv * DV * DK,
        // at the head counts this PTX was baked with, and an nv-block launch of Dv
        // threads touches only those.
        unsafe {
            cuda_core::launch_kernel_on_stream(&func, (nv as u32, 1, 1), (BLOCK, 1, 1), 0, &stream, &mut params)
        }
        .expect("hand PTX launch");
    };
    let mut parity = Parity::new();
    let mut last = None;
    for _ in 0..STEPS {
        let s = make_step(&mut rng, nv);
        let d = upload(&stream, &s);
        launch(&d);
        let mut want = vec![0.0f32; nv * DV];
        cpu_delta_rule(&s.q, &s.k, &s.v, &s.beta, &s.gate, &mut host_state, &mut want, nv);
        parity.fold(&d_out.to_host_vec(&stream).expect("download out"), &want);
        parity.fold(&d_state.to_host_vec(&stream).expect("download state"), &host_state);
        last = Some(d);
    }
    let d = last.expect("steps");
    let (us, eager_us) = if perf {
        (time_graph_us(&stream, || launch(&d)), time_eager_us(&stream, || launch(&d)))
    } else {
        (0.0, 0.0)
    };
    (Measured { parity, us, eager_us }, regs)
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

    println!("== #3522 cuda-oxide gated delta-rule recurrence ({sm}) ==");
    println!("   Dk=Dv={DK} nk={NK} 1 thread/state row, {STEPS} chained steps, out+state");

    let mut all_ok = true;
    let mut timing_all_ok = true;
    let mut regs_hand = 0;
    for nv in NV_SHAPES {
        let (h, regs) = run_handptx(&ctx, &sm, nv, false);
        regs_hand = regs;
        println!(
            "  parity handPTX nv={nv}: cos={:.7} maxdiff={:.3e} rel={:.3e} {} regs={regs}",
            h.parity.cos,
            h.parity.maxdiff,
            h.parity.rel,
            if h.parity.pass() { "PASS" } else { "FAIL" }
        );
    }
    for ex2 in [false, true] {
        let (variant, entry) = if ex2 { ("ex2", "delta_rule_ex2") } else { ("exp", "delta_rule_exp") };
        println!("\n  -- ({variant}) {entry}");
        println!("  nv | oxide us | handPTX us | ratio | verdict | eager oxide/hand");
        let regs = oxide_registers(&ctx, entry);
        let mut worst_ratio = 0.0f64;
        let mut worst = (0.0f64, 0.0f64, 0usize);
        let mut worst_eager = 0.0f64;
        let mut parity = Parity::new();
        for nv in NV_SHAPES {
            // Three rounds, alternating which kernel is timed first; the round with
            // the median ratio is the row (see GDN-DECISIONS.md, causal conv1d seq).
            let mut rounds: Vec<(Measured, Measured)> = (0..3)
                .map(|round| {
                    if round % 2 == 0 {
                        let o = run_oxide(&ctx, &module, nv, ex2, true);
                        (o, run_handptx(&ctx, &sm, nv, true).0)
                    } else {
                        let h = run_handptx(&ctx, &sm, nv, true).0;
                        (run_oxide(&ctx, &module, nv, ex2, true), h)
                    }
                })
                .collect();
            rounds.sort_by(|a, b| (a.0.us / a.1.us).total_cmp(&(b.0.us / b.1.us)));
            let (o, h) = rounds.swap_remove(1);
            let ratio = o.us / h.us;
            let ok = ratio <= TIMING_RATIO_MAX;
            let eager_ratio = o.eager_us / h.eager_us;
            println!(
                "  {nv:>2} | {:>8.3} | {:>10.3} | {ratio:.3} | {:>7} | {:.3}/{:.3} = {eager_ratio:.3}  parity cos={:.7} maxdiff={:.3e} rel={:.3e}",
                o.us,
                h.us,
                if ok { "GO" } else { "NO-GO" },
                o.eager_us,
                h.eager_us,
                o.parity.cos,
                o.parity.maxdiff,
                o.parity.rel,
            );
            if ratio > worst_ratio {
                worst_ratio = ratio;
                worst = (o.us, h.us, nv);
            }
            worst_eager = worst_eager.max(eager_ratio);
            parity.cos = parity.cos.min(o.parity.cos);
            parity.maxdiff = parity.maxdiff.max(o.parity.maxdiff);
            parity.rel = parity.rel.max(o.parity.rel);
        }
        let parity_ok = parity.pass();
        let timing_ok = worst_ratio <= TIMING_RATIO_MAX;
        all_ok &= parity_ok;
        timing_all_ok &= timing_ok;
        println!(
            "RECEIPT {{\"schema\":\"apr-kernel-receipt/v1\",\"kernel\":\"gdn_delta_rule\",\"variant\":\"{variant}\",\"entry\":\"{entry}\",\"authoring\":\"oxide\",\"cc\":\"{sm}\",\"head_k_dim\":{DK},\"head_v_dim\":{DV},\"num_k_heads\":{NK},\"steps\":{STEPS},\"parity\":{{\"cos_min\":{:.9},\"maxdiff_max\":{:.3e},\"rel_max\":{:.3e},\"cos_threshold\":{PARITY_COS},\"maxdiff_threshold\":{PARITY_MAXDIFF},\"pass\":{parity_ok}}},\"timing\":{{\"oxide_us\":{:.3},\"handptx_us\":{:.3},\"ratio\":{worst_ratio:.4},\"worst_nv\":{},\"ratio_max\":{TIMING_RATIO_MAX},\"method\":\"cuda-graph-100\",\"eager_ratio_max\":{worst_eager:.4},\"pass\":{timing_ok}}},\"register_budget\":{{\"oxide\":{},\"handptx\":{regs_hand}}}}}",
            parity.cos,
            parity.maxdiff,
            parity.rel,
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
    println!("#3522 DELTA RULE DONE");
}
