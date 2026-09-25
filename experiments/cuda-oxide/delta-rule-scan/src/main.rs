// #3522 OXIDE-001 — pure-Rust cuda-oxide port of the GDN gated delta-rule CHUNK
// SCAN: the per-token recurrence over T consecutive prefill tokens in one launch,
// the state row held on chip for the whole chunk.
//
// Target = hand-PTX `DeltaRuleChunkScanKernel`
// (crates/aprender-gpu/src/kernels/gdn/delta_rule_scan.rs, entry
// `gdn_delta_rule_chunk_scan`): grid (num_v_heads), block (head_v_dim); head dims and
// both row strides baked into the PTX; params (q, k, v, beta, gate, state, output,
// t_count). Token t reads q/k/v at row t of the `[T][q | k | v]` conv output, beta
// and gate at `[t][h]`, and writes output row t; per token it is exactly
// `DeltaRuleRecurrenceKernel` (see experiments/cuda-oxide/delta-rule/).
//
// CPU reference = T calls of the verbatim `delta_rule_recurrence_gqa` port, one per
// token, on the same state. Parity runs a CHAINED schedule of launches, t_count
// 1, 17 then CHUNK, on one state (82 tokens), and checks every output row of every
// launch and the state each launch leaves behind.
//
// SAFETY SHAPE (kernel-safety, #3522): the device kernel is safe Rust — no `unsafe`
// block, no raw pointer — and that decides two things the hand kernel does
// differently:
//   * NO SHARED STAGING. The hand kernel copies each token's k and q heads into
//     shared memory behind two `bar.sync`s. cuda-oxide's `SharedArray` is a
//     `static mut` whose every access is `unsafe`, so this port reads k and q from
//     global memory instead: all Dv threads of a block read the same address at the
//     same step, one broadcast L1 transaction. The timing gate decides whether that
//     costs anything; with no shared memory there is also no barrier, so no thread
//     has to reach one and an early exit is allowed.
//   * THE OUTPUT IS A COLUMN. Thread x = h*Dv + j writes element x of every output
//     row: `RuntimeRowMajorTiles<CHUNK, 1>`, the matrix row width (out_row_stride)
//     bound to the slice by the host. Tile rows are a type constant, so ONE launch
//     covers at most CHUNK tokens: `t_count <= CHUNK`, the output buffer is CHUNK
//     rows, and a longer prompt is several launches on the same state (the parity
//     schedule does exactly that). The hand kernel takes any T in one launch.
//   * The state row is a `RuntimeRowMajorTiles<1, DK>` tile of the one-row matrix
//     `[nv * Dv * Dk]`: thread x owns elements x*Dk.., which IS memory row j of head
//     h's state, the hand kernel's exclusive-row ownership as a type. It is read once
//     into a local `[f32; DK]` and written back once, as the hand kernel's registers.
//   * q/k heads are read as one checked `&[f32; DK]` each (`first_chunk`).
//
// Geometry: the hand kernel's launch, grid (num_v_heads), block Dv = 128, typed as a
// 2D domain (row 0) because the tile views are 2D. Both passes over i run ascending,
// as the CPU loop does.
//
// Two decay variants, as for the other exp-bearing rows:
//   (A) delta_rule_scan_exp — `gate.exp()` (libm), the CPU reference's form.
//   (B) delta_rule_scan_ex2 — `exp2(gate * log2 e)`, the hand PTX's `ex2.approx` form.
//
// Output: a human table on stdout plus one `apr-kernel-receipt/v1` JSON line per
// variant prefixed `RECEIPT `; `receipt.sh` adds host facts and writes
// evidence/kernels/gdn_delta_rule_scan/<host>.json.

use cuda_core::{CudaContext, DeviceBuffer, IntoResult, LaunchConfig2D, sys};
use cuda_device::{
    DisjointSlice, LocalIndex32, RuntimeRowMajorTiles, cuda_module, kernel, launch_bounds,
    launch_contract, thread,
};
use std::sync::Arc;

/// Qwen3.5 head widths (every size): Dk = Dv = 128. num_k_heads = 16.
const DK: usize = 128;
const DV: usize = 128;
const NK: usize = 16;
const BLOCK: u32 = DV as u32;
/// Tokens one oxide launch can cover (the output tile's row count).
const CHUNK: usize = 64;

// F-OXIDE-ROPE-PARITY-001 form, as #3522's kernel-parity shape states it.
const PARITY_COS: f64 = 0.9999;
const PARITY_MAXDIFF: f32 = 1e-3;
const TIMING_RATIO_MAX: f64 = 1.2;

/// num_v_heads: 0.8B/2B (16), 4B/9B (32), 27B (48). Each has a committed hand-PTX
/// baseline, since the hand kernel bakes the head counts and row strides in.
const NV_SHAPES: [usize; 3] = [16, 32, 48];
/// Chained parity launches (t_count each), on one state.
const SCHEDULE: [usize; 3] = [1, 17, CHUNK];

/// The `[q | k | v]` conv-output row width at `nv` value heads.
const fn qkv_stride(nv: usize) -> usize {
    2 * NK * DK + nv * DV
}

/// Both decay variants from one body: a `#[device]` helper is emitted as a real
/// `.func` call (and its variant flag as a runtime branch), so the body is expanded
/// into each `#[kernel]` instead, and the macro emits the whole
/// `#[cuda_module]`, which looks for `#[kernel]` items before macros expand. `#[unroll]` needs counted `while` loops with no
/// early exit; full unrolling is what keeps the `[f32; DK]` row in registers (the
/// `for` form left it in `.local` memory).
macro_rules! scan_kernels {
    ($($(#[$doc:meta])* $name:ident, |$g:ident| $decay:expr;)+) => {
    #[cuda_module]
    mod kernels {
        use super::*;
    $(
        $(#[$doc])*
        #[kernel(launch_context = launch_context)]
        #[launch_bounds(128)]
        #[launch_contract(domain = 2, coordinates = u32, block = (128, 1, 1))]
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
            rows: &[f32],
            stride: u32,
            beta: &[f32],
            gate: &[f32],
            nv: u32,
            t_count: u32,
            mut state: DisjointSlice<f32, RuntimeRowMajorTiles<1, DK>>,
            mut out: DisjointSlice<f32, RuntimeRowMajorTiles<CHUNK, 1>>,
        ) {
            let x = thread::coord_2d_u32(launch_context).col() as usize;
            let (Some(mut s), Some(mut o)) = (
                state.tile_2d32_rt(thread::coord_2d_u32(launch_context)),
                out.tile_2d32_rt(thread::coord_2d_u32(launch_context)),
            ) else {
                return;
            };
            let (stride, nv) = (stride as usize, nv as usize);
            let h = x / DV;
            let kh = h % NK;
            let mut row = [0.0f32; DK];
            let mut i = 0u32;
            #[unroll]
            while i < DK as u32 {
                if let Some(li) = LocalIndex32::<DK>::new(i) {
                    row[i as usize] = s.at(LocalIndex32::<1>::constant::<0>(), li).read();
                }
                i += 1;
            }
            let mut t = 0u32;
            while t < t_count {
                let Some(ot) = LocalIndex32::<CHUNK>::new(t) else { break };
                let base = t as usize * stride;
                let hi = t as usize * nv + h;
                let (Some(qh), Some(kv), Some(&vj), Some(&b), Some(&$g)) = (
                    rows.get(base + kh * DK..).and_then(|r| r.first_chunk::<DK>()),
                    rows.get(base + NK * DK + kh * DK..).and_then(|r| r.first_chunk::<DK>()),
                    rows.get(base + 2 * NK * DK + x),
                    beta.get(hi),
                    gate.get(hi),
                ) else {
                    break;
                };
                let decay = $decay;
                let mut sum = 0.0f32;
                let mut i = 0usize;
                #[unroll]
                while i < DK {
                    row[i] *= decay;
                    sum += row[i] * kv[i];
                    i += 1;
                }
                let delta = (vj - sum) * b;
                let mut acc = 0.0f32;
                let mut i = 0usize;
                #[unroll]
                while i < DK {
                    row[i] += kv[i] * delta;
                    acc += row[i] * qh[i];
                    i += 1;
                }
                o.at(ot, LocalIndex32::<1>::constant::<0>()).write(acc * (1.0 / 128.0f32.sqrt()));
                t += 1;
            }
            // A malformed launch breaks out above; the row still goes back whole.
            let mut i = 0u32;
            #[unroll]
            while i < DK as u32 {
                if let Some(li) = LocalIndex32::<DK>::new(i) {
                    s.at(LocalIndex32::<1>::constant::<0>(), li).write(row[i as usize]);
                }
                i += 1;
            }
        }
    )+
    }
    };
}

scan_kernels! {
    /// (A) decay via `exp` (libm).
    delta_rule_scan_exp, |g| g.exp();
    /// (B) decay via `exp2(gate * log2 e)`, the hand PTX's form.
    delta_rule_scan_ex2, |g| (g * std::f32::consts::LOG2_E).exp2();
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
    0x3522_000D + (nv as u64) * 65_536
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


/// One launch's worth of tokens: `[T][q | k | v]` rows, `[T][nv]` beta and gate.
struct Chunk {
    t: usize,
    steps: Vec<Step>,
    rows: Vec<f32>,
    beta: Vec<f32>,
    gate: Vec<f32>,
}

fn make_chunk(rng: &mut Lcg, nv: usize, t: usize) -> Chunk {
    let steps: Vec<Step> = (0..t).map(|_| make_step(rng, nv)).collect();
    let mut rows = Vec::with_capacity(t * qkv_stride(nv));
    let (mut beta, mut gate) = (Vec::new(), Vec::new());
    for s in &steps {
        rows.extend_from_slice(&s.q);
        rows.extend_from_slice(&s.k);
        rows.extend_from_slice(&s.v);
        beta.extend_from_slice(&s.beta);
        gate.extend_from_slice(&s.gate);
    }
    Chunk { t, steps, rows, beta, gate }
}

/// T per-token CPU steps on `state`; returns the `[T][nv * Dv]` outputs.
fn cpu_chunk(c: &Chunk, state: &mut [f32], nv: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; c.t * nv * DV];
    for (s, o) in c.steps.iter().zip(out.chunks_exact_mut(nv * DV)) {
        cpu_delta_rule(&s.q, &s.k, &s.v, &s.beta, &s.gate, state, o, nv);
    }
    out
}

struct DevChunk {
    t: usize,
    rows: DeviceBuffer<f32>,
    beta: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
}

fn upload(stream: &Arc<cuda_core::CudaStream>, c: &Chunk) -> DevChunk {
    DevChunk {
        t: c.t,
        rows: DeviceBuffer::from_host(stream, &c.rows).expect("rows"),
        beta: DeviceBuffer::from_host(stream, &c.beta).expect("beta"),
        gate: DeviceBuffer::from_host(stream, &c.gate).expect("gate"),
    }
}

/// The chained SCHEDULE against the CPU port (parity: every output row of every
/// launch, and the state after each), then, when `perf`, one CHUNK-token launch
/// timed on the state the schedule left behind.
fn run_oxide(ctx: &Arc<CudaContext>, module: &kernels::LoadedModule, nv: usize, ex2: bool, perf: bool) -> Measured {
    let stream = ctx.new_stream().expect("stream");
    let mut rng = Lcg(seed(nv));
    let mut host_state = initial_state(&mut rng, nv);
    let mut d_state = DeviceBuffer::from_host(&stream, &host_state).expect("state");
    // CHUNK rows always: the output tile is CHUNK x 1, checked against the length.
    let mut d_out = DeviceBuffer::<f32>::zeroed(&stream, CHUNK * nv * DV).expect("out");
    let config = LaunchConfig2D::new((nv as u32, 1), (BLOCK, 1), 0);
    let stride = qkv_stride(nv) as u32;
    let state_width = (nv * DV * DK) as u32;
    let out_width = (nv * DV) as u32;
    let launch = |d: &DevChunk, d_state: &mut DeviceBuffer<f32>, d_out: &mut DeviceBuffer<f32>| {
        let t = d.t as u32;
        if ex2 {
            let p = module.prepare_delta_rule_scan_ex2(config).expect("prepare ex2");
            module
                .delta_rule_scan_ex2(
                    &stream,
                    &p,
                    &d.rows,
                    stride,
                    &d.beta,
                    &d.gate,
                    nv as u32,
                    t,
                    cuda_host::RowWidth::new(d_state, state_width),
                    cuda_host::RowWidth::new(d_out, out_width),
                )
                .expect("launch ex2");
        } else {
            let p = module.prepare_delta_rule_scan_exp(config).expect("prepare exp");
            module
                .delta_rule_scan_exp(
                    &stream,
                    &p,
                    &d.rows,
                    stride,
                    &d.beta,
                    &d.gate,
                    nv as u32,
                    t,
                    cuda_host::RowWidth::new(d_state, state_width),
                    cuda_host::RowWidth::new(d_out, out_width),
                )
                .expect("launch exp");
        }
    };
    let mut parity = Parity::new();
    for t in SCHEDULE {
        let c = make_chunk(&mut rng, nv, t);
        let d = upload(&stream, &c);
        launch(&d, &mut d_state, &mut d_out);
        let want = cpu_chunk(&c, &mut host_state, nv);
        let got = d_out.to_host_vec(&stream).expect("download out");
        parity.fold(&got[..want.len()], &want);
        parity.fold(&d_state.to_host_vec(&stream).expect("download state"), &host_state);
    }
    let (us, eager_us) = if perf {
        let d = upload(&stream, &make_chunk(&mut rng, nv, CHUNK));
        (
            time_graph_us(&stream, || launch(&d, &mut d_state, &mut d_out)),
            time_eager_us(&stream, || launch(&d, &mut d_state, &mut d_out)),
        )
    } else {
        (0.0, 0.0)
    };
    Measured { parity, us, eager_us }
}

/// The hand PTX, loaded from its committed baseline and run through the same
/// schedule: grid (nv), block Dv, params (q, k, v, beta, gate, state, out, t_count),
/// q/k/v the conv-output buffer offset to its three sections.
fn run_handptx(ctx: &Arc<CudaContext>, sm: &str, nv: usize, perf: bool) -> (Measured, u32) {
    let ptx_path = format!("baseline-ptx/gdn_delta_rule_chunk_scan.nv{nv}.{sm}.ptx");
    let ptx = std::fs::read_to_string(&ptx_path).unwrap_or_else(|e| {
        eprintln!("missing hand-PTX baseline {ptx_path}: {e}");
        eprintln!("regenerate: APR_BLESS_PTX=1 cargo test -p aprender-gpu --features cuda --lib gdn_delta_rule_scan_ptx_golden");
        std::process::exit(2);
    });
    let stream = ctx.new_stream().expect("stream");
    let module = ctx.load_module_from_ptx_src(&ptx).expect("load hand PTX");
    let func = module
        .load_function("gdn_delta_rule_chunk_scan")
        .expect("gdn_delta_rule_chunk_scan");
    let regs = func.num_registers().expect("regs");
    let mut rng = Lcg(seed(nv));
    let mut host_state = initial_state(&mut rng, nv);
    let d_state = DeviceBuffer::from_host(&stream, &host_state).expect("state");
    let d_out = DeviceBuffer::<f32>::zeroed(&stream, CHUNK * nv * DV).expect("out");
    let launch = |d: &DevChunk| {
        let base = d.rows.cu_deviceptr();
        let mut ptrs = [
            base,
            base + (NK * DK * 4) as u64,
            base + (2 * NK * DK * 4) as u64,
            d.beta.cu_deviceptr(),
            d.gate.cu_deviceptr(),
            d_state.cu_deviceptr(),
            d_out.cu_deviceptr(),
        ];
        let mut t_count = d.t as u32;
        let mut params: [*mut std::ffi::c_void; 8] = std::array::from_fn(|i| {
            if i < 7 { (&raw mut ptrs[i]).cast() } else { (&raw mut t_count).cast() }
        });
        // SAFETY: the eight params match the entry's seven .u64 and one .u32 in
        // order; rows holds t_count rows of the qkv stride this PTX was baked with,
        // beta and gate t_count * nv, state nv * Dv * Dk, out CHUNK >= t_count rows
        // of nv * Dv, and an nv-block launch of Dv threads touches only those. The
        // k/q staging buffer is static shared memory declared by the PTX.
        unsafe {
            cuda_core::launch_kernel_on_stream(&func, (nv as u32, 1, 1), (BLOCK, 1, 1), 0, &stream, &mut params)
        }
        .expect("hand PTX launch");
    };
    let mut parity = Parity::new();
    for t in SCHEDULE {
        let c = make_chunk(&mut rng, nv, t);
        let d = upload(&stream, &c);
        launch(&d);
        let want = cpu_chunk(&c, &mut host_state, nv);
        let got = d_out.to_host_vec(&stream).expect("download out");
        parity.fold(&got[..want.len()], &want);
        parity.fold(&d_state.to_host_vec(&stream).expect("download state"), &host_state);
    }
    let (us, eager_us) = if perf {
        let d = upload(&stream, &make_chunk(&mut rng, nv, CHUNK));
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

    println!("== #3522 cuda-oxide gated delta-rule chunk scan ({sm}) ==");
    println!("   Dk=Dv={DK} nk={NK} 1 thread/state row, chained t_count {SCHEDULE:?}, timed T={CHUNK}");

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
        let (variant, entry) = if ex2 { ("ex2", "delta_rule_scan_ex2") } else { ("exp", "delta_rule_scan_exp") };
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
            "RECEIPT {{\"schema\":\"apr-kernel-receipt/v1\",\"kernel\":\"gdn_delta_rule_scan\",\"variant\":\"{variant}\",\"entry\":\"{entry}\",\"authoring\":\"oxide\",\"cc\":\"{sm}\",\"head_k_dim\":{DK},\"head_v_dim\":{DV},\"num_k_heads\":{NK},\"chunk\":{CHUNK},\"schedule\":{SCHEDULE:?},\"parity\":{{\"cos_min\":{:.9},\"maxdiff_max\":{:.3e},\"rel_max\":{:.3e},\"cos_threshold\":{PARITY_COS},\"maxdiff_threshold\":{PARITY_MAXDIFF},\"pass\":{parity_ok}}},\"timing\":{{\"oxide_us\":{:.3},\"handptx_us\":{:.3},\"ratio\":{worst_ratio:.4},\"worst_nv\":{},\"tokens\":{CHUNK},\"ratio_max\":{TIMING_RATIO_MAX},\"method\":\"cuda-graph-100\",\"eager_ratio_max\":{worst_eager:.4},\"pass\":{timing_ok}}},\"register_budget\":{{\"oxide\":{},\"handptx\":{regs_hand}}}}}",
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
    println!("#3522 DELTA RULE SCAN DONE");
}
