// #3522 OXIDE-001 — pure-Rust cuda-oxide port of the GDN causal GQA PREFILL FLASH
// ATTENTION, head_dim 256 (Qwen3.5's gated full-attention layers, prompt path).
//
// Target = hand-PTX `PrefillFlashAttention256Kernel`
// (crates/aprender-gpu/src/kernels/gdn/prefill_flash_attention.rs, entry
// `gdn_prefill_flash_attention_256`): grid (ceil(rows / 16), num_kv_heads), block
// 32 * heads_per_kv; params (q, k, v, out, rows, pos0). q/out are
// `[rows][num_heads * 256]`, k/v `[pos0 + rows][num_kv_heads * 256]`; query row r sits
// at position pos0 + r and attends keys 0 ..= pos0 + r. The hand kernel is FA2 on the
// tensor cores: it stages 16-key K/V tiles in shared memory as f16, forms S = Q K^T and
// O += P V with `mma.sync.m16n8k16` (f16 inputs, f32 accumulation) fed by `ldmatrix`,
// and keeps the online-softmax state in registers.
//
// CPU reference = causal GQA attention in f64 over the f32 inputs (the hand kernel's
// device-test `reference` with `f16_inputs = false`): dot / sqrt(256), max-subtracted
// softmax, out = sum w v / sum w.
//
// SAFETY SHAPE (kernel-safety, #3522): the device kernels are safe Rust (no `unsafe`
// block, no raw pointer). Both things the hand kernel is built on are `unsafe` in
// cuda-oxide: every `mma`/`wmma` intrinsic is an `unsafe fn`, and `SharedArray` is a
// `static mut`. So this port has neither tensor cores nor shared staging. It is the
// scalar f32 form of the same online softmax:
//   * one WARP per (query row, query head); lane l owns the 8 contiguous head elements
//     l*8..l*8+8 of q, k, v and the output. Per key the lanes form partial dots and a
//     shfl.xor butterfly gives every lane the same full dot, so the softmax state
//     (m, l) is warp-uniform and no lane diverges.
//   * warps are numbered (row, head) with heads innermost, so the heads_per_kv warps
//     of one KV group are adjacent and read the same K/V rows (L1 reuse stands in for
//     the hand kernel's shared tile).
//   * every load is `get(..).first_chunk::<8>()` with a zero fallback, never an early
//     exit inside a shuffle. The only return is warp-uniform (padding warps).
// The math is full f32, so the port is closer to the reference than the hand kernel
// (whose f16 inputs cost ~1e-3 relative); parity is gated on the port only.
//
// Two exp variants, as for the other exp-bearing rows:
//   (A) prefill_attn_exp — `exp` (libm), the CPU reference's form.
//   (B) prefill_attn_ex2 — `exp2(x * log2 e)`, the hand PTX's `ex2.approx` form.
//
// Output: a human table on stdout plus one `apr-kernel-receipt/v1` JSON line per
// variant prefixed `RECEIPT `; `receipt.sh` adds host facts and writes
// evidence/kernels/gdn_prefill_flash_attention_256/<host>.json.

use cuda_core::{CudaContext, DeviceBuffer, IntoResult, LaunchConfig1D, sys};
use cuda_device::{DisjointSlice, LinearTiles, cuda_module, kernel, launch_bounds, launch_contract, thread, warp};
use std::sync::Arc;

/// Qwen3.5 attention head width.
const HD: usize = 256;
const LANES: usize = 32;
/// Head elements per lane.
const PER: usize = HD / LANES;
/// Oxide block: 4 warps.
const BLOCK: u32 = 128;
const WARPS_PER_BLOCK: usize = BLOCK as usize / LANES;
const FULL: u32 = 0xFFFF_FFFF;

/// Gates (contracts/kernel-receipt-v1.yaml).
const PARITY_COS: f64 = 0.9999;
const PARITY_MAXDIFF: f32 = 1e-3;
const TIMING_RATIO_MAX: f64 = 1.2;

/// (num_heads, num_kv_heads): the hand kernel's device-test shapes (27B, 9B, small).
/// 8 heads per KV group does not fit the hand kernel's 48 KiB of shared memory.
const SHAPES: [(usize, usize); 3] = [(8, 2), (16, 4), (24, 4)];
/// (rows, pos0) timed: a short chunk, a medium one, a long prompt chunk.
const TIMED: [(usize, usize); 3] = [(16, 0), (128, 0), (512, 0)];
/// (rows, pos0) checked: the hand kernel's device-test cases (partial last block,
/// cached prefix, many key tiles) plus the timed ones.
const PARITY: [(usize, usize); 6] = [(16, 0), (20, 3), (37, 29), (100, 500), (128, 0), (512, 0)];
/// Keys the fixture holds: max(pos0 + rows).
const MAX_KEYS: usize = 600;
const MAX_ROWS: usize = 512;

#[cuda_module]
mod kernels {
    use super::*;

    #[inline(always)]
    fn exp_of(x: f32, ex2: bool) -> f32 {
        if ex2 { (x * std::f32::consts::LOG2_E).exp2() } else { x.exp() }
    }

    /// This lane's 8 elements at `at`, zeros when out of range (never an early exit:
    /// the lane is inside a warp shuffle).
    #[inline(always)]
    fn load8(s: &[f32], at: usize) -> [f32; PER] {
        match s.get(at..).and_then(|r| r.first_chunk::<PER>()) {
            Some(c) => *c,
            None => [0.0; PER],
        }
    }

    /// One warp, one (row, head): online softmax over keys 0 ..= pos0 + row, then
    /// this lane's 8 normalised outputs.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn attend(
        t: u32,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        nh: usize,
        nkv: usize,
        rows: usize,
        pos0: usize,
        ex2: bool,
    ) -> Option<[f32; PER]> {
        let g = t as usize / LANES;
        let lane = t as usize % LANES;
        let row = g / nh;
        let h = g % nh;
        if row >= rows {
            return None; // warp-uniform: the grid's padding warps
        }
        let kvh = h / (nh / nkv);
        let kv_row = nkv * HD;
        let qv = load8(q, (row * nh + h) * HD + lane * PER);
        let mut m = f32::NEG_INFINITY;
        let mut l = 0.0f32;
        let mut acc = [0.0f32; PER];
        let end = pos0 + row + 1;
        let mut j = 0;
        while j < end {
            let at = j * kv_row + kvh * HD + lane * PER;
            let kj = load8(k, at);
            let mut d = 0.0f32;
            for i in 0..PER {
                d += qv[i] * kj[i];
            }
            d += warp::shuffle_xor_f32_sync(FULL, d, 16);
            d += warp::shuffle_xor_f32_sync(FULL, d, 8);
            d += warp::shuffle_xor_f32_sync(FULL, d, 4);
            d += warp::shuffle_xor_f32_sync(FULL, d, 2);
            d += warp::shuffle_xor_f32_sync(FULL, d, 1);
            let s = d / 16.0; // sqrt(256)
            // One exp per key: whichever of (old max, s) is smaller is scaled.
            let (corr, w) = if s > m { (exp_of(m - s, ex2), 1.0f32) } else { (1.0f32, exp_of(s - m, ex2)) };
            if s > m {
                m = s;
            }
            l = l * corr + w;
            let vj = load8(v, at);
            for i in 0..PER {
                acc[i] = acc[i] * corr + w * vj[i];
            }
            j += 1;
        }
        let inv = 1.0 / l; // key 0 is always attended, so l >= 1
        for a in &mut acc {
            *a *= inv;
        }
        Some(acc)
    }

    /// This lane's 8-float run of out, `[rows][nh][256]` = `[warp][256]`.
    #[inline(always)]
    fn write_out(o: Option<[f32; PER]>, run: Option<cuda_device::ThreadRunMut32<'_, f32, PER>>) {
        let (Some(o), Some(mut run)) = (o, run) else {
            return;
        };
        for k in 0..PER as u32 {
            if let Some(mut slot) = run.at(k) {
                slot.write(o[k as usize]);
            }
        }
    }

    /// (A) `exp` (libm).
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_attn_exp(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        nh: u32,
        nkv: u32,
        rows: u32,
        pos0: u32,
        mut out: DisjointSlice<f32, LinearTiles<PER>>,
    ) {
        let t = thread::index_1d_u32(launch_context);
        let o = attend(t.get(), q, k, v, nh as usize, nkv as usize, rows as usize, pos0 as usize, false);
        write_out(o, out.thread_run32(t));
    }

    /// (B) `exp2(x * log2 e)`, the hand PTX's form.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_attn_ex2(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        nh: u32,
        nkv: u32,
        rows: u32,
        pos0: u32,
        mut out: DisjointSlice<f32, LinearTiles<PER>>,
    ) {
        let t = thread::index_1d_u32(launch_context);
        let o = attend(t.get(), q, k, v, nh as usize, nkv as usize, rows as usize, pos0 as usize, true);
        write_out(o, out.thread_run32(t));
    }
}

/// Causal GQA attention in f64 over the f32 inputs: the hand kernel's device-test
/// `reference` (prefill_flash_attention.rs) with `f16_inputs = false`.
#[allow(clippy::too_many_arguments)]
fn cpu_prefill_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    rows: usize,
    pos0: usize,
    heads: usize,
    kv_heads: usize,
) -> Vec<f32> {
    let d = HD;
    let hpk = heads / kv_heads;
    let mut out = vec![0.0f32; rows * heads * d];
    let mut acc = vec![0.0f64; d];
    for row in 0..rows {
        let p = pos0 + row;
        for h in 0..heads {
            let g = h / hpk;
            let qv = &q[(row * heads + h) * d..(row * heads + h + 1) * d];
            let scores: Vec<f64> = (0..=p)
                .map(|j| {
                    let kv = &k[(j * kv_heads + g) * d..(j * kv_heads + g + 1) * d];
                    qv.iter().zip(kv).map(|(&a, &b)| f64::from(a) * f64::from(b)).sum::<f64>() / (d as f64).sqrt()
                })
                .collect();
            let mx = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let w: Vec<f64> = scores.iter().map(|s| (s - mx).exp()).collect();
            let sum: f64 = w.iter().sum();
            acc.iter_mut().for_each(|a| *a = 0.0);
            for (j, &wj) in w.iter().enumerate() {
                let vv = &v[(j * kv_heads + g) * d..(j * kv_heads + g + 1) * d];
                acc.iter_mut().zip(vv).for_each(|(a, &x)| *a += wj * f64::from(x));
            }
            for (i, a) in acc.iter().enumerate() {
                out[(row * heads + h) * d + i] = (a / sum) as f32;
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


/// Worst over the parity lengths.
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



/// q, k, v at the hand kernel's device-test magnitudes (O(1) after per-head RMSNorm,
/// spread wide enough that the softmax is not uniform).
struct Fixture {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    /// CPU reference per PARITY case.
    want: Vec<Vec<f32>>,
}

fn fixture(nh: usize, nkv: usize) -> Fixture {
    let mut rng = Lcg(0x3522_00FA + (nh * 16 + nkv) as u64 * 65_536);
    let q = rng.vec(MAX_ROWS * nh * HD, 1.5);
    let k = rng.vec(MAX_KEYS * nkv * HD, 1.5);
    let v = rng.vec(MAX_KEYS * nkv * HD, 1.0);
    let want = PARITY
        .iter()
        .map(|&(rows, pos0)| cpu_prefill_attention(&q, &k, &v, rows, pos0, nh, nkv))
        .collect();
    Fixture { q, k, v, want }
}

struct Dev {
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    out: DeviceBuffer<f32>,
}

fn upload(stream: &Arc<cuda_core::CudaStream>, f: &Fixture, nh: usize) -> Dev {
    Dev {
        q: DeviceBuffer::from_host(stream, &f.q).expect("q"),
        k: DeviceBuffer::from_host(stream, &f.k).expect("k"),
        v: DeviceBuffer::from_host(stream, &f.v).expect("v"),
        out: DeviceBuffer::<f32>::zeroed(stream, MAX_ROWS * nh * HD).expect("out"),
    }
}

/// The first `rows * nh * 256` outputs (the buffer is sized for MAX_ROWS).
fn download(stream: &Arc<cuda_core::CudaStream>, out: &DeviceBuffer<f32>, rows: usize, nh: usize) -> Vec<f32> {
    let mut o = out.to_host_vec(stream).expect("download out");
    o.truncate(rows * nh * HD);
    o
}

/// Every PARITY case against the CPU reference, then, when given, one TIMED case.
fn run_oxide(
    ctx: &Arc<CudaContext>,
    module: &kernels::LoadedModule,
    f: &Fixture,
    (nh, nkv): (usize, usize),
    ex2: bool,
    timed: Option<(usize, usize)>,
) -> Measured {
    let stream = ctx.new_stream().expect("stream");
    let mut d = upload(&stream, f, nh);
    let launch = |(rows, pos0): (usize, usize), d: &mut Dev| {
        let cfg = LaunchConfig1D::new((rows * nh).div_ceil(WARPS_PER_BLOCK) as u32, BLOCK, 0);
        let args = (nh as u32, nkv as u32, rows as u32, pos0 as u32);
        if ex2 {
            let p = module.prepare_prefill_attn_ex2(cfg).expect("prepare ex2");
            module
                .prefill_attn_ex2(&stream, &p, &d.q, &d.k, &d.v, args.0, args.1, args.2, args.3, &mut d.out)
                .expect("launch ex2");
        } else {
            let p = module.prepare_prefill_attn_exp(cfg).expect("prepare exp");
            module
                .prefill_attn_exp(&stream, &p, &d.q, &d.k, &d.v, args.0, args.1, args.2, args.3, &mut d.out)
                .expect("launch exp");
        }
    };
    let mut parity = Parity::new();
    for (i, &case) in PARITY.iter().enumerate() {
        launch(case, &mut d);
        parity.fold(&download(&stream, &d.out, case.0, nh), &f.want[i]);
    }
    let (us, eager_us) = match timed {
        Some(case) => (
            time_graph_us(&stream, || launch(case, &mut d)),
            time_eager_us(&stream, || launch(case, &mut d)),
        ),
        None => (0.0, 0.0),
    };
    Measured { parity, us, eager_us }
}

/// The hand PTX, loaded from its committed baseline and run over the same cases:
/// grid (ceil(rows / 16), nkv), block 32 * heads_per_kv, params (q, k, v, out, rows,
/// pos0). Its parity is printed, never gated (f16 inputs, see the header).
fn run_handptx(
    ctx: &Arc<CudaContext>,
    sm: &str,
    f: &Fixture,
    (nh, nkv): (usize, usize),
    timed: Option<(usize, usize)>,
) -> (Measured, u32) {
    let ptx_path = format!("baseline-ptx/gdn_prefill_flash_attention_256.h{nh}kv{nkv}.{sm}.ptx");
    let ptx = std::fs::read_to_string(&ptx_path).unwrap_or_else(|e| {
        eprintln!("missing hand-PTX baseline {ptx_path}: {e}");
        eprintln!(
            "regenerate: APR_BLESS_PTX=1 cargo test -p aprender-gpu --features cuda --lib gdn_prefill_flash_attention_ptx_golden"
        );
        std::process::exit(2);
    });
    let stream = ctx.new_stream().expect("stream");
    let module = ctx.load_module_from_ptx_src(&ptx).expect("load hand PTX");
    let func = module.load_function("gdn_prefill_flash_attention_256").expect("gdn_prefill_flash_attention_256");
    let regs = func.num_registers().expect("regs");
    let d = upload(&stream, f, nh);
    let launch = |(rows, pos0): (usize, usize)| {
        let mut ptrs = [d.q.cu_deviceptr(), d.k.cu_deviceptr(), d.v.cu_deviceptr(), d.out.cu_deviceptr()];
        let mut scalars = [rows as u32, pos0 as u32];
        let mut params: [*mut std::ffi::c_void; 6] = std::array::from_fn(|i| {
            if i < 4 { (&raw mut ptrs[i]).cast() } else { (&raw mut scalars[i - 4]).cast() }
        });
        let grid = (rows.div_ceil(16) as u32, nkv as u32, 1);
        let block = (32 * (nh / nkv) as u32, 1, 1);
        // SAFETY: the six params match the entry's four .u64 and two .u32 in order; q
        // and out hold MAX_ROWS >= rows rows of nh * 256, k/v MAX_KEYS >= pos0 + rows
        // rows of nkv * 256 (the head counts this PTX was baked with). Its K/V/P/Q
        // tiles are static shared memory declared by the PTX (<= 48 KiB).
        unsafe { cuda_core::launch_kernel_on_stream(&func, grid, block, 0, &stream, &mut params) }
            .expect("hand PTX launch");
    };
    let mut parity = Parity::new();
    for (i, &case) in PARITY.iter().enumerate() {
        launch(case);
        parity.fold(&download(&stream, &d.out, case.0, nh), &f.want[i]);
    }
    let (us, eager_us) = match timed {
        Some(case) => (time_graph_us(&stream, || launch(case)), time_eager_us(&stream, || launch(case))),
        None => (0.0, 0.0),
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

    println!("== #3522 cuda-oxide GDN prefill flash attention, head_dim {HD} ({sm}) ==");
    println!("   scalar f32: 1 warp per (row, head); parity (rows, pos0) {PARITY:?}, timed {TIMED:?}");

    let fixtures: Vec<Fixture> = SHAPES.iter().map(|&(nh, nkv)| fixture(nh, nkv)).collect();
    let mut all_ok = true;
    let mut timing_all_ok = true;
    let mut regs_hand = 0;
    for (f, &shape) in fixtures.iter().zip(&SHAPES) {
        let (h, regs) = run_handptx(&ctx, &sm, f, shape, None);
        regs_hand = regs_hand.max(regs);
        println!(
            "  parity handPTX h{}kv{} (f16 inputs, not gated): cos={:.7} maxdiff={:.3e} rel={:.3e} regs={regs}",
            shape.0, shape.1, h.parity.cos, h.parity.maxdiff, h.parity.rel,
        );
    }
    for ex2 in [false, true] {
        let variant = if ex2 { "ex2" } else { "exp" };
        let entry = format!("prefill_attn_{variant}");
        println!("\n  -- ({variant}) {entry}");
        println!("  shape   | rows,pos0 | oxide us | handPTX us | ratio | verdict | eager oxide/hand");
        let regs = oxide_registers(&ctx, &entry);
        let mut worst_ratio = 0.0f64;
        let mut worst = (0.0f64, 0.0f64, (0usize, 0usize), (0usize, 0usize));
        let mut worst_eager = 0.0f64;
        let mut parity = Parity::new();
        for (f, &shape) in fixtures.iter().zip(&SHAPES) {
            for case in TIMED {
                // Three rounds, alternating which kernel is timed first; the round with
                // the median ratio is the row (see GDN-DECISIONS.md, causal conv1d seq).
                let mut rounds: Vec<(Measured, Measured)> = (0..3)
                    .map(|round| {
                        if round % 2 == 0 {
                            let o = run_oxide(&ctx, &module, f, shape, ex2, Some(case));
                            (o, run_handptx(&ctx, &sm, f, shape, Some(case)).0)
                        } else {
                            let h = run_handptx(&ctx, &sm, f, shape, Some(case)).0;
                            (run_oxide(&ctx, &module, f, shape, ex2, Some(case)), h)
                        }
                    })
                    .collect();
                rounds.sort_by(|a, b| (a.0.us / a.1.us).total_cmp(&(b.0.us / b.1.us)));
                let (o, h) = rounds.swap_remove(1);
                let ratio = o.us / h.us;
                let ok = ratio <= TIMING_RATIO_MAX;
                let eager_ratio = o.eager_us / h.eager_us;
                println!(
                    "  h{:>2}kv{} | {:>4},{:<4} | {:>8.3} | {:>10.3} | {ratio:.3} | {:>7} | {:.3}/{:.3} = {eager_ratio:.3}  parity cos={:.7} maxdiff={:.3e} rel={:.3e}",
                    shape.0,
                    shape.1,
                    case.0,
                    case.1,
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
                    worst = (o.us, h.us, shape, case);
                }
                worst_eager = worst_eager.max(eager_ratio);
                parity.cos = parity.cos.min(o.parity.cos);
                parity.maxdiff = parity.maxdiff.max(o.parity.maxdiff);
                parity.rel = parity.rel.max(o.parity.rel);
            }
        }
        let parity_ok = parity.pass();
        let timing_ok = worst_ratio <= TIMING_RATIO_MAX;
        all_ok &= parity_ok;
        timing_all_ok &= timing_ok;
        let reg = |r: Option<u32>| r.map_or("null".to_string(), |r| r.to_string());
        println!(
            "RECEIPT {{\"schema\":\"apr-kernel-receipt/v1\",\"kernel\":\"gdn_prefill_flash_attention_256\",\"variant\":\"{variant}\",\"entry\":\"{entry}\",\"authoring\":\"oxide\",\"cc\":\"{sm}\",\"head_dim\":{HD},\"shapes\":{:?},\"parity_cases\":{:?},\"timed_cases\":{:?},\"parity\":{{\"cos_min\":{:.9},\"maxdiff_max\":{:.3e},\"rel_max\":{:.3e},\"cos_threshold\":{PARITY_COS},\"maxdiff_threshold\":{PARITY_MAXDIFF},\"pass\":{parity_ok}}},\"timing\":{{\"oxide_us\":{:.3},\"handptx_us\":{:.3},\"ratio\":{worst_ratio:.4},\"worst_shape\":[{},{}],\"worst_case\":[{},{}],\"ratio_max\":{TIMING_RATIO_MAX},\"method\":\"cuda-graph-100\",\"eager_ratio_max\":{worst_eager:.4},\"pass\":{timing_ok}}},\"register_budget\":{{\"oxide\":{},\"handptx\":{regs_hand}}}}}",
            SHAPES.map(|(a, b)| [a, b]),
            PARITY.map(|(a, b)| [a, b]),
            TIMED.map(|(a, b)| [a, b]),
            parity.cos,
            parity.maxdiff,
            parity.rel,
            worst.0,
            worst.1,
            worst.2.0,
            worst.2.1,
            worst.3.0,
            worst.3.1,
            reg(regs),
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
    println!("#3522 PREFILL FLASH ATTENTION DONE");
}
