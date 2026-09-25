// #3522 OXIDE-001 — pure-Rust cuda-oxide port of the GDN single-query DECODE
// ATTENTION over a resident KV cache, head_dim 256 (Qwen3.5's gated full-attention
// layers).
//
// Target = hand-PTX `DecodeAttention256Kernel`
// (crates/aprender-gpu/src/kernels/gdn/decode_attention.rs, entry
// `gdn_decode_attention`): grid (num_heads), block 256; head counts and head_dim baked
// into the PTX; params (q, k_cache, v_cache, out, seq_len). k/v caches are
// `[max_len][num_kv_heads * 256]`. The hand kernel scores up to 4096 positions per
// pass into shared memory (one thread per position, a full 256-wide scalar dot),
// block-reduces max and sum, then thread d accumulates output element d over the
// pass's positions.
//
// CPU reference = the verbatim port of `forward_attention`'s attention block
// (forward_qwen35.rs:1052) that the hand kernel's device tests use: dot / sqrt(256),
// max-subtracted softmax, `out += w * v`, positions ascending.
//
// SAFETY SHAPE (kernel-safety, #3522): the device kernels are safe Rust (no `unsafe`
// block, no raw pointer). cuda-oxide's `SharedArray` is a `static mut` whose every
// access is `unsafe`, so the hand kernel's shared scores buffer and block reductions
// are out. This port is instead the split-K ("flash decoding") form, which needs no
// shared memory:
//   * CHUNK KERNEL: one WARP per (head, chunk of positions). Lane l owns the 8
//     contiguous head elements l*8..l*8+8 of q, k, v and the output. Per position the
//     lanes form partial dots and a shfl.xor butterfly gives every lane the same full
//     dot, so the softmax state is warp-uniform and no lane diverges. The warp keeps an
//     online softmax (running max m, sum l, acc[8]) and writes its unnormalised
//     partial: acc to `[warp][256]`, (m, l) to its lanes' own 2-float runs.
//   * COMBINE KERNEL: grid (num_heads), block 256, thread d of head h folds the
//     chunks: M = max m_c, L = sum l_c * exp(m_c - M), out = sum acc_c[d] *
//     exp(m_c - M) / L. Every chunk is non-empty (the host picks nc = ceil(seq /
//     chunk_len)), so L >= 1.
//   * Every load is `get(..).first_chunk::<8>()` with a zero fallback instead of an
//     early exit, so a lane can never leave a shuffle its warp is still in. The only
//     returns are warp-uniform (a whole warp past the last (head, chunk)).
//
// Both kernels are timed together as one "launch" against the one hand launch.
//
// Two exp variants, as for the other exp-bearing rows (both kernels use it):
//   (A) decode_attn_{chunk,combine}_exp — `exp` (libm), the CPU reference's form.
//   (B) decode_attn_{chunk,combine}_ex2 — `exp2(x * log2 e)`, the hand PTX's
//       `ex2.approx` form.
//
// Output: a human table on stdout plus one `apr-kernel-receipt/v1` JSON line per
// variant prefixed `RECEIPT `; `receipt.sh` adds host facts and writes
// evidence/kernels/gdn_decode_attention/<host>.json.

use cuda_core::{CudaContext, DeviceBuffer, IntoResult, LaunchConfig1D, sys};
use cuda_device::{DisjointSlice, LinearTiles, cuda_module, kernel, launch_bounds, launch_contract, thread, warp};
use std::sync::Arc;

/// Qwen3.5 attention head width.
const HD: usize = 256;
const LANES: usize = 32;
/// Head elements one lane owns.
const PER: usize = HD / LANES;
/// Chunk-kernel block: 4 warps, each its own (head, chunk).
const BLOCK_CHUNK: u32 = 128;
const WARPS_PER_BLOCK: usize = BLOCK_CHUNK as usize / LANES;
/// Combine-kernel block and the hand kernel's block.
const BLOCK: u32 = 256;
const FULL: u32 = 0xFFFF_FFFF;

// F-OXIDE-ROPE-PARITY-001 form, as #3522's kernel-parity shape states it.
const PARITY_COS: f64 = 0.9999;
const PARITY_MAXDIFF: f32 = 1e-3;
const TIMING_RATIO_MAX: f64 = 1.2;

/// (num_heads, num_kv_heads): 0.8B/2B (16/2), 4B/9B (16/4), 27B (24/4). Each has a
/// committed hand-PTX baseline, since the hand kernel bakes the head counts in.
const SHAPES: [(usize, usize); 3] = [(16, 2), (16, 4), (24, 4)];
/// Timed context lengths. 4101 > 4096 takes the hand kernel through two passes.
const SEQ_TIMED: [usize; 3] = [128, 1024, 4101];
/// Parity lengths: one position, a few, a ragged last chunk, and every timed length.
const SEQ_PARITY: [usize; 6] = [1, 5, 37, 128, 1024, 4101];
const MAX_LEN: usize = 4101;

/// Positions per chunk: at least 16, and at most ~32 chunks per head, so short
/// contexts still spread over the SMs and long ones do not flood the combine.
const fn chunk_len(seq_len: usize) -> usize {
    let c = seq_len.div_ceil(32);
    if c < 16 { 16 } else { c }
}

const fn n_chunks(seq_len: usize) -> usize {
    seq_len.div_ceil(chunk_len(seq_len))
}

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

    /// One warp, one (head, chunk): online softmax over the chunk's positions.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn chunk(
        t: u32,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        nh: usize,
        nkv: usize,
        seq_len: usize,
        chunk_len: usize,
        nc: usize,
        ex2: bool,
    ) -> Option<(f32, f32, [f32; PER])> {
        let g = t as usize / LANES;
        let lane = t as usize % LANES;
        let h = g / nc;
        let c = g % nc;
        if h >= nh {
            return None; // warp-uniform: the grid's padding warps
        }
        let kvh = h / (nh / nkv);
        let row = nkv * HD;
        let qv = load8(q, h * HD + lane * PER);
        let end = if (c + 1) * chunk_len < seq_len { (c + 1) * chunk_len } else { seq_len };
        let mut m = f32::NEG_INFINITY;
        let mut l = 0.0f32;
        let mut acc = [0.0f32; PER];
        let mut p = c * chunk_len;
        while p < end {
            let at = p * row + kvh * HD + lane * PER;
            let kv = load8(k_cache, at);
            let mut d = 0.0f32;
            for i in 0..PER {
                d += qv[i] * kv[i];
            }
            d += warp::shuffle_xor_f32_sync(FULL, d, 16);
            d += warp::shuffle_xor_f32_sync(FULL, d, 8);
            d += warp::shuffle_xor_f32_sync(FULL, d, 4);
            d += warp::shuffle_xor_f32_sync(FULL, d, 2);
            d += warp::shuffle_xor_f32_sync(FULL, d, 1);
            let s = d / 16.0; // sqrt(256)
            // One exp per position: whichever of (old max, s) is smaller is scaled.
            let (corr, w) = if s > m { (exp_of(m - s, ex2), 1.0f32) } else { (1.0f32, exp_of(s - m, ex2)) };
            if s > m {
                m = s;
            }
            l = l * corr + w;
            let vv = load8(v_cache, at);
            for i in 0..PER {
                acc[i] = acc[i] * corr + w * vv[i];
            }
            p += 1;
        }
        Some((m, l, acc))
    }

    /// The chunk kernel's writes: acc to this lane's 8-float run of `[warp][256]`,
    /// (m, l) to this lane's own 2-float run (the combine reads lane 0's).
    #[inline(always)]
    fn write_partial(
        partial: Option<(f32, f32, [f32; PER])>,
        acc_run: Option<cuda_device::ThreadRunMut32<'_, f32, PER>>,
        ml_run: Option<cuda_device::ThreadRunMut32<'_, f32, 2>>,
    ) {
        let (Some((m, l, acc)), Some(mut a), Some(mut ml)) = (partial, acc_run, ml_run) else {
            return;
        };
        for k in 0..PER as u32 {
            if let Some(mut slot) = a.at(k) {
                slot.write(acc[k as usize]);
            }
        }
        if let Some(mut slot) = ml.at(0) {
            slot.write(m);
        }
        if let Some(mut slot) = ml.at(1) {
            slot.write(l);
        }
    }

    /// Thread d of head h: fold the head's `nc` chunk partials into output d.
    #[inline(always)]
    fn combine(t: usize, acc: &[f32], ml: &[f32], nc: usize, ex2: bool) -> f32 {
        let h = t / HD;
        let d = t % HD;
        let mut mx = f32::NEG_INFINITY;
        for c in 0..nc {
            let mc = ml.get((h * nc + c) * LANES * 2).copied().unwrap_or(f32::NEG_INFINITY);
            if mc > mx {
                mx = mc;
            }
        }
        let mut l = 0.0f32;
        let mut o = 0.0f32;
        for c in 0..nc {
            let g = h * nc + c;
            let (Some(&mc), Some(&lc), Some(&ac)) =
                (ml.get(g * LANES * 2), ml.get(g * LANES * 2 + 1), acc.get(g * HD + d))
            else {
                break;
            };
            let sc = exp_of(mc - mx, ex2);
            l += lc * sc;
            o += ac * sc;
        }
        o / l
    }

    /// (A) chunk partials, `exp` (libm).
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
    #[allow(clippy::too_many_arguments)]
    pub fn decode_attn_chunk_exp(
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        nh: u32,
        nkv: u32,
        seq_len: u32,
        chunk_len: u32,
        nc: u32,
        mut acc: DisjointSlice<f32, LinearTiles<PER>>,
        mut ml: DisjointSlice<f32, LinearTiles<2>>,
    ) {
        let t = thread::index_1d_u32(launch_context).get();
        let partial = chunk(
            t, q, k_cache, v_cache, nh as usize, nkv as usize, seq_len as usize, chunk_len as usize, nc as usize, false,
        );
        write_partial(
            partial,
            acc.thread_run32(thread::index_1d_u32(launch_context)),
            ml.thread_run32(thread::index_1d_u32(launch_context)),
        );
    }

    /// (B) chunk partials, `exp2(x * log2 e)`, the hand PTX's form.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
    #[allow(clippy::too_many_arguments)]
    pub fn decode_attn_chunk_ex2(
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        nh: u32,
        nkv: u32,
        seq_len: u32,
        chunk_len: u32,
        nc: u32,
        mut acc: DisjointSlice<f32, LinearTiles<PER>>,
        mut ml: DisjointSlice<f32, LinearTiles<2>>,
    ) {
        let t = thread::index_1d_u32(launch_context).get();
        let partial = chunk(
            t, q, k_cache, v_cache, nh as usize, nkv as usize, seq_len as usize, chunk_len as usize, nc as usize, true,
        );
        write_partial(
            partial,
            acc.thread_run32(thread::index_1d_u32(launch_context)),
            ml.thread_run32(thread::index_1d_u32(launch_context)),
        );
    }

    /// (A) combine, `exp` (libm).
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub fn decode_attn_combine_exp(acc: &[f32], ml: &[f32], nc: u32, mut out: DisjointSlice<f32, LinearTiles<1>>) {
        let t = thread::index_1d_u32(launch_context);
        let v = combine(t.get() as usize, acc, ml, nc as usize, false);
        if let Some(mut run) = out.thread_run32(t)
            && let Some(mut slot) = run.at(0)
        {
            slot.write(v);
        }
    }

    /// (B) combine, `exp2(x * log2 e)`.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub fn decode_attn_combine_ex2(acc: &[f32], ml: &[f32], nc: u32, mut out: DisjointSlice<f32, LinearTiles<1>>) {
        let t = thread::index_1d_u32(launch_context);
        let v = combine(t.get() as usize, acc, ml, nc as usize, true);
        if let Some(mut run) = out.thread_run32(t)
            && let Some(mut slot) = run.at(0)
        {
            slot.write(v);
        }
    }
}

/// Verbatim port of `aprender-serve`'s `gguf::ops::softmax` (ops.rs:348).
fn softmax(logits: &mut [f32]) {
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in logits.iter_mut() {
        *x = (*x - max_val).exp();
        sum += *x;
    }
    let inv_sum = 1.0 / sum;
    for x in logits.iter_mut() {
        *x *= inv_sum;
    }
}

/// Verbatim port of the attention block of `forward_attention`
/// (forward_qwen35.rs:1052), as the hand kernel's device tests use it.
fn cpu_decode_attention(q: &[f32], k_cache: &[f32], v_cache: &[f32], seq_len: usize, nh: usize, nkv: usize) -> Vec<f32> {
    let head_dim = HD;
    let group_size = nh / nkv;
    let mut out = vec![0.0; nh * head_dim];
    for h in 0..nh {
        let kv_h = h / group_size;
        let q_h = &q[h * head_dim..(h + 1) * head_dim];
        let mut scores = vec![0.0; seq_len];
        for (p, score) in scores.iter_mut().enumerate() {
            let mut dot = 0.0;
            let k_p = &k_cache[p * (nkv * head_dim) + kv_h * head_dim..p * (nkv * head_dim) + (kv_h + 1) * head_dim];
            for i in 0..head_dim {
                dot += q_h[i] * k_p[i];
            }
            *score = dot / (head_dim as f32).sqrt();
        }
        softmax(&mut scores);
        let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
        for (p, &w) in scores.iter().enumerate() {
            let v_p = &v_cache[p * (nkv * head_dim) + kv_h * head_dim..p * (nkv * head_dim) + (kv_h + 1) * head_dim];
            for i in 0..head_dim {
                out_h[i] += w * v_p[i];
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



/// q, and a KV cache whose rows differ in scale (0.5 .. 2.05, period 32) so a kernel
/// reading the wrong position or KV head lands somewhere else. q at unit scale keeps
/// the softmax far from uniform.
struct Fixture {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    /// CPU reference per SEQ_PARITY length.
    want: Vec<Vec<f32>>,
}

fn fixture(nh: usize, nkv: usize) -> Fixture {
    let mut rng = Lcg(0x3522_00DA + (nh * 16 + nkv) as u64 * 65_536);
    let q = rng.vec(nh * HD, 1.0);
    let row = nkv * HD;
    let (mut k, mut v) = (Vec::with_capacity(MAX_LEN * row), Vec::with_capacity(MAX_LEN * row));
    for p in 0..MAX_LEN {
        let scale = 0.05f32.mul_add((p % 32) as f32, 0.5);
        k.extend(rng.vec(row, scale));
        v.extend(rng.vec(row, scale));
    }
    let want = SEQ_PARITY.iter().map(|&s| cpu_decode_attention(&q, &k, &v, s, nh, nkv)).collect();
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
        out: DeviceBuffer::<f32>::zeroed(stream, nh * HD).expect("out"),
    }
}

/// Every SEQ_PARITY length against the CPU port, then, when `perf`, `seq_timed`
/// timed. One "launch" is the chunk kernel plus the combine kernel.
fn run_oxide(
    ctx: &Arc<CudaContext>,
    module: &kernels::LoadedModule,
    f: &Fixture,
    (nh, nkv): (usize, usize),
    ex2: bool,
    seq_timed: Option<usize>,
) -> Measured {
    let stream = ctx.new_stream().expect("stream");
    let mut d = upload(&stream, f, nh);
    let nc_max = n_chunks(MAX_LEN);
    let mut acc = DeviceBuffer::<f32>::zeroed(&stream, nh * nc_max * HD).expect("acc");
    let mut ml = DeviceBuffer::<f32>::zeroed(&stream, nh * nc_max * LANES * 2).expect("ml");
    let launch = |seq: usize, d: &mut Dev, acc: &mut DeviceBuffer<f32>, ml: &mut DeviceBuffer<f32>| {
        let (cl, nc) = (chunk_len(seq), n_chunks(seq));
        let warps = nh * nc;
        let cfg_a = LaunchConfig1D::new(warps.div_ceil(WARPS_PER_BLOCK) as u32, BLOCK_CHUNK, 0);
        let cfg_b = LaunchConfig1D::new(nh as u32, BLOCK, 0);
        let args = (nh as u32, nkv as u32, seq as u32, cl as u32, nc as u32);
        if ex2 {
            let p = module.prepare_decode_attn_chunk_ex2(cfg_a).expect("prepare chunk ex2");
            module
                .decode_attn_chunk_ex2(&stream, &p, &d.q, &d.k, &d.v, args.0, args.1, args.2, args.3, args.4, acc, ml)
                .expect("launch chunk ex2");
            let p = module.prepare_decode_attn_combine_ex2(cfg_b).expect("prepare combine ex2");
            module
                .decode_attn_combine_ex2(&stream, &p, acc, ml, args.4, &mut d.out)
                .expect("launch combine ex2");
        } else {
            let p = module.prepare_decode_attn_chunk_exp(cfg_a).expect("prepare chunk exp");
            module
                .decode_attn_chunk_exp(&stream, &p, &d.q, &d.k, &d.v, args.0, args.1, args.2, args.3, args.4, acc, ml)
                .expect("launch chunk exp");
            let p = module.prepare_decode_attn_combine_exp(cfg_b).expect("prepare combine exp");
            module
                .decode_attn_combine_exp(&stream, &p, acc, ml, args.4, &mut d.out)
                .expect("launch combine exp");
        }
    };
    let mut parity = Parity::new();
    for (i, &seq) in SEQ_PARITY.iter().enumerate() {
        launch(seq, &mut d, &mut acc, &mut ml);
        parity.fold(&d.out.to_host_vec(&stream).expect("download out"), &f.want[i]);
    }
    let (us, eager_us) = match seq_timed {
        Some(seq) => (
            time_graph_us(&stream, || launch(seq, &mut d, &mut acc, &mut ml)),
            time_eager_us(&stream, || launch(seq, &mut d, &mut acc, &mut ml)),
        ),
        None => (0.0, 0.0),
    };
    Measured { parity, us, eager_us }
}

/// The hand PTX, loaded from its committed baseline and run over the same lengths:
/// grid (nh), block 256, params (q, k_cache, v_cache, out, seq_len).
fn run_handptx(
    ctx: &Arc<CudaContext>,
    sm: &str,
    f: &Fixture,
    (nh, nkv): (usize, usize),
    seq_timed: Option<usize>,
) -> (Measured, u32) {
    let ptx_path = format!("baseline-ptx/gdn_decode_attention.h{nh}kv{nkv}.{sm}.ptx");
    let ptx = std::fs::read_to_string(&ptx_path).unwrap_or_else(|e| {
        eprintln!("missing hand-PTX baseline {ptx_path}: {e}");
        eprintln!("regenerate: APR_BLESS_PTX=1 cargo test -p aprender-gpu --features cuda --lib gdn_decode_attention_ptx_golden");
        std::process::exit(2);
    });
    let stream = ctx.new_stream().expect("stream");
    let module = ctx.load_module_from_ptx_src(&ptx).expect("load hand PTX");
    let func = module.load_function("gdn_decode_attention").expect("gdn_decode_attention");
    let regs = func.num_registers().expect("regs");
    let d = upload(&stream, f, nh);
    let launch = |seq: usize| {
        let mut ptrs = [d.q.cu_deviceptr(), d.k.cu_deviceptr(), d.v.cu_deviceptr(), d.out.cu_deviceptr()];
        let mut seq_len = seq as u32;
        let mut params: [*mut std::ffi::c_void; 5] = std::array::from_fn(|i| {
            if i < 4 { (&raw mut ptrs[i]).cast() } else { (&raw mut seq_len).cast() }
        });
        // SAFETY: the five params match the entry's four .u64 and one .u32 in order;
        // q holds nh * 256, k/v MAX_LEN >= seq_len rows of nkv * 256 (the head counts
        // this PTX was baked with), out nh * 256, and an nh-block launch of 256
        // threads touches only those. The scores buffer is static shared memory
        // declared by the PTX.
        unsafe {
            cuda_core::launch_kernel_on_stream(&func, (nh as u32, 1, 1), (BLOCK, 1, 1), 0, &stream, &mut params)
        }
        .expect("hand PTX launch");
    };
    let mut parity = Parity::new();
    for (i, &seq) in SEQ_PARITY.iter().enumerate() {
        launch(seq);
        parity.fold(&d.out.to_host_vec(&stream).expect("download out"), &f.want[i]);
    }
    let (us, eager_us) = match seq_timed {
        Some(seq) => (time_graph_us(&stream, || launch(seq)), time_eager_us(&stream, || launch(seq))),
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

    println!("== #3522 cuda-oxide GDN decode attention, head_dim {HD} ({sm}) ==");
    println!("   split-K: 1 warp per (head, chunk) + combine; parity seq {SEQ_PARITY:?}, timed seq {SEQ_TIMED:?}");

    let fixtures: Vec<Fixture> = SHAPES.iter().map(|&(nh, nkv)| fixture(nh, nkv)).collect();
    let mut all_ok = true;
    let mut timing_all_ok = true;
    let mut regs_hand = 0;
    for (f, &shape) in fixtures.iter().zip(&SHAPES) {
        let (h, regs) = run_handptx(&ctx, &sm, f, shape, None);
        regs_hand = regs;
        println!(
            "  parity handPTX h{}kv{}: cos={:.7} maxdiff={:.3e} rel={:.3e} {} regs={regs}",
            shape.0,
            shape.1,
            h.parity.cos,
            h.parity.maxdiff,
            h.parity.rel,
            if h.parity.pass() { "PASS" } else { "FAIL" }
        );
    }
    for ex2 in [false, true] {
        let variant = if ex2 { "ex2" } else { "exp" };
        let entry = format!("decode_attn_chunk_{variant}");
        let combine = format!("decode_attn_combine_{variant}");
        println!("\n  -- ({variant}) {entry} + {combine}");
        println!("  shape   |  seq | oxide us | handPTX us | ratio | verdict | eager oxide/hand");
        let regs = oxide_registers(&ctx, &entry);
        let regs_combine = oxide_registers(&ctx, &combine);
        let mut worst_ratio = 0.0f64;
        let mut worst = (0.0f64, 0.0f64, (0usize, 0usize), 0usize);
        let mut worst_eager = 0.0f64;
        let mut parity = Parity::new();
        for (f, &shape) in fixtures.iter().zip(&SHAPES) {
            for seq in SEQ_TIMED {
                // Three rounds, alternating which kernel is timed first; the round with
                // the median ratio is the row (see GDN-DECISIONS.md, causal conv1d seq).
                let mut rounds: Vec<(Measured, Measured)> = (0..3)
                    .map(|round| {
                        if round % 2 == 0 {
                            let o = run_oxide(&ctx, &module, f, shape, ex2, Some(seq));
                            (o, run_handptx(&ctx, &sm, f, shape, Some(seq)).0)
                        } else {
                            let h = run_handptx(&ctx, &sm, f, shape, Some(seq)).0;
                            (run_oxide(&ctx, &module, f, shape, ex2, Some(seq)), h)
                        }
                    })
                    .collect();
                rounds.sort_by(|a, b| (a.0.us / a.1.us).total_cmp(&(b.0.us / b.1.us)));
                let (o, h) = rounds.swap_remove(1);
                let ratio = o.us / h.us;
                let ok = ratio <= TIMING_RATIO_MAX;
                let eager_ratio = o.eager_us / h.eager_us;
                println!(
                    "  h{:>2}kv{} | {seq:>4} | {:>8.3} | {:>10.3} | {ratio:.3} | {:>7} | {:.3}/{:.3} = {eager_ratio:.3}  parity cos={:.7} maxdiff={:.3e} rel={:.3e}",
                    shape.0,
                    shape.1,
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
                    worst = (o.us, h.us, shape, seq);
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
            "RECEIPT {{\"schema\":\"apr-kernel-receipt/v1\",\"kernel\":\"gdn_decode_attention\",\"variant\":\"{variant}\",\"entry\":\"{entry}\",\"entries\":[\"{entry}\",\"{combine}\"],\"authoring\":\"oxide\",\"cc\":\"{sm}\",\"head_dim\":{HD},\"shapes\":{:?},\"seq_parity\":{SEQ_PARITY:?},\"seq_timed\":{SEQ_TIMED:?},\"parity\":{{\"cos_min\":{:.9},\"maxdiff_max\":{:.3e},\"rel_max\":{:.3e},\"cos_threshold\":{PARITY_COS},\"maxdiff_threshold\":{PARITY_MAXDIFF},\"pass\":{parity_ok}}},\"timing\":{{\"oxide_us\":{:.3},\"handptx_us\":{:.3},\"ratio\":{worst_ratio:.4},\"worst_shape\":[{},{}],\"worst_seq\":{},\"ratio_max\":{TIMING_RATIO_MAX},\"method\":\"cuda-graph-100\",\"eager_ratio_max\":{worst_eager:.4},\"pass\":{timing_ok}}},\"register_budget\":{{\"oxide\":{},\"oxide_combine\":{},\"handptx\":{regs_hand}}}}}",
            SHAPES.map(|(a, b)| [a, b]),
            parity.cos,
            parity.maxdiff,
            parity.rel,
            worst.0,
            worst.1,
            worst.2.0,
            worst.2.1,
            worst.3,
            reg(regs),
            reg(regs_combine),
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
    println!("#3522 DECODE ATTENTION DONE");
}
