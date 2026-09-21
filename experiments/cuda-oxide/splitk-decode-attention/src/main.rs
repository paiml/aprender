// PMAT-3725 / aprender#3725 — cuda-oxide port of the split-K (flash-decoding) decode
// attention for Qwen3.5's full-attention layers (head_dim = 256), f32 and f16 KV.
//
// Source of record for the PRODUCTION kernel: the builder-PTX pair
//   crates/aprender-gpu/src/kernels/gdn/decode_attention_splitk.rs
//   (DecodeAttentionSplitKKernel + DecodeAttentionSplitKReduceKernel).
// Cop ruling on #3725 (2026-09-21): the production kernel stays builder-PTX until
// OXIDE-001 O-1 lands a path that ships oxide PTX; this port exists to measure
// parity and an A/B against it on sm_89 and sm_121.
//
// Same algorithm, same layout, same launch shape:
//   kernel A: grid (num_kv_heads, n_splits), block 256 = 8 warps. A block serves all
//     G query heads of one KV head; warp w takes positions start+w, start+w+8, …;
//     lane l holds row elements l, l+32, …, l+224; per-(warp, head) online softmax
//     in registers; the 8 warps merge through shared memory; one un-normalised
//     partial (m, l, acc[256]) per (query head, split).
//   kernel B: grid num_heads, block 256, one thread per element: the log-sum-exp
//     merge  out = Σ e^(m_s−M) acc_s / Σ e^(m_s−M) l_s.
//
// Checked here, per host:
//   parity: oxide vs the CPU twin (same split/warp/merge order) and vs an f64
//     reference, and vs the builder kernel on the same inputs, at several seq_len
//     including a forced many-split plan;
//   A/B:    GPU-event µs per call, oxide pair vs builder pair (PTX emitted by
//     `gdn_splitk_rungs --emit-ptx DIR`), same buffers, same stream.
//
// Run (lambda: put /usr/local/cuda-13.3/bin first on PATH):
//   cargo oxide run splitk_decode_attention_oxide -- --builder-ptx DIR [--out receipt.json]

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, CudaFunction, CudaStream, DeviceBuffer};
use cuda_device::shared::SharedArray;
use cuda_device::{convert, kernel, thread, warp};
use cuda_host::cuda_module;

const HEAD_DIM: usize = 256;
const PER_LANE: usize = HEAD_DIM / 32;
const WARPS: usize = 8;
const BLOCK: u32 = 256;
/// Planner constants — must equal SPLITK_DEFAULT_TARGET_SPLITS / SPLITK_MIN_CHUNK.
const TARGET_SPLITS: u32 = 256;
const MIN_CHUNK: u32 = 64;

#[cuda_module]
mod kernels {
    use super::*;

    /// One warp's butterfly sum; every lane ends with the total.
    #[inline(always)]
    fn warp_sum(x: f32) -> f32 {
        let mut v = x;
        v += warp::shuffle_xor_f32_sync(0xFFFF_FFFF, v, 16);
        v += warp::shuffle_xor_f32_sync(0xFFFF_FFFF, v, 8);
        v += warp::shuffle_xor_f32_sync(0xFFFF_FFFF, v, 4);
        v += warp::shuffle_xor_f32_sync(0xFFFF_FFFF, v, 2);
        v += warp::shuffle_xor_f32_sync(0xFFFF_FFFF, v, 1);
        v
    }

    /// Fold one position into every query head's online softmax — the per-position
    /// update of a warp (`kk`, `vv` are this lane's K and V elements).
    #[inline(always)]
    fn fold_position<const G: usize>(
        m: &mut [f32; G],
        l: &mut [f32; G],
        acc: &mut [[f32; PER_LANE]; G],
        qr: &[[f32; PER_LANE]; G],
        kk: &[f32; PER_LANE],
        vv: &[f32; PER_LANE],
    ) {
        let mut g = 0;
        while g < G {
            let mut part = 0.0f32;
            let mut e = 0;
            while e < PER_LANE {
                part += qr[g][e] * kk[e];
                e += 1;
            }
            // divided by sqrt(head_dim) = 16, as the CPU reference does
            let score = warp_sum(part) / 16.0;
            let new_m = if score > m[g] { score } else { m[g] };
            let corr = (m[g] - new_m).exp();
            let w = (score - new_m).exp();
            l[g] = l[g] * corr + w;
            let mut e = 0;
            while e < PER_LANE {
                acc[g][e] = acc[g][e] * corr + w * vv[e];
                e += 1;
            }
            m[g] = new_m;
            g += 1;
        }
    }

    /// Where one block's merged partial for one query head goes.
    struct Slot {
        pacc: *mut f32,
        pml: *mut f32,
        index: usize,
    }

    /// Merge the 8 warps' state for ONE query head through shared memory (`ml` is
    /// `[WARPS][2]`, `accs` `[WARPS][HEAD_DIM]`) and write the block's partial.
    #[inline(always)]
    unsafe fn merge_head(m: f32, l: f32, acc: &[f32; PER_LANE], ml: *mut f32, accs: *mut f32, out: &Slot) {
        unsafe {
            let tid = thread::threadIdx_x() as usize;
            let lane = tid & 31;
            let wid = tid >> 5;
            if lane == 0 {
                *ml.add(wid * 2) = m;
                *ml.add(wid * 2 + 1) = l;
            }
            let mut e = 0;
            while e < PER_LANE {
                *accs.add(wid * HEAD_DIM + lane + 32 * e) = acc[e];
                e += 1;
            }
            thread::sync_threads();

            let mut bm = f32::NEG_INFINITY;
            let mut w = 0;
            while w < WARPS {
                bm = bm.max(*ml.add(w * 2));
                w += 1;
            }
            let mut bl = 0.0f32;
            let mut ba = 0.0f32;
            let mut w = 0;
            while w < WARPS {
                let scale = (*ml.add(w * 2) - bm).exp();
                bl += *ml.add(w * 2 + 1) * scale;
                ba += *accs.add(w * HEAD_DIM + tid) * scale;
                w += 1;
            }
            *out.pacc.add(out.index * HEAD_DIM + tid) = ba;
            if tid == 0 {
                *out.pml.add(out.index * 2) = bm;
                *out.pml.add(out.index * 2 + 1) = bl;
            }
            thread::sync_threads();
        }
    }

    /// The shared body of the two partial kernels, generic over the stored element.
    /// `ml` / `acc` are the kernel's shared arrays (`[WARPS][2]`, `[WARPS][HEAD_DIM]`).
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn partial_body<const G: usize, T: Copy, L: Fn(T) -> f32>(
        q: &[f32],
        k: &[T],
        v: &[T],
        pacc: *mut f32,
        pml: *mut f32,
        seq_len: u32,
        chunk: u32,
        n_splits: u32,
        num_kv_heads: u32,
        ml: *mut f32,
        accs: *mut f32,
        load: L,
    ) {
        unsafe {
            let kv_h = thread::blockIdx_x() as usize;
            let split = thread::blockIdx_y();
            let lane = thread::threadIdx_x() as usize & 31;
            let start = split * chunk;
            if start >= seq_len {
                return;
            }
            let end = seq_len.min(start + chunk);
            let row = num_kv_heads as usize * HEAD_DIM;
            let first = kv_h * G;

            let mut qr = [[0.0f32; PER_LANE]; G];
            let mut i = 0;
            while i < G * PER_LANE {
                qr[i / PER_LANE][i % PER_LANE] = q[(first + i / PER_LANE) * HEAD_DIM + lane + 32 * (i % PER_LANE)];
                i += 1;
            }
            let mut m = [f32::NEG_INFINITY; G];
            let mut l = [0.0f32; G];
            let mut acc = [[0.0f32; PER_LANE]; G];

            let mut p = start + (thread::threadIdx_x() >> 5);
            while p < end {
                let base = p as usize * row + kv_h * HEAD_DIM + lane;
                let mut kk = [0.0f32; PER_LANE];
                let mut vv = [0.0f32; PER_LANE];
                let mut e = 0;
                while e < PER_LANE {
                    kk[e] = load(k[base + 32 * e]);
                    vv[e] = load(v[base + 32 * e]);
                    e += 1;
                }
                fold_position::<G>(&mut m, &mut l, &mut acc, &qr, &kk, &vv);
                p += WARPS as u32;
            }

            let mut g = 0;
            while g < G {
                let slot = Slot {
                    pacc,
                    pml,
                    index: (first + g) * n_splits as usize + split as usize,
                };
                merge_head(m[g], l[g], &acc[g], ml, accs, &slot);
                g += 1;
            }
        }
    }

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn splitk_partial_f32<const G: usize>(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        pacc: *mut f32,
        pml: *mut f32,
        seq_len: u32,
        chunk: u32,
        n_splits: u32,
        num_kv_heads: u32,
    ) {
        static mut ML: SharedArray<f32, { WARPS * 2 }> = SharedArray::UNINIT;
        static mut ACC: SharedArray<f32, { WARPS * HEAD_DIM }> = SharedArray::UNINIT;
        unsafe {
            let ml = SharedArray::as_raw_mut_ptr(&raw mut ML);
            let accs = SharedArray::as_raw_mut_ptr(&raw mut ACC);
            partial_body::<G, f32, _>(q, k, v, pacc, pml, seq_len, chunk, n_splits, num_kv_heads, ml, accs, |x| x);
        }
    }

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn splitk_partial_f16<const G: usize>(
        q: &[f32],
        k: &[u16],
        v: &[u16],
        pacc: *mut f32,
        pml: *mut f32,
        seq_len: u32,
        chunk: u32,
        n_splits: u32,
        num_kv_heads: u32,
    ) {
        static mut ML: SharedArray<f32, { WARPS * 2 }> = SharedArray::UNINIT;
        static mut ACC: SharedArray<f32, { WARPS * HEAD_DIM }> = SharedArray::UNINIT;
        unsafe {
            let ml = SharedArray::as_raw_mut_ptr(&raw mut ML);
            let accs = SharedArray::as_raw_mut_ptr(&raw mut ACC);
            partial_body::<G, u16, _>(q, k, v, pacc, pml, seq_len, chunk, n_splits, num_kv_heads, ml, accs, |x: u16| {
                convert::cvt_f32_f16x2_lo(u32::from(x))
            });
        }
    }

    #[kernel]
    pub unsafe fn splitk_reduce(pacc: &[f32], pml: &[f32], out: *mut f32, n_splits: u32) {
        unsafe {
            let h = thread::blockIdx_x() as usize;
            let tid = thread::threadIdx_x() as usize;
            let n = n_splits as usize;
            let mut big_m = f32::NEG_INFINITY;
            let mut s = 0;
            while s < n {
                let ms = pml[(h * n + s) * 2];
                if ms > big_m {
                    big_m = ms;
                }
                s += 1;
            }
            let mut num = 0.0f32;
            let mut den = 0.0f32;
            let mut s = 0;
            while s < n {
                let slot = h * n + s;
                let scale = (pml[slot * 2] - big_m).exp();
                num += pacc[slot * HEAD_DIM + tid] * scale;
                den += pml[slot * 2 + 1] * scale;
                s += 1;
            }
            *out.add(h * HEAD_DIM + tid) = num / den;
        }
    }
}

// ---------------------------------------------------------------- host side

#[derive(Clone, Copy, Debug)]
struct Plan {
    chunk: u32,
    n_splits: u32,
}

fn plan(seq_len: u32, target: u32, min_chunk: u32) -> Plan {
    let chunk = seq_len.div_ceil(target).max(min_chunk);
    Plan {
        chunk,
        n_splits: seq_len.div_ceil(chunk),
    }
}

/// One online-softmax partial (running max, sum, un-normalised accumulator).
#[derive(Clone)]
struct Partial {
    m: f32,
    l: f32,
    acc: Vec<f32>,
}

impl Partial {
    fn empty() -> Self {
        Self { m: f32::NEG_INFINITY, l: 0.0, acc: vec![0.0; HEAD_DIM] }
    }

    fn push(&mut self, score: f32, v: &[f32]) {
        let new_m = self.m.max(score);
        let correction = (self.m - new_m).exp();
        let weight = (score - new_m).exp();
        self.l = self.l * correction + weight;
        for (a, x) in self.acc.iter_mut().zip(v) {
            *a = *a * correction + weight * x;
        }
        self.m = new_m;
    }

    fn merge(parts: &[Self]) -> Self {
        let m = parts.iter().fold(f32::NEG_INFINITY, |m, p| m.max(p.m));
        let mut merged = Self { m, l: 0.0, acc: vec![0.0; HEAD_DIM] };
        for part in parts {
            let scale = (part.m - m).exp();
            merged.l += part.l * scale;
            for (a, x) in merged.acc.iter_mut().zip(&part.acc) {
                *a += x * scale;
            }
        }
        merged
    }
}

/// Verbatim port of `decode_attention_splitk_cpu` (aprender-gpu): the kernel pair's
/// split / warp / merge order, in f32.
fn twin(q: &[f32], k: &[f32], v: &[f32], nh: usize, nkv: usize, seq_len: usize, p: Plan) -> Vec<f32> {
    let row = nkv * HEAD_DIM;
    let group = nh / nkv;
    let (chunk, n_splits) = (p.chunk as usize, p.n_splits as usize);
    let warp_partial = |q_h: &[f32], kv_h: usize, start: usize, end: usize, w: usize| {
        let mut part = Partial::empty();
        for pos in (start + w..end).step_by(WARPS) {
            let base = pos * row + kv_h * HEAD_DIM;
            let dot: f32 = q_h.iter().zip(&k[base..base + HEAD_DIM]).map(|(a, b)| a * b).sum();
            part.push(dot / (HEAD_DIM as f32).sqrt(), &v[base..base + HEAD_DIM]);
        }
        part
    };
    let mut out = vec![0.0f32; nh * HEAD_DIM];
    for (h, out_h) in out.chunks_exact_mut(HEAD_DIM).enumerate() {
        let q_h = &q[h * HEAD_DIM..(h + 1) * HEAD_DIM];
        let splits: Vec<Partial> = (0..n_splits)
            .map(|s| {
                let start = (s * chunk).min(seq_len);
                let end = (start + chunk).min(seq_len);
                let warps: Vec<Partial> = (0..WARPS).map(|w| warp_partial(q_h, h / group, start, end, w)).collect();
                Partial::merge(&warps)
            })
            .collect();
        let merged = Partial::merge(&splits);
        for (o, a) in out_h.iter_mut().zip(&merged.acc) {
            *o = a / merged.l;
        }
    }
    out
}

/// Exact softmax attention in f64, positions ascending.
fn reference_f64(q: &[f32], k: &[f32], v: &[f32], nh: usize, nkv: usize, seq_len: usize) -> Vec<f64> {
    let row = nkv * HEAD_DIM;
    let group = nh / nkv;
    let mut out = vec![0.0f64; nh * HEAD_DIM];
    for h in 0..nh {
        let kv_h = h / group;
        let q_h = &q[h * HEAD_DIM..(h + 1) * HEAD_DIM];
        let scores: Vec<f64> = (0..seq_len)
            .map(|p| {
                let b = p * row + kv_h * HEAD_DIM;
                q_h.iter().zip(&k[b..b + HEAD_DIM]).map(|(a, x)| f64::from(*a) * f64::from(*x)).sum::<f64>() / 16.0
            })
            .collect();
        let mx = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let ws: Vec<f64> = scores.iter().map(|s| (s - mx).exp()).collect();
        let sum: f64 = ws.iter().sum();
        for (p, w) in ws.iter().enumerate() {
            let b = p * row + kv_h * HEAD_DIM;
            for (o, x) in out[h * HEAD_DIM..(h + 1) * HEAD_DIM].iter_mut().zip(&v[b..b + HEAD_DIM]) {
                *o += w / sum * f64::from(*x);
            }
        }
    }
    out
}

fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let a = x.abs();
    if a < 6.103_515_6e-5 {
        return sign | (a * 16_777_216.0).round() as u16;
    }
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let mant = b & 0x7f_ffff;
    let mut h = ((exp as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && h & 1 == 1) {
        h += 1;
    }
    sign | h as u16
}

fn widen(b: u16) -> f32 {
    let sign = if b & 0x8000 == 0 { 1.0f32 } else { -1.0 };
    let e = i32::from((b >> 10) & 0x1f);
    let m = f32::from(b & 0x3ff);
    if e == 0 { sign * m * 2f32.powi(-24) } else { sign * (1.0 + m / 1024.0) * 2f32.powi(e - 15) }
}

struct Lcg(u32);
impl Lcg {
    fn next(&mut self, scale: f32) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (f32::from(((self.0 >> 16) & 0xFFFF) as u16) / 32768.0 - 1.0) * scale
    }
}

/// Relative max |Δ| against `want`'s own scale, and cosine.
fn compare(got: &[f32], want: &[f64]) -> (f64, f64) {
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let (mut worst, mut dot, mut ng, mut nw) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (g, w) in got.iter().zip(want) {
        let g = f64::from(*g);
        worst = worst.max((g - w).abs());
        dot += g * w;
        ng += g * g;
        nw += w * w;
    }
    (worst / scale, dot / (ng.sqrt() * nw.sqrt()))
}

type Res<T> = Result<T, Box<dyn std::error::Error>>;

struct Args {
    builder_dir: Option<String>,
    out_path: Option<String>,
    host: String,
}

fn parse_args() -> Res<Args> {
    let mut a = Args { builder_dir: None, out_path: None, host: "unknown".into() };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    for pair in argv.chunks(2) {
        let val = pair.get(1).cloned();
        match pair[0].as_str() {
            "--builder-ptx" => a.builder_dir = val,
            "--out" => a.out_path = val,
            "--host" => a.host = val.unwrap_or_default(),
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    Ok(a)
}

const NH: usize = 16; // the 9B's geometry
const NKV: usize = 4;
const MAX_LEN: usize = 60_000;

/// Host data: q, the f32 cache, the f16 cache and its widened values.
struct Data {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    k16: Vec<u16>,
    v16: Vec<u16>,
    k16w: Vec<f32>,
    v16w: Vec<f32>,
}

fn make_data() -> Data {
    let row = NKV * HEAD_DIM;
    let mut rng = Lcg(0x3725_0A0A);
    let q: Vec<f32> = (0..NH * HEAD_DIM).map(|_| rng.next(0.3)).collect();
    let mut k = Vec::with_capacity(MAX_LEN * row);
    let mut v = Vec::with_capacity(MAX_LEN * row);
    for p in 0..MAX_LEN {
        let s = 0.5 + 0.5 * ((p % 97) as f32 / 97.0);
        k.extend((0..row).map(|_| rng.next(s)));
        v.extend((0..row).map(|_| rng.next(s)));
    }
    let k16: Vec<u16> = k.iter().map(|x| f16_bits(*x)).collect();
    let v16: Vec<u16> = v.iter().map(|x| f16_bits(*x)).collect();
    let k16w = k16.iter().map(|b| widen(*b)).collect();
    let v16w = v16.iter().map(|b| widen(*b)).collect();
    Data { q, k, v, k16, v16, k16w, v16w }
}

/// Device buffers shared by every case.
struct Dev {
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    k16: DeviceBuffer<u16>,
    v16: DeviceBuffer<u16>,
    pacc: DeviceBuffer<f32>,
    pml: DeviceBuffer<f32>,
    out: DeviceBuffer<f32>,
    out_b: DeviceBuffer<f32>,
}

fn upload(stream: &CudaStream, d: &Data) -> Res<Dev> {
    Ok(Dev {
        q: DeviceBuffer::from_host(stream, &d.q)?,
        k: DeviceBuffer::from_host(stream, &d.k)?,
        v: DeviceBuffer::from_host(stream, &d.v)?,
        k16: DeviceBuffer::from_host(stream, &d.k16)?,
        v16: DeviceBuffer::from_host(stream, &d.v16)?,
        pacc: DeviceBuffer::zeroed(stream, NH * TARGET_SPLITS as usize * HEAD_DIM)?,
        pml: DeviceBuffer::zeroed(stream, NH * TARGET_SPLITS as usize * 2)?,
        out: DeviceBuffer::zeroed(stream, NH * HEAD_DIM)?,
        out_b: DeviceBuffer::zeroed(stream, NH * HEAD_DIM)?,
    })
}

/// The builder-PTX pair emitted by `gdn_splitk_rungs --emit-ptx` for this device.
struct Builder {
    fa32: CudaFunction,
    fa16: CudaFunction,
    fb: CudaFunction,
}

fn load_builder(ctx: &std::sync::Arc<CudaContext>, dir: &str, target: &str) -> Res<Builder> {
    let load = |name: &str, file: String| -> Res<CudaFunction> {
        let src = std::fs::read_to_string(format!("{dir}/{file}"))?;
        Ok(ctx.load_module_from_ptx_src(&src)?.load_function(name)?)
    };
    Ok(Builder {
        fa32: load("gdn_decode_attention_splitk_f32", format!("gdn_decode_attention_splitk_f32_16_4.{target}.ptx"))?,
        fa16: load("gdn_decode_attention_splitk_f16", format!("gdn_decode_attention_splitk_f16_16_4.{target}.ptx"))?,
        fb: load("gdn_decode_attention_splitk_reduce", format!("gdn_decode_attention_splitk_reduce_16.{target}.ptx"))?,
    })
}

/// CUDA-event µs per call: 3 warmup, then the median of 5 × 20 launches.
fn time_events(stream: &CudaStream, f: &dyn Fn() -> Result<(), cuda_core::DriverError>) -> Res<f64> {
    for _ in 0..3 {
        f()?;
    }
    let mut v = Vec::new();
    for _ in 0..5 {
        let s = stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        for _ in 0..20 {
            f()?;
        }
        let e = stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        e.synchronize()?;
        v.push(f64::from(s.elapsed_ms(&e)?) * 1000.0 / 20.0);
    }
    v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    Ok(v[v.len() / 2])
}

fn configs(p: Plan) -> (LaunchConfig, LaunchConfig) {
    let block = (BLOCK, 1, 1);
    (
        LaunchConfig { grid_dim: (NKV as u32, p.n_splits, 1), block_dim: block, shared_mem_bytes: 0 },
        LaunchConfig { grid_dim: (NH as u32, 1, 1), block_dim: block, shared_mem_bytes: 0 },
    )
}

/// Launch the oxide pair once.
fn run_oxide(module: &kernels::LoadedModule, stream: &CudaStream, dev: &Dev, f16: bool, seq_len: usize, p: Plan) -> Result<(), cuda_core::DriverError> {
    let (cfg_a, cfg_b) = configs(p);
    let (pacc, pml) = (dev.pacc.cu_deviceptr() as *mut f32, dev.pml.cu_deviceptr() as *mut f32);
    // SAFETY: the raw configs are the kernels' own launch shape and every pointer is a
    // live allocation sized for TARGET_SPLITS splits.
    unsafe {
        if f16 {
            module.splitk_partial_f16::<4>(stream, cfg_a, &dev.q, &dev.k16, &dev.v16, pacc, pml, seq_len as u32, p.chunk, p.n_splits, NKV as u32)?;
        } else {
            module.splitk_partial_f32::<4>(stream, cfg_a, &dev.q, &dev.k, &dev.v, pacc, pml, seq_len as u32, p.chunk, p.n_splits, NKV as u32)?;
        }
        module.splitk_reduce(stream, cfg_b, &dev.pacc, &dev.pml, dev.out.cu_deviceptr() as *mut f32, p.n_splits)?;
    }
    Ok(())
}

/// Launch the builder pair once, into `out_b`.
fn run_builder(b: &Builder, stream: &CudaStream, dev: &Dev, f16: bool, seq_len: usize, p: Plan) -> Result<(), cuda_core::DriverError> {
    let (cfg_a, cfg_b) = configs(p);
    let (k, v) = if f16 { (dev.k16.cu_deviceptr(), dev.v16.cu_deviceptr()) } else { (dev.k.cu_deviceptr(), dev.v.cu_deviceptr()) };
    let mut ptrs = [dev.q.cu_deviceptr(), k, v, dev.pacc.cu_deviceptr(), dev.pml.cu_deviceptr()];
    // The builder kernels declare u32 scalars; pass them as u32.
    let mut scalars = [seq_len as u32, p.chunk, p.n_splits];
    let mut pa: Vec<*mut std::ffi::c_void> = ptrs.iter_mut().map(|x| std::ptr::from_mut(x).cast()).collect();
    pa.extend(scalars.iter_mut().map(|x| std::ptr::from_mut(x).cast::<std::ffi::c_void>()));
    let mut ptrs_b = [dev.pacc.cu_deviceptr(), dev.pml.cu_deviceptr(), dev.out_b.cu_deviceptr()];
    let mut n = p.n_splits;
    let mut pb: Vec<*mut std::ffi::c_void> = ptrs_b.iter_mut().map(|x| std::ptr::from_mut(x).cast()).collect();
    pb.push(std::ptr::from_mut(&mut n).cast());
    let fa = if f16 { &b.fa16 } else { &b.fa32 };
    // SAFETY: argument order and widths are the builder kernels' .param declarations;
    // buffers are sized for TARGET_SPLITS splits.
    unsafe {
        cuda_core::launch_kernel_on_stream(fa, cfg_a.grid_dim, cfg_a.block_dim, 0, stream, &mut pa)?;
        cuda_core::launch_kernel_on_stream(&b.fb, cfg_b.grid_dim, cfg_b.block_dim, 0, stream, &mut pb)?;
    }
    Ok(())
}

fn to_f64(x: &[f32]) -> Vec<f64> {
    x.iter().map(|v| f64::from(*v)).collect()
}

/// One (storage, seq_len) case: parity against the twin, f64 and the builder pair,
/// and the event-timed A/B. Returns the JSON row and whether parity passed.
#[allow(clippy::too_many_arguments)]
fn run_case(module: &kernels::LoadedModule, builder: Option<&Builder>, stream: &CudaStream, data: &Data, dev: &Dev, f16: bool, seq_len: usize, p: Plan) -> Res<(String, bool)> {
    let (k_seen, v_seen) = if f16 { (&data.k16w, &data.v16w) } else { (&data.k, &data.v) };
    run_oxide(module, stream, dev, f16, seq_len, p)?;
    let got = dev.out.to_host_vec(stream)?;
    let (rel_twin, _) = compare(&got, &to_f64(&twin(&data.q, k_seen, v_seen, NH, NKV, seq_len, p)));
    let (rel_ref, cos_ref) = compare(&got, &reference_f64(&data.q, k_seen, v_seen, NH, NKV, seq_len));
    let pass = rel_twin < 1e-4 && rel_ref < 1e-4 && cos_ref > 0.9999;
    let oxide_us = time_events(stream, &|| run_oxide(module, stream, dev, f16, seq_len, p))?;
    let (builder_us, rel_builder) = match builder {
        Some(b) => {
            run_builder(b, stream, dev, f16, seq_len, p)?;
            let (rel, _) = compare(&got, &to_f64(&dev.out_b.to_host_vec(stream)?));
            (Some(time_events(stream, &|| run_builder(b, stream, dev, f16, seq_len, p))?), Some(rel))
        }
        None => (None, None),
    };
    let kv = if f16 { "f16" } else { "f32" };
    let verdict = if pass { "PASS" } else { "FAIL" };
    let rel_b = rel_builder.map_or("null".to_string(), |v| format!("{v:.3e}"));
    let b_us = builder_us.map_or("null".to_string(), |b| format!("{b:.2}"));
    let ratio = builder_us.map_or("null".to_string(), |b| format!("{:.3}", oxide_us / b));
    println!(
        "  {kv} L={seq_len:>6} splits={:>3}  vs twin {rel_twin:.2e}  vs f64 {rel_ref:.2e} cos {cos_ref:.7}  vs builder {rel_b}  oxide {oxide_us:>8.1}us  builder {b_us}us  {verdict}",
        p.n_splits
    );
    let row = format!(
        "    {{\"kv\": \"{kv}\", \"seq_len\": {seq_len}, \"chunk\": {}, \"n_splits\": {}, \"oxide_vs_twin_rel_max\": {rel_twin:.3e}, \
         \"oxide_vs_f64_rel_max\": {rel_ref:.3e}, \"oxide_vs_f64_cos\": {cos_ref:.9}, \"oxide_vs_builder_rel_max\": {rel_b}, \
         \"oxide_us\": {oxide_us:.2}, \"builder_us\": {b_us}, \"oxide_over_builder\": {ratio}, \"parity\": \"{verdict}\"}}",
        p.chunk, p.n_splits
    );
    Ok((row, pass))
}

fn main() -> Res<()> {
    let args = parse_args()?;
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // SAFETY: this package owns the embedded device bundle for `kernels`.
    let module = unsafe { kernels::load(&ctx)? };
    let (major, minor) = ctx.compute_capability()?;
    let target = format!("sm_{major}{minor}");
    println!("== PMAT-3725 cuda-oxide split-K decode attention — {target} ==");

    let data = make_data();
    let dev = upload(&stream, &data)?;
    let builder = match &args.builder_dir {
        Some(dir) => Some(load_builder(&ctx, dir, &target).map_err(|e| format!("builder PTX: {e}"))?),
        None => None,
    };
    let cases = [
        (37usize, plan(37, TARGET_SPLITS, MIN_CHUNK)),
        (203, plan(203, 29, 1)),
        (4096, plan(4096, TARGET_SPLITS, MIN_CHUNK)),
        (20_000, plan(20_000, TARGET_SPLITS, MIN_CHUNK)),
        (60_000, plan(60_000, TARGET_SPLITS, MIN_CHUNK)),
    ];
    let mut rows = Vec::new();
    let mut all_pass = true;
    for f16 in [false, true] {
        for &(seq_len, p) in &cases {
            let (row, pass) = run_case(&module, builder.as_ref(), &stream, &data, &dev, f16, seq_len, p)?;
            rows.push(row);
            all_pass &= pass;
        }
    }

    let verdict = if all_pass { "PASS" } else { "FAIL" };
    let json = format!(
        "{{\n  \"schema\": \"apr-kernel-receipt/v1\",\n  \"kernel\": \"gdn_decode_attention_splitk\",\n  \"authoring\": \"oxide\",\n  \
         \"host\": \"{}\",\n  \"cc\": \"{major}.{minor}\",\n  \"cuda_oxide\": \"b9847e95\",\n  \"geometry\": \"16/4, head_dim 256\",\n  \
         \"timing\": \"CUDA events, median of 5 x 20 launches after 3 warmup, kernel A + kernel B\",\n  \"parity\": \"{verdict}\",\n  \"rows\": [\n{}\n  ]\n}}\n",
        args.host,
        rows.join(",\n"),
    );
    if let Some(p) = &args.out_path {
        std::fs::write(p, json)?;
        println!("wrote {p}");
    }
    println!("PARITY {verdict}");
    if !all_pass {
        std::process::exit(1);
    }
    Ok(())
}
