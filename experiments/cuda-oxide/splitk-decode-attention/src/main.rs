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
use cuda_core::{CudaContext, DeviceBuffer};
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
            let tid = thread::threadIdx_x() as usize;
            let lane = tid & 31;
            let wid = tid >> 5;
            let start = split * chunk;
            if start >= seq_len {
                return;
            }
            let end = if start + chunk < seq_len { start + chunk } else { seq_len };
            let row = num_kv_heads as usize * HEAD_DIM;
            let first = kv_h * G;

            let mut qr = [[0.0f32; PER_LANE]; G];
            let mut g = 0;
            while g < G {
                let mut e = 0;
                while e < PER_LANE {
                    qr[g][e] = q[(first + g) * HEAD_DIM + lane + 32 * e];
                    e += 1;
                }
                g += 1;
            }
            let mut m = [f32::NEG_INFINITY; G];
            let mut l = [0.0f32; G];
            let mut acc = [[0.0f32; PER_LANE]; G];

            let mut p = start + wid as u32;
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
                p += WARPS as u32;
            }

            let mut g = 0;
            while g < G {
                if lane == 0 {
                    *ml.add(wid * 2) = m[g];
                    *ml.add(wid * 2 + 1) = l[g];
                }
                let mut e = 0;
                while e < PER_LANE {
                    *accs.add(wid * HEAD_DIM + lane + 32 * e) = acc[g][e];
                    e += 1;
                }
                thread::sync_threads();

                let mut bm = f32::NEG_INFINITY;
                let mut w = 0;
                while w < WARPS {
                    let mw = *ml.add(w * 2);
                    if mw > bm {
                        bm = mw;
                    }
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
                let slot = (first + g) * n_splits as usize + split as usize;
                *pacc.add(slot * HEAD_DIM + tid) = ba;
                if tid == 0 {
                    *pml.add(slot * 2) = bm;
                    *pml.add(slot * 2 + 1) = bl;
                }
                thread::sync_threads();
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

/// Verbatim port of `decode_attention_splitk_cpu` (aprender-gpu).
#[allow(clippy::too_many_arguments)]
fn twin(q: &[f32], k: &[f32], v: &[f32], nh: usize, nkv: usize, seq_len: usize, p: Plan) -> Vec<f32> {
    let row = nkv * HEAD_DIM;
    let group = nh / nkv;
    let (chunk, n_splits) = (p.chunk as usize, p.n_splits as usize);
    let sqrt_hd = (HEAD_DIM as f32).sqrt();
    let mut partials = vec![(f32::NEG_INFINITY, 0.0f32, vec![0.0f32; HEAD_DIM]); nh * n_splits];
    for kv_h in 0..nkv {
        for s in 0..n_splits {
            let start = s * chunk;
            if start >= seq_len {
                continue;
            }
            let end = (start + chunk).min(seq_len);
            for g in 0..group {
                let h = kv_h * group + g;
                let q_h = &q[h * HEAD_DIM..(h + 1) * HEAD_DIM];
                let mut ws = vec![(f32::NEG_INFINITY, 0.0f32, vec![0.0f32; HEAD_DIM]); WARPS];
                for (w, st) in ws.iter_mut().enumerate() {
                    let mut pos = start + w;
                    while pos < end {
                        let base = pos * row + kv_h * HEAD_DIM;
                        let dot: f32 = q_h.iter().zip(&k[base..base + HEAD_DIM]).map(|(a, b)| a * b).sum();
                        let score = dot / sqrt_hd;
                        let nm = st.0.max(score);
                        let corr = (st.0 - nm).exp();
                        let wgt = (score - nm).exp();
                        st.1 = st.1 * corr + wgt;
                        for (a, x) in st.2.iter_mut().zip(&v[base..base + HEAD_DIM]) {
                            *a = *a * corr + wgt * x;
                        }
                        st.0 = nm;
                        pos += WARPS;
                    }
                }
                let bm = ws.iter().fold(f32::NEG_INFINITY, |m, w| m.max(w.0));
                let mut bl = 0.0f32;
                let mut ba = vec![0.0f32; HEAD_DIM];
                for w in &ws {
                    let sc = (w.0 - bm).exp();
                    bl += w.1 * sc;
                    for (a, x) in ba.iter_mut().zip(&w.2) {
                        *a += x * sc;
                    }
                }
                partials[h * n_splits + s] = (bm, bl, ba);
            }
        }
    }
    let mut out = vec![0.0f32; nh * HEAD_DIM];
    for h in 0..nh {
        let parts = &partials[h * n_splits..(h + 1) * n_splits];
        let big_m = parts.iter().fold(f32::NEG_INFINITY, |m, p| m.max(p.0));
        let mut den = 0.0f32;
        let mut num = vec![0.0f32; HEAD_DIM];
        for part in parts {
            let sc = (part.0 - big_m).exp();
            den += part.1 * sc;
            for (n, a) in num.iter_mut().zip(&part.2) {
                *n += a * sc;
            }
        }
        for (o, n) in out[h * HEAD_DIM..(h + 1) * HEAD_DIM].iter_mut().zip(num) {
            *o = n / den;
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut builder_dir: Option<String> = None;
    let mut out_path: Option<String> = None;
    let mut host = String::from("unknown");
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let val = argv.get(i + 1).cloned();
        match argv[i].as_str() {
            "--builder-ptx" => builder_dir = val,
            "--out" => out_path = val,
            "--host" => host = val.unwrap_or_default(),
            other => return Err(format!("unknown argument {other}").into()),
        }
        i += 2;
    }

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // SAFETY: this package owns the embedded device bundle for `kernels`.
    let module = unsafe { kernels::load(&ctx)? };
    let (major, minor) = ctx.compute_capability()?;
    let target = format!("sm_{major}{minor}");
    println!("== PMAT-3725 cuda-oxide split-K decode attention — {target} ==");

    let mut rows: Vec<String> = Vec::new();
    let mut all_pass = true;
    let (nh, nkv) = (16usize, 4usize); // the 9B's geometry
    let max_len = 60_000usize;
    let row = nkv * HEAD_DIM;
    let mut rng = Lcg(0x3725_0A0A);
    let q: Vec<f32> = (0..nh * HEAD_DIM).map(|_| rng.next(0.3)).collect();
    let mut k = Vec::with_capacity(max_len * row);
    let mut v = Vec::with_capacity(max_len * row);
    for p in 0..max_len {
        let s = 0.5 + 0.5 * ((p % 97) as f32 / 97.0);
        for _ in 0..row {
            k.push(rng.next(s));
        }
        for _ in 0..row {
            v.push(rng.next(s));
        }
    }
    let k16: Vec<u16> = k.iter().map(|x| f16_bits(*x)).collect();
    let v16: Vec<u16> = v.iter().map(|x| f16_bits(*x)).collect();
    let k16w: Vec<f32> = k16.iter().map(|b| widen(*b)).collect();
    let v16w: Vec<f32> = v16.iter().map(|b| widen(*b)).collect();

    let q_dev = DeviceBuffer::from_host(&stream, &q)?;
    let k_dev = DeviceBuffer::from_host(&stream, &k)?;
    let v_dev = DeviceBuffer::from_host(&stream, &v)?;
    let k16_dev = DeviceBuffer::from_host(&stream, &k16)?;
    let v16_dev = DeviceBuffer::from_host(&stream, &v16)?;
    let pacc = DeviceBuffer::<f32>::zeroed(&stream, nh * TARGET_SPLITS as usize * HEAD_DIM)?;
    let pml = DeviceBuffer::<f32>::zeroed(&stream, nh * TARGET_SPLITS as usize * 2)?;
    let out = DeviceBuffer::<f32>::zeroed(&stream, nh * HEAD_DIM)?;
    let out_b = DeviceBuffer::<f32>::zeroed(&stream, nh * HEAD_DIM)?;

    // The builder kernels, if their PTX was emitted for this device.
    let builder = builder_dir.as_ref().map(|dir| -> Result<_, Box<dyn std::error::Error>> {
        let load = |name: &str, file: &str| -> Result<_, Box<dyn std::error::Error>> {
            let src = std::fs::read_to_string(format!("{dir}/{file}"))?;
            let m = ctx.load_module_from_ptx_src(&src)?;
            Ok(m.load_function(name)?)
        };
        Ok((
            load("gdn_decode_attention_splitk_f32", &format!("gdn_decode_attention_splitk_f32_16_4.{target}.ptx"))?,
            load("gdn_decode_attention_splitk_f16", &format!("gdn_decode_attention_splitk_f16_16_4.{target}.ptx"))?,
            load("gdn_decode_attention_splitk_reduce", &format!("gdn_decode_attention_splitk_reduce_16.{target}.ptx"))?,
        ))
    });
    let builder = match builder {
        Some(Ok(b)) => Some(b),
        Some(Err(e)) => return Err(format!("builder PTX: {e}").into()),
        None => None,
    };

    for f16 in [false, true] {
        let (k_seen, v_seen) = if f16 { (&k16w, &v16w) } else { (&k, &v) };
        let (k_ptr, v_ptr) = if f16 {
            (k16_dev.cu_deviceptr(), v16_dev.cu_deviceptr())
        } else {
            (k_dev.cu_deviceptr(), v_dev.cu_deviceptr())
        };
        for (seq_len, p) in [
            (37usize, plan(37, TARGET_SPLITS, MIN_CHUNK)),
            (203, plan(203, 29, 1)),
            (4096, plan(4096, TARGET_SPLITS, MIN_CHUNK)),
            (20_000, plan(20_000, TARGET_SPLITS, MIN_CHUNK)),
            (60_000, plan(60_000, TARGET_SPLITS, MIN_CHUNK)),
        ] {
            let cfg_a = LaunchConfig {
                grid_dim: (nkv as u32, p.n_splits, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            };
            let cfg_b = LaunchConfig {
                grid_dim: (nh as u32, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            };
            let pacc_ptr = pacc.cu_deviceptr() as *mut f32;
            let pml_ptr = pml.cu_deviceptr() as *mut f32;
            let out_ptr = out.cu_deviceptr() as *mut f32;
            let oxide = || -> Result<(), cuda_core::DriverError> {
                // SAFETY: the raw configs are the kernels' own launch shape and every
                // pointer is a live allocation sized for TARGET_SPLITS splits.
                unsafe {
                    if f16 {
                        module.splitk_partial_f16::<4>(&stream, cfg_a, &q_dev, &k16_dev, &v16_dev, pacc_ptr, pml_ptr, seq_len as u32, p.chunk, p.n_splits, nkv as u32)?;
                    } else {
                        module.splitk_partial_f32::<4>(&stream, cfg_a, &q_dev, &k_dev, &v_dev, pacc_ptr, pml_ptr, seq_len as u32, p.chunk, p.n_splits, nkv as u32)?;
                    }
                    module.splitk_reduce(&stream, cfg_b, &pacc, &pml, out_ptr, p.n_splits)?;
                }
                Ok(())
            };
            oxide()?;
            let got = out.to_host_vec(&stream)?;

            let tw = twin(&q, k_seen, v_seen, nh, nkv, seq_len, p);
            let tw64: Vec<f64> = tw.iter().map(|x| f64::from(*x)).collect();
            let (rel_twin, _) = compare(&got, &tw64);
            let reference = reference_f64(&q, k_seen, v_seen, nh, nkv, seq_len);
            let (rel_ref, cos_ref) = compare(&got, &reference);
            let pass = rel_twin < 1e-4 && rel_ref < 1e-4 && cos_ref > 0.9999;
            all_pass &= pass;

            // A/B against the builder pair on the same buffers.
            let time = |f: &dyn Fn() -> Result<(), cuda_core::DriverError>| -> Result<f64, Box<dyn std::error::Error>> {
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
            };
            let oxide_us = time(&oxide)?;
            let (builder_us, rel_builder) = if let Some((fa32, fa16, fb)) = &builder {
                let fa = if f16 { fa16 } else { fa32 };
                let builder_run = || -> Result<(), cuda_core::DriverError> {
                    let mut a = [
                        q_dev.cu_deviceptr(), k_ptr, v_ptr, pacc.cu_deviceptr(), pml.cu_deviceptr(),
                        seq_len as u64, u64::from(p.chunk), u64::from(p.n_splits),
                    ];
                    // The builder kernels declare u32 scalars; pass them as u32.
                    let mut s32 = [seq_len as u32, p.chunk, p.n_splits];
                    let mut pa: Vec<*mut std::ffi::c_void> = a[..5].iter_mut().map(|x| std::ptr::from_mut(x).cast()).collect();
                    pa.extend(s32.iter_mut().map(|x| std::ptr::from_mut(x).cast::<std::ffi::c_void>()));
                    let mut b = [pacc.cu_deviceptr(), pml.cu_deviceptr(), out_b.cu_deviceptr()];
                    let mut n = p.n_splits;
                    let mut pb: Vec<*mut std::ffi::c_void> = b.iter_mut().map(|x| std::ptr::from_mut(x).cast()).collect();
                    pb.push(std::ptr::from_mut(&mut n).cast());
                    // SAFETY: argument order and widths are the builder kernels' .param
                    // declarations; buffers are sized for TARGET_SPLITS splits.
                    unsafe {
                        cuda_core::launch_kernel_on_stream(fa, cfg_a.grid_dim, cfg_a.block_dim, 0, &stream, &mut pa)?;
                        cuda_core::launch_kernel_on_stream(fb, cfg_b.grid_dim, cfg_b.block_dim, 0, &stream, &mut pb)?;
                    }
                    Ok(())
                };
                builder_run()?;
                let gb = out_b.to_host_vec(&stream)?;
                let gb64: Vec<f64> = gb.iter().map(|x| f64::from(*x)).collect();
                let (rel, _) = compare(&got, &gb64);
                (Some(time(&builder_run)?), Some(rel))
            } else {
                (None, None)
            };
            let fmt = |x: Option<f64>, prec: usize| x.map_or("null".to_string(), |v| format!("{v:.prec$e}"));
            println!(
                "  {} L={:>6} splits={:>3}  vs twin {:.2e}  vs f64 {:.2e} cos {:.7}  vs builder {}  oxide {:>8.1}us  builder {}us  {}",
                if f16 { "f16" } else { "f32" },
                seq_len,
                p.n_splits,
                rel_twin,
                rel_ref,
                cos_ref,
                fmt(rel_builder, 2),
                oxide_us,
                builder_us.map_or("null".into(), |b| format!("{b:.1}")),
                if pass { "PASS" } else { "FAIL" },
            );
            rows.push(format!(
                "    {{\"kv\": \"{}\", \"seq_len\": {seq_len}, \"chunk\": {}, \"n_splits\": {}, \"oxide_vs_twin_rel_max\": {rel_twin:.3e}, \
                 \"oxide_vs_f64_rel_max\": {rel_ref:.3e}, \"oxide_vs_f64_cos\": {cos_ref:.9}, \"oxide_vs_builder_rel_max\": {}, \
                 \"oxide_us\": {oxide_us:.2}, \"builder_us\": {}, \"oxide_over_builder\": {}, \"parity\": \"{}\"}}",
                if f16 { "f16" } else { "f32" },
                p.chunk,
                p.n_splits,
                fmt(rel_builder, 3),
                builder_us.map_or("null".into(), |b| format!("{b:.2}")),
                builder_us.map_or("null".into(), |b| format!("{:.3}", oxide_us / b)),
                if pass { "PASS" } else { "FAIL" },
            ));
        }
    }

    let json = format!(
        "{{\n  \"schema\": \"apr-kernel-receipt/v1\",\n  \"kernel\": \"gdn_decode_attention_splitk\",\n  \"authoring\": \"oxide\",\n  \
         \"host\": \"{host}\",\n  \"cc\": \"{major}.{minor}\",\n  \"cuda_oxide\": \"b9847e95\",\n  \"geometry\": \"16/4, head_dim 256\",\n  \
         \"timing\": \"CUDA events, median of 5 x 20 launches after 3 warmup, kernel A + kernel B\",\n  \"parity\": \"{}\",\n  \"rows\": [\n{}\n  ]\n}}\n",
        if all_pass { "PASS" } else { "FAIL" },
        rows.join(",\n"),
    );
    if let Some(p) = out_path {
        std::fs::write(&p, json)?;
        println!("wrote {p}");
    }
    println!("PARITY {}", if all_pass { "PASS" } else { "FAIL" });
    if !all_pass {
        std::process::exit(1);
    }
    Ok(())
}
