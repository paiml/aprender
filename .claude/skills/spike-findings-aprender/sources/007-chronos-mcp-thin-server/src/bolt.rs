//! Chronos-Bolt inference — the spike-005 port (parity to 1e-6), with every matrix product routed
//! through trueno's packed GEMM now that aarch64 has a NEON microkernel (spike 008): projections,
//! feed-forward, the patch-embedding residual blocks, and (new here) attention scores and context.
//! `fast = false` / `attn_gemm = false` keep the plain-loop paths for A/B measurement.
use crate::safetensors::{Tensor, Weights};

#[derive(Clone, Debug)]
pub struct Config { pub d_model: usize, pub d_ff: usize, pub d_kv: usize, pub heads: usize, pub enc_layers: usize, pub dec_layers: usize, pub eps: f32, pub buckets: usize, pub max_distance: usize, pub context_length: usize, pub prediction_length: usize, pub patch: usize, pub quantiles: Vec<f32>, pub use_reg_token: bool }

impl Config {
    pub fn from_json(v: &serde_json::Value) -> Config {
        let c = &v["chronos_config"];
        Config { d_model: v["d_model"].as_u64().expect("d_model") as usize, d_ff: v["d_ff"].as_u64().expect("d_ff") as usize, d_kv: v["d_kv"].as_u64().expect("d_kv") as usize, heads: v["num_heads"].as_u64().expect("heads") as usize,
            enc_layers: v["num_layers"].as_u64().expect("layers") as usize, dec_layers: v["num_decoder_layers"].as_u64().expect("dec layers") as usize, eps: v["layer_norm_epsilon"].as_f64().expect("eps") as f32,
            buckets: v["relative_attention_num_buckets"].as_u64().expect("buckets") as usize, max_distance: v["relative_attention_max_distance"].as_u64().expect("maxd") as usize,
            context_length: c["context_length"].as_u64().expect("ctx") as usize, prediction_length: c["prediction_length"].as_u64().expect("pl") as usize, patch: c["input_patch_size"].as_u64().expect("patch") as usize,
            quantiles: c["quantiles"].as_array().expect("q").iter().map(|q| q.as_f64().expect("f") as f32).collect(), use_reg_token: c["use_reg_token"].as_bool().unwrap_or(false) }
    }
}

// ------------------------------------------------------------ primitives ----
/// y[r][o] = Σ_i x[r][i] w[o][i] (+ b[o]); w is [out, in] row-major as in torch. Plain loops.
pub fn linear(x: &[f32], rows: usize, inp: usize, w: &[f32], out: usize, b: Option<&[f32]>) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * out];
    for r in 0..rows {
        let xr = &x[r * inp..(r + 1) * inp];
        let yr = &mut y[r * out..(r + 1) * out];
        for o in 0..out { yr[o] = dot8(xr, &w[o * inp..(o + 1) * inp]) + b.map_or(0.0, |b| b[o]); }
    }
    y
}

/// Dot product with 8 independent accumulators: an explicit reduction order LLVM can vectorise.
#[inline]
pub fn dot8(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut acc = [0.0f32; 8];
    let chunks = n / 8;
    for c in 0..chunks {
        let (ac, bc) = (&a[c * 8..c * 8 + 8], &b[c * 8..c * 8 + 8]);
        for k in 0..8 { acc[k] += ac[k] * bc[k]; }
    }
    let mut s = acc.iter().sum::<f32>();
    for i in chunks * 8..n { s += a[i] * b[i]; }
    s
}

/// When set, multi-row products go through `trueno::blis::gemm` (rayon-parallel with the crate's
/// `parallel` feature) instead of the single-threaded `gemm_blis`.
pub static PARALLEL_GEMM: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// x[rows×in] · Wᵀ[in×out] through trueno (weights pre-transposed); gemv for a single row.
pub fn linear_fast(x: &[f32], rows: usize, inp: usize, wt: &[f32], out: usize, b: Option<&[f32]>) -> Vec<f32> {
    if rows == 1 {
        let mut y = vec![0.0f32; out];
        trueno::blis::gemv::gemv(inp, out, x, wt, &mut y);
        if let Some(b) = b { for (v, bb) in y.iter_mut().zip(b) { *v += bb; } }
        return y;
    }
    let mut y = vec![0.0f32; rows * out];
    if PARALLEL_GEMM.load(std::sync::atomic::Ordering::Relaxed) { trueno::blis::gemm(rows, out, inp, x, wt, &mut y).expect("gemm"); } else { trueno::blis::gemm_blis(rows, out, inp, x, wt, &mut y, None).expect("gemm_blis"); }
    if let Some(b) = b { for r in 0..rows { for o in 0..out { y[r * out + o] += b[o]; } } }
    y
}

/// One projection, choosing the path: plain loops when `fast` is off; for a single row either
/// trueno's `gemv` over the transposed copy (`row1_gemv`) or a contiguous `dot8` per output over the
/// row-major `[out, in]` weights; for several rows the packed GEMM over the transposed copy.
#[allow(clippy::too_many_arguments)]
pub fn proj(fast: bool, row1_gemv: bool, x: &[f32], rows: usize, inp: usize, w: &[f32], wt: &[f32], out: usize, b: Option<&[f32]>) -> Vec<f32> {
    if !fast { return linear(x, rows, inp, w, out, b); }
    if rows == 1 && !row1_gemv { return linear(x, 1, inp, w, out, b); }
    linear_fast(x, rows, inp, wt, out, b)
}

pub fn t5_layer_norm(x: &[f32], rows: usize, d: usize, w: &[f32], eps: f32) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * d];
    for r in 0..rows {
        let xr = &x[r * d..(r + 1) * d];
        let var = xr.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for i in 0..d { y[r * d + i] = xr[i] * inv * w[i]; }
    }
    y
}

/// transformers' `_relative_position_bucket`.
pub fn relative_bucket(rel: i64, bidirectional: bool, num_buckets: usize, max_distance: usize) -> usize {
    let mut nb = num_buckets as i64;
    let mut bucket = 0i64;
    let mut rp = rel;
    if bidirectional { nb /= 2; if rp > 0 { bucket += nb; } rp = rp.abs(); } else { rp = -(rp.min(0)); }
    let max_exact = nb / 2;
    let large = max_exact + (((rp as f32 / max_exact as f32).ln() / (max_distance as f32 / max_exact as f32).ln()) * (nb - max_exact) as f32) as i64;
    bucket += if rp < max_exact { rp } else { large.min(nb - 1) };
    bucket as usize
}

/// [out, in] → [in, out]
pub fn transpose(w: &[f32], out: usize, inp: usize) -> Vec<f32> { let mut t = vec![0.0f32; w.len()]; for o in 0..out { for i in 0..inp { t[i * out + o] = w[o * inp + i]; } } t }

pub struct Attn { pub q: Vec<f32>, pub k: Vec<f32>, pub v: Vec<f32>, pub o: Vec<f32>, pub qt: Vec<f32>, pub kt: Vec<f32>, pub vt: Vec<f32>, pub ot: Vec<f32> }
pub struct Block { pub ln_sa: Vec<f32>, pub sa: Attn, pub ln_ca: Option<Vec<f32>>, pub ca: Option<Attn>, pub ln_ff: Vec<f32>, pub wi: Vec<f32>, pub wo: Vec<f32>, pub wit: Vec<f32>, pub wot: Vec<f32> }
pub struct Stack { pub blocks: Vec<Block>, pub rel_bias: Vec<f32>, pub final_ln: Vec<f32>, pub is_decoder: bool }
pub struct Residual { pub wh: Vec<f32>, pub bh: Vec<f32>, pub wo: Vec<f32>, pub bo: Vec<f32>, pub wr: Vec<f32>, pub br: Vec<f32>, pub wht: Vec<f32>, pub wot: Vec<f32>, pub wrt: Vec<f32>, pub inp: usize, pub hid: usize, pub out: usize }
pub struct Bolt { pub fast: bool, pub attn_gemm: bool, pub row1_gemv: bool, pub cfg: Config, pub shared: Vec<f32>, pub in_embed: Residual, pub out_embed: Residual, pub encoder: Stack, pub decoder: Stack }

fn take(w: &Weights, name: &str) -> Vec<f32> { w.get(name).unwrap_or_else(|| panic!("missing tensor {name}")).data.clone() }
fn tensor<'a>(w: &'a Weights, name: &str) -> &'a Tensor { w.get(name).unwrap_or_else(|| panic!("missing tensor {name}")) }

fn residual(w: &Weights, prefix: &str) -> Residual {
    let wh = tensor(w, &format!("{prefix}.hidden_layer.weight")); let wo = tensor(w, &format!("{prefix}.output_layer.weight")); let wr = tensor(w, &format!("{prefix}.residual_layer.weight"));
    let (inp, hid, out) = (wh.shape[1], wh.shape[0], wo.shape[0]);
    Residual { inp, hid, out, wht: transpose(&wh.data, hid, inp), wot: transpose(&wo.data, out, hid), wrt: transpose(&wr.data, out, inp), wh: wh.data.clone(), bh: take(w, &format!("{prefix}.hidden_layer.bias")), wo: wo.data.clone(), bo: take(w, &format!("{prefix}.output_layer.bias")), wr: wr.data.clone(), br: take(w, &format!("{prefix}.residual_layer.bias")) }
}
impl Residual {
    pub fn forward(&self, x: &[f32], rows: usize, fast: bool, row1_gemv: bool) -> Vec<f32> {
        let mut h = proj(fast, row1_gemv, x, rows, self.inp, &self.wh, &self.wht, self.hid, Some(&self.bh));
        for v in h.iter_mut() { if *v < 0.0 { *v = 0.0; } }
        let mut out = proj(fast, row1_gemv, &h, rows, self.hid, &self.wo, &self.wot, self.out, Some(&self.bo));
        let res = proj(fast, row1_gemv, x, rows, self.inp, &self.wr, &self.wrt, self.out, Some(&self.br));
        for (o, r) in out.iter_mut().zip(&res) { *o += r; }
        out
    }
}

fn attn(w: &Weights, prefix: &str) -> Attn {
    let (q, k, v, o) = (take(w, &format!("{prefix}.q.weight")), take(w, &format!("{prefix}.k.weight")), take(w, &format!("{prefix}.v.weight")), take(w, &format!("{prefix}.o.weight")));
    let s = tensor(w, &format!("{prefix}.q.weight")).shape.clone();
    Attn { qt: transpose(&q, s[0], s[1]), kt: transpose(&k, s[0], s[1]), vt: transpose(&v, s[0], s[1]), ot: transpose(&o, s[0], s[1]), q, k, v, o }
}

fn stack(w: &Weights, name: &str, layers: usize, is_decoder: bool) -> Stack {
    let blocks = (0..layers).map(|i| {
        let p = format!("{name}.block.{i}.layer");
        let (ln_ca, ca, ff) = if is_decoder { (Some(take(w, &format!("{p}.1.layer_norm.weight"))), Some(attn(w, &format!("{p}.1.EncDecAttention"))), 2) } else { (None, None, 1) };
        let wi = take(w, &format!("{p}.{ff}.DenseReluDense.wi.weight")); let wo = take(w, &format!("{p}.{ff}.DenseReluDense.wo.weight"));
        let (dff, d) = (tensor(w, &format!("{p}.{ff}.DenseReluDense.wi.weight")).shape[0], tensor(w, &format!("{p}.{ff}.DenseReluDense.wi.weight")).shape[1]);
        Block { ln_sa: take(w, &format!("{p}.0.layer_norm.weight")), sa: attn(w, &format!("{p}.0.SelfAttention")), ln_ca, ca, ln_ff: take(w, &format!("{p}.{ff}.layer_norm.weight")), wit: transpose(&wi, dff, d), wot: transpose(&wo, d, dff), wi, wo }
    }).collect();
    Stack { blocks, rel_bias: take(w, &format!("{name}.block.0.layer.0.SelfAttention.relative_attention_bias.weight")), final_ln: take(w, &format!("{name}.final_layer_norm.weight")), is_decoder }
}

impl Bolt {
    pub fn load(w: &Weights, cfg: Config) -> Bolt {
        Bolt { fast: true, attn_gemm: true, row1_gemv: false, shared: take(w, "shared.weight"), in_embed: residual(w, "input_patch_embedding"), out_embed: residual(w, "output_patch_embedding"), encoder: stack(w, "encoder", cfg.enc_layers, false), decoder: stack(w, "decoder", cfg.dec_layers, true), cfg }
    }

    /// Multi-head attention; `bias[h*lq*lk ..]` optional (relative position), `key_mask[k]` false → excluded.
    pub fn attention(&self, a: &Attn, xq: &[f32], lq: usize, xkv: &[f32], lk: usize, bias: Option<&[f32]>, key_mask: &[bool]) -> Vec<f32> {
        let (d, h, dk) = (self.cfg.d_model, self.cfg.heads, self.cfg.d_kv);
        let (f, g) = (self.fast, self.row1_gemv);
        let (q, k, v) = (proj(f, g, xq, lq, d, &a.q, &a.qt, h * dk, None), proj(f, g, xkv, lk, d, &a.k, &a.kt, h * dk, None), proj(f, g, xkv, lk, d, &a.v, &a.vt, h * dk, None));
        let mut ctx = vec![0.0f32; lq * h * dk];
        if self.attn_gemm && lq > 1 {
            // per head: scores[lq×lk] = Q_h · K_hᵀ and ctx_h[lq×dk] = P · V_h through the packed GEMM
            let (mut qh, mut kt, mut vh) = (vec![0.0f32; lq * dk], vec![0.0f32; dk * lk], vec![0.0f32; lk * dk]);
            let mut scores = vec![0.0f32; lq * lk];
            let mut ch = vec![0.0f32; lq * dk];
            for hh in 0..h {
                for i in 0..lq { qh[i * dk..(i + 1) * dk].copy_from_slice(&q[i * h * dk + hh * dk..i * h * dk + (hh + 1) * dk]); }
                for j in 0..lk { let kj = &k[j * h * dk + hh * dk..j * h * dk + (hh + 1) * dk]; for t in 0..dk { kt[t * lk + j] = kj[t]; } vh[j * dk..(j + 1) * dk].copy_from_slice(&v[j * h * dk + hh * dk..j * h * dk + (hh + 1) * dk]); }
                scores.iter_mut().for_each(|s| *s = 0.0);
                trueno::blis::gemm_blis(lq, lk, dk, &qh, &kt, &mut scores, None).expect("gemm scores");
                for i in 0..lq {
                    let row = &mut scores[i * lk..(i + 1) * lk];
                    let mut mx = f32::NEG_INFINITY;
                    for j in 0..lk { if let Some(b) = bias { row[j] += b[hh * lq * lk + i * lk + j]; } if !key_mask[j] { row[j] += f32::MIN; } if row[j] > mx { mx = row[j]; } }
                    let mut den = 0.0f32;
                    for j in 0..lk { row[j] = (row[j] - mx).exp(); den += row[j]; }
                    let inv = 1.0 / den;
                    for j in 0..lk { row[j] *= inv; }
                }
                ch.iter_mut().for_each(|s| *s = 0.0);
                trueno::blis::gemm_blis(lq, dk, lk, &scores, &vh, &mut ch, None).expect("gemm context");
                for i in 0..lq { ctx[i * h * dk + hh * dk..i * h * dk + (hh + 1) * dk].copy_from_slice(&ch[i * dk..(i + 1) * dk]); }
            }
        } else {
            let mut scores = vec![0.0f32; lk];
            for hh in 0..h {
                for i in 0..lq {
                    let qi = &q[i * h * dk + hh * dk..i * h * dk + (hh + 1) * dk];
                    let mut mx = f32::NEG_INFINITY;
                    for j in 0..lk {
                        let kj = &k[j * h * dk + hh * dk..j * h * dk + (hh + 1) * dk];
                        let mut s = dot8(qi, kj);
                        if let Some(b) = bias { s += b[hh * lq * lk + i * lk + j]; }
                        if !key_mask[j] { s += f32::MIN; }
                        scores[j] = s; if s > mx { mx = s; }
                    }
                    let mut den = 0.0f32;
                    for j in 0..lk { scores[j] = (scores[j] - mx).exp(); den += scores[j]; }
                    let out = &mut ctx[i * h * dk + hh * dk..i * h * dk + (hh + 1) * dk];
                    for j in 0..lk {
                        let p = scores[j] / den;
                        if p == 0.0 { continue; }
                        let vj = &v[j * h * dk + hh * dk..j * h * dk + (hh + 1) * dk];
                        for t in 0..dk { out[t] += p * vj[t]; }
                    }
                }
            }
        }
        proj(f, g, &ctx, lq, h * dk, &a.o, &a.ot, d, None)
    }

    pub fn position_bias(&self, st: &Stack, lq: usize, lk: usize) -> Vec<f32> {
        let h = self.cfg.heads;
        let mut b = vec![0.0f32; h * lq * lk];
        for i in 0..lq { for j in 0..lk {
            let bucket = relative_bucket(j as i64 - i as i64, !st.is_decoder, self.cfg.buckets, self.cfg.max_distance);
            for hh in 0..h { b[hh * lq * lk + i * lk + j] = st.rel_bias[bucket * h + hh]; }
        } }
        b
    }

    pub fn run_stack(&self, st: &Stack, x0: &[f32], l: usize, self_mask: &[bool], enc: Option<(&[f32], usize, &[bool])>) -> Vec<f32> {
        let (d, dff, eps) = (self.cfg.d_model, self.cfg.d_ff, self.cfg.eps);
        let bias = self.position_bias(st, l, l);
        let mut x = x0.to_vec();
        for blk in &st.blocks {
            let hn = t5_layer_norm(&x, l, d, &blk.ln_sa, eps);
            let sa = self.attention(&blk.sa, &hn, l, &hn, l, Some(&bias), self_mask);
            for (a, b) in x.iter_mut().zip(&sa) { *a += b; }
            if let (Some(ln), Some(ca), Some((eh, el, emask))) = (&blk.ln_ca, &blk.ca, enc) {
                let hn = t5_layer_norm(&x, l, d, ln, eps);
                let out = self.attention(ca, &hn, l, eh, el, None, emask);
                for (a, b) in x.iter_mut().zip(&out) { *a += b; }
            }
            let hn = t5_layer_norm(&x, l, d, &blk.ln_ff, eps);
            let mut h = proj(self.fast, self.row1_gemv, &hn, l, d, &blk.wi, &blk.wit, dff, None);
            for v in h.iter_mut() { if *v < 0.0 { *v = 0.0; } }
            let ff = proj(self.fast, self.row1_gemv, &h, l, dff, &blk.wo, &blk.wot, d, None);
            for (a, b) in x.iter_mut().zip(&ff) { *a += b; }
        }
        t5_layer_norm(&x, l, d, &st.final_ln, eps)
    }

    /// One forward: context (may contain NaN) → 9×64 quantiles on the original scale.
    pub fn forward(&self, context: &[f32]) -> Vec<Vec<f32>> { self.forward_stages(context).0 }

    /// `forward` plus per-stage milliseconds (embedding, encoder, decoder, head).
    pub fn forward_stages(&self, context: &[f32]) -> (Vec<Vec<f32>>, Vec<(&'static str, f64)>) {
        use std::time::Instant;
        let mut stages = Vec::with_capacity(4);
        let cfg = &self.cfg;
        let ctx: &[f32] = if context.len() > cfg.context_length { &context[context.len() - cfg.context_length..] } else { context };
        let finite: Vec<f32> = ctx.iter().cloned().filter(|v| v.is_finite()).collect();
        let loc = if finite.is_empty() { 0.0 } else { finite.iter().sum::<f32>() / finite.len() as f32 };
        let mut scale = if finite.is_empty() { 1.0 } else { (finite.iter().map(|v| (v - loc) * (v - loc)).sum::<f32>() / finite.len() as f32).sqrt() };
        if scale == 0.0 { scale = 1e-5; }
        let p = cfg.patch;
        let pad = if ctx.len() % p == 0 { 0 } else { p - ctx.len() % p };
        let n_patches = (ctx.len() + pad) / p;
        let mut feats = Vec::with_capacity(n_patches * 2 * p);
        let mut attn_mask = Vec::with_capacity(n_patches + 1);
        for pi in 0..n_patches {
            let mut vals = vec![0.0f32; p]; let mut mask = vec![0.0f32; p];
            for t in 0..p {
                let idx = pi * p + t;
                if idx >= pad { let v = ctx[idx - pad]; if v.is_finite() { vals[t] = (v - loc) / scale; mask[t] = 1.0; } }
            }
            attn_mask.push(mask.iter().sum::<f32>() > 0.0);
            feats.extend(vals); feats.extend(mask);
        }
        let t = Instant::now();
        let mut embeds = self.in_embed.forward(&feats, n_patches, self.fast, self.row1_gemv);
        let mut l = n_patches;
        if cfg.use_reg_token { embeds.extend_from_slice(&self.shared[cfg.d_model..2 * cfg.d_model]); attn_mask.push(true); l += 1; }
        stages.push(("patch embedding", t.elapsed().as_secs_f64() * 1e3));
        let t = Instant::now();
        let enc = self.run_stack(&self.encoder, &embeds, l, &attn_mask, None);
        stages.push(("encoder", t.elapsed().as_secs_f64() * 1e3));
        let t = Instant::now();
        let dec_in = self.shared[..cfg.d_model].to_vec();
        let dec = self.run_stack(&self.decoder, &dec_in, 1, &[true], Some((&enc, l, &attn_mask)));
        stages.push(("decoder (1 token)", t.elapsed().as_secs_f64() * 1e3));
        let t = Instant::now();
        let head = self.out_embed.forward(&dec, 1, self.fast, self.row1_gemv);
        let nq = cfg.quantiles.len();
        let q = (0..nq).map(|qi| (0..cfg.prediction_length).map(|t| head[qi * cfg.prediction_length + t] * scale + loc).collect()).collect();
        stages.push(("quantile head", t.elapsed().as_secs_f64() * 1e3));
        (q, stages)
    }

    /// `ChronosBoltPipeline.predict`: direct block, then per-quantile paths batched and re-quantiled.
    /// Returns `[quantile][t]` and the number of forwards spent.
    pub fn predict(&self, context: &[f32], prediction_length: usize) -> (Vec<Vec<f32>>, usize) {
        let cfg = &self.cfg;
        let nq = cfg.quantiles.len();
        let first = self.forward(context);
        let mut forwards = 1;
        let mut out: Vec<Vec<f32>> = first.clone();
        let mut remaining = prediction_length as i64 - cfg.prediction_length as i64;
        if remaining <= 0 { for q in out.iter_mut() { q.truncate(prediction_length); } return (out, forwards); }
        let base: Vec<f32> = if context.len() > cfg.context_length { context[context.len() - cfg.context_length..].to_vec() } else { context.to_vec() };
        let mut paths: Vec<Vec<f32>> = (0..nq).map(|_| base.clone()).collect();
        let mut last = first;
        while remaining > 0 {
            for (path, q) in paths.iter_mut().zip(&last) { path.extend_from_slice(q); let n = path.len(); if n > cfg.context_length { path.drain(..n - cfg.context_length); } }
            let preds: Vec<Vec<Vec<f32>>> = paths.iter().map(|p| { forwards += 1; self.forward(p) }).collect();
            let mut next = vec![vec![0.0f32; cfg.prediction_length]; nq];
            let mut pool = Vec::with_capacity(nq * nq);
            for t in 0..cfg.prediction_length {
                pool.clear();
                for pr in &preds { for q in pr { pool.push(q[t]); } }
                pool.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
                for (qi, &lvl) in cfg.quantiles.iter().enumerate() { next[qi][t] = torch_quantile(&pool, lvl); }
            }
            for (o, n) in out.iter_mut().zip(&next) { o.extend_from_slice(n); }
            remaining -= cfg.prediction_length as i64;
            last = next;
        }
        for q in out.iter_mut() { q.truncate(prediction_length); }
        (out, forwards)
    }
}

/// torch.quantile default (linear interpolation) on a sorted slice.
pub fn torch_quantile(sorted: &[f32], q: f32) -> f32 {
    let pos = q as f64 * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize; let hi = (lo + 1).min(sorted.len() - 1);
    (sorted[lo] as f64 + (pos - lo as f64) * (sorted[hi] as f64 - sorted[lo] as f64)) as f32
}
