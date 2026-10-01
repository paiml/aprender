//! Chronos-Bolt inference: instance scaling, patching, residual patch embedding, REG token,
//! T5 encoder (relative position buckets, no score scaling), one-token T5 decoder with
//! cross-attention, residual quantile head, inverse scaling, and the 9-path rollout.
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
/// y[r][o] = Σ_i x[r][i] w[o][i] (+ b[o]); w is [out, in] row-major as in torch.
pub fn linear(x: &[f32], rows: usize, inp: usize, w: &[f32], out: usize, b: Option<&[f32]>) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * out];
    for r in 0..rows {
        let xr = &x[r * inp..(r + 1) * inp];
        let yr = &mut y[r * out..(r + 1) * out];
        for o in 0..out {
            let wo = &w[o * inp..(o + 1) * inp];
            yr[o] = dot8(xr, wo) + b.map_or(0.0, |b| b[o]);
        }
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

pub struct Attn { pub q: Vec<f32>, pub k: Vec<f32>, pub v: Vec<f32>, pub o: Vec<f32>, pub qt: Vec<f32>, pub kt: Vec<f32>, pub vt: Vec<f32>, pub ot: Vec<f32> }
pub struct Block { pub ln_sa: Vec<f32>, pub sa: Attn, pub ln_ca: Option<Vec<f32>>, pub ca: Option<Attn>, pub ln_ff: Vec<f32>, pub wi: Vec<f32>, pub wo: Vec<f32>, pub wit: Vec<f32>, pub wot: Vec<f32> }
pub struct Stack { pub blocks: Vec<Block>, pub rel_bias: Vec<f32>, pub final_ln: Vec<f32>, pub is_decoder: bool }
pub struct Residual { pub wh: Vec<f32>, pub bh: Vec<f32>, pub wo: Vec<f32>, pub bo: Vec<f32>, pub wr: Vec<f32>, pub br: Vec<f32>, pub inp: usize, pub hid: usize, pub out: usize }
pub struct Bolt { pub fast: bool, pub cfg: Config, pub shared: Vec<f32>, pub in_embed: Residual, pub out_embed: Residual, pub encoder: Stack, pub decoder: Stack }

fn take(w: &Weights, name: &str) -> Vec<f32> { w.get(name).unwrap_or_else(|| panic!("missing tensor {name}")).data.clone() }
fn tensor<'a>(w: &'a Weights, name: &str) -> &'a Tensor { w.get(name).unwrap_or_else(|| panic!("missing tensor {name}")) }

fn residual(w: &Weights, prefix: &str) -> Residual {
    let wh = tensor(w, &format!("{prefix}.hidden_layer.weight")); let wo = tensor(w, &format!("{prefix}.output_layer.weight"));
    Residual { inp: wh.shape[1], hid: wh.shape[0], out: wo.shape[0], wh: wh.data.clone(), bh: take(w, &format!("{prefix}.hidden_layer.bias")), wo: wo.data.clone(), bo: take(w, &format!("{prefix}.output_layer.bias")), wr: take(w, &format!("{prefix}.residual_layer.weight")), br: take(w, &format!("{prefix}.residual_layer.bias")) }
}
impl Residual {
    pub fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut h = linear(x, rows, self.inp, &self.wh, self.hid, Some(&self.bh));
        for v in h.iter_mut() { if *v < 0.0 { *v = 0.0; } }
        let mut out = linear(&h, rows, self.hid, &self.wo, self.out, Some(&self.bo));
        let res = linear(x, rows, self.inp, &self.wr, self.out, Some(&self.br));
        for (o, r) in out.iter_mut().zip(&res) { *o += r; }
        out
    }
}

/// [out, in] → [in, out]
pub fn transpose(w: &[f32], out: usize, inp: usize) -> Vec<f32> { let mut t = vec![0.0f32; w.len()]; for o in 0..out { for i in 0..inp { t[i * out + o] = w[o * inp + i]; } } t }
fn attn(w: &Weights, prefix: &str) -> Attn {
    let (q, k, v, o) = (take(w, &format!("{prefix}.q.weight")), take(w, &format!("{prefix}.k.weight")), take(w, &format!("{prefix}.v.weight")), take(w, &format!("{prefix}.o.weight")));
    let s = tensor(w, &format!("{prefix}.q.weight")).shape.clone();
    Attn { qt: transpose(&q, s[0], s[1]), kt: transpose(&k, s[0], s[1]), vt: transpose(&v, s[0], s[1]), ot: transpose(&o, s[0], s[1]), q, k, v, o }
}
/// x[rows×in] · Wᵀ[in×out] through trueno's SIMD GEMM (weights pre-transposed).
pub fn linear_fast(x: &[f32], rows: usize, inp: usize, wt: &[f32], out: usize, b: Option<&[f32]>) -> Vec<f32> {
    if rows == 1 {
        let mut y = vec![0.0f32; out];
        trueno::blis::gemv::gemv(inp, out, x, wt, &mut y);
        if let Some(b) = b { for (v, bb) in y.iter_mut().zip(b) { *v += bb; } }
        return y;
    }
    // packed BLIS-style GEMM: C[rows×out] = A[rows×in] · B[in×out]
    let mut y = vec![0.0f32; rows * out];
    trueno::blis::gemm_blis(rows, out, inp, x, wt, &mut y, None).expect("gemm_blis");
    if let Some(b) = b { for r in 0..rows { for o in 0..out { y[r * out + o] += b[o]; } } }
    y
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
        Bolt { fast: false, shared: take(w, "shared.weight"), in_embed: residual(w, "input_patch_embedding"), out_embed: residual(w, "output_patch_embedding"), encoder: stack(w, "encoder", cfg.enc_layers, false), decoder: stack(w, "decoder", cfg.dec_layers, true), cfg }
    }

    /// Multi-head attention; `bias[h*lq*lk ..]` optional (relative position), `key_mask[k]` false → excluded.
    pub fn attention(&self, a: &Attn, xq: &[f32], lq: usize, xkv: &[f32], lk: usize, bias: Option<&[f32]>, key_mask: &[bool]) -> Vec<f32> {
        let (d, h, dk) = (self.cfg.d_model, self.cfg.heads, self.cfg.d_kv);
        let (q, k, v) = if self.fast { (linear_fast(xq, lq, d, &a.qt, h * dk, None), linear_fast(xkv, lk, d, &a.kt, h * dk, None), linear_fast(xkv, lk, d, &a.vt, h * dk, None)) } else { (linear(xq, lq, d, &a.q, h * dk, None), linear(xkv, lk, d, &a.k, h * dk, None), linear(xkv, lk, d, &a.v, h * dk, None)) };
        let mut ctx = vec![0.0f32; lq * h * dk];
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
        if self.fast { linear_fast(&ctx, lq, h * dk, &a.ot, d, None) } else { linear(&ctx, lq, h * dk, &a.o, d, None) }
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
            let mut h = if self.fast { linear_fast(&hn, l, d, &blk.wit, dff, None) } else { linear(&hn, l, d, &blk.wi, dff, None) };
            for v in h.iter_mut() { if *v < 0.0 { *v = 0.0; } }
            let ff = if self.fast { linear_fast(&h, l, dff, &blk.wot, d, None) } else { linear(&h, l, dff, &blk.wo, d, None) };
            for (a, b) in x.iter_mut().zip(&ff) { *a += b; }
        }
        t5_layer_norm(&x, l, d, &st.final_ln, eps)
    }

    /// One forward: context (may contain NaN) → 9×64 quantiles on the original scale, plus the ladder tensors.
    pub fn forward(&self, context: &[f32]) -> Forward {
        let cfg = &self.cfg;
        let ctx: &[f32] = if context.len() > cfg.context_length { &context[context.len() - cfg.context_length..] } else { context };
        // instance norm (nanmean / population std), nan_to_num semantics
        let finite: Vec<f32> = ctx.iter().cloned().filter(|v| v.is_finite()).collect();
        let loc = if finite.is_empty() { 0.0 } else { finite.iter().sum::<f32>() / finite.len() as f32 };
        let mut scale = if finite.is_empty() { 1.0 } else { (finite.iter().map(|v| (v - loc) * (v - loc)).sum::<f32>() / finite.len() as f32).sqrt() };
        if scale == 0.0 { scale = 1e-5; }
        // left-pad with NaN to a multiple of the patch size, then patch
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
        let mut embeds = self.in_embed.forward(&feats, n_patches);
        let mut l = n_patches;
        if cfg.use_reg_token { embeds.extend_from_slice(&self.shared[cfg.d_model..2 * cfg.d_model]); attn_mask.push(true); l += 1; }
        let enc = self.run_stack(&self.encoder, &embeds, l, &attn_mask, None);
        let dec_in = self.shared[..cfg.d_model].to_vec();
        let dec = self.run_stack(&self.decoder, &dec_in, 1, &[true], Some((&enc, l, &attn_mask)));
        let head = self.out_embed.forward(&dec, 1);
        let nq = cfg.quantiles.len();
        let quantiles: Vec<Vec<f32>> = (0..nq).map(|qi| (0..cfg.prediction_length).map(|t| head[qi * cfg.prediction_length + t] * scale + loc).collect()).collect();
        Forward { loc, scale, attention_mask: attn_mask, input_embeds: embeds, encoder_hidden: enc, decoder_hidden: dec, quantiles }
    }

    /// `ChronosBoltPipeline.predict`: direct block, then per-quantile paths batched and re-quantiled.
    pub fn predict(&self, context: &[f32], prediction_length: usize) -> Vec<Vec<f32>> {
        let cfg = &self.cfg;
        let nq = cfg.quantiles.len();
        let first = self.forward(context).quantiles;
        let mut out: Vec<Vec<f32>> = first.clone();
        let mut remaining = prediction_length as i64 - cfg.prediction_length as i64;
        if remaining <= 0 { for q in out.iter_mut() { q.truncate(prediction_length); } return out; }
        let base: Vec<f32> = if context.len() > cfg.context_length { context[context.len() - cfg.context_length..].to_vec() } else { context.to_vec() };
        let mut paths: Vec<Vec<f32>> = (0..nq).map(|_| base.clone()).collect();
        let mut last = first;
        while remaining > 0 {
            for (path, q) in paths.iter_mut().zip(&last) { path.extend_from_slice(q); let n = path.len(); if n > cfg.context_length { path.drain(..n - cfg.context_length); } }
            let preds: Vec<Vec<Vec<f32>>> = paths.iter().map(|p| self.forward(p).quantiles).collect(); // [path][q][t]
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
        out
    }
}

/// torch.quantile default (linear interpolation) on a sorted slice.
pub fn torch_quantile(sorted: &[f32], q: f32) -> f32 {
    let pos = q as f64 * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize; let hi = (lo + 1).min(sorted.len() - 1);
    (sorted[lo] as f64 + (pos - lo as f64) * (sorted[hi] as f64 - sorted[lo] as f64)) as f32
}

pub struct Forward { pub loc: f32, pub scale: f32, pub attention_mask: Vec<bool>, pub input_embeds: Vec<f32>, pub encoder_hidden: Vec<f32>, pub decoder_hidden: Vec<f32>, pub quantiles: Vec<Vec<f32>> }

/// Stage timing of one encoder forward (129 tokens) to see what dominates.
impl Bolt {
    pub fn profile(&self, context: &[f32]) -> Vec<(String, f64)> {
        use std::time::Instant;
        let cfg = &self.cfg;
        let ctx: &[f32] = if context.len() > cfg.context_length { &context[context.len() - cfg.context_length..] } else { context };
        let l = ctx.len() / cfg.patch + 1;
        let (d, dff, h, dk) = (cfg.d_model, cfg.d_ff, cfg.heads, cfg.d_kv);
        let x = vec![0.1f32; l * d];
        let mask = vec![true; l];
        let mut out = Vec::new();
        let t = Instant::now(); let bias = self.position_bias(&self.encoder, l, l); out.push(("position bias (129×129×4 buckets)".into(), t.elapsed().as_secs_f64() * 1e3));
        let blk = &self.encoder.blocks[0];
        let t = Instant::now(); let hn = t5_layer_norm(&x, l, d, &blk.ln_sa, cfg.eps); out.push(("layer norm".into(), t.elapsed().as_secs_f64() * 1e3));
        let t = Instant::now(); let _ = if self.fast { linear_fast(&hn, l, d, &blk.sa.qt, h * dk, None) } else { linear(&hn, l, d, &blk.sa.q, h * dk, None) }; out.push(("one 256→256 projection".into(), t.elapsed().as_secs_f64() * 1e3));
        let t = Instant::now(); let _ = self.attention(&blk.sa, &hn, l, &hn, l, Some(&bias), &mask); out.push(("self-attention (4 projections + scores + context)".into(), t.elapsed().as_secs_f64() * 1e3));
        let t = Instant::now(); let hh = if self.fast { linear_fast(&hn, l, d, &blk.wit, dff, None) } else { linear(&hn, l, d, &blk.wi, dff, None) }; out.push(("FF wi 256→1024".into(), t.elapsed().as_secs_f64() * 1e3));
        let t = Instant::now(); let _ = if self.fast { linear_fast(&hh, l, dff, &blk.wot, d, None) } else { linear(&hh, l, dff, &blk.wo, d, None) }; out.push(("FF wo 1024→256".into(), t.elapsed().as_secs_f64() * 1e3));
        let t = Instant::now(); let _ = self.run_stack(&self.encoder, &x, l, &mask, None); out.push(("full encoder (4 layers)".into(), t.elapsed().as_secs_f64() * 1e3));
        out
    }
}
