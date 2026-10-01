//! Chronos-2 inference (amazon/chronos-2, chronos-forecasting 2.3.1): arcsinh instance scaling,
//! 16-step patches with a time encoding, a residual patch embedding, a REG token, empty future
//! patches, a 12-layer encoder of RoPE time attention + group attention + T5 feed-forward, and a
//! residual quantile head over the future positions (21 quantiles × 16 steps per patch, up to 64
//! patches in one pass). Single-series only: group attention over a batch of one collapses to
//! `x += o(v(ln(x)))`. All products go through trueno's packed GEMM (spike 008 kernel).
use crate::safetensors::{Tensor, Weights};

#[derive(Clone, Debug)]
pub struct Config { pub d_model: usize, pub d_ff: usize, pub d_kv: usize, pub heads: usize, pub layers: usize, pub eps: f32, pub rope_theta: f32, pub context_length: usize, pub patch: usize, pub max_output_patches: usize, pub quantiles: Vec<f32>, pub time_scale: f32, pub use_arcsinh: bool, pub use_reg_token: bool }

impl Config {
    pub fn from_json(v: &serde_json::Value) -> Config {
        let c = &v["chronos_config"];
        let context_length = c["context_length"].as_u64().expect("ctx") as usize;
        Config { d_model: v["d_model"].as_u64().expect("d_model") as usize, d_ff: v["d_ff"].as_u64().expect("d_ff") as usize, d_kv: v["d_kv"].as_u64().expect("d_kv") as usize, heads: v["num_heads"].as_u64().expect("heads") as usize, layers: v["num_layers"].as_u64().expect("layers") as usize,
            eps: v["layer_norm_epsilon"].as_f64().expect("eps") as f32, rope_theta: v["rope_theta"].as_f64().unwrap_or(10000.0) as f32, context_length, patch: c["input_patch_size"].as_u64().expect("patch") as usize,
            max_output_patches: c["max_output_patches"].as_u64().unwrap_or(1) as usize, quantiles: c["quantiles"].as_array().expect("q").iter().map(|q| q.as_f64().expect("f") as f32).collect(),
            time_scale: c["time_encoding_scale"].as_u64().map_or(context_length as f32, |t| t as f32), use_arcsinh: c["use_arcsinh"].as_bool().unwrap_or(false), use_reg_token: c["use_reg_token"].as_bool().unwrap_or(false) }
    }
}

// ------------------------------------------------------------ primitives ----
/// x[rows×in] · Wᵀ[in×out] (+ b) through trueno's packed GEMM; `wt` is the transposed [in, out] weight.
pub static PARALLEL_GEMM: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub fn linear_t(x: &[f32], rows: usize, inp: usize, wt: &[f32], out: usize, b: Option<&[f32]>) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * out];
    if PARALLEL_GEMM.load(std::sync::atomic::Ordering::Relaxed) { trueno::blis::gemm(rows, out, inp, x, wt, &mut y).expect("gemm"); } else { trueno::blis::gemm_blis(rows, out, inp, x, wt, &mut y, None).expect("gemm_blis"); }
    if let Some(b) = b { for r in 0..rows { for o in 0..out { y[r * out + o] += b[o]; } } }
    y
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
/// [out, in] → [in, out]
pub fn transpose(w: &[f32], out: usize, inp: usize) -> Vec<f32> { let mut t = vec![0.0f32; w.len()]; for o in 0..out { for i in 0..inp { t[i * out + o] = w[o * inp + i]; } } t }

pub struct Lin { pub wt: Vec<f32>, pub inp: usize, pub out: usize }
pub struct Residual { pub h: Lin, pub bh: Vec<f32>, pub o: Lin, pub bo: Vec<f32>, pub r: Lin, pub br: Vec<f32> }
pub struct Attn { pub q: Lin, pub k: Lin, pub v: Lin, pub o: Lin }
pub struct Block { pub ln_t: Vec<f32>, pub time: Attn, pub ln_g: Vec<f32>, pub group: Attn, pub ln_ff: Vec<f32>, pub wi: Lin, pub wo: Lin }
pub struct Chronos2 { pub cfg: Config, pub reg: Vec<f32>, pub in_embed: Residual, pub out_embed: Residual, pub blocks: Vec<Block>, pub final_ln: Vec<f32> }

fn tensor<'a>(w: &'a Weights, name: &str) -> &'a Tensor { w.get(name).unwrap_or_else(|| panic!("missing tensor {name}")) }
fn take(w: &Weights, name: &str) -> Vec<f32> { tensor(w, name).data.clone() }
fn lin(w: &Weights, name: &str) -> Lin { let t = tensor(w, name); let (out, inp) = (t.shape[0], t.shape[1]); Lin { wt: transpose(&t.data, out, inp), inp, out } }
fn residual(w: &Weights, p: &str) -> Residual { Residual { h: lin(w, &format!("{p}.hidden_layer.weight")), bh: take(w, &format!("{p}.hidden_layer.bias")), o: lin(w, &format!("{p}.output_layer.weight")), bo: take(w, &format!("{p}.output_layer.bias")), r: lin(w, &format!("{p}.residual_layer.weight")), br: take(w, &format!("{p}.residual_layer.bias")) } }
fn attn(w: &Weights, p: &str) -> Attn { Attn { q: lin(w, &format!("{p}.q.weight")), k: lin(w, &format!("{p}.k.weight")), v: lin(w, &format!("{p}.v.weight")), o: lin(w, &format!("{p}.o.weight")) } }
impl Residual {
    pub fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut h = linear_t(x, rows, self.h.inp, &self.h.wt, self.h.out, Some(&self.bh));
        for v in h.iter_mut() { if *v < 0.0 { *v = 0.0; } }
        let mut out = linear_t(&h, rows, self.o.inp, &self.o.wt, self.o.out, Some(&self.bo));
        let res = linear_t(x, rows, self.r.inp, &self.r.wt, self.r.out, Some(&self.br));
        for (o, r) in out.iter_mut().zip(&res) { *o += r; }
        out
    }
}

pub struct Forward { pub loc: f32, pub scale: f32, pub n_patches: usize, pub attention_mask: Vec<bool>, pub patch_feat_first: Vec<f32>, pub patch_feat_last: Vec<f32>, pub embed_first: Vec<f32>, pub embed_last: Vec<f32>, pub hidden_first: Vec<f32>, pub hidden_reg: Vec<f32>, pub hidden_last: Vec<f32>, pub quantiles: Vec<Vec<f32>>, pub tokens: usize }

impl Chronos2 {
    pub fn load(w: &Weights, cfg: Config) -> Chronos2 {
        let shared = take(w, "shared.weight");
        let blocks = (0..cfg.layers).map(|i| { let p = format!("encoder.block.{i}.layer"); Block {
            ln_t: take(w, &format!("{p}.0.layer_norm.weight")), time: attn(w, &format!("{p}.0.self_attention")),
            ln_g: take(w, &format!("{p}.1.layer_norm.weight")), group: attn(w, &format!("{p}.1.self_attention")),
            ln_ff: take(w, &format!("{p}.2.layer_norm.weight")), wi: lin(w, &format!("{p}.2.mlp.wi.weight")), wo: lin(w, &format!("{p}.2.mlp.wo.weight")) } }).collect();
        Chronos2 { reg: shared[cfg.d_model..2 * cfg.d_model].to_vec(), in_embed: residual(w, "input_patch_embedding"), out_embed: residual(w, "output_patch_embedding"), blocks, final_ln: take(w, "encoder.final_layer_norm.weight"), cfg }
    }

    /// RoPE tables for positions 0..l: cos/sin[pos][i], i < d_kv, with the (freqs, freqs) layout.
    fn rope(&self, l: usize) -> (Vec<f32>, Vec<f32>) {
        let dk = self.cfg.d_kv; let half = dk / 2;
        let inv: Vec<f32> = (0..half).map(|i| 1.0 / self.cfg.rope_theta.powf((2 * i) as f32 / dk as f32)).collect();
        let (mut c, mut s) = (vec![0.0f32; l * dk], vec![0.0f32; l * dk]);
        for pos in 0..l { for i in 0..half { let f = pos as f32 * inv[i]; let (sn, cs) = f.sin_cos(); c[pos * dk + i] = cs; c[pos * dk + half + i] = cs; s[pos * dk + i] = sn; s[pos * dk + half + i] = sn; } }
        (c, s)
    }
    fn apply_rope(&self, x: &mut [f32], l: usize, cos: &[f32], sin: &[f32]) {
        let (h, dk) = (self.cfg.heads, self.cfg.d_kv); let half = dk / 2;
        let mut tmp = vec![0.0f32; dk];
        for pos in 0..l { for hh in 0..h {
            let v = &mut x[pos * h * dk + hh * dk..pos * h * dk + (hh + 1) * dk];
            for i in 0..half { tmp[i] = -v[half + i]; tmp[half + i] = v[i]; }
            for i in 0..dk { v[i] = v[i] * cos[pos * dk + i] + tmp[i] * sin[pos * dk + i]; }
        } }
    }

    /// Bidirectional multi-head attention without score scaling; masked keys get `f32::MIN` added.
    fn attention(&self, a: &Attn, x: &[f32], l: usize, key_mask: &[bool], rope: Option<(&[f32], &[f32])>) -> Vec<f32> {
        let (d, h, dk) = (self.cfg.d_model, self.cfg.heads, self.cfg.d_kv);
        let mut q = linear_t(x, l, d, &a.q.wt, h * dk, None);
        let mut k = linear_t(x, l, d, &a.k.wt, h * dk, None);
        let v = linear_t(x, l, d, &a.v.wt, h * dk, None);
        if let Some((c, s)) = rope { self.apply_rope(&mut q, l, c, s); self.apply_rope(&mut k, l, c, s); }
        let mut ctx = vec![0.0f32; l * h * dk];
        let (mut qh, mut kt, mut vh) = (vec![0.0f32; l * dk], vec![0.0f32; dk * l], vec![0.0f32; l * dk]);
        let mut scores = vec![0.0f32; l * l];
        let mut ch = vec![0.0f32; l * dk];
        for hh in 0..h {
            for i in 0..l { qh[i * dk..(i + 1) * dk].copy_from_slice(&q[i * h * dk + hh * dk..i * h * dk + (hh + 1) * dk]); vh[i * dk..(i + 1) * dk].copy_from_slice(&v[i * h * dk + hh * dk..i * h * dk + (hh + 1) * dk]); let kj = &k[i * h * dk + hh * dk..i * h * dk + (hh + 1) * dk]; for t in 0..dk { kt[t * l + i] = kj[t]; } }
            scores.iter_mut().for_each(|s| *s = 0.0);
            trueno::blis::gemm_blis(l, l, dk, &qh, &kt, &mut scores, None).expect("scores");
            for i in 0..l {
                let row = &mut scores[i * l..(i + 1) * l];
                let mut mx = f32::NEG_INFINITY;
                for j in 0..l { if !key_mask[j] { row[j] += f32::MIN; } if row[j] > mx { mx = row[j]; } }
                let mut den = 0.0f32;
                for j in 0..l { row[j] = (row[j] - mx).exp(); den += row[j]; }
                let inv = 1.0 / den; for j in 0..l { row[j] *= inv; }
            }
            ch.iter_mut().for_each(|s| *s = 0.0);
            trueno::blis::gemm_blis(l, dk, l, &scores, &vh, &mut ch, None).expect("context");
            for i in 0..l { ctx[i * h * dk + hh * dk..i * h * dk + (hh + 1) * dk].copy_from_slice(&ch[i * dk..(i + 1) * dk]); }
        }
        linear_t(&ctx, l, h * dk, &a.o.wt, d, None)
    }

    /// One pass: context (NaN = missing) and `num_output_patches` → 21 × (16·patches) quantiles.
    pub fn forward(&self, context: &[f32], num_output_patches: usize) -> Forward {
        let cfg = &self.cfg;
        let (d, p) = (cfg.d_model, cfg.patch);
        let ctx: &[f32] = if context.len() > cfg.context_length { &context[context.len() - cfg.context_length..] } else { context };
        // instance norm (nanmean / population std), then arcsinh
        let finite: Vec<f32> = ctx.iter().cloned().filter(|v| v.is_finite()).collect();
        let loc = if finite.is_empty() { 0.0 } else { finite.iter().sum::<f32>() / finite.len() as f32 };
        let mut scale = if finite.is_empty() { 1.0 } else { (finite.iter().map(|v| (v - loc) * (v - loc)).sum::<f32>() / finite.len() as f32).sqrt() };
        if scale == 0.0 { scale = 1e-5; }
        let pad = (p - ctx.len() % p) % p;
        let n_patches = (ctx.len() + pad) / p;
        let final_len = (n_patches * p) as f32;
        let mut feats = vec![0.0f32; n_patches * 3 * p];
        let mut mask = Vec::with_capacity(n_patches + 1 + num_output_patches);
        for pi in 0..n_patches {
            let f = &mut feats[pi * 3 * p..(pi + 1) * 3 * p];
            let mut any = false;
            for t in 0..p {
                let idx = pi * p + t;
                f[t] = (-final_len + idx as f32) / cfg.time_scale;
                if idx >= pad { let v = ctx[idx - pad]; if v.is_finite() { let s = (v - loc) / scale; f[p + t] = if cfg.use_arcsinh { s.asinh() } else { s }; f[2 * p + t] = 1.0; any = true; } }
            }
            mask.push(any);
        }
        let mut x = self.in_embed.forward(&feats, n_patches);
        let (embed_first, embed_last) = (x[..d].to_vec(), x[(n_patches - 1) * d..n_patches * d].to_vec());
        if cfg.use_reg_token { x.extend_from_slice(&self.reg); mask.push(true); }
        let n_ctx = x.len() / d;
        let mut ff = vec![0.0f32; num_output_patches * 3 * p];
        for pi in 0..num_output_patches { for t in 0..p { ff[pi * 3 * p + t] = (pi * p + t) as f32 / cfg.time_scale; } }
        x.extend(self.in_embed.forward(&ff, num_output_patches));
        for _ in 0..num_output_patches { mask.push(true); }
        let l = x.len() / d;
        let (cos, sin) = self.rope(l);
        for blk in &self.blocks {
            let hn = t5_layer_norm(&x, l, d, &blk.ln_t, cfg.eps);
            let sa = self.attention(&blk.time, &hn, l, &mask, Some((&cos, &sin)));
            for (a, b) in x.iter_mut().zip(&sa) { *a += b; }
            // group attention over a batch of one: softmax over a single key = 1 ⇒ x += o(v(ln(x)))
            let hn = t5_layer_norm(&x, l, d, &blk.ln_g, cfg.eps);
            let v = linear_t(&hn, l, d, &blk.group.v.wt, cfg.heads * cfg.d_kv, None);
            let g = linear_t(&v, l, cfg.heads * cfg.d_kv, &blk.group.o.wt, d, None);
            for (a, b) in x.iter_mut().zip(&g) { *a += b; }
            let hn = t5_layer_norm(&x, l, d, &blk.ln_ff, cfg.eps);
            let mut hdn = linear_t(&hn, l, d, &blk.wi.wt, cfg.d_ff, None);
            for v in hdn.iter_mut() { if *v < 0.0 { *v = 0.0; } }
            let f2 = linear_t(&hdn, l, cfg.d_ff, &blk.wo.wt, d, None);
            for (a, b) in x.iter_mut().zip(&f2) { *a += b; }
        }
        let x = t5_layer_norm(&x, l, d, &self.final_ln, cfg.eps);
        let fut = &x[(l - num_output_patches) * d..];
        let head = self.out_embed.forward(fut, num_output_patches); // [patch][q*16 + t]
        let nq = cfg.quantiles.len();
        let quantiles: Vec<Vec<f32>> = (0..nq).map(|qi| (0..num_output_patches * p).map(|i| { let (pi, t) = (i / p, i % p); let v = head[pi * nq * p + qi * p + t]; (if cfg.use_arcsinh { v.sinh() } else { v }) * scale + loc }).collect()).collect();
        Forward { loc, scale, n_patches, attention_mask: mask, patch_feat_first: feats[..3 * p].to_vec(), patch_feat_last: feats[(n_patches - 1) * 3 * p..n_patches * 3 * p].to_vec(), embed_first, embed_last, hidden_first: x[..d].to_vec(), hidden_reg: x[(n_ctx - 1) * d..n_ctx * d].to_vec(), hidden_last: x[(l - 1) * d..].to_vec(), quantiles, tokens: l }
    }

    /// The pipeline's direct path: ceil(h/16) patches (≤ max_output_patches), truncated to h.
    pub fn predict(&self, context: &[f32], h: usize) -> Result<(Vec<Vec<f32>>, usize), String> {
        let nop = (h + self.cfg.patch - 1) / self.cfg.patch;
        if nop > self.cfg.max_output_patches { return Err(format!("horizon {h} needs {nop} output patches > max_output_patches {}; the pipeline's long-horizon unrolling is not ported", self.cfg.max_output_patches)); }
        let f = self.forward(context, nop);
        Ok((f.quantiles.into_iter().map(|mut q| { q.truncate(h); q }).collect(), f.tokens))
    }
}
