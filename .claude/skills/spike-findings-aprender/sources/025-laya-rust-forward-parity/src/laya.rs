//! Laya decision model forward in plain Rust: ModernBERT-large encoder + Laya's 2-layer `nn.TransformerEncoder`
//! head + shared marker scorer. One bidirectional pass per question row; no generation, no KV cache.
//!
//! Every dense product goes through [`Linear::forward`] as `C^T = W . X^T` on trueno's BLIS GEMM (the spike-020
//! layout: the checkpoint's `[out, in]` row-major weight IS the GEMM's A operand, so no weight is transposed), with
//! W's output rows banded across the rayon pool. Weights are F16 in the checkpoint and widened to f32 at load.
//!
//! Reference semantics (transformers 5.17 `modeling_modernbert.py`, Laya `laya/common.py` @ 4066d5d):
//! embeddings = LayerNorm(tok_embeddings(ids)), no position embedding; layer 0 has no attn_norm (Identity);
//! attention = fused Wqkv (no bias) -> rotate-half RoPE (theta 160k global / 10k local) -> scaled dot product,
//! local layers masked to |i - j| <= local_attention / 2; MLP = Wi -> chunk(input, gate) -> gelu(input) * gate -> Wo;
//! final LayerNorm; head: h + type_emb[qtype] -> 2 x pre-norm TransformerEncoderLayer (16 heads, relu FFN 4d, biases);
//! scorer on each [MASK] marker: LayerNorm -> Linear -> GELU(erf) -> Linear -> logit; p = softmax(z / T_bucket).
#![allow(clippy::many_single_char_names)]

use rayon::prelude::*;
use std::collections::HashMap;

pub const EPS: f32 = 1e-5;

/// `[rows, cols]` -> `[cols, rows]`, 32x32 tiles, parallel over output bands (spike 020).
pub fn transpose(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    const T: usize = 32;
    let mut dst = vec![0.0f32; rows * cols];
    dst.par_chunks_mut(T * rows).enumerate().for_each(|(bi, band)| {
        let c0 = bi * T;
        let cn = band.len() / rows;
        for r0 in (0..rows).step_by(T) {
            for c in 0..cn {
                for r in r0..(r0 + T).min(rows) {
                    band[c * rows + r] = src[r * cols + c0 + c];
                }
            }
        }
    });
    dst
}

pub struct Linear {
    pub w: Vec<f32>, // [out, inp] row-major, as stored
    pub b: Option<Vec<f32>>,
    pub out: usize,
    pub inp: usize,
}

impl Linear {
    /// `x` is `[m, inp]`; returns `[m, out]`.
    pub fn forward(&self, x: &[f32], m: usize) -> Vec<f32> {
        let (n, k) = (self.out, self.inp);
        let xt = transpose(x, m, k);
        let mut ct = vec![0.0f32; n * m];
        let bands = (rayon::current_num_threads() * 2).max(1);
        let band = n.div_ceil(bands).next_multiple_of(8).max(8);
        ct.par_chunks_mut(band * m).enumerate().for_each(|(i, c)| {
            let r0 = i * band;
            let rows = c.len() / m;
            trueno::blis::gemm_blis(rows, m, k, &self.w[r0 * k..(r0 + rows) * k], &xt, c, None).expect("gemm_blis");
        });
        let mut y = transpose(&ct, n, m);
        if let Some(b) = &self.b {
            y.par_chunks_mut(n).for_each(|r| r.iter_mut().zip(b).for_each(|(v, bb)| *v += bb));
        }
        y
    }
}

pub fn layer_norm(x: &[f32], d: usize, w: &[f32], b: Option<&[f32]>) -> Vec<f32> {
    let mut y = vec![0.0f32; x.len()];
    y.par_chunks_mut(d).zip(x.par_chunks(d)).for_each(|(o, r)| {
        let mean = r.iter().map(|&v| f64::from(v)).sum::<f64>() / d as f64;
        let var = r.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() / d as f64;
        let inv = 1.0 / (var + f64::from(EPS)).sqrt();
        for i in 0..d {
            let v = ((f64::from(r[i]) - mean) * inv) as f32 * w[i];
            o[i] = b.map_or(v, |b| v + b[i]);
        }
    });
    y
}

pub fn gelu(x: f32) -> f32 {
    let x = f64::from(x);
    (0.5 * x * (1.0 + libm::erf(x / std::f64::consts::SQRT_2))) as f32
}

/// Rotate-half RoPE in place on `[L, heads * hd]`, position = row index.
fn rope(x: &mut [f32], l: usize, heads: usize, hd: usize, theta: f64) {
    let half = hd / 2;
    // torch: inv_freq = 1 / base ** (arange(0, hd, 2) / hd) in f32; angle = pos * inv_freq in f32
    let inv: Vec<f32> = (0..half).map(|p| (1.0 / theta.powf((2 * p) as f64 / hd as f64)) as f32).collect();
    x.par_chunks_mut(heads * hd).enumerate().take(l).for_each(|(pos, row)| {
        for h in 0..heads {
            let v = &mut row[h * hd..(h + 1) * hd];
            for p in 0..half {
                let ang = pos as f32 * inv[p];
                let (s, c) = ang.sin_cos();
                let (a, b) = (v[p], v[p + half]);
                v[p] = a * c - b * s;
                v[p + half] = b * c + a * s;
            }
        }
    });
}

/// Bidirectional multi-head attention over one row. `q`,`k`,`v` are `[L, heads*hd]`; `window` = Some(w) keeps
/// |i - j| <= w. Returns `[L, heads*hd]`.
fn attention(q: &[f32], k: &[f32], v: &[f32], l: usize, heads: usize, hd: usize, window: Option<usize>) -> Vec<f32> {
    let d = heads * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0.0f32; l * d];
    out.par_chunks_mut(d).enumerate().for_each(|(i, o)| {
        let (lo, hi) = match window {
            Some(w) => (i.saturating_sub(w), (i + w + 1).min(l)),
            None => (0, l),
        };
        let mut s = vec![0.0f32; hi - lo];
        for h in 0..heads {
            let qi = &q[i * d + h * hd..i * d + (h + 1) * hd];
            let mut mx = f32::NEG_INFINITY;
            for (jj, sv) in s.iter_mut().enumerate() {
                let j = lo + jj;
                let kj = &k[j * d + h * hd..j * d + (h + 1) * hd];
                *sv = qi.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * scale;
                mx = mx.max(*sv);
            }
            let mut z = 0.0f32;
            for sv in s.iter_mut() {
                *sv = (*sv - mx).exp();
                z += *sv;
            }
            let oh = &mut o[h * hd..(h + 1) * hd];
            for (jj, &w) in s.iter().enumerate() {
                let j = lo + jj;
                let vj = &v[j * d + h * hd..j * d + (h + 1) * hd];
                let w = w / z;
                oh.iter_mut().zip(vj).for_each(|(a, b)| *a += w * b);
            }
        }
    });
    out
}

pub struct EncLayer {
    attn_norm: Option<Vec<f32>>,
    wqkv: Linear,
    wo: Linear,
    mlp_norm: Vec<f32>,
    wi: Linear,
    wo_mlp: Linear,
    global: bool,
}

pub struct HeadLayer {
    norm1: (Vec<f32>, Vec<f32>),
    in_proj: Linear,
    out_proj: Linear,
    norm2: (Vec<f32>, Vec<f32>),
    lin1: Linear,
    lin2: Linear,
}

pub struct Laya {
    pub d: usize,
    heads: usize,
    hd: usize,
    tok_emb: Vec<f32>,
    emb_norm: Vec<f32>,
    layers: Vec<EncLayer>,
    final_norm: Vec<f32>,
    type_emb: Vec<f32>,
    head: Vec<HeadLayer>,
    sc_norm: (Vec<f32>, Vec<f32>),
    sc_1: Linear,
    sc_2: Linear,
    pub window: usize,
    theta_global: f64,
    theta_local: f64,
}

/// Widen every F16/F32 tensor of a safetensors file to f32.
pub fn load_tensors(bytes: &[u8]) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let st = safetensors::SafeTensors::deserialize(bytes).expect("safetensors");
    st.tensors()
        .into_par_iter()
        .map(|(name, t)| {
            let data: Vec<f32> = match t.dtype() {
                safetensors::Dtype::F16 => {
                    t.data().chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect()
                }
                safetensors::Dtype::F32 => t.data().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
                other => panic!("{name}: unsupported dtype {other:?}"),
            };
            (name, (data, t.shape().to_vec()))
        })
        .collect()
}

impl Laya {
    /// `cfg` is the encoder `config.json`; `n_head` is `rl_agent_config.json::head_layers`.
    pub fn from_tensors(mut t: HashMap<String, (Vec<f32>, Vec<usize>)>, cfg: &serde_json::Value, n_head: usize) -> Laya {
        let mut take = |n: &str| t.remove(n).unwrap_or_else(|| panic!("missing tensor {n}"));
        let lin = |(w, s): (Vec<f32>, Vec<usize>), b: Option<(Vec<f32>, Vec<usize>)>| Linear { out: s[0], inp: s[1], w, b: b.map(|x| x.0) };
        let d = cfg["hidden_size"].as_u64().expect("hidden_size") as usize;
        let heads = cfg["num_attention_heads"].as_u64().expect("heads") as usize;
        let n_layers = cfg["num_hidden_layers"].as_u64().expect("layers") as usize;
        let every = cfg["global_attn_every_n_layers"].as_u64().unwrap_or(3) as usize;
        let rope = &cfg["rope_parameters"];
        let theta_global = rope["full_attention"]["rope_theta"].as_f64().expect("global theta");
        let theta_local = rope["sliding_attention"]["rope_theta"].as_f64().expect("local theta");
        let window = cfg["local_attention"].as_u64().unwrap_or(128) as usize / 2;
        let mut layers = Vec::new();
        for i in 0..n_layers {
            let p = format!("encoder.layers.{i}.");
            layers.push(EncLayer {
                attn_norm: if i == 0 { None } else { Some(take(&format!("{p}attn_norm.weight")).0) },
                wqkv: lin(take(&format!("{p}attn.Wqkv.weight")), None),
                wo: lin(take(&format!("{p}attn.Wo.weight")), None),
                mlp_norm: take(&format!("{p}mlp_norm.weight")).0,
                wi: lin(take(&format!("{p}mlp.Wi.weight")), None),
                wo_mlp: lin(take(&format!("{p}mlp.Wo.weight")), None),
                global: i % every == 0,
            });
        }
        let mut head = Vec::new();
        for hi in 0..n_head {
            let p = format!("head.layers.{hi}.");
            head.push(HeadLayer {
                norm1: (take(&format!("{p}norm1.weight")).0, take(&format!("{p}norm1.bias")).0),
                in_proj: lin(take(&format!("{p}self_attn.in_proj_weight")), Some(take(&format!("{p}self_attn.in_proj_bias")))),
                out_proj: lin(take(&format!("{p}self_attn.out_proj.weight")), Some(take(&format!("{p}self_attn.out_proj.bias")))),
                norm2: (take(&format!("{p}norm2.weight")).0, take(&format!("{p}norm2.bias")).0),
                lin1: lin(take(&format!("{p}linear1.weight")), Some(take(&format!("{p}linear1.bias")))),
                lin2: lin(take(&format!("{p}linear2.weight")), Some(take(&format!("{p}linear2.bias")))),
            });
        }
        Laya {
            d,
            heads,
            hd: d / heads,
            tok_emb: take("encoder.embeddings.tok_embeddings.weight").0,
            emb_norm: take("encoder.embeddings.norm.weight").0,
            layers,
            final_norm: take("encoder.final_norm.weight").0,
            type_emb: take("type_emb.weight").0,
            head,
            sc_norm: (take("scorer.0.weight").0, take("scorer.0.bias").0),
            sc_1: lin(take("scorer.1.weight"), Some(take("scorer.1.bias"))),
            sc_2: lin(take("scorer.3.weight"), Some(take("scorer.3.bias"))),
            window,
            theta_global,
            theta_local,
        }
    }

    /// Scorer logits for one question row; `tap(name, [L,d])` sees every ladder block.
    pub fn forward(&self, ids: &[u32], markers: &[usize], qtype: usize, mut tap: impl FnMut(&str, &[f32])) -> Vec<f32> {
        let (d, l) = (self.d, ids.len());
        let mut x: Vec<f32> = ids.iter().flat_map(|&t| self.tok_emb[t as usize * d..(t as usize + 1) * d].iter().copied()).collect();
        x = layer_norm(&x, d, &self.emb_norm, None);
        tap("emb", &x);
        for (li, ly) in self.layers.iter().enumerate() {
            let xn = ly.attn_norm.as_ref().map_or_else(|| x.clone(), |w| layer_norm(&x, d, w, None));
            let qkv = ly.wqkv.forward(&xn, l);
            let (mut q, mut k, mut v) = (vec![0.0f32; l * d], vec![0.0f32; l * d], vec![0.0f32; l * d]);
            for i in 0..l {
                let r = &qkv[i * 3 * d..(i + 1) * 3 * d];
                q[i * d..(i + 1) * d].copy_from_slice(&r[..d]);
                k[i * d..(i + 1) * d].copy_from_slice(&r[d..2 * d]);
                v[i * d..(i + 1) * d].copy_from_slice(&r[2 * d..]);
            }
            let theta = if ly.global { self.theta_global } else { self.theta_local };
            rope(&mut q, l, self.heads, self.hd, theta);
            rope(&mut k, l, self.heads, self.hd, theta);
            let a = attention(&q, &k, &v, l, self.heads, self.hd, if ly.global { None } else { Some(self.window) });
            let a = ly.wo.forward(&a, l);
            x.par_iter_mut().zip(a.par_iter()).for_each(|(h, o)| *h += o);
            let xn = layer_norm(&x, d, &ly.mlp_norm, None);
            let h = ly.wi.forward(&xn, l);
            let inter = ly.wi.out / 2;
            let mut g = vec![0.0f32; l * inter];
            g.par_chunks_mut(inter).zip(h.par_chunks(2 * inter)).for_each(|(o, r)| {
                for j in 0..inter {
                    o[j] = gelu(r[j]) * r[inter + j];
                }
            });
            let m = ly.wo_mlp.forward(&g, l);
            x.par_iter_mut().zip(m.par_iter()).for_each(|(h, o)| *h += o);
            tap(&format!("layer{li}"), &x);
        }
        x = layer_norm(&x, d, &self.final_norm, None);
        tap("final", &x);
        let te = &self.type_emb[qtype * d..(qtype + 1) * d];
        x.par_chunks_mut(d).for_each(|r| r.iter_mut().zip(te).for_each(|(a, b)| *a += b));
        for (hi, hl) in self.head.iter().enumerate() {
            let xn = layer_norm(&x, d, &hl.norm1.0, Some(&hl.norm1.1));
            let qkv = hl.in_proj.forward(&xn, l);
            let (mut q, mut k, mut v) = (vec![0.0f32; l * d], vec![0.0f32; l * d], vec![0.0f32; l * d]);
            for i in 0..l {
                let r = &qkv[i * 3 * d..(i + 1) * 3 * d];
                q[i * d..(i + 1) * d].copy_from_slice(&r[..d]);
                k[i * d..(i + 1) * d].copy_from_slice(&r[d..2 * d]);
                v[i * d..(i + 1) * d].copy_from_slice(&r[2 * d..]);
            }
            let a = attention(&q, &k, &v, l, d / 64, 64, None);
            let a = hl.out_proj.forward(&a, l);
            x.par_iter_mut().zip(a.par_iter()).for_each(|(h, o)| *h += o);
            let xn = layer_norm(&x, d, &hl.norm2.0, Some(&hl.norm2.1));
            let mut f = hl.lin1.forward(&xn, l);
            f.par_iter_mut().for_each(|v| *v = v.max(0.0));
            let f = hl.lin2.forward(&f, l);
            x.par_iter_mut().zip(f.par_iter()).for_each(|(h, o)| *h += o);
            tap(&format!("head{hi}"), &x);
        }
        let m: Vec<f32> = markers.iter().flat_map(|&p| x[p * d..(p + 1) * d].iter().copied()).collect();
        tap("m_opts", &m);
        let kk = markers.len();
        let mn = layer_norm(&m, d, &self.sc_norm.0, Some(&self.sc_norm.1));
        let mut s = self.sc_1.forward(&mn, kk);
        s.iter_mut().for_each(|v| *v = gelu(*v));
        self.sc_2.forward(&s, kk)
    }
}

/// Laya's temperature bucket (`laya/common.py::temp_bucket`) clamped to [0.5, 5] (`clamp_temperature`).
pub fn temperature(cfg: &serde_json::Value, qtype: usize, k: usize) -> f32 {
    let names = ["choice", "score", "noul"];
    let size = if k <= 2 { "2" } else if k <= 5 { "3-5" } else if k <= 10 { "6-10" } else { "11+" };
    let key = format!("{}:{size}", names[qtype]);
    let t = cfg["temperature_by_options"][&key].as_f64().or_else(|| cfg["temperature"][qtype].as_f64()).unwrap_or(1.0);
    if t.is_finite() { t.clamp(0.5, 5.0) as f32 } else { 1.0 }
}

pub fn softmax_t(z: &[f32], t: f32) -> Vec<f32> {
    let m = z.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b / t));
    let e: Vec<f64> = z.iter().map(|&v| f64::from(v / t - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|v| (v / s) as f32).collect()
}

/// Port of `laya.common.build_sequence` (string state, no option reordering, right truncation).
pub struct Builder {
    pub tok: tokenizers::Tokenizer,
    pub cls: u32,
    pub sep: u32,
    pub mask: u32,
    pub mask_token: String,
    pub max_len: usize,
    pub head_max_len: usize,
}

impl Builder {
    fn enc(&self, s: &str) -> Vec<u32> {
        self.tok.encode(s, false).expect("tokenize").get_ids().to_vec()
    }
    pub fn build(&self, state: &str, t: &str, ins: &str, options: &[String]) -> (Vec<u32>, Vec<usize>) {
        let mt = self.mask_token.as_str();
        let ins = ins.replace(mt, " ");
        let head_ids = self.enc(&format!("{t} question: {ins}"));
        let mut opt_ids: Vec<Vec<u32>> = options
            .iter()
            .map(|o| {
                let mut v = vec![self.mask];
                v.extend(self.enc(&format!(" {}", o.replace(mt, " "))).into_iter().take(48));
                v
            })
            .collect();
        let hm = self.head_max_len as isize;
        let mut budget = hm - opt_ids.iter().map(|o| o.len() as isize).sum::<isize>();
        if budget < 16 {
            let per = 4.max((hm - 16) / (opt_ids.len().max(1) as isize)) as usize;
            opt_ids.iter_mut().for_each(|o| o.truncate(per));
            budget = hm - opt_ids.iter().map(|o| o.len() as isize).sum::<isize>();
        }
        let keep = 8.max(budget) as usize;
        let mut ids = vec![self.cls];
        ids.extend(head_ids.into_iter().take(keep));
        ids.push(self.sep);
        let mut markers = Vec::new();
        for o in opt_ids {
            markers.push(ids.len());
            ids.extend(o);
        }
        ids.push(self.sep);
        let room = self.max_len.saturating_sub(ids.len() + 1);
        ids.extend(self.enc(&state.replace(mt, " ")).into_iter().take(room));
        ids.push(self.sep);
        ids.truncate(self.max_len);
        markers.retain(|&m| m < self.max_len);
        (ids, markers)
    }
}
