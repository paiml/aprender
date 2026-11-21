//! `qwen4_exp` gated RMSNorm, n-gram hashed embedding and per-layer embedding
//! (PLE) with its dilated causal conv (#7175, master #7169).
//!
//! Pure fp32 CPU reference ops on row-major slices, validated against the
//! transformers goldens in `tests/fixtures/qwen4_exp/goldens/` by
//! `tests/qwen4_exp_embed_parity.rs`. Decode (#7332): [`NgramState`] carries
//! the n-gram context and [`PleState`] the dilated-conv history. Reference:
//! `modeling_qwen4_exp.py` @ transformers 36e937ae7 (`RMSNormGated` :170,
//! `NGramEmbedding` :1037-1178, `PLELayer` :1180-1253).

const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const SPLITMIX_M1: u64 = 0xBF58_476D_1CE4_E5B9;
const SPLITMIX_M2: u64 = 0x94D0_49BB_1331_11EB;
const PRIME_1: u64 = 10007;

/// Activation applied to the gate in [`gated_rms_norm`] (`output_gate_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateAct {
    /// `output_gate_type = "sigmoid"` (Qwen 4).
    Sigmoid,
    /// `output_gate_type = "silu"` (Qwen 3.5).
    Silu,
}

/// N-gram hashed embedding hyperparameters (one PLE layer).
#[derive(Debug, Clone)]
pub struct NgramConfig {
    /// Unigram vocabulary size (divisor bound of the multipliers).
    pub vocab_size: u64,
    /// `ngram_size` (n-grams 2..=ngram_size are hashed).
    pub ngram_size: usize,
    /// Hash heads per n-gram order.
    pub heads_per_ngram: usize,
    /// `ngram_vocab_size_base` (head vocab = nth prime after `base - 1`).
    pub ngram_vocab_size_base: u64,
    /// `config.seed` (default 1234; read it from the checkpoint config).
    pub seed: u64,
    /// EOS id; resets n-gram segments.
    pub eos_token_id: i64,
    /// `ple_embed_dim` (output width of the embedding).
    pub embed_dim: usize,
    /// Position of this layer in `ple_layer_ids`.
    pub ple_layer_index: usize,
}

/// PLE parameters (borrowed row-major slices in torch layouts).
pub struct PleParams<'a> {
    /// N-gram embedding config; the conv dilation equals `ngram.ngram_size`.
    pub ngram: &'a NgramConfig,
    /// Model hidden size `H`.
    pub hidden_size: usize,
    /// Hyper-connection stream count `C`.
    pub hc_count: usize,
    /// `ple_conv_kernel_size`.
    pub kernel: usize,
    /// RMSNorm epsilon.
    pub eps: f32,
    /// N-gram embedding table `[rows, embed_dim / ngram_heads]`.
    pub table: &'a [f32],
    /// `key_proj.weight` `[C*H, embed_dim]`.
    pub key_proj: &'a [f32],
    /// `value_proj.weight` `[H, embed_dim]`.
    pub value_proj: &'a [f32],
    /// `norm_key.weight` `[C*H]` (offset form, scale is `1 + w`).
    pub norm_key: &'a [f32],
    /// `norm_query.weight` `[C*H]`.
    pub norm_query: &'a [f32],
    /// `norm_conv.weight` `[C*H]`.
    pub norm_conv: &'a [f32],
    /// `conv1d.weight` `[C*H, 1, kernel]`.
    pub conv_w: &'a [f32],
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// Gated RMSNorm (`Qwen4ExpTextRMSNormGated`): `w * rmsnorm(x) * act(gate)`,
/// per row of `weight.len()`. The weight is plain `w`, not `1 + w`.
pub fn gated_rms_norm(x: &[f32], gate: &[f32], weight: &[f32], eps: f32, act: GateAct) -> Vec<f32> {
    let d = weight.len().max(1);
    let mut out = Vec::with_capacity(x.len());
    for (xr, zr) in x.chunks(d).zip(gate.chunks(d)) {
        let ms = xr.iter().map(|v| v * v).sum::<f32>() / xr.len() as f32;
        let inv = (ms + eps).sqrt().recip();
        out.extend(xr.iter().zip(zr).zip(weight).map(|((&v, &z), &w)| {
            let g = match act {
                GateAct::Sigmoid => sigmoid(z),
                GateAct::Silu => silu(z),
            };
            w * (v * inv) * g
        }));
    }
    out
}

fn splitmix64(value: u64) -> u64 {
    let v = value.wrapping_add(SPLITMIX_GAMMA);
    let v = (v ^ (v >> 30)).wrapping_mul(SPLITMIX_M1);
    let v = (v ^ (v >> 27)).wrapping_mul(SPLITMIX_M2);
    v ^ (v >> 31)
}

/// Per-layer hash multipliers (`_build_layer_multipliers`).
pub fn layer_multipliers(cfg: &NgramConfig) -> Vec<i64> {
    let half = ((i64::MAX as u64) / cfg.vocab_size.max(1) / 2).max(1);
    let base = cfg
        .seed
        .wrapping_add(PRIME_1.wrapping_mul(cfg.ple_layer_index as u64));
    (0..cfg.ngram_size as u64)
        .map(|i| {
            let s = base.wrapping_add(SPLITMIX_GAMMA.wrapping_mul(i + 1));
            (2 * (splitmix64(s) % half) + 1) as i64
        })
        .collect()
}

fn is_prime(v: u64) -> bool {
    if v < 2 || (v % 2 == 0 && v != 2) {
        return v == 2;
    }
    (3..)
        .step_by(2)
        .take_while(|d| d * d <= v)
        .all(|d| v % d != 0)
}

fn nth_prime_after(start: u64, count: usize) -> u64 {
    let mut p = start;
    for _ in 0..count {
        p += 1;
        while !is_prime(p) {
            p += 1;
        }
    }
    p
}

/// Per-head `(vocab_size, offset)` pairs for the n-gram table.
pub fn head_tables(cfg: &NgramConfig) -> Vec<(u64, u64)> {
    let heads = (cfg.ngram_size - 1) * cfg.heads_per_ngram;
    let mut off = 0;
    (0..heads)
        .map(|h| {
            let g = cfg.ple_layer_index * heads + h + 1;
            let size = nth_prime_after(cfg.ngram_vocab_size_base - 1, g);
            let pair = (size, off);
            off += size;
            pair
        })
        .collect()
}

/// `_shift_right_ignore_eos` for one sequence: positions that cross an EOS
/// boundary or fall before the start become EOS.
fn shift_ignore_eos(hist: &[i64], shift: usize, eos: i64) -> Vec<i64> {
    let mut seg_start = 0usize;
    let mut out = Vec::with_capacity(hist.len());
    for (p, &tok) in hist.iter().enumerate() {
        let valid = p >= shift && p - seg_start >= shift;
        out.push(if shift == 0 {
            tok
        } else if valid {
            hist[p - shift]
        } else {
            eos
        });
        if tok == eos {
            seg_start = p + 1;
        }
    }
    out
}

/// Hashed n-gram table ids for one sequence: `[len(ids), ngram_heads]`
/// row-major. `ctx` is the previous `ngram_size - 1` tokens (default EOS).
pub fn ngram_ids(cfg: &NgramConfig, ids: &[i64], ctx: Option<&[i64]>) -> Vec<i64> {
    let ctx_len = cfg.ngram_size - 1;
    let mut hist: Vec<i64> = match ctx {
        Some(c) => c.to_vec(),
        None => vec![cfg.eos_token_id; ctx_len],
    };
    hist.extend_from_slice(ids);
    let shifted: Vec<Vec<i64>> = (0..cfg.ngram_size)
        .map(|k| shift_ignore_eos(&hist, k, cfg.eos_token_id))
        .collect();
    let mult = layer_multipliers(cfg);
    let heads = head_tables(cfg);
    let first = hist.len() - ids.len();
    let mut out = Vec::with_capacity(ids.len() * heads.len());
    for p in first..hist.len() {
        for n in 2..=cfg.ngram_size {
            let mixed = (0..n).fold(0i64, |acc, j| {
                let t = shifted[j][p].wrapping_mul(mult[j]);
                if j == 0 {
                    t
                } else {
                    acc ^ t
                }
            });
            let lo = (n - 2) * cfg.heads_per_ngram;
            for &(size, off) in &heads[lo..lo + cfg.heads_per_ngram] {
                out.push(mixed.rem_euclid(size as i64) + off as i64);
            }
        }
    }
    out
}

/// Table lookup of hashed ids `[tokens, ngram_heads]` -> `[tokens, embed_dim]`.
fn lookup(cfg: &NgramConfig, table: &[f32], hashed: &[i64]) -> Vec<f32> {
    let heads = (cfg.ngram_size - 1) * cfg.heads_per_ngram;
    let w = cfg.embed_dim / heads.max(1);
    hashed
        .iter()
        .flat_map(|&row| {
            let s = row as usize * w;
            table.get(s..s + w).unwrap_or(&[]).iter().copied()
        })
        .collect()
}

/// N-gram hashed embedding for one sequence: `[len(ids), embed_dim]`.
pub fn ngram_embed(cfg: &NgramConfig, table: &[f32], ids: &[i64]) -> Vec<f32> {
    lookup(cfg, table, &ngram_ids(cfg, ids, None))
}

/// Decode-time n-gram context carry-over (#7332): the last `ngram_size - 1`
/// token ids of the sequence so far (EOS before the first token).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NgramState {
    ctx: Vec<i64>,
}

impl NgramState {
    /// Fresh state for a new sequence.
    pub fn new(cfg: &NgramConfig) -> Self {
        Self {
            ctx: vec![cfg.eos_token_id; cfg.ngram_size.saturating_sub(1)],
        }
    }

    /// The carried context.
    pub fn context(&self) -> &[i64] {
        &self.ctx
    }

    /// Hashed ids for the next chunk `ids`, then carries its tail forward.
    pub fn ids(&mut self, cfg: &NgramConfig, ids: &[i64]) -> Vec<i64> {
        let out = ngram_ids(cfg, ids, Some(&self.ctx));
        let keep = self.ctx.len();
        let mut hist = std::mem::take(&mut self.ctx);
        hist.extend_from_slice(ids);
        self.ctx = hist.split_off(hist.len() - keep);
        out
    }

    /// Embedding of the next chunk `ids` (`[len(ids), embed_dim]`).
    pub fn embed(&mut self, cfg: &NgramConfig, table: &[f32], ids: &[i64]) -> Vec<f32> {
        let hashed = self.ids(cfg, ids);
        lookup(cfg, table, &hashed)
    }
}

/// `y = x @ w^T` with `w` as `[out, inp]`, `x` as `[rows, inp]`.
fn linear(x: &[f32], w: &[f32], inp: usize, out: usize) -> Vec<f32> {
    x.chunks(inp)
        .flat_map(|row| {
            w.chunks(inp)
                .take(out)
                .map(|wr| row.iter().zip(wr).map(|(a, b)| a * b).sum::<f32>())
        })
        .collect()
}

/// Grouped RMSNorm with `(1 + w)` scale; `x` rows are `weight.len()` wide and
/// are split into groups of `group` normalised independently.
fn grouped_rms_norm(x: &[f32], weight: &[f32], group: usize, eps: f32) -> Vec<f32> {
    let d = weight.len();
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks(d) {
        for (gi, g) in row.chunks(group).enumerate() {
            let ms = g.iter().map(|v| v * v).sum::<f32>() / g.len() as f32;
            let inv = (ms + eps).sqrt().recip();
            let base = gi * group;
            out.extend(
                g.iter()
                    .enumerate()
                    .map(|(i, v)| v * inv * (1.0 + weight[base + i])),
            );
        }
    }
    out
}

/// Depthwise dilated causal conv + SiLU over `x: [seq, ch]`; `w: [ch, kernel]`.
fn dilated_causal_conv_silu(
    x: &[f32],
    w: &[f32],
    ch: usize,
    kernel: usize,
    dil: usize,
) -> Vec<f32> {
    let seq = x.len() / ch.max(1);
    let mut out = vec![0.0f32; seq * ch];
    for s in 0..seq {
        for c in 0..ch {
            let acc: f32 = (0..kernel)
                .filter_map(|j| {
                    let back = (kernel - 1 - j) * dil;
                    s.checked_sub(back)
                        .map(|t| w[c * kernel + j] * x[t * ch + c])
                })
                .sum();
            out[s * ch + c] = silu(acc);
        }
    }
    out
}

/// Per-stream signed-sqrt sigmoid gate, applied to `value` (`[seq, H]`),
/// returning `gated_value` `[seq, C*H]`.
fn gate_value(key: &[f32], query: &[f32], value: &[f32], h: usize, c: usize) -> Vec<f32> {
    let scale = (h as f32).sqrt();
    let mut out = Vec::with_capacity(key.len());
    for ((kr, qr), vr) in key
        .chunks(c * h)
        .zip(query.chunks(c * h))
        .zip(value.chunks(h))
    {
        for (kc, qc) in kr.chunks(h).zip(qr.chunks(h)) {
            let raw = kc.iter().zip(qc).map(|(a, b)| a * b).sum::<f32>() / scale;
            let g = raw.abs().max(1e-6).sqrt()
                * if raw < 0.0 {
                    -1.0
                } else if raw > 0.0 {
                    1.0
                } else {
                    0.0
                };
            out.extend(vr.iter().map(|v| sigmoid(g) * v));
        }
    }
    out
}

/// PLE for one sequence: `hidden: [seq, C*H]`, `ids: [seq]` -> `[seq, C*H]`.
pub fn ple_forward(p: &PleParams<'_>, hidden: &[f32], ids: &[i64]) -> Vec<f32> {
    PleState::new(p).step(p, hidden, ids)
}

/// Decode-time PLE state for one sequence (#7332): the n-gram context and the
/// last `(kernel - 1) * dilation` rows of the conv input (normed gated value,
/// `[rows, C*H]`, zeros before the first token).
#[derive(Debug, Clone)]
pub struct PleState {
    ngram: NgramState,
    conv: Vec<f32>,
}

impl PleState {
    /// Fresh state for a new sequence.
    pub fn new(p: &PleParams<'_>) -> Self {
        let rows = p.kernel.saturating_sub(1) * p.ngram.ngram_size;
        Self {
            ngram: NgramState::new(p.ngram),
            conv: vec![0.0; rows * p.hc_count * p.hidden_size],
        }
    }

    /// Carried n-gram context.
    pub fn ngram(&self) -> &NgramState {
        &self.ngram
    }

    /// PLE of the next chunk: `hidden: [n, C*H]`, `ids: [n]` -> `[n, C*H]`.
    pub fn step(&mut self, p: &PleParams<'_>, hidden: &[f32], ids: &[i64]) -> Vec<f32> {
        let (h, c, e) = (p.hidden_size, p.hc_count, p.ngram.embed_dim);
        let emb = self.ngram.embed(p.ngram, p.table, ids);
        let key = grouped_rms_norm(&linear(&emb, p.key_proj, e, c * h), p.norm_key, h, p.eps);
        let value = linear(&emb, p.value_proj, e, h);
        let query = grouped_rms_norm(hidden, p.norm_query, h, p.eps);
        let gv = gate_value(&key, &query, &value, h, c);
        let normed = grouped_rms_norm(&gv, p.norm_conv, h, p.eps);
        let skip = self.conv.len();
        let mut ext = std::mem::take(&mut self.conv);
        ext.extend_from_slice(&normed);
        let conv = dilated_causal_conv_silu(&ext, p.conv_w, c * h, p.kernel, p.ngram.ngram_size);
        self.conv = ext[ext.len() - skip..].to_vec();
        gv.iter().zip(&conv[skip..]).map(|(a, b)| a + b).collect()
    }
}
