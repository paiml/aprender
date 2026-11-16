//! `qwen4_exp` attention-side ops (#7175, master #7169): the sparse-attention
//! lightning indexer, the indexed attention module with its sigmoid output
//! gate, and the hc multi-stream residual.
//!
//! Reference: transformers @ 36e937ae761dea806a5b9fbb382e880378ad5435,
//! `modeling_qwen4_exp.py` (`Qwen4ExpTextQSAIndexer`, `Qwen4ExpTextAttention`,
//! `Qwen4ExpTextGatedResidual`). Pure fp32 CPU functions on row-major slices
//! with explicit shapes; one sequence per call (batches are looped by the caller).
//! Not covered: KV/indexer cache decode (queries == keys), bf16 islands.

use crate::gguf::forward_qwen35::apply_sigmoid_gate;
use crate::gguf::ops::{rms_norm_unit_offset, softmax};

/// Result of an op: shape violations are reported, never panicked on.
pub type OpResult<T> = Result<T, String>;

fn ensure(name: &str, got: usize, want: usize) -> OpResult<()> {
    if got == want {
        Ok(())
    } else {
        Err(format!("{name}: length {got}, expected {want}"))
    }
}

/// `y[r] = sum_c w[r*cols + c] * x[c]` (torch Linear, weight `[out, in]`, no bias).
fn linear(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    w.chunks_exact(cols)
        .take(rows)
        .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}

fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

fn silu(v: f32) -> f32 {
    v * sigmoid(v)
}

/// Grouped `(1 + w)` RMSNorm: `x` is split in groups of `weight.len() / n_groups`
/// ... here simply: each `group`-sized chunk of `x` is normalised with the
/// matching slice of `weight` (`weight.len() == x.len()`).
fn grouped_rms_norm(x: &[f32], weight: &[f32], group: usize, eps: f32) -> Vec<f32> {
    x.chunks_exact(group)
        .zip(weight.chunks_exact(group))
        .flat_map(|(xg, wg)| rms_norm_unit_offset(xg, wg, eps))
        .collect()
}

/// Partial rotate-half RoPE on every head of one token with given cos/sin
/// (`rotary_dim = cos.len()`; dims past it pass through).
fn rope_token(x: &mut [f32], head_dim: usize, cos: &[f32], sin: &[f32]) {
    let half = cos.len() / 2;
    for head in x.chunks_exact_mut(head_dim) {
        for i in 0..half {
            let (a, b) = (head[i], head[i + half]);
            head[i] = a * cos[i] - b * sin[i];
            head[i + half] = b * cos[i + half] + a * sin[i + half];
        }
    }
}

/// Indexer weights (`Qwen4ExpTextQSAIndexer`).
pub struct IndexerParams<'a> {
    /// `index_qk_proj.weight` `[(n_heads + 1) * head_dim, hidden]`.
    pub qk_proj: &'a [f32],
    /// `q_layernorm.weight` `[head_dim]` (offset from 1).
    pub q_norm: &'a [f32],
    /// `k_layernorm.weight` `[head_dim]` (offset from 1).
    pub k_norm: &'a [f32],
}

/// Indexer hyperparameters.
#[derive(Clone, Copy)]
pub struct IndexerShape {
    /// Model hidden size.
    pub hidden: usize,
    /// Indexer query heads (`indexer_n_heads`); there is exactly one key head.
    pub n_heads: usize,
    /// `indexer_head_dim`.
    pub head_dim: usize,
    /// `indexer_budget` (tokens).
    pub budget: usize,
    /// `indexer_compress_ratio` (tokens per block).
    pub ratio: usize,
    /// RMSNorm epsilon.
    pub eps: f32,
}

/// Rotated, normed indexer queries `[seq, n_heads * head_dim]` and raw keys `[seq, head_dim]`.
fn indexer_qk(
    p: &IndexerParams,
    s: &IndexerShape,
    hidden: &[f32],
    cos: &[f32],
    sin: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let (dh, rot) = (s.head_dim, cos.len() / (hidden.len() / s.hidden));
    let qd = s.n_heads * dh;
    let (mut qs, mut ks) = (Vec::new(), Vec::new());
    for (t, h) in hidden.chunks_exact(s.hidden).enumerate() {
        let qk = linear(p.qk_proj, qd + dh, s.hidden, h);
        let mut q: Vec<f32> = qk[..qd]
            .chunks_exact(dh)
            .flat_map(|hd| rms_norm_unit_offset(hd, p.q_norm, s.eps))
            .collect();
        rope_token(
            &mut q,
            dh,
            &cos[t * rot..(t + 1) * rot],
            &sin[t * rot..(t + 1) * rot],
        );
        qs.extend(q);
        ks.extend_from_slice(&qk[qd..]);
    }
    (qs, ks)
}

/// Block scores of one query against the pooled blocks of its visible tokens.
fn block_scores(
    p: &IndexerParams,
    s: &IndexerShape,
    q: &[f32],
    keys: &[f32],
    blocks: &[usize],
    cs: (&[f32], &[f32]),
) -> Vec<f32> {
    let (dh, rot) = (s.head_dim, cs.0.len() / (keys.len() / s.head_dim));
    blocks
        .chunks_exact(s.ratio)
        .map(|blk| {
            let mut pooled = vec![0f32; dh];
            for &tok in blk {
                for (a, b) in pooled.iter_mut().zip(&keys[tok * dh..(tok + 1) * dh]) {
                    *a += *b;
                }
            }
            for v in pooled.iter_mut() {
                *v /= s.ratio as f32;
            }
            let mut pooled = rms_norm_unit_offset(&pooled, p.k_norm, s.eps);
            let g = blk[0];
            rope_token(
                &mut pooled,
                dh,
                &cs.0[g * rot..(g + 1) * rot],
                &cs.1[g * rot..(g + 1) * rot],
            );
            let dot: f32 = q
                .chunks_exact(dh)
                .map(|qh| {
                    qh.iter()
                        .zip(&pooled)
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                        .max(0.0)
                })
                .sum();
            dot / (dh as f32).sqrt()
        })
        .collect()
}

/// Indices of the `k` largest scores. Ties resolve to the lower block index
/// (stable); torch.topk documents no tie order, so a tie is ambiguous vs the reference.
fn top_k(scores: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    idx.truncate(k);
    idx
}

/// Selected-token row for one query.
fn select_row(
    p: &IndexerParams,
    s: &IndexerShape,
    q: &[f32],
    keys: &[f32],
    visible: &[bool],
    cs: (&[f32], &[f32]),
) -> Vec<bool> {
    let vis: Vec<usize> = (0..visible.len()).filter(|&i| visible[i]).collect();
    let nb = vis.len() / s.ratio;
    let (full, tail) = vis.split_at(nb * s.ratio);
    let scores = block_scores(p, s, q, keys, full, cs);
    let mut row = vec![false; visible.len()];
    for b in top_k(&scores, (s.budget / s.ratio).min(nb)) {
        full[b * s.ratio..(b + 1) * s.ratio]
            .iter()
            .for_each(|&t| row[t] = true);
    }
    for &t in tail {
        row[t] = true;
    }
    row
}

/// Lightning indexer for one sequence of `seq` tokens (queries == keys).
///
/// `hidden` `[seq, hidden]`, `cos`/`sin` `[seq, rotary_dim]`, `visible` `[seq, seq]`
/// (row = query). Returns the selected-token mask `[seq, seq]`.
pub fn indexer_select(
    p: &IndexerParams,
    s: &IndexerShape,
    hidden: &[f32],
    cos: &[f32],
    sin: &[f32],
    visible: &[bool],
) -> OpResult<Vec<bool>> {
    ensure("hidden", hidden.len() % s.hidden, 0)?;
    let seq = hidden.len() / s.hidden;
    ensure("visible", visible.len(), seq * seq)?;
    ensure("cos", cos.len() % seq, 0)?;
    ensure("sin", sin.len(), cos.len())?;
    ensure(
        "qk_proj",
        p.qk_proj.len(),
        (s.n_heads + 1) * s.head_dim * s.hidden,
    )?;
    let (q, k) = indexer_qk(p, s, hidden, cos, sin);
    let qd = s.n_heads * s.head_dim;
    let mut mask = Vec::with_capacity(seq * seq);
    for t in 0..seq {
        mask.extend(select_row(
            p,
            s,
            &q[t * qd..(t + 1) * qd],
            &k,
            &visible[t * seq..(t + 1) * seq],
            (cos, sin),
        ));
    }
    Ok(mask)
}

/// Weights of `Qwen4ExpTextAttention`.
pub struct AttentionParams<'a> {
    /// `q_proj.weight` `[n_heads * head_dim * 2, hidden]` (per head: query then gate).
    pub q_proj: &'a [f32],
    /// `k_proj.weight` `[n_kv * head_dim, hidden]`.
    pub k_proj: &'a [f32],
    /// `v_proj.weight` `[n_kv * head_dim, hidden]`.
    pub v_proj: &'a [f32],
    /// `o_proj.weight` `[hidden, n_heads * head_dim]`.
    pub o_proj: &'a [f32],
    /// `q_norm.weight` `[head_dim]`.
    pub q_norm: &'a [f32],
    /// `k_norm.weight` `[head_dim]`.
    pub k_norm: &'a [f32],
    /// The layer's indexer.
    pub indexer: IndexerParams<'a>,
}

/// Attention hyperparameters.
#[derive(Clone, Copy)]
pub struct AttentionShape {
    /// Attention query heads.
    pub n_heads: usize,
    /// KV heads (GQA).
    pub n_kv: usize,
    /// Per-head dim.
    pub head_dim: usize,
    /// Indexer settings (also carries `hidden` and `eps`).
    pub indexer: IndexerShape,
}

/// Per-token projections: queries, sigmoid-gate pre-activations, keys, values.
struct Projected {
    q: Vec<f32>,
    gate: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
}

fn project_qkv(
    p: &AttentionParams,
    a: &AttentionShape,
    hidden: &[f32],
    cos: &[f32],
    sin: &[f32],
) -> Projected {
    let (d, h) = (a.head_dim, a.indexer.hidden);
    let seq = hidden.len() / h;
    let rot = cos.len() / seq;
    let mut out = Projected {
        q: vec![],
        gate: vec![],
        k: vec![],
        v: vec![],
    };
    for (t, x) in hidden.chunks_exact(h).enumerate() {
        let qg = linear(p.q_proj, a.n_heads * d * 2, h, x);
        let mut q = Vec::new();
        for head in qg.chunks_exact(2 * d) {
            q.extend(rms_norm_unit_offset(&head[..d], p.q_norm, a.indexer.eps));
            out.gate.extend_from_slice(&head[d..]);
        }
        let kp = linear(p.k_proj, a.n_kv * d, h, x);
        let mut k: Vec<f32> = kp
            .chunks_exact(d)
            .flat_map(|c| rms_norm_unit_offset(c, p.k_norm, a.indexer.eps))
            .collect();
        let (c, s) = (&cos[t * rot..(t + 1) * rot], &sin[t * rot..(t + 1) * rot]);
        rope_token(&mut q, d, c, s);
        rope_token(&mut k, d, c, s);
        out.q.extend(q);
        out.k.extend(k);
        out.v.extend(linear(p.v_proj, a.n_kv * d, h, x));
    }
    out
}

/// Softmax attention (GQA) of one query head over the allowed keys.
fn attend_head(
    a: &AttentionShape,
    pr: &Projected,
    t: usize,
    head: usize,
    allowed: &[bool],
) -> Vec<f32> {
    let (d, nq, nk) = (a.head_dim, a.n_heads * a.head_dim, a.n_kv * a.head_dim);
    let kv = head / (a.n_heads / a.n_kv);
    let q = &pr.q[t * nq + head * d..t * nq + (head + 1) * d];
    let keys: Vec<usize> = (0..allowed.len()).filter(|&i| allowed[i]).collect();
    let scale = (d as f32).powf(-0.5);
    let mut w: Vec<f32> = keys
        .iter()
        .map(|&j| {
            q.iter()
                .zip(&pr.k[j * nk + kv * d..j * nk + (kv + 1) * d])
                .map(|(x, y)| x * y)
                .sum::<f32>()
                * scale
        })
        .collect();
    softmax(&mut w);
    let mut out = vec![0f32; d];
    for (&j, wj) in keys.iter().zip(&w) {
        for (o, v) in out
            .iter_mut()
            .zip(&pr.v[j * nk + kv * d..j * nk + (kv + 1) * d])
        {
            *o += wj * v;
        }
    }
    out
}

/// `Qwen4ExpTextAttention.forward` for one sequence: indexer mask AND `visible`,
/// gated q_proj, per-head q/k norm, partial RoPE, GQA softmax, `* sigmoid(gate)`, `o_proj`.
///
/// `hidden` `[seq, hidden]`, `cos`/`sin` `[seq, rotary_dim]`, `visible` `[seq, seq]`.
/// Returns `[seq, hidden]`.
pub fn indexed_attention(
    p: &AttentionParams,
    a: &AttentionShape,
    hidden: &[f32],
    cos: &[f32],
    sin: &[f32],
    visible: &[bool],
) -> OpResult<Vec<f32>> {
    let h = a.indexer.hidden;
    ensure("hidden", hidden.len() % h, 0)?;
    let seq = hidden.len() / h;
    ensure("q_proj", p.q_proj.len(), a.n_heads * a.head_dim * 2 * h)?;
    ensure("o_proj", p.o_proj.len(), h * a.n_heads * a.head_dim)?;
    ensure("gqa", a.n_heads % a.n_kv, 0)?;
    let sel = indexer_select(&p.indexer, &a.indexer, hidden, cos, sin, visible)?;
    let pr = project_qkv(p, a, hidden, cos, sin);
    let nq = a.n_heads * a.head_dim;
    let mut out = Vec::with_capacity(seq * h);
    for t in 0..seq {
        let allowed: Vec<bool> = (0..seq)
            .map(|j| sel[t * seq + j] && visible[t * seq + j])
            .collect();
        let mut o: Vec<f32> = (0..a.n_heads)
            .flat_map(|hd| attend_head(a, &pr, t, hd, &allowed))
            .collect();
        apply_sigmoid_gate(&mut o, &pr.gate[t * nq..(t + 1) * nq]);
        out.extend(linear(p.o_proj, h, nq, &o));
    }
    Ok(out)
}

/// `Qwen4ExpTextGatedResidual` weights.
pub struct HcParams<'a> {
    /// `hc_norm.weight` `[count * hidden]`.
    pub norm: &'a [f32],
    /// `input_mix_weight_down.weight` `[lowrank, count * hidden]`.
    pub down: &'a [f32],
    /// `input_mix_weight_up.weight` `[count * hidden, lowrank]`.
    pub up: &'a [f32],
    /// `block_inject_weight.weight` `[count, count * hidden]`; `None` for the final mixer.
    pub inject: Option<&'a [f32]>,
}

/// hc hyperparameters.
#[derive(Clone, Copy)]
pub struct HcShape {
    /// Model hidden size.
    pub hidden: usize,
    /// `hc_count` (streams).
    pub count: usize,
    /// `hc_lowrank`.
    pub lowrank: usize,
    /// RMSNorm epsilon.
    pub eps: f32,
}

/// Output of the gated residual.
pub struct HcOut {
    /// `[tokens, hidden]`: stream mix that feeds the sublayer.
    pub mixed: Vec<f32>,
    /// `[tokens, count]` injection weights (empty for the final mixer).
    pub inject: Vec<f32>,
}

/// Gated residual over `tokens` rows of `[count * hidden]` streams. The raw
/// streams (the reference's second return value) are `x` itself.
pub fn hc_gated_residual(p: &HcParams, s: &HcShape, x: &[f32]) -> OpResult<HcOut> {
    let ch = s.count * s.hidden;
    ensure("x", x.len() % ch, 0)?;
    ensure("norm", p.norm.len(), ch)?;
    ensure("down", p.down.len(), s.lowrank * ch)?;
    ensure("up", p.up.len(), ch * s.lowrank)?;
    let mut out = HcOut {
        mixed: vec![],
        inject: vec![],
    };
    for row in x.chunks_exact(ch) {
        let xn = grouped_rms_norm(row, p.norm, s.hidden, s.eps);
        let low: Vec<f32> = linear(p.down, s.lowrank, ch, &xn)
            .iter()
            .map(|v| silu(v / s.count as f32))
            .collect();
        let mix: Vec<f32> = linear(p.up, ch, s.lowrank, &low)
            .into_iter()
            .map(sigmoid)
            .collect();
        let mut m = vec![0f32; s.hidden];
        for (mg, xg) in mix.chunks_exact(s.hidden).zip(xn.chunks_exact(s.hidden)) {
            for (o, (a, b)) in m.iter_mut().zip(mg.iter().zip(xg)) {
                *o += a * b / s.count as f32;
            }
        }
        out.mixed.extend(m);
        if let Some(w) = p.inject {
            ensure("inject", w.len(), s.count * ch)?;
            out.inject.extend(
                linear(w, s.count, ch, &xn)
                    .iter()
                    .map(|v| 2.0 * sigmoid(v / s.count as f32)),
            );
        }
    }
    Ok(out)
}

/// Decoder-layer residual update: `x_raw + flatten(block_out[..., None, :] * inject[..., :, None])`.
/// `x` `[tokens, count*hidden]`, `block_out` `[tokens, hidden]`, `inject` `[tokens, count]`.
pub fn hc_combine(s: &HcShape, x: &[f32], block_out: &[f32], inject: &[f32]) -> OpResult<Vec<f32>> {
    let tokens = block_out.len() / s.hidden;
    ensure("x", x.len(), tokens * s.count * s.hidden)?;
    ensure("inject", inject.len(), tokens * s.count)?;
    let mut out = x.to_vec();
    for t in 0..tokens {
        for c in 0..s.count {
            for j in 0..s.hidden {
                out[(t * s.count + c) * s.hidden + j] +=
                    block_out[t * s.hidden + j] * inject[t * s.count + c];
            }
        }
    }
    Ok(out)
}

/// Final mixer (`use_combine=False`): collapses streams to `[tokens, hidden]`.
pub fn hc_final_mixer(p: &HcParams, s: &HcShape, x: &[f32]) -> OpResult<Vec<f32>> {
    hc_gated_residual(p, s, x).map(|o| o.mixed)
}

/// Stream init: `hidden.repeat(1, 1, count)` over `[tokens, hidden]` rows.
pub fn hc_stream_init(s: &HcShape, embeds: &[f32]) -> Vec<f32> {
    embeds
        .chunks_exact(s.hidden)
        .flat_map(|r| r.repeat(s.count))
        .collect()
}

/// Block scores of query `t` (diagnostics for tie checks): one score per complete
/// block of the query's visible tokens, in block order.
pub fn indexer_block_scores_row(
    p: &IndexerParams,
    s: &IndexerShape,
    hidden: &[f32],
    cos: &[f32],
    sin: &[f32],
    visible: &[bool],
    t: usize,
) -> OpResult<Vec<f32>> {
    ensure("hidden", hidden.len() % s.hidden, 0)?;
    let seq = hidden.len() / s.hidden;
    ensure("visible", visible.len(), seq * seq)?;
    ensure("query", usize::from(t < seq), 1)?;
    let (q, k) = indexer_qk(p, s, hidden, cos, sin);
    let qd = s.n_heads * s.head_dim;
    let vis: Vec<usize> = (0..seq).filter(|&i| visible[t * seq + i]).collect();
    let full = &vis[..vis.len() / s.ratio * s.ratio];
    Ok(block_scores(
        p,
        s,
        &q[t * qd..(t + 1) * qd],
        &k,
        full,
        (cos, sin),
    ))
}
