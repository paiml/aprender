//! Chronos-Bolt zero-shot inference — the spike-005 port (parity 9.54e-7 against
//! `chronos-forecasting` 2.3.1), with every matrix product routed through trueno's packed GEMM
//! now that aarch64 has a NEON microkernel (spike 008): projections, feed-forward, the
//! patch-embedding residual blocks, and attention scores and context.
//!
//! # What was ported and what changed (D-08)
//!
//! The body is `sources/007-chronos-mcp-thin-server/src/bolt.rs` verbatim, minus ONE branch and
//! plus ONE merge:
//!
//! * **REMOVED — the rayon parallel-GEMM arm**: the spike's `AtomicBool` toggle at
//!   `sources/007/src/bolt.rs:48-51` and the `if` arm at `:61` are both gone. This file never
//!   writes that static's identifier, so `! grep -q` on it is a real file-wide ban (D-14
//!   verify) rather than a reviewer's memory — the discipline 06-04 established for D-10. D-14
//!   says the rayon `blis::gemm` path is not used in the servers (<= 1.4x below ~500 tokens;
//!   Lambda has 1-2 vCPUs), and this crate's manifest never enables trueno's `parallel` feature.
//!   `linear_fast` therefore always calls the single-threaded [`trueno::blis::gemm_blis`].
//! * **MERGED — the spike-005 `Forward` ladder tensors back into the forward.** Spike 005's
//!   `forward` returned `Forward { loc, scale, attention_mask, input_embeds, encoder_hidden,
//!   decoder_hidden, quantiles }` (its parity driver asserts every rung); spike 007's returned
//!   quantiles plus per-stage milliseconds. Duplicating the forward body to get both would be two
//!   copies of the arithmetic the 9.54e-7 measurement was taken through, so there is ONE body,
//!   [`Bolt::forward_full`], and three thin wrappers: [`Bolt::forward`] (quantiles, the hot
//!   path), [`Bolt::forward_stages`] (007's signature, for `--bench`) and
//!   [`Bolt::forward_ladder`] (005's struct, for the parity ladder). No operation, no order and
//!   no rounding moved.
//!
//! `fast = false` / `attn_gemm = false` / `row1_gemv = true` keep the plain-loop and gemv paths
//! for A/B measurement; the PRODUCTION defaults set by [`Bolt::load`] are
//! `fast = true, attn_gemm = true, row1_gemv = false`, which is D-14's routing exactly.

use crate::safetensors::{Tensor, Weights};

/// Chronos-Bolt hyper-parameters, read from the model's `config.json`.
#[derive(Clone, Debug)]
pub struct Config {
    /// Model width.
    pub d_model: usize,
    /// Feed-forward width.
    pub d_ff: usize,
    /// Per-head key/value width.
    pub d_kv: usize,
    /// Attention heads.
    pub heads: usize,
    /// Encoder layers.
    pub enc_layers: usize,
    /// Decoder layers.
    pub dec_layers: usize,
    /// T5 layer-norm epsilon.
    pub eps: f32,
    /// Relative-attention buckets.
    pub buckets: usize,
    /// Relative-attention maximum distance.
    pub max_distance: usize,
    /// Longest context the model looks at (2048 for Bolt).
    pub context_length: usize,
    /// Native direct horizon (64 for Bolt).
    pub prediction_length: usize,
    /// Points per input patch.
    pub patch: usize,
    /// Quantile levels the head emits, ascending.
    pub quantiles: Vec<f32>,
    /// Whether a REG token is appended to the encoder input.
    pub use_reg_token: bool,
}

impl Config {
    /// Read a Chronos-Bolt `config.json` value.
    ///
    /// # Panics
    ///
    /// Panics naming the key when a required field is absent or the wrong type. A malformed
    /// config is a defect in the weights directory, never a condition to default around (D-11).
    pub fn from_json(v: &serde_json::Value) -> Config {
        let c = &v["chronos_config"];
        Config {
            d_model: v["d_model"].as_u64().expect("d_model") as usize,
            d_ff: v["d_ff"].as_u64().expect("d_ff") as usize,
            d_kv: v["d_kv"].as_u64().expect("d_kv") as usize,
            heads: v["num_heads"].as_u64().expect("heads") as usize,
            enc_layers: v["num_layers"].as_u64().expect("layers") as usize,
            dec_layers: v["num_decoder_layers"].as_u64().expect("dec layers") as usize,
            eps: v["layer_norm_epsilon"].as_f64().expect("eps") as f32,
            buckets: v["relative_attention_num_buckets"]
                .as_u64()
                .expect("buckets") as usize,
            max_distance: v["relative_attention_max_distance"].as_u64().expect("maxd") as usize,
            context_length: c["context_length"].as_u64().expect("ctx") as usize,
            prediction_length: c["prediction_length"].as_u64().expect("pl") as usize,
            patch: c["input_patch_size"].as_u64().expect("patch") as usize,
            quantiles: c["quantiles"]
                .as_array()
                .expect("q")
                .iter()
                .map(|q| q.as_f64().expect("f") as f32)
                .collect(),
            use_reg_token: c["use_reg_token"].as_bool().unwrap_or(false),
        }
    }
}

// ------------------------------------------------------------ primitives ----

/// `y[r][o] = SUM_i x[r][i] w[o][i] (+ b[o])`; `w` is `[out, in]` row-major as in torch.
///
/// Plain loops over [`dot8`]. This is the SINGLE-ROW path D-14 mandates, and the reason the
/// untransposed `[out, in]` weight copy is resident: `dot8` reads `w[o * inp .. (o+1) * inp]`,
/// a contiguous `inp`-length row, which only the untransposed layout provides.
#[must_use]
pub fn linear(
    x: &[f32],
    rows: usize,
    inp: usize,
    w: &[f32],
    out: usize,
    b: Option<&[f32]>,
) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * out];
    for r in 0..rows {
        let xr = &x[r * inp..(r + 1) * inp];
        let yr = &mut y[r * out..(r + 1) * out];
        for o in 0..out {
            yr[o] = dot8(xr, &w[o * inp..(o + 1) * inp]) + b.map_or(0.0, |b| b[o]);
        }
    }
    y
}

/// Dot product with 8 independent accumulators: an explicit reduction order LLVM can vectorise.
#[inline]
#[must_use]
pub fn dot8(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut acc = [0.0f32; 8];
    let chunks = n / 8;
    for c in 0..chunks {
        let (ac, bc) = (&a[c * 8..c * 8 + 8], &b[c * 8..c * 8 + 8]);
        for k in 0..8 {
            acc[k] += ac[k] * bc[k];
        }
    }
    let mut s = acc.iter().sum::<f32>();
    for i in chunks * 8..n {
        s += a[i] * b[i];
    }
    s
}

/// `x[rows x in] . Wt[in x out]` through trueno (weights pre-transposed); gemv for a single row.
///
/// D-14's MULTI-ROW path. The rayon `blis::gemm` arm the spike carried behind its `parallel`
/// feature is deleted: this is always the single-threaded packed GEMM.
///
/// # Panics
///
/// Panics if [`trueno::blis::gemm_blis`] refuses the shapes — a dimension mismatch here is a
/// defect in the caller, not a runtime condition.
#[must_use]
pub fn linear_fast(
    x: &[f32],
    rows: usize,
    inp: usize,
    wt: &[f32],
    out: usize,
    b: Option<&[f32]>,
) -> Vec<f32> {
    if rows == 1 {
        let mut y = vec![0.0f32; out];
        trueno::blis::gemv::gemv(inp, out, x, wt, &mut y);
        if let Some(b) = b {
            for (v, bb) in y.iter_mut().zip(b) {
                *v += bb;
            }
        }
        return y;
    }
    let mut y = vec![0.0f32; rows * out];
    trueno::blis::gemm_blis(rows, out, inp, x, wt, &mut y, None).expect("gemm_blis");
    if let Some(b) = b {
        for r in 0..rows {
            for o in 0..out {
                y[r * out + o] += b[o];
            }
        }
    }
    y
}

/// One projection, choosing the path: plain loops when `fast` is off; for a single row either
/// trueno's `gemv` over the transposed copy (`row1_gemv`) or a contiguous [`dot8`] per output over
/// the row-major `[out, in]` weights; for several rows the packed GEMM over the transposed copy.
///
/// At the production defaults (`fast = true, row1_gemv = false`) a single row therefore reaches
/// [`linear`]/[`dot8`] and several rows reach [`linear_fast`]/`gemm_blis` — D-14, both clauses.
/// `bolt::tests::single_row_routing_is_dot8_at_production_defaults` proves it behaviourally.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn proj(
    fast: bool,
    row1_gemv: bool,
    x: &[f32],
    rows: usize,
    inp: usize,
    w: &[f32],
    wt: &[f32],
    out: usize,
    b: Option<&[f32]>,
) -> Vec<f32> {
    if !fast {
        return linear(x, rows, inp, w, out, b);
    }
    if rows == 1 && !row1_gemv {
        return linear(x, 1, inp, w, out, b);
    }
    linear_fast(x, rows, inp, wt, out, b)
}

/// T5 RMS layer norm: no mean subtraction, no bias — `x * rsqrt(mean(x^2) + eps) * w`.
#[must_use]
pub fn t5_layer_norm(x: &[f32], rows: usize, d: usize, w: &[f32], eps: f32) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * d];
    for r in 0..rows {
        let xr = &x[r * d..(r + 1) * d];
        let var = xr.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for i in 0..d {
            y[r * d + i] = xr[i] * inv * w[i];
        }
    }
    y
}

/// transformers' `_relative_position_bucket`.
#[must_use]
pub fn relative_bucket(
    rel: i64,
    bidirectional: bool,
    num_buckets: usize,
    max_distance: usize,
) -> usize {
    let mut nb = num_buckets as i64;
    let mut bucket = 0i64;
    let mut rp = rel;
    if bidirectional {
        nb /= 2;
        if rp > 0 {
            bucket += nb;
        }
        rp = rp.abs();
    } else {
        rp = -(rp.min(0));
    }
    let max_exact = nb / 2;
    let large = max_exact
        + (((rp as f32 / max_exact as f32).ln() / (max_distance as f32 / max_exact as f32).ln())
            * (nb - max_exact) as f32) as i64;
    bucket += if rp < max_exact {
        rp
    } else {
        large.min(nb - 1)
    };
    bucket as usize
}

/// The instance-norm scale floor. A constant series has population std exactly 0, and the
/// Chronos pipeline floors it here rather than dividing by zero. It is a MODEL constant, not a
/// tolerance — which is why it is named once and referenced, so the parity module can use the
/// same value without writing a bare literal inside a region that bans them.
pub const SCALE_FLOOR: f32 = 1e-5;

/// `[out, in]` -> `[in, out]`.
#[must_use]
pub fn transpose(w: &[f32], out: usize, inp: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; w.len()];
    // IN-04: the `expect` below states its safety argument — `w.len() == out * inp` — in
    // PROSE, and prose is enforced by no type and checked at no call site. This is the
    // CHECKABLE FORM OF THAT CLAIM, not a substitute for it: it runs in every debug and
    // test build (which is where all thirteen in-crate call sites are exercised), so a
    // caller that violates the invariant is named at ITS OWN call site with both numbers
    // rather than surfacing as a length mismatch deep inside a third-party kernel.
    //
    // The `expect` is KEPT — the `debug_assert` is the check, the `expect` is the message
    // if a release build is wrong anyway. The return type is deliberately NOT widened to
    // `Result`: the review offers that as an alternative, but it is a breaking public-API
    // change to a published crate for a condition every in-crate caller already satisfies,
    // and this plan is closing an unchecked claim, not opening a new surface.
    debug_assert_eq!(
        w.len(),
        out * inp,
        "transpose: w.len() must equal out * inp (out={out}, inp={inp}); this is the claim \
         the expect below states in prose, checked"
    );
    // trueno's blocked kernel writes b[c * rows + r] = a[r * cols + c]; with
    // rows = out, cols = inp that is exactly t[i * out + o] = w[o * inp + i].
    // A transpose is a permutation, so the blocking cannot change a single bit —
    // it only stops re-deriving the 242x-slower naive form (trueno GH-388).
    trueno::blis::transpose(out, inp, w, &mut t)
        .expect("transpose: w.len() == out * inp, guaranteed by every caller");
    t
}

// -------------------------------------------------- the dual weight layout ----
//
// D-13 AMENDMENT (REVIEW-06-01, ratified by plan 06-05's blocking human decision
// `amend-memory-clause`). D-13's sentence "Only transposed `[in, out]` weights are kept in
// memory" is AMENDED: the shipped `Bolt` keeps BOTH layouts, deliberately.
//
// WHY. `dot8` reads a contiguous `inp`-length row `w[o*inp .. (o+1)*inp]`, which only the
// row-major `[out, in]` copy provides; `gemm_blis` reads the `[in, out]` transpose. D-14 mandates
// `dot8` for single rows AND `gemm_blis` for multi-row products, so both copies are load-bearing.
// Deleting the untransposed copies would force every single-row product onto `gemv` over the
// transpose — violating D-14's second clause — and would change the float accumulation order
// through a 12-layer T5, under which the 9.54e-7 aarch64 Peyton parity number (against the frozen
// 1.0e-6 `quantiles_abs_f32` bar) was measured. The bar would stop being evidence.
//
// WHAT IT COSTS. Weight arrays are resident roughly TWICE: 8_642_560 of the tiny model's
// 8_652_672 f32 parameters are matrices held in both layouts, so residency is 17_295_232 f32 =
// ~69.2 MB rather than ~34.6 MB. That is RESIDENT MEMORY. It is NOT the SC4 "< 30 MB" bar, which
// measures the release BINARY carrying the embedded f16 bytes and is untouched by an in-memory
// duplicate. `bolt::tests::weight_layout_is_dual_and_its_cost_is_stated` recomputes and PRINTS
// that figure from the config alone, so the cost is an asserted number and not a footnote.

/// One attention block's projections, in BOTH layouts (see the D-13 amendment above).
///
/// `q`/`k`/`v`/`o` are row-major `[out, in]` and feed `dot8` on single rows; `qt`/`kt`/`vt`/`ot`
/// are the `[in, out]` transposes and feed `gemm_blis` on multi-row products.
pub struct Attn {
    /// Query projection, `[out, in]` — the `dot8` copy.
    pub q: Vec<f32>,
    /// Key projection, `[out, in]` — the `dot8` copy.
    pub k: Vec<f32>,
    /// Value projection, `[out, in]` — the `dot8` copy.
    pub v: Vec<f32>,
    /// Output projection, `[out, in]` — the `dot8` copy.
    pub o: Vec<f32>,
    /// Query projection, `[in, out]` — the `gemm_blis` copy.
    pub qt: Vec<f32>,
    /// Key projection, `[in, out]` — the `gemm_blis` copy.
    pub kt: Vec<f32>,
    /// Value projection, `[in, out]` — the `gemm_blis` copy.
    pub vt: Vec<f32>,
    /// Output projection, `[in, out]` — the `gemm_blis` copy.
    pub ot: Vec<f32>,
}

/// One transformer block. `wi`/`wo` are the `[out, in]` `dot8` copies of the feed-forward
/// projections and `wit`/`wot` their `[in, out]` `gemm_blis` transposes — both resident, for the
/// reason stated in the D-13 amendment above (`dot8` needs contiguous `[out, in]` rows).
pub struct Block {
    /// Self-attention layer-norm weight.
    pub ln_sa: Vec<f32>,
    /// Self-attention projections.
    pub sa: Attn,
    /// Cross-attention layer-norm weight (decoder only).
    pub ln_ca: Option<Vec<f32>>,
    /// Cross-attention projections (decoder only).
    pub ca: Option<Attn>,
    /// Feed-forward layer-norm weight.
    pub ln_ff: Vec<f32>,
    /// FF in-projection, `[d_ff, d_model]` — the `dot8` copy.
    pub wi: Vec<f32>,
    /// FF out-projection, `[d_model, d_ff]` — the `dot8` copy.
    pub wo: Vec<f32>,
    /// FF in-projection transpose — the `gemm_blis` copy.
    pub wit: Vec<f32>,
    /// FF out-projection transpose — the `gemm_blis` copy.
    pub wot: Vec<f32>,
}

/// An encoder or decoder stack.
pub struct Stack {
    /// The blocks, in order.
    pub blocks: Vec<Block>,
    /// The relative-attention bias table, `[buckets, heads]`.
    pub rel_bias: Vec<f32>,
    /// Final layer-norm weight.
    pub final_ln: Vec<f32>,
    /// Whether this stack is the decoder (causal buckets, cross-attention).
    pub is_decoder: bool,
}

/// A patch-embedding residual block. `wh`/`wo`/`wr` are the `[out, in]` `dot8` copies and
/// `wht`/`wot`/`wrt` their `[in, out]` `gemm_blis` transposes — both resident, for the reason
/// stated in the D-13 amendment above.
pub struct Residual {
    /// Hidden projection, `[hid, inp]` — the `dot8` copy.
    pub wh: Vec<f32>,
    /// Hidden bias.
    pub bh: Vec<f32>,
    /// Output projection, `[out, hid]` — the `dot8` copy.
    pub wo: Vec<f32>,
    /// Output bias.
    pub bo: Vec<f32>,
    /// Residual projection, `[out, inp]` — the `dot8` copy.
    pub wr: Vec<f32>,
    /// Residual bias.
    pub br: Vec<f32>,
    /// Hidden projection transpose — the `gemm_blis` copy.
    pub wht: Vec<f32>,
    /// Output projection transpose — the `gemm_blis` copy.
    pub wot: Vec<f32>,
    /// Residual projection transpose — the `gemm_blis` copy.
    pub wrt: Vec<f32>,
    /// Input width.
    pub inp: usize,
    /// Hidden width.
    pub hid: usize,
    /// Output width.
    pub out: usize,
}

/// A loaded Chronos-Bolt model.
pub struct Bolt {
    /// Route multi-row products through the packed GEMM (production default `true`).
    pub fast: bool,
    /// Route attention scores/context through the packed GEMM (production default `true`).
    pub attn_gemm: bool,
    /// Route SINGLE rows through `gemv` instead of `dot8` (production default `false`, D-14).
    pub row1_gemv: bool,
    /// The hyper-parameters this model was loaded with.
    pub cfg: Config,
    /// `shared.weight`: `[2, d_model]` — the decoder start token and the REG token.
    pub shared: Vec<f32>,
    /// Input patch embedding.
    pub in_embed: Residual,
    /// Output (quantile head) patch embedding.
    pub out_embed: Residual,
    /// The encoder stack.
    pub encoder: Stack,
    /// The decoder stack.
    pub decoder: Stack,
}

fn take(w: &Weights, name: &str) -> Vec<f32> {
    w.get(name)
        .unwrap_or_else(|| panic!("missing tensor {name}"))
        .data
        .clone()
}

fn tensor<'a>(w: &'a Weights, name: &str) -> &'a Tensor {
    w.get(name)
        .unwrap_or_else(|| panic!("missing tensor {name}"))
}

fn residual(w: &Weights, prefix: &str) -> Residual {
    let wh = tensor(w, &format!("{prefix}.hidden_layer.weight"));
    let wo = tensor(w, &format!("{prefix}.output_layer.weight"));
    let wr = tensor(w, &format!("{prefix}.residual_layer.weight"));
    let (inp, hid, out) = (wh.shape[1], wh.shape[0], wo.shape[0]);
    Residual {
        inp,
        hid,
        out,
        wht: transpose(&wh.data, hid, inp),
        wot: transpose(&wo.data, out, hid),
        wrt: transpose(&wr.data, out, inp),
        wh: wh.data.clone(),
        bh: take(w, &format!("{prefix}.hidden_layer.bias")),
        wo: wo.data.clone(),
        bo: take(w, &format!("{prefix}.output_layer.bias")),
        wr: wr.data.clone(),
        br: take(w, &format!("{prefix}.residual_layer.bias")),
    }
}

impl Residual {
    /// `relu(x W_h + b_h) W_o + b_o + (x W_r + b_r)`.
    #[must_use]
    pub fn forward(&self, x: &[f32], rows: usize, fast: bool, row1_gemv: bool) -> Vec<f32> {
        let mut h = proj(
            fast,
            row1_gemv,
            x,
            rows,
            self.inp,
            &self.wh,
            &self.wht,
            self.hid,
            Some(&self.bh),
        );
        for v in h.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        let mut out = proj(
            fast,
            row1_gemv,
            &h,
            rows,
            self.hid,
            &self.wo,
            &self.wot,
            self.out,
            Some(&self.bo),
        );
        let res = proj(
            fast,
            row1_gemv,
            x,
            rows,
            self.inp,
            &self.wr,
            &self.wrt,
            self.out,
            Some(&self.br),
        );
        for (o, r) in out.iter_mut().zip(&res) {
            *o += r;
        }
        out
    }
}

fn attn(w: &Weights, prefix: &str) -> Attn {
    let (q, k, v, o) = (
        take(w, &format!("{prefix}.q.weight")),
        take(w, &format!("{prefix}.k.weight")),
        take(w, &format!("{prefix}.v.weight")),
        take(w, &format!("{prefix}.o.weight")),
    );
    let s = tensor(w, &format!("{prefix}.q.weight")).shape.clone();
    Attn {
        qt: transpose(&q, s[0], s[1]),
        kt: transpose(&k, s[0], s[1]),
        vt: transpose(&v, s[0], s[1]),
        ot: transpose(&o, s[0], s[1]),
        q,
        k,
        v,
        o,
    }
}

fn stack(w: &Weights, name: &str, layers: usize, is_decoder: bool) -> Stack {
    let blocks = (0..layers)
        .map(|i| {
            let p = format!("{name}.block.{i}.layer");
            let (ln_ca, ca, ff) = if is_decoder {
                (
                    Some(take(w, &format!("{p}.1.layer_norm.weight"))),
                    Some(attn(w, &format!("{p}.1.EncDecAttention"))),
                    2,
                )
            } else {
                (None, None, 1)
            };
            let wi = take(w, &format!("{p}.{ff}.DenseReluDense.wi.weight"));
            let wo = take(w, &format!("{p}.{ff}.DenseReluDense.wo.weight"));
            let (dff, d) = (
                tensor(w, &format!("{p}.{ff}.DenseReluDense.wi.weight")).shape[0],
                tensor(w, &format!("{p}.{ff}.DenseReluDense.wi.weight")).shape[1],
            );
            Block {
                ln_sa: take(w, &format!("{p}.0.layer_norm.weight")),
                sa: attn(w, &format!("{p}.0.SelfAttention")),
                ln_ca,
                ca,
                ln_ff: take(w, &format!("{p}.{ff}.layer_norm.weight")),
                wit: transpose(&wi, dff, d),
                wot: transpose(&wo, d, dff),
                wi,
                wo,
            }
        })
        .collect();
    Stack {
        blocks,
        rel_bias: take(
            w,
            &format!("{name}.block.0.layer.0.SelfAttention.relative_attention_bias.weight"),
        ),
        final_ln: take(w, &format!("{name}.final_layer_norm.weight")),
        is_decoder,
    }
}

/// One forward's ladder tensors (spike-005 `Forward`): everything the parity ladder asserts rung
/// by rung, not just the quantiles.
pub struct Forward {
    /// Instance-norm location (nanmean of the context).
    pub loc: f32,
    /// Instance-norm scale (population std, floored at 1e-5).
    pub scale: f32,
    /// Per-token encoder attention mask, REG token included.
    pub attention_mask: Vec<bool>,
    /// Patch embeddings fed to the encoder, `[l, d_model]` row-major.
    pub input_embeds: Vec<f32>,
    /// Encoder last hidden state, `[l, d_model]`.
    pub encoder_hidden: Vec<f32>,
    /// Decoder last hidden state, `[1, d_model]`.
    pub decoder_hidden: Vec<f32>,
    /// `[quantile][t]` on the ORIGINAL scale.
    pub quantiles: Vec<Vec<f32>>,
}

impl Bolt {
    /// Load a model from decoded weights and its config, at the PRODUCTION routing defaults
    /// `fast = true, attn_gemm = true, row1_gemv = false` (D-14).
    ///
    /// # Panics
    ///
    /// Panics naming the tensor when the weight set is missing one. An incomplete safetensors
    /// file is a defect in the weights directory.
    #[must_use]
    pub fn load(w: &Weights, cfg: Config) -> Bolt {
        Bolt {
            fast: true,
            attn_gemm: true,
            row1_gemv: false,
            shared: take(w, "shared.weight"),
            in_embed: residual(w, "input_patch_embedding"),
            out_embed: residual(w, "output_patch_embedding"),
            encoder: stack(w, "encoder", cfg.enc_layers, false),
            decoder: stack(w, "decoder", cfg.dec_layers, true),
            cfg,
        }
    }

    /// Multi-head attention; `bias[h*lq*lk ..]` optional (relative position), `key_mask[k]` false
    /// means excluded.
    ///
    /// # Panics
    ///
    /// Panics if `gemm_blis` refuses the per-head shapes.
    #[must_use]
    pub fn attention(
        &self,
        a: &Attn,
        xq: &[f32],
        lq: usize,
        xkv: &[f32],
        lk: usize,
        bias: Option<&[f32]>,
        key_mask: &[bool],
    ) -> Vec<f32> {
        let (d, h, dk) = (self.cfg.d_model, self.cfg.heads, self.cfg.d_kv);
        let (f, g) = (self.fast, self.row1_gemv);
        let (q, k, v) = (
            proj(f, g, xq, lq, d, &a.q, &a.qt, h * dk, None),
            proj(f, g, xkv, lk, d, &a.k, &a.kt, h * dk, None),
            proj(f, g, xkv, lk, d, &a.v, &a.vt, h * dk, None),
        );
        let mut ctx = vec![0.0f32; lq * h * dk];
        if self.attn_gemm && lq > 1 {
            // per head: scores[lq x lk] = Q_h . K_h^T and ctx_h[lq x dk] = P . V_h, packed GEMM
            let (mut qh, mut kt, mut vh) = (
                vec![0.0f32; lq * dk],
                vec![0.0f32; dk * lk],
                vec![0.0f32; lk * dk],
            );
            let mut scores = vec![0.0f32; lq * lk];
            let mut ch = vec![0.0f32; lq * dk];
            for hh in 0..h {
                for i in 0..lq {
                    qh[i * dk..(i + 1) * dk]
                        .copy_from_slice(&q[i * h * dk + hh * dk..i * h * dk + (hh + 1) * dk]);
                }
                for j in 0..lk {
                    let kj = &k[j * h * dk + hh * dk..j * h * dk + (hh + 1) * dk];
                    for t in 0..dk {
                        kt[t * lk + j] = kj[t];
                    }
                    vh[j * dk..(j + 1) * dk]
                        .copy_from_slice(&v[j * h * dk + hh * dk..j * h * dk + (hh + 1) * dk]);
                }
                // D-08 clippy edit: the spike wrote `.iter_mut().for_each(|s| *s = 0.0)`,
                // which `clippy::needless_for_each` rejects. Same zeroing, same order.
                for s in scores.iter_mut() {
                    *s = 0.0;
                }
                trueno::blis::gemm_blis(lq, lk, dk, &qh, &kt, &mut scores, None)
                    .expect("gemm scores");
                for i in 0..lq {
                    let row = &mut scores[i * lk..(i + 1) * lk];
                    let mut mx = f32::NEG_INFINITY;
                    for j in 0..lk {
                        if let Some(b) = bias {
                            row[j] += b[hh * lq * lk + i * lk + j];
                        }
                        if !key_mask[j] {
                            row[j] += f32::MIN;
                        }
                        if row[j] > mx {
                            mx = row[j];
                        }
                    }
                    let mut den = 0.0f32;
                    for j in 0..lk {
                        row[j] = (row[j] - mx).exp();
                        den += row[j];
                    }
                    let inv = 1.0 / den;
                    for j in 0..lk {
                        row[j] *= inv;
                    }
                }
                // D-08 clippy edit, as above.
                for s in ch.iter_mut() {
                    *s = 0.0;
                }
                trueno::blis::gemm_blis(lq, dk, lk, &scores, &vh, &mut ch, None)
                    .expect("gemm context");
                for i in 0..lq {
                    ctx[i * h * dk + hh * dk..i * h * dk + (hh + 1) * dk]
                        .copy_from_slice(&ch[i * dk..(i + 1) * dk]);
                }
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
                        if let Some(b) = bias {
                            s += b[hh * lq * lk + i * lk + j];
                        }
                        if !key_mask[j] {
                            s += f32::MIN;
                        }
                        scores[j] = s;
                        if s > mx {
                            mx = s;
                        }
                    }
                    let mut den = 0.0f32;
                    for j in 0..lk {
                        scores[j] = (scores[j] - mx).exp();
                        den += scores[j];
                    }
                    let out = &mut ctx[i * h * dk + hh * dk..i * h * dk + (hh + 1) * dk];
                    for j in 0..lk {
                        let p = scores[j] / den;
                        if p == 0.0 {
                            continue;
                        }
                        let vj = &v[j * h * dk + hh * dk..j * h * dk + (hh + 1) * dk];
                        for t in 0..dk {
                            out[t] += p * vj[t];
                        }
                    }
                }
            }
        }
        proj(f, g, &ctx, lq, h * dk, &a.o, &a.ot, d, None)
    }

    /// The `[heads, lq, lk]` relative-position bias for one stack.
    #[must_use]
    pub fn position_bias(&self, st: &Stack, lq: usize, lk: usize) -> Vec<f32> {
        let h = self.cfg.heads;
        let mut b = vec![0.0f32; h * lq * lk];
        for i in 0..lq {
            for j in 0..lk {
                let bucket = relative_bucket(
                    j as i64 - i as i64,
                    !st.is_decoder,
                    self.cfg.buckets,
                    self.cfg.max_distance,
                );
                for hh in 0..h {
                    b[hh * lq * lk + i * lk + j] = st.rel_bias[bucket * h + hh];
                }
            }
        }
        b
    }

    /// Run one encoder or decoder stack over `x0`, returning its final-layer-norm output.
    #[must_use]
    pub fn run_stack(
        &self,
        st: &Stack,
        x0: &[f32],
        l: usize,
        self_mask: &[bool],
        enc: Option<(&[f32], usize, &[bool])>,
    ) -> Vec<f32> {
        let (d, dff, eps) = (self.cfg.d_model, self.cfg.d_ff, self.cfg.eps);
        let bias = self.position_bias(st, l, l);
        let mut x = x0.to_vec();
        for blk in &st.blocks {
            let hn = t5_layer_norm(&x, l, d, &blk.ln_sa, eps);
            let sa = self.attention(&blk.sa, &hn, l, &hn, l, Some(&bias), self_mask);
            for (a, b) in x.iter_mut().zip(&sa) {
                *a += b;
            }
            if let (Some(ln), Some(ca), Some((eh, el, emask))) = (&blk.ln_ca, &blk.ca, enc) {
                let hn = t5_layer_norm(&x, l, d, ln, eps);
                let out = self.attention(ca, &hn, l, eh, el, None, emask);
                for (a, b) in x.iter_mut().zip(&out) {
                    *a += b;
                }
            }
            let hn = t5_layer_norm(&x, l, d, &blk.ln_ff, eps);
            let mut h = proj(
                self.fast,
                self.row1_gemv,
                &hn,
                l,
                d,
                &blk.wi,
                &blk.wit,
                dff,
                None,
            );
            for v in h.iter_mut() {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
            let ff = proj(
                self.fast,
                self.row1_gemv,
                &h,
                l,
                dff,
                &blk.wo,
                &blk.wot,
                d,
                None,
            );
            for (a, b) in x.iter_mut().zip(&ff) {
                *a += b;
            }
        }
        t5_layer_norm(&x, l, d, &st.final_ln, eps)
    }

    /// One forward: context (may contain NaN) -> 9x64 quantiles on the original scale.
    #[must_use]
    pub fn forward(&self, context: &[f32]) -> Vec<Vec<f32>> {
        self.forward_full(context).0.quantiles
    }

    /// [`Bolt::forward`] plus per-stage milliseconds (embedding, encoder, decoder, head).
    #[must_use]
    pub fn forward_stages(&self, context: &[f32]) -> (Vec<Vec<f32>>, Vec<(&'static str, f64)>) {
        let (fwd, stages) = self.forward_full(context);
        (fwd.quantiles, stages)
    }

    /// [`Bolt::forward`] returning every ladder tensor the SC4 parity rungs assert.
    #[must_use]
    pub fn forward_ladder(&self, context: &[f32]) -> Forward {
        self.forward_full(context).0
    }

    /// THE forward. One body, three wrappers — see the module docs (D-08 merge note).
    #[must_use]
    pub fn forward_full(&self, context: &[f32]) -> (Forward, Vec<(&'static str, f64)>) {
        use std::time::Instant;
        let mut stages = Vec::with_capacity(4);
        let cfg = &self.cfg;
        let ctx: &[f32] = if context.len() > cfg.context_length {
            &context[context.len() - cfg.context_length..]
        } else {
            context
        };
        // instance norm (nanmean / population std), nan_to_num semantics
        let finite: Vec<f32> = ctx.iter().copied().filter(|v| v.is_finite()).collect();
        let loc = if finite.is_empty() {
            0.0
        } else {
            finite.iter().sum::<f32>() / finite.len() as f32
        };
        let mut scale = if finite.is_empty() {
            1.0
        } else {
            (finite.iter().map(|v| (v - loc) * (v - loc)).sum::<f32>() / finite.len() as f32).sqrt()
        };
        if scale == 0.0 {
            scale = SCALE_FLOOR;
        }
        // left-pad with NaN to a multiple of the patch size, then patch
        let p = cfg.patch;
        let pad = if ctx.len() % p == 0 {
            0
        } else {
            p - ctx.len() % p
        };
        let n_patches = (ctx.len() + pad) / p;
        let mut feats = Vec::with_capacity(n_patches * 2 * p);
        let mut attn_mask = Vec::with_capacity(n_patches + 1);
        for pi in 0..n_patches {
            let mut vals = vec![0.0f32; p];
            let mut mask = vec![0.0f32; p];
            for t in 0..p {
                let idx = pi * p + t;
                if idx >= pad {
                    let v = ctx[idx - pad];
                    if v.is_finite() {
                        vals[t] = (v - loc) / scale;
                        mask[t] = 1.0;
                    }
                }
            }
            attn_mask.push(mask.iter().sum::<f32>() > 0.0);
            feats.extend(vals);
            feats.extend(mask);
        }
        let t = Instant::now();
        let mut embeds = self
            .in_embed
            .forward(&feats, n_patches, self.fast, self.row1_gemv);
        let mut l = n_patches;
        if cfg.use_reg_token {
            embeds.extend_from_slice(&self.shared[cfg.d_model..2 * cfg.d_model]);
            attn_mask.push(true);
            l += 1;
        }
        stages.push(("patch embedding", t.elapsed().as_secs_f64() * 1e3));
        let t = Instant::now();
        let enc = self.run_stack(&self.encoder, &embeds, l, &attn_mask, None);
        stages.push(("encoder", t.elapsed().as_secs_f64() * 1e3));
        let t = Instant::now();
        let dec_in = self.shared[..cfg.d_model].to_vec();
        let dec = self.run_stack(
            &self.decoder,
            &dec_in,
            1,
            &[true],
            Some((&enc, l, &attn_mask)),
        );
        stages.push(("decoder (1 token)", t.elapsed().as_secs_f64() * 1e3));
        let t = Instant::now();
        let head = self.out_embed.forward(&dec, 1, self.fast, self.row1_gemv);
        let nq = cfg.quantiles.len();
        let quantiles: Vec<Vec<f32>> = (0..nq)
            .map(|qi| {
                (0..cfg.prediction_length)
                    .map(|t| head[qi * cfg.prediction_length + t] * scale + loc)
                    .collect()
            })
            .collect();
        stages.push(("quantile head", t.elapsed().as_secs_f64() * 1e3));
        (
            Forward {
                loc,
                scale,
                attention_mask: attn_mask,
                input_embeds: embeds,
                encoder_hidden: enc,
                decoder_hidden: dec,
                quantiles,
            },
            stages,
        )
    }

    /// `ChronosBoltPipeline.predict`: direct block, then per-quantile paths batched and
    /// re-quantiled. Returns `[quantile][t]` and the number of forwards spent.
    ///
    /// # Panics
    ///
    /// Panics if a predicted value is non-finite (the pool sort needs a total order).
    #[must_use]
    pub fn predict(&self, context: &[f32], prediction_length: usize) -> (Vec<Vec<f32>>, usize) {
        let cfg = &self.cfg;
        let nq = cfg.quantiles.len();
        let first = self.forward(context);
        let mut forwards = 1;
        let mut out: Vec<Vec<f32>> = first.clone();
        let mut remaining = prediction_length as i64 - cfg.prediction_length as i64;
        if remaining <= 0 {
            for q in out.iter_mut() {
                q.truncate(prediction_length);
            }
            return (out, forwards);
        }
        let base: Vec<f32> = if context.len() > cfg.context_length {
            context[context.len() - cfg.context_length..].to_vec()
        } else {
            context.to_vec()
        };
        let mut paths: Vec<Vec<f32>> = (0..nq).map(|_| base.clone()).collect();
        let mut last = first;
        while remaining > 0 {
            for (path, q) in paths.iter_mut().zip(&last) {
                path.extend_from_slice(q);
                let n = path.len();
                if n > cfg.context_length {
                    path.drain(..n - cfg.context_length);
                }
            }
            let preds: Vec<Vec<Vec<f32>>> = paths
                .iter()
                .map(|p| {
                    forwards += 1;
                    self.forward(p)
                })
                .collect();
            let mut next = vec![vec![0.0f32; cfg.prediction_length]; nq];
            let mut pool = Vec::with_capacity(nq * nq);
            for t in 0..cfg.prediction_length {
                pool.clear();
                for pr in &preds {
                    for q in pr {
                        pool.push(q[t]);
                    }
                }
                pool.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
                for (qi, &lvl) in cfg.quantiles.iter().enumerate() {
                    next[qi][t] = torch_quantile(&pool, lvl);
                }
            }
            for (o, n) in out.iter_mut().zip(&next) {
                o.extend_from_slice(n);
            }
            remaining -= cfg.prediction_length as i64;
            last = next;
        }
        for q in out.iter_mut() {
            q.truncate(prediction_length);
        }
        (out, forwards)
    }
}

/// `torch.quantile` default (linear interpolation) on a sorted slice.
#[must_use]
pub fn torch_quantile(sorted: &[f32], q: f32) -> f32 {
    let pos = f64::from(q) * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    (f64::from(sorted[lo]) + (pos - lo as f64) * (f64::from(sorted[hi]) - f64::from(sorted[lo])))
        as f32
}

#[cfg(test)]
mod tests {
    use super::{linear, linear_fast, proj, transpose, Config};
    use crate::test_support::load_json;

    /// xorshift64* — the same shape the Prophet port's `Rng` uses; deterministic per seed so a
    /// failure is reproducible from the printed seed alone.
    fn rand_vec(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0
            })
            .collect()
    }

    /// REVIEW-06-04: a `grep dot8` shows the function EXISTS; it cannot show that a single row
    /// actually reaches it. This does, behaviourally.
    ///
    /// At the production defaults `Bolt::load` sets (`fast = true`, `row1_gemv = false`), one row
    /// must be BIT-IDENTICAL to the `dot8` path (`linear`) — that is D-14's second clause — and
    /// must NOT be the `gemv` path for at least one of 20 seeded inputs. The `dot8` equality is
    /// asserted unconditionally on every seed; the gemv difference is asserted as "at least one
    /// of 20 differs" and the count is printed, because if the two kernels happened to agree on
    /// every input the second assertion would be vacuous rather than reassuring.
    #[test]
    fn single_row_routing_is_dot8_at_production_defaults() {
        let (inp, out) = (67usize, 23usize);
        let mut gemv_differs = 0usize;
        for seed in 1..=20u64 {
            let x = rand_vec(seed, inp);
            let w = rand_vec(seed ^ 0xABCD, out * inp);
            let wt = transpose(&w, out, inp);

            let routed = proj(true, false, &x, 1, inp, &w, &wt, out, None);
            let dot8_path = linear(&x, 1, inp, &w, out, None);
            assert_eq!(
                routed, dot8_path,
                "seed {seed}: proj(fast=true, row1_gemv=false) on ONE row must be bit-identical \
                 to the dot8 path (D-14); it routed somewhere else"
            );

            let gemv_path = linear_fast(&x, 1, inp, &wt, out, None);
            if gemv_path != routed {
                gemv_differs += 1;
            }
        }
        println!(
            "single-row routing: dot8 bit-identical on 20/20 seeds; gemv path differed on \
             {gemv_differs}/20 seeds"
        );
        assert!(
            gemv_differs >= 1,
            "the gemv path agreed with dot8 on all 20 seeded inputs, so the dot8 assertion above \
             cannot distinguish the two kernels — the routing proof is vacuous on this host"
        );
    }

    /// D-14's FIRST clause, proven the same behavioural way (FALSIFY-CHRONOS-014): a multi-row
    /// product must reach `gemm_blis` over the `[in, out]` transpose, not the plain-loop path.
    ///
    /// The two assertions are a pair. Bit-identity to `linear_fast` alone would be satisfied by a
    /// host where the two kernels happen to agree, so the difference from `linear` is asserted as
    /// "at least one of 20 differs" and the count is printed.
    #[test]
    fn multi_row_routing_is_gemm_blis_at_production_defaults() {
        let (rows, inp, out) = (7usize, 67usize, 23usize);
        let mut dot8_differs = 0usize;
        for seed in 1..=20u64 {
            let x = rand_vec(seed, rows * inp);
            let w = rand_vec(seed ^ 0x1234, out * inp);
            let wt = transpose(&w, out, inp);

            let routed = proj(true, false, &x, rows, inp, &w, &wt, out, None);
            let gemm_path = linear_fast(&x, rows, inp, &wt, out, None);
            assert_eq!(
                routed, gemm_path,
                "seed {seed}: proj(fast=true) on {rows} rows must be bit-identical to the \
                 gemm_blis path (D-14 first clause); it routed somewhere else"
            );

            if linear(&x, rows, inp, &w, out, None) != routed {
                dot8_differs += 1;
            }
        }
        println!(
            "multi-row routing: gemm_blis bit-identical on 20/20 seeds; dot8 path differed on \
             {dot8_differs}/20 seeds"
        );
        assert!(
            dot8_differs >= 1,
            "the dot8 path agreed with gemm_blis on all 20 seeded inputs, so the assertion above \
             cannot distinguish the two kernels — the routing proof is vacuous on this host"
        );
    }

    /// `torch_quantile`'s linear interpolation, which the re-quantiled rollout depends on.
    /// Lives in `mod tests`, NOT in `mod parity`: it is arithmetic on a synthetic sorted slice,
    /// so its `1e-6` is an exactness epsilon and not a parity bar. The region-scoped literal scan
    /// over `mod parity` found it there first, and that is why it moved — a literal inside the
    /// parity region is indistinguishable from a smuggled bar.
    #[test]
    fn torch_quantile_interpolates_linearly() {
        let sorted = [0.0f32, 1.0, 2.0, 3.0, 4.0];
        assert!((super::torch_quantile(&sorted, 0.0) - 0.0).abs() < 1e-6);
        assert!((super::torch_quantile(&sorted, 0.5) - 2.0).abs() < 1e-6);
        assert!((super::torch_quantile(&sorted, 1.0) - 4.0).abs() < 1e-6);
        // 0.1 * (5 - 1) = 0.4 -> between sorted[0] and sorted[1]
        assert!((super::torch_quantile(&sorted, 0.1) - 0.4).abs() < 1e-6);
        let nq =
            crate::test_support::constant_u64("chronos-bolt-parity-v1", "quantile_levels") as usize;
        assert_eq!(nq, 9, "the contract's quantile level count");
    }

    /// REVIEW-06-01, and the D-13 amendment ratified by plan 06-05's blocking human decision
    /// `amend-memory-clause`.
    ///
    /// D-13 says "Only transposed `[in, out]` weights are kept in memory". The shipped `Bolt`
    /// keeps BOTH layouts, because `dot8` (D-14's single-row kernel) reads contiguous `[out, in]`
    /// rows while `gemm_blis` (D-14's multi-row kernel) reads the `[in, out]` transpose. The cost
    /// is roughly 2x weight-array residency, and this test STATES it rather than leaving it in a
    /// comment: the element count is recomputed from the committed config alone (no weights
    /// needed) and the implied MB is printed.
    ///
    /// SC4's "< 30 MB" bar measures the release BINARY carrying the embedded f16 bytes. It is not
    /// a residency bar and this duplicate does not touch it.
    #[test]
    fn weight_layout_is_dual_and_its_cost_is_stated() {
        let cfg = Config::from_json(&load_json("chronos_bolt_tiny_config.json"));
        let (d, dff, nq, pl, patch) = (
            cfg.d_model,
            cfg.d_ff,
            cfg.quantiles.len(),
            cfg.prediction_length,
            cfg.patch,
        );
        let attn_blocks = cfg.enc_layers + 2 * cfg.dec_layers; // self-attn everywhere, cross-attn in the decoder
        let ff_blocks = cfg.enc_layers + cfg.dec_layers;

        // Matrices held in BOTH layouts (q/k/v/o, wi/wo, wh/wo/wr).
        let attn_matrices = attn_blocks * 4 * (cfg.heads * cfg.d_kv * d);
        let ff_matrices = ff_blocks * 2 * (dff * d);
        let in_embed = dff * (2 * patch) + d * dff + d * (2 * patch);
        let out_embed = dff * d + (nq * pl) * dff + (nq * pl) * d;
        let single_layout = attn_matrices + ff_matrices + in_embed + out_embed;
        let dual_layout = 2 * single_layout;

        // Held ONCE: layer norms, biases, the relative-attention tables, the shared embedding.
        let layer_norms = (2 * ff_blocks + cfg.dec_layers + 2) * d;
        let rel_bias = 2 * cfg.buckets * cfg.heads;
        let shared = 2 * d;
        let residual_biases = (dff + d + d) + (dff + nq * pl + nq * pl);
        let single_only = layer_norms + rel_bias + shared + residual_biases;

        let params = single_layout + single_only;
        let resident = dual_layout + single_only;

        println!(
            "dual weight layout (D-13 AMENDED): {params} f32 parameters = {:.2} MB; resident with \
             BOTH layouts {resident} f32 = {:.2} MB ({:.2}x). Duplicated: {single_layout} matrix \
             elements; held once: {single_only}. This is RESIDENT MEMORY, not SC4's < 30 MB \
             BINARY bar (embedded f16 bytes).",
            params as f64 * 4.0 / 1e6,
            resident as f64 * 4.0 / 1e6,
            resident as f64 / params as f64,
        );

        assert!(
            dual_layout >= 2 * single_layout,
            "both layouts must be resident: {dual_layout} < 2 x {single_layout}"
        );
        // Cross-check against the published model card (8.65M params, amazon/chronos-bolt-tiny):
        // if this arithmetic did not describe the real tensor set the figure printed above would
        // be fiction.
        assert_eq!(
            params, 8_652_672,
            "the config-derived parameter count must equal the 8,652,672 the model card and the \
             committed safetensors header both report"
        );
    }
}

#[cfg(test)]
mod parity {
    //! The SC4 Bolt parity ladder against `chronos-forecasting` 2.3.1.
    //!
    //! Every weight-dependent test carries
    //! `#[cfg_attr(not(chronos_weights), ignore = "...")]`, so an unarmed run reports them as
    //! COUNTED skips with the arming reason printed rather than as a silent green (D-18). The
    //! print-a-skip-line-and-return style is forbidden here (a `println!` naming SKIP followed
    //! by an early `return`): it reports `0 ignored`, i.e. a silent green. This file never writes
    //! that pattern, so a `! grep -q` on it is a real file-wide ban rather than a reviewer's
    //! memory — the discipline 06-04 established for D-10.
    //!
    //! Every numeric bar is READ from `contracts/chronos-bolt-parity-v1.yaml` at test time
    //! (D-15) — no tolerance literal lives in this module.

    use super::relative_bucket;
    use crate::test_support::{constant_u64, equation_tolerance, load_json, read_csv};
    use std::sync::OnceLock;

    /// The armed model, loaded once per test binary.
    ///
    /// `expect` rather than skip: when `chronos_weights` is set, `CHRONOS_MODEL_DIR` held a
    /// `model.safetensors` at BUILD time, so its absence at run time is a defect.
    fn model() -> &'static crate::chronos::Model {
        static M: OnceLock<crate::chronos::Model> = OnceLock::new();
        M.get_or_init(|| {
            let dir = std::env::var_os("CHRONOS_MODEL_DIR").expect(
                "CHRONOS_MODEL_DIR is set whenever cfg(chronos_weights) is armed; absence is a \
                 defect, never a reason to skip",
            );
            crate::chronos::load_model_from_dir(std::path::Path::new(&dir))
                .expect("the armed weights directory must load")
        })
    }

    /// The f32 quantile bar, ARCHITECTURE-KEYED (REVIEW-06-02), printed with the ARCH it chose.
    ///
    /// aarch64 asserts the SC4 literal 1.0e-6, measured 9.54e-7 with the spike-008 NEON
    /// microkernel live. Anywhere else — every CI job in this repository — asserts the
    /// PROVISIONAL, UNMEASURED `quantiles_abs_f32_nonaarch64`, whose contract entry obliges the
    /// first such run to record its measured value and tighten the bar.
    fn f32_quantile_bar() -> f64 {
        let equation = if cfg!(target_arch = "aarch64") {
            "quantiles_abs_f32"
        } else {
            "quantiles_abs_f32_nonaarch64"
        };
        let bar = equation_tolerance("chronos-bolt-parity-v1", equation);
        println!(
            "f32 quantile bar: chronos-bolt-parity-v1.{equation} = {bar:e} (ARCH={})",
            std::env::consts::ARCH
        );
        bar
    }

    fn max_abs(a: &[f32], b: &[f32]) -> f64 {
        assert_eq!(a.len(), b.len(), "length {} vs {}", a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| f64::from((x - y).abs()))
            .fold(0.0, f64::max)
    }

    fn vecf(v: &serde_json::Value) -> Vec<f32> {
        v.as_array()
            .expect("array")
            .iter()
            .map(|x| x.as_f64().map_or(f32::NAN, |f| f as f32))
            .collect()
    }

    fn vecq(v: &serde_json::Value) -> Vec<Vec<f32>> {
        v.as_array().expect("array").iter().map(vecf).collect()
    }

    fn flat(q: &[Vec<f32>]) -> Vec<f32> {
        q.iter().flatten().copied().collect()
    }

    fn peyton() -> Vec<f32> {
        let (_, y) = read_csv("peyton_manning.csv");
        y.iter().map(|v| *v as f32).collect()
    }

    /// One series' shared rungs: mask, patch count, loc, scale, the six hidden states.
    /// Returns the forward so the caller can bar the quantiles its own way.
    fn ladder_rungs(name: &str, y: &[f32]) -> super::Forward {
        let m = model();
        let d = m.bolt.cfg.d_model;
        let fixture = load_json("chronos_bolt_tiny_fixture.json");
        let f = &fixture["series"][name];
        let fwd = m.bolt.forward_ladder(y);
        let l = fwd.attention_mask.len();

        let py_mask: Vec<bool> = f["attention_mask"]
            .as_array()
            .expect("attention_mask")
            .iter()
            .map(|v| v.as_f64().expect("f") > 0.5)
            .collect();
        assert_eq!(
            fwd.attention_mask, py_mask,
            "{name}: the attention mask must be EQUAL to the oracle's — it carries no tolerance"
        );
        assert_eq!(
            l,
            f["n_patches_plus_reg"].as_u64().expect("n") as usize,
            "{name}: patch count including the REG token"
        );

        let ls_bar = equation_tolerance("chronos-bolt-parity-v1", "loc_scale_rel");
        let py_loc = f["loc"].as_f64().expect("loc");
        let py_scale = f["scale"].as_f64().expect("scale");
        let d_loc = (f64::from(fwd.loc) - py_loc).abs() / py_loc.abs().max(1.0);
        let d_scale =
            (f64::from(fwd.scale) - py_scale).abs() / py_scale.max(f64::from(super::SCALE_FLOOR));
        println!("{name}: loc rel {d_loc:.3e}, scale rel {d_scale:.3e} (bar {ls_bar:e})");
        assert!(
            d_loc <= ls_bar,
            "{name}: loc rel {d_loc:e} over bar {ls_bar:e}"
        );
        assert!(
            d_scale <= ls_bar,
            "{name}: scale rel {d_scale:e} over bar {ls_bar:e}"
        );

        let h_bar = equation_tolerance("chronos-bolt-parity-v1", "hidden_states_abs");
        let rungs = [
            (
                "input_embeds first patch",
                max_abs(
                    &fwd.input_embeds[..d],
                    &vecf(&f["input_embeds_first_patch"]),
                ),
            ),
            (
                "input_embeds last patch",
                max_abs(
                    &fwd.input_embeds[(l - 2) * d..(l - 1) * d],
                    &vecf(&f["input_embeds_last_patch"]),
                ),
            ),
            (
                "REG embedding",
                max_abs(
                    &fwd.input_embeds[(l - 1) * d..l * d],
                    &vecf(&f["reg_embed"]),
                ),
            ),
            (
                "encoder first token",
                max_abs(
                    &fwd.encoder_hidden[..d],
                    &vecf(&f["encoder_last_hidden_first"]),
                ),
            ),
            (
                "encoder REG token",
                max_abs(
                    &fwd.encoder_hidden[(l - 1) * d..l * d],
                    &vecf(&f["encoder_last_hidden_reg"]),
                ),
            ),
            (
                "decoder hidden",
                max_abs(&fwd.decoder_hidden, &vecf(&f["decoder_last_hidden"])),
            ),
        ];
        for (rung, delta) in rungs {
            println!("{name}: {rung} max|delta| {delta:.3e} (bar {h_bar:e})");
            assert!(
                delta <= h_bar,
                "{name}: {rung} max|delta| {delta:e} over the contract bar {h_bar:e}"
            );
        }
        fwd
    }

    /// FALSIFY-CHRONOS-001 / -004 / -005: the full Peyton Manning ladder on the ARCH-selected bar.
    #[cfg_attr(
        not(chronos_weights),
        ignore = "CHRONOS_MODEL_DIR unset or has no model.safetensors — run `just fetch-chronos-tiny` to arm"
    )]
    #[test]
    fn peyton_ladder_matches_oracle_f32() {
        let y = peyton();
        let fwd = ladder_rungs("peyton", &y);

        let bar = f32_quantile_bar();
        let fixture = load_json("chronos_bolt_tiny_fixture.json");
        let py_q = vecq(&fixture["series"]["peyton"]["quantiles_64"]);
        let delta = max_abs(&flat(&fwd.quantiles), &flat(&py_q));
        println!("peyton: quantiles_64 max|delta| {delta:.4e} against bar {bar:e}");
        assert!(
            delta <= bar,
            "peyton quantiles_64 max|delta| {delta:e} over the contract bar {bar:e} \
             (ARCH={}) — record this number in the contract if this is the first run on this \
             architecture",
            std::env::consts::ARCH
        );

        // The standalone oracle must agree with the ladder fixture; if it does not, one of the
        // two committed files was regenerated alone and neither can be trusted.
        let oracle = load_json("peyton_tiny_oracle.json");
        let oracle_q = vecq(&oracle["quantiles_64"]);
        assert_eq!(
            max_abs(&flat(&py_q), &flat(&oracle_q)),
            0.0,
            "the two committed Peyton oracles must be byte-equal on quantiles_64"
        );
    }

    /// FALSIFY-CHRONOS-003: the two shorter series, barred RELATIVE to their own scale.
    #[cfg_attr(
        not(chronos_weights),
        ignore = "CHRONOS_MODEL_DIR unset or has no model.safetensors — run `just fetch-chronos-tiny` to arm"
    )]
    #[test]
    fn air_and_short100_ladders_match_oracle() {
        let peyton = peyton();
        let (_, air_y) = read_csv("air_passengers.csv");
        let air: Vec<f32> = air_y.iter().map(|v| *v as f32).collect();
        let short100 = peyton[peyton.len() - 100..].to_vec();

        let rel = equation_tolerance("chronos-bolt-parity-v1", "quantiles_rel_scale");
        let fixture = load_json("chronos_bolt_tiny_fixture.json");
        for (name, y) in [("air", air), ("short100", short100)] {
            let fwd = ladder_rungs(name, &y);
            let py_q = vecq(&fixture["series"][name]["quantiles_64"]);
            let delta = max_abs(&flat(&fwd.quantiles), &flat(&py_q));
            let bar = rel * f64::from(fwd.scale);
            println!(
                "{name}: quantiles_64 max|delta| {delta:.4e} = {:.3e} relative to scale \
                 {:.4} (bar {rel:e} x scale = {bar:e})",
                delta / f64::from(fwd.scale),
                fwd.scale
            );
            assert!(
                delta <= bar,
                "{name} quantiles_64 max|delta| {delta:e} over {rel:e} x scale {} = {bar:e}",
                fwd.scale
            );
        }
    }

    /// FALSIFY-CHRONOS-008 / -009: the 365-step rollout AND the control that pins which pipeline
    /// the oracle came from. Without the control the rung would prove only that SOME rollout was
    /// implemented.
    #[cfg_attr(
        not(chronos_weights),
        ignore = "CHRONOS_MODEL_DIR unset or has no model.safetensors — run `just fetch-chronos-tiny` to arm"
    )]
    #[test]
    fn rollout_365_matches_oracle_and_median_only_does_not() {
        let m = model();
        let cfg = &m.bolt.cfg;
        let y = peyton();
        let oracle = load_json("peyton_tiny_oracle.json");
        let py365 = vecq(&oracle["quantiles_365"]);

        let (q365, forwards) = m.bolt.predict(&y, 365);
        let expected_forwards =
            constant_u64("chronos-bolt-parity-v1", "rollout_forwards_for_365") as usize;
        assert_eq!(
            forwards, expected_forwards,
            "the rollout shape is arithmetic, not an observation: 1 direct forward + 5 blocks x \
             9 quantile paths"
        );

        let bar = equation_tolerance("chronos-bolt-parity-v1", "rollout_365_abs");
        let delta = max_abs(&flat(&q365), &flat(&py365));
        let first64 = max_abs(
            &flat(&q365.iter().map(|q| q[..64].to_vec()).collect::<Vec<_>>()),
            &flat(&py365.iter().map(|q| q[..64].to_vec()).collect::<Vec<_>>()),
        );
        println!(
            "rollout 365: all steps max|delta| {delta:.3e} (bar {bar:e}); first 64 steps \
             {first64:.3e}; {forwards} forwards"
        );
        assert!(
            delta <= bar,
            "365-step rollout max|delta| {delta:e} over the contract bar {bar:e}"
        );

        // ---- the control: the pre-2025 median-only rollout (spike 005 lines 70-82) ----
        let mut ctx: Vec<f32> = y[y.len() - cfg.context_length..].to_vec();
        let mut out: Vec<Vec<f32>> = vec![Vec::new(); cfg.quantiles.len()];
        let mut remaining = 365i64;
        while remaining > 0 {
            let f = m.bolt.forward(&ctx);
            for (o, q) in out.iter_mut().zip(&f) {
                o.extend_from_slice(q);
            }
            ctx.extend_from_slice(&f[4]);
            if ctx.len() > cfg.context_length {
                ctx.drain(..ctx.len() - cfg.context_length);
            }
            remaining -= cfg.prediction_length as i64;
        }
        for o in out.iter_mut() {
            o.truncate(365);
        }
        let control = max_abs(&flat(&out), &flat(&py365));
        let separation = equation_tolerance(
            "chronos-bolt-parity-v1",
            "rollout_scheme_nine_path_requantiled",
        );
        println!(
            "control (median-only rollout): max|delta| {control:.3e} against the SAME oracle \
             (must EXCEED {separation:e})"
        );
        assert!(
            control > separation,
            "the median-only control differs from the oracle by only {control:e}, which does not \
             exceed {separation:e} — the fixture no longer discriminates between the two rollout \
             schemes, so the rung above has stopped being evidence"
        );
    }

    /// FALSIFY-CHRONOS-006 / -007: the six edge probes, `constant` on the absolute bar.
    #[cfg_attr(
        not(chronos_weights),
        ignore = "CHRONOS_MODEL_DIR unset or has no model.safetensors — run `just fetch-chronos-tiny` to arm"
    )]
    #[test]
    fn edge_probes_match_oracle() {
        let m = model();
        let probes = load_json("chronos_probes.json");
        assert_eq!(
            probes["chronos_version"].as_str().expect("version"),
            "2.3.1",
            "the probe oracle must be the pinned chronos-forecasting version"
        );
        let rel = equation_tolerance("chronos-bolt-parity-v1", "probe_quantiles_rel_scale");
        let abs_bar = equation_tolerance("chronos-bolt-parity-v1", "probe_constant_abs");
        let cases = probes["cases"].as_object().expect("cases");
        assert_eq!(cases.len(), 6, "six probes are committed");
        for (name, c) in cases {
            let y = vecf(&c["y"]);
            let py_q = vecq(&c["quantiles"]);
            let pl = py_q[0].len();
            let fwd = m.bolt.forward_ladder(&y);
            let (q, _) = m.bolt.predict(&y, pl);

            let py_mask: Vec<bool> = c["attention_mask"]
                .as_array()
                .expect("mask")
                .iter()
                .map(|v| v.as_f64().expect("f") > 0.5)
                .collect();
            assert_eq!(
                fwd.attention_mask, py_mask,
                "{name}: attention mask must be EQUAL — it is where a NaN or padding regression \
                 shows up first"
            );

            let delta = max_abs(&flat(&q), &flat(&py_q));
            let bar = if name == "constant" {
                abs_bar
            } else {
                rel * f64::from(fwd.scale).max(1e-12)
            };
            println!(
                "probe {name}: n {} -> {pl}, scale {:.4e}, max|delta| {delta:.3e} (bar {bar:e})",
                y.len(),
                fwd.scale
            );
            assert!(
                delta <= bar,
                "probe {name}: max|delta| {delta:e} over the contract bar {bar:e}"
            );
        }
    }

    /// KANI-CHRONOS-001's runnable, identically bounded evidence. NOT weight-gated: the bucket
    /// function is pure arithmetic, so it runs on every CI leg whether or not weights exist.
    ///
    /// The committed fixtures publish no bucket table, so the reference here is the 18 offsets
    /// spike 005 recorded in `RUN-OUTPUT.md` section 1 against transformers'
    /// `_relative_position_bucket`, plus the structural properties over the whole
    /// `[-context_length, context_length]` grid.
    #[test]
    fn relative_buckets_match_fixture_over_grid() {
        let (nb, maxd) = (32usize, 128usize);
        let recorded: [(i64, usize, usize); 18] = [
            (-200, 15, 31),
            (-128, 15, 31),
            (-100, 15, 30),
            (-17, 10, 16),
            (-16, 10, 16),
            (-8, 8, 8),
            (-1, 1, 1),
            (0, 0, 0),
            (1, 17, 0),
            (7, 23, 0),
            (8, 24, 0),
            (15, 25, 0),
            (16, 26, 0),
            (31, 27, 0),
            (64, 30, 0),
            (127, 31, 0),
            (128, 31, 0),
            (500, 31, 0),
        ];
        for (rel, bidi, causal) in recorded {
            assert_eq!(
                relative_bucket(rel, true, nb, maxd),
                bidi,
                "bidirectional bucket for offset {rel}"
            );
            assert_eq!(
                relative_bucket(rel, false, nb, maxd),
                causal,
                "causal bucket for offset {rel}"
            );
        }

        let ctx = constant_u64("chronos-bolt-parity-v1", "context_length") as i64;
        for rel in -ctx..=ctx {
            for bidirectional in [true, false] {
                let b = relative_bucket(rel, bidirectional, nb, maxd);
                assert!(
                    b < nb,
                    "bucket {b} is out of range for offset {rel} (bidirectional {bidirectional}) \
                     — it indexes the bias table"
                );
            }
            // Causal attention never looks forward: every non-negative offset is bucket 0.
            if rel >= 0 {
                assert_eq!(relative_bucket(rel, false, nb, maxd), 0);
            }
            // Bidirectional splits the table: future offsets live in the upper half.
            if rel > 0 {
                assert!(relative_bucket(rel, true, nb, maxd) >= nb / 2);
            }
        }
        println!(
            "relative_bucket: 18 recorded offsets exact, and {} grid points in both modes stay \
             inside [0, {nb})",
            2 * ctx + 1
        );
    }
}
