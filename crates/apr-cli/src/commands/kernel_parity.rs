//! `apr kernel parity` — the PRODUCER whose output `apr attn-parity-lint` reads
//! (aprender#2377 finding 3).
//!
//! CRUX-L-02 shipped a consumer with no producer: `attn-parity-lint`'s help
//! documented `apr kernel parity --impl flash2 --ref naive --json` and the
//! shipped binary had no `kernel` command at all, so the parity, provenance and
//! head-dim gates had never run on real data and could not.
//!
//! ## What this measures, and what it refuses to claim
//!
//! `--impl tiled` runs the **in-tree** tiled online-softmax attention kernel
//! (`realizar::brick::FlashAttentionBrick`, the FlashAttention-2 tiling scheme
//! of Dao 2023 / Milakov & Gimelshein 2018) against a naive reference written
//! here, which materialises the score row and does a plain max-subtracted
//! softmax. Both consume the same seeded Q/K/V, so `max_abs_diff` and
//! `cosine_sim` are a real measurement of two independent implementations — it
//! can fail, and a regression in the shipped brick would make it fail.
//!
//! `--impl flash2` means the pinned `hf-kernels-community:flash-attn2@<sha>`
//! CUDA kernel. **This binary embeds no such kernel.** Asking for it is
//! REFUSED with a non-zero exit and a message saying so. It is never quietly
//! answered by the tiled path, and `attn_impl: "flash2"` is never emitted with
//! a `kernel_source` we did not load — that fabricated-provenance line is the
//! whole reason CRUX-L-02 pins the `pkg@sha` format.
//!
//! Shape note: the brick is a decode-step kernel — ONE query position attending
//! over a `seq_len`-long KV cache. The observation says so. No claim is made
//! about prefill or about causal masking across a query block.

use std::path::Path;

use serde::Serialize;

use crate::error::{refuse_overwrite, CliError, Result};

/// `--impl` / `--ref` are declared at the crate root (`extended_commands.rs`)
/// because `ExtendedCommands` is public and `mod commands` is not.
pub(crate) use crate::{KernelImpl, KernelRef};

/// head_dim values the pinned flash-attn2 kernel dispatches
/// (`contracts/crux-L-02-v1.yaml` § `flash_attn2_dispatch`, arXiv:2307.08691).
pub(crate) const FLASH2_SUPPORTED_HEAD_DIMS: [usize; 2] = [64, 128];

/// The parity + provenance observation `apr attn-parity-lint` consumes.
///
/// One body serves both `--parity-file` and `--provenance-file`: the numerics
/// gate reads `max_abs_diff`/`cosine_sim`, the provenance gate reads
/// `attn_impl`/`kernel_source`/`fallback`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct ParityObservation {
    /// Implementation under test, as named on the command line.
    pub kernel: String,
    /// Reference implementation.
    pub reference: String,
    /// KV cache length attended over.
    pub seq_len: usize,
    /// Query head count.
    pub num_heads: usize,
    /// Key/value head count (GQA groups when smaller than `num_heads`).
    pub num_kv_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Seed for the Q/K/V draw.
    pub seed: u64,
    /// Attention shape measured — decode step, one query position.
    pub regime: String,
    /// Largest absolute elementwise difference between the two outputs.
    pub max_abs_diff: f64,
    /// Cosine similarity of the two flattened outputs.
    pub cosine_sim: f64,
    /// Provenance discriminant read by `classify_provenance`.
    pub attn_impl: String,
    /// Pinned `pkg@sha` when — and only when — that kernel actually ran.
    pub kernel_source: Option<String>,
    /// Why `attn_impl` is not `flash2`. Never empty when it is not.
    pub fallback: Option<String>,
}

/// Dimensions requested on the command line.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ParityDims {
    pub seq_len: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub seed: u64,
}

/// Run the producer.
pub(crate) fn run(
    kernel: KernelImpl,
    reference: KernelRef,
    dims: ParityDims,
    json: bool,
    output: Option<&Path>,
    force: bool,
) -> Result<()> {
    if let Some(out) = output {
        refuse_overwrite(out, force)?;
    }
    if let Err(err) = validate_dims(&dims) {
        emit_error(&err, json, output)?;
        return Err(CliError::ValidationFailed(err));
    }
    if kernel == KernelImpl::Flash2 {
        return refuse_flash2(&dims, json, output);
    }

    let obs = measure_tiled(reference, dims)?;
    let rendered = if json {
        serde_json::to_string_pretty(&obs).map_err(|e| {
            CliError::InvalidInput(format!("apr kernel parity: cannot serialize: {e}"))
        })?
    } else {
        render_text(&obs)
    };
    write_out(&rendered, output)
}

fn write_out(rendered: &str, output: Option<&Path>) -> Result<()> {
    match output {
        Some(out) => std::fs::write(out, format!("{rendered}\n"))?,
        None => println!("{rendered}"),
    }
    Ok(())
}

fn render_text(o: &ParityObservation) -> String {
    format!(
        "kernel parity {} vs {}\n  regime      : {}\n  dims        : seq_len={} heads={} \
         kv_heads={} head_dim={} seed={}\n  max_abs_diff: {:e}\n  cosine_sim  : {}\n  \
         attn_impl   : {}\n  provenance  : {}",
        o.kernel,
        o.reference,
        o.regime,
        o.seq_len,
        o.num_heads,
        o.num_kv_heads,
        o.head_dim,
        o.seed,
        o.max_abs_diff,
        o.cosine_sim,
        o.attn_impl,
        o.kernel_source
            .clone()
            .or_else(|| o.fallback.clone())
            .unwrap_or_default()
    )
}

/// Refuse `--impl flash2`, emitting the error JSON the head-dim gate reads.
fn refuse_flash2(dims: &ParityDims, json: bool, output: Option<&Path>) -> Result<()> {
    if !FLASH2_SUPPORTED_HEAD_DIMS.contains(&dims.head_dim) {
        // `{:?}` on the array would render `[64, 128]`, which reads as a closed
        // INTERVAL — head_dim 96 would look supported. It is a two-element SET.
        let supported = FLASH2_SUPPORTED_HEAD_DIMS
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let msg = format!(
            "unsupported-head-dim: {} — flash2 dispatches only head_dim ∈ {{{supported}}} \
             (contracts/crux-L-02-v1.yaml § flash_attn2_dispatch, arXiv:2307.08691)",
            dims.head_dim
        );
        emit_error(&msg, json, output)?;
        return Err(CliError::ValidationFailed(msg));
    }
    let msg = format!(
        "flash2-kernel-unavailable: this binary embeds no \
         hf-kernels-community:flash-attn2 kernel, so no flash2 measurement exists to report. \
         Rerun with `--impl tiled` to measure the in-tree tiled kernel at head_dim {}.",
        dims.head_dim
    );
    emit_error(&msg, json, output)?;
    Err(CliError::NotImplemented(msg))
}

/// Write `{"error": ...}` so a refusal is still a capturable observation.
fn emit_error(message: &str, json: bool, output: Option<&Path>) -> Result<()> {
    if !json {
        return Ok(());
    }
    let body = serde_json::json!({ "error": message });
    let rendered = serde_json::to_string_pretty(&body).unwrap_or_default();
    write_out(&rendered, output)
}

fn validate_dims(d: &ParityDims) -> std::result::Result<(), String> {
    if d.head_dim == 0 {
        return Err("unsupported-head-dim: 0 — head_dim must be positive".to_string());
    }
    if d.seq_len == 0 {
        return Err("apr kernel parity: --seq-len must be positive".to_string());
    }
    if d.num_heads == 0 || d.num_kv_heads == 0 {
        return Err(
            "apr kernel parity: --num-heads and --num-kv-heads must be positive".to_string(),
        );
    }
    if d.num_heads % d.num_kv_heads != 0 {
        return Err(format!(
            "apr kernel parity: --num-heads {} is not a multiple of --num-kv-heads {} \
             (GQA needs whole groups)",
            d.num_heads, d.num_kv_heads
        ));
    }
    Ok(())
}

// ── the measurement ──────────────────────────────────────────────────────

/// Deterministic SplitMix64 draw, so `--seed` really pins the inputs.
///
/// Shared with `embed_viz`'s random projection: one seeded stream, one place to
/// audit its determinism.
pub(crate) struct SplitMix64(u64);

impl SplitMix64 {
    /// Seed the stream.
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [-1, 1).
    pub(crate) fn next_unit(&mut self) -> f32 {
        // Top 24 bits scaled into [0, 1), then mapped to [-1, 1).
        const SCALE: f32 = 1.0 / 16_777_216.0;
        let unit = (self.next_u64() >> 40) as f32 * SCALE;
        unit.mul_add(2.0, -1.0)
    }
}

fn draw(rng: &mut SplitMix64, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.next_unit()).collect()
}

/// Naive reference: materialise the score row, max-subtracted softmax, weighted V.
pub(crate) fn naive_attention(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    dims: &ParityDims,
) -> Vec<f32> {
    let scale = 1.0 / (dims.head_dim as f32).sqrt();
    let group = dims.num_heads / dims.num_kv_heads;
    let mut out = vec![0.0f32; dims.num_heads * dims.head_dim];
    for h in 0..dims.num_heads {
        let kv = h / group;
        let q = &query[h * dims.head_dim..(h + 1) * dims.head_dim];
        let mut scores = Vec::with_capacity(dims.seq_len);
        for s in 0..dims.seq_len {
            let base = (s * dims.num_kv_heads + kv) * dims.head_dim;
            let dot: f32 = (0..dims.head_dim).map(|d| q[d] * keys[base + d]).sum();
            scores.push(dot * scale);
        }
        let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut denom = 0.0f32;
        for s in &mut scores {
            *s = (*s - m).exp();
            denom += *s;
        }
        for (s, p) in scores.iter().enumerate() {
            let base = (s * dims.num_kv_heads + kv) * dims.head_dim;
            for d in 0..dims.head_dim {
                out[h * dims.head_dim + d] += p * values[base + d];
            }
        }
        for d in 0..dims.head_dim {
            out[h * dims.head_dim + d] /= denom;
        }
    }
    out
}

/// Largest absolute elementwise difference.
pub(crate) fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| f64::from((x - y).abs()))
        .fold(0.0f64, f64::max)
}

/// Cosine similarity of two flattened outputs; `None` when either has zero norm.
pub(crate) fn cosine_sim(a: &[f32], b: &[f32]) -> Option<f64> {
    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let na: f64 = a
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = b
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt();
    if na == 0.0 || nb == 0.0 {
        return None;
    }
    Some((dot / (na * nb)).clamp(-1.0, 1.0))
}

/// Draw seeded Q/K/V for `dims`.
pub(crate) fn draw_qkv(dims: &ParityDims) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut rng = SplitMix64(dims.seed ^ 0x5DEE_CE66_D1F5_1A3B);
    let q = draw(&mut rng, dims.num_heads * dims.head_dim);
    let kv_len = dims.seq_len * dims.num_kv_heads * dims.head_dim;
    let k = draw(&mut rng, kv_len);
    let v = draw(&mut rng, kv_len);
    (q, k, v)
}

const TILED_FALLBACK_REASON: &str =
    "in-tree tiled online-softmax kernel (realizar::brick::FlashAttentionBrick); \
     this binary embeds no hf-kernels-community:flash-attn2 kernel, so no flash2 \
     provenance sha exists to pin";

#[cfg(feature = "inference")]
fn measure_tiled(reference: KernelRef, dims: ParityDims) -> Result<ParityObservation> {
    use realizar::brick::FlashAttentionBrick;

    let (q, k, v) = draw_qkv(&dims);
    let brick = FlashAttentionBrick::new(dims.num_heads, dims.num_kv_heads, dims.head_dim);
    let tiled = brick.forward(&q, &k, &v, dims.seq_len).map_err(|e| {
        CliError::ValidationFailed(format!(
            "apr kernel parity: tiled kernel refused input: {e}"
        ))
    })?;
    let naive = naive_attention(&q, &k, &v, &dims);
    let cos = cosine_sim(&tiled, &naive).ok_or_else(|| {
        CliError::ValidationFailed(
            "apr kernel parity: an output vector has zero norm, so cosine similarity is undefined"
                .to_string(),
        )
    })?;

    Ok(ParityObservation {
        kernel: "tiled".to_string(),
        reference: match reference {
            KernelRef::Naive => "naive".to_string(),
        },
        seq_len: dims.seq_len,
        num_heads: dims.num_heads,
        num_kv_heads: dims.num_kv_heads,
        head_dim: dims.head_dim,
        seed: dims.seed,
        regime: "decode-step: 1 query position over a seq_len KV cache".to_string(),
        max_abs_diff: max_abs_diff(&tiled, &naive),
        cosine_sim: cos,
        attn_impl: "fallback".to_string(),
        kernel_source: None,
        fallback: Some(TILED_FALLBACK_REASON.to_string()),
    })
}

#[cfg(not(feature = "inference"))]
fn measure_tiled(_reference: KernelRef, _dims: ParityDims) -> Result<ParityObservation> {
    Err(CliError::FeatureDisabled(
        "apr kernel parity --impl tiled needs the `inference` feature (it runs \
         realizar::brick::FlashAttentionBrick); rebuild with --features inference"
            .to_string(),
    ))
}

#[cfg(test)]
#[path = "kernel_parity_tests.rs"]
mod tests;
