//! PRA-001 T11: the `sparse-logits-v1` blob (spec PRA-001 §2.7).
//!
//! Per output token: the sampled id, the top-k ids with their log-softmax at
//! temperature 1.0 (before any sampling filter), the full-vocab logsumexp, and
//! the retained mass `M = Σ exp(logprobs) ≤ 1`. Storing `M` is the point: a
//! top-k cache without its residual mass is a biased teacher estimate, and
//! the tail-bucket correction of a sparse forward-KL objective needs it.
//!
//! Layout: `MAGIC`, a u32-LE header length, the header JSON, then one zstd
//! frame holding the columns (all LE): sampled ids `n×u32`, ids `n·k×u32`,
//! logprobs `n·k×f16`, logsumexp `n×f32`, mass `n×f16`. The decoder is
//! strict: an unknown header key, a short column, a trailing byte, or a mass
//! that disagrees with its own logprobs is an error, never a best effort.

use half::f16;
use serde::{Deserialize, Serialize};

pub const SCHEME: &str = "sparse-logits-v1";
/// Magic prefix; the trailing newline keeps `head -c` output readable.
pub const MAGIC: &[u8; 8] = b"SPLOGV1\n";
/// PRA-001 §2.7 default `k` `[A]`.
pub const DEFAULT_K: u8 = 20;
/// zstd level for the column frame.
const ZSTD_LEVEL: i32 = 3;
/// f16 has an 11-bit significand: Σ of k rounded exps can drift this much
/// from the rounded mass. Anything larger is a producer defect.
const MASS_TOLERANCE: f32 = 1.0 / 256.0;

/// How the token set was chosen. `RsSample` is reserved (§2.7) so the 27B
/// teacher can write importance-sampled sets later without a schema change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Mode {
    Topk,
    RsSample,
}

/// Per-blob header (§2.7).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Header {
    pub schema: String,
    pub mode: Mode,
    pub model_sha256: String,
    pub tokenizer_sha256: String,
    pub apr_tag: String,
    /// The backend that actually served the row (`served_by`).
    pub backend: String,
    /// Sampling temperature of the recorded run. Logprobs are always T = 1.0.
    pub temperature_of_record: f32,
    pub k: u8,
    pub n_tokens: u32,
}

/// One output token's sparse distribution.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenLogits {
    pub token_id_sampled: u32,
    /// Top-k ids, descending by logprob. u32: Qwen's vocab exceeds 2^16.
    pub ids: Vec<u32>,
    /// log-softmax at T = 1.0, same order as `ids`.
    pub logprobs: Vec<f16>,
    pub logsumexp_full: f32,
    /// Retained mass `M = Σ exp(logprobs)`.
    pub topk_mass: f16,
}

impl TokenLogits {
    /// Build from a full-vocab logit row at T = 1.0. `None` when `k` is 0,
    /// exceeds the vocab, or a logit is not finite.
    #[must_use]
    pub fn from_full_logits(logits: &[f32], k: u8, token_id_sampled: u32) -> Option<Self> {
        let k = usize::from(k);
        if k == 0 || k > logits.len() || logits.iter().any(|x| !x.is_finite()) {
            return None;
        }
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f64 = logits.iter().map(|&x| f64::from(x - max).exp()).sum();
        let lse = f64::from(max) + sum.ln();
        let mut order: Vec<usize> = (0..logits.len()).collect();
        // Descending logit, ascending id on ties: deterministic.
        order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
        order.truncate(k);
        let lp: Vec<f64> = order.iter().map(|&i| f64::from(logits[i]) - lse).collect();
        let mass: f64 = lp.iter().map(|x| x.exp()).sum();
        Some(Self {
            token_id_sampled,
            ids: order
                .iter()
                .map(|&i| u32::try_from(i).ok())
                .collect::<Option<_>>()?,
            logprobs: lp.iter().map(|&x| f16::from_f64(x)).collect(),
            logsumexp_full: lse as f32,
            topk_mass: f16::from_f64(mass.min(1.0)),
        })
    }

    /// Residual (tail) mass `1 − M`, clamped at 0.
    #[must_use]
    pub fn residual_mass(&self) -> f32 {
        (1.0 - self.topk_mass.to_f32()).max(0.0)
    }

    fn check(&self, k: usize, at: usize) -> Result<(), String> {
        if self.ids.len() != k || self.logprobs.len() != k {
            return Err(format!(
                "token {at}: {} ids / {} logprobs, header k = {k}",
                self.ids.len(),
                self.logprobs.len()
            ));
        }
        if !self.logsumexp_full.is_finite() {
            return Err(format!("token {at}: logsumexp_full is not finite"));
        }
        let mut sum = 0.0_f32;
        for lp in &self.logprobs {
            let v = lp.to_f32();
            if !v.is_finite() || v > 0.0 {
                return Err(format!(
                    "token {at}: logprob {v} is not a finite log-probability"
                ));
            }
            sum += v.exp();
        }
        let m = self.topk_mass.to_f32();
        if !(0.0..=1.0 + MASS_TOLERANCE).contains(&m) {
            return Err(format!("token {at}: topk_mass {m} outside [0, 1]"));
        }
        if (sum - m).abs() > MASS_TOLERANCE {
            return Err(format!("token {at}: topk_mass {m} != Σexp(logprobs) {sum}"));
        }
        Ok(())
    }
}

/// PRM C11 capture: the `choices[0].logprobs.content` of an `apr serve`
/// `/v1/chat/completions` body (#4026, `top_logprobs = k`) as one
/// [`TokenLogits`] per output token, ready for [`encode`].
///
/// Refused, never guessed: a body without `logsumexp_full` (a hosted or
/// foreign server — its tail mass is unknowable, and hosted output never
/// enters the corpus), fewer than `k` alternatives, an entry without
/// `token_id`, or logprobs that are not a descending, finite log-softmax.
pub fn from_chat_completion(body: &serde_json::Value, k: u8) -> Result<Vec<TokenLogits>, String> {
    let content = body
        .pointer("/choices/0/logprobs/content")
        .and_then(serde_json::Value::as_array)
        .ok_or("no choices[0].logprobs.content: the request did not ask for logprobs")?;
    content
        .iter()
        .enumerate()
        .map(|(at, entry)| {
            token_from_serve_entry(entry, usize::from(k)).map_err(|e| format!("token {at}: {e}"))
        })
        .collect()
}

fn serve_id(v: &serde_json::Value) -> Result<u32, String> {
    v.get("token_id")
        .and_then(serde_json::Value::as_u64)
        .and_then(|id| u32::try_from(id).ok())
        .ok_or_else(|| "no u32 token_id".to_string())
}

fn token_from_serve_entry(entry: &serde_json::Value, k: usize) -> Result<TokenLogits, String> {
    if k == 0 {
        return Err("k = 0".into());
    }
    let lse = entry
        .get("logsumexp_full")
        .and_then(serde_json::Value::as_f64)
        .filter(|x| x.is_finite())
        .ok_or("no finite logsumexp_full: not an apr serve #4026 body")?;
    let top = entry
        .get("top_logprobs")
        .and_then(serde_json::Value::as_array)
        .ok_or("no top_logprobs")?;
    if top.len() < k {
        return Err(format!("{} alternatives, k = {k}", top.len()));
    }
    let mut ids = Vec::with_capacity(k);
    let mut lps = Vec::with_capacity(k);
    for alt in &top[..k] {
        let lp = alt
            .get("logprob")
            .and_then(serde_json::Value::as_f64)
            .filter(|x| x.is_finite() && *x <= 0.0)
            .ok_or("an alternative's logprob is not a finite value ≤ 0")?;
        if lps.last().is_some_and(|&prev: &f64| lp > prev) {
            return Err("top_logprobs are not descending".into());
        }
        ids.push(serve_id(alt)?);
        lps.push(lp);
    }
    let mass: f64 = lps.iter().map(|x| x.exp()).sum();
    Ok(TokenLogits {
        token_id_sampled: serve_id(entry)?,
        ids,
        logprobs: lps.iter().map(|&x| f16::from_f64(x)).collect(),
        logsumexp_full: lse as f32,
        topk_mass: f16::from_f64(mass.min(1.0)),
    })
}

fn check_all(h: &Header, tokens: &[TokenLogits]) -> Result<(), String> {
    if h.schema != SCHEME {
        return Err(format!("schema {} is not {SCHEME}", h.schema));
    }
    if h.k == 0 {
        return Err("k = 0".into());
    }
    if usize::try_from(h.n_tokens).ok() != Some(tokens.len()) {
        return Err(format!(
            "header n_tokens {} != {} tokens",
            h.n_tokens,
            tokens.len()
        ));
    }
    if !h.temperature_of_record.is_finite() || h.temperature_of_record < 0.0 {
        return Err("temperature_of_record is not a finite value ≥ 0".into());
    }
    for (name, v) in [
        ("model_sha256", &h.model_sha256),
        ("tokenizer_sha256", &h.tokenizer_sha256),
    ] {
        if v.len() != 64 || !v.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(format!("{name} is not a sha256 hex digest"));
        }
    }
    tokens
        .iter()
        .enumerate()
        .try_for_each(|(i, t)| t.check(usize::from(h.k), i))
}

/// Encode one blob. Refuses a header/token set the decoder would reject, so
/// a blob that exists is a blob that reads.
pub fn encode(h: &Header, tokens: &[TokenLogits]) -> Result<Vec<u8>, String> {
    check_all(h, tokens)?;
    let head = serde_json::to_vec(h).map_err(|e| e.to_string())?;
    let mut cols = Vec::with_capacity(tokens.len() * (usize::from(h.k) * 6 + 10));
    for t in tokens {
        cols.extend_from_slice(&t.token_id_sampled.to_le_bytes());
    }
    for t in tokens {
        t.ids
            .iter()
            .for_each(|x| cols.extend_from_slice(&x.to_le_bytes()));
    }
    for t in tokens {
        t.logprobs
            .iter()
            .for_each(|x| cols.extend_from_slice(&x.to_bits().to_le_bytes()));
    }
    for t in tokens {
        cols.extend_from_slice(&t.logsumexp_full.to_le_bytes());
    }
    for t in tokens {
        cols.extend_from_slice(&t.topk_mass.to_bits().to_le_bytes());
    }
    let body = zstd::bulk::compress(&cols, ZSTD_LEVEL).map_err(|e| e.to_string())?;
    let len = u32::try_from(head.len()).map_err(|_| "header too large".to_string())?;
    let mut out = Vec::with_capacity(MAGIC.len() + 4 + head.len() + body.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&head);
    out.extend_from_slice(&body);
    Ok(out)
}

/// Uncompressed column bytes for `n` tokens at `k` (the M6 "raw" figure).
#[must_use]
pub fn raw_column_bytes(n: usize, k: usize) -> usize {
    n * (4 + k * (4 + 2) + 4 + 2)
}

struct Cursor<'a>(&'a [u8]);

impl Cursor<'_> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N], String> {
        if self.0.len() < N {
            return Err("column data is short".into());
        }
        let (a, b) = self.0.split_at(N);
        self.0 = b;
        a.try_into().map_err(|_| "column data is short".to_string())
    }
}

/// Decode and validate one blob.
pub fn decode(blob: &[u8]) -> Result<(Header, Vec<TokenLogits>), String> {
    let rest = blob.strip_prefix(MAGIC.as_slice()).ok_or("bad magic")?;
    if rest.len() < 4 {
        return Err("truncated header length".into());
    }
    let (len, rest) = rest.split_at(4);
    let len = u32::from_le_bytes(len.try_into().map_err(|_| "header length")?) as usize;
    if rest.len() < len {
        return Err("truncated header".into());
    }
    let (head, body) = rest.split_at(len);
    let h: Header = serde_json::from_slice(head).map_err(|e| format!("header: {e}"))?;
    let (n, k) = (h.n_tokens as usize, usize::from(h.k));
    let want = raw_column_bytes(n, k);
    let cols = zstd::bulk::decompress(body, want + 1).map_err(|e| format!("columns: {e}"))?;
    if cols.len() != want {
        return Err(format!(
            "columns hold {} bytes, header implies {want}",
            cols.len()
        ));
    }
    let mut c = Cursor(&cols);
    let mut sampled = Vec::with_capacity(n);
    for _ in 0..n {
        sampled.push(u32::from_le_bytes(c.take()?));
    }
    let mut ids = Vec::with_capacity(n * k);
    for _ in 0..n * k {
        ids.push(u32::from_le_bytes(c.take()?));
    }
    let mut lps = Vec::with_capacity(n * k);
    for _ in 0..n * k {
        lps.push(f16::from_bits(u16::from_le_bytes(c.take()?)));
    }
    let mut lse = Vec::with_capacity(n);
    for _ in 0..n {
        lse.push(f32::from_le_bytes(c.take()?));
    }
    let mut tokens = Vec::with_capacity(n);
    for (i, (&s, &l)) in sampled.iter().zip(&lse).enumerate() {
        tokens.push(TokenLogits {
            token_id_sampled: s,
            ids: ids[i * k..(i + 1) * k].to_vec(),
            logprobs: lps[i * k..(i + 1) * k].to_vec(),
            logsumexp_full: l,
            topk_mass: f16::from_bits(u16::from_le_bytes(c.take()?)),
        });
    }
    check_all(&h, &tokens)?;
    Ok((h, tokens))
}

#[cfg(test)]
#[path = "sparse_logits_tests.rs"]
mod tests;
