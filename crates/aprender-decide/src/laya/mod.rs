//! Laya: a ModernBERT encoder with a typed decision head — the first (and today only)
//! [`DecisionMethod`](crate::DecisionMethod) (D-14).
//!
//! ```text
//! ids --ModernBertEncoder (core, prefix "encoder.")--> [l, d]
//!     + type_emb[qtype]
//!     --head_layers x HeadLayer (pre-norm TransformerEncoderLayer, nhead = max(1, d/64))-->
//!     gather at the [MASK] markers --Scorer (LayerNorm -> Linear -> GELU -> Linear)--> K logits
//!     --softmax(z / T), T = calibrated bucket temperature--> K probabilities
//! ```
//!
//! qtype indices are Laya's: `choice 0, score 1, noul 2`. A decision artifact serves
//! `choice` only (the task type); the other two exist so the parity ladder can
//! reproduce every row Laya's oracle recorded.
//!
//! The encoder is core's `aprender::models::modernbert` (D-13); head and scorer reuse
//! its `Linear`, `layer_norm`, `gelu_exact` and `attention` (OPS-03). Every head,
//! scorer and `type_emb` tensor goes through core's own `load_tensor` (the same F16
//! widening and the same refusals as the encoder) and is refused BY NAME when missing,
//! of the wrong shape, undecodable or non-finite ([`LayaError::Load`]).
//!
//! **Marker rule** (`contracts/decide-apr-v1.yaml` `marker_rule`): options longer than
//! `head_max_len` are SHRUNK exactly as Laya does and served. A task is refused only
//! when its built row keeps fewer markers than it has criteria
//! ([`LayaError::MarkersLost`]). Markers precede the state, so [`Laya::from_parts`]
//! checks this ONCE with an empty state: a task that fails it can never be loaded, and
//! it is never a per-request surprise.

pub mod builder;
pub mod head;
pub mod scorer;
pub mod temperature;

use crate::{DecideError, Decision, DecisionMethod, PreparedRow, Task};
use aprender::format::v2::AprV2ReaderRef;
use aprender::models::modernbert::{
    load_tensor, Linear, ModernBertConfig, ModernBertConfigError, ModernBertEncoder,
    ModernBertLoadError,
};
use builder::{Builder, RowPrefix};
use head::{head_geometry, HeadLayer};
use rayon::prelude::*;
use scorer::Scorer;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fmt;

/// Laya's question type (index = Laya's qtype id).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QType {
    /// Pick one of K named criteria (index 0).
    Choice,
    /// An ordinal level (index 1).
    Score,
    /// A yes/no statement (index 2).
    Noul,
}

impl QType {
    /// Laya's qtype index (the `type_emb` row).
    #[must_use]
    pub fn index(self) -> usize {
        match self {
            Self::Choice => 0,
            Self::Score => 1,
            Self::Noul => 2,
        }
    }

    /// Laya's qtype name (`QTYPE_NAMES`).
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Choice => "choice",
            Self::Score => "score",
            Self::Noul => "noul",
        }
    }

    /// The qtype for Laya's index, if any.
    #[must_use]
    pub fn from_index(i: usize) -> Option<Self> {
        match i {
            0 => Some(Self::Choice),
            1 => Some(Self::Score),
            2 => Some(Self::Noul),
            _ => None,
        }
    }
}

fn default_max_len() -> usize {
    512
}

fn default_head_max_len() -> usize {
    192
}

fn default_temperature() -> Vec<f64> {
    vec![1.0, 1.0, 1.0]
}

/// The fields of Laya's `rl_agent_config.json` inference uses. Laya's other fields
/// (`encoder`, `act_costs`, `amp_dtype`, ...) are ignored. Defaults are Laya's own
/// (`agent.py`: `max_len` 512, `head_max_len` 192, temperatures 1.0); `head_layers`
/// is required, as Laya requires it.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct AgentConfig {
    /// Number of decision-head layers.
    pub head_layers: usize,
    /// Row cap in tokens.
    #[serde(default = "default_max_len")]
    pub max_len: usize,
    /// Head + options budget in tokens.
    #[serde(default = "default_head_max_len")]
    pub head_max_len: usize,
    /// Per-qtype fallback temperature (`[choice, score, noul]`).
    #[serde(default = "default_temperature")]
    pub temperature: Vec<f64>,
    /// Calibrated temperature per bucket key (`"choice:3-5"`, ...).
    #[serde(default)]
    pub temperature_by_options: BTreeMap<String, f64>,
}

impl AgentConfig {
    /// Parse `rl_agent_config.json`.
    ///
    /// # Errors
    ///
    /// [`LayaError::AgentConfig`] for malformed JSON, a missing `head_layers`, a
    /// `head_layers` above [`MAX_HEAD_LAYERS`] or a non-positive `max_len`.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, LayaError> {
        let c: Self =
            serde_json::from_slice(bytes).map_err(|e| LayaError::AgentConfig(e.to_string()))?;
        if c.max_len == 0 {
            return Err(LayaError::AgentConfig("max_len must be positive".into()));
        }
        // The count is untrusted (an artifact's own config blob): bounded BEFORE the per-layer
        // tensor names are derived from it.
        if c.head_layers > MAX_HEAD_LAYERS {
            return Err(LayaError::AgentConfig(format!(
                "head_layers {} is over the supported {MAX_HEAD_LAYERS}",
                c.head_layers
            )));
        }
        Ok(c)
    }
}

#[cfg(test)]
thread_local! {
    /// Rows [`Laya::forward_row`] was entered for on this thread. Test-only: a bound that must
    /// bite BEFORE any forward pass (the load-time probe row budget) is proven by this
    /// counter not moving, since the refusal itself is the same before or after a forward.
    pub(crate) static FORWARD_ROWS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The most decision-head layers an agent config may declare. Laya ships 2; the bound keeps
/// the per-layer tensor names derived from an untrusted count from sizing an allocation.
pub const MAX_HEAD_LAYERS: usize = 64;

/// Why the Laya method refused.
#[derive(Debug, Clone, PartialEq)]
pub enum LayaError {
    /// The encoder `config.json` is outside core's supported domain.
    EncoderConfig(ModernBertConfigError),
    /// An encoder, head, scorer or `type_emb` tensor was refused by core's loader
    /// (missing, wrong shape, undecodable or non-finite — named in the error).
    Load(ModernBertLoadError),
    /// `rl_agent_config.json` is malformed.
    AgentConfig(String),
    /// Laya's `nhead = max(1, d / 64)` does not divide the hidden size.
    HeadDoesNotDivide {
        /// Hidden size.
        d: usize,
        /// The head count the rule gives.
        nhead: usize,
    },
    /// The tokenizer bytes or a text were refused by `tokenizers`.
    Tokenizer(String),
    /// `[CLS]`, `[SEP]` or `[MASK]` is not in the tokenizer's vocabulary.
    MissingSpecialToken(String),
    /// The task's built row keeps fewer `[MASK]` markers than the task has criteria
    /// (checked once, at load).
    MarkersLost {
        /// Criteria in the task.
        criteria: usize,
        /// Markers that survived the builder.
        markers: usize,
    },
    /// A row's marker points outside the row.
    MarkerOutOfRange {
        /// The marker position.
        marker: usize,
        /// Row length.
        tokens: usize,
    },
    /// A row's marker count differs from the task's criteria count.
    RowMarkerCount {
        /// Expected (task criteria).
        expected: usize,
        /// Observed in the row.
        observed: usize,
    },
}

impl fmt::Display for LayaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EncoderConfig(e) => write!(f, "laya: encoder config: {e}"),
            Self::Load(e) => write!(f, "laya: {e}"),
            Self::AgentConfig(e) => write!(f, "laya: rl_agent_config.json: {e}"),
            Self::HeadDoesNotDivide { d, nhead } => write!(
                f,
                "laya: head count {nhead} (max(1, d / 64)) does not divide hidden size {d}"
            ),
            Self::Tokenizer(e) => write!(f, "laya: tokenizer: {e}"),
            Self::MissingSpecialToken(t) => {
                write!(f, "laya: tokenizer has no {t} token")
            }
            Self::MarkersLost { criteria, markers } => write!(
                f,
                "laya: the task's row keeps {markers} of {criteria} option markers; \
                 its options do not fit max_len"
            ),
            Self::MarkerOutOfRange { marker, tokens } => {
                write!(f, "laya: marker {marker} is outside a {tokens}-token row")
            }
            Self::RowMarkerCount { expected, observed } => write!(
                f,
                "laya: row has {observed} markers, the task has {expected} criteria"
            ),
        }
    }
}

impl std::error::Error for LayaError {}

/// One non-encoder tensor through core's loader (the encoder's own refusals).
fn tensor(reader: &AprV2ReaderRef<'_>, name: &str, shape: &[usize]) -> Result<Vec<f32>, LayaError> {
    load_tensor(reader, name, shape).map_err(LayaError::Load)
}

/// A LayerNorm's `(weight, bias)` pair under `name`, each `[d]`.
fn load_norm(
    reader: &AprV2ReaderRef<'_>,
    name: &str,
    d: usize,
) -> Result<(Vec<f32>, Vec<f32>), LayaError> {
    Ok((
        tensor(reader, &format!("{name}.weight"), &[d])?,
        tensor(reader, &format!("{name}.bias"), &[d])?,
    ))
}

/// A biased `Linear` `[out, inp]` from explicitly named weight and bias tensors.
fn load_linear_named(
    reader: &AprV2ReaderRef<'_>,
    weight: &str,
    bias: &str,
    out: usize,
    inp: usize,
) -> Result<Linear, LayaError> {
    Ok(Linear {
        w: tensor(reader, weight, &[out, inp])?,
        b: Some(tensor(reader, bias, &[out])?),
        out,
        inp,
    })
}

/// A biased `Linear` stored as `{name}.weight` / `{name}.bias`.
fn load_linear(
    reader: &AprV2ReaderRef<'_>,
    name: &str,
    out: usize,
    inp: usize,
) -> Result<Linear, LayaError> {
    load_linear_named(
        reader,
        &format!("{name}.weight"),
        &format!("{name}.bias"),
        out,
        inp,
    )
}

/// A loaded Laya model bound to one task.
#[derive(Debug)]
pub struct Laya {
    encoder: ModernBertEncoder,
    type_emb: Vec<f32>,
    head: Vec<HeadLayer>,
    scorer: Scorer,
    builder: Builder,
    agent: AgentConfig,
    task: Task,
    /// The task's state-independent row head, built once at load.
    row_prefix: RowPrefix,
    options: Vec<String>,
    temperature: f32,
}

impl Laya {
    /// Build Laya from an APR v2 reader (every tensor under `prefix`, the encoder under
    /// `{prefix}encoder.`), the encoder `config.json`, `rl_agent_config.json`, the
    /// byte-identical `tokenizer.json`, and the task it will answer.
    ///
    /// # Errors
    ///
    /// A [`LayaError`] naming the refused config, tensor or tokenizer, or
    /// [`LayaError::MarkersLost`] when the task's options do not all survive the
    /// builder.
    pub fn from_parts(
        reader: &AprV2ReaderRef<'_>,
        prefix: &str,
        encoder_config_bytes: &[u8],
        agent_config_bytes: &[u8],
        tokenizer_bytes: &[u8],
        task: Task,
    ) -> Result<Self, LayaError> {
        let config = ModernBertConfig::from_json_bytes(encoder_config_bytes)
            .map_err(LayaError::EncoderConfig)?;
        let agent = AgentConfig::from_json_bytes(agent_config_bytes)?;
        let d = config.hidden_size();
        let (nhead, hd) = head_geometry(d)?;
        // Tokenizer and marker rule first: they are cheap and refuse before any weight.
        let builder = Builder::from_bytes(tokenizer_bytes, agent.max_len, agent.head_max_len)?;
        let options = task.render_options();
        let row_prefix = builder.prefix(QType::Choice.name(), task.instructions(), &options)?;
        let probe = builder.finish(&row_prefix, "")?;
        if probe.markers.len() != options.len() {
            return Err(LayaError::MarkersLost {
                criteria: options.len(),
                markers: probe.markers.len(),
            });
        }
        let encoder = ModernBertEncoder::from_apr(reader, &format!("{prefix}encoder."), &config)
            .map_err(LayaError::Load)?;
        let ffn = 4 * d;
        let mut head = Vec::with_capacity(agent.head_layers);
        for i in 0..agent.head_layers {
            let p = format!("{prefix}head.layers.{i}.");
            head.push(HeadLayer {
                norm1: load_norm(reader, &format!("{p}norm1"), d)?,
                in_proj: load_linear_named(
                    reader,
                    &format!("{p}self_attn.in_proj_weight"),
                    &format!("{p}self_attn.in_proj_bias"),
                    3 * d,
                    d,
                )?,
                out_proj: load_linear(reader, &format!("{p}self_attn.out_proj"), d, d)?,
                norm2: load_norm(reader, &format!("{p}norm2"), d)?,
                linear1: load_linear(reader, &format!("{p}linear1"), ffn, d)?,
                linear2: load_linear(reader, &format!("{p}linear2"), d, ffn)?,
                nhead,
                hd,
            });
        }
        let type_emb = tensor(reader, &format!("{prefix}type_emb.weight"), &[3, d])?;
        let scorer = Scorer {
            norm: load_norm(reader, &format!("{prefix}scorer.0"), d)?,
            fc1: load_linear(reader, &format!("{prefix}scorer.1"), d, d)?,
            fc2: load_linear(reader, &format!("{prefix}scorer.3"), 1, d)?,
        };
        let temperature = temperature::temperature_for(&agent, QType::Choice, options.len());
        Ok(Self {
            encoder,
            type_emb,
            head,
            scorer,
            builder,
            agent,
            task,
            row_prefix,
            options,
            temperature,
        })
    }

    /// The request builder (tokenizer + Laya's `build_sequence`).
    #[must_use]
    pub fn builder(&self) -> &Builder {
        &self.builder
    }

    /// The agent config this model was loaded with.
    #[must_use]
    pub fn agent_config(&self) -> &AgentConfig {
        &self.agent
    }

    /// The calibrated temperature applied to the task's `choice` rows.
    #[must_use]
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Scorer logits for one built row: encoder -> `+ type_emb[qtype]` -> head layers
    /// -> hidden states at `markers` -> scorer. `tap(name, block)` sees the encoder's
    /// `emb` / `layer{i}` / `final`, then `head{i}` and `m_opts`.
    ///
    /// # Errors
    ///
    /// [`LayaError::MarkerOutOfRange`] for a marker outside the row, or an encoder /
    /// primitive refusal.
    #[provable_contracts_macros::contract("laya-parity-v1", equation = "logits_abs")]
    pub fn forward_row(
        &self,
        ids: &[u32],
        markers: &[usize],
        qtype: QType,
        mut tap: impl FnMut(&str, &[f32]),
    ) -> Result<Vec<f32>, DecideError> {
        #[cfg(test)]
        FORWARD_ROWS.with(|n| n.set(n.get() + 1));
        let l = ids.len();
        if let Some(&marker) = markers.iter().find(|&&m| m >= l) {
            return Err(LayaError::MarkerOutOfRange { marker, tokens: l }.into());
        }
        let mut x = self.encoder.forward(ids, &mut tap)?;
        let d = self.encoder.config().hidden_size();
        let q = qtype.index();
        let te = &self.type_emb[q * d..(q + 1) * d];
        x.par_chunks_mut(d)
            .for_each(|r| r.iter_mut().zip(te).for_each(|(a, b)| *a += b));
        for (i, layer) in self.head.iter().enumerate() {
            layer.forward(&mut x, l)?;
            tap(&format!("head{i}"), &x);
        }
        let m: Vec<f32> = markers
            .iter()
            .flat_map(|&p| x[p * d..(p + 1) * d].iter().copied())
            .collect();
        tap("m_opts", &m);
        Ok(self.scorer.forward(&m, markers.len())?)
    }

    /// Score one built `choice` row whose marker count must be `k`, at temperature `t`.
    fn score_choice_row(
        &self,
        ids: &[u32],
        markers: &[usize],
        truncated: bool,
        k: usize,
        t: f32,
    ) -> Result<Decision, DecideError> {
        if markers.len() != k {
            return Err(LayaError::RowMarkerCount {
                expected: k,
                observed: markers.len(),
            }
            .into());
        }
        let z = self.forward_row(ids, markers, QType::Choice, |_, _| {})?;
        let probabilities = temperature::softmax_t(&z, t);
        Ok(Decision {
            label_index: argmax(&probabilities),
            probabilities,
            tokens: ids.len(),
            truncated,
        })
    }

    /// Score `texts` against ANOTHER `choice` task with this model's weights, at that
    /// task's own bucket temperature — the task this model was loaded for is untouched.
    ///
    /// This exists for the decide-apr-v1 probes, which run the contract's synthetic
    /// probe task (`probe_policy.probe_task`) rather than the served task, without
    /// loading (and widening) the weights a second time.
    ///
    /// # Errors
    ///
    /// [`LayaError::RowMarkerCount`] when `task`'s built row loses a marker, or a
    /// builder / forward refusal.
    pub fn classify_for_task(
        &self,
        task: &Task,
        texts: &[String],
    ) -> Result<Vec<Decision>, DecideError> {
        let options = task.render_options();
        let t = temperature::temperature_for(&self.agent, QType::Choice, options.len());
        let prefix = self
            .builder
            .prefix(QType::Choice.name(), task.instructions(), &options)?;
        texts
            .iter()
            .map(|text| {
                let row = self.builder.finish(&prefix, text)?;
                self.score_choice_row(&row.ids, &row.markers, row.truncated, options.len(), t)
            })
            .collect()
    }
}

/// Index of the first maximum; NaN never wins (IN-01, plan 08-31). The fold starts
/// at the first element that compares equal to itself (so not NaN): a NaN at index 0
/// can no longer hold the running maximum, and every comparison against a NaN is
/// false, so no later NaN displaces it. Ties keep the earlier index. An empty or
/// all-NaN slice returns 0, as before. Pinned by `laya::tests::argmax_nan_never_wins`.
/// Shared with the verifier, which must re-derive the served decision with the SAME
/// rule.
pub(crate) fn argmax<T: PartialOrd>(p: &[T]) -> usize {
    #[allow(clippy::eq_op)] // `x == x` is the generic NaN test for a PartialOrd element
    let Some(start) = p.iter().position(|x| x == x) else {
        return 0;
    };
    (start + 1..p.len()).fold(start, |m, i| if p[i] > p[m] { i } else { m })
}

impl DecisionMethod for Laya {
    fn task(&self) -> &Task {
        &self.task
    }

    fn prepare(&self, texts: &[String]) -> Result<Vec<PreparedRow>, DecideError> {
        texts
            .iter()
            .map(|t| {
                let row = self.builder.finish(&self.row_prefix, t)?;
                Ok(PreparedRow {
                    ids: row.ids,
                    markers: row.markers,
                    truncated: row.truncated,
                })
            })
            .collect()
    }

    fn classify_prepared(&self, rows: &[PreparedRow]) -> Result<Vec<Decision>, DecideError> {
        rows.iter()
            .map(|r| {
                self.score_choice_row(
                    &r.ids,
                    &r.markers,
                    r.truncated,
                    self.options.len(),
                    self.temperature,
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests;
