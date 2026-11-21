//! GH-7334: GGUF loader surface for `general.architecture = "qwen4exp"`.
//!
//! The safetensors side of `qwen4_exp` lives in [`crate::qwen4_exp`]; this module
//! is the GGUF side and reuses its [`TensorFamily`]. Names, keys and shapes
//! follow llama.cpp b12557 `src/models/qwen4exp.cpp` (`load_arch_hparams`,
//! `load_arch_tensors`).
//!
//! What it does, all on the header (no tensor bytes are read):
//! - reads the `qwen4exp.*` hyperparameters, refusing a missing required key by
//!   name (no silent defaults: PMAT-3477);
//! - maps every GGUF tensor name to a [`TensorFamily`], by layer kind (Gated
//!   DeltaNet vs full attention), PLE layer and MTP layer, and lists the names
//!   it cannot map;
//! - checks the shape contract (LAYOUT-001): expected shapes are written in
//!   ggml order as in llama.cpp and compared, through [`row_major_shape`], with
//!   the row-major dims the parser stores in [`TensorInfo::dims`];
//! - validates a split (`gguf-split`) shard set and locates the n-gram engram
//!   `per_layer_token_embd.weight`;
//! - refuses the vision tower (mmproj / `clip`) and the MTP directory by name;
//! - lists tensors whose quant type has no CPU dequantizer here
//!   (UD-IQ1_S and UD-IQ2_XS: IQ1_S and IQ2_XS).
//!
//! It does not run a forward pass: the MTP head is #7333 and inference refusal
//! is #7331.
//!
//! Wired into a production load (PR #7356 round 3, lane 1 F4): only
//! [`load_qwen4exp_gguf`], which both dense GGUF loaders call through
//! [`dense_loader_refusal`] for a `qwen4exp` file and for an arch-less shard that holds
//! `blk.N.nextn.*` tensors ([`dense_loader_must_refuse`]). [`validate_split_set`],
//! [`refuse_non_text_file`], [`dequant_gaps`], [`unmapped_tensors`] and [`mtp_tensors`]
//! are library surface with tests only: no production split-GGUF loader exists, and no
//! `apr` path calls them yet, so an `MTP/` directory path or an mmproj file is refused
//! today only through the dense loaders' architecture and MTP checks.

use crate::error::{RealizarError, Result};
use crate::gguf::types::{GGUFValue, TensorInfo, GGUF_TYPE_F32};
use crate::gguf::GGUFModel;
pub use crate::qwen4_exp::TensorFamily;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// `general.architecture` of a qwen4exp GGUF.
pub const GGUF_ARCH: &str = "qwen4exp";

/// The engram (n-gram hash embedding) tensor.
pub const ENGRAM_TENSOR: &str = "per_layer_token_embd.weight";

const OP: &str = "load_qwen4exp_gguf";

fn refuse(reason: String) -> RealizarError {
    RealizarError::UnsupportedOperation {
        operation: OP.to_string(),
        reason,
    }
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

fn as_u64(v: &GGUFValue) -> Option<u64> {
    match v {
        GGUFValue::UInt8(x) => Some(u64::from(*x)),
        GGUFValue::UInt16(x) => Some(u64::from(*x)),
        GGUFValue::UInt32(x) => Some(u64::from(*x)),
        GGUFValue::UInt64(x) => Some(*x),
        GGUFValue::Int8(x) => u64::try_from(*x).ok(),
        GGUFValue::Int16(x) => u64::try_from(*x).ok(),
        GGUFValue::Int32(x) => u64::try_from(*x).ok(),
        GGUFValue::Int64(x) => u64::try_from(*x).ok(),
        _ => None,
    }
}

/// The largest scalar a `qwen4exp.*` header key may hold (PR #7356 round 3, lane 1 F1).
///
/// Every scalar key is a dimension, a count or a layer index, and the shape contract
/// multiplies at most three of them (`head_dim * n_head * 2`,
/// `(ngram - 1) * heads_per_ngram * ple_head_dim`), so with each factor at most 2^20 no
/// product or sum the loader forms can overflow `u64`, and `block_count` cannot ask for a
/// per-layer table larger than 2^20 entries. Published checkpoints are far below it
/// (`embedding_length` ~ 10^4, `block_count` ~ 10^2). A value above it is refused by
/// name, so a crafted header gets an error, never an overflow panic or an abort on a
/// huge allocation. `ple.eos_token_id` is a token id, not a dimension, and is exempt.
pub const MAX_HEADER_VALUE: u64 = 1 << 20;

struct Md<'a>(&'a HashMap<String, GGUFValue>);

impl Md<'_> {
    fn key(k: &str) -> String {
        format!("{GGUF_ARCH}.{k}")
    }

    fn opt_u64(&self, k: &str) -> Result<Option<u64>> {
        let v = self.opt_u64_unbounded(k)?;
        match v {
            Some(v) if v > MAX_HEADER_VALUE => Err(refuse(format!(
                "metadata key {} is {v}, above the {MAX_HEADER_VALUE} bound for a header value",
                Self::key(k)
            ))),
            v => Ok(v),
        }
    }

    fn opt_u64_unbounded(&self, k: &str) -> Result<Option<u64>> {
        let key = Self::key(k);
        match self.0.get(&key) {
            None => Ok(None),
            Some(v) => as_u64(v)
                .map(Some)
                .ok_or_else(|| refuse(format!("metadata key {key} is not an unsigned integer"))),
        }
    }

    fn req(&self, k: &str) -> Result<u64> {
        self.opt_u64(k)?
            .ok_or_else(|| refuse(format!("missing required metadata key {}", Self::key(k))))
    }

    fn nonzero(&self, k: &str) -> Result<u64> {
        let v = self.req(k)?;
        if v == 0 {
            return Err(refuse(format!(
                "metadata key {} must be non-zero",
                Self::key(k)
            )));
        }
        Ok(v)
    }

    /// An integer array, or a scalar read as a one-element array.
    fn u64s(&self, k: &str) -> Result<Option<Vec<u64>>> {
        let key = Self::key(k);
        match self.0.get(&key) {
            None => Ok(None),
            Some(GGUFValue::Array(xs)) => xs
                .iter()
                .map(|x| match x {
                    GGUFValue::Bool(b) => Some(u64::from(*b)),
                    other => as_u64(other),
                })
                .collect::<Option<Vec<_>>>()
                .map(Some)
                .ok_or_else(|| refuse(format!("metadata key {key} is not an integer array"))),
            Some(v) => as_u64(v)
                .map(|x| Some(vec![x]))
                .ok_or_else(|| refuse(format!("metadata key {key} is not an integer"))),
        }
    }

    /// `ple.layer_multipliers` are the n-gram hash multipliers, positive 64-bit integers
    /// (`qwen4_exp_embed::layer_multipliers`). A bool, float or string element, or a
    /// value <= 0, is refused (PR #7356 lane-1 finding 3). Oddness is not checked here.
    fn hash_multipliers(&self, k: &str) -> Result<()> {
        let key = Self::key(k);
        let ok = match self.0.get(&key) {
            Some(GGUFValue::Array(xs)) => xs
                .iter()
                .all(|x| !matches!(x, GGUFValue::Bool(_)) && as_u64(x).is_some_and(|v| v > 0)),
            _ => false,
        };
        if ok {
            Ok(())
        } else {
            Err(refuse(format!(
                "metadata key {key} must be an array of positive 64-bit integers"
            )))
        }
    }

    fn array_len(&self, k: &str) -> Option<usize> {
        match self.0.get(&Self::key(k)) {
            Some(GGUFValue::Array(xs)) => Some(xs.len()),
            Some(_) => Some(1),
            None => None,
        }
    }
}

/// The hyperparameters of a qwen4exp GGUF that the loader and shape contract use.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Qwen4ExpGgufMeta {
    /// `qwen4exp.block_count`: trunk layers plus MTP layers.
    pub block_count: u64,
    /// `qwen4exp.nextn_predict_layers` (0 when absent, as llama.cpp reads it).
    pub nextn_predict_layers: u64,
    /// Trunk layers: `block_count - nextn_predict_layers`.
    pub n_layer: u64,
    /// `qwen4exp.embedding_length`.
    pub n_embd: u64,
    /// Rows of `token_embd.weight`.
    pub n_vocab: u64,
    /// `qwen4exp.attention.head_count`.
    pub n_head: u64,
    /// `qwen4exp.attention.head_count_kv`.
    pub n_head_kv: u64,
    /// `qwen4exp.attention.key_length`.
    pub head_dim: u64,
    /// `qwen4exp.attention.value_length`.
    pub value_dim: u64,
    /// `qwen4exp.expert_count`.
    pub n_expert: u64,
    /// `qwen4exp.expert_feed_forward_length` (one value for every layer).
    pub n_ff_exp: u64,
    /// `qwen4exp.expert_shared_feed_forward_length`, else `feed_forward_length`.
    pub n_ff_shexp: u64,
    /// `qwen4exp.hyper_connection.count` (> 1).
    pub hc_count: u64,
    /// `qwen4exp.hyper_connection.low_rank`.
    pub hc_low_rank: u64,
    /// `qwen4exp.ssm.conv_kernel`.
    pub ssm_d_conv: u64,
    /// `qwen4exp.ssm.state_size`: head_k_dim = head_v_dim.
    pub ssm_d_state: u64,
    /// `qwen4exp.ssm.time_step_rank`: linear-attention value heads.
    pub ssm_dt_rank: u64,
    /// `qwen4exp.ssm.group_count`: linear-attention key heads.
    pub ssm_n_group: u64,
    /// `qwen4exp.attention.indexer.head_count`.
    pub indexer_n_head: u64,
    /// `qwen4exp.attention.indexer.key_length`.
    pub indexer_head_dim: u64,
    /// Per layer: Gated DeltaNet (`true`) or full attention (`false`).
    pub recurrent: Vec<bool>,
    /// `qwen4exp.ple.layers` (empty when the PLE group is absent).
    pub ple_layers: Vec<u64>,
    /// `(ple.ngram_size - 1) * ple.heads_per_ngram`.
    pub ple_n_heads: u64,
    /// `qwen4exp.embedding_length_per_layer_input`: engram row width.
    pub ple_head_dim: u64,
    /// `qwen4exp.ple.conv_kernel`.
    pub ple_conv_kernel: u64,
    /// `max(head_offset + head_vocab_size)`: fewest engram rows the gather needs.
    pub ple_min_rows: u64,
}

impl Qwen4ExpGgufMeta {
    /// Read the hyperparameters of a single-file qwen4exp GGUF.
    ///
    /// # Errors
    /// The file is not qwen4exp, or a required key is missing, zero or malformed;
    /// the error names the key.
    pub fn from_model(model: &GGUFModel) -> Result<Self> {
        let tensors: Vec<&TensorInfo> = model.tensors.iter().collect();
        Self::from_parts(&model.metadata, &tensors)
    }

    fn from_parts(md: &HashMap<String, GGUFValue>, tensors: &[&TensorInfo]) -> Result<Self> {
        let arch = match md.get(crate::gguf::keys::GENERAL_ARCHITECTURE) {
            Some(GGUFValue::String(s)) => s.as_str(),
            _ => "",
        };
        if arch != GGUF_ARCH {
            return Err(refuse(format!(
                "general.architecture is {arch:?}, expected {GGUF_ARCH:?}"
            )));
        }
        let md = Md(md);
        let mut m = Self::default();
        m.read_trunk(&md)?;
        m.read_moe(&md)?;
        m.read_ssm_hc_indexer(&md)?;
        m.read_layer_kinds(&md)?;
        m.read_ple(&md)?;
        let embd = tensors
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .ok_or_else(|| refuse("missing required tensor token_embd.weight".to_string()))?;
        // Row-major [n_vocab, n_embd]; a 0-row or non-2D embedding has no vocabulary
        // (PR #7356 lane-1 finding 4).
        m.n_vocab = match embd.dims.as_slice() {
            [v, _] if *v > 0 => *v,
            dims => {
                return Err(refuse(format!(
                    "tensor token_embd.weight has row-major shape {dims:?}; it needs \
                     [n_vocab > 0, n_embd]"
                )))
            },
        };
        Ok(m)
    }

    fn read_trunk(&mut self, md: &Md<'_>) -> Result<()> {
        self.block_count = md.nonzero("block_count")?;
        self.nextn_predict_layers = md.opt_u64("nextn_predict_layers")?.unwrap_or(0);
        if self.nextn_predict_layers >= self.block_count {
            return Err(refuse(format!(
                "{GGUF_ARCH}.nextn_predict_layers {} leaves no trunk layer of block_count {}",
                self.nextn_predict_layers, self.block_count
            )));
        }
        self.n_layer = self.block_count - self.nextn_predict_layers;
        self.n_embd = md.nonzero("embedding_length")?;
        self.n_head = md.nonzero("attention.head_count")?;
        self.n_head_kv = md.nonzero("attention.head_count_kv")?;
        self.head_dim = md.nonzero("attention.key_length")?;
        self.value_dim = md.nonzero("attention.value_length")?;
        Ok(())
    }

    fn read_moe(&mut self, md: &Md<'_>) -> Result<()> {
        self.n_expert = md.nonzero("expert_count")?;
        md.nonzero("expert_used_count")?;
        let key = "expert_feed_forward_length";
        let ff = md
            .u64s(key)?
            .ok_or_else(|| refuse(format!("missing required metadata key {}", Md::key(key))))?;
        self.n_ff_exp = match ff.as_slice() {
            [first, rest @ ..]
                if *first != 0 && *first <= MAX_HEADER_VALUE && rest.iter().all(|x| x == first) =>
            {
                *first
            },
            _ => {
                return Err(refuse(format!(
                    "{} must be one non-zero value (at most {MAX_HEADER_VALUE}) for every layer, got {ff:?}",
                    Md::key(key)
                )))
            },
        };
        self.n_ff_shexp = match md.opt_u64("expert_shared_feed_forward_length")? {
            Some(v) if v != 0 => v,
            _ => md.nonzero("feed_forward_length")?,
        };
        Ok(())
    }

    fn read_ssm_hc_indexer(&mut self, md: &Md<'_>) -> Result<()> {
        self.ssm_d_conv = md.nonzero("ssm.conv_kernel")?;
        md.nonzero("ssm.inner_size")?;
        self.ssm_d_state = md.nonzero("ssm.state_size")?;
        self.ssm_dt_rank = md.nonzero("ssm.time_step_rank")?;
        self.ssm_n_group = md.nonzero("ssm.group_count")?;
        self.hc_count = md.req("hyper_connection.count")?;
        if self.hc_count <= 1 {
            return Err(refuse(format!(
                "{} must be greater than one, got {}",
                Md::key("hyper_connection.count"),
                self.hc_count
            )));
        }
        self.hc_low_rank = md.nonzero("hyper_connection.low_rank")?;
        self.indexer_n_head = md.nonzero("attention.indexer.head_count")?;
        self.indexer_head_dim = md.nonzero("attention.indexer.key_length")?;
        md.nonzero("attention.indexer.top_k")?;
        Ok(())
    }

    /// `attention.recurrent_layers`, else `full_attention_interval` (4 when
    /// absent, as llama.cpp reads it): layer `i` is linear iff
    /// `i < n_layer && (i + 1) % interval != 0`.
    fn read_layer_kinds(&mut self, md: &Md<'_>) -> Result<()> {
        if let Some(flags) = md.u64s("attention.recurrent_layers")? {
            if flags.len() as u64 != self.block_count {
                return Err(refuse(format!(
                    "{} has {} entries, block_count is {}",
                    Md::key("attention.recurrent_layers"),
                    flags.len(),
                    self.block_count
                )));
            }
            self.recurrent = flags.iter().map(|f| *f != 0).collect();
            return Ok(());
        }
        let interval = md.opt_u64("full_attention_interval")?.unwrap_or(4);
        if interval == 0 {
            return Err(refuse(format!(
                "metadata key {} must be non-zero",
                Md::key("full_attention_interval")
            )));
        }
        self.recurrent = (0..self.block_count)
            .map(|i| i < self.n_layer && (i + 1) % interval != 0)
            .collect();
        Ok(())
    }

    /// The PLE group is optional; when `ple.layers` is present every key of it
    /// is required (llama.cpp `load_arch_hparams`).
    fn read_ple(&mut self, md: &Md<'_>) -> Result<()> {
        let Some(layers) = md.u64s("ple.layers")? else {
            return Ok(());
        };
        if layers.len() != 1 {
            return Err(refuse(format!(
                "{} lists {} layers, but only one PLE layer is supported",
                Md::key("ple.layers"),
                layers.len()
            )));
        }
        for &il in &layers {
            if !self.is_recurrent(il) {
                return Err(refuse(format!(
                    "PLE layer {il} is not a linear attention layer ({})",
                    Md::key("ple.layers")
                )));
            }
        }
        self.ple_layers = layers;
        let ngram = md.req("ple.ngram_size")?;
        if !(2..=16).contains(&ngram) {
            return Err(refuse(format!("PLE n-gram size {ngram} is out of range")));
        }
        let per = md.nonzero("ple.heads_per_ngram")?;
        self.ple_conv_kernel = md.nonzero("ple.conv_kernel")?;
        md.opt_u64_unbounded("ple.eos_token_id")?.ok_or_else(|| {
            refuse(format!(
                "missing required metadata key {}",
                Md::key("ple.eos_token_id")
            ))
        })?;
        self.ple_head_dim = md.nonzero("embedding_length_per_layer_input")?;
        self.ple_n_heads = (ngram - 1) * per;
        if md.array_len("ple.layer_multipliers").unwrap_or(0) < ngram as usize {
            return Err(refuse(format!(
                "{} needs at least {ngram} entries",
                Md::key("ple.layer_multipliers")
            )));
        }
        md.hash_multipliers("ple.layer_multipliers")?;
        self.ple_min_rows = self.read_ple_heads(md)?;
        Ok(())
    }

    fn read_ple_heads(&self, md: &Md<'_>) -> Result<u64> {
        let heads = |k: &str| -> Result<Vec<u64>> {
            let xs = md.u64s(k)?.unwrap_or_default();
            if xs.len() as u64 != self.ple_n_heads {
                return Err(refuse(format!(
                    "{} has {} entries, PLE head count is {}",
                    Md::key(k),
                    xs.len(),
                    self.ple_n_heads
                )));
            }
            Ok(xs)
        };
        let offsets = heads("ple.head_offsets")?;
        let sizes = heads("ple.head_vocab_sizes")?;
        let mut rows = 0u64;
        for (h, (o, s)) in offsets.iter().zip(&sizes).enumerate() {
            let end = o
                .checked_add(*s)
                .filter(|e| *s != 0 && i32::try_from(*e).is_ok());
            let end = end.ok_or_else(|| {
                refuse(format!(
                    "PLE head {h} range does not fit the int32 row index"
                ))
            })?;
            rows = rows.max(end);
        }
        Ok(rows)
    }

    /// Layer `il` is a Gated DeltaNet (linear attention) layer.
    #[must_use]
    pub fn is_recurrent(&self, il: u64) -> bool {
        usize::try_from(il)
            .ok()
            .and_then(|i| self.recurrent.get(i))
            .copied()
            .unwrap_or(false)
    }

    /// Layer `il` is an MTP (nextn) layer: `n_layer <= il < block_count`.
    #[must_use]
    pub fn is_mtp(&self, il: u64) -> bool {
        il >= self.n_layer && il < self.block_count
    }

    fn is_ple(&self, il: u64) -> bool {
        self.ple_layers.contains(&il)
    }
}

/// The file's `general.architecture` is `qwen4exp`.
#[must_use]
pub fn is_qwen4exp_gguf(model: &GGUFModel) -> bool {
    model.architecture() == Some(GGUF_ARCH)
}

// ---------------------------------------------------------------------------
// Tensor-name mapping
// ---------------------------------------------------------------------------

const GLOBAL: [(&str, TensorFamily); 7] = [
    ("token_embd.weight", TensorFamily::Embedding),
    ("output.weight", TensorFamily::Embedding),
    ("output_hc_norm.weight", TensorFamily::HyperConnection),
    ("output_hc_down.weight", TensorFamily::HyperConnection),
    ("output_hc_up.weight", TensorFamily::HyperConnection),
    (ENGRAM_TENSOR, TensorFamily::NgramEmbedding),
    ("per_layer_token_embd.scale", TensorFamily::NgramEmbedding),
];

const HC: [&str; 10] = [
    "hc_attn_norm.weight",
    "hc_attn_up.weight",
    "hc_attn_down_inject.weight",
    "hc_attn_down.weight",
    "hc_attn_inject.weight",
    "hc_ffn_norm.weight",
    "hc_ffn_up.weight",
    "hc_ffn_down_inject.weight",
    "hc_ffn_down.weight",
    "hc_ffn_inject.weight",
];

const MOE: [&str; 9] = [
    "ffn_gate_inp.weight",
    "ffn_down_exps.weight",
    "ffn_gate_up_exps.weight",
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_gate_inp_shexp.weight",
    "ffn_gate_shexp.weight",
    "ffn_up_shexp.weight",
    "ffn_down_shexp.weight",
];

const LINEAR: [&str; 11] = [
    "attn_qkv_gate.weight",
    "attn_qkv.weight",
    "attn_gate.weight",
    "ssm_conv1d.weight",
    "ssm_dt.bias",
    "ssm_a",
    "ssm_beta_alpha.weight",
    "ssm_beta.weight",
    "ssm_alpha.weight",
    "ssm_norm.weight",
    "ssm_out.weight",
];

const FULL: [&str; 11] = [
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_qkv.weight",
    "attn_qkv.bias",
    "attn_q.bias",
    "attn_k.bias",
    "attn_v.bias",
    "attn_output.weight",
    "attn_q_norm.weight",
    "attn_k_norm.weight",
];

const INDEXER: [&str; 4] = [
    "indexer.q_proj.weight",
    "indexer.k_proj.weight",
    "indexer.q_norm.weight",
    "indexer.k_norm.weight",
];

const PLE: [&str; 6] = [
    "ple_key.weight",
    "ple_value.weight",
    "ple_norm_key.weight",
    "ple_norm_query.weight",
    "ple_norm_conv.weight",
    "ple_conv1d.weight",
];

const NEXTN: [&str; 6] = [
    "nextn.eh_proj.weight",
    "nextn.enorm.weight",
    "nextn.hnorm.weight",
    "nextn.hc_head_norm.weight",
    "nextn.hc_head_down.weight",
    "nextn.hc_head_up.weight",
];

/// `blk.{il}.{rest}` → `(il, rest)`; `il` must be all digits.
fn split_block(name: &str) -> Option<(u64, &str)> {
    let tail = name.strip_prefix("blk.")?;
    let (idx, rest) = tail.split_once('.')?;
    if idx.is_empty() || !idx.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some((idx.parse().ok()?, rest))
}

/// The family of a per-block tensor `rest` of layer `il`, by layer kind.
fn block_family(rest: &str, il: u64, meta: &Qwen4ExpGgufMeta) -> Option<TensorFamily> {
    if HC.contains(&rest) {
        return Some(TensorFamily::HyperConnection);
    }
    if MOE.contains(&rest) {
        return Some(TensorFamily::Moe);
    }
    if meta.is_ple(il) && PLE.contains(&rest) {
        return Some(TensorFamily::PerLayerEmbedding);
    }
    if meta.is_recurrent(il) {
        return LINEAR
            .contains(&rest)
            .then_some(TensorFamily::LinearAttention);
    }
    if FULL.contains(&rest) {
        return Some(TensorFamily::FullAttention);
    }
    INDEXER.contains(&rest).then_some(TensorFamily::Indexer)
}

/// Map a qwen4exp GGUF tensor name to its family; `None` when the name is not
/// part of the architecture at that layer (wrong layer kind, out of range,
/// another arch's spelling, a vision tensor).
#[must_use]
pub fn map_gguf_tensor(name: &str, meta: &Qwen4ExpGgufMeta) -> Option<TensorFamily> {
    if let Some((_, fam)) = GLOBAL.iter().find(|(n, _)| *n == name) {
        return Some(*fam);
    }
    let (il, rest) = split_block(name)?;
    if il >= meta.block_count {
        return None;
    }
    if meta.is_mtp(il) {
        let ok = NEXTN.contains(&rest) || block_family(rest, il, meta).is_some();
        return ok.then_some(TensorFamily::Mtp);
    }
    block_family(rest, il, meta)
}

/// Names in `model` that [`map_gguf_tensor`] cannot map, in file order.
#[must_use]
pub fn unmapped_tensors(model: &GGUFModel, meta: &Qwen4ExpGgufMeta) -> Vec<String> {
    model
        .tensors
        .iter()
        .filter(|t| map_gguf_tensor(&t.name, meta).is_none())
        .map(|t| t.name.clone())
        .collect()
}

/// A `blk.N.nextn.*` name, at any layer: an MTP tensor whatever the metadata says.
fn is_nextn_name(name: &str) -> bool {
    split_block(name).is_some_and(|(_, rest)| rest.starts_with("nextn."))
}

/// The MTP tensors among `names`: every tensor of layers `n_layer..block_count`, plus every
/// `blk.N.nextn.*` tensor even when `nextn_predict_layers` is 0 or absent (PR #7356 lane-1
/// finding 2). The MTP head itself is #7333.
fn mtp_names<'a>(names: impl Iterator<Item = &'a str>, meta: &Qwen4ExpGgufMeta) -> Vec<String> {
    names
        .filter(|n| map_gguf_tensor(n, meta) == Some(TensorFamily::Mtp) || is_nextn_name(n))
        .map(str::to_string)
        .collect()
}

/// The MTP tensors of `model` (see [`mtp_names`]), in file order.
#[must_use]
pub fn mtp_tensors(model: &GGUFModel, meta: &Qwen4ExpGgufMeta) -> Vec<String> {
    mtp_names(model.tensors.iter().map(|t| t.name.as_str()), meta)
}

/// The MTP refusal for a tensor set whose metadata may not parse (a lone shard, a broken
/// header): named `blk.N.nextn.*` tensors are refused first, else the metadata error.
fn meta_or_mtp_refusal(names: &[&str], meta: Result<Qwen4ExpGgufMeta>) -> Result<Qwen4ExpGgufMeta> {
    let meta = match meta {
        Ok(m) => m,
        Err(e) => {
            let nextn: Vec<&str> = names.iter().copied().filter(|n| is_nextn_name(n)).collect();
            crate::qwen4_exp::mtp_refusal(&nextn).map_err(refuse)?;
            return Err(e);
        },
    };
    crate::qwen4_exp::mtp_refusal(&mtp_names(names.iter().copied(), &meta)).map_err(refuse)?;
    Ok(meta)
}

// ---------------------------------------------------------------------------
// Shape contract (LAYOUT-001)
// ---------------------------------------------------------------------------

/// ggml dims (`ne0` first, as on disk and in llama.cpp) → row-major shape.
#[must_use]
pub fn row_major_shape(ggml_dims: &[u64]) -> Vec<usize> {
    ggml_dims.iter().rev().map(|d| *d as usize).collect()
}

/// How one expected tensor is matched.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Need {
    /// Must be present with exactly this shape.
    Required,
    /// May be absent; when present the shape must match.
    Optional,
    /// Must be present; element count must match (llama.cpp `TENSOR_ALLOW_RESHAPE`).
    Reshape,
}

struct Shapes<'a> {
    by_name: HashMap<&'a str, &'a TensorInfo>,
}

impl Shapes<'_> {
    fn get(&self, name: &str) -> Option<&TensorInfo> {
        self.by_name.get(name).copied()
    }

    fn check(&self, name: &str, ggml: &[u64], need: Need) -> Result<()> {
        let Some(t) = self.get(name) else {
            if need == Need::Optional {
                return Ok(());
            }
            return Err(refuse(format!("missing required tensor {name}")));
        };
        let want = row_major_shape(ggml);
        let got: Vec<usize> = t.dims.iter().map(|d| *d as usize).collect();
        let ok = if need == Need::Reshape {
            // Checked: a crafted file's dims must not overflow the count (round 3, F1).
            let count = |d: &[usize]| d.iter().try_fold(1usize, |a, x| a.checked_mul(*x));
            count(&got).is_some() && count(&got) == count(&want)
        } else {
            got == want
        };
        if ok {
            return Ok(());
        }
        Err(refuse(format!(
            "tensor {name} has row-major shape {got:?}, the contract wants {want:?} (LAYOUT-001)"
        )))
    }

    /// Either the merged `[a | b]` tensor, or both halves.
    fn merged(&self, merged: &str, a: &str, b: &str, n_in: u64, n_a: u64, n_b: u64) -> Result<()> {
        if self.get(merged).is_some() {
            return self.check(merged, &[n_in, n_a + n_b], Need::Required);
        }
        self.check(a, &[n_in, n_a], Need::Required)?;
        self.check(b, &[n_in, n_b], Need::Required)
    }
}

fn blk(il: u64, rest: &str) -> String {
    format!("blk.{il}.{rest}")
}

fn check_globals(s: &Shapes<'_>, m: &Qwen4ExpGgufMeta) -> Result<()> {
    let (e, hc, lr) = (m.n_embd, m.hc_count, m.hc_low_rank);
    s.check("token_embd.weight", &[e, m.n_vocab], Need::Required)?;
    s.check("output.weight", &[e, m.n_vocab], Need::Optional)?;
    s.check("output_hc_norm.weight", &[e, hc], Need::Reshape)?;
    s.check("output_hc_down.weight", &[hc * e, lr], Need::Required)?;
    s.check("output_hc_up.weight", &[lr, hc * e], Need::Required)?;
    if m.ple_n_heads == 0 {
        return Ok(());
    }
    let t = s
        .get(ENGRAM_TENSOR)
        .ok_or_else(|| refuse(format!("missing required tensor {ENGRAM_TENSOR}")))?;
    // Row-major [rows, head_dim]; rows may be padded above the head ranges.
    let ok =
        matches!(t.dims.as_slice(), [rows, w] if *w == m.ple_head_dim && *rows >= m.ple_min_rows);
    if !ok {
        return Err(refuse(format!(
            "tensor {ENGRAM_TENSOR} has row-major shape {:?}, the contract wants [>= {}, {}] (LAYOUT-001)",
            t.dims, m.ple_min_rows, m.ple_head_dim
        )));
    }
    if let Some(scale) = s.get("per_layer_token_embd.scale") {
        if scale.qtype != GGUF_TYPE_F32 {
            return Err(refuse("per_layer_token_embd.scale must be F32".to_string()));
        }
    }
    Ok(())
}

fn check_hc(s: &Shapes<'_>, il: u64, m: &Qwen4ExpGgufMeta) -> Result<()> {
    let (e, hc, lr) = (m.n_embd, m.hc_count, m.hc_low_rank);
    let hd = hc * e;
    for part in ["attn", "ffn"] {
        s.check(
            &blk(il, &format!("hc_{part}_norm.weight")),
            &[e, hc],
            Need::Reshape,
        )?;
        s.check(
            &blk(il, &format!("hc_{part}_up.weight")),
            &[lr, hd],
            Need::Required,
        )?;
        s.merged(
            &blk(il, &format!("hc_{part}_down_inject.weight")),
            &blk(il, &format!("hc_{part}_down.weight")),
            &blk(il, &format!("hc_{part}_inject.weight")),
            hd,
            lr,
            hc,
        )?;
    }
    Ok(())
}

fn check_full_attention(s: &Shapes<'_>, il: u64, m: &Qwen4ExpGgufMeta) -> Result<()> {
    let e = m.n_embd;
    let q = m.head_dim * m.n_head * 2; // [q | gate] per head
    let k = m.head_dim * m.n_head_kv;
    let v = m.value_dim * m.n_head_kv;
    if s.get(&blk(il, "attn_qkv.weight")).is_some() {
        s.check(&blk(il, "attn_qkv.weight"), &[e, q + k + v], Need::Required)?;
    } else {
        s.check(&blk(il, "attn_q.weight"), &[e, q], Need::Required)?;
        s.check(&blk(il, "attn_k.weight"), &[e, k], Need::Required)?;
        s.check(&blk(il, "attn_v.weight"), &[e, v], Need::Required)?;
    }
    s.check(
        &blk(il, "attn_output.weight"),
        // Each head emits value_dim values (attn_v is value_dim * n_head_kv wide).
        &[m.value_dim * m.n_head, e],
        Need::Required,
    )?;
    s.check(
        &blk(il, "attn_q_norm.weight"),
        &[m.head_dim],
        Need::Required,
    )?;
    s.check(
        &blk(il, "attn_k_norm.weight"),
        &[m.head_dim],
        Need::Required,
    )?;
    let idx = m.indexer_head_dim;
    s.check(
        &blk(il, "indexer.q_proj.weight"),
        &[e, m.indexer_n_head * idx],
        Need::Required,
    )?;
    s.check(&blk(il, "indexer.k_proj.weight"), &[e, idx], Need::Required)?;
    s.check(&blk(il, "indexer.q_norm.weight"), &[idx], Need::Required)?;
    s.check(&blk(il, "indexer.k_norm.weight"), &[idx], Need::Required)
}

fn check_linear_attention(s: &Shapes<'_>, il: u64, m: &Qwen4ExpGgufMeta) -> Result<()> {
    let e = m.n_embd;
    let key_dim = m.ssm_d_state * m.ssm_n_group;
    let value_dim = m.ssm_d_state * m.ssm_dt_rank;
    let conv_dim = key_dim * 2 + value_dim;
    s.merged(
        &blk(il, "attn_qkv_gate.weight"),
        &blk(il, "attn_qkv.weight"),
        &blk(il, "attn_gate.weight"),
        e,
        conv_dim,
        value_dim,
    )?;
    s.check(
        &blk(il, "ssm_conv1d.weight"),
        &[m.ssm_d_conv, conv_dim],
        Need::Required,
    )?;
    s.check(&blk(il, "ssm_dt.bias"), &[m.ssm_dt_rank], Need::Required)?;
    s.check(&blk(il, "ssm_a"), &[m.ssm_dt_rank], Need::Required)?;
    s.merged(
        &blk(il, "ssm_beta_alpha.weight"),
        &blk(il, "ssm_beta.weight"),
        &blk(il, "ssm_alpha.weight"),
        e,
        m.ssm_dt_rank,
        m.ssm_dt_rank,
    )?;
    s.check(
        &blk(il, "ssm_norm.weight"),
        &[m.ssm_d_state],
        Need::Required,
    )?;
    s.check(&blk(il, "ssm_out.weight"), &[value_dim, e], Need::Required)
}

fn check_ple(s: &Shapes<'_>, il: u64, m: &Qwen4ExpGgufMeta) -> Result<()> {
    let (e, hc) = (m.n_embd, m.hc_count);
    // The projections read the gathered engram rows: ple_n_heads rows of ple_head_dim each
    // (`build_inp_ple`, qwen4_exp_embed.rs `key_proj [C*H, embed_dim]`, `value_proj
    // [H, embed_dim]`). llama.cpp b12557 declares the input as n_embd (qwen4exp.cpp:292-293),
    // which only equals ple_embd on the published config; the width is ple_embd.
    let pe = m.ple_n_heads * m.ple_head_dim;
    s.check(&blk(il, "ple_key.weight"), &[pe, hc * e], Need::Required)?;
    s.check(&blk(il, "ple_value.weight"), &[pe, e], Need::Required)?;
    for n in [
        "ple_norm_key.weight",
        "ple_norm_query.weight",
        "ple_norm_conv.weight",
    ] {
        s.check(&blk(il, n), &[e, hc], Need::Reshape)?;
    }
    s.check(
        &blk(il, "ple_conv1d.weight"),
        &[m.ple_conv_kernel, hc * e],
        Need::Required,
    )
}

fn check_moe(s: &Shapes<'_>, il: u64, m: &Qwen4ExpGgufMeta) -> Result<()> {
    let (e, ff, x, sh) = (m.n_embd, m.n_ff_exp, m.n_expert, m.n_ff_shexp);
    s.check(&blk(il, "ffn_gate_inp.weight"), &[e, x], Need::Required)?;
    s.check(
        &blk(il, "ffn_down_exps.weight"),
        &[ff, e, x],
        Need::Required,
    )?;
    if s.get(&blk(il, "ffn_gate_up_exps.weight")).is_some() {
        s.check(
            &blk(il, "ffn_gate_up_exps.weight"),
            &[e, 2 * ff, x],
            Need::Required,
        )?;
    } else {
        s.check(
            &blk(il, "ffn_gate_exps.weight"),
            &[e, ff, x],
            Need::Required,
        )?;
        s.check(&blk(il, "ffn_up_exps.weight"), &[e, ff, x], Need::Required)?;
    }
    s.check(&blk(il, "ffn_gate_inp_shexp.weight"), &[e], Need::Required)?;
    s.check(&blk(il, "ffn_gate_shexp.weight"), &[e, sh], Need::Required)?;
    s.check(&blk(il, "ffn_up_shexp.weight"), &[e, sh], Need::Required)?;
    s.check(&blk(il, "ffn_down_shexp.weight"), &[sh, e], Need::Required)
}

fn check_nextn(s: &Shapes<'_>, il: u64, m: &Qwen4ExpGgufMeta) -> Result<()> {
    let (e, hc, lr) = (m.n_embd, m.hc_count, m.hc_low_rank);
    s.check(
        &blk(il, "nextn.eh_proj.weight"),
        &[2 * e, e],
        Need::Required,
    )?;
    s.check(&blk(il, "nextn.enorm.weight"), &[e], Need::Required)?;
    s.check(&blk(il, "nextn.hnorm.weight"), &[e, hc], Need::Reshape)?;
    s.check(
        &blk(il, "nextn.hc_head_norm.weight"),
        &[e, hc],
        Need::Reshape,
    )?;
    s.check(
        &blk(il, "nextn.hc_head_down.weight"),
        &[hc * e, lr],
        Need::Required,
    )?;
    s.check(
        &blk(il, "nextn.hc_head_up.weight"),
        &[lr, hc * e],
        Need::Required,
    )
}

fn check_block(s: &Shapes<'_>, il: u64, m: &Qwen4ExpGgufMeta) -> Result<()> {
    check_hc(s, il, m)?;
    if m.is_recurrent(il) {
        check_linear_attention(s, il, m)?;
    } else {
        check_full_attention(s, il, m)?;
    }
    if m.is_ple(il) {
        check_ple(s, il, m)?;
    }
    check_moe(s, il, m)?;
    if m.is_mtp(il) {
        check_nextn(s, il, m)?;
    }
    Ok(())
}

fn check_shapes_of(tensors: &[&TensorInfo], meta: &Qwen4ExpGgufMeta) -> Result<()> {
    let s = Shapes {
        by_name: tensors.iter().map(|t| (t.name.as_str(), *t)).collect(),
    };
    check_globals(&s, meta)?;
    for il in 0..meta.block_count {
        check_block(&s, il, meta)?;
    }
    Ok(())
}

/// Check every required tensor is present with its contract shape.
///
/// # Errors
/// A required tensor is missing, or a tensor's row-major shape is not the one
/// the ggml-order contract implies; the error names the tensor.
pub fn check_shapes(model: &GGUFModel, meta: &Qwen4ExpGgufMeta) -> Result<()> {
    let tensors: Vec<&TensorInfo> = model.tensors.iter().collect();
    check_shapes_of(&tensors, meta)
}

// ---------------------------------------------------------------------------
// Refusals: the vision tower (mmproj) and the MTP directory
// ---------------------------------------------------------------------------

fn is_vision_tensor(name: &str) -> bool {
    name.starts_with("v.") || name.starts_with("mm.")
}

/// Refuse a file that is not the text model: the mmproj vision tower (by file
/// name, a `clip` architecture, or any `v.*` / `mm.*` tensor) and anything
/// under an `MTP` directory.
///
/// # Errors
/// The file is the vision tower or the MTP directory; the error names which.
pub fn refuse_non_text_file(path: &Path, model: &GGUFModel) -> Result<()> {
    let file = path.file_name().and_then(|f| f.to_str()).unwrap_or("");
    if file.to_ascii_lowercase().contains("mmproj") || model.architecture() == Some("clip") {
        return Err(refuse(format!(
            "{} is the qwen4exp vision tower (mmproj); only the text model loads",
            path.display()
        )));
    }
    if let Some(t) = model.tensors.iter().find(|t| is_vision_tensor(&t.name)) {
        return Err(refuse(format!(
            "{} carries vision tower tensor {}; only the text model loads",
            path.display(),
            t.name
        )));
    }
    let in_mtp_dir = path
        .parent()
        .into_iter()
        .flat_map(Path::components)
        .any(|c| {
            c.as_os_str()
                .to_str()
                .is_some_and(|s| s.eq_ignore_ascii_case("mtp"))
        });
    if in_mtp_dir {
        return Err(refuse(format!(
            "{}: {} is under the MTP directory; the MTP head is not loaded here (#7333)",
            crate::qwen4_exp::MTP_REFUSAL_ID,
            path.display()
        )));
    }
    Ok(())
}

/// The qwen4exp GGUF load entry (#7334, #7333): metadata, then the MTP refusal by tensor
/// name, then the LAYOUT-001 shape contract. The dense loader
/// (`QuantizedGGUFTransformer::from_gguf`) calls it for every `qwen4exp` file before it reads
/// a config or a weight, so a real load refuses `blk.N.nextn.*` by name
/// ([`crate::qwen4_exp::MTP_REFUSAL_ID`]) instead of falling through to a missing dense
/// tensor.
///
/// # Errors
/// A missing or malformed key, MTP tensors present (led by `QWEN4EXP-MTP-REFUSED`), or a
/// tensor that breaks the shape contract.
pub fn load_qwen4exp_gguf(model: &GGUFModel) -> Result<Qwen4ExpGgufMeta> {
    let names: Vec<&str> = model.tensors.iter().map(|t| t.name.as_str()).collect();
    let meta = meta_or_mtp_refusal(&names, Qwen4ExpGgufMeta::from_model(model))?;
    check_shapes(model, &meta)?;
    Ok(meta)
}

/// The dense GGUF loaders must hand `model` to [`dense_loader_refusal`]: its architecture is
/// `qwen4exp`, or it has no `general.architecture` at all and carries `blk.N.nextn.*`
/// tensors. The second case is a non-first `gguf-split` shard opened on its own, which has
/// only the split keys (PR #7356 round 3, lane 1 F2); without it that shard fails on the
/// missing architecture and its MTP tensors are never named.
#[must_use]
pub fn dense_loader_must_refuse(model: &GGUFModel) -> bool {
    match model.architecture() {
        Some(arch) => arch == GGUF_ARCH,
        None => model.tensors.iter().any(|t| is_nextn_name(&t.name)),
    }
}

/// What the dense loader returns for a `qwen4exp` file: the [`load_qwen4exp_gguf`] refusal
/// when it has one, else the named refusal that no qwen4exp forward exists on this path.
#[must_use]
pub fn dense_loader_refusal(model: &GGUFModel) -> RealizarError {
    match load_qwen4exp_gguf(model) {
        Err(e) => e,
        Ok(_) => refuse(format!(
            "architecture '{GGUF_ARCH}' passed the load contract, but the dense GGUF forward \
             does not run it (hyper-connections, PLE and MoE are not wired here)"
        )),
    }
}

// ---------------------------------------------------------------------------
// Dequant coverage
// ---------------------------------------------------------------------------

/// A tensor whose quant type has no CPU dequantizer in this crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DequantGap {
    /// The tensor name.
    pub tensor: String,
    /// The ggml type id.
    pub qtype: u32,
    /// The ggml type name (`IQ1_S`, `IQ2_XS`, …).
    pub type_name: &'static str,
}

/// Every tensor whose quant type [`GGUFModel::has_cpu_dequant`] rejects.
#[must_use]
pub fn dequant_gaps(model: &GGUFModel) -> Vec<DequantGap> {
    model
        .tensors
        .iter()
        .filter(|t| !GGUFModel::has_cpu_dequant(t.qtype))
        .map(|t| DequantGap {
            tensor: t.name.clone(),
            qtype: t.qtype,
            type_name: crate::gguf::ggml_type_table::name(t.qtype).unwrap_or("unknown"),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Split GGUF
// ---------------------------------------------------------------------------

/// A validated `gguf-split` shard set.
#[derive(Debug, Clone)]
pub struct SplitSet {
    /// Hyperparameters, read from shard `split.no = 0`.
    pub meta: Qwen4ExpGgufMeta,
    /// `split.no` of the shard holding [`ENGRAM_TENSOR`]; `None` when the set has no PLE
    /// (`qwen4exp.ple.layers` absent), which needs no engram (PR #7356 lane-1 finding 5).
    pub engram_shard: Option<usize>,
    /// The engram's tensor info, `None` exactly when `engram_shard` is.
    pub engram: Option<TensorInfo>,
    /// Tensors across every shard (= `split.tensors.count`).
    pub total_tensors: usize,
    /// Names across the set that do not map.
    pub unmapped: Vec<String>,
}

fn split_key(model: &GGUFModel, key: &str) -> Result<u64> {
    model
        .metadata
        .get(key)
        .and_then(as_u64)
        .ok_or_else(|| refuse(format!("shard is missing split key {key}")))
}

/// Order `shards` by `split.no`, checking `split.count` and the numbering.
fn order_shards(shards: &[GGUFModel]) -> Result<Vec<&GGUFModel>> {
    let mut ordered: Vec<Option<&GGUFModel>> = vec![None; shards.len()];
    for s in shards {
        let count = split_key(s, "split.count")?;
        if count != shards.len() as u64 {
            return Err(refuse(format!(
                "split.count is {count} but {} shards were given",
                shards.len()
            )));
        }
        let no = split_key(s, "split.no")? as usize;
        match ordered.get_mut(no) {
            Some(slot @ None) => *slot = Some(s),
            _ => return Err(refuse(format!("split.no {no} is out of range or repeated"))),
        }
    }
    Ok(ordered.into_iter().flatten().collect())
}

/// Validate a split qwen4exp GGUF and locate the engram.
///
/// # Errors
/// A shard is missing or repeated, `split.tensors.count` disagrees with the
/// tensors present, a tensor appears twice, MTP tensors are present (led by
/// `QWEN4EXP-MTP-REFUSED`, the same refusal as a single-file load), the set has PLE but no
/// engram, or the metadata or shape contract fails.
pub fn validate_split_set(shards: &[GGUFModel]) -> Result<SplitSet> {
    let ordered = order_shards(shards)?;
    let first = ordered
        .first()
        .ok_or_else(|| refuse("an empty shard set has no split.count".to_string()))?;
    let declared = split_key(first, "split.tensors.count")? as usize;
    let mut seen = HashSet::new();
    let mut all: Vec<&TensorInfo> = Vec::new();
    let mut engram = None;
    for (no, shard) in ordered.iter().enumerate() {
        for t in &shard.tensors {
            if !seen.insert(t.name.as_str()) {
                return Err(refuse(format!(
                    "tensor {} appears in more than one shard",
                    t.name
                )));
            }
            if t.name == ENGRAM_TENSOR {
                engram = Some((no, t.clone()));
            }
            all.push(t);
        }
    }
    if all.len() != declared {
        return Err(refuse(format!(
            "split.tensors.count is {declared} but the shards hold {}",
            all.len()
        )));
    }
    let names: Vec<&str> = all.iter().map(|t| t.name.as_str()).collect();
    let meta = meta_or_mtp_refusal(&names, Qwen4ExpGgufMeta::from_parts(&first.metadata, &all))?;
    if !meta.ple_layers.is_empty() && engram.is_none() {
        // Named for the split set; `check_globals` would also refuse, but only as a
        // generic missing tensor (round 3, lane 1 F3).
        return Err(refuse(format!(
            "the split set has PLE layers, but no shard holds {ENGRAM_TENSOR}"
        )));
    }
    let (engram_shard, engram) = engram.unzip();
    check_shapes_of(&all, &meta)?;
    let unmapped = all
        .iter()
        .filter(|t| map_gguf_tensor(&t.name, &meta).is_none())
        .map(|t| t.name.clone())
        .collect();
    Ok(SplitSet {
        meta,
        engram_shard,
        engram,
        total_tensors: all.len(),
        unmapped,
    })
}

#[cfg(test)]
#[path = "qwen4exp_load_tests.rs"]
mod tests;
