//! T1-R2 step 2b (#4000, contract `qwen35-train-gdn-v1`): a whole Qwen3.5 model for
//! training — embedding, the hybrid layer stack, final norm, `lm_head` — loaded from a
//! GGUF into f32, with a sequence forward to logits.
//!
//! The loader reads the file on its own: `aprender`'s `GgufReader` dequantises each
//! tensor by name. It does not share serve's `Qwen35OwnedLayer` mapping, so the
//! full-logit parity test (`QTG-001`) compares two independent readings of the file.
//! Layer kind comes from which tensors a block carries (`ssm_a` = Gated `DeltaNet`,
//! `attn_q` = gated attention), and every width comes from a tensor's length where one
//! records it (head width = `attn_q_norm`, value heads = `ssm_a`). Metadata is used only
//! for what no tensor records: head counts, `state_size`, rope sections, theta, eps.

use std::path::Path;

use aprender::format::gguf::{GgufReader, GgufValue};

use super::gdn::project;
use super::qwen35_layer::{
    qwen35_block_forward, rms_norm_chunks, GatedAttnDims, GatedAttnWeights, Qwen35Mixer,
    SwiGluWeights,
};
use super::{GdnDims, GdnWeights};
use crate::{Error, Result};

/// A Gated `DeltaNet` mixer's weights, owned.
#[derive(Debug, Clone)]
struct OwnedGdn {
    qkv: Vec<f32>,
    gate: Vec<f32>,
    alpha: Vec<f32>,
    beta: Vec<f32>,
    a: Vec<f32>,
    dt_bias: Vec<f32>,
    conv: Vec<f32>,
    norm: Vec<f32>,
    out: Vec<f32>,
    dims: GdnDims,
}

/// A gated full-attention mixer's weights, owned.
#[derive(Debug, Clone)]
struct OwnedAttn {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    out: Vec<f32>,
    dims: GatedAttnDims,
}

/// Which mixer a layer has.
#[derive(Debug, Clone)]
enum OwnedMixer {
    Gdn(OwnedGdn),
    Attention(OwnedAttn),
}

/// One Qwen3.5 layer, owned: the mixer plus the norms and `SwiGLU` both kinds share.
#[derive(Debug, Clone)]
struct Qwen35Layer {
    attn_norm: Vec<f32>,
    mixer: OwnedMixer,
    post_norm: Vec<f32>,
    ffn_gate: Vec<f32>,
    ffn_up: Vec<f32>,
    ffn_down: Vec<f32>,
}

impl Qwen35Layer {
    fn forward(&self, hidden: &[f32], eps: f32) -> Vec<f32> {
        let mixer = match &self.mixer {
            OwnedMixer::Gdn(g) => Qwen35Mixer::Gdn(
                GdnWeights {
                    qkv: &g.qkv,
                    gate: &g.gate,
                    alpha: &g.alpha,
                    beta: &g.beta,
                    a: &g.a,
                    dt_bias: &g.dt_bias,
                    conv: &g.conv,
                    norm: &g.norm,
                    out: &g.out,
                },
                g.dims,
            ),
            OwnedMixer::Attention(a) => Qwen35Mixer::Attention(
                GatedAttnWeights {
                    q: &a.q,
                    k: &a.k,
                    v: &a.v,
                    q_norm: &a.q_norm,
                    k_norm: &a.k_norm,
                    out: &a.out,
                },
                a.dims,
            ),
        };
        let ffn = SwiGluWeights { gate: &self.ffn_gate, up: &self.ffn_up, down: &self.ffn_down };
        qwen35_block_forward(hidden, &self.attn_norm, &mixer, &self.post_norm, &ffn, eps)
    }
}

/// A whole Qwen3.5 (hybrid Gated `DeltaNet` + gated attention) model in f32.
#[derive(Debug, Clone)]
pub struct Qwen35Model {
    hidden_dim: usize,
    vocab_size: usize,
    eps: f32,
    embed: Vec<f32>,
    layers: Vec<Qwen35Layer>,
    final_norm: Vec<f32>,
    /// `None` when the file ties `lm_head` to the embedding (no `output.weight`).
    lm_head: Option<Vec<f32>>,
}

/// The file and the metadata lookups the loader needs.
struct Gguf {
    reader: GgufReader,
    arch: String,
}

impl Gguf {
    fn value(&self, key: &str) -> Option<&GgufValue> {
        self.reader.metadata.get(&format!("{}.{key}", self.arch))
    }

    fn usize(&self, key: &str) -> Result<usize> {
        match self.value(key) {
            Some(GgufValue::Uint32(v)) => Ok(*v as usize),
            Some(GgufValue::Int32(v)) => usize::try_from(*v).map_err(|e| parse(key, e)),
            Some(GgufValue::Uint64(v)) => usize::try_from(*v).map_err(|e| parse(key, e)),
            other => Err(Error::Parse(format!(
                "{}.{key}: missing or not an integer ({other:?})",
                self.arch
            ))),
        }
    }

    fn f32(&self, key: &str) -> Result<f32> {
        match self.value(key) {
            Some(GgufValue::Float32(v)) => Ok(*v),
            other => {
                Err(Error::Parse(format!("{}.{key}: missing or not f32 ({other:?})", self.arch)))
            }
        }
    }

    fn rope_sections(&self) -> Result<Vec<usize>> {
        let key = "rope.dimension_sections";
        match self.value(key) {
            Some(GgufValue::ArrayUint32(v)) => Ok(v.iter().map(|&x| x as usize).collect()),
            Some(GgufValue::ArrayInt32(v)) => {
                v.iter().map(|&x| usize::try_from(x).map_err(|e| parse(key, e))).collect()
            }
            other => Err(Error::Parse(format!(
                "{}.{key}: missing or not an int array ({other:?})",
                self.arch
            ))),
        }
    }

    fn has(&self, name: &str) -> bool {
        self.reader.tensors.iter().any(|t| t.name == name)
    }

    fn tensor(&self, name: &str) -> Result<Vec<f32>> {
        self.reader
            .get_tensor_f32(name)
            .map(|(data, _)| data)
            .map_err(|e| Error::Parse(format!("tensor {name}: {e}")))
    }
}

fn parse(key: &str, e: impl std::fmt::Display) -> Error {
    Error::Parse(format!("{key}: {e}"))
}

impl Qwen35Model {
    /// Load a Qwen3.5 GGUF, dequantising every tensor to f32.
    ///
    /// # Errors
    /// The file is unreadable, a tensor or a required metadata key is missing, or a
    /// block carries neither a Gated `DeltaNet` nor an attention mixer.
    pub fn from_gguf(path: impl AsRef<Path>) -> Result<Self> {
        let reader = GgufReader::from_file_full(path.as_ref())
            .map_err(|e| Error::Io(format!("{}: {e}", path.as_ref().display())))?;
        let arch = reader
            .architecture()
            .ok_or_else(|| Error::Parse("general.architecture missing".to_string()))?;
        let g = Gguf { reader, arch };

        let hidden_dim = g.usize("embedding_length")?;
        let eps = g.f32("attention.layer_norm_rms_epsilon")?;
        let embed = g.tensor("token_embd.weight")?;
        let vocab_size = embed.len() / hidden_dim;
        let final_norm = g.tensor("output_norm.weight")?;
        let lm_head = if g.has("output.weight") { Some(g.tensor("output.weight")?) } else { None };

        let layers = (0..g.usize("block_count")?)
            .map(|i| load_layer(&g, i, hidden_dim, eps))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { hidden_dim, vocab_size, eps, embed, layers, final_norm, lm_head })
    }

    /// Vocabulary size (rows of the embedding).
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Number of layers.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// For each layer, whether it is full attention (`true`) or Gated `DeltaNet`.
    #[must_use]
    pub fn attention_schedule(&self) -> Vec<bool> {
        self.layers.iter().map(|l| matches!(l.mixer, OwnedMixer::Attention(_))).collect()
    }

    /// Causal forward over `tokens` from an empty state; logits are `[len × vocab]`.
    ///
    /// # Panics
    /// If a token id is outside the vocabulary.
    #[must_use]
    pub fn forward(&self, tokens: &[u32]) -> Vec<f32> {
        let d = self.hidden_dim;
        let mut h = Vec::with_capacity(tokens.len() * d);
        for &t in tokens {
            let t = t as usize;
            assert!(t < self.vocab_size, "token {t} outside vocab {}", self.vocab_size);
            h.extend_from_slice(&self.embed[t * d..(t + 1) * d]);
        }
        for layer in &self.layers {
            h = layer.forward(&h, self.eps);
        }
        rms_norm_chunks(&mut h, &self.final_norm, self.eps);
        project(&h, self.lm_head.as_deref().unwrap_or(&self.embed), d, self.vocab_size)
    }
}

fn load_layer(g: &Gguf, i: usize, hidden_dim: usize, eps: f32) -> Result<Qwen35Layer> {
    let t = |s: &str| g.tensor(&format!("blk.{i}.{s}"));
    let mixer = if g.has(&format!("blk.{i}.ssm_a")) {
        let (a, norm, qkv) = (t("ssm_a")?, t("ssm_norm.weight")?, t("attn_qkv.weight")?);
        let conv = t("ssm_conv1d.weight")?;
        let (num_v_heads, head_v_dim) = (a.len(), norm.len());
        let head_k_dim = g.usize("ssm.state_size")?;
        let conv_dim = qkv.len() / hidden_dim;
        let dims = GdnDims {
            hidden_dim,
            num_k_heads: (conv_dim - num_v_heads * head_v_dim) / (2 * head_k_dim),
            head_k_dim,
            num_v_heads,
            head_v_dim,
            conv_kernel: conv.len() / conv_dim,
            eps,
        };
        OwnedMixer::Gdn(OwnedGdn {
            qkv,
            gate: t("attn_gate.weight")?,
            alpha: t("ssm_alpha.weight")?,
            beta: t("ssm_beta.weight")?,
            a,
            dt_bias: t("ssm_dt.bias")?,
            conv,
            norm,
            out: t("ssm_out.weight")?,
            dims,
        })
    } else if g.has(&format!("blk.{i}.attn_q.weight")) {
        let (q_norm, k) = (t("attn_q_norm.weight")?, t("attn_k.weight")?);
        let head_dim = q_norm.len();
        let dims = GatedAttnDims {
            hidden_dim,
            num_heads: g.usize("attention.head_count")?,
            num_kv_heads: k.len() / hidden_dim / head_dim,
            head_dim,
            n_rot: 2 * g.rope_sections()?.iter().sum::<usize>(),
            rope_theta: g.f32("rope.freq_base")?,
            eps,
        };
        OwnedMixer::Attention(OwnedAttn {
            q: t("attn_q.weight")?,
            k,
            v: t("attn_v.weight")?,
            q_norm,
            k_norm: t("attn_k_norm.weight")?,
            out: t("attn_output.weight")?,
            dims,
        })
    } else {
        return Err(Error::Parse(format!(
            "blk.{i}: neither ssm_a nor attn_q — not a Qwen3.5 block"
        )));
    };
    Ok(Qwen35Layer {
        attn_norm: t("attn_norm.weight")?,
        mixer,
        post_norm: t("post_attention_norm.weight")?,
        ffn_gate: t("ffn_gate.weight")?,
        ffn_up: t("ffn_up.weight")?,
        ffn_down: t("ffn_down.weight")?,
    })
}

#[cfg(test)]
#[path = "qwen35_model_tests.rs"]
mod tests;
