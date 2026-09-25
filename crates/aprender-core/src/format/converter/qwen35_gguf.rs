//! Qwen3.5 (`qwen35`) GGUF export (#4418).
//!
//! Qwen3.5 is a hybrid: Gated DeltaNet linear-attention layers with a full
//! (gated) attention layer every `full_attention_interval` layers. The generic
//! exporter knew none of its tensor names, so every tensor passed through under
//! its HF name, the conv kernel stayed 3-D, and the metadata said
//! `qwen3_5` / 0 heads: llama.cpp refused the file.
//!
//! This module is the llama.cpp `convert_hf_to_gguf.py` `Qwen3_5TextModel`
//! contract, checked element-wise against a reference GGUF written by
//! llama.cpp `d1d3c3396` from a seeded tiny checkpoint:
//!
//! * names: `model.language_model.layers.N.*` -> `blk.N.*` ([`qwen35_gguf_name`]);
//!   `mtp.*` and `model.visual.*` are dropped (the published Q4_K_M ships neither)
//! * values ([`transform_qwen35_tensor`]): `+1` on the RMSNorm weights that HF
//!   stores zero-centred (never on `linear_attn.norm`), `A_log -> -exp(A_log)`,
//!   the conv kernel squeezed `[C,1,K] -> [C,K]`, and every V-head-indexed axis
//!   regrouped from key-head-major to value-head-major ([`reorder_v_heads`])
//! * metadata: `qwen35.ssm.*`, `full_attention_interval`, M-RoPE sections,
//!   partial rotary dims, and a `gpt2`/`qwen35` tokenizer with token types
//! * quantization ([`qwen35_q4km_type`]): llama.cpp's Q4_K_M layer policy

use crate::error::{AprenderError, Result};
use crate::format::gguf::{GgmlType, GgufTensor, GgufValue};
use std::collections::BTreeMap;

/// True when the tensor set is a Qwen3.5 hybrid checkpoint.
pub(crate) fn is_qwen35_tensor_set(tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>) -> bool {
    tensors.keys().any(|k| k.contains(".linear_attn.A_log"))
}

/// GH-279 for a hybrid stack. The dense `qwen3` completeness key demands
/// `self_attn.*` in every layer, which a Qwen3.5 linear-attention layer never
/// has. Each layer must instead carry ONE token mixer (full `self_attn` or
/// Gated-DeltaNet `linear_attn`, complete) plus its norms and SwiGLU MLP.
pub(crate) fn enforce_hybrid_layer_completeness(names: &[&str], num_layers: usize) -> Result<()> {
    const COMMON: &[&str] = &[
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ];
    for i in 0..num_layers {
        let has = |suffix: &str| {
            let want = format!("layers.{i}.{suffix}");
            names
                .iter()
                .any(|n| n.ends_with(&want) && !n.starts_with("mtp."))
        };
        let mixer: Vec<&str> = if has("linear_attn.A_log") {
            QWEN35_LAYER_NAME_MAP
                .iter()
                .map(|(h, _)| *h)
                .filter(|h| h.starts_with("linear_attn."))
                .collect()
        } else {
            QWEN35_LAYER_NAME_MAP
                .iter()
                .map(|(h, _)| *h)
                .filter(|h| h.starts_with("self_attn."))
                .collect()
        };
        if let Some(missing) = COMMON.iter().copied().chain(mixer).find(|s| !has(s)) {
            return Err(AprenderError::FormatError {
                message: format!("hybrid qwen3_5 layer {i} is missing '{missing}' ({num_layers} layers declared)"),
            });
        }
    }
    Ok(())
}

/// HF per-layer suffix -> GGUF per-layer suffix. This table IS the name map;
/// `qwen35_name_map_is_pinned` fails if a row is dropped or altered.
pub(crate) const QWEN35_LAYER_NAME_MAP: &[(&str, &str)] = &[
    ("input_layernorm.weight", "attn_norm.weight"),
    (
        "post_attention_layernorm.weight",
        "post_attention_norm.weight",
    ),
    ("mlp.gate_proj.weight", "ffn_gate.weight"),
    ("mlp.up_proj.weight", "ffn_up.weight"),
    ("mlp.down_proj.weight", "ffn_down.weight"),
    ("linear_attn.in_proj_qkv.weight", "attn_qkv.weight"),
    ("linear_attn.in_proj_z.weight", "attn_gate.weight"),
    ("linear_attn.in_proj_a.weight", "ssm_alpha.weight"),
    ("linear_attn.in_proj_b.weight", "ssm_beta.weight"),
    ("linear_attn.A_log", "ssm_a"),
    ("linear_attn.dt_bias", "ssm_dt.bias"),
    ("linear_attn.conv1d.weight", "ssm_conv1d.weight"),
    ("linear_attn.norm.weight", "ssm_norm.weight"),
    ("linear_attn.out_proj.weight", "ssm_out.weight"),
    ("self_attn.q_proj.weight", "attn_q.weight"),
    ("self_attn.k_proj.weight", "attn_k.weight"),
    ("self_attn.v_proj.weight", "attn_v.weight"),
    ("self_attn.o_proj.weight", "attn_output.weight"),
    ("self_attn.q_norm.weight", "attn_q_norm.weight"),
    ("self_attn.k_norm.weight", "attn_k_norm.weight"),
];

/// Strip the language-model prefix; `None` for vision-tower and MTP tensors.
fn strip_lm_prefix(hf: &str) -> Option<&str> {
    if hf.starts_with("mtp.") || hf.contains("visual.") {
        return None;
    }
    Some(
        hf.strip_prefix("model.language_model.")
            .or_else(|| hf.strip_prefix("language_model.model."))
            .or_else(|| hf.strip_prefix("model."))
            .unwrap_or(hf),
    )
}

/// Map one HF tensor name to its GGUF name, or `None` when it is not exported.
pub(crate) fn qwen35_gguf_name(hf: &str) -> Option<String> {
    let base = strip_lm_prefix(hf)?;
    match base {
        "embed_tokens.weight" => return Some("token_embd.weight".to_string()),
        "norm.weight" => return Some("output_norm.weight".to_string()),
        "lm_head.weight" => return Some("output.weight".to_string()),
        _ => {}
    }
    let rest = base.strip_prefix("layers.")?;
    let (idx, suffix) = rest.split_once('.')?;
    let layer: usize = idx.parse().ok()?;
    QWEN35_LAYER_NAME_MAP
        .iter()
        .find(|(h, _)| *h == suffix)
        .map(|(_, g)| format!("blk.{layer}.{g}"))
}

/// Hybrid-block dimensions the value transforms need.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LinearAttnDims {
    /// Key heads (`linear_num_key_heads`, GGUF `ssm.group_count`).
    pub nk: usize,
    /// Value heads (`linear_num_value_heads`, GGUF `ssm.time_step_rank`).
    pub nv: usize,
    /// Key head dim (`linear_key_head_dim`, GGUF `ssm.state_size`).
    pub dk: usize,
    /// Value head dim (`linear_value_head_dim`).
    pub dv: usize,
}

/// Regroup `nv` value heads of `hd` elements, stored key-head-major
/// (`[nk, nv/nk, hd]`), into value-head-major order (`[nv/nk, nk, hd]`), the
/// layout llama.cpp's broadcast of `nk` key heads over `nv` value heads reads.
/// `src` holds `nv * hd` items of `item` elements each. Identity when `nv == nk`.
pub(crate) fn reorder_v_heads(
    src: &[f32],
    nk: usize,
    nv: usize,
    hd: usize,
    item: usize,
) -> Vec<f32> {
    let nvpk = nv / nk;
    let mut dst = vec![0.0f32; src.len()];
    for k in 0..nk {
        for j in 0..nvpk {
            for c in 0..hd {
                let s = ((k * nvpk + j) * hd + c) * item;
                let d = ((j * nk + k) * hd + c) * item;
                dst[d..d + item].copy_from_slice(&src[s..s + item]);
            }
        }
    }
    dst
}

/// Column form of [`reorder_v_heads`]: regroup the `nv * hd` columns of a
/// `[rows, nv * hd]` row-major matrix (`out_proj`).
fn reorder_v_head_cols(src: &[f32], rows: usize, nk: usize, nv: usize, hd: usize) -> Vec<f32> {
    let cols = nv * hd;
    let mut dst = Vec::with_capacity(src.len());
    for r in 0..rows {
        dst.extend(reorder_v_heads(
            &src[r * cols..(r + 1) * cols],
            nk,
            nv,
            hd,
            1,
        ));
    }
    dst
}

/// Reorder the rows after the first `head_rows` rows (the q/k part of the fused
/// qkv projection and conv kernel stays put; only the v part is per-value-head).
fn reorder_tail_rows(
    data: &[f32],
    row_len: usize,
    head_rows: usize,
    d: LinearAttnDims,
) -> Vec<f32> {
    let split = head_rows * row_len;
    let mut out = data[..split].to_vec();
    out.extend(reorder_v_heads(&data[split..], d.nk, d.nv, d.dv, row_len));
    out
}

/// HF suffixes whose RMSNorm weight is stored zero-centred (`w - 1`).
/// `linear_attn.norm` is deliberately absent: it is a plain gated RMSNorm.
const PLUS_ONE_NORMS: &[&str] = &[
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
];

/// Apply the llama.cpp value transform for one tensor. Returns the new data
/// and the (row-major) shape to write.
pub(crate) fn transform_qwen35_tensor(
    hf: &str,
    data: &[f32],
    shape: &[usize],
    d: LinearAttnDims,
) -> Result<(Vec<f32>, Vec<usize>)> {
    let base = strip_lm_prefix(hf).unwrap_or(hf);
    let suffix = base
        .strip_prefix("layers.")
        .and_then(|r| r.split_once('.'))
        .map_or(base, |(_, s)| s);
    let qk_rows = 2 * d.nk * d.dk;
    let out = match suffix {
        s if PLUS_ONE_NORMS.contains(&s) || base == "norm.weight" => {
            (data.iter().map(|v| v + 1.0).collect(), shape.to_vec())
        }
        "linear_attn.A_log" => {
            let neg: Vec<f32> = data.iter().map(|v| -v.exp()).collect();
            (reorder_v_heads(&neg, d.nk, d.nv, 1, 1), shape.to_vec())
        }
        "linear_attn.dt_bias" => (reorder_v_heads(data, d.nk, d.nv, 1, 1), shape.to_vec()),
        "linear_attn.in_proj_a.weight" | "linear_attn.in_proj_b.weight" => (
            reorder_v_heads(data, d.nk, d.nv, 1, row_len(shape)?),
            shape.to_vec(),
        ),
        "linear_attn.in_proj_z.weight" => (
            reorder_v_heads(data, d.nk, d.nv, d.dv, row_len(shape)?),
            shape.to_vec(),
        ),
        "linear_attn.in_proj_qkv.weight" => (
            reorder_tail_rows(data, row_len(shape)?, qk_rows, d),
            shape.to_vec(),
        ),
        "linear_attn.conv1d.weight" => {
            let (c, k) = conv_dims(shape)?;
            (reorder_tail_rows(data, k, qk_rows, d), vec![c, k])
        }
        "linear_attn.out_proj.weight" => {
            let rows = *shape.first().unwrap_or(&0);
            (
                reorder_v_head_cols(data, rows, d.nk, d.nv, d.dv),
                shape.to_vec(),
            )
        }
        _ => (data.to_vec(), shape.to_vec()),
    };
    Ok(out)
}

fn row_len(shape: &[usize]) -> Result<usize> {
    match shape {
        [_, cols] => Ok(*cols),
        _ => Err(AprenderError::FormatError {
            message: format!("[#4418] qwen35: expected a 2-D projection, got shape {shape:?}"),
        }),
    }
}

/// `[C, 1, K]` (HF) or `[C, K]` -> `(C, K)`.
fn conv_dims(shape: &[usize]) -> Result<(usize, usize)> {
    match shape {
        [c, 1, k] | [c, k] => Ok((*c, *k)),
        _ => Err(AprenderError::FormatError {
            message: format!("[#4418] qwen35: conv1d shape {shape:?} is not [C,1,K]"),
        }),
    }
}

/// llama.cpp `use_more_bits`: the first and last eighth of the layers, and
/// every third layer between, get the larger quant in Q4_K_M.
pub(crate) fn use_more_bits(i: usize, n: usize) -> bool {
    i < n / 8 || i >= 7 * n / 8 || (i >= n / 8 && (i - n / 8) % 3 == 2)
}

/// Q4_K_M tensor type, mirroring llama.cpp `llama_tensor_get_type` for this
/// architecture (and the published unsloth Q4_K_M: token_embd Q6_K, attn_qkv
/// and ssm_out Q5_K, attn_v / ffn_down Q6_K on `use_more_bits` layers).
/// `attn_v_rank` is the tensor's index among the `attn_v` tensors; llama.cpp
/// counts it against `n_layer`, not against the number of attn_v tensors.
/// ssm_alpha/ssm_beta stay F32: llama.cpp uses Q8_0 there and trueno_quant has
/// no Q8_0 encoder (they are 2 x nv x hidden, well under 1% of the file).
pub(crate) fn qwen35_q4km_type(
    gguf: &str,
    layer: Option<usize>,
    attn_v_rank: usize,
    n_layer: usize,
) -> GgmlType {
    let kind = gguf.rsplit_once('.').map_or(gguf, |(head, tail)| {
        if tail == "weight" {
            head.rsplit('.').next().unwrap_or(head)
        } else {
            tail
        }
    });
    match kind {
        "token_embd" | "output" => GgmlType::Q6K,
        "attn_qkv" | "ssm_out" => GgmlType::Q5K,
        "ssm_alpha" | "ssm_beta" => GgmlType::F32,
        "attn_v" if use_more_bits(attn_v_rank, n_layer) => GgmlType::Q6K,
        "ffn_down" if layer.is_some_and(|l| use_more_bits(l, n_layer)) => GgmlType::Q6K,
        _ => GgmlType::Q4K,
    }
}

/// Encode one tensor. 1-D tensors, the conv kernel, and any matrix whose row
/// is not a multiple of the 256-element super-block stay F32.
fn encode(data: &[f32], shape: &[usize], want: GgmlType) -> (GgmlType, Vec<u8>) {
    let quantizable = shape.len() == 2 && shape[1] % 256 == 0;
    match want {
        GgmlType::Q4K if quantizable => (want, super::quantize_q4_k_matrix(data, shape)),
        GgmlType::Q5K if quantizable => (want, trueno_quant::quantize_q5_k_matrix(data, shape)),
        GgmlType::Q6K if quantizable => (want, trueno_quant::quantize_q6_k_matrix(data, shape)),
        _ => (
            GgmlType::F32,
            data.iter().flat_map(|f| f.to_le_bytes()).collect(),
        ),
    }
}

/// Row-major `[rows, cols]` -> GGUF `[ne0 = cols, ne1 = rows]`.
fn gguf_shape(shape: &[usize]) -> Vec<u64> {
    shape.iter().rev().map(|&d| d as u64).collect()
}

fn layer_of(gguf: &str) -> Option<usize> {
    gguf.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

/// Map, transform and encode every exported tensor. `quantize` selects Q4_K_M;
/// otherwise every tensor is written F32.
pub(crate) fn build_qwen35_tensors(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    d: LinearAttnDims,
    n_layer: usize,
    quantize: bool,
) -> Result<Vec<GgufTensor>> {
    let embed = tensors
        .iter()
        .find(|(k, _)| k.ends_with("embed_tokens.weight"))
        .map(|(_, v)| v);
    let mut mapped: Vec<(usize, String, &String)> = Vec::new();
    for hf in tensors.keys() {
        let Some(g) = qwen35_gguf_name(hf) else {
            continue;
        };
        // Tied (the APR import materialises lm_head as a copy of the
        // embedding): llama.cpp falls back to token_embd for the output.
        if g == "output.weight" && embed.is_some_and(|e| e.0 == tensors[hf].0) {
            continue;
        }
        mapped.push((layer_of(&g).unwrap_or(usize::MAX), g, hf));
    }
    mapped.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let mut out = Vec::with_capacity(mapped.len());
    let mut attn_v_rank = 0usize;
    for (_, g, hf) in mapped {
        let (data, shape) = &tensors[hf];
        let (data, shape) = transform_qwen35_tensor(hf, data, shape, d)?;
        let want = if quantize {
            qwen35_q4km_type(&g, layer_of(&g), attn_v_rank, n_layer)
        } else {
            GgmlType::F32
        };
        if g.ends_with(".attn_v.weight") {
            attn_v_rank += 1;
        }
        let (dtype, bytes) = encode(&data, &shape, want);
        out.push(GgufTensor {
            name: g,
            shape: gguf_shape(&shape),
            dtype,
            data: bytes,
        });
    }
    Ok(out)
}

/// Everything the `qwen35.*` metadata block needs.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Qwen35Hparams {
    pub n_layer: usize,
    pub hidden: usize,
    pub ffn: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub ctx: usize,
    pub rope_theta: f32,
    pub eps: f32,
    pub rot_dim: usize,
    pub mrope: [i32; 4],
    pub conv_kernel: usize,
    pub full_attn_interval: usize,
    pub dims: LinearAttnDims,
}

fn hp_usize(h: &serde_json::Map<String, serde_json::Value>, k: &str) -> Result<usize> {
    h.get(k)
        .and_then(serde_json::Value::as_u64)
        .map(|v| v as usize)
        .ok_or_else(|| AprenderError::FormatError {
            message: format!(
                "[#4418] qwen35 export: `{k}` missing from linear_attn_hparams \
                 (re-import from a config.json that has it; no default is guessed)"
            ),
        })
}

fn need(v: Option<usize>, what: &str) -> Result<usize> {
    v.filter(|&x| x > 0)
        .ok_or_else(|| AprenderError::FormatError {
            message: format!("[#4418] qwen35 export: model config has no `{what}`"),
        })
}

/// First dimension of the tensor whose name ends with `suffix`.
fn dim0(tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>, suffix: &str) -> Option<usize> {
    tensors
        .iter()
        .find(|(k, _)| k.ends_with(suffix))
        .and_then(|(_, (_, s))| s.first().copied())
}

/// Standard fields of the resolved model config (APR metadata or config.json).
#[derive(Debug, Clone, Default)]
pub(crate) struct Qwen35Base {
    pub n_layer: Option<usize>,
    pub hidden: Option<usize>,
    pub ffn: Option<usize>,
    pub n_head: Option<usize>,
    pub n_head_kv: Option<usize>,
    pub head_dim: Option<usize>,
    pub ctx: Option<usize>,
    pub rope_theta: Option<f32>,
    pub eps: Option<f32>,
}

/// Resolve the hyperparameters, cross-checking the hparams against the tensor
/// shapes (a mismatch is a loud error, never a silently wrong file).
pub(crate) fn resolve_qwen35_hparams(
    base: &Qwen35Base,
    h: &serde_json::Map<String, serde_json::Value>,
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
) -> Result<Qwen35Hparams> {
    let dims = LinearAttnDims {
        nk: hp_usize(h, "linear_num_key_heads")?,
        nv: hp_usize(h, "linear_num_value_heads")?,
        dk: hp_usize(h, "linear_key_head_dim")?,
        dv: hp_usize(h, "linear_value_head_dim")?,
    };
    check_dim(
        dim0(tensors, "linear_attn.A_log"),
        dims.nv,
        "A_log length vs linear_num_value_heads",
    )?;
    check_dim(
        dim0(tensors, "linear_attn.norm.weight"),
        dims.dv,
        "linear_attn.norm vs linear_value_head_dim",
    )?;
    if dims.nk == 0 || dims.nv % dims.nk != 0 {
        return Err(AprenderError::FormatError {
            message: format!(
                "[#4418] qwen35: {} value heads not a multiple of {} key heads",
                dims.nv, dims.nk
            ),
        });
    }
    let head_dim = need(
        base.head_dim
            .or_else(|| dim0(tensors, "self_attn.q_norm.weight")),
        "head_dim",
    )?;
    let partial = h
        .get("partial_rotary_factor")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(1.0);
    Ok(Qwen35Hparams {
        n_layer: need(base.n_layer, "num_hidden_layers")?,
        hidden: need(base.hidden, "hidden_size")?,
        ffn: need(base.ffn, "intermediate_size")?,
        n_head: need(base.n_head, "num_attention_heads")?,
        n_head_kv: need(base.n_head_kv, "num_key_value_heads")?,
        head_dim,
        ctx: base.ctx.unwrap_or(0),
        rope_theta: base.rope_theta.unwrap_or(10_000_000.0),
        eps: base.eps.unwrap_or(1e-6),
        rot_dim: (head_dim as f64 * partial).round() as usize,
        mrope: mrope_sections(h),
        conv_kernel: hp_usize(h, "linear_conv_kernel_dim")?,
        full_attn_interval: hp_usize(h, "full_attention_interval")?,
        dims,
    })
}

fn check_dim(got: Option<usize>, want: usize, what: &str) -> Result<()> {
    match got {
        Some(g) if g != want => Err(AprenderError::FormatError {
            message: format!("[#4418] qwen35: {what}: tensor says {g}, config says {want}"),
        }),
        _ => Ok(()),
    }
}

/// M-RoPE sections padded to 4 (llama.cpp default `[11, 11, 10, 0]`).
fn mrope_sections(h: &serde_json::Map<String, serde_json::Value>) -> [i32; 4] {
    let mut out = [11, 11, 10, 0];
    if let Some(arr) = h.get("mrope_section").and_then(serde_json::Value::as_array) {
        out = [0; 4];
        for (slot, v) in out.iter_mut().zip(arr) {
            *slot = v.as_i64().unwrap_or(0) as i32;
        }
    }
    out
}

/// The `general.*` + `qwen35.*` metadata block.
pub(crate) fn qwen35_config_metadata(
    hp: &Qwen35Hparams,
    name: &str,
    quantize: bool,
) -> Vec<(String, GgufValue)> {
    let u = |v: usize| GgufValue::Uint32(v as u32);
    let a = "qwen35";
    let d = hp.dims;
    vec![
        ("general.architecture".into(), GgufValue::String(a.into())),
        ("general.type".into(), GgufValue::String("model".into())),
        ("general.name".into(), GgufValue::String(name.into())),
        ("general.file_type".into(), u(if quantize { 15 } else { 0 })),
        ("general.quantization_version".into(), u(2)),
        (format!("{a}.block_count"), u(hp.n_layer)),
        (format!("{a}.context_length"), u(hp.ctx)),
        (format!("{a}.embedding_length"), u(hp.hidden)),
        (format!("{a}.feed_forward_length"), u(hp.ffn)),
        (format!("{a}.attention.head_count"), u(hp.n_head)),
        (format!("{a}.attention.head_count_kv"), u(hp.n_head_kv)),
        (format!("{a}.attention.key_length"), u(hp.head_dim)),
        (format!("{a}.attention.value_length"), u(hp.head_dim)),
        (
            format!("{a}.attention.layer_norm_rms_epsilon"),
            GgufValue::Float32(hp.eps),
        ),
        (
            format!("{a}.rope.freq_base"),
            GgufValue::Float32(hp.rope_theta),
        ),
        (format!("{a}.rope.dimension_count"), u(hp.rot_dim)),
        (
            format!("{a}.rope.dimension_sections"),
            GgufValue::ArrayInt32(hp.mrope.to_vec()),
        ),
        (format!("{a}.ssm.conv_kernel"), u(hp.conv_kernel)),
        (format!("{a}.ssm.state_size"), u(d.dk)),
        (format!("{a}.ssm.group_count"), u(d.nk)),
        (format!("{a}.ssm.time_step_rank"), u(d.nv)),
        (format!("{a}.ssm.inner_size"), u(d.nv * d.dv)),
        (
            format!("{a}.full_attention_interval"),
            u(hp.full_attn_interval),
        ),
    ]
}

/// llama.cpp token types.
const TT_NORMAL: i32 = 1;
const TT_CONTROL: i32 = 3;
const TT_USER_DEFINED: i32 = 4;
const TT_UNUSED: i32 = 5;

fn is_special(tok: &str) -> bool {
    tok.len() > 4 && tok.starts_with("<|") && tok.ends_with("|>")
}

/// Qwen3.5 lists seven audio/TTS specials (`<|audio_start|>` ..) only in
/// `tokenizer_config.json` `added_tokens_decoder`, never in `tokenizer.json`,
/// so the vocab arrives with filler holes at their ids. Fill those holes
/// (and only those) so the vocab matches `convert_hf_to_gguf.py`.
/// One `added_tokens_decoder` entry of `tokenizer_config.json`.
#[derive(Debug, Clone)]
pub(crate) struct AddedToken {
    pub(crate) id: usize,
    pub(crate) content: String,
    pub(crate) special: bool,
}

pub(crate) fn fill_added_token_holes(metadata: &mut [(String, GgufValue)], added: &[AddedToken]) {
    let Some(tokens) = metadata.iter_mut().find_map(|(k, v)| match v {
        GgufValue::ArrayString(t) if k == "tokenizer.ggml.tokens" => Some(t),
        _ => None,
    }) else {
        return;
    };
    for AddedToken { id, content, .. } in added {
        if let Some(t) = tokens.get_mut(*id) {
            // Hole fillers: `<|pad_<id>|>` (APR vocab padded to the embed rows,
            // metadata.rs), `<unk>` / `<unk>_N` (tokenizer.json loader + GH-279
            // dedup), `[PAD<id>]` (GH-277). Qwen has no real token of any form.
            let hole = *t == format!("<|pad_{id}|>")
                || t == "<unk>"
                || t.starts_with("<unk>_")
                || *t == format!("[PAD{id}]");
            if hole {
                t.clone_from(content);
            }
        }
    }
}

/// `convert_hf_to_gguf.py` types an added token CONTROL when its config entry
/// says `special` or it looks like `<|..|>`, else USER_DEFINED; `fix_qwen35_tokenizer_metadata` can only
/// guess from the `<|..|>` shape (`<tts_pad>` is special, `<think>` is not).
pub(crate) fn apply_added_token_types(metadata: &mut [(String, GgufValue)], added: &[AddedToken]) {
    let tokens = metadata.iter().find_map(|(k, v)| match v {
        GgufValue::ArrayString(t) if k == "tokenizer.ggml.tokens" => Some(t.clone()),
        _ => None,
    });
    let Some(tokens) = tokens else { return };
    let Some(types) = metadata.iter_mut().find_map(|(k, v)| match v {
        GgufValue::ArrayInt32(t) if k == "tokenizer.ggml.token_type" => Some(t),
        _ => None,
    }) else {
        return;
    };
    for a in added {
        if tokens.get(a.id) == Some(&a.content) {
            if let Some(ty) = types.get_mut(a.id) {
                // conversion/base.py: `special or does_token_look_special(token)`
                *ty = if a.special || is_special(&a.content) {
                    TT_CONTROL
                } else {
                    TT_USER_DEFINED
                };
            }
        }
    }
}

/// Rewrite the generic tokenizer block into what llama.cpp's `qwen35` loader
/// expects: model `gpt2`, pre `qwen35`, `[PADn]` placeholders typed UNUSED,
/// a `token_type` array, eos = `<|im_end|>`, padding = `<|endoftext|>`, and
/// no BOS. Added tokens start at the first `<|...|>` id: from there `<|...|>`
/// is CONTROL and anything else USER_DEFINED (`<think>`, `<tool_call>`, ...).
pub(crate) fn fix_qwen35_tokenizer_metadata(metadata: &mut Vec<(String, GgufValue)>) {
    let Some(tokens) = metadata.iter_mut().find_map(|(k, v)| match v {
        GgufValue::ArrayString(t) if k == "tokenizer.ggml.tokens" => Some(t),
        _ => None,
    }) else {
        return;
    };
    let mut types = Vec::with_capacity(tokens.len());
    let first_added = tokens
        .iter()
        .position(|t| is_special(t))
        .unwrap_or(tokens.len());
    for (i, t) in tokens.iter_mut().enumerate() {
        if t.starts_with("<|pad_") && t.ends_with("|>") {
            *t = format!("[PAD{i}]");
            types.push(TT_UNUSED);
        } else if i < first_added {
            types.push(TT_NORMAL);
        } else if is_special(t) {
            types.push(TT_CONTROL);
        } else {
            types.push(TT_USER_DEFINED);
        }
    }
    let id_of = |name: &str| tokens.iter().position(|t| t == name).map(|i| i as u32);
    let eos = id_of("<|im_end|>");
    let pad = id_of("<|endoftext|>");
    metadata.retain(|(k, _)| {
        !matches!(
            k.as_str(),
            "tokenizer.ggml.model"
                | "tokenizer.ggml.pre"
                | "tokenizer.ggml.token_type"
                | "tokenizer.ggml.bos_token_id"
                | "tokenizer.ggml.eos_token_id"
                | "tokenizer.ggml.padding_token_id"
                | "tokenizer.ggml.add_bos_token"
        )
    });
    metadata.push((
        "tokenizer.ggml.model".into(),
        GgufValue::String("gpt2".into()),
    ));
    metadata.push((
        "tokenizer.ggml.pre".into(),
        GgufValue::String("qwen35".into()),
    ));
    metadata.push((
        "tokenizer.ggml.token_type".into(),
        GgufValue::ArrayInt32(types),
    ));
    if let Some(e) = eos {
        metadata.push(("tokenizer.ggml.eos_token_id".into(), GgufValue::Uint32(e)));
    }
    if let Some(p) = pad {
        metadata.push((
            "tokenizer.ggml.padding_token_id".into(),
            GgufValue::Uint32(p),
        ));
    }
    metadata.push((
        "tokenizer.ggml.add_bos_token".into(),
        GgufValue::Bool(false),
    ));
}

#[cfg(test)]
#[path = "qwen35_gguf_tests.rs"]
mod tests;
