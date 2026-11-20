//! GH-7334: `gguf_qwen4exp_*` tests over a header-only GGUF built in-test.
//!
//! The fixture has a header, metadata and tensor infos and no tensor data, the
//! same as a real file's header with its weight bytes cut off. The names come
//! from llama.cpp b12557 `src/models/qwen4exp.cpp` (`load_arch_tensors`); they
//! are written out here by hand and are not generated from the code under test.

use super::*;
use crate::gguf::GGUFModel;

// ---------------------------------------------------------------------------
// Minimal GGUF v3 writer (header + KV + tensor infos, no data)
// ---------------------------------------------------------------------------

enum Kv {
    Str(&'static str),
    U32(u32),
    U16(u16),
    I32(i32),
    F32(f32),
    U32s(Vec<u32>),
    U64s(Vec<u64>),
    F32s(Vec<f32>),
    Bools(Vec<bool>),
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn put_kv(out: &mut Vec<u8>, key: &str, v: &Kv) {
    put_str(out, key);
    match v {
        Kv::Str(s) => {
            out.extend_from_slice(&8u32.to_le_bytes());
            put_str(out, s);
        },
        Kv::U32(x) => {
            out.extend_from_slice(&4u32.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        },
        Kv::U16(x) => {
            out.extend_from_slice(&2u32.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        },
        Kv::I32(x) => {
            out.extend_from_slice(&5u32.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        },
        Kv::F32(x) => {
            out.extend_from_slice(&6u32.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        },
        Kv::U32s(xs) => {
            out.extend_from_slice(&9u32.to_le_bytes());
            out.extend_from_slice(&4u32.to_le_bytes());
            out.extend_from_slice(&(xs.len() as u64).to_le_bytes());
            for x in xs {
                out.extend_from_slice(&x.to_le_bytes());
            }
        },
        Kv::U64s(xs) => {
            out.extend_from_slice(&9u32.to_le_bytes());
            out.extend_from_slice(&10u32.to_le_bytes());
            out.extend_from_slice(&(xs.len() as u64).to_le_bytes());
            for x in xs {
                out.extend_from_slice(&x.to_le_bytes());
            }
        },
        Kv::F32s(xs) => {
            out.extend_from_slice(&9u32.to_le_bytes());
            out.extend_from_slice(&6u32.to_le_bytes());
            out.extend_from_slice(&(xs.len() as u64).to_le_bytes());
            for x in xs {
                out.extend_from_slice(&x.to_le_bytes());
            }
        },
        Kv::Bools(xs) => {
            out.extend_from_slice(&9u32.to_le_bytes());
            out.extend_from_slice(&7u32.to_le_bytes());
            out.extend_from_slice(&(xs.len() as u64).to_le_bytes());
            for x in xs {
                out.push(u8::from(*x));
            }
        },
    }
}

struct Fixture {
    kvs: Vec<(String, Kv)>,
    tensors: Vec<(String, Vec<u64>, u32)>,
}

impl Fixture {
    fn kv(&mut self, key: &str, v: Kv) {
        self.kvs.retain(|(k, _)| k != key);
        self.kvs.push((key.to_string(), v));
    }
    fn drop_kv(&mut self, key: &str) {
        self.kvs.retain(|(k, _)| k != key);
    }
    fn t(&mut self, name: &str, dims: &[u64], qtype: u32) {
        self.tensors.push((name.to_string(), dims.to_vec(), qtype));
    }
    fn retype(&mut self, name: &str, qtype: u32) {
        for t in &mut self.tensors {
            if t.0 == name {
                t.2 = qtype;
            }
        }
    }
    fn reshape(&mut self, name: &str, dims: &[u64]) {
        for t in &mut self.tensors {
            if t.0 == name {
                t.1 = dims.to_vec();
            }
        }
    }
    fn bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&0x4655_4747u32.to_le_bytes());
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.kvs.len() as u64).to_le_bytes());
        for (k, v) in &self.kvs {
            put_kv(&mut out, k, v);
        }
        for (i, (name, dims, qtype)) in self.tensors.iter().enumerate() {
            put_str(&mut out, name);
            out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in dims {
                out.extend_from_slice(&d.to_le_bytes());
            }
            out.extend_from_slice(&qtype.to_le_bytes());
            out.extend_from_slice(&((i as u64) * 32).to_le_bytes());
        }
        while out.len() % 32 != 0 {
            out.push(0);
        }
        out
    }
    fn model(&self) -> GGUFModel {
        GGUFModel::from_bytes(&self.bytes()).expect("fixture parses")
    }
}

// Fixture shape: 4 trunk layers + 1 MTP layer; full_attention_interval 4, so
// layers 0..=2 are Gated DeltaNet, 3 is full attention, 4 is the MTP block.
const E: u64 = 8; // n_embd
const V: u64 = 16; // n_vocab
const HEADS: u64 = 2;
const KV_HEADS: u64 = 1;
const HEAD_DIM: u64 = 4;
const N_EXPERT: u64 = 4;
const FF_EXP: u64 = 4;
const FF_SH: u64 = 4;
const HC: u64 = 2;
const HC_LR: u64 = 3;
const D_STATE: u64 = 4; // head_k_dim = head_v_dim
const N_GROUP: u64 = 2; // n_k_heads
const DT_RANK: u64 = 2; // n_v_heads
const D_CONV: u64 = 4;
const IDX_HEADS: u64 = 2;
const IDX_DIM: u64 = 4;
const PLE_DIM: u64 = 4; // embedding_length_per_layer_input
const PLE_ROWS: u64 = 40;
const F32: u32 = 0;
const Q2_K: u32 = 10;
const IQ2_XS: u32 = 17;
const IQ1_S: u32 = 19;
const IQ4_NL: u32 = 20;

fn hparams() -> Vec<(String, Kv)> {
    let p = |k: &str| format!("qwen4exp.{k}");
    vec![
        ("general.architecture".into(), Kv::Str("qwen4exp")),
        (p("block_count"), Kv::U32(5)),
        (p("nextn_predict_layers"), Kv::U32(1)),
        (p("context_length"), Kv::U32(4096)),
        (p("embedding_length"), Kv::U32(E as u32)),
        (p("attention.head_count"), Kv::U32(HEADS as u32)),
        (p("attention.head_count_kv"), Kv::U32(KV_HEADS as u32)),
        (p("attention.key_length"), Kv::U32(HEAD_DIM as u32)),
        (p("attention.value_length"), Kv::U32(HEAD_DIM as u32)),
        (p("attention.layer_norm_rms_epsilon"), Kv::F32(1e-6)),
        (p("expert_count"), Kv::U32(N_EXPERT as u32)),
        (p("expert_used_count"), Kv::U32(2)),
        (p("expert_feed_forward_length"), Kv::U32(FF_EXP as u32)),
        (
            p("expert_shared_feed_forward_length"),
            Kv::U32(FF_SH as u32),
        ),
        (p("rope.freq_base"), Kv::F32(10_000_000.0)),
        (p("rope.dimension_count"), Kv::U32(HEAD_DIM as u32)),
        (p("ssm.conv_kernel"), Kv::U32(D_CONV as u32)),
        (p("ssm.inner_size"), Kv::U32((D_STATE * DT_RANK) as u32)),
        (p("ssm.state_size"), Kv::U32(D_STATE as u32)),
        (p("ssm.time_step_rank"), Kv::U32(DT_RANK as u32)),
        (p("ssm.group_count"), Kv::U32(N_GROUP as u32)),
        (p("hyper_connection.count"), Kv::U32(HC as u32)),
        (p("hyper_connection.low_rank"), Kv::U32(HC_LR as u32)),
        (p("attention.indexer.head_count"), Kv::U32(IDX_HEADS as u32)),
        (p("attention.indexer.key_length"), Kv::U32(IDX_DIM as u32)),
        (p("attention.indexer.top_k"), Kv::U32(16)),
        (p("full_attention_interval"), Kv::U32(4)),
        (p("ple.layers"), Kv::U32s(vec![1])),
        (p("ple.ngram_size"), Kv::U32(3)),
        (p("ple.heads_per_ngram"), Kv::U32(2)),
        (p("ple.conv_kernel"), Kv::U32(4)),
        (p("ple.eos_token_id"), Kv::U32(7)),
        (
            p("embedding_length_per_layer_input"),
            Kv::U32(PLE_DIM as u32),
        ),
        (p("ple.layer_multipliers"), Kv::F32s(vec![1.0, 0.5, 0.25])),
        (p("ple.head_offsets"), Kv::U64s(vec![0, 10, 20, 30])),
        (p("ple.head_vocab_sizes"), Kv::U64s(vec![10, 10, 10, 10])),
    ]
}

/// The tensors of one block `il` of a qwen4exp GGUF, written out from llama.cpp
/// `load_block` (the unmerged HC / recurrent spellings on odd layers so both
/// spellings are covered).
fn block(f: &mut Fixture, il: u64, recurrent: bool, ple: bool) {
    let n = |s: &str| format!("blk.{il}.{s}");
    let hc_dim = HC * E;
    f.t(&n("hc_attn_norm.weight"), &[E, HC], F32);
    f.t(&n("hc_attn_up.weight"), &[HC_LR, hc_dim], F32);
    f.t(&n("hc_ffn_norm.weight"), &[E, HC], F32);
    f.t(&n("hc_ffn_up.weight"), &[HC_LR, hc_dim], F32);
    if il % 2 == 0 {
        f.t(&n("hc_attn_down_inject.weight"), &[hc_dim, HC_LR + HC], F32);
        f.t(&n("hc_ffn_down_inject.weight"), &[hc_dim, HC_LR + HC], F32);
    } else {
        f.t(&n("hc_attn_down.weight"), &[hc_dim, HC_LR], F32);
        f.t(&n("hc_attn_inject.weight"), &[hc_dim, HC], F32);
        f.t(&n("hc_ffn_down.weight"), &[hc_dim, HC_LR], F32);
        f.t(&n("hc_ffn_inject.weight"), &[hc_dim, HC], F32);
    }
    if recurrent {
        let key_dim = D_STATE * N_GROUP;
        let value_dim = D_STATE * DT_RANK;
        let conv_dim = key_dim * 2 + value_dim;
        if il % 2 == 0 {
            f.t(&n("attn_qkv_gate.weight"), &[E, conv_dim + value_dim], Q2_K);
            f.t(&n("ssm_beta_alpha.weight"), &[E, 2 * DT_RANK], F32);
        } else {
            f.t(&n("attn_qkv.weight"), &[E, conv_dim], Q2_K);
            f.t(&n("attn_gate.weight"), &[E, value_dim], Q2_K);
            f.t(&n("ssm_beta.weight"), &[E, DT_RANK], F32);
            f.t(&n("ssm_alpha.weight"), &[E, DT_RANK], F32);
        }
        f.t(&n("ssm_conv1d.weight"), &[D_CONV, conv_dim], F32);
        f.t(&n("ssm_dt.bias"), &[DT_RANK], F32);
        f.t(&n("ssm_a"), &[DT_RANK], F32);
        f.t(&n("ssm_norm.weight"), &[D_STATE], F32);
        f.t(&n("ssm_out.weight"), &[value_dim, E], Q2_K);
    } else {
        f.t(&n("attn_q.weight"), &[E, HEAD_DIM * HEADS * 2], Q2_K);
        f.t(&n("attn_k.weight"), &[E, HEAD_DIM * KV_HEADS], Q2_K);
        f.t(&n("attn_v.weight"), &[E, HEAD_DIM * KV_HEADS], Q2_K);
        f.t(&n("attn_output.weight"), &[HEAD_DIM * HEADS, E], Q2_K);
        f.t(&n("attn_q_norm.weight"), &[HEAD_DIM], F32);
        f.t(&n("attn_k_norm.weight"), &[HEAD_DIM], F32);
        f.t(&n("indexer.q_proj.weight"), &[E, IDX_HEADS * IDX_DIM], Q2_K);
        f.t(&n("indexer.k_proj.weight"), &[E, IDX_DIM], Q2_K);
        f.t(&n("indexer.q_norm.weight"), &[IDX_DIM], F32);
        f.t(&n("indexer.k_norm.weight"), &[IDX_DIM], F32);
    }
    if ple {
        f.t(&n("ple_key.weight"), &[E, hc_dim], F32);
        f.t(&n("ple_value.weight"), &[E, E], F32);
        f.t(&n("ple_norm_key.weight"), &[E, HC], F32);
        f.t(&n("ple_norm_query.weight"), &[E, HC], F32);
        f.t(&n("ple_norm_conv.weight"), &[E, HC], F32);
        f.t(&n("ple_conv1d.weight"), &[4, hc_dim], F32);
    }
    f.t(&n("ffn_gate_inp.weight"), &[E, N_EXPERT], F32);
    f.t(&n("ffn_down_exps.weight"), &[FF_EXP, E, N_EXPERT], Q2_K);
    if il % 2 == 0 {
        f.t(
            &n("ffn_gate_up_exps.weight"),
            &[E, 2 * FF_EXP, N_EXPERT],
            Q2_K,
        );
    } else {
        f.t(&n("ffn_gate_exps.weight"), &[E, FF_EXP, N_EXPERT], Q2_K);
        f.t(&n("ffn_up_exps.weight"), &[E, FF_EXP, N_EXPERT], Q2_K);
    }
    f.t(&n("ffn_gate_inp_shexp.weight"), &[E], F32);
    f.t(&n("ffn_gate_shexp.weight"), &[E, FF_SH], Q2_K);
    f.t(&n("ffn_up_shexp.weight"), &[E, FF_SH], Q2_K);
    f.t(&n("ffn_down_shexp.weight"), &[FF_SH, E], Q2_K);
}

fn mtp(f: &mut Fixture, il: u64) {
    let n = |s: &str| format!("blk.{il}.{s}");
    let hc_dim = HC * E;
    f.t(&n("nextn.eh_proj.weight"), &[2 * E, E], Q2_K);
    f.t(&n("nextn.enorm.weight"), &[E], F32);
    f.t(&n("nextn.hnorm.weight"), &[E, HC], F32);
    f.t(&n("nextn.hc_head_norm.weight"), &[E, HC], F32);
    f.t(&n("nextn.hc_head_down.weight"), &[hc_dim, HC_LR], F32);
    f.t(&n("nextn.hc_head_up.weight"), &[HC_LR, hc_dim], F32);
}

/// A full single-file qwen4exp header: every global, 4 trunk blocks, 1 MTP block.
fn fixture() -> Fixture {
    let mut f = Fixture {
        kvs: hparams(),
        tensors: Vec::new(),
    };
    let hc_dim = HC * E;
    f.t("token_embd.weight", &[E, V], Q2_K);
    f.t("output.weight", &[E, V], Q2_K);
    f.t("output_hc_norm.weight", &[E, HC], F32);
    f.t("output_hc_down.weight", &[hc_dim, HC_LR], F32);
    f.t("output_hc_up.weight", &[HC_LR, hc_dim], F32);
    f.t("per_layer_token_embd.weight", &[PLE_DIM, PLE_ROWS], IQ4_NL);
    f.t("per_layer_token_embd.scale", &[1], F32);
    for il in 0..4 {
        block(&mut f, il, il != 3, il == 1);
    }
    block(&mut f, 4, false, false);
    mtp(&mut f, 4);
    f
}

fn meta_of(f: &Fixture) -> Qwen4ExpGgufMeta {
    Qwen4ExpGgufMeta::from_model(&f.model()).expect("fixture metadata reads")
}

fn err_text<T: std::fmt::Debug>(r: crate::error::Result<T>) -> String {
    format!("{}", r.expect_err("expected an error"))
}

// ---------------------------------------------------------------------------
// Architecture detection and metadata
// ---------------------------------------------------------------------------

#[test]
fn gguf_qwen4exp_arch_is_detected_from_general_architecture() {
    let f = fixture();
    assert!(is_qwen4exp_gguf(&f.model()));
    let mut other = fixture();
    other.kv("general.architecture", Kv::Str("qwen35"));
    assert!(!is_qwen4exp_gguf(&other.model()));
}

#[test]
fn gguf_qwen4exp_metadata_reads_arch_prefixed_keys() {
    let m = meta_of(&fixture());
    assert_eq!(m.block_count, 5);
    assert_eq!(m.nextn_predict_layers, 1);
    assert_eq!(m.n_layer, 4, "n_layer = block_count - nextn_predict_layers");
    assert_eq!(m.n_embd, E);
    assert_eq!(m.n_vocab, V, "n_vocab is ne1 of token_embd.weight");
    assert_eq!(m.n_head, HEADS);
    assert_eq!(m.n_head_kv, KV_HEADS);
    assert_eq!(m.head_dim, HEAD_DIM);
    assert_eq!(m.n_expert, N_EXPERT);
    assert_eq!(m.n_ff_exp, FF_EXP);
    assert_eq!(m.n_ff_shexp, FF_SH);
    assert_eq!(m.hc_count, HC);
    assert_eq!(m.hc_low_rank, HC_LR);
    assert_eq!(m.ple_layers, vec![1]);
    assert_eq!(m.ple_n_heads, 4, "(ngram_size - 1) * heads_per_ngram");
    assert_eq!(m.ple_min_rows, 40, "max(offset + vocab_size)");
}

#[test]
fn gguf_qwen4exp_layer_kinds_follow_full_attention_interval() {
    let m = meta_of(&fixture());
    let kinds: Vec<bool> = (0..5).map(|i| m.is_recurrent(i)).collect();
    assert_eq!(
        kinds,
        vec![true, true, true, false, false],
        "MTP layer 4 is full attention"
    );
    assert!(m.is_mtp(4) && !m.is_mtp(3));
}

#[test]
fn gguf_qwen4exp_recurrent_layers_array_overrides_interval() {
    let mut f = fixture();
    f.kv(
        "qwen4exp.attention.recurrent_layers",
        Kv::Bools(vec![true, true, true, false, false]),
    );
    f.drop_kv("qwen4exp.full_attention_interval");
    let m = meta_of(&f);
    assert!(m.is_recurrent(0) && !m.is_recurrent(3));
}

#[test]
fn gguf_qwen4exp_missing_required_key_is_named_not_defaulted() {
    for key in [
        "qwen4exp.block_count",
        "qwen4exp.embedding_length",
        "qwen4exp.hyper_connection.count",
        "qwen4exp.ssm.state_size",
        "qwen4exp.attention.indexer.head_count",
        "qwen4exp.embedding_length_per_layer_input",
    ] {
        let mut f = fixture();
        f.drop_kv(key);
        let msg = err_text(Qwen4ExpGgufMeta::from_model(&f.model()));
        assert!(
            msg.contains(key),
            "error for missing {key} must name it: {msg}"
        );
    }
}

#[test]
fn gguf_qwen4exp_hyper_connection_count_one_is_refused() {
    let mut f = fixture();
    f.kv("qwen4exp.hyper_connection.count", Kv::U32(1));
    let msg = err_text(Qwen4ExpGgufMeta::from_model(&f.model()));
    assert!(msg.contains("hyper_connection.count"), "{msg}");
}

#[test]
fn gguf_qwen4exp_ple_layer_must_be_a_linear_attention_layer() {
    let mut f = fixture();
    f.kv("qwen4exp.ple.layers", Kv::U32s(vec![3]));
    let msg = err_text(Qwen4ExpGgufMeta::from_model(&f.model()));
    assert!(msg.contains("PLE layer 3"), "{msg}");
}

#[test]
fn gguf_qwen4exp_ple_head_arrays_must_match_head_count() {
    let mut f = fixture();
    f.kv("qwen4exp.ple.head_offsets", Kv::U64s(vec![0, 10]));
    let msg = err_text(Qwen4ExpGgufMeta::from_model(&f.model()));
    assert!(msg.contains("ple.head_offsets"), "{msg}");
}

// ---------------------------------------------------------------------------
// Tensor-name mapping
// ---------------------------------------------------------------------------

#[test]
fn gguf_qwen4exp_every_fixture_tensor_is_mapped() {
    let f = fixture();
    let model = f.model();
    let m = meta_of(&f);
    let unmapped = unmapped_tensors(&model, &m);
    assert!(unmapped.is_empty(), "unmapped: {unmapped:?}");
    assert_eq!(model.tensors.len(), f.tensors.len());
}

#[test]
fn gguf_qwen4exp_tensor_families() {
    let m = meta_of(&fixture());
    let cases: &[(&str, Option<TensorFamily>)] = &[
        ("token_embd.weight", Some(TensorFamily::Embedding)),
        ("output.weight", Some(TensorFamily::Embedding)),
        ("output_hc_down.weight", Some(TensorFamily::HyperConnection)),
        (
            "per_layer_token_embd.weight",
            Some(TensorFamily::NgramEmbedding),
        ),
        (
            "per_layer_token_embd.scale",
            Some(TensorFamily::NgramEmbedding),
        ),
        (
            "blk.0.hc_attn_down_inject.weight",
            Some(TensorFamily::HyperConnection),
        ),
        (
            "blk.0.attn_qkv_gate.weight",
            Some(TensorFamily::LinearAttention),
        ),
        ("blk.1.attn_qkv.weight", Some(TensorFamily::LinearAttention)),
        ("blk.0.ssm_a", Some(TensorFamily::LinearAttention)),
        ("blk.0.ssm_dt.bias", Some(TensorFamily::LinearAttention)),
        ("blk.3.attn_q.weight", Some(TensorFamily::FullAttention)),
        ("blk.3.attn_qkv.weight", Some(TensorFamily::FullAttention)),
        ("blk.3.attn_q.bias", Some(TensorFamily::FullAttention)),
        ("blk.3.indexer.q_proj.weight", Some(TensorFamily::Indexer)),
        (
            "blk.1.ple_conv1d.weight",
            Some(TensorFamily::PerLayerEmbedding),
        ),
        ("blk.2.ffn_gate_up_exps.weight", Some(TensorFamily::Moe)),
        ("blk.2.ffn_gate_inp_shexp.weight", Some(TensorFamily::Moe)),
        ("blk.4.attn_q.weight", Some(TensorFamily::Mtp)),
        ("blk.4.nextn.eh_proj.weight", Some(TensorFamily::Mtp)),
        // Wrong layer kind: full-attention weights on a linear layer and
        // DeltaNet weights on a full layer are not part of the arch.
        ("blk.0.attn_q.weight", None),
        ("blk.3.ssm_a", None),
        ("blk.3.attn_gate.weight", None),
        // PLE tensors only on the declared PLE layer.
        ("blk.2.ple_key.weight", None),
        // nextn only on MTP layers.
        ("blk.3.nextn.enorm.weight", None),
        // Out of range, other archs' spellings, vision.
        ("blk.5.attn_q.weight", None),
        ("output_norm.weight", None),
        ("blk.0.attn_norm.weight", None),
        ("blk.0.ffn_gate_exps.bias", None),
        ("v.blk.0.attn_q.weight", None),
        ("mm.input_projection.weight", None),
        ("blk.x.attn_q.weight", None),
    ];
    for (name, want) in cases {
        assert_eq!(map_gguf_tensor(name, &m), *want, "{name}");
    }
}

#[test]
fn gguf_qwen4exp_unknown_tensor_is_reported_by_name() {
    let mut f = fixture();
    f.t("blk.0.attn_norm.weight", &[E], F32);
    let m = meta_of(&f);
    assert_eq!(
        unmapped_tensors(&f.model(), &m),
        vec!["blk.0.attn_norm.weight".to_string()]
    );
}

// ---------------------------------------------------------------------------
// LAYOUT-001: ggml dims (ne0 first) and the row-major shape
// ---------------------------------------------------------------------------

#[test]
fn gguf_qwen4exp_shape_contract_holds_on_fixture() {
    let f = fixture();
    let m = meta_of(&f);
    check_shapes(&f.model(), &m).expect("fixture shapes match the contract");
}

#[test]
fn gguf_qwen4exp_row_major_shape_reverses_ggml_dims() {
    // token_embd ggml {n_embd, n_vocab} is row-major [n_vocab, n_embd].
    assert_eq!(row_major_shape(&[E, V]), vec![V as usize, E as usize]);
    assert_eq!(
        row_major_shape(&[FF_EXP, E, N_EXPERT]),
        vec![N_EXPERT as usize, E as usize, FF_EXP as usize]
    );
    // The parsed TensorInfo already carries the row-major shape.
    let model = fixture().model();
    let embd = model
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight")
        .expect("embd");
    assert_eq!(embd.dims, vec![V, E]);
}

#[test]
fn gguf_qwen4exp_transposed_tensor_is_refused_by_name() {
    for (name, bad) in [
        ("token_embd.weight", vec![V, E]),
        ("blk.3.attn_q.weight", vec![E, HEAD_DIM * HEADS]),
        // attn_output and ssm_out are square in this fixture (E = 8 =
        // HEAD_DIM*HEADS = D_STATE*DT_RANK): a transpose of a square matrix is
        // invisible to a shape contract, so non-square tensors stand in.
        ("blk.3.attn_k.weight", vec![HEAD_DIM * KV_HEADS, E]),
        ("blk.0.ffn_down_exps.weight", vec![E, FF_EXP, N_EXPERT]),
        ("blk.1.ffn_gate_inp.weight", vec![N_EXPERT, E]),
        (
            "blk.1.ssm_conv1d.weight",
            vec![(2 * D_STATE * N_GROUP) + D_STATE * DT_RANK, D_CONV],
        ),
        ("blk.0.hc_attn_up.weight", vec![HC * E, HC_LR]),
        ("per_layer_token_embd.weight", vec![PLE_ROWS, PLE_DIM]),
    ] {
        let mut f = fixture();
        f.reshape(name, &bad);
        let m = meta_of(&f);
        let msg = err_text(check_shapes(&f.model(), &m));
        assert!(msg.contains(name), "{name}: {msg}");
    }
}

#[test]
fn gguf_qwen4exp_missing_required_tensor_is_refused_by_name() {
    let mut f = fixture();
    f.tensors.retain(|t| t.0 != "blk.2.ssm_out.weight");
    let m = meta_of(&f);
    let msg = err_text(check_shapes(&f.model(), &m));
    assert!(msg.contains("blk.2.ssm_out.weight"), "{msg}");
}

#[test]
fn gguf_qwen4exp_engram_with_too_few_rows_is_refused() {
    let mut f = fixture();
    f.reshape("per_layer_token_embd.weight", &[PLE_DIM, 39]);
    let m = meta_of(&f);
    let msg = err_text(check_shapes(&f.model(), &m));
    assert!(msg.contains("per_layer_token_embd.weight"), "{msg}");
}

// ---------------------------------------------------------------------------
// Split GGUF: the engram in the last shard
// ---------------------------------------------------------------------------

fn split_kvs(f: &mut Fixture, no: u16, count: u16, total: i32) {
    f.kv("split.no", Kv::U16(no));
    f.kv("split.count", Kv::U16(count));
    f.kv("split.tensors.count", Kv::I32(total));
}

/// Two shards in llama.cpp `gguf-split` form: shard 0 carries the arch
/// metadata and every tensor but the engram; shard 1 carries only the split
/// keys and `per_layer_token_embd.weight` (IQ4_NL).
fn split_pair() -> (Fixture, Fixture) {
    let full = fixture();
    let total = full.tensors.len() as i32;
    let mut first = Fixture {
        kvs: full.kvs,
        tensors: Vec::new(),
    };
    let mut last = Fixture {
        kvs: Vec::new(),
        tensors: Vec::new(),
    };
    for t in full.tensors {
        if t.0 == "per_layer_token_embd.weight" {
            last.tensors.push(t);
        } else {
            first.tensors.push(t);
        }
    }
    split_kvs(&mut first, 0, 2, total);
    split_kvs(&mut last, 1, 2, total);
    (first, last)
}

#[test]
fn gguf_qwen4exp_split_set_finds_engram_in_last_shard() {
    let (a, b) = split_pair();
    let shards = vec![a.model(), b.model()];
    let set = validate_split_set(&shards).expect("split set validates");
    assert_eq!(set.engram_shard, 1);
    assert_eq!(set.total_tensors, a.tensors.len() + b.tensors.len());
    assert_eq!(set.engram.qtype, IQ4_NL);
    // TensorInfo dims are row-major (the parser reverses ggml order).
    assert_eq!(set.engram.dims, vec![PLE_ROWS, PLE_DIM]);
    // The arch metadata is read from shard 0 and every tensor maps.
    assert!(set.unmapped.is_empty(), "{:?}", set.unmapped);
}

#[test]
fn gguf_qwen4exp_split_set_accepts_any_shard_order() {
    let (a, b) = split_pair();
    let shards = vec![b.model(), a.model()];
    let set = validate_split_set(&shards).expect("order-independent");
    assert_eq!(
        set.engram_shard, 1,
        "reported by split.no, not slice position"
    );
}

#[test]
fn gguf_qwen4exp_split_set_missing_shard_is_refused() {
    let (a, _) = split_pair();
    let msg = err_text(validate_split_set(&[a.model()]));
    assert!(msg.contains("split.count"), "{msg}");
}

#[test]
fn gguf_qwen4exp_split_set_tensor_count_mismatch_is_refused() {
    let (mut a, b) = split_pair();
    split_kvs(&mut a, 0, 2, 9999);
    let msg = err_text(validate_split_set(&[a.model(), b.model()]));
    assert!(msg.contains("split.tensors.count"), "{msg}");
}

#[test]
fn gguf_qwen4exp_split_set_without_engram_is_refused() {
    let (a, mut b) = split_pair();
    b.tensors.clear();
    b.t("blk.0.ssm_a.extra", &[1], F32);
    let msg = err_text(validate_split_set(&[a.model(), b.model()]));
    assert!(msg.contains("per_layer_token_embd.weight"), "{msg}");
}

#[test]
fn gguf_qwen4exp_split_set_duplicate_tensor_is_refused() {
    let (a, mut b) = split_pair();
    b.t("token_embd.weight", &[E, V], Q2_K);
    let total = (a.tensors.len() + b.tensors.len()) as i32;
    let mut a = a;
    split_kvs(&mut a, 0, 2, total);
    split_kvs(&mut b, 1, 2, total);
    let msg = err_text(validate_split_set(&[a.model(), b.model()]));
    assert!(msg.contains("token_embd.weight"), "{msg}");
}

// ---------------------------------------------------------------------------
// Refusals by name: the vision tower (mmproj) and the MTP head
// ---------------------------------------------------------------------------

fn mmproj() -> Fixture {
    let mut f = Fixture {
        kvs: vec![("general.architecture".into(), Kv::Str("clip"))],
        tensors: Vec::new(),
    };
    f.t("v.patch_embd.weight", &[16, 16, 3, 32], F32);
    f.t("v.blk.0.attn_q.weight", &[32, 32], F32);
    f.t("mm.input_projection.weight", &[32, E], F32);
    f
}

#[test]
fn gguf_qwen4exp_vision_mmproj_file_is_refused_by_name() {
    let model = mmproj().model();
    let msg = err_text(refuse_non_text_file(
        std::path::Path::new("Qwen3.8-Flash-Next-GGUF/mmproj-F16.gguf"),
        &model,
    ));
    assert!(msg.contains("vision tower"), "{msg}");
    assert!(msg.contains("mmproj"), "{msg}");
}

#[test]
fn gguf_qwen4exp_vision_tensors_are_refused_even_under_a_text_name() {
    let mut f = fixture();
    f.t("v.blk.0.attn_q.weight", &[32, 32], F32);
    let msg = err_text(refuse_non_text_file(
        std::path::Path::new("model-00001-of-00002.gguf"),
        &f.model(),
    ));
    assert!(msg.contains("vision tower"), "{msg}");
    assert!(msg.contains("v.blk.0.attn_q.weight"), "{msg}");
}

#[test]
fn gguf_qwen4exp_mtp_dir_is_refused_by_name() {
    let f = fixture();
    for p in [
        "Qwen3.8-Flash-Next-GGUF/MTP/model.gguf",
        "Qwen3.8-Flash-Next-GGUF/mtp/model.gguf",
    ] {
        let msg = err_text(refuse_non_text_file(std::path::Path::new(p), &f.model()));
        assert!(msg.contains("MTP"), "{p}: {msg}");
    }
}

#[test]
fn gguf_qwen4exp_text_file_passes_the_refusals() {
    let f = fixture();
    refuse_non_text_file(
        std::path::Path::new("Qwen3.8-Flash-Next-GGUF/UD-Q2_K_XL/model-00001-of-00002.gguf"),
        &f.model(),
    )
    .expect("text shard is not refused");
}

#[test]
fn gguf_qwen4exp_mtp_tensors_are_split_out_of_the_trunk() {
    let f = fixture();
    let m = meta_of(&f);
    let mtp = mtp_tensors(&f.model(), &m);
    assert!(!mtp.is_empty());
    assert!(mtp.iter().all(|n| n.starts_with("blk.4.")), "{mtp:?}");
    assert!(mtp.contains(&"blk.4.nextn.eh_proj.weight".to_string()));
}

// ---------------------------------------------------------------------------
// Dequant coverage: UD-IQ1_S / UD-IQ2_XS have no dequantizer here
// ---------------------------------------------------------------------------

#[test]
fn gguf_qwen4exp_q2k_and_iq4nl_mix_has_full_dequant_coverage() {
    let f = fixture();
    assert!(dequant_gaps(&f.model()).is_empty());
}

#[test]
fn gguf_qwen4exp_iq1s_and_iq2xs_are_reported_as_dequant_gaps() {
    let mut f = fixture();
    f.retype("blk.0.ffn_down_exps.weight", IQ1_S);
    f.retype("blk.1.ffn_gate_exps.weight", IQ2_XS);
    let gaps = dequant_gaps(&f.model());
    let names: Vec<(&str, &str)> = gaps
        .iter()
        .map(|g| (g.tensor.as_str(), g.type_name))
        .collect();
    assert_eq!(
        names,
        vec![
            ("blk.0.ffn_down_exps.weight", "IQ1_S"),
            ("blk.1.ffn_gate_exps.weight", "IQ2_XS")
        ]
    );
}
