//! Tests for the EmbeddingGemma 2 CPU forward (APR-EMBED-001 EG-2, #4901).
//!
//! A synthetic F32 `gemma-embedding2` file small enough to run in a unit test: hidden 8,
//! per-layer input 4, two layers (0 local with head width 2, 1 global with head width 4),
//! two query heads over one KV head, FFN 8, output 6, vocabulary 5, window 2.
//!
//! Tests cover:
//! - The output: unit length, `output_dim()` wide, deterministic, order-sensitive (RoPE)
//! - The fixture's embeddings span more than a plane, so the reach test has room to see a move
//! - The fixture holds each weight's shape as a real file does, `[out, in]` once read
//! - Every tensor the graph loads reaches the embedding
//! - The last layer's output scale cancels in `output_norm` (the scale sits before it)
//! - Refusals: architecture, pooling, causal, RoPE scaling, shared KV, tensor count,
//!   a missing tensor, wrong dims, the sliding-window pattern, head widths, inputs
//! - Token rows of a quantized table, and a row that is not whole blocks
//! - Matryoshka truncation

use super::*;
use crate::gguf::test_factory::GGUFBuilder;

const ARCH: &str = "gemma-embedding2";
const H: usize = 8;
const P: usize = 4;
const LAYERS: usize = 2;
const HEADS: usize = 2;
const FF: usize = 8;
const OUT: usize = 6;
const VOCAB: usize = 5;
/// Head width of layer 0 (local) and layer 1 (global).
const HEAD_DIMS: [usize; LAYERS] = [2, 4];
const TOKENS: [u32; 4] = [1, 2, 3, 4];

#[derive(Debug, Clone)]
enum Meta {
    U32(u32),
    F32(f32),
    Bool(bool),
    Str(&'static str),
    I32s(Vec<i32>),
    Bools(Vec<bool>),
}

/// The valid synthetic file, with knobs a refusal test turns.
#[derive(Debug, Clone)]
struct Fixture {
    arch: &'static str,
    meta: Vec<(String, Meta)>,
    tensors: Vec<(String, Vec<u64>, Vec<f32>)>,
}

/// Deterministic values in `[-scale, scale)`: no RNG, so a failure reproduces exactly. An
/// integer hash of the index, not a sinusoid of it: `sin(a·i + b)` along a matrix's flattened
/// index is `sin(A_row + B_col)`, which is rank 2, and with every weight rank 2 the whole
/// fixture embedded onto one circle, where a weight could move the vector by less than the
/// cosine can resolve.
fn ramp(n: usize, seed: f32, scale: f32) -> Vec<f32> {
    (0u32..)
        .take(n)
        .map(|i| {
            let k = (i ^ seed.to_bits()).wrapping_mul(0x9E37_79B9);
            let k = (k ^ (k >> 16)).wrapping_mul(0x85EB_CA6B);
            let k = k ^ (k >> 13);
            ((k >> 8) as f32 / 16_777_216.0 * 2.0 - 1.0) * scale
        })
        .collect()
}

/// A norm weight near 1.
fn norm(n: usize, seed: f32) -> Vec<f32> {
    ramp(n, seed, 0.1).into_iter().map(|v| v + 1.0).collect()
}

impl Fixture {
    fn new() -> Self {
        let mut f = Self {
            arch: ARCH,
            meta: Vec::new(),
            tensors: Vec::new(),
        };
        for (suffix, value) in [
            ("embedding_length", Meta::U32(H as u32)),
            ("embedding_length_per_layer_input", Meta::U32(P as u32)),
            ("embedding_length_out", Meta::U32(OUT as u32)),
            ("block_count", Meta::U32(LAYERS as u32)),
            ("feed_forward_length", Meta::U32(FF as u32)),
            ("context_length", Meta::U32(16)),
            ("pooling_type", Meta::U32(1)),
            ("attention.head_count", Meta::U32(HEADS as u32)),
            ("attention.head_count_kv", Meta::I32s(vec![1, 1])),
            ("attention.causal", Meta::Bool(false)),
            ("attention.shared_kv_layers", Meta::U32(0)),
            ("attention.layer_norm_rms_epsilon", Meta::F32(1e-6)),
            ("attention.sliding_window", Meta::U32(2)),
            (
                "attention.sliding_window_pattern",
                Meta::Bools(vec![true, false]),
            ),
            ("attention.key_length", Meta::U32(4)),
            ("attention.value_length", Meta::U32(4)),
            ("attention.key_length_swa", Meta::U32(2)),
            ("attention.value_length_swa", Meta::U32(2)),
            ("rope.dimension_count", Meta::U32(4)),
            ("rope.dimension_count_swa", Meta::U32(2)),
            ("rope.freq_base", Meta::F32(1e6)),
            ("rope.freq_base_swa", Meta::F32(1e4)),
        ] {
            f = f.set(suffix, value);
        }
        let mut seed = 0.0;
        let mut next = || {
            seed += 1.7;
            seed
        };
        f = f
            .weight("token_embd.weight", H, VOCAB, next())
            .weight("per_layer_model_proj.weight", H, P * LAYERS, next())
            .vector("per_layer_proj_norm.weight", P, next())
            .vector("output_norm.weight", H, next())
            .weight("output.weight", H, OUT, next());
        for (i, d) in HEAD_DIMS.into_iter().enumerate() {
            let blk = |t: &str| format!("blk.{i}.{t}.weight");
            f = f
                .vector(&blk("attn_norm"), H, next())
                .weight(&blk("attn_q"), H, HEADS * d, next())
                .weight(&blk("attn_k"), H, d, next())
                .weight(&blk("attn_v"), H, d, next())
                .weight(&blk("attn_output"), HEADS * d, H, next())
                .vector(&blk("attn_q_norm"), d, next())
                .vector(&blk("attn_k_norm"), d, next())
                .vector(&blk("post_attention_norm"), H, next())
                .vector(&blk("ffn_norm"), H, next())
                .weight(&blk("ffn_gate"), H, FF, next())
                .weight(&blk("ffn_up"), H, FF, next())
                .weight(&blk("ffn_down"), FF, H, next())
                .vector(&blk("post_ffw_norm"), H, next())
                .weight(&blk("inp_gate"), H, P, next())
                .weight(&blk("proj"), P, H, next())
                .vector(&blk("post_norm"), H, next())
                .tensor(&blk("layer_output_scale"), &[1], vec![0.9]);
        }
        f
    }

    fn set(mut self, suffix: &str, value: Meta) -> Self {
        let key = format!("{ARCH}.{suffix}");
        self.meta.retain(|(k, _)| *k != key);
        self.meta.push((key, value));
        self
    }

    fn unset(mut self, suffix: &str) -> Self {
        let key = format!("{ARCH}.{suffix}");
        self.meta.retain(|(k, _)| *k != key);
        self
    }

    fn tensor(mut self, name: &str, dims: &[u64], data: Vec<f32>) -> Self {
        self.tensors.retain(|(n, _, _)| n != name);
        self.tensors.push((name.to_string(), dims.to_vec(), data));
        self
    }

    /// `out_dim` rows of `in_dim`. The builder takes the shape row-major, `[out_dim, in_dim]`,
    /// and writes it reversed, as GGUF's `ne0, ne1`, which is how a real file holds it.
    fn weight(self, name: &str, in_dim: usize, out_dim: usize, seed: f32) -> Self {
        let dims = [out_dim as u64, in_dim as u64];
        self.tensor(name, &dims, ramp(in_dim * out_dim, seed, 0.3))
    }

    fn vector(self, name: &str, len: usize, seed: f32) -> Self {
        self.tensor(name, &[len as u64], norm(len, seed))
    }

    fn rename(mut self, from: &str, to: &str) -> Self {
        for t in &mut self.tensors {
            if t.0 == from {
                t.0 = to.to_string();
            }
        }
        self
    }

    fn bytes(&self) -> Vec<u8> {
        self.builder().build()
    }

    fn builder(&self) -> GGUFBuilder {
        let mut b = GGUFBuilder::new().architecture(self.arch);
        for (key, value) in &self.meta {
            b = match value {
                Meta::U32(v) => b.add_u32(key, *v),
                Meta::F32(v) => b.add_f32(key, *v),
                Meta::Bool(v) => b.add_bool(key, *v),
                Meta::Str(v) => b.add_string(key, v),
                Meta::I32s(v) => b.add_i32_array(key, v),
                Meta::Bools(v) => b.add_bool_array(key, v),
            };
        }
        for (name, dims, data) in &self.tensors {
            b = b.add_f32_tensor(name, dims, data);
        }
        b
    }

    fn load(&self) -> Result<EmbeddingGemma2> {
        let data = self.bytes();
        let model = GGUFModel::from_bytes(&data)?;
        EmbeddingGemma2::from_gguf(&model, &data)
    }

    fn embed(&self) -> Vec<f32> {
        self.load()
            .expect("the fixture loads")
            .embed(&TOKENS)
            .expect("the fixture embeds")
    }

    /// The refusal reason, which must name `needle`.
    fn refused(&self, needle: &str) {
        let err = self.load().expect_err("the file must be refused");
        let text = err.to_string();
        assert!(
            text.contains(needle),
            "refusal {text:?} does not name {needle:?}"
        );
    }
}

/// The vocabulary of [`served_fixture`]: `embedding_gemma2_input`'s test vocabulary, the four
/// specials at the ids the real files use, then pieces.
const SERVED_TOKENS: [&str; 10] = [
    "<pad>",
    "<eos>",
    "<bos>",
    "<unk>",
    "a",
    "b",
    "ab",
    "\u{2581}",
    "\u{2581}a",
    "\u{2581}ab",
];

/// A whole file `apr serve` and `apr embed` can load (APR-EMBED-001 EG-3, #4902): this
/// module's forward with an `out`-wide projection and a token table as long as
/// [`SERVED_TOKENS`], plus the SPM tokenizer metadata the input layer reads.
pub(crate) fn served_fixture(out: usize) -> Vec<u8> {
    let types: Vec<i32> = SERVED_TOKENS
        .iter()
        .map(|t| if t.starts_with('<') { 3 } else { 1 })
        .collect();
    Fixture::new()
        .set("embedding_length_out", Meta::U32(out as u32))
        .weight("token_embd.weight", H, SERVED_TOKENS.len(), 11.3)
        .weight("output.weight", H, out, 12.7)
        .builder()
        .add_string("tokenizer.ggml.model", "gemma4")
        .add_string_array("tokenizer.ggml.tokens", &SERVED_TOKENS)
        .add_i32_array("tokenizer.ggml.token_type", &types)
        .add_string_array(
            "tokenizer.ggml.merges",
            &["\u{2581} a", "a b", "\u{2581}a b"],
        )
        .add_bool("tokenizer.ggml.add_bos_token", true)
        .add_bool("tokenizer.ggml.add_eos_token", true)
        .add_u32("tokenizer.ggml.bos_token_id", 2)
        .add_u32("tokenizer.ggml.eos_token_id", 1)
        .build()
}

fn length(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>() / (length(a) * length(b))
}

// ============================================================================
// The output
// ============================================================================

#[test]
fn embeds_to_a_unit_vector_of_the_output_width_eg2() {
    let model = Fixture::new().load().expect("the fixture loads");
    assert_eq!(model.output_dim(), OUT);
    assert_eq!(model.vocab_size(), VOCAB);
    let v = model.embed(&TOKENS).expect("embeds");
    assert_eq!(v.len(), OUT);
    assert!(v.iter().all(|x| x.is_finite()), "{v:?}");
    assert!((length(&v) - 1.0).abs() < 1e-5, "length {}", length(&v));
    // A single token is a valid input too.
    assert_eq!(model.embed(&[0]).expect("one token").len(), OUT);
}

#[test]
fn the_same_tokens_give_the_same_vector_eg2() {
    let model = Fixture::new().load().expect("the fixture loads");
    assert_eq!(
        model.embed(&TOKENS).expect("embeds"),
        model.embed(&TOKENS).expect("embeds")
    );
}

/// Mean pooling over a non-causal stack is order-blind without positions; RoPE is what makes
/// the order count.
#[test]
fn token_order_changes_the_vector_eg2() {
    let model = Fixture::new().load().expect("the fixture loads");
    let forward = model.embed(&[1, 2, 3, 4]).expect("embeds");
    let reversed = model.embed(&[4, 3, 2, 1]).expect("embeds");
    assert!(
        cosine(&forward, &reversed) < 0.999_99,
        "order ignored: {forward:?}"
    );
}

/// The reach test below needs room to move: three embeddings of the fixture must span three
/// dimensions. With rank-2 weights they lay in one plane and the Gram determinant was 0.
#[test]
fn the_fixture_embeds_beyond_a_plane_eg2() {
    let model = Fixture::new().load().expect("the fixture loads");
    let inputs: [&[u32]; 3] = [&[1, 2, 3, 4], &[4, 3, 2, 1], &[0, 2]];
    let [a, b, c] = inputs.map(|t| model.embed(t).expect("embeds"));
    let dot = |x: &[f32], y: &[f32]| -> f64 {
        x.iter()
            .zip(y)
            .map(|(p, q)| f64::from(*p) * f64::from(*q))
            .sum()
    };
    let (ab, bc, ca) = (dot(&a, &b), dot(&b, &c), dot(&c, &a));
    let gram = dot(&a, &a) * dot(&b, &b) * dot(&c, &c) + 2.0 * ab * bc * ca
        - ab * ab * dot(&c, &c)
        - bc * bc * dot(&a, &a)
        - ca * ca * dot(&b, &b);
    assert!(gram > 1e-4, "coplanar embeddings: Gram determinant {gram}");
}

/// A weight that is loaded but never applied would survive this: each tensor is perturbed in
/// turn, non-uniformly, and the vector must move.
#[test]
fn every_tensor_reaches_the_embedding_eg2() {
    let base = Fixture::new();
    let reference = base.embed();
    // The last layer's output scale multiplies the whole stream just before `output_norm`,
    // which divides it back out; the next test pins that.
    let cancels = format!("blk.{}.layer_output_scale.weight", LAYERS - 1);
    let unreached: Vec<String> = base
        .tensors
        .iter()
        .filter(|(name, _, _)| *name != cancels)
        .filter_map(|(name, dims, data)| {
            let moved: Vec<f32> = data
                .iter()
                .enumerate()
                .map(|(i, v)| v + 0.3 * (i as f32 * 1.3).cos() + 0.2)
                .collect();
            let c = cosine(&reference, &base.clone().tensor(name, dims, moved).embed());
            (c >= 0.999_999).then(|| format!("{name} (cosine {c})"))
        })
        .collect();
    assert!(
        unreached.is_empty(),
        "did not reach the embedding: {unreached:?}"
    );
}

#[test]
fn the_last_layer_scale_cancels_in_the_output_norm_eg2() {
    let base = Fixture::new();
    let name = format!("blk.{}.layer_output_scale.weight", LAYERS - 1);
    let scaled = base.clone().tensor(&name, &[1], vec![3.0]).embed();
    assert!(cosine(&base.embed(), &scaled) > 0.999_99);
    // The first layer's scale does not cancel: layer 1 adds its branches to the scaled stream.
    let first = base
        .clone()
        .tensor("blk.0.layer_output_scale.weight", &[1], vec![3.0])
        .embed();
    assert!(cosine(&base.embed(), &first) < 0.999_99);
}

// ============================================================================
// Refusals: the file
// ============================================================================

#[test]
fn another_architecture_is_refused_eg2() {
    let mut f = Fixture::new();
    f.arch = "gemma3";
    f.refused("is not gemma-embedding2");
}

#[test]
fn a_pooling_type_other_than_mean_is_refused_eg2() {
    Fixture::new()
        .set("pooling_type", Meta::U32(2))
        .refused("pooling_type 2");
    Fixture::new()
        .unset("pooling_type")
        .refused("pooling_type missing");
}

#[test]
fn a_causal_flag_is_refused_eg2() {
    Fixture::new()
        .set("attention.causal", Meta::Bool(true))
        .refused("causal");
    // Absent means not causal, as llama.cpp reads it.
    assert!(Fixture::new().unset("attention.causal").load().is_ok());
}

#[test]
fn rope_scaling_and_shared_kv_are_refused_eg2() {
    Fixture::new()
        .set("rope.scaling.type", Meta::Str("yarn"))
        .refused("rope.scaling.type");
    Fixture::new()
        .set("attention.shared_kv_layers", Meta::U32(1))
        .refused("shared_kv_layers");
    assert!(Fixture::new()
        .unset("attention.shared_kv_layers")
        .load()
        .is_ok());
}

#[test]
fn a_tensor_count_other_than_the_graph_is_refused_eg2() {
    Fixture::new()
        .tensor("extra.weight", &[H as u64], norm(H, 0.0))
        .refused("the graph consumes 39");
    Fixture::new()
        .rename("blk.1.proj.weight", "blk.1.projx.weight")
        .refused("blk.1.proj.weight missing");
}

/// The reader reverses GGUF's `ne0, ne1`, so a weight reads back as `[out, in]`: `token_embd` as
/// `[vocab, hidden]`, where the generative config reads its vocabulary (C-13), and `attn_q` as
/// `[heads × head width, hidden]`. A fixture written the other way round once loaded a forward that
/// read the other way round, and only the real file showed it.
#[test]
fn the_fixture_holds_shapes_as_a_real_file_does_eg2() {
    let data = Fixture::new().bytes();
    let model = GGUFModel::from_bytes(&data).expect("parses");
    let dims = |name: &str| tensor_dims(&model, name).expect("the fixture has it");
    assert_eq!(dims("token_embd.weight"), [VOCAB, H]);
    for (i, d) in HEAD_DIMS.into_iter().enumerate() {
        assert_eq!(dims(&format!("blk.{i}.attn_q.weight")), [HEADS * d, H]);
    }
}

#[test]
fn a_tensor_with_other_dims_is_refused_eg2() {
    Fixture::new()
        .weight("blk.0.ffn_up.weight", H, FF + 1, 0.5)
        .refused("blk.0.ffn_up.weight has dims [9, 8], expected [8, 8]");
    Fixture::new()
        .tensor(
            "output.weight",
            &[H as u64, OUT as u64],
            ramp(H * OUT, 0.5, 0.3),
        )
        .refused("output.weight has dims [8, 6], expected [6, 8]");
    Fixture::new()
        .tensor("blk.1.attn_q_norm.weight", &[2], norm(2, 0.5))
        .refused("blk.1.attn_q_norm.weight has dims [2], expected [4]");
}

#[test]
fn a_sliding_window_pattern_that_disagrees_is_refused_eg2() {
    let key = "attention.sliding_window_pattern";
    Fixture::new()
        .set(key, Meta::Bools(vec![false, true]))
        .refused("layer 0");
    Fixture::new()
        .set(key, Meta::Bools(vec![true]))
        .refused("1 entries for 2 layers");
    Fixture::new()
        .set(key, Meta::I32s(vec![1, 0]))
        .refused("non-bool");
    // Absent, llama.cpp's default makes both layers local; the widths say layer 1 is global.
    Fixture::new().unset(key).refused("layer 1");
}

#[test]
fn head_widths_that_do_not_fit_are_refused_eg2() {
    Fixture::new()
        .set("attention.value_length_swa", Meta::U32(3))
        .refused("value_length_swa 3");
    Fixture::new()
        .set("rope.dimension_count", Meta::U32(3))
        .refused("rope.dimension_count 3");
    Fixture::new()
        .set("rope.dimension_count_swa", Meta::U32(4))
        .refused("1..=2");
    Fixture::new()
        .set("attention.head_count_kv", Meta::I32s(vec![1, 3]))
        .refused("2 query heads over 3 KV heads");
    Fixture::new()
        .set("embedding_length_out", Meta::U32(0))
        .refused("a zero width");
}

// ============================================================================
// Refusals: the input
// ============================================================================

#[test]
fn inputs_the_model_cannot_embed_are_refused_eg2() {
    let model = Fixture::new().load().expect("the fixture loads");
    let empty = model.embed(&[]).expect_err("no tokens");
    assert!(empty.to_string().contains("no tokens"), "{empty}");
    let oov = model
        .embed(&[1, VOCAB as u32])
        .expect_err("out of vocabulary");
    assert!(oov.to_string().contains("outside the vocabulary"), "{oov}");
    let long = model.embed(&[1; 17]).expect_err("past the context");
    assert!(
        matches!(
            long,
            RealizarError::ContextLimitExceeded {
                provided: 17,
                maximum: 16
            }
        ),
        "{long:?}"
    );
    assert!(model.embed(&[1; 16]).is_ok(), "the full context fits");
}

// ============================================================================
// Token rows and block sizes
// ============================================================================

/// Two rows of one Q4_K block each: `d` = 1.0, `dmin` = 0, varied scales and nibbles.
fn q4_k_rows(rows: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(rows * 144);
    for r in 0..rows {
        bytes.extend_from_slice(&0x3C00u16.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend((0..12).map(|i| (i * 5 + r * 3 + 1) as u8 & 0x3F));
        bytes.extend((0..128).map(|i| (i * 7 + r * 11) as u8));
    }
    bytes
}

#[test]
fn a_quantized_token_row_matches_the_whole_table_eg2() {
    let data = GGUFBuilder::new()
        .architecture(ARCH)
        .add_q4_k_tensor("t.weight", &[2, 256], &q4_k_rows(2))
        .build();
    let model = GGUFModel::from_bytes(&data).expect("parses");
    let table = weight(&model, &data, "t.weight", 256, 2).expect("whole blocks");
    let all = model
        .get_tensor_f32("t.weight", &data)
        .expect("dequantizes");
    assert_eq!(table_row(&table, 0).expect("row 0"), all[..256]);
    assert_eq!(table_row(&table, 1).expect("row 1"), all[256..]);
    assert_ne!(
        all[..256],
        all[256..],
        "the rows must differ for the test to bite"
    );
    assert!(table_row(&table, 2).is_err(), "row 2 of 2");
}

#[test]
fn a_row_that_is_not_whole_blocks_is_refused_eg2() {
    let data = GGUFBuilder::new()
        .architecture(ARCH)
        .add_q4_k_tensor("t.weight", &[2, 128], &q4_k_rows(1))
        .build();
    let model = GGUFModel::from_bytes(&data).expect("parses");
    let err = weight(&model, &data, "t.weight", 128, 2).expect_err("half a block per row");
    assert!(
        err.to_string().contains("not whole Q4_K blocks of 256"),
        "{err}"
    );
}

#[test]
fn bool_arrays_round_trip_through_the_reader_eg2() {
    let data = GGUFBuilder::new()
        .architecture(ARCH)
        .add_bool_array("k", &[true, false, true])
        .build();
    let model = GGUFModel::from_bytes(&data).expect("parses");
    let want = GGUFValue::Array(vec![
        GGUFValue::Bool(true),
        GGUFValue::Bool(false),
        GGUFValue::Bool(true),
    ]);
    assert_eq!(model.metadata.get("k"), Some(&want));
}

// ============================================================================
// Matryoshka truncation and pooling
// ============================================================================

#[test]
fn truncation_keeps_the_prefix_direction_at_unit_length_eg2() {
    let v = unit_length(ramp(768, 0.5, 1.0)).expect("non-zero");
    for dim in MATRYOSHKA_DIMS {
        let t = truncate_to(&v, dim).expect("a trained width");
        assert_eq!(t.len(), dim);
        assert!((length(&t) - 1.0).abs() < 1e-5);
        assert!(cosine(&t, &v[..dim]) > 0.999_999);
    }
    // At the full width the vector is scaled by 1 / |v|, which is 1 to within rounding.
    let full = truncate_to(&v, 768).expect("full width");
    let moved = full
        .iter()
        .zip(&v)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(moved <= 1e-6, "full width moved the vector by {moved}");
}

#[test]
fn truncation_to_an_untrained_or_wider_width_is_refused_eg2() {
    let v = unit_length(ramp(768, 0.5, 1.0)).expect("non-zero");
    for dim in [0, 100, 1024] {
        assert!(truncate_to(&v, dim).is_err(), "dim {dim}");
    }
    assert!(
        truncate_to(&v[..512], 768).is_err(),
        "wider than the vector"
    );
    assert!(
        truncate_to(&[0.0; 768], 128).is_err(),
        "a zero prefix has no direction"
    );
}

#[test]
fn pooling_is_the_mean_over_tokens_eg2() {
    assert_eq!(mean_rows(&[1.0, 2.0, 3.0, 6.0], 2), [2.0, 4.0]);
    assert!(unit_length(vec![0.0; 3]).is_err());
    assert!(unit_length(vec![f32::NAN, 1.0]).is_err());
    let u = unit_length(vec![3.0, 4.0]).expect("non-zero");
    assert!((u[0] - 0.6).abs() < 1e-7 && (u[1] - 0.8).abs() < 1e-7);
}
