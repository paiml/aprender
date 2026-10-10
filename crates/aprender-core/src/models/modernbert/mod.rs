//! ModernBERT encoder (D-13): a reusable, prefix-aware, fp32 forward proven against
//! transformers in CI.
//!
//! A lift of the spike-025 encoder numerics (`.claude/skills/spike-findings-aprender/
//! sources/025-laya-rust-forward-parity/src/laya.rs`) onto the house GEMM
//! (`trueno::blis::gemm_blis`) and the house `.apr` loader
//! (`AprV2DequantExt::get_tensor_as_f32`, the one F16 widening path). Every numeric
//! operation and its order is the spike's, which matched torch fp32 to 3.8e-6 on
//! probabilities; the only numeric substitution is the erf, now the house
//! `batuta_common::math::erfc_precise` (re-proved by `tests::tiny_parity`).
//!
//! It is NOT built on `models::bert` or `autograd`: HF BERT is post-norm with
//! position embeddings and biases, which is the wrong architecture on every block.
//!
//! # Architecture (reference: transformers 5.17 `modeling_modernbert.py`, SDPA)
//!
//! | piece | semantics |
//! |---|---|
//! | embeddings | `LayerNorm(tok_embeddings(ids))`, no bias, **no position embedding** |
//! | layer 0 | `attn_norm = Identity`; layers 1.. are pre-norm LayerNorm without bias |
//! | attention | fused `Wqkv` `[3d, d]`, no bias, split into q/k/v; rotate-half RoPE, theta `rope_parameters.full_attention` on global layers, `rope_parameters.sliding_attention` on local ones; scale `hd^-0.5`; bidirectional |
//! | global vs local | `layer_types` when present, else `i % global_attn_every_n_layers == 0`; both present must agree |
//! | **local window** | `\|i - j\| <= local_attention / 2`, **inclusive** (`tests::window_mutation` proves it) |
//! | MLP | `Wi` `[2 * intermediate, d]` -> `chunk(input, gate)` -> `gelu_exact(input) * gate` -> `Wo` |
//! | final | `LayerNorm(final_norm)`, no bias |
//! | layer norm | f64 mean/var accumulation, `eps = norm_eps` from the config at every call site |
//!
//! Every dense product is [`Linear::forward`]: `C^T = W . X^T` on `gemm_blis`, the
//! checkpoint's `[out, in]` row-major weight IS the GEMM's A operand (spike-020 layout,
//! no weight transpose), and W's output rows are banded across the rayon pool.
//!
//! # Tensor names (the import/load contract)
//!
//! [`expected_modernbert_tensor_names`] with a `prefix`: a plain HF ModernBERT uses
//! `""`; Laya's encoder copy uses `"encoder."`. Weights may be F16 or F32 in the
//! `.apr`; both widen through `get_tensor_as_f32`.
//!
//! # Supported domain
//!
//! [`ModernBertConfig::from_json_bytes`] refuses, with a typed
//! [`ModernBertConfigError`] naming the field and BEFORE any shape is derived or any
//! buffer allocated:
//!
//! - any of `vocab_size`, `hidden_size`, `intermediate_size`, `num_hidden_layers`,
//!   `num_attention_heads`, `global_attn_every_n_layers`, `local_attention` equal to 0;
//! - `num_hidden_layers` above [`MAX_NUM_HIDDEN_LAYERS`] (checked before any per-layer
//!   derivation, so an untrusted count never sizes an allocation);
//! - a `hidden_activation` other than `gelu`, a `rope_type` other than `default`, or a
//!   non-null `rope_scaling` (HF would compute a different forward; this encoder computes
//!   only exact GELU and unscaled rotate-half RoPE);
//! - `hidden_size % num_attention_heads != 0`, or an odd head dim (rotate-half pairs dims);
//! - an odd `local_attention` (the window is `local_attention / 2` on each side);
//! - `layer_types` whose length is not `num_hidden_layers`, or a value other than
//!   `full_attention` / `sliding_attention`, or one that disagrees with the modulo rule;
//! - `norm_eps` not finite and > 0; a missing, non-finite or non-positive rope theta;
//! - `norm_bias`, `attention_bias` or `mlp_bias` true (this encoder has no biases);
//! - `vocab_size x hidden_size`, `3 x hidden_size x hidden_size` or
//!   `2 x intermediate_size x hidden_size` overflowing `usize`.
//!
//! At runtime an out-of-vocab id, an empty row, a GEMM failure or an inconsistent
//! buffer length is a typed [`ModernBertError`], never an index panic.
//!
//! Contract: `contracts/laya-parity-v1.yaml` (embeddings / per-layer / final-norm rungs
//! and the window mutation). Gated on the default `parallel` feature (rayon).

pub mod config;
pub mod embeddings;
pub mod encoder;
pub mod gemm;
pub mod layer;
pub mod load;

pub use config::{ModernBertConfig, ModernBertConfigError, MAX_NUM_HIDDEN_LAYERS};
pub use embeddings::ModernBertEmbeddings;
pub use encoder::ModernBertEncoder;
pub use gemm::Linear;
pub use layer::{attention, gelu_exact, layer_norm, rope_rotate_half, ModernBertLayer};
pub use load::{expected_modernbert_tensor_names, load_tensor, ModernBertLoadError};

use std::fmt;

/// A runtime refusal of the ModernBERT forward or one of its reusable primitives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModernBertError {
    /// The input row has no tokens.
    EmptyInput,
    /// A token id is not below the vocabulary size.
    OutOfVocab {
        /// Position of the offending id in the row.
        position: usize,
        /// The id itself.
        id: u32,
        /// The model's vocabulary size.
        vocab_size: usize,
    },
    /// A buffer length disagrees with the dimensions it was passed with.
    InputShape {
        /// Which operand (e.g. `"linear.x"`, `"layer_norm.w"`).
        what: &'static str,
        /// Length implied by the dimensions (`None` when that product overflows).
        expected: Option<usize>,
        /// Length actually observed.
        observed: usize,
    },
    /// The BLIS GEMM refused its operands.
    Gemm(String),
}

impl fmt::Display for ModernBertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "modernbert: empty input row"),
            Self::OutOfVocab {
                position,
                id,
                vocab_size,
            } => write!(
                f,
                "modernbert: token id {id} at position {position} is out of vocabulary (size {vocab_size})"
            ),
            Self::InputShape {
                what,
                expected,
                observed,
            } => match expected {
                Some(e) => write!(f, "modernbert: {what} has length {observed}, expected {e}"),
                None => write!(
                    f,
                    "modernbert: {what} has length {observed}, and its dimensions overflow usize"
                ),
            },
            Self::Gemm(e) => write!(f, "modernbert: gemm_blis failed: {e}"),
        }
    }
}

impl std::error::Error for ModernBertError {}

/// Refuse `observed != product(dims)` (or an overflowing product) as a typed error.
pub(crate) fn check_len(
    what: &'static str,
    observed: usize,
    dims: &[usize],
) -> Result<(), ModernBertError> {
    let expected = dims.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d));
    if expected == Some(observed) {
        Ok(())
    } else {
        Err(ModernBertError::InputShape {
            what,
            expected,
            observed,
        })
    }
}

/// Fixture plumbing shared by the in-module tests: the tiny HF ModernBERT from plan
/// 08-02, written into an in-memory `.apr`, and the `laya-parity-v1` bars.
#[cfg(test)]
pub(crate) mod test_support {
    use super::{ModernBertConfig, ModernBertEncoder, ModernBertLoadError};
    use crate::format::v2::{AprV2Metadata, AprV2ReaderRef, AprV2Writer, TensorDType};
    use std::cmp::Ordering;
    use std::path::{Path, PathBuf};

    /// `(name, shape, raw little-endian F16 bytes)` as stored in the fixture.
    pub(crate) type RawTensor = (String, Vec<usize>, Vec<u8>);

    pub(crate) fn fixture_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/modernbert_tiny")
    }

    pub(crate) fn config_bytes() -> Vec<u8> {
        std::fs::read(fixture_dir().join("config.json")).expect("read modernbert_tiny/config.json")
    }

    pub(crate) fn fixture_config() -> ModernBertConfig {
        ModernBertConfig::from_json_bytes(&config_bytes()).expect("fixture config is supported")
    }

    /// Every tensor of `model.safetensors`, raw F16 bytes, sorted by name.
    pub(crate) fn fixture_tensors() -> Vec<RawTensor> {
        let bytes = std::fs::read(fixture_dir().join("model.safetensors"))
            .expect("read modernbert_tiny/model.safetensors");
        let st = safetensors::SafeTensors::deserialize(&bytes).expect("parse safetensors");
        let mut out: Vec<RawTensor> = st
            .tensors()
            .into_iter()
            .map(|(name, view)| {
                assert_eq!(
                    view.dtype(),
                    safetensors::Dtype::F16,
                    "{name}: fixture is F16"
                );
                (name, view.shape().to_vec(), view.data().to_vec())
            })
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// Write `tensors` as F16 under `prefix` into an in-memory `.apr`.
    pub(crate) fn write_apr(tensors: &[RawTensor], prefix: &str) -> Vec<u8> {
        let mut w = AprV2Writer::new(AprV2Metadata::default());
        for (name, shape, raw) in tensors {
            w.add_tensor(
                format!("{prefix}{name}"),
                TensorDType::F16,
                shape.clone(),
                raw.clone(),
            );
        }
        w.write().expect("write in-memory .apr")
    }

    pub(crate) fn load(
        apr: &[u8],
        prefix: &str,
        config: &ModernBertConfig,
    ) -> Result<ModernBertEncoder, ModernBertLoadError> {
        let reader = AprV2ReaderRef::from_bytes(apr).expect("open in-memory .apr");
        ModernBertEncoder::from_apr(&reader, prefix, config)
    }

    /// The tiny fixture loaded with prefix `""`.
    pub(crate) fn fixture_encoder() -> ModernBertEncoder {
        let cfg = fixture_config();
        load(&write_apr(&fixture_tensors(), ""), "", &cfg).expect("load tiny fixture")
    }

    /// One oracle row: ids plus the fp32 ladder blocks.
    pub(crate) struct OracleRow {
        pub(crate) ids: Vec<u32>,
        pub(crate) blocks: Vec<(String, Vec<f32>)>,
    }

    impl OracleRow {
        pub(crate) fn block(&self, name: &str) -> &[f32] {
            &self
                .blocks
                .iter()
                .find(|(n, _)| n == name)
                .unwrap_or_else(|| panic!("oracle block {name} missing"))
                .1
        }
    }

    pub(crate) fn oracle_json() -> serde_json::Value {
        let bytes = std::fs::read(fixture_dir().join("oracle.json"))
            .expect("read modernbert_tiny/oracle.json");
        serde_json::from_slice(&bytes).expect("parse oracle.json")
    }

    pub(crate) fn oracle_rows() -> Vec<OracleRow> {
        let o = oracle_json();
        let names: Vec<String> = o["blocks"]
            .as_array()
            .expect("oracle blocks")
            .iter()
            .map(|b| b.as_str().expect("block name").to_string())
            .collect();
        o["rows"]
            .as_array()
            .expect("oracle rows")
            .iter()
            .map(|r| OracleRow {
                ids: r["ids"]
                    .as_array()
                    .expect("ids")
                    .iter()
                    .map(|v| u32::try_from(v.as_u64().expect("id")).expect("id fits u32"))
                    .collect(),
                blocks: names
                    .iter()
                    .map(|n| {
                        let vals = r[n.as_str()]
                            .as_array()
                            .unwrap_or_else(|| panic!("block {n}"))
                            .iter()
                            // exact f32 values carried as their f64 repr
                            .map(|v| v.as_f64().expect("f32 value") as f32)
                            .collect();
                        (n.clone(), vals)
                    })
                    .collect(),
            })
            .collect()
    }

    /// `equations.<equation>.float_tolerance` from `contracts/laya-parity-v1.yaml`,
    /// read at test time (never a Rust literal).
    pub(crate) fn tolerance(equation: &str) -> f64 {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts/laya-parity-v1.yaml");
        let text = std::fs::read_to_string(&path).expect("read laya-parity-v1.yaml");
        let y: serde_yaml::Value = serde_yaml::from_str(&text).expect("parse laya-parity-v1.yaml");
        y["equations"][equation]["float_tolerance"]
            .as_f64()
            .unwrap_or_else(|| panic!("laya-parity-v1 equations.{equation}.float_tolerance"))
    }

    /// NaN-visible `delta <= bound`: a NaN on either side never passes.
    pub(crate) fn within(delta: f64, bound: f64) -> bool {
        matches!(
            delta.partial_cmp(&bound),
            Some(Ordering::Less | Ordering::Equal)
        )
    }

    /// `max |a - b|` in f64; NaN-propagating, and NaN on a length mismatch.
    pub(crate) fn max_abs(a: &[f32], b: &[f32]) -> f64 {
        if a.len() != b.len() {
            return f64::NAN;
        }
        a.iter().zip(b).fold(0.0f64, |m, (&x, &y)| {
            let d = (f64::from(x) - f64::from(y)).abs();
            if d.is_nan() || m.is_nan() {
                f64::NAN
            } else {
                m.max(d)
            }
        })
    }

    /// `rms(a - reference) / rms(reference)`; NaN on a length mismatch.
    pub(crate) fn rel_rms(a: &[f32], reference: &[f32]) -> f64 {
        if a.len() != reference.len() || reference.is_empty() {
            return f64::NAN;
        }
        let n = reference.len() as f64;
        let num = a
            .iter()
            .zip(reference)
            .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
            .sum::<f64>()
            / n;
        let den = reference.iter().map(|&y| f64::from(y).powi(2)).sum::<f64>() / n;
        (num / den).sqrt()
    }

    /// Run `encoder` on `ids`, collecting every ladder tap in order.
    pub(crate) fn run_taps(
        encoder: &ModernBertEncoder,
        ids: &[u32],
    ) -> (Vec<f32>, Vec<(String, Vec<f32>)>) {
        let mut taps = Vec::new();
        let out = encoder
            .forward(ids, |name, block| {
                taps.push((name.to_string(), block.to_vec()))
            })
            .expect("forward");
        (out, taps)
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{
        config_bytes, fixture_config, fixture_encoder, fixture_tensors, load, max_abs, oracle_json,
        oracle_rows, rel_rms, run_taps, tolerance, within, write_apr,
    };
    use super::{ModernBertConfig, ModernBertError, ModernBertLoadError};

    /// FALSIFY-LAYA-PARITY-002 on the tiny fixture: a plain HF ModernBERT
    /// (F16 safetensors -> in-memory .apr -> prefix "" load -> forward) matches the
    /// transformers fp32 oracle on embeddings, every layer and the final norm, with
    /// every bar read from `contracts/laya-parity-v1.yaml`.
    #[test]
    fn tiny_parity() {
        let emb_bar = tolerance("embeddings_abs");
        let layer_bar = tolerance("per_layer_rel_rms");
        let final_bar = tolerance("final_norm_abs");
        let encoder = fixture_encoder();
        let rows = oracle_rows();
        assert!(!rows.is_empty(), "oracle has rows");
        println!("modernbert tiny_parity on ARCH={}", std::env::consts::ARCH);
        for (ri, row) in rows.iter().enumerate() {
            let (out, taps) = run_taps(&encoder, &row.ids);
            let n_layers = encoder.config().num_hidden_layers();
            assert_eq!(
                taps.len(),
                n_layers + 2,
                "row {ri}: emb + layers + final taps"
            );
            for (name, got) in &taps {
                let want = row.block(name);
                if name.starts_with("layer") {
                    let r = rel_rms(got, want);
                    println!(
                        "row {ri} {name}: rel_rms {r:.3e} (bar {layer_bar:.1e}) max|delta| {:.3e}",
                        max_abs(got, want)
                    );
                    assert!(
                        within(r, layer_bar),
                        "row {ri} {name}: rel_rms {r:e} > {layer_bar:e}"
                    );
                } else {
                    let bar = if name == "emb" { emb_bar } else { final_bar };
                    let d = max_abs(got, want);
                    println!("row {ri} {name}: max|delta| {d:.3e} (bar {bar:.1e})");
                    assert!(
                        within(d, bar),
                        "row {ri} {name}: max|delta| {d:e} > {bar:e}"
                    );
                }
            }
            let final_tap = &taps.last().expect("final tap").1;
            assert_eq!(
                &out, final_tap,
                "row {ri}: forward returns the final-norm block"
            );
        }
    }

    /// FALSIFY-LAYA-PARITY-004: the local window is `|i - j| <= local_attention / 2`
    /// inclusive. Mutating it to w-1 or w+1 pushes the FIRST local layer past the
    /// per-layer bar while the FIRST global layer still passes.
    #[test]
    fn window_mutation() {
        let bar = tolerance("per_layer_rel_rms");
        let mut encoder = fixture_encoder();
        let cfg = encoder.config().clone();
        let w = cfg.window();
        let oracle = oracle_json();
        let recorded = oracle["window_mutation"]["half_window"]
            .as_u64()
            .expect("oracle window_mutation.half_window");
        assert_eq!(w as u64, recorded, "config window agrees with the oracle's");
        let first_global = cfg
            .layer_is_global()
            .iter()
            .position(|&g| g)
            .expect("a global layer");
        let first_local = cfg
            .layer_is_global()
            .iter()
            .position(|&g| !g)
            .expect("a local layer");
        assert!(
            first_global < first_local,
            "the mutation must not reach the global layer first"
        );
        let rows = oracle_rows();
        println!(
            "modernbert window_mutation on ARCH={} (w={w}, bar {bar:.1e})",
            std::env::consts::ARCH
        );
        for mutated in [w - 1, w + 1] {
            encoder.set_window_override(Some(mutated));
            for (ri, row) in rows.iter().enumerate() {
                let (_, taps) = run_taps(&encoder, &row.ids);
                let g_name = format!("layer{first_global}");
                let l_name = format!("layer{first_local}");
                let g = rel_rms(&taps[first_global + 1].1, row.block(&g_name));
                let l = rel_rms(&taps[first_local + 1].1, row.block(&l_name));
                println!(
                    "window {mutated} row {ri}: {g_name} rel_rms {g:.3e}, {l_name} rel_rms {l:.3e}"
                );
                assert!(
                    within(g, bar),
                    "window {mutated} row {ri}: {g_name} should still pass, got {g:e}"
                );
                assert!(
                    matches!(l.partial_cmp(&bar), Some(std::cmp::Ordering::Greater)),
                    "window {mutated} row {ri}: {l_name} should exceed {bar:e}, got {l:e}"
                );
            }
        }
        encoder.set_window_override(None);
        let (_, taps) = run_taps(&encoder, &rows[0].ids);
        let l = rel_rms(
            &taps[first_local + 1].1,
            rows[0].block(&format!("layer{first_local}")),
        );
        assert!(
            within(l, bar),
            "override cleared restores the true window: {l:e}"
        );
    }

    /// The same tensors under prefix `encoder.` (Laya's layout) load with that prefix and
    /// give bit-identical output to prefix `""`.
    #[test]
    fn prefix_reuse() {
        let cfg = fixture_config();
        let tensors = fixture_tensors();
        let plain = load(&write_apr(&tensors, ""), "", &cfg).expect("plain load");
        let prefixed_apr = write_apr(&tensors, "encoder.");
        let prefixed = load(&prefixed_apr, "encoder.", &cfg).expect("prefixed load");
        for row in oracle_rows() {
            let (a, ta) = run_taps(&plain, &row.ids);
            let (b, tb) = run_taps(&prefixed, &row.ids);
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&a), bits(&b), "final output bit-identical");
            for ((na, va), (nb, vb)) in ta.iter().zip(&tb) {
                assert_eq!(na, nb);
                assert_eq!(bits(va), bits(vb), "{na} bit-identical");
            }
        }
        // The prefix is load-bearing: the prefixed file does not load as plain HF.
        match load(&prefixed_apr, "", &cfg) {
            Err(ModernBertLoadError::MissingTensor { name }) => {
                assert_eq!(name, "embeddings.tok_embeddings.weight");
            }
            other => panic!("expected MissingTensor, got {other:?}"),
        }
    }

    /// `norm_eps` is used, not parsed and ignored: 1e-1 moves the final norm by more
    /// than the final-norm bar relative to the fixture's 1e-5.
    #[test]
    fn norm_eps_honoured() {
        let bar = tolerance("final_norm_abs");
        let mut cfg_json: serde_json::Value =
            serde_json::from_slice(&config_bytes()).expect("config json");
        cfg_json["norm_eps"] = serde_json::json!(1e-1);
        let big_eps = ModernBertConfig::from_json_bytes(
            &serde_json::to_vec(&cfg_json).expect("serialize config"),
        )
        .expect("eps 1e-1 is supported");
        assert!((big_eps.norm_eps() - 1e-1).abs() < 1e-12);
        let tensors = fixture_tensors();
        let apr = write_apr(&tensors, "");
        let base = load(&apr, "", &fixture_config()).expect("base load");
        let moved = load(&apr, "", &big_eps).expect("eps load");
        let ids = &oracle_rows()[0].ids;
        let (a, _) = run_taps(&base, ids);
        let (b, _) = run_taps(&moved, ids);
        let d = max_abs(&a, &b);
        println!("norm_eps 1e-5 vs 1e-1: final max|delta| {d:.3e} (bar {bar:.1e})");
        assert!(
            matches!(d.partial_cmp(&bar), Some(std::cmp::Ordering::Greater)),
            "a changed norm_eps must change the output by more than {bar:e}, got {d:e}"
        );
    }

    /// An out-of-vocab id and an empty row are typed errors, never index panics.
    #[test]
    fn forward_refuses_out_of_vocab_and_empty() {
        let encoder = fixture_encoder();
        let vocab = encoder.config().vocab_size();
        let bad = u32::try_from(vocab).expect("vocab fits u32");
        let err = encoder
            .forward(&[1, 5, bad, 2], |_, _| {})
            .expect_err("oov refused");
        assert_eq!(
            err,
            ModernBertError::OutOfVocab {
                position: 2,
                id: bad,
                vocab_size: vocab
            }
        );
        let err = encoder.forward(&[], |_, _| {}).expect_err("empty refused");
        assert_eq!(err, ModernBertError::EmptyInput);
    }
}
