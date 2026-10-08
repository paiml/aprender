//! ModernBERT weight loading from APR v2 (prefix-aware).
//!
//! Tensor names are the HF ModernBERT names under a caller-supplied prefix — `""` for
//! a plain HF ModernBERT, `"encoder."` for Laya's encoder copy:
//!
//! ```text
//! {prefix}embeddings.tok_embeddings.weight      [vocab, d]
//! {prefix}embeddings.norm.weight                [d]
//! {prefix}layers.{i}.attn_norm.weight           [d]           (i >= 1; layer 0 is Identity)
//! {prefix}layers.{i}.attn.Wqkv.weight           [3d, d]
//! {prefix}layers.{i}.attn.Wo.weight             [d, d]
//! {prefix}layers.{i}.mlp_norm.weight            [d]
//! {prefix}layers.{i}.mlp.Wi.weight              [2 * intermediate, d]
//! {prefix}layers.{i}.mlp.Wo.weight              [d, intermediate]
//! {prefix}final_norm.weight                     [d]
//! ```
//!
//! The `.apr` bytes are untrusted: a missing tensor, a shape that disagrees with the
//! config, an undecodable dtype and a non-finite value are each refused BY NAME with a
//! [`ModernBertLoadError`]. Widening is `AprV2DequantExt::get_tensor_as_f32` — the one
//! F16 path in the workspace, no hand-written f16 code here.

use super::{Linear, ModernBertConfig, ModernBertEmbeddings, ModernBertEncoder, ModernBertLayer};
use crate::format::v2::{AprV2ReaderRef, TensorDType};
use crate::format::AprV2DequantExt;
use std::fmt;

/// Why a ModernBERT `.apr` was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModernBertLoadError {
    /// A required tensor is absent.
    MissingTensor {
        /// Full (prefixed) tensor name.
        name: String,
    },
    /// A tensor's stored shape disagrees with the config.
    ShapeMismatch {
        /// Full (prefixed) tensor name.
        name: String,
        /// Shape the config implies.
        expected: Vec<usize>,
        /// Shape stored in the file.
        observed: Vec<usize>,
    },
    /// A tensor's dtype cannot be widened to f32, or its data is truncated.
    Undecodable {
        /// Full (prefixed) tensor name.
        name: String,
        /// The stored dtype.
        dtype: String,
    },
    /// A tensor holds a NaN or an infinity.
    NonFinite {
        /// Full (prefixed) tensor name.
        name: String,
    },
}

impl fmt::Display for ModernBertLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingTensor { name } => write!(f, "modernbert load: missing tensor {name}"),
            Self::ShapeMismatch {
                name,
                expected,
                observed,
            } => write!(
                f,
                "modernbert load: tensor {name} has shape {observed:?}, config implies {expected:?}"
            ),
            Self::Undecodable { name, dtype } => write!(
                f,
                "modernbert load: tensor {name} ({dtype}) cannot be widened to f32"
            ),
            Self::NonFinite { name } => {
                write!(f, "modernbert load: tensor {name} holds a non-finite value")
            }
        }
    }
}

impl std::error::Error for ModernBertLoadError {}

/// `(name, shape)` for every tensor the encoder needs, in load order.
fn expected_tensors(config: &ModernBertConfig, prefix: &str) -> Vec<(String, Vec<usize>)> {
    let d = config.hidden_size();
    let inter = config.intermediate_size();
    let mut t = vec![
        (
            format!("{prefix}embeddings.tok_embeddings.weight"),
            vec![config.vocab_size(), d],
        ),
        (format!("{prefix}embeddings.norm.weight"), vec![d]),
    ];
    for i in 0..config.num_hidden_layers() {
        let p = format!("{prefix}layers.{i}.");
        if i > 0 {
            t.push((format!("{p}attn_norm.weight"), vec![d]));
        }
        t.push((format!("{p}attn.Wqkv.weight"), vec![3 * d, d]));
        t.push((format!("{p}attn.Wo.weight"), vec![d, d]));
        t.push((format!("{p}mlp_norm.weight"), vec![d]));
        t.push((format!("{p}mlp.Wi.weight"), vec![2 * inter, d]));
        t.push((format!("{p}mlp.Wo.weight"), vec![d, inter]));
    }
    t.push((format!("{prefix}final_norm.weight"), vec![d]));
    t
}

/// The canonical tensor names a ModernBERT `.apr` must contain for `config`, under
/// `prefix` (`""` for plain HF, `"encoder."` for Laya). This is the import/load
/// contract, mirroring `expected_bert_tensor_names`.
#[must_use]
pub fn expected_modernbert_tensor_names(config: &ModernBertConfig, prefix: &str) -> Vec<String> {
    expected_tensors(config, prefix)
        .into_iter()
        .map(|(n, _)| n)
        .collect()
}

/// Load, shape-check, widen and finiteness-check one tensor.
///
/// Public so a model built ON the encoder (Laya's decision head, `aprender-decide`)
/// loads its own tensors through this one path rather than a copy of it.
///
/// # Errors
///
/// [`ModernBertLoadError`] naming `name`: missing, a stored shape other than `shape`,
/// a dtype other than F16 / F32 or a payload that is not exactly `product(shape) x width`
/// bytes (truncated or padded data), or a non-finite value.
pub fn load_tensor(
    reader: &AprV2ReaderRef<'_>,
    name: &str,
    shape: &[usize],
) -> Result<Vec<f32>, ModernBertLoadError> {
    let entry = reader
        .get_tensor(name)
        .ok_or_else(|| ModernBertLoadError::MissingTensor {
            name: name.to_string(),
        })?;
    if entry.shape != shape {
        return Err(ModernBertLoadError::ShapeMismatch {
            name: name.to_string(),
            expected: shape.to_vec(),
            observed: entry.shape.clone(),
        });
    }
    let undecodable = || ModernBertLoadError::Undecodable {
        name: name.to_string(),
        dtype: format!("{:?}", entry.dtype),
    };
    // Only the documented domain (F16 / F32), and only a payload of EXACTLY
    // product(shape) x width bytes: the quantized decoders zero-fill truncated data and zero
    // non-finite scales, so widening first and checking the output length after would load
    // a truncated or corrupt quantized weight as zeros.
    let width: u64 = match entry.dtype {
        TensorDType::F16 => 2,
        TensorDType::F32 => 4,
        _ => return Err(undecodable()),
    };
    // Checked product: a caller's shape need not come from a checked_mul-proven config.
    let elements = shape.iter().try_fold(1usize, |a, &b| a.checked_mul(b));
    let expected_bytes = elements
        .and_then(|n| u64::try_from(n).ok())
        .and_then(|n| n.checked_mul(width));
    if expected_bytes != Some(entry.size) {
        return Err(undecodable());
    }
    let data = reader.get_tensor_as_f32(name).ok_or_else(undecodable)?;
    if Some(data.len()) != elements {
        return Err(undecodable());
    }
    if !data.iter().all(|v| v.is_finite()) {
        return Err(ModernBertLoadError::NonFinite {
            name: name.to_string(),
        });
    }
    Ok(data)
}

impl ModernBertEncoder {
    /// Load a ModernBERT encoder from an APR v2 reader, with every tensor name under
    /// `prefix`.
    ///
    /// # Errors
    ///
    /// [`ModernBertLoadError`] naming the tensor for a missing tensor, a shape that
    /// disagrees with `config`, an undecodable dtype or a non-finite value.
    #[provable_contracts_macros::contract("laya-parity-v1", equation = "embeddings_abs")]
    pub fn from_apr(
        reader: &AprV2ReaderRef<'_>,
        prefix: &str,
        config: &ModernBertConfig,
    ) -> Result<Self, ModernBertLoadError> {
        let mut tensors = expected_tensors(config, prefix).into_iter();
        let mut next = || -> Result<(Vec<f32>, Vec<usize>), ModernBertLoadError> {
            // expected_tensors yields exactly the sequence consumed below.
            let (name, shape) =
                tensors
                    .next()
                    .ok_or_else(|| ModernBertLoadError::MissingTensor {
                        name: format!("{prefix}<load order exhausted>"),
                    })?;
            let data = load_tensor(reader, &name, &shape)?;
            Ok((data, shape))
        };
        let lin = |(w, s): (Vec<f32>, Vec<usize>)| Linear {
            w,
            b: None,
            out: s[0],
            inp: s[1],
        };
        let d = config.hidden_size();
        let tok = next()?.0;
        let emb_norm = next()?.0;
        let embeddings = ModernBertEmbeddings::from_parts(tok, emb_norm, config.vocab_size(), d);
        let mut layers = Vec::with_capacity(config.num_hidden_layers());
        for (i, &global) in config.layer_is_global().iter().enumerate() {
            let attn_norm = if i == 0 { None } else { Some(next()?.0) };
            let wqkv = lin(next()?);
            let wo = lin(next()?);
            let mlp_norm = next()?.0;
            let wi = lin(next()?);
            let wo_mlp = lin(next()?);
            layers.push(ModernBertLayer::from_parts(
                attn_norm, wqkv, wo, mlp_norm, wi, wo_mlp, global,
            ));
        }
        let final_norm = next()?.0;
        Ok(Self::from_parts(
            config.clone(),
            embeddings,
            layers,
            final_norm,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{expected_modernbert_tensor_names, ModernBertLoadError};
    use crate::models::modernbert::test_support::{
        fixture_config, fixture_tensors, load, write_apr, RawTensor,
    };

    /// The import/load contract names exactly the fixture's tensors (plain HF, no
    /// prefix), and the prefix is prepended verbatim.
    #[test]
    fn expected_names_match_the_hf_fixture() {
        let cfg = fixture_config();
        let mut want = expected_modernbert_tensor_names(&cfg, "");
        want.sort();
        let got: Vec<String> = fixture_tensors().into_iter().map(|t| t.0).collect();
        assert_eq!(want, got);
        assert!(expected_modernbert_tensor_names(&cfg, "encoder.")
            .iter()
            .all(|n| n.starts_with("encoder.")));
        assert!(
            !want.contains(&"layers.0.attn_norm.weight".to_string()),
            "layer 0 is Identity"
        );
    }

    fn without(name: &str) -> Vec<RawTensor> {
        fixture_tensors()
            .into_iter()
            .filter(|t| t.0 != name)
            .collect()
    }

    #[test]
    fn refuses_missing_tensor_by_name() {
        let cfg = fixture_config();
        let missing = "layers.1.mlp.Wi.weight";
        let err = load(&write_apr(&without(missing), ""), "", &cfg).expect_err("refused");
        assert_eq!(
            err,
            ModernBertLoadError::MissingTensor {
                name: missing.to_string()
            }
        );
        // Under a prefix the refusal names the full prefixed tensor.
        let err =
            load(&write_apr(&without(missing), "encoder."), "encoder.", &cfg).expect_err("refused");
        assert_eq!(
            err,
            ModernBertLoadError::MissingTensor {
                name: format!("encoder.{missing}")
            }
        );
    }

    #[test]
    fn refuses_shape_mismatch() {
        let cfg = fixture_config();
        let name = "layers.0.attn.Wo.weight";
        let tensors: Vec<RawTensor> = fixture_tensors()
            .into_iter()
            .map(|(n, s, raw)| {
                if n == name {
                    (n, vec![16, 64], raw)
                } else {
                    (n, s, raw)
                }
            })
            .collect();
        let err = load(&write_apr(&tensors, ""), "", &cfg).expect_err("refused");
        assert_eq!(
            err,
            ModernBertLoadError::ShapeMismatch {
                name: name.to_string(),
                expected: vec![32, 32],
                observed: vec![16, 64],
            }
        );
    }

    /// A NaN (F16 bit pattern 0x7E00) and an infinity (0x7C00) injected into one weight
    /// are each refused by name at load.
    #[test]
    fn refuses_non_finite() {
        let cfg = fixture_config();
        let name = "layers.2.mlp.Wo.weight";
        for bits in [0x7E00u16, 0x7C00u16] {
            let tensors: Vec<RawTensor> = fixture_tensors()
                .into_iter()
                .map(|(n, s, mut raw)| {
                    if n == name {
                        raw[10..12].copy_from_slice(&bits.to_le_bytes());
                    }
                    (n, s, raw)
                })
                .collect();
            let err = load(&write_apr(&tensors, ""), "", &cfg).expect_err("refused");
            assert_eq!(
                err,
                ModernBertLoadError::NonFinite {
                    name: name.to_string()
                },
                "f16 bits {bits:#06x}"
            );
        }
    }

    /// A dtype outside F16 / F32, or a payload that is not exactly `product(shape) x width`
    /// bytes, is refused BY NAME before widening. The quantized decoders zero-fill missing
    /// data, so an empty APR-Q4 weight must not load as a tensor of zeros.
    #[test]
    fn refuses_foreign_dtype_and_truncated_payload() {
        use crate::format::v2::{AprV2Metadata, AprV2Writer, TensorDType};
        let cfg = fixture_config();
        let name = "final_norm.weight";
        let build = |dtype: TensorDType, cut: usize| {
            let mut w = AprV2Writer::new(AprV2Metadata::default());
            for (n, shape, raw) in fixture_tensors() {
                if n == name {
                    let keep = raw.len().saturating_sub(cut);
                    let payload = if dtype == TensorDType::F16 {
                        raw[..keep].to_vec()
                    } else {
                        Vec::new()
                    };
                    w.add_tensor(n, dtype, shape, payload);
                } else {
                    w.add_tensor(n, TensorDType::F16, shape, raw);
                }
            }
            w.write().expect("write in-memory .apr")
        };
        for (label, apr) in [
            ("empty APR-Q4", build(TensorDType::AprQ4, 0)),
            ("F16 two bytes short", build(TensorDType::F16, 2)),
        ] {
            let err = load(&apr, "", &cfg).expect_err(label);
            assert!(
                matches!(&err, ModernBertLoadError::Undecodable { name: n, .. } if n == name),
                "{label}: {err:?}"
            );
        }
    }
}
