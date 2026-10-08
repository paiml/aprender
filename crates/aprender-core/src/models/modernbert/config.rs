//! ModernBERT `config.json`: a typed HF parse that validates the supported domain
//! BEFORE any shape is derived or any buffer allocated.
//!
//! The primitives downstream slice by `hidden_size`, `head_dim` and
//! `2 * intermediate_size` and chunk by the same, so a zero, an odd rotary dim or an
//! overflowing product would otherwise surface as an index panic deep in the forward.
//! Every such config is refused here with a [`ModernBertConfigError`] naming the field.

use serde::Deserialize;
use std::cmp::Ordering;
use std::fmt;

/// Layer-type strings HF writes in `layer_types`.
const FULL_ATTENTION: &str = "full_attention";
const SLIDING_ATTENTION: &str = "sliding_attention";

/// The one activation this encoder computes (`gelu_exact`): HF's `ACT2FN["gelu"]`.
const SUPPORTED_ACTIVATION: &str = "gelu";
/// The one RoPE variant this encoder computes: unscaled rotate-half.
const SUPPORTED_ROPE_TYPE: &str = "default";

/// The most encoder layers a config may declare. Far above any ModernBERT (base 22, large
/// 28), and low enough that the per-layer flags and tensor names derived from an UNTRUSTED
/// count (a `.apr`'s embedded config) cannot allocate proportionally to it.
pub const MAX_NUM_HIDDEN_LAYERS: usize = 1024;

/// Why a ModernBERT `config.json` is outside the supported domain.
#[derive(Debug, Clone, PartialEq)]
pub enum ModernBertConfigError {
    /// The bytes are not a JSON object with the required fields and types.
    Json(String),
    /// A dimension that must be positive is 0.
    ZeroDimension {
        /// The HF field name.
        field: &'static str,
    },
    /// `hidden_size % num_attention_heads != 0`.
    HeadsDoNotDivideHidden {
        /// `hidden_size`.
        hidden_size: usize,
        /// `num_attention_heads`.
        num_attention_heads: usize,
    },
    /// The head dim is odd; rotate-half RoPE pairs dims `p` and `p + hd/2`.
    OddHeadDim {
        /// `hidden_size / num_attention_heads`.
        head_dim: usize,
    },
    /// `local_attention` is odd; the window is `local_attention / 2` on each side.
    OddLocalAttention {
        /// `local_attention`.
        local_attention: usize,
    },
    /// `layer_types` does not have one entry per layer.
    LayerTypesLength {
        /// `num_hidden_layers`.
        expected: usize,
        /// `layer_types.len()`.
        observed: usize,
    },
    /// A `layer_types` entry is neither `full_attention` nor `sliding_attention`.
    UnknownLayerType {
        /// Layer index.
        layer: usize,
        /// The unknown value.
        value: String,
    },
    /// `layer_types[layer]` disagrees with `layer % global_attn_every_n_layers == 0`.
    LayerTypesDisagree {
        /// Layer index.
        layer: usize,
    },
    /// `norm_eps` is not finite and positive.
    BadNormEps {
        /// The value parsed.
        value: f64,
    },
    /// A rope theta is absent.
    MissingTheta {
        /// Dotted HF path of the absent field.
        field: &'static str,
    },
    /// A rope theta is not finite and positive.
    BadTheta {
        /// Dotted HF path of the field.
        field: &'static str,
        /// The value parsed.
        value: f64,
    },
    /// A bias flag is true; this encoder has no biases.
    UnsupportedBias {
        /// The HF field name.
        field: &'static str,
    },
    /// A weight-shape product overflows `usize`.
    DimensionOverflow {
        /// The product, e.g. `"vocab_size x hidden_size"`.
        field: &'static str,
    },
    /// `num_hidden_layers` is above [`MAX_NUM_HIDDEN_LAYERS`].
    TooManyLayers {
        /// The declared count.
        value: usize,
        /// The cap.
        max: usize,
    },
    /// `hidden_activation` is not the exact GELU this encoder computes.
    UnsupportedActivation {
        /// The declared activation.
        value: String,
    },
    /// A RoPE variant or scaling other than unscaled `default` RoPE.
    UnsupportedRope {
        /// Dotted HF path of the field.
        field: &'static str,
        /// The declared value.
        value: String,
    },
}

impl fmt::Display for ModernBertConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(e) => write!(f, "modernbert config: invalid JSON: {e}"),
            Self::ZeroDimension { field } => write!(f, "modernbert config: {field} must be > 0"),
            Self::HeadsDoNotDivideHidden {
                hidden_size,
                num_attention_heads,
            } => write!(
                f,
                "modernbert config: num_attention_heads {num_attention_heads} does not divide hidden_size {hidden_size}"
            ),
            Self::OddHeadDim { head_dim } => write!(
                f,
                "modernbert config: head dim {head_dim} is odd (rotate-half RoPE needs it even)"
            ),
            Self::OddLocalAttention { local_attention } => write!(
                f,
                "modernbert config: local_attention {local_attention} is odd (window is local_attention / 2 per side)"
            ),
            Self::LayerTypesLength { expected, observed } => write!(
                f,
                "modernbert config: layer_types has {observed} entries, num_hidden_layers is {expected}"
            ),
            Self::UnknownLayerType { layer, value } => write!(
                f,
                "modernbert config: layer_types[{layer}] = {value:?} is not {FULL_ATTENTION} or {SLIDING_ATTENTION}"
            ),
            Self::LayerTypesDisagree { layer } => write!(
                f,
                "modernbert config: layer_types[{layer}] disagrees with layer % global_attn_every_n_layers"
            ),
            Self::BadNormEps { value } => {
                write!(f, "modernbert config: norm_eps {value} must be finite and > 0")
            }
            Self::MissingTheta { field } => write!(f, "modernbert config: {field} is missing"),
            Self::BadTheta { field, value } => {
                write!(f, "modernbert config: {field} = {value} must be finite and > 0")
            }
            Self::UnsupportedBias { field } => write!(
                f,
                "modernbert config: {field} = true is unsupported (this encoder has no biases)"
            ),
            Self::DimensionOverflow { field } => {
                write!(f, "modernbert config: {field} overflows usize")
            }
            Self::TooManyLayers { value, max } => write!(
                f,
                "modernbert config: num_hidden_layers {value} is over the supported {max}"
            ),
            Self::UnsupportedActivation { value } => write!(
                f,
                "modernbert config: hidden_activation {value:?} is unsupported (this encoder computes {SUPPORTED_ACTIVATION:?})"
            ),
            Self::UnsupportedRope { field, value } => write!(
                f,
                "modernbert config: {field} = {value} is unsupported (this encoder computes unscaled {SUPPORTED_ROPE_TYPE:?} RoPE)"
            ),
        }
    }
}

impl std::error::Error for ModernBertConfigError {}

#[derive(Deserialize)]
struct RawRope {
    rope_theta: Option<f64>,
    /// HF selects the RoPE init function (and its scaling) by this; older configs spell it
    /// `type`. Read so a non-default variant is refused rather than computed as default.
    #[serde(default, alias = "type")]
    rope_type: Option<String>,
}

#[derive(Deserialize)]
struct RawRopeParameters {
    full_attention: Option<RawRope>,
    sliding_attention: Option<RawRope>,
}

/// The HF fields this encoder reads. Every other HF field is ignored, except those that
/// change the forward (`hidden_activation`, `rope_type`, `rope_scaling`), which are read so
/// an unsupported value is refused by name.
#[derive(Deserialize)]
struct RawConfig {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    global_attn_every_n_layers: usize,
    local_attention: usize,
    norm_eps: f64,
    #[serde(default)]
    layer_types: Option<Vec<String>>,
    #[serde(default)]
    rope_parameters: Option<RawRopeParameters>,
    #[serde(default)]
    norm_bias: bool,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    mlp_bias: bool,
    /// HF default `"gelu"` (exact erf GELU) when absent.
    #[serde(default)]
    hidden_activation: Option<String>,
    /// The legacy scaling block HF folds into `rope_parameters`; only absent/null is served.
    #[serde(default)]
    rope_scaling: Option<serde_json::Value>,
}

/// A validated ModernBERT configuration. Only [`ModernBertConfig::from_json_bytes`]
/// constructs one, so every instance is inside the supported domain.
#[derive(Debug, Clone, PartialEq)]
pub struct ModernBertConfig {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    global_attn_every_n_layers: usize,
    local_attention: usize,
    norm_eps: f64,
    rope_theta_global: f64,
    rope_theta_local: f64,
    layer_is_global: Vec<bool>,
}

fn positive_finite(v: f64) -> bool {
    v.is_finite() && matches!(v.partial_cmp(&0.0), Some(Ordering::Greater))
}

fn theta(rope: Option<&RawRope>, field: &'static str) -> Result<f64, ModernBertConfigError> {
    let value = rope
        .and_then(|r| r.rope_theta)
        .ok_or(ModernBertConfigError::MissingTheta { field })?;
    if positive_finite(value) {
        Ok(value)
    } else {
        Err(ModernBertConfigError::BadTheta { field, value })
    }
}

fn checked_product(dims: &[usize], field: &'static str) -> Result<usize, ModernBertConfigError> {
    dims.iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or(ModernBertConfigError::DimensionOverflow { field })
}

impl ModernBertConfig {
    /// Parse and validate an HF ModernBERT `config.json`.
    ///
    /// # Errors
    ///
    /// [`ModernBertConfigError`] naming the field for every config outside the
    /// supported domain (see the module docs of `models::modernbert`).
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, ModernBertConfigError> {
        let raw: RawConfig = serde_json::from_slice(bytes)
            .map_err(|e| ModernBertConfigError::Json(e.to_string()))?;
        Self::validate(raw)
    }

    fn validate(raw: RawConfig) -> Result<Self, ModernBertConfigError> {
        for (field, v) in [
            ("vocab_size", raw.vocab_size),
            ("hidden_size", raw.hidden_size),
            ("intermediate_size", raw.intermediate_size),
            ("num_hidden_layers", raw.num_hidden_layers),
            ("num_attention_heads", raw.num_attention_heads),
            ("global_attn_every_n_layers", raw.global_attn_every_n_layers),
            ("local_attention", raw.local_attention),
        ] {
            if v == 0 {
                return Err(ModernBertConfigError::ZeroDimension { field });
            }
        }
        // Before anything per-layer is derived: the count is untrusted input.
        if raw.num_hidden_layers > MAX_NUM_HIDDEN_LAYERS {
            return Err(ModernBertConfigError::TooManyLayers {
                value: raw.num_hidden_layers,
                max: MAX_NUM_HIDDEN_LAYERS,
            });
        }
        if raw.hidden_size % raw.num_attention_heads != 0 {
            return Err(ModernBertConfigError::HeadsDoNotDivideHidden {
                hidden_size: raw.hidden_size,
                num_attention_heads: raw.num_attention_heads,
            });
        }
        let head_dim = raw.hidden_size / raw.num_attention_heads;
        if head_dim % 2 != 0 {
            return Err(ModernBertConfigError::OddHeadDim { head_dim });
        }
        if raw.local_attention % 2 != 0 {
            return Err(ModernBertConfigError::OddLocalAttention {
                local_attention: raw.local_attention,
            });
        }
        let declared = match &raw.layer_types {
            None => None,
            Some(types) => {
                if types.len() != raw.num_hidden_layers {
                    return Err(ModernBertConfigError::LayerTypesLength {
                        expected: raw.num_hidden_layers,
                        observed: types.len(),
                    });
                }
                let mut globals = Vec::with_capacity(types.len());
                for (layer, t) in types.iter().enumerate() {
                    match t.as_str() {
                        FULL_ATTENTION => globals.push(true),
                        SLIDING_ATTENTION => globals.push(false),
                        _ => {
                            return Err(ModernBertConfigError::UnknownLayerType {
                                layer,
                                value: t.clone(),
                            })
                        }
                    }
                }
                Some(globals)
            }
        };
        if !positive_finite(raw.norm_eps) {
            return Err(ModernBertConfigError::BadNormEps {
                value: raw.norm_eps,
            });
        }
        for (field, on) in [
            ("norm_bias", raw.norm_bias),
            ("attention_bias", raw.attention_bias),
            ("mlp_bias", raw.mlp_bias),
        ] {
            if on {
                return Err(ModernBertConfigError::UnsupportedBias { field });
            }
        }
        if let Some(act) = raw
            .hidden_activation
            .as_deref()
            .filter(|a| *a != SUPPORTED_ACTIVATION)
        {
            return Err(ModernBertConfigError::UnsupportedActivation {
                value: act.to_string(),
            });
        }
        if let Some(scaling) = raw.rope_scaling.as_ref().filter(|s| !s.is_null()) {
            return Err(ModernBertConfigError::UnsupportedRope {
                field: "rope_scaling",
                value: scaling.to_string(),
            });
        }
        let rope = raw.rope_parameters.as_ref();
        for (field, r) in [
            (
                "rope_parameters.full_attention.rope_type",
                rope.and_then(|r| r.full_attention.as_ref()),
            ),
            (
                "rope_parameters.sliding_attention.rope_type",
                rope.and_then(|r| r.sliding_attention.as_ref()),
            ),
        ] {
            if let Some(t) = r
                .and_then(|r| r.rope_type.as_deref())
                .filter(|t| *t != SUPPORTED_ROPE_TYPE)
            {
                return Err(ModernBertConfigError::UnsupportedRope {
                    field,
                    value: t.to_string(),
                });
            }
        }
        let rope_theta_global = theta(
            rope.and_then(|r| r.full_attention.as_ref()),
            "rope_parameters.full_attention.rope_theta",
        )?;
        let rope_theta_local = theta(
            rope.and_then(|r| r.sliding_attention.as_ref()),
            "rope_parameters.sliding_attention.rope_theta",
        )?;
        checked_product(
            &[raw.vocab_size, raw.hidden_size],
            "vocab_size x hidden_size",
        )?;
        checked_product(
            &[3, raw.hidden_size, raw.hidden_size],
            "3 x hidden_size x hidden_size",
        )?;
        checked_product(
            &[2, raw.intermediate_size, raw.hidden_size],
            "2 x intermediate_size x hidden_size",
        )?;
        let modulo: Vec<bool> = (0..raw.num_hidden_layers)
            .map(|i| i % raw.global_attn_every_n_layers == 0)
            .collect();
        if let Some(declared) = declared {
            if let Some(layer) = declared.iter().zip(&modulo).position(|(a, b)| a != b) {
                return Err(ModernBertConfigError::LayerTypesDisagree { layer });
            }
        }
        Ok(Self {
            vocab_size: raw.vocab_size,
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            global_attn_every_n_layers: raw.global_attn_every_n_layers,
            local_attention: raw.local_attention,
            norm_eps: raw.norm_eps,
            rope_theta_global,
            rope_theta_local,
            layer_is_global: modulo,
        })
    }

    /// Vocabulary size (rows of `tok_embeddings`).
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Hidden size `d`.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// MLP intermediate size (`Wi` has `2 x` this many rows).
    pub fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }

    /// Number of encoder layers.
    pub fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }

    /// Number of attention heads.
    pub fn num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }

    /// `hidden_size / num_attention_heads` (even by construction).
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Global-attention interval.
    pub fn global_attn_every_n_layers(&self) -> usize {
        self.global_attn_every_n_layers
    }

    /// `local_attention` as configured (the full window width).
    pub fn local_attention(&self) -> usize {
        self.local_attention
    }

    /// The local half-window: a local layer keeps `|i - j| <= window()`.
    pub fn window(&self) -> usize {
        self.local_attention / 2
    }

    /// LayerNorm epsilon, honoured by every layer norm.
    pub fn norm_eps(&self) -> f64 {
        self.norm_eps
    }

    /// RoPE theta on global layers.
    pub fn rope_theta_global(&self) -> f64 {
        self.rope_theta_global
    }

    /// RoPE theta on local layers.
    pub fn rope_theta_local(&self) -> f64 {
        self.rope_theta_local
    }

    /// Per-layer global flag (`layer_types` reconciled with the modulo rule).
    pub fn layer_is_global(&self) -> &[bool] {
        &self.layer_is_global
    }
}

#[cfg(test)]
mod tests {
    use super::{ModernBertConfig, ModernBertConfigError};
    use serde_json::{json, Value};

    /// The tiny fixture's config as a JSON value, for per-test mutation.
    fn base() -> Value {
        let bytes = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/modernbert_tiny/config.json"),
        )
        .expect("read modernbert_tiny/config.json");
        serde_json::from_slice(&bytes).expect("config json")
    }

    fn parse(v: &Value) -> Result<ModernBertConfig, ModernBertConfigError> {
        ModernBertConfig::from_json_bytes(&serde_json::to_vec(v).expect("serialize"))
    }

    fn with(mutate: impl FnOnce(&mut Value)) -> Result<ModernBertConfig, ModernBertConfigError> {
        let mut v = base();
        mutate(&mut v);
        parse(&v)
    }

    #[test]
    fn parses_the_tiny_fixture() {
        let c = parse(&base()).expect("fixture config is supported");
        assert_eq!(
            (c.vocab_size(), c.hidden_size(), c.intermediate_size()),
            (512, 32, 64)
        );
        assert_eq!(
            (c.num_hidden_layers(), c.num_attention_heads(), c.head_dim()),
            (3, 2, 16)
        );
        assert_eq!((c.local_attention(), c.window()), (8, 4));
        assert_eq!(c.layer_is_global(), &[true, false, false]);
        assert!((c.rope_theta_global() - 160_000.0).abs() < 1e-9);
        assert!((c.rope_theta_local() - 10_000.0).abs() < 1e-9);
        assert!((c.norm_eps() - 1e-5).abs() < 1e-18);
    }

    /// Every structural violation is refused with the matching typed error naming the
    /// field, before any shape is derived — the nine refusals of the supported domain.
    #[test]
    fn config_domain() {
        use ModernBertConfigError as E;
        let cases: Vec<(&str, Box<dyn FnOnce(&mut Value)>, E)> = vec![
            (
                "zero hidden_size",
                Box::new(|v| v["hidden_size"] = json!(0)),
                E::ZeroDimension {
                    field: "hidden_size",
                },
            ),
            (
                "zero num_attention_heads",
                Box::new(|v| v["num_attention_heads"] = json!(0)),
                E::ZeroDimension {
                    field: "num_attention_heads",
                },
            ),
            (
                "heads do not divide hidden",
                Box::new(|v| {
                    v["hidden_size"] = json!(30);
                    v["num_attention_heads"] = json!(4);
                }),
                E::HeadsDoNotDivideHidden {
                    hidden_size: 30,
                    num_attention_heads: 4,
                },
            ),
            (
                "odd head dim",
                Box::new(|v| {
                    v["hidden_size"] = json!(6);
                    v["num_attention_heads"] = json!(2);
                }),
                E::OddHeadDim { head_dim: 3 },
            ),
            (
                "odd local_attention",
                Box::new(|v| v["local_attention"] = json!(7)),
                E::OddLocalAttention { local_attention: 7 },
            ),
            (
                "zero global_attn_every_n_layers",
                Box::new(|v| v["global_attn_every_n_layers"] = json!(0)),
                E::ZeroDimension {
                    field: "global_attn_every_n_layers",
                },
            ),
            (
                "layer_types one short",
                Box::new(|v| v["layer_types"] = json!(["full_attention", "sliding_attention"])),
                E::LayerTypesLength {
                    expected: 3,
                    observed: 2,
                },
            ),
            (
                "unknown layer type",
                Box::new(|v| {
                    v["layer_types"] =
                        json!(["full_attention", "chunked_attention", "sliding_attention"]);
                }),
                E::UnknownLayerType {
                    layer: 1,
                    value: "chunked_attention".to_string(),
                },
            ),
            (
                "vocab_size x hidden_size overflows usize",
                Box::new(|v| v["vocab_size"] = json!(1u64 << 60)),
                E::DimensionOverflow {
                    field: "vocab_size x hidden_size",
                },
            ),
        ];
        assert_eq!(cases.len(), 9, "the nine refusals of the supported domain");
        for (label, mutate, want) in cases {
            let got = with(mutate);
            assert_eq!(got, Err(want), "{label}");
        }
    }

    /// `norm_eps` and the thetas must be finite and positive.
    #[test]
    fn config_domain_eps_and_theta() {
        assert_eq!(
            with(|v| v["norm_eps"] = json!(0.0)),
            Err(ModernBertConfigError::BadNormEps { value: 0.0 })
        );
        assert_eq!(
            with(|v| v["rope_parameters"]["full_attention"]["rope_theta"] = json!(-1.0)),
            Err(ModernBertConfigError::BadTheta {
                field: "rope_parameters.full_attention.rope_theta",
                value: -1.0
            })
        );
    }

    /// `layer_types` must agree with `i % global_attn_every_n_layers == 0`; without
    /// `layer_types` the modulo rule alone decides.
    #[test]
    fn layer_types_agree() {
        let got = with(|v| {
            v["layer_types"] = json!([
                "sliding_attention",
                "sliding_attention",
                "sliding_attention"
            ]);
        });
        assert_eq!(
            got,
            Err(ModernBertConfigError::LayerTypesDisagree { layer: 0 })
        );
        let fallback = with(|v| {
            v.as_object_mut().expect("object").remove("layer_types");
            v["num_hidden_layers"] = json!(7);
        })
        .expect("modulo fallback");
        assert_eq!(
            fallback.layer_is_global(),
            &[true, false, false, true, false, false, true]
        );
    }

    /// Unsupported features are refused by name, never silently ignored.
    #[test]
    fn unsupported_config() {
        assert_eq!(
            with(|v| v["norm_bias"] = json!(true)),
            Err(ModernBertConfigError::UnsupportedBias { field: "norm_bias" })
        );
        assert_eq!(
            with(|v| {
                v["rope_parameters"]
                    .as_object_mut()
                    .expect("rope_parameters")
                    .remove("sliding_attention");
            }),
            Err(ModernBertConfigError::MissingTheta {
                field: "rope_parameters.sliding_attention.rope_theta"
            })
        );
        let err = with(|v| {
            v.as_object_mut().expect("object").remove("hidden_size");
        })
        .expect_err("missing hidden_size");
        assert!(
            matches!(&err, ModernBertConfigError::Json(m) if m.contains("hidden_size")),
            "{err:?}"
        );
    }

    /// Fields HF honours in the forward are refused unless they name what this encoder
    /// computes: exact GELU and unscaled default RoPE (an HF config with SiLU or YaRN would
    /// otherwise load and produce silently different hidden states).
    #[test]
    fn unsupported_forward_semantics_are_refused() {
        use ModernBertConfigError as E;
        assert_eq!(
            with(|v| v["hidden_activation"] = json!("silu")),
            Err(E::UnsupportedActivation {
                value: "silu".to_string()
            })
        );
        assert_eq!(
            with(|v| v["rope_parameters"]["full_attention"]["rope_type"] = json!("yarn")),
            Err(E::UnsupportedRope {
                field: "rope_parameters.full_attention.rope_type",
                value: "yarn".to_string()
            })
        );
        assert_eq!(
            with(|v| {
                let s = v["rope_parameters"]["sliding_attention"]
                    .as_object_mut()
                    .expect("sliding_attention");
                s.remove("rope_type");
                s.insert("type".to_string(), json!("linear"));
            }),
            Err(E::UnsupportedRope {
                field: "rope_parameters.sliding_attention.rope_type",
                value: "linear".to_string()
            })
        );
        assert!(matches!(
            with(|v| v["rope_scaling"] = json!({"rope_type": "linear", "factor": 4.0})),
            Err(E::UnsupportedRope {
                field: "rope_scaling",
                ..
            })
        ));
        // The supported spellings still parse: explicit gelu/default, and both absent.
        with(|v| {
            v["hidden_activation"] = json!("gelu");
            v["rope_scaling"] = Value::Null;
        })
        .expect("gelu / default / null scaling");
        with(|v| {
            let o = v.as_object_mut().expect("object");
            o.remove("hidden_activation");
            o.remove("rope_scaling");
            for layer in ["full_attention", "sliding_attention"] {
                v["rope_parameters"][layer]
                    .as_object_mut()
                    .expect("rope entry")
                    .remove("rope_type");
            }
        })
        .expect("absent activation / rope_type / scaling mean the HF defaults");
    }

    /// An untrusted layer count is capped BEFORE the per-layer flags are collected: a huge
    /// declared count is a typed refusal, never an allocation proportional to it.
    #[test]
    fn layer_count_is_capped_before_allocation() {
        let huge = with(|v| {
            v.as_object_mut().expect("object").remove("layer_types");
            v["num_hidden_layers"] = json!(1u64 << 62);
        });
        assert_eq!(
            huge,
            Err(ModernBertConfigError::TooManyLayers {
                value: 1usize << 62,
                max: super::MAX_NUM_HIDDEN_LAYERS,
            })
        );
        let at_cap = with(|v| {
            v.as_object_mut().expect("object").remove("layer_types");
            v["num_hidden_layers"] = json!(super::MAX_NUM_HIDDEN_LAYERS);
        })
        .expect("the cap itself is supported");
        assert_eq!(at_cap.num_hidden_layers(), super::MAX_NUM_HIDDEN_LAYERS);
    }
}
