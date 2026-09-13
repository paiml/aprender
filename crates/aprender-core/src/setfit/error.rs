//! Typed failures of the SetFit import / tokenize / batch boundary.
//!
//! Contract: `setfit-encoder-conformance-v1` (ENC-01, ENC-02).
//!
//! Everything in this module sits on a trust boundary: `config.json`, tokenizer
//! bytes and `.apr` tensors are untrusted artifacts, and `encode_batch` takes
//! arbitrary user text. Each variant therefore names the *specific* field or
//! condition that failed, so a rejection is actionable rather than a generic
//! "bad model". No path here panics, `assert!`s, or silently substitutes a
//! default.

use crate::autograd::OpError;
use crate::models::bert::load::BertLoadError;

/// A SetFit import, tokenization, or batch-construction failure.
///
/// `Op` wraps [`OpError`] so the differentiable primitives (01-01/01-03), which
/// are deliberately ungated (D-03), compose with `?` inside functions that
/// return `SetFitError`. [`std::fmt::Display`] forwards to the inner error so an
/// op's own diagnostic (row index, flat position, offending vocab id) survives
/// the wrap instead of being flattened into a generic message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SetFitError {
    /// A modelled, behavior-affecting config field did not match the pin.
    ///
    /// `field` is the field's name **as it appears in `config.json`**, so the
    /// error text can be matched against the artifact the user actually edited.
    ImportConfigMismatch {
        /// Config field name (e.g. `"hidden_size"`).
        field: String,
        /// Value required by the pin.
        expected: String,
        /// Value found in the artifact.
        got: String,
    },

    /// A required file was missing or unreadable under the model directory.
    ImportIo {
        /// Path (relative to the model directory) that could not be read.
        path: String,
        /// One-line description of the underlying I/O or parse failure.
        reason: String,
    },

    /// A tensor read failed: missing, wrong dtype, or wrong element count.
    ImportTensor(BertLoadError),

    /// The tokenizer bytes could not be parsed by `tokenizers`.
    TokenizerLoad {
        /// One-line description from the tokenizers crate.
        reason: String,
    },

    /// The tokenizer bytes did not hash to the pinned digest.
    ///
    /// Distinct from [`Self::TokenizerLoad`]: the bytes parsed fine, they are
    /// simply not the tokenizer this model was pinned against.
    TokenizerHashMismatch {
        /// Pinned sha256 (hex, lowercase).
        expected: String,
        /// Sha256 of the bytes actually supplied.
        got: String,
    },

    /// The sentence-transformers module stack did not request mean pooling
    /// (plus the L2 `Normalize` module) exactly as the pin requires.
    UnsupportedPooling {
        /// The pooling mode actually configured.
        got: String,
    },

    /// `architectures` named something other than `BertModel`.
    UnsupportedArchitecture {
        /// The architecture actually declared.
        got: String,
    },

    /// `hidden_act` selected an activation this crate does not implement.
    ///
    /// `"gelu_new"` / `"gelu_pytorch_tanh"` land here: they select the *tanh
    /// approximation*, which differs from the pinned erf form by ~4.7e-4 — two
    /// orders above the frozen activation tolerance.
    UnsupportedActivation {
        /// The activation actually declared.
        got: String,
    },

    /// A batch could not be formed from the supplied inputs.
    BatchInvalid {
        /// Why the batch is not constructible (e.g. `"empty text list"`).
        reason: String,
    },

    /// An input exceeded the tokenizer's configured maximum length in a context
    /// where truncation was not permitted.
    OversizeInput {
        /// Token length of the offending input.
        len: usize,
        /// Maximum permitted length.
        max: usize,
    },

    /// A canonical vocabulary id fell outside the slice closure.
    ///
    /// Returned rather than zero-filling: a zero embedding row is
    /// indistinguishable from a legitimately zero row downstream, which is the
    /// exact `aprender-train` trap this phase exists to avoid.
    VocabOutOfSlice {
        /// The canonical id with no slice row.
        canonical_id: u32,
    },

    /// A loaded tensor contained a `NaN` or `±Inf`.
    NonFiniteTensor {
        /// Tensor name in the `.apr` file.
        tensor: String,
        /// Flattened index of the offending element.
        position: usize,
    },

    /// A vocabulary remap was internally inconsistent or out of range.
    RemapInvalid {
        /// Why the remap is unusable.
        reason: String,
    },

    /// A requested freeze group could not be applied (01-07, D-22).
    ///
    /// Two conditions reach here, and both are configuration errors rather than
    /// data errors: a layer index outside the encoder's layer range, and a
    /// structurally valid group whose prefix set addresses **zero** named
    /// parameters. The second is the naming-drift guard — a policy that
    /// silently freezes nothing is worse than one that fails, because the
    /// resulting run looks like a successful partial freeze.
    ///
    /// The stored freeze policy and every `requires_grad` flag are left
    /// UNCHANGED when this is returned: validation completes for every group
    /// before any flag is touched, so there is no partial application.
    FreezeGroupInvalid {
        /// Which group failed and why.
        reason: String,
    },

    /// A dropout-mask derivation rejected its coordinates (03-02, TRN-06).
    ///
    /// Two conditions reach here: an unusable dropout RATE at construction, and a
    /// forward-call ORDINAL that does not fit the `u32` counter lane.
    ///
    /// The payload is the rendered [`super::dropout_rng::DropoutRngError`] rather
    /// than the error itself, and that is a deliberate trade. `SetFitError`
    /// derives `Eq`, which the whole crate's rejection tests compare on;
    /// `DropoutRngError` carries `f32` payloads and therefore cannot be `Eq`.
    /// Wrapping it directly would have forced `Eq` off `SetFitError` and rewritten
    /// assertions across the module for one variant's benefit. The typed error is
    /// still available UNWRAPPED at the `dropout_rng` boundary, which is where the
    /// rate and ordinal gates are actually tested, and the rendered message keeps
    /// naming the offending value.
    DropoutRng {
        /// The rendered `DropoutRngError`, naming the offending value.
        reason: String,
    },

    /// A differentiable op rejected its arguments.
    Op(OpError),
}

impl From<OpError> for SetFitError {
    fn from(e: OpError) -> Self {
        Self::Op(e)
    }
}

impl From<super::dropout_rng::DropoutRngError> for SetFitError {
    fn from(e: super::dropout_rng::DropoutRngError) -> Self {
        Self::DropoutRng {
            reason: e.to_string(),
        }
    }
}

impl From<BertLoadError> for SetFitError {
    fn from(e: BertLoadError) -> Self {
        Self::ImportTensor(e)
    }
}

impl std::fmt::Display for SetFitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ImportConfigMismatch {
                field,
                expected,
                got,
            } => write!(
                f,
                "SetFitError::ImportConfigMismatch(field {field}: expected {expected}, got {got})"
            ),
            Self::ImportIo { path, reason } => {
                write!(f, "SetFitError::ImportIo({path}: {reason})")
            }
            // Forward, do not flatten: BertLoadError names the tensor and the
            // exact element-count mismatch.
            Self::ImportTensor(e) => write!(f, "SetFitError::ImportTensor({e})"),
            Self::TokenizerLoad { reason } => {
                write!(f, "SetFitError::TokenizerLoad({reason})")
            }
            Self::TokenizerHashMismatch { expected, got } => write!(
                f,
                "SetFitError::TokenizerHashMismatch(expected {expected}, got {got})"
            ),
            // Each of the three below names the CONFIG KEY it read, not only the
            // offending value: a rejection that says `"gelu_new"` without saying
            // which field carried it is not actionable, and the ENC-01 mutation
            // tests assert on the field name for exactly that reason.
            Self::UnsupportedPooling { got } => {
                write!(f, "SetFitError::UnsupportedPooling({got})")
            }
            Self::UnsupportedArchitecture { got } => {
                write!(
                    f,
                    "SetFitError::UnsupportedArchitecture(architectures = {got}; only BertModel is pinned)"
                )
            }
            Self::UnsupportedActivation { got } => {
                write!(
                    f,
                    "SetFitError::UnsupportedActivation(hidden_act = \"{got}\" is not the pinned exact-erf \"gelu\")"
                )
            }
            Self::BatchInvalid { reason } => {
                write!(f, "SetFitError::BatchInvalid({reason})")
            }
            Self::OversizeInput { len, max } => write!(
                f,
                "SetFitError::OversizeInput(length {len} exceeds maximum {max})"
            ),
            Self::VocabOutOfSlice { canonical_id } => write!(
                f,
                "SetFitError::VocabOutOfSlice(canonical id {canonical_id} is outside the slice closure)"
            ),
            Self::NonFiniteTensor { tensor, position } => write!(
                f,
                "SetFitError::NonFiniteTensor({tensor} has a non-finite value at flat position {position})"
            ),
            Self::RemapInvalid { reason } => {
                write!(f, "SetFitError::RemapInvalid({reason})")
            }
            Self::FreezeGroupInvalid { reason } => {
                write!(f, "SetFitError::FreezeGroupInvalid({reason})")
            }
            // Forward the dropout derivation's own diagnostic verbatim: the
            // offending rate or ordinal is already named inside `reason`.
            Self::DropoutRng { reason } => {
                write!(f, "SetFitError::DropoutRng({reason})")
            }
            // Forward the op's own diagnostic verbatim (W5).
            Self::Op(e) => write!(f, "SetFitError::Op({e})"),
        }
    }
}

impl std::error::Error for SetFitError {}

#[cfg(all(test, feature = "setfit"))]
mod tests {
    use super::*;

    #[test]
    fn setfit_error_op_wraps_and_forwards_the_inner_diagnostic() {
        let inner = OpError::OutOfVocabulary {
            id: 40_000,
            vocab_size: 97,
            position: 7,
        };
        // `?`-composability is the whole point of the variant (W5).
        let wrapped: SetFitError = inner.clone().into();
        assert_eq!(wrapped, SetFitError::Op(inner.clone()));
        let text = wrapped.to_string();
        // Not flattened: the op's own row/position/vocab detail survives.
        assert!(text.contains(&inner.to_string()), "got {text}");
        assert!(text.contains("40000"), "got {text}");
        assert!(text.contains("position 7"), "got {text}");
    }

    #[test]
    fn setfit_error_bert_load_error_wraps_and_names_the_tensor() {
        let inner = BertLoadError {
            tensor: "embeddings.word_embeddings.weight".to_string(),
            reason: "tensor not present in APR file".to_string(),
        };
        let wrapped: SetFitError = inner.clone().into();
        assert_eq!(wrapped, SetFitError::ImportTensor(inner));
        assert!(
            wrapped.to_string().contains("word_embeddings"),
            "got {wrapped}"
        );
    }

    #[test]
    fn setfit_error_display_names_the_mismatched_field() {
        let e = SetFitError::ImportConfigMismatch {
            field: "hidden_size".to_string(),
            expected: "384".to_string(),
            got: "512".to_string(),
        };
        let text = e.to_string();
        assert!(text.contains("hidden_size"), "got {text}");
        assert!(text.contains("384"), "got {text}");
        assert!(text.contains("512"), "got {text}");
    }

    #[test]
    fn setfit_error_is_a_std_error() {
        fn assert_error<E: std::error::Error>(_: &E) {}
        assert_error(&SetFitError::BatchInvalid {
            reason: "empty text list".to_string(),
        });
    }
}
