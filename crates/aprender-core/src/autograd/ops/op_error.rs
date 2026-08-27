
// ============================================================================
// Typed errors for the SetFit differentiable primitives
// Contract: setfit-encoder-conformance-v1
// ============================================================================

/// Failure modes of the batched SetFit primitives in [`crate::autograd`].
///
/// These ops sit on a **trust boundary**: ids, masks and shapes arrive from
/// tokenizer output today and from model files from Phase 4 onward, so every
/// argument is untrusted. Each variant therefore names the *specific* condition
/// that failed rather than collapsing to a generic "bad input".
///
/// Two known traps in this repository are exactly what this enum exists to
/// close:
///
/// * `crates/aprender-train/src/transformer/embedding.rs` silently **zero-fills**
///   out-of-vocabulary ids. A zero row is indistinguishable from a legitimately
///   zero embedding downstream, so the corruption never raises an error — it
///   only ever surfaces as unexplained accuracy loss.
/// * `crates/aprender-core/src/models/bert/embeddings.rs` `assert!`s on
///   over-length input and then slices unchecked.
///
/// Neither is acceptable here. Every op that returns this error fails **closed**:
/// it never panics, never zero-fills, and never lets a NaN into the autograd
/// graph.
///
/// Plans 01-03 and 01-09 extend this enum with further variants. Do **not**
/// rename the existing ones — they are named in
/// `contracts/setfit-encoder-conformance-v1.yaml`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpError {
    /// A token id was at or beyond `vocab_size`.
    ///
    /// `position` is the index into the FLATTENED `B*S` id slice, so a caller
    /// can recover `(batch, seq)` as `(position / seq, position % seq)`.
    OutOfVocabulary {
        /// The offending token id.
        id: u32,
        /// Rows available in the embedding table.
        vocab_size: usize,
        /// Flattened `b * seq + s` index of the offending id.
        position: usize,
    },

    /// A tensor did not have the required shape.
    ///
    /// A `0` extent inside `expected` means **unconstrained** — it encodes a
    /// rank requirement whose extents the op cannot know in advance. `expected:
    /// [0, 0]` therefore reads as "any 2-D shape".
    ShapeMismatch {
        /// Required shape; `0` marks an unconstrained extent.
        expected: Vec<usize>,
        /// Shape actually supplied.
        got: Vec<usize>,
    },

    /// A batch row had no valid (non-padding) position.
    ///
    /// This is the checked-denominator guard (D-03). Pooling such a row would
    /// divide by zero, and masking it would produce an all-`-1e9` softmax row.
    AllPaddingRow {
        /// Index of the offending batch row.
        row: usize,
    },

    /// A mask length did not match the position count implied by the shape.
    ///
    /// `ids` carries the **expected** element count derived from the batch and
    /// sequence dimensions; `mask` carries the length actually supplied.
    LengthMismatch {
        /// Expected number of positions (`batch * seq`).
        ids: usize,
        /// Length of the mask slice actually supplied.
        mask: usize,
    },

    /// A dimension was zero.
    ///
    /// Returned instead of an empty tensor, because an empty tensor silently
    /// no-ops every downstream op and the failure then surfaces far from its
    /// cause.
    ZeroDimension {
        /// Which dimension was zero (`"batch"`, `"seq"`, `"hidden"`, …).
        which: &'static str,
    },

    /// A shape product would overflow `usize`.
    ///
    /// Detected with `checked_mul` **before** any allocation, so a wrapping
    /// element count can never become an under-sized buffer.
    ShapeOverflow {
        /// The dimensions whose product overflowed.
        dims: Vec<usize>,
    },

    /// An attention-mask entry was neither `0` nor `1`.
    ///
    /// A `2` must never be silently treated as "keep": that would let a
    /// malformed mask quietly widen attention over padding.
    NonBinaryMaskValue {
        /// The offending value.
        value: u8,
        /// Flattened index of the offending value.
        position: usize,
    },

    /// An input tensor contained a non-finite value (`NaN` or `±Inf`).
    ///
    /// Rejected at the op boundary so a corrupt weight cannot poison every
    /// downstream gradient with a NaN that is untraceable to its source.
    NonFiniteInput {
        /// Flattened index of the offending element.
        position: usize,
    },

    /// The epsilon floor was not a positive, finite number (plan 01-03).
    ///
    /// `l2_normalize_rows` and `cosine_similarity_rows` divide by
    /// `max(norm, eps)`. A zero, negative, `NaN` or infinite `eps` therefore
    /// removes the only guard standing between a zero-norm row and a `NaN` (or,
    /// with a negative `eps`, silently flips the sign of a whole row). The floor
    /// is an explicit parameter with no hidden default precisely so that it can
    /// be validated here.
    ///
    /// # Why the value is stored as BITS rather than as an `f32`
    ///
    /// Two independent reasons, both of which bite:
    ///
    /// 1. `OpError` derives `Eq`. An `f32` field would forbid that derive for
    ///    the whole enum, changing the API of seven pre-existing variants for
    ///    the sake of one.
    /// 2. `NaN != NaN` under `PartialEq`. Had the variant carried an `f32`,
    ///    `assert_eq!(err, OpError::InvalidEpsilon { eps: f32::NAN })` would be
    ///    **unsatisfiable against a correct implementation** — for exactly the
    ///    `NaN` input this variant exists to reject. That is the same class of
    ///    self-defeating assertion the ENC-04 gradient gate was rewritten to
    ///    avoid.
    ///
    /// [`OpError::epsilon`] recovers the original value for display or
    /// inspection.
    InvalidEpsilon {
        /// IEEE-754 bit pattern of the offending epsilon.
        eps_bits: u32,
    },
}

impl OpError {
    /// Build an [`OpError::InvalidEpsilon`] from the offending value.
    pub(crate) fn invalid_epsilon(eps: f32) -> Self {
        Self::InvalidEpsilon {
            eps_bits: eps.to_bits(),
        }
    }

    /// Recover the epsilon carried by [`OpError::InvalidEpsilon`].
    ///
    /// Returns `None` for every other variant.
    #[must_use]
    pub fn epsilon(&self) -> Option<f32> {
        match self {
            Self::InvalidEpsilon { eps_bits } => Some(f32::from_bits(*eps_bits)),
            _ => None,
        }
    }
}

impl std::fmt::Display for OpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfVocabulary {
                id,
                vocab_size,
                position,
            } => write!(
                f,
                "OpError::OutOfVocabulary(id {id} >= vocab_size {vocab_size} at flat position {position})"
            ),
            Self::ShapeMismatch { expected, got } => {
                let want: Vec<String> = expected
                    .iter()
                    .map(|d| if *d == 0 { "*".to_string() } else { d.to_string() })
                    .collect();
                write!(
                    f,
                    "OpError::ShapeMismatch(expected [{}], got {got:?})",
                    want.join(", ")
                )
            }
            Self::AllPaddingRow { row } => write!(
                f,
                "OpError::AllPaddingRow(row {row} has no valid position; denominator would be zero)"
            ),
            Self::LengthMismatch { ids, mask } => write!(
                f,
                "OpError::LengthMismatch(expected {ids} positions, got a slice of length {mask})"
            ),
            Self::ZeroDimension { which } => {
                write!(f, "OpError::ZeroDimension({which} is zero)")
            }
            Self::ShapeOverflow { dims } => write!(
                f,
                "OpError::ShapeOverflow(product of {dims:?} overflows usize)"
            ),
            Self::NonBinaryMaskValue { value, position } => write!(
                f,
                "OpError::NonBinaryMaskValue({value} at flat position {position}; only 0 and 1 are valid)"
            ),
            Self::NonFiniteInput { position } => write!(
                f,
                "OpError::NonFiniteInput(non-finite value at flat position {position})"
            ),
            Self::InvalidEpsilon { eps_bits } => write!(
                f,
                "OpError::InvalidEpsilon({}; the epsilon floor must be finite and > 0)",
                f32::from_bits(*eps_bits)
            ),
        }
    }
}

impl std::error::Error for OpError {}
