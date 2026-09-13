//! The differentiable MiniLM sentence encoder (ENC-03, ENC-05).
//!
//! Contract: `setfit-encoder-conformance-v1`, equation `setfit_encoder_forward`.
//!
//! # One forward implementation
//!
//! [`BertSentenceEncoder::forward_layers`] is the ONLY place a layer is run.
//! [`BertSentenceEncoder::forward_tokens`] returns its last layer output and
//! [`BertSentenceEncoder::forward_tokens_per_layer`] returns the whole
//! `(embeddings_out, layer_outputs)` pair. Collecting the intermediates is
//! unconditional — the `Vec` is built in every build — so the conformance build
//! and the production build compute identically; only the public accessor is
//! `cfg`-gated. A `to_bits` test asserts the two entry points agree elementwise,
//! which is what makes "one implementation" structural rather than aspirational
//! (T-1-28): a per-layer parity gate that compared against a path production
//! never runs would localize failures in a model nobody ships.
//!
//! # Composition
//!
//! Built ONLY from `nn/` building blocks and the ungated autograd primitives.
//! `models/bert/` is untouched beyond 01-05's sanctioned A-01 `read_tensor`
//! visibility line (D-01) — in particular the BERT embeddings module, which
//! `assert!`s on over-length input and then slices unchecked, is never reached.
//!
//! Every intermediate flows through an autograd-aware operation. A
//! `Tensor::new` / `Tensor::from_vec` over a COMPUTED value with no adjacent
//! `grad_fn` is the PMAT-913/914/922 severed-graph class and is forbidden here;
//! the only raw tensors this module builds are the position/token-type id
//! vectors, which are integers and carry no gradient by construction.
//!
//! # Mode (ENC-05)
//!
//! [`Module::set_training`] is the recursive propagation channel (D-17) and
//! [`Module::train`] / [`Module::eval`] delegate to it, so both spellings flip
//! every dropout site including the one inside `MultiHeadAttention`. Flipping
//! mode never adds, removes or mutates a registered parameter.
//!
//! `from_import` returns an encoder in **eval** mode, matching HuggingFace
//! `from_pretrained`, which calls `model.eval()` before handing the model back.
//! That is also the mode the frozen fixtures were generated in (D-16). Training
//! callers flip it explicitly with `set_training(true)`.

// D32 CLOSED (01-07): the module-wide `#![allow(dead_code)]` 01-06 added here is
// GONE. 01-06 recorded that `from_import` was `pub(crate)` under the D-08 seal
// with no non-test caller, so a library-only build walked the whole construction
// path — `site_seed`, `DROPOUT_P`, `install_projection`, the site-name helpers —
// as unreachable. `SetFitMiniLm::from_pretrained_dir` is now that caller, and
// the removal was MEASURED rather than assumed: `cargo check -p aprender-core
// --features setfit` reports zero dead-code findings in this file afterwards.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::autograd::{
    additive_attention_mask, embedding_gather, l2_normalize_rows, masked_mean_pool, OpError, Tensor,
};
use crate::models::bert::load::{read_tensor, BertLoadError};
use crate::nn::transformer::AttentionDropoutMasks;
use crate::nn::{LayerNorm, Linear, Module, MultiHeadAttention};

use super::dropout_rng::{self, SiteDropout};
use super::error::SetFitError;
use super::import::{MiniLmImport, ModelDims, VocabRemap, PINNED_ACTIVATION};
use super::tokenizer::{SentenceBatch, MAX_SEQUENCE_LENGTH};
use super::EncoderArchitecture;

/// Epsilon of the trailing L2 normalization, matching the pinned
/// sentence-transformers `Normalize` module.
///
/// `pub` so the pair objective (01-07 `setfit/loss.rs`) clamps its cosine norms
/// with the SAME constant this encoder normalized with, and so a persistence
/// artifact (plan 03-08) records the constant that was actually applied rather
/// than a literal of its own. Same single-source-of-truth reasoning 01-06 applied
/// to the two pinned dropout probabilities.
pub const L2_EPS: f32 = 1e-12;

/// The pooling policy this encoder implements: masked mean over the attention mask.
///
/// Named here, beside [`BertSentenceEncoder::encode`] which applies it, so that a
/// downstream artifact records the policy from its definition rather than from a
/// string it believes to be true.
pub const POOLING_POLICY: &str = "masked_mean";

/// The normalization policy this encoder implements: row-wise L2 at [`L2_EPS`].
pub const NORMALIZATION_POLICY: &str = "l2";

/// Dropout probability at every HF-verified site.
///
/// Single source of truth check: the ENC-01 pin rejects any checkpoint whose
/// `hidden_dropout_prob` or `attention_probs_dropout_prob` differs from `0.1`
/// (`import::PINNED_HIDDEN_DROPOUT_PROB` / `PINNED_ATTENTION_DROPOUT_PROB`), and
/// `encoder_dropout_probability_agrees_with_the_enc01_pin` asserts this constant
/// equals both of them at `f32`. So the value cannot drift in one place only.
const DROPOUT_P: f32 = 0.1;

// ---------------------------------------------------------------------------
// Dropout site naming and seeding
// ---------------------------------------------------------------------------

/// Site 1: embeddings, after LayerNorm.
const EMBEDDINGS_DROPOUT_SITE: &str = "embeddings.dropout";

/// Site 2 of layer `i`: attention probabilities, after softmax and before `@V`.
///
/// Named for HF's `BertSelfAttention.dropout`.
fn attention_probs_site(layer: usize) -> String {
    format!("encoder.layer.{layer}.attention.self.dropout")
}

/// Site 3 of layer `i`: attention output dense, before the residual add.
fn attention_output_site(layer: usize) -> String {
    format!("encoder.layer.{layer}.attention.output.dropout")
}

/// Site 4 of layer `i`: FFN output dense, before the residual add.
fn ffn_output_site(layer: usize) -> String {
    format!("encoder.layer.{layer}.output.dropout")
}

// THE site-keying decision (plan 03-02, resolving what 03-RESEARCH left open).
//
// There is now EXACTLY ONE derivation on the SetFit dropout path:
// `dropout_rng::derive_key(root_seed, dotted_site)`, i.e. a truncated SHA-256
// over `b"apr-setfit-dropout-v1\0" ‖ root_seed_le ‖ site`.
//
// Phase 1's `site_seed` — FNV-1a over the name folded through SplitMix64's
// finaliser (`nn::transformer::mix_call_seed`) — is GONE, along with
// `mix_call_seed` itself, which had no other caller in the workspace (measured,
// not assumed: the only two call sites were this function and the per-call seed
// advance inside `MultiHeadAttention`, both replaced here). Keeping both would
// have left two half-documented schemes deriving streams for the same four
// sites, and the failure mode of that is a site quietly keyed by the WRONG one
// after a refactor, which no test distinguishes from a legitimately different
// mask.
//
// What survives from `site_seed` is its actual insight, carried into
// `derive_key`'s documentation: key on the DOTTED NAME, never on a position.

/// The four dotted dropout sites of one encoder, in construction order.
///
/// Kept as a free function so the site names have exactly one spelling: the
/// builders above produce them and this is the only place they are turned into
/// mask sources.
///
/// # Errors
///
/// [`SetFitError::DropoutRng`] if `p` is not a usable dropout rate.
fn site_dropout(root_seed: u64, site: &str, p: f32) -> Result<Arc<SiteDropout>, SetFitError> {
    Ok(Arc::new(SiteDropout::new(root_seed, site, p)?))
}

// ---------------------------------------------------------------------------
// Layer
// ---------------------------------------------------------------------------

/// One HF BERT encoder layer (post-norm).
///
/// `attention.out_proj` IS `attention.output.dense`: `MultiHeadAttention`
/// applies it inside `forward_self`, so this struct holds only what comes after.
struct EncoderLayer {
    attention: MultiHeadAttention,
    /// Site 2, ALSO installed into `attention` as its mask source.
    ///
    /// Held here as well as there so introspection can read the site's dotted
    /// name and derived key directly. `MultiHeadAttention` sees it as an opaque
    /// `dyn AttentionDropoutMasks`, which is the right shape for `nn/` and the
    /// wrong shape for asking "which stream is this?".
    attention_probs_dropout: Arc<SiteDropout>,
    attention_output_dropout: Arc<SiteDropout>,
    attention_layer_norm: LayerNorm,
    intermediate: Linear,
    output_dense: Linear,
    output_dropout: Arc<SiteDropout>,
    output_layer_norm: LayerNorm,
}

// ---------------------------------------------------------------------------
// Execution-derived backend identity (D-12, review B6)
// ---------------------------------------------------------------------------

/// The identity of the compute path that ACTUALLY RAN an encode (D-12).
///
/// A value of this type is obtained in exactly one way: by calling
/// [`BertSentenceEncoder::encode_with_backend`] and receiving it back. There is
/// no public constructor, no setter, no `From` and no field a caller can write,
/// so the `backend` a response reports cannot be minted — only returned.
///
/// # What the identity means, and what it does not
///
/// (a) It names the kernel ENTRY POINT the encode path invoked. The SetFit
/// encoder's matmuls reach `trueno::Matrix::matmul` through `crate::autograd`
/// (`autograd/gradient.rs`, `matmul_2d`), and `autograd-trueno-matmul` is the
/// name of that entry point.
///
/// (b) trueno exposes NO per-dispatch execution report, and
/// `trueno::Matrix::matmul` selects among `matmul_naive`,
/// `blis::parallel::gemm_blis_parallel` and a GPU path BY SIZE
/// (`aprender-compute/src/matrix/ops/arithmetic.rs`, `SIMD_THRESHOLD == 64`).
/// So a CPU-feature detection value would describe the HOST rather than the run:
/// an AVX2 detection result is fully consistent with a scalar execution of this
/// very batch, because a small input takes the naive path regardless of what the
/// silicon can do.
///
/// (c) Reporting a capability value from trueno's `Backend` enum (its AVX2 /
/// NEON variants) or from its backend-selection family is therefore FORBIDDEN
/// here, and the conservative kernel-entry identity is deliberate rather than an
/// oversight. CLAUDE.md Verification Discipline rule 2: *never label a run by
/// intent — prove the mechanism engaged.* Reviewer B6's finding was exactly
/// this: detecting that AVX2 is AVAILABLE does not prove the encoder USED it.
///
/// Those symbol names are spelled here in PROSE, never as the literal tokens.
/// The D-12 gate greps this whole directory for them and requires zero matches,
/// so a doc comment quoting them verbatim would turn its own guard red — the
/// same defect orchestrator note F-05 records for `skip_serializing_if`, and it
/// is exactly what happened on this file's first green run.
///
/// (d) The v1 limitation is recorded rather than papered over.
/// `autograd-trueno-matmul` is a TRUE statement about what the encoder called
/// and an INCOMPLETE one about what silicon executed. Upgrading to per-dispatch
/// reporting requires a trueno API that does not exist; it is a recorded
/// DEFERRED ITEM, not a Phase 4 deliverable.
///
/// The grammar is three colon-separated segments and stays three, so a future
/// GPU identity is comparable to this one rather than a differently-shaped
/// string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionBackend {
    device: &'static str,
    kernel: &'static str,
}

impl ExecutionBackend {
    /// The three-segment identity string, `<device>:setfit-core:<kernel>`.
    #[must_use]
    pub fn identity(&self) -> String {
        format!("{}:setfit-core:{}", self.device, self.kernel)
    }

    /// The device segment.
    #[must_use]
    pub fn device(&self) -> &'static str {
        self.device
    }

    /// The kernel segment.
    #[must_use]
    pub fn kernel(&self) -> &'static str {
        self.kernel
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// A graph-connected BERT sentence encoder built from a validated import.
pub struct BertSentenceEncoder {
    word_embeddings: Tensor,
    position_embeddings: Tensor,
    token_type_embeddings: Tensor,
    embeddings_layer_norm: LayerNorm,
    embeddings_dropout: Arc<SiteDropout>,
    layers: Vec<EncoderLayer>,
    dims: ModelDims,
    /// `Some` for a slice fixture; `None` for the full pin.
    remap: Option<VocabRemap>,
    /// The LayerNorm epsilon every `LayerNorm` above was constructed with.
    ///
    /// RETAINED (plan 03-08) rather than only consumed: it is a behaviour
    /// constant of this encoder, and a rebuild that had to guess it would
    /// produce a structurally identical model with different arithmetic.
    layer_norm_eps: f32,
    /// The upstream revision this encoder's weights came from.
    ///
    /// Retained for the same reason: it is provenance an artifact must record,
    /// and it exists only on the import that is dropped after construction.
    source_revision: String,
    /// Sha256 of the tokenizer this encoder is paired with (D-08 defense in
    /// depth — the structural guarantee is the `pub(crate)` constructor).
    tokenizer_sha256: String,
    root_seed: u64,
    training: bool,
    /// D-15's `block`, mirrored so it can be read back.
    ///
    /// The authoritative copy lives in each [`SiteDropout`]; this field exists so
    /// a caller can ask what it last set without reaching into a site.
    forward_ordinal: u64,
}

impl BertSentenceEncoder {
    /// The architecture fingerprint: the dimensions that decide whether a calibration
    /// measured elsewhere applies to this encoder.
    ///
    /// Read-only and derived entirely from `dims`, which the constructor already validated.
    /// It exists because a gate that fails closed OUTSIDE its calibration regime has to be
    /// able to observe the architecture it is judging; without this it would be comparing a
    /// caller-supplied label, which is a claim rather than a measurement.
    ///
    /// Deliberately NOT behind `conformance-fixtures`: this is production behaviour, not test
    /// support. It exposes no tensor, no weight and no tokenizer material — only the shape
    /// numbers that are already implied by every parameter the model publishes.
    #[must_use]
    pub fn architecture_fingerprint(&self) -> String {
        format!(
            "minilm-slice-h{}-l{}-a{}-i{}-v{}",
            self.dims.hidden,
            self.dims.layers,
            self.dims.heads,
            self.dims.intermediate,
            self.dims.vocab,
        )
    }
}

impl std::fmt::Debug for BertSentenceEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BertSentenceEncoder")
            .field("dims", &self.dims)
            .field("max_seq", &self.max_seq())
            .field("is_slice", &self.remap.is_some())
            .field("training", &self.training)
            .finish_non_exhaustive()
    }
}

impl BertSentenceEncoder {
    /// Build an encoder from a validated import.
    ///
    /// SEALED (D-08, user decision 2026-08-08): `pub(crate)`. Out-of-crate
    /// callers construct via `SetFitMiniLm` (01-07), which pairs this encoder
    /// with the tokenizer from the same source, so a mismatched pair is not
    /// CONSTRUCTIBLE rather than merely detected. The read and forward methods
    /// stay `pub`.
    ///
    /// Returns an encoder in eval mode (see the module docs).
    ///
    /// # Errors
    ///
    /// [`SetFitError::ImportTensor`] if a tensor is missing or the wrong size.
    /// `MiniLmImport` has already validated presence, shape and finiteness of
    /// every tensor read here, so this is defense in depth rather than the
    /// primary gate.
    pub(crate) fn from_import(import: &MiniLmImport, root_seed: u64) -> Result<Self, SetFitError> {
        let reader = import.reader();
        let prefix = import.tensor_prefix();
        Self::assemble(
            import.dims().clone(),
            import.layer_norm_eps(),
            import.vocab_remap().cloned(),
            import.revision().to_string(),
            import.tokenizer_sha256().to_string(),
            root_seed,
            &|name: &str, shape: &[usize]| -> Result<Tensor, SetFitError> {
                // A-01 reuse: the checked-read semantics (presence, dtype path,
                // element count) come from the loader they were written for, they
                // are not reimplemented here.
                Ok(read_tensor(reader, &format!("{prefix}{name}"), shape)?.requires_grad())
            },
        )
    }

    /// Rebuild an encoder from a name -> (shape, data) map (plan 03-08, D-07).
    ///
    /// The reload counterpart of [`Self::from_import`]. It does not mirror that
    /// function's construction order — it IS that construction order:
    /// [`Self::assemble`] is the single body, and both constructors differ only
    /// in the `read` they supply. A copied second construction sequence is how a
    /// rebuilt encoder ends up structurally identical and behaviourally
    /// different, so there is no second sequence to diverge.
    ///
    /// SEALED (D-08): `pub(crate)`. The public door is
    /// `SetFitMiniLm::from_bundle_parts`, which pairs this with a tokenizer
    /// rebuilt from the same artifact.
    ///
    /// # Errors
    ///
    /// [`SetFitError::UnsupportedActivation`] if the record does not name the
    /// pinned activation; [`SetFitError::RemapInvalid`] if the remap does not
    /// describe this vocabulary; [`SetFitError::ImportTensor`] naming the tensor
    /// if one is missing, has the wrong shape, or the wrong element count.
    pub(crate) fn from_named_tensors(
        arch: &EncoderArchitecture,
        tensors: BTreeMap<String, (Vec<usize>, Vec<f32>)>,
        root_seed: u64,
    ) -> Result<Self, SetFitError> {
        if arch.hidden_act != PINNED_ACTIVATION {
            return Err(SetFitError::UnsupportedActivation {
                got: arch.hidden_act.clone(),
            });
        }
        if arch.heads == 0 || arch.head_dim.checked_mul(arch.heads) != Some(arch.hidden) {
            return Err(SetFitError::ImportConfigMismatch {
                field: "head_dim".to_string(),
                expected: format!("hidden / heads = {} / {}", arch.hidden, arch.heads),
                got: arch.head_dim.to_string(),
            });
        }
        // EVERY DIMENSION IS BOUNDED BY THE TENSORS ACTUALLY SUPPLIED, BEFORE `assemble`
        // ALLOCATES FROM IT.
        //
        // `assemble` builds a `LayerNorm` of width `hidden` and a `Vec` of capacity
        // `num_layers` before it reads a single tensor, so an architecture record is an
        // allocation request that arrives ahead of every validation the reader has. The
        // record travels inside a bundle whose four contracted bounds cover the input
        // length, the tensor count and the element counts -- and none of them covers this
        // struct, so `num_layers: 2^60` aborted the process on a payload of a few hundred
        // bytes, which is precisely the class of attack those bounds exist to refuse.
        //
        // The bound is not an arbitrary ceiling: every dimension indexes a tensor that must
        // be present, so a dimension larger than the total number of `f32`s supplied cannot
        // possibly be satisfied and the read that would discover it is already too late.
        // `num_layers` is bounded by the tensor COUNT for the same reason -- each layer
        // consumes named tensors of its own.
        let supplied_elements: usize = tensors.values().map(|(_, data)| data.len()).sum();
        for (field, value, ceiling, ceiling_of) in [
            (
                "num_layers",
                arch.num_layers,
                tensors.len(),
                "supplied tensors",
            ),
            (
                "hidden",
                arch.hidden,
                supplied_elements,
                "supplied elements",
            ),
            (
                "intermediate",
                arch.intermediate,
                supplied_elements,
                "supplied elements",
            ),
            ("vocab", arch.vocab, supplied_elements, "supplied elements"),
            (
                "positions",
                arch.positions,
                supplied_elements,
                "supplied elements",
            ),
            (
                "type_vocab_size",
                arch.type_vocab_size,
                supplied_elements,
                "supplied elements",
            ),
        ] {
            if value > ceiling {
                return Err(SetFitError::ImportConfigMismatch {
                    field: field.to_string(),
                    expected: format!("at most {ceiling} ({ceiling_of})"),
                    got: value.to_string(),
                });
            }
        }
        let dims = ModelDims {
            hidden: arch.hidden,
            layers: arch.num_layers,
            heads: arch.heads,
            intermediate: arch.intermediate,
            vocab: arch.vocab,
            max_positions: arch.positions,
            type_vocab: arch.type_vocab_size,
            pad_token_id: arch.pad_token_id,
        };
        let remap = match &arch.vocab_remap {
            Some(slice_to_orig) => {
                let remap = VocabRemap::from_slice_to_orig(slice_to_orig.clone())?;
                if remap.slice_vocab() != dims.vocab {
                    return Err(SetFitError::RemapInvalid {
                        reason: format!(
                            "the remap has {} rows but the embedding table has {}",
                            remap.slice_vocab(),
                            dims.vocab
                        ),
                    });
                }
                Some(remap)
            }
            None => None,
        };

        // TAKEN BY VALUE and drained, so each tensor's `Vec<f32>` MOVES into its `Tensor`.
        // Reading through a `&BTreeMap` forced a `data.clone()` per tensor: the caller
        // (`verify::rebuild_from`) owns the map and drops it immediately afterwards, so every
        // one of those copies was pure waste. It is not a small waste — a full model pin is
        // 22,565,376 elements, so the clone duplicated ~90 MB and held ~180 MB of tensor data
        // live at once on the reload path. `Tensor::from_vec` already takes its `Vec` by
        // value, so nothing but this signature stood in the way.
        //
        // The `RefCell` is what lets a `&dyn Fn` mutate: `assemble` deliberately takes the
        // reader as `Fn` so every constructor shares ONE ordering, and widening it to `FnMut`
        // to serve this one caller would be the tail wagging the dog.
        let remaining = core::cell::RefCell::new(tensors);

        let encoder = Self::assemble(
            dims,
            arch.layer_norm_eps as f32,
            remap,
            arch.source_revision.clone(),
            arch.tokenizer_sha256.clone(),
            root_seed,
            &|name: &str, shape: &[usize]| -> Result<Tensor, SetFitError> {
                // `remove`, not `get`. Besides enabling the move, it makes a SECOND read of
                // the same name report "not present" — `assemble` reads each name exactly
                // once, so a repeat read is a defect in the ordering, not a legitimate call.
                let (got_shape, data) = remaining.borrow_mut().remove(name).ok_or_else(|| {
                    SetFitError::ImportTensor(BertLoadError {
                        tensor: name.to_string(),
                        reason: "tensor not present in the bundle".to_string(),
                    })
                })?;
                if got_shape.as_slice() != shape {
                    return Err(SetFitError::ImportTensor(BertLoadError {
                        tensor: name.to_string(),
                        reason: format!("shape mismatch: got {got_shape:?}, expected {shape:?}"),
                    }));
                }
                let expected: usize = shape.iter().product();
                if data.len() != expected {
                    return Err(SetFitError::ImportTensor(BertLoadError {
                        tensor: name.to_string(),
                        reason: format!(
                            "element count mismatch: got {}, expected {expected} (shape {shape:?})",
                            data.len()
                        ),
                    }));
                }
                Ok(Tensor::from_vec(data, shape).requires_grad())
            },
        )?;

        // NOTHING MAY BE LEFT OVER. `assemble` drains exactly the names
        // `Module::named_parameters` emits, so a residue is a tensor this architecture does
        // not name — and every other check is blind to it: the shape and element checks only
        // fire on names that ARE read, the bundle's `deny_unknown_fields` rejects unknown
        // KEYS of the struct and says nothing about entries of the tensor MAP, and the
        // verify policy's round-trip closure check re-serializes the residue happily because
        // it is genuinely part of the bytes that were hashed. Without this, an artifact could
        // carry arbitrary unreferenced payload and still be declared a faithful reload.
        let leftover = remaining.into_inner();
        if let Some((name, _)) = leftover.into_iter().next() {
            return Err(SetFitError::ImportTensor(BertLoadError {
                tensor: name,
                reason: "the bundle carries a tensor this architecture does not name".to_string(),
            }));
        }
        Ok(encoder)
    }

    /// THE construction sequence, shared by every constructor above.
    ///
    /// Takes the `read` as a parameter precisely so that "from an APR reader" and
    /// "from a bundle map" cannot be two different orders, two different shape
    /// expectations or two different seeding schedules.
    fn assemble(
        dims: ModelDims,
        eps: f32,
        remap: Option<VocabRemap>,
        source_revision: String,
        tokenizer_sha256: String,
        root_seed: u64,
        read: &dyn Fn(&str, &[usize]) -> Result<Tensor, SetFitError>,
    ) -> Result<Self, SetFitError> {
        let h = dims.hidden;

        let mut embeddings_layer_norm = LayerNorm::with_eps(&[h], eps);
        embeddings_layer_norm.set_weight(read("embeddings.LayerNorm.weight", &[h])?);
        embeddings_layer_norm.set_bias(read("embeddings.LayerNorm.bias", &[h])?);

        let mut layers = Vec::with_capacity(dims.layers);
        for i in 0..dims.layers {
            let p = format!("encoder.layer.{i}");

            // Site 2 lives INSIDE MultiHeadAttention, between softmax and @V.
            // It was the one unseedable site (A5). 01-06 reached it with a
            // construction-time u64 seed; 03-02 hands it the SAME mask source
            // the other three sites use, so the forward ordinal reaches all four
            // and D-15's branch independence does not hold at three sites and
            // silently fail at the fourth.
            let attention_probs_dropout =
                site_dropout(root_seed, &attention_probs_site(i), DROPOUT_P)?;
            // The `Arc<SiteDropout>` is UNSIZE-COERCED to `Arc<dyn ...>` at this
            // binding. The fully-qualified `Arc::clone(&x)` form does NOT compile
            // here: it unifies its type parameter with the annotated `dyn` type
            // and then fails on the `&Arc<SiteDropout>` argument, before any
            // coercion can apply. The method form resolves `T` from the receiver
            // and coerces the result, which is the whole difference.
            let attention_masks: Arc<dyn AttentionDropoutMasks> = attention_probs_dropout.clone();
            let mut attention = MultiHeadAttention::new(h, dims.heads)
                .with_dropout(DROPOUT_P)
                .with_attention_dropout_masks(attention_masks);
            install_projection(attention.q_proj_mut(), &read, &p, "query", h)?;
            install_projection(attention.k_proj_mut(), &read, &p, "key", h)?;
            install_projection(attention.v_proj_mut(), &read, &p, "value", h)?;
            let out_proj = attention.out_proj_mut();
            out_proj.set_weight(read(
                &format!("{p}.attention.output.dense.weight"),
                &[h, h],
            )?);
            out_proj.set_bias(read(&format!("{p}.attention.output.dense.bias"), &[h])?);

            let mut attention_layer_norm = LayerNorm::with_eps(&[h], eps);
            attention_layer_norm.set_weight(read(
                &format!("{p}.attention.output.LayerNorm.weight"),
                &[h],
            )?);
            attention_layer_norm
                .set_bias(read(&format!("{p}.attention.output.LayerNorm.bias"), &[h])?);

            let im = dims.intermediate;
            let mut intermediate = Linear::new(h, im);
            intermediate.set_weight(read(&format!("{p}.intermediate.dense.weight"), &[im, h])?);
            intermediate.set_bias(read(&format!("{p}.intermediate.dense.bias"), &[im])?);

            let mut output_dense = Linear::new(im, h);
            output_dense.set_weight(read(&format!("{p}.output.dense.weight"), &[h, im])?);
            output_dense.set_bias(read(&format!("{p}.output.dense.bias"), &[h])?);

            let mut output_layer_norm = LayerNorm::with_eps(&[h], eps);
            output_layer_norm.set_weight(read(&format!("{p}.output.LayerNorm.weight"), &[h])?);
            output_layer_norm.set_bias(read(&format!("{p}.output.LayerNorm.bias"), &[h])?);

            layers.push(EncoderLayer {
                attention,
                attention_probs_dropout,
                attention_output_dropout: site_dropout(
                    root_seed,
                    &attention_output_site(i),
                    DROPOUT_P,
                )?,
                attention_layer_norm,
                intermediate,
                output_dense,
                output_dropout: site_dropout(root_seed, &ffn_output_site(i), DROPOUT_P)?,
                output_layer_norm,
            });
        }

        let mut encoder = Self {
            word_embeddings: read("embeddings.word_embeddings.weight", &[dims.vocab, h])?,
            position_embeddings: read(
                "embeddings.position_embeddings.weight",
                &[dims.max_positions, h],
            )?,
            token_type_embeddings: read(
                "embeddings.token_type_embeddings.weight",
                &[dims.type_vocab, h],
            )?,
            embeddings_layer_norm,
            embeddings_dropout: site_dropout(root_seed, EMBEDDINGS_DROPOUT_SITE, DROPOUT_P)?,
            layers,
            dims,
            remap,
            layer_norm_eps: eps,
            source_revision,
            tokenizer_sha256,
            root_seed,
            training: true,
            forward_ordinal: 0,
        };
        // HF `from_pretrained` hands back an eval-mode model; so does this.
        encoder.set_training(false);
        Ok(encoder)
    }

    /// Maximum accepted padded sequence length for THIS encoder.
    ///
    /// `min(MAX_SEQUENCE_LENGTH, max_position_embeddings)`. The minimum, not the
    /// constant: the slice carries only 64 position rows, so a hardcoded `<=
    /// 256` would admit a batch whose position gather runs off the end of the
    /// table.
    #[must_use]
    pub fn max_seq(&self) -> usize {
        MAX_SEQUENCE_LENGTH.min(self.dims.max_positions)
    }

    /// The root seed every dropout site's stream is derived from.
    #[must_use]
    pub fn root_seed(&self) -> u64 {
        self.root_seed
    }

    /// The forward-call ordinal every dropout site currently draws at (D-15).
    #[must_use]
    pub fn forward_ordinal(&self) -> u64 {
        self.forward_ordinal
    }

    /// Point every dropout site at forward-call ordinal `forward_ordinal`.
    ///
    /// This is D-15's `block` coordinate, `2 * training_step + branch`, and it is
    /// what makes the SetFit dropout policy replay-exact AND independent between
    /// the pair objective's two siamese branches. Use
    /// [`super::dropout_rng::forward_ordinal`] to compute it from a
    /// `(step, branch)` pair rather than open-coding the arithmetic.
    ///
    /// Advancing it is the caller's job precisely because the encoder cannot know
    /// it. A self-advancing counter would make the mask a function of how many
    /// forwards had run — including forwards from an unrelated evaluation — and
    /// that is the property TRN-06 needs to NOT have.
    ///
    /// # Errors
    ///
    /// [`SetFitError::DropoutRng`] if `forward_ordinal` does not fit the `u32`
    /// counter lane. Validation happens ONCE before any site is touched, so a
    /// rejected call leaves every site at its previous ordinal rather than
    /// half-advanced.
    pub fn set_forward_ordinal(&mut self, forward_ordinal: u64) -> Result<(), SetFitError> {
        // Validate first, apply second. A validate-as-you-go loop would already
        // have moved the embeddings site by the time it rejected, leaving the
        // encoder in a mixed-ordinal state no caller asked for.
        dropout_rng::checked_forward_ordinal(forward_ordinal)?;
        for site in self.dropout_modules() {
            site.set_forward_ordinal(forward_ordinal)?;
        }
        self.forward_ordinal = forward_ordinal;
        Ok(())
    }

    /// Every dropout site of this encoder, in construction order.
    ///
    /// ONE traversal, used by both the mode channel and the ordinal channel: a
    /// second hand-written loop is how a newly added site ends up following one
    /// and not the other.
    ///
    /// Written as an iterator chain rather than a `for` loop on purpose. This is
    /// a TRAVERSAL, not the compute path, and
    /// `encoder_has_exactly_one_layer_loop` asserts TEXTUALLY that the shared-ref
    /// layer-loop header occurs exactly once in this file, inside
    /// `forward_layers`. Spelling a bookkeeping walk the same way as the one
    /// compute loop would either break that gate or, worse, invite someone to
    /// weaken it to "at least one". (The header is deliberately not quoted in
    /// this comment either — the gate counts occurrences in the SOURCE TEXT, and
    /// a doc comment is source text. That is not pedantry: it is how this very
    /// paragraph first turned the gate red.)
    ///
    /// Returns a lazy ITERATOR, not a `Vec`. Both channels run on the hot path —
    /// `set_forward_ordinal` fires twice per training step and `set_training`
    /// once per batch — and a walk that only reads each site's atomic has no
    /// reason to heap-allocate `1 + 3L` pointers first. The elements are
    /// `&SiteDropout` rather than `&Arc<SiteDropout>` for the same reason: no
    /// caller needs the handle, only the site.
    fn dropout_modules(&self) -> impl Iterator<Item = &SiteDropout> {
        std::iter::once(&self.embeddings_dropout)
            .chain(self.layers.iter().flat_map(|l| {
                [
                    &l.attention_probs_dropout,
                    &l.attention_output_dropout,
                    &l.output_dropout,
                ]
            }))
            .map(Arc::as_ref)
    }

    /// Number of encoder layers this model was built with.
    ///
    /// A READ accessor (the D-08 seal is about constructors). 01-07's
    /// `FreezeGroup` validation needs it: `LayerAttention(7)` against a 2-layer
    /// slice must be a typed rejection, and the only honest source for "how many
    /// layers" is the encoder that was actually built.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Sha256 of the tokenizer this encoder is paired with.
    ///
    /// A READ accessor. It exists so the pairing `SetFitMiniLm` establishes can
    /// be ASSERTED rather than assumed — the forward-time equality check is the
    /// runtime half, this is what lets a test see the value it compares.
    #[must_use]
    pub fn tokenizer_sha256(&self) -> &str {
        &self.tokenizer_sha256
    }

    /// The dimensions this encoder was built at.
    #[must_use]
    pub fn dims(&self) -> &ModelDims {
        &self.dims
    }

    /// The epsilon every `LayerNorm` in this encoder was constructed with.
    #[must_use]
    pub fn layer_norm_eps(&self) -> f32 {
        self.layer_norm_eps
    }

    /// The upstream revision this encoder's weights came from.
    #[must_use]
    pub fn source_revision(&self) -> &str {
        &self.source_revision
    }

    /// The vocabulary remap a slice encoder gathers through; `None` for the pin.
    #[must_use]
    pub fn vocab_remap(&self) -> Option<&VocabRemap> {
        self.remap.as_ref()
    }

    /// Ordered dotted names of every ACTIVE dropout site.
    ///
    /// Real introspection, not a re-derived name list: each entry is emitted
    /// only if the module that implements it exists AND is active, so a site
    /// that was never wired cannot appear. That distinction is the whole point —
    /// a behavioural proxy ("the output changed") cannot tell "site missing"
    /// from "site present but `p` effectively 0", and both are ways ENC-05's
    /// dropout placement can be quietly wrong.
    #[cfg(test)]
    pub(crate) fn dropout_sites(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.embeddings_dropout.probability() > 0.0 {
            out.push(EMBEDDINGS_DROPOUT_SITE.to_string());
        }
        for (i, layer) in self.layers.iter().enumerate() {
            if layer.attention.dropout_p() > 0.0
                && layer.attention.has_attention_dropout_masks()
                && layer.attention_probs_dropout.probability() > 0.0
            {
                out.push(attention_probs_site(i));
            }
            if layer.attention_output_dropout.probability() > 0.0 {
                out.push(attention_output_site(i));
            }
            if layer.output_dropout.probability() > 0.0 {
                out.push(ffn_output_site(i));
            }
        }
        out
    }

    /// Graph-connected token states `[B, S, H]`.
    ///
    /// A thin wrapper over `forward_layers` — it runs no layer of its own.
    ///
    /// # Errors
    ///
    /// A typed [`SetFitError`] for a foreign tokenizer, a malformed batch, an
    /// oversize sequence, or an id outside the vocabulary / slice closure.
    /// Op failures arrive as [`SetFitError::Op`].
    #[provable_contracts_macros::contract(
        "setfit-encoder-conformance-v1",
        equation = "setfit_encoder_forward"
    )]
    pub fn forward_tokens(&self, batch: &SentenceBatch) -> Result<Tensor, SetFitError> {
        contract_pre_setfit_encoder_forward!(batch.input_ids());
        let (_, mut layer_outputs) = self.forward_layers(batch)?;
        // A zero-layer configuration is rejected at import, so an empty vec is
        // an internal invariant violation rather than a user-reachable state.
        let result = layer_outputs.pop().ok_or(SetFitError::BatchInvalid {
            reason: "encoder has no layers".to_string(),
        })?;
        contract_post_setfit_encoder_forward!(result.data());
        Ok(result)
    }

    /// Per-layer intermediates for the D-15 localization gate (01-08).
    ///
    /// Returns `(embeddings_out [B,S,H], layer_outputs: one [B,S,H] per encoder
    /// layer, in order)`. `layer_outputs.last()` IS the tensor
    /// [`Self::forward_tokens`] returns — the fixture's `final_tokens`.
    ///
    /// `pub`, not `pub(crate)`: 01-08 is an out-of-crate integration test and
    /// reaches this through 01-07's conformance-gated `SetFitMiniLm::encoder()`.
    /// It is a READ method — it constructs no encoder and no tokenizer, so the
    /// D-08 seal is untouched.
    ///
    /// # Errors
    ///
    /// Identical to [`Self::forward_tokens`]: both inherit the one boundary
    /// validation inside `forward_layers`.
    #[cfg(feature = "conformance-fixtures")]
    pub fn forward_tokens_per_layer(
        &self,
        batch: &SentenceBatch,
    ) -> Result<(Tensor, Vec<Tensor>), SetFitError> {
        self.forward_layers(batch)
    }

    /// `forward_tokens` -> masked mean pool -> L2 normalize: `[B, H]`,
    /// graph-connected, unit-norm rows.
    ///
    /// # Errors
    ///
    /// As [`Self::forward_tokens`], plus [`SetFitError::Op`] from the pooling
    /// and normalization primitives.
    pub fn encode(&self, batch: &SentenceBatch) -> Result<Tensor, SetFitError> {
        let tokens = self.forward_tokens(batch)?;
        let pooled = masked_mean_pool(&tokens, batch.attention_mask())?;
        Ok(l2_normalize_rows(&pooled, L2_EPS)?)
    }

    /// The identity of the path [`Self::encode`] above runs, and the ONLY place
    /// this repository spells it.
    ///
    /// Deliberately adjacent to `encode` — the code it names is the three lines
    /// above — so the name and the implementation move together. Module-private,
    /// and it stays that way: [`Self::encode_with_backend`] is the only door,
    /// which is what makes "the identity is RETURNED by the encode call" true
    /// rather than merely conventional. No other module may name this constant;
    /// a reader who wants to change the reported value must change the encode
    /// path, which is the point.
    ///
    /// See [`ExecutionBackend`] for why this is a kernel entry point and not a
    /// CPU-feature detection result (review B6, D-12).
    const ENCODE_BACKEND: ExecutionBackend = ExecutionBackend {
        device: "cpu",
        kernel: "autograd-trueno-matmul",
    };

    /// [`Self::encode`], plus the identity of the path that ran it (D-12).
    ///
    /// Added BESIDE [`Self::encode`], which is unchanged: the training path and
    /// every Phase 1 conformance fixture call it, and their bytes must stay
    /// byte-identical.
    ///
    /// # Errors
    ///
    /// Exactly [`Self::encode`]'s.
    pub fn encode_with_backend(
        &self,
        batch: &SentenceBatch,
    ) -> Result<(Tensor, ExecutionBackend), SetFitError> {
        let pooled = self.encode(batch)?;
        Ok((pooled, Self::ENCODE_BACKEND))
    }

    // -----------------------------------------------------------------------
    // The ONE forward implementation
    // -----------------------------------------------------------------------

    /// Validate the boundary once, gather embeddings, run every layer.
    ///
    /// Both public entry points delegate here, so neither can drift from the
    /// other and neither can skip the validation (T-1-11, T-1-28).
    fn forward_layers(&self, batch: &SentenceBatch) -> Result<(Tensor, Vec<Tensor>), SetFitError> {
        let ids = self.validate(batch)?;
        let b = batch.batch;
        let s = batch.seq;

        // ---- Embeddings --------------------------------------------------
        let word = embedding_gather(&self.word_embeddings, &ids, b, s)?;
        // Position ids are 0..S for every row. Integers, no gradient.
        let position_ids: Vec<u32> = (0..b)
            .flat_map(|_| (0..s).map(|p| u32::try_from(p).unwrap_or(u32::MAX)))
            .collect();
        let position = embedding_gather(&self.position_embeddings, &position_ids, b, s)?;
        let token_type =
            embedding_gather(&self.token_type_embeddings, &batch.token_type_ids, b, s)?;

        let summed = word.add(&position).add(&token_type);
        let normalized = self.embeddings_layer_norm.forward(&summed);
        // The tensor AFTER site 1 is `embeddings_out` — the same quantity
        // forward_per_layer.json records. Dropout is inert in eval, the mode the
        // fixtures were generated in (D-16).
        let embeddings_out = self.embeddings_dropout.forward(&normalized);

        // ---- Mask ---------------------------------------------------------
        // [B,1,1,S] so it broadcasts over [B,heads,S,S] scores through 01-09's
        // repaired `add_mask`, which keeps the autograd edge.
        let attention_mask = additive_attention_mask(&batch.attention_mask, b, s)?;

        // ---- Layers -------------------------------------------------------
        let mut x = embeddings_out.clone();
        let mut layer_outputs = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let (attended, _) = layer.attention.forward_self(&x, Some(&attention_mask));
            let attended = layer.attention_output_dropout.forward(&attended);
            x = layer.attention_layer_norm.forward(&x.add(&attended));

            // gelu_exact (01-09), matching the pin's `hidden_act: "gelu"`. The
            // tanh form is a different function, 4.73e-4 away, and ENC-01
            // rejects any checkpoint that asks for it.
            let intermediate = layer.intermediate.forward(&x).gelu_exact();
            let ffn = layer.output_dense.forward(&intermediate);
            let ffn = layer.output_dropout.forward(&ffn);
            x = layer.output_layer_norm.forward(&x.add(&ffn));

            layer_outputs.push(x.clone());
        }

        Ok((embeddings_out, layer_outputs))
    }

    /// Fail-closed boundary validation, run exactly once per forward.
    ///
    /// Returns the EMBEDDING-TABLE row for each position: canonical ids resolved
    /// through the remap when the import carries one, canonical ids verbatim
    /// otherwise. The [`SentenceBatch`] itself is never mutated.
    fn validate(&self, batch: &SentenceBatch) -> Result<Vec<u32>, SetFitError> {
        // 1. Tokenizer identity, BEFORE any compute (D-08 defense in depth).
        if batch.tokenizer_sha256 != self.tokenizer_sha256 {
            return Err(SetFitError::TokenizerHashMismatch {
                expected: self.tokenizer_sha256.clone(),
                got: batch.tokenizer_sha256.clone(),
            });
        }

        // 2. Shape.
        let b = batch.batch;
        let s = batch.seq;
        if b == 0 || s == 0 {
            return Err(SetFitError::BatchInvalid {
                reason: format!("batch {b} x seq {s}: neither dimension may be zero"),
            });
        }
        let positions = b.checked_mul(s).ok_or_else(|| SetFitError::BatchInvalid {
            reason: format!("batch {b} x seq {s} overflows usize"),
        })?;
        for (field, len) in [
            ("input_ids", batch.input_ids.len()),
            ("token_type_ids", batch.token_type_ids.len()),
            ("attention_mask", batch.attention_mask.len()),
        ] {
            if len != positions {
                return Err(SetFitError::BatchInvalid {
                    reason: format!(
                        "{field} has {len} entries but batch {b} x seq {s} needs {positions}"
                    ),
                });
            }
        }

        // 3. Sequence bound — min(256, max_position_embeddings), so an
        //    over-long batch cannot drive an out-of-range position gather.
        let max = self.max_seq();
        if s > max {
            return Err(SetFitError::OversizeInput { len: s, max });
        }

        // 4. Mask values, then per-row validity. Order matters: a row of all
        //    `2`s is a broken mask, not a padded row.
        for (position, v) in batch.attention_mask.iter().enumerate() {
            if *v > 1 {
                return Err(OpError::NonBinaryMaskValue {
                    value: *v,
                    position,
                }
                .into());
            }
        }
        for row in 0..b {
            let base = row * s;
            if !batch.attention_mask[base..base + s].iter().any(|v| *v == 1) {
                return Err(OpError::AllPaddingRow { row }.into());
            }
        }

        // 5. Token type ids inside the type table.
        for (position, t) in batch.token_type_ids.iter().enumerate() {
            if *t as usize >= self.dims.type_vocab {
                return Err(OpError::OutOfVocabulary {
                    id: *t,
                    vocab_size: self.dims.type_vocab,
                    position,
                }
                .into());
            }
        }

        // 6. Word ids. Canonical ids are remapped to slice rows HERE; the batch
        //    keeps its canonical ids so provenance survives the forward.
        let mut rows = Vec::with_capacity(positions);
        match &self.remap {
            Some(remap) => {
                for id in &batch.input_ids {
                    rows.push(remap.to_slice_row(*id)?);
                }
            }
            None => {
                for (position, id) in batch.input_ids.iter().enumerate() {
                    if *id as usize >= self.dims.vocab {
                        return Err(OpError::OutOfVocabulary {
                            id: *id,
                            vocab_size: self.dims.vocab,
                            position,
                        }
                        .into());
                    }
                    rows.push(*id);
                }
            }
        }
        Ok(rows)
    }

    // -----------------------------------------------------------------------
    // Naming (D-18)
    // -----------------------------------------------------------------------

    /// HF dotted prefix of layer `i`.
    fn layer_prefix(i: usize) -> String {
        format!("encoder.layer.{i}")
    }
}

/// Install one `attention.self.{query,key,value}` projection.
///
/// A free function rather than an inline loop: `q_proj_mut()`, `k_proj_mut()`
/// and `v_proj_mut()` each borrow the whole `MultiHeadAttention` mutably, so
/// they cannot be collected into one iterable.
fn install_projection<F>(
    proj: &mut Linear,
    read: &F,
    layer_prefix: &str,
    hf_name: &str,
    hidden: usize,
) -> Result<(), SetFitError>
where
    F: Fn(&str, &[usize]) -> Result<Tensor, SetFitError>,
{
    proj.set_weight(read(
        &format!("{layer_prefix}.attention.self.{hf_name}.weight"),
        &[hidden, hidden],
    )?);
    proj.set_bias(read(
        &format!("{layer_prefix}.attention.self.{hf_name}.bias"),
        &[hidden],
    )?);
    Ok(())
}

/// Translate a `MultiHeadAttention` local parameter name to its HF path.
///
/// The mapping is by NAME, not by position, so a reordering inside
/// `MultiHeadAttention` surfaces as an out-of-order parameter list against the
/// frozen `parameter_order` rather than as silently mislabelled tensors. An
/// unrecognised local name is passed through verbatim, which fails that same
/// gate loudly instead of inventing a plausible HF path.
fn hf_attention_name(local: &str) -> String {
    match local.split_once('.') {
        Some(("q_proj", leaf)) => format!("attention.self.query.{leaf}"),
        Some(("k_proj", leaf)) => format!("attention.self.key.{leaf}"),
        Some(("v_proj", leaf)) => format!("attention.self.value.{leaf}"),
        Some(("out_proj", leaf)) => format!("attention.output.dense.{leaf}"),
        _ => format!("attention.{local}"),
    }
}

impl Module for BertSentenceEncoder {
    /// Not the ENC-03 entry point.
    ///
    /// The `Module` trait's `forward` takes a bare tensor, which carries no
    /// attention mask and no tokenizer identity — the two things this encoder
    /// validates. Running the layer stack from here would silently attend to
    /// padding. Use [`BertSentenceEncoder::encode`] or
    /// [`BertSentenceEncoder::forward_tokens`]; this impl exists so the encoder
    /// participates in parameter traversal and mode propagation.
    fn forward(&self, input: &Tensor) -> Tensor {
        input.clone()
    }

    fn parameters(&self) -> Vec<&Tensor> {
        self.named_parameters()
            .into_iter()
            .map(|(_, t)| t)
            .collect()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        self.named_parameters_mut()
            .into_iter()
            .map(|(_, t)| t)
            .collect()
    }

    /// HF dotted names, verbatim, in torch `named_parameters()` order, pooler
    /// excluded.
    ///
    /// Verbatim is the whole point (D-18): `gradients.json`'s keys are torch's
    /// own names, so any translation layer between them and these is a place the
    /// two can disagree. The order is asserted against `parameter_order`, an
    /// ordered ARRAY — JSON object keys carry no ordering guarantee.
    fn named_parameters(&self) -> Vec<(String, &Tensor)> {
        let mut out: Vec<(String, &Tensor)> = vec![
            (
                "embeddings.word_embeddings.weight".to_string(),
                &self.word_embeddings,
            ),
            (
                "embeddings.position_embeddings.weight".to_string(),
                &self.position_embeddings,
            ),
            (
                "embeddings.token_type_embeddings.weight".to_string(),
                &self.token_type_embeddings,
            ),
        ];
        out.extend(
            self.embeddings_layer_norm
                .named_parameters()
                .into_iter()
                .map(|(n, t)| (format!("embeddings.LayerNorm.{n}"), t)),
        );

        for (i, layer) in self.layers.iter().enumerate() {
            let p = Self::layer_prefix(i);
            out.extend(
                layer
                    .attention
                    .named_parameters()
                    .into_iter()
                    .map(|(n, t)| (format!("{p}.{}", hf_attention_name(&n)), t)),
            );
            out.extend(
                layer
                    .attention_layer_norm
                    .named_parameters()
                    .into_iter()
                    .map(|(n, t)| (format!("{p}.attention.output.LayerNorm.{n}"), t)),
            );
            out.extend(
                layer
                    .intermediate
                    .named_parameters()
                    .into_iter()
                    .map(|(n, t)| (format!("{p}.intermediate.dense.{n}"), t)),
            );
            out.extend(
                layer
                    .output_dense
                    .named_parameters()
                    .into_iter()
                    .map(|(n, t)| (format!("{p}.output.dense.{n}"), t)),
            );
            out.extend(
                layer
                    .output_layer_norm
                    .named_parameters()
                    .into_iter()
                    .map(|(n, t)| (format!("{p}.output.LayerNorm.{n}"), t)),
            );
        }
        out
    }

    fn named_parameters_mut(&mut self) -> Vec<(String, &mut Tensor)> {
        let mut out: Vec<(String, &mut Tensor)> = vec![
            (
                "embeddings.word_embeddings.weight".to_string(),
                &mut self.word_embeddings,
            ),
            (
                "embeddings.position_embeddings.weight".to_string(),
                &mut self.position_embeddings,
            ),
            (
                "embeddings.token_type_embeddings.weight".to_string(),
                &mut self.token_type_embeddings,
            ),
        ];
        out.extend(
            self.embeddings_layer_norm
                .named_parameters_mut()
                .into_iter()
                .map(|(n, t)| (format!("embeddings.LayerNorm.{n}"), t)),
        );

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let p = Self::layer_prefix(i);
            out.extend(
                layer
                    .attention
                    .named_parameters_mut()
                    .into_iter()
                    .map(|(n, t)| (format!("{p}.{}", hf_attention_name(&n)), t)),
            );
            out.extend(
                layer
                    .attention_layer_norm
                    .named_parameters_mut()
                    .into_iter()
                    .map(|(n, t)| (format!("{p}.attention.output.LayerNorm.{n}"), t)),
            );
            out.extend(
                layer
                    .intermediate
                    .named_parameters_mut()
                    .into_iter()
                    .map(|(n, t)| (format!("{p}.intermediate.dense.{n}"), t)),
            );
            out.extend(
                layer
                    .output_dense
                    .named_parameters_mut()
                    .into_iter()
                    .map(|(n, t)| (format!("{p}.output.dense.{n}"), t)),
            );
            out.extend(
                layer
                    .output_layer_norm
                    .named_parameters_mut()
                    .into_iter()
                    .map(|(n, t)| (format!("{p}.output.LayerNorm.{n}"), t)),
            );
        }
        out
    }

    /// ENC-05: flip every dropout site, recursively, and nothing else.
    ///
    /// RNG state, seeds and the mode flag are module state, never parameters, so
    /// this changes no registered tensor — proven bytewise by the
    /// train -> eval -> train snapshot test.
    fn set_training(&mut self, training: bool) {
        self.training = training;
        // The four dotted dropout sites, through the ONE traversal the ordinal
        // channel also uses. `SiteDropout::set_training` takes `&self` because
        // the flag is an atomic coordinate, not accumulated state.
        for site in self.dropout_modules() {
            site.set_training(training);
        }
        self.embeddings_layer_norm.set_training(training);
        for layer in &mut self.layers {
            // MultiHeadAttention gates site 2 on ITS OWN flag — the mask source's
            // flag is irrelevant there — so the attention module still has to be
            // flipped explicitly.
            layer.attention.set_training(training);
            layer.attention_layer_norm.set_training(training);
            layer.intermediate.set_training(training);
            layer.output_dense.set_training(training);
            layer.output_layer_norm.set_training(training);
        }
    }

    /// Delegates to [`Module::set_training`].
    ///
    /// The crate convention is that `train`/`eval` are leaf-local and
    /// `set_training` is the propagation channel (D-17). That is a real footgun
    /// on a module whose whole point is dropout placement: `encoder.eval()`
    /// leaving dropout active would silently make every "inference" run
    /// stochastic. Both spellings therefore route through the one channel.
    fn train(&mut self) {
        self.set_training(true);
    }

    /// Delegates to [`Module::set_training`]; see [`Module::train`].
    fn eval(&mut self) {
        self.set_training(false);
    }

    fn training(&self) -> bool {
        self.training
    }
}

#[cfg(all(test, feature = "setfit"))]
#[path = "encoder_tests.rs"]
mod encoder_tests;
