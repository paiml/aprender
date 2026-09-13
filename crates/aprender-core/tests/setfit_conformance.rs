#![cfg(all(feature = "setfit", feature = "conformance-fixtures"))]
//! # SetFit / MiniLM conformance harness (Phase 1, plan 01-08)
//!
//! Contract: `setfit-encoder-conformance-v1`. This is the falsifiable Phase 1
//! gate: everything 01-01..01-07 and 01-09 built, compared against the frozen
//! Python reference plane 01-04 recorded.
//!
//! Run the whole suite:
//!
//! ```text
//! cargo test -p aprender-core --features setfit,conformance-fixtures --test setfit_conformance
//! ```
//!
//! `cargo test -p aprender-core --lib` compiles **none** of this — the setfit
//! module and every fixture gate sit behind the two features above. A green
//! default-feature run says nothing about this file.
//!
//! ## The three structural rules of this harness
//!
//! **1. Every model input goes through the real tokenizer.** [`batch_from_case`]
//! is the ONLY place `SentenceBatch` is produced, and it produces it by calling
//! `SetFitMiniLm::tokenize(case.texts)` — the only public batch producer after
//! the D-08 seal. No submodule hand-builds a batch, and the fixtures'
//! pre-remapped slice-id arrays are deliberately not even deserialized, so the
//! vocabulary remap is exercised INSIDE the encoder exactly where production
//! exercises it. [`conformance_the_tokenizer_boundary_is_the_only_batch_source`]
//! holds that line by scanning this file and its submodules.
//!
//! **2. Every epsilon comes from the contract.** There is no hand-written
//! tolerance anywhere except [`tolerances_generated`], which is derived and
//! carries a DO-NOT-HAND-EDIT header.
//! [`conformance_no_hand_written_tolerance_literal_outside_the_generated_file`]
//! scans for the shape.
//!
//! **3. Nothing is constructed through a sealed path.** This is an out-of-crate
//! integration test. `MiniLmImport::{open, open_slice_fixture}`,
//! `MiniLmTokenizer::from_bytes` and `BertSentenceEncoder::from_import` are all
//! `pub(crate)` (01-05/01-06, proven by four `E0624`s in 01-07 — **not** E0603,
//! see D41). Models come from `SetFitMiniLm::{from_slice_fixture,
//! from_pretrained_dir}` and nothing else.
//!
//! ## What these gates CAN and CANNOT detect
//!
//! Stated plainly, because 01-06's mutation F is the standard: an FFN output
//! dropout that was constructed, seeded, mode-aware and reported — but never
//! called — survived 41 of 42 tests there.
//!
//! **Detected here:** wrong per-layer arithmetic (localized to the layer),
//! a wrong activation form, a wrong pooling denominator, a wrong normalization
//! epsilon branch, a severed autograd edge anywhere the pair objective's
//! backward should reach, a frozen parameter that moves, a tokenizer that
//! disagrees with the pinned HF one, a tolerance loosened without a contract
//! edit, and — since D55 closed — a wrong AdamW learning rate, an update that
//! is not sign-saturated at step 1, and wrong beta1/beta2.
//!
//! **The optimizer's detection scope, precisely.** Each row was established by
//! running the mutation against this suite, not by reading the assertions:
//!
//! | Mutation | This suite | `--lib adamw` |
//! |---|---|---|
//! | betas hardcoded to `(0.5, 0.5)` | **RED** — trajectory off by 4.05e-04 vs 7.63e-06 tol | green |
//! | update scale halved (`0.5 * lr`) | **RED** — post-step off by 1.00e-05 vs 1.89e-06 tol | — |
//! | decoupled weight decay deleted | green (see below) | **RED** — `falsify_aw_001` |
//!
//! The betas row is what `OBLIG-ENC-04-MULTISTEP-TRAJECTORY-PARITY` exists for:
//! at step 1 bias correction gives `m_hat = (1-b1)g/(1-b1) = g` and
//! `v_hat = (1-b2)g^2/(1-b2) = g^2`, so the update is `lr*g/(|g|+eps)` for every
//! choice of betas and NO single-step fixture can constrain them at any
//! tolerance. Replaying 20 steps can, because the moments carry history from
//! step 2 onward.
//!
//! The halved-update row is what the tolerance fix bought: the same mutation
//! deviates 1.00e-05, which the pre-D55 tolerance of 3.05e-05 accepted.
//!
//! **Still NOT detected here — decoupled weight decay.** This is structural and
//! not a tolerance to tighten: at `lr = 2e-5, wd = 0.01` the decay term is
//! `lr*wd*|p| <= 1.75e-07`, about 3 f32 ulp of the parameters it acts on, and
//! measurement puts it within 2.6x of the f32/f64 noise floor at every step
//! count tried up to 50. No f32 reference comparison separates it. It is owned
//! by `contracts/adamw-kernel-v1` and its `falsify_aw_001_decoupled...` lib
//! test, which compares AdamW against Adam algebraically rather than against a
//! reference, and which was confirmed RED under that mutation.
//!
//! **NOT detected here:** anything that is inert in EVAL mode. Every numerical
//! fixture in this corpus was generated with dropout disabled (D-16), so a
//! dropout site that is constructed and never applied is invisible to every
//! parity gate in this file. That class is covered by
//! `setfit::encoder::encoder_tests::encoder_mode_every_site_is_actually_applied_in_the_forward`
//! (01-06), which lives in the library's unit tests, not here. Nor does this
//! file exercise the full architecture: the slice is 2 layers / hidden 64 /
//! 2 heads / 64 positions, so a hidden-384 or 6-layer defect is only reachable
//! through the `#[ignore]`d full-weight suite (D-10).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use aprender::autograd::{get_grad, Tensor};
use aprender::nn::Module;
use aprender::setfit::{FreezeGroup, SentenceBatch, SetFitError, SetFitMiniLm};

use serde::Deserialize;
use sha2::{Digest, Sha256};

// `#[path]` is required: submodules of an integration-test crate root resolve
// against `tests/`, not against `tests/setfit_conformance/`.
#[path = "setfit_conformance/tolerances_generated.rs"]
pub mod tolerances_generated;

#[path = "setfit_conformance/detach_negative.rs"]
mod detach_negative;
#[path = "setfit_conformance/forward_parity.rs"]
mod forward_parity;
#[path = "setfit_conformance/frozen_gate.rs"]
mod frozen_gate;
#[path = "setfit_conformance/full_weight_parity.rs"]
mod full_weight_parity;
#[path = "setfit_conformance/gradient_gate.rs"]
mod gradient_gate;

use tolerances_generated as tol;

// ===========================================================================
// Fixture location and construction
// ===========================================================================

/// The frozen fixture corpus 01-04 committed.
///
/// `APRENDER_SETFIT_FIXTURES` overrides it, matching the library-side helper in
/// `src/setfit/model_tests.rs` so a relocated corpus moves both at once.
pub fn fixtures_dir() -> PathBuf {
    if let Ok(p) = std::env::var("APRENDER_SETFIT_FIXTURES") {
        let p = PathBuf::from(p);
        if p.is_dir() {
            return p;
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/setfit")
}

/// The path of the phase contract, or `None` in a packaged-crate context.
///
/// Walks up from `CARGO_MANIFEST_DIR`. In every developer and CI checkout this
/// resolves; inside an unpacked `.crate` it does not, because the
/// workspace-root `contracts/` directory is not part of the published package.
pub fn contract_path() -> Option<PathBuf> {
    let mut dir: Option<&Path> = Some(Path::new(env!("CARGO_MANIFEST_DIR")));
    while let Some(d) = dir {
        let candidate = d.join("contracts/setfit-encoder-conformance-v1.yaml");
        if candidate.is_file() {
            return Some(candidate);
        }
        dir = d.parent();
    }
    None
}

/// Seed every model in this harness is built with.
///
/// Only the dropout streams depend on it, and every fixture comparison runs in
/// eval mode where those are inert (D-16) — so no parity result here is a
/// function of this number.
pub const SEED: u64 = 0x0108_5E7F_1701;

/// A fresh eval-mode model over the frozen slice.
///
/// `from_slice_fixture` is one of exactly two public constructors (01-07); the
/// lower-level ones are sealed and would not compile from here.
pub fn slice_model() -> SetFitMiniLm {
    let mut m = SetFitMiniLm::from_slice_fixture(&fixtures_dir(), SEED)
        .expect("the frozen slice fixture must load through the bound type");
    // The fixtures were generated in eval mode (D-16, 01-06 departure 5). Models
    // already load in eval mode; this is belt to those braces and documents the
    // requirement at every call site.
    m.set_training(false);
    m
}

/// **The one batch builder.**
///
/// Every model input in this harness comes from here, and this is the only
/// place `tokenize` is called. Driving from `texts` rather than from recorded
/// ids is what makes the slice vocabulary remap run inside the encoder, which
/// is where production runs it — a hand-built batch carrying pre-remapped rows
/// would satisfy every numeric gate in this file while bypassing the boundary
/// those gates exist to cover (T-1-27).
///
/// # Errors
///
/// Whatever `SetFitMiniLm::tokenize` returns for an empty list or a tokenizer
/// failure.
pub fn batch_from_case(
    model: &SetFitMiniLm,
    texts: &[String],
) -> Result<SentenceBatch, SetFitError> {
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    model.tokenize(&refs)
}

// ===========================================================================
// Fixture schemas (01-04)
// ===========================================================================

pub fn read_fixture<T: for<'de> Deserialize<'de>>(name: &str) -> T {
    let path = fixtures_dir().join(name);
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("fixture `{}` is unreadable: {e}", path.display()));
    serde_json::from_slice(&bytes).unwrap_or_else(|e| {
        panic!(
            "fixture `{}` does not match its schema: {e}",
            path.display()
        )
    })
}

#[derive(Debug, Deserialize)]
pub struct Shape3 {
    pub batch: usize,
    pub seq: usize,
    pub hidden: usize,
}

#[derive(Debug, Deserialize)]
pub struct Shape2 {
    pub batch: usize,
    pub hidden: usize,
}

/// NOTE: the fixture's pre-remapped slice-id array is intentionally absent from
/// this struct. Not deserializing it is strictly stronger than promising not to
/// feed it — the harness cannot pass ids the encoder should have remapped
/// itself, because it never reads them.
#[derive(Debug, Deserialize)]
pub struct ForwardCase {
    pub case_id: String,
    pub texts: Vec<String>,
    pub input_ids_canonical: Vec<Vec<u32>>,
    pub attention_mask: Vec<Vec<u8>>,
    pub shape: Shape3,
    pub embeddings_out: Vec<f32>,
    pub layer_outputs: Vec<Vec<f32>>,
    pub final_tokens: Vec<f32>,
}

#[derive(Debug, Deserialize)]
pub struct ForwardFixture {
    pub cases: Vec<ForwardCase>,
}

#[derive(Debug, Deserialize)]
pub struct PoolingCase {
    pub case_id: String,
    pub texts: Vec<String>,
    pub input_ids_canonical: Vec<Vec<u32>>,
    pub attention_mask: Vec<Vec<u8>>,
    pub shape: Shape2,
    pub pooled: Vec<f32>,
    pub normalized: Vec<f32>,
}

#[derive(Debug, Deserialize)]
pub struct PoolingFixture {
    pub cases: Vec<PoolingCase>,
}

#[derive(Debug, Deserialize)]
pub struct ActivationFixture {
    pub op: String,
    pub approximate: String,
    pub tanh_vs_exact_max_delta: f32,
    pub x: Vec<f32>,
    pub y: Vec<f32>,
}

/// The pair batch every ENC-04 / ENC-06 gate runs on.
///
/// The pre-remapped id arrays are again deliberately not deserialized.
#[derive(Debug, Deserialize)]
pub struct LossPair {
    pub a_case_id: String,
    pub a_texts: Vec<String>,
    pub a_ids_canonical: Vec<Vec<u32>>,
    pub b_case_id: String,
    pub b_texts: Vec<String>,
    pub b_ids_canonical: Vec<Vec<u32>>,
    pub labels: Vec<f32>,
}

#[derive(Debug, Deserialize)]
pub struct LossFixture {
    pub pair: LossPair,
    pub cosine: Vec<f32>,
    pub mse: f32,
}

/// Which fixture and which cases a derived fixture was recorded from.
///
/// Asserted against what the gate actually tokenized, so a future fixture
/// regeneration that moves the pair batch fails loudly instead of silently
/// comparing against the wrong reference.
#[derive(Debug, Deserialize)]
pub struct SourcePointer {
    pub fixture: String,
    pub a_case_id: String,
    pub b_case_id: String,
}

#[derive(Debug, Deserialize)]
pub struct NamedGrad {
    pub shape: Vec<usize>,
    pub grad: Vec<f32>,
}

#[derive(Debug, Deserialize)]
pub struct AnalyticallyZero {
    pub name: String,
    pub max_abs_grad: f32,
    pub justification: String,
}

#[derive(Debug, Deserialize)]
pub struct GradientsFixture {
    pub source: SourcePointer,
    pub parameter_order: Vec<String>,
    pub zero_grad_floor: f32,
    pub analytically_zero: Vec<AnalyticallyZero>,
    pub grads: BTreeMap<String, NamedGrad>,
}

impl GradientsFixture {
    /// The exemption set, as DATA. Rust never decides its own exemptions.
    pub fn exempt_names(&self) -> Vec<String> {
        self.analytically_zero
            .iter()
            .map(|e| e.name.clone())
            .collect()
    }
}

#[derive(Debug, Deserialize)]
pub struct AdamWSpec {
    pub lr: f32,
    pub betas: Vec<f32>,
    pub eps: f32,
    pub weight_decay: f32,
}

#[derive(Debug, Deserialize)]
pub struct OptimizerStepFixture {
    pub source: SourcePointer,
    pub adamw: AdamWSpec,
    pub all_trainable: bool,
    pub loss_before: f32,
    pub loss_after: f32,
    pub post_step: BTreeMap<String, Vec<f32>>,
}

/// `optimizer_multistep.json` — the trajectory that constrains the betas.
///
/// `separation` is not read by any assertion: it is the generator's measurement
/// of what each mutation actually moves, carried into the fixture so a reader
/// can see WHY this obligation is falsifiable (and why the decay control is
/// recorded but not gated here) without re-deriving it.
#[derive(Debug, Deserialize)]
pub struct OptimizerMultistepFixture {
    pub source: SourcePointer,
    pub adamw: AdamWSpec,
    pub all_trainable: bool,
    pub steps: usize,
    pub losses: Vec<f32>,
}

#[derive(Debug, Deserialize)]
pub struct InvarianceSingle {
    pub case_id: String,
    pub texts: Vec<String>,
    pub input_ids_canonical: Vec<Vec<u32>>,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
pub struct InvariancePadded {
    pub case_id: String,
    pub texts: Vec<String>,
    pub input_ids_canonical: Vec<Vec<u32>>,
    pub embeddings: Vec<f32>,
    pub target_row: usize,
}

#[derive(Debug, Deserialize)]
pub struct InvarianceFixture {
    pub single: InvarianceSingle,
    pub padded_batch: InvariancePadded,
}

#[derive(Debug, Deserialize)]
pub struct FullModelFixture {
    pub case_id: String,
    pub texts: Vec<String>,
    pub shape: Shape2,
    pub embeddings: Vec<f32>,
}

#[derive(Debug, Deserialize)]
pub struct TokenizerCase {
    pub id: String,
    pub texts: Vec<String>,
    pub input_ids: Vec<Vec<u32>>,
    pub attention_mask: Vec<Vec<u8>>,
}

#[derive(Debug, Deserialize)]
pub struct TokenizerCases {
    pub cases: Vec<TokenizerCase>,
}

impl TokenizerCases {
    pub fn get(&self, id: &str) -> &TokenizerCase {
        self.cases
            .iter()
            .find(|c| c.id == id)
            .unwrap_or_else(|| panic!("case_id `{id}` does not resolve in tokenizer_cases.json"))
    }
}

// ===========================================================================
// Comparison helpers
// ===========================================================================

/// Elementwise comparison, reporting the WORST index and both values.
///
/// `tol` always arrives from [`tolerances_generated`]; this function has no
/// default and no fallback, so a caller cannot accidentally compare without a
/// contract-derived epsilon.
pub fn assert_close(actual: &[f32], expected: &[f32], tol: f32, what: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{what}: length {} != fixture length {}",
        actual.len(),
        expected.len()
    );
    let mut worst = 0.0f32;
    let mut worst_at = usize::MAX;
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(a.is_finite(), "{what}: element {i} is {a}, not finite");
        let d = (a - e).abs();
        if d > worst {
            worst = d;
            worst_at = i;
        }
    }
    assert!(
        worst <= tol,
        "{what}: max |rust - fixture| = {worst:e} at index {worst_at} \
         (rust {}, fixture {}), tolerance {tol:e}",
        actual[worst_at],
        expected[worst_at]
    );
}

/// True when every element agrees within `tol`. Used to prove a gate can FAIL.
pub fn all_within(actual: &[f32], expected: &[f32], tol: f32) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected.iter())
            .all(|(a, e)| (a - e).abs() <= tol)
}

pub fn l2(v: &[f32]) -> f32 {
    v.iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt() as f32
}

pub fn max_abs(v: &[f32]) -> f32 {
    v.iter().fold(0.0f32, |m, x| m.max(x.abs()))
}

// ===========================================================================
// The ENC-04 gate — ONE implementation, shared by the positive and the negative
// ===========================================================================

/// One named tensor's gradient, or `None` when it received none at all.
pub type NamedGrads = Vec<(String, Option<Vec<f32>>)>;

/// Inputs to the canonical ENC-04 gate.
pub struct GateInput<'a> {
    /// Every named tensor in the partition under test, in traversal order.
    pub grads: &'a NamedGrads,
    /// Per-tensor post-step parameter movement, when a step was taken.
    pub deltas: Option<&'a [(String, f32)]>,
    /// The lr of the step that produced `deltas`, when that step was the FIRST
    /// one taken by a fresh optimizer. Clause (f) then asserts each non-exempt
    /// tensor moved by lr rather than merely by something positive. `None` from
    /// callers that pass no deltas, and from any future caller stepping at t > 1,
    /// where the update no longer saturates to `lr*sign(g)`.
    pub step_lr: Option<f32>,
    /// `gradients.json.analytically_zero[].name` — DATA, never a Rust decision.
    pub exemptions: &'a [String],
    /// `gradients.json.zero_grad_floor`, carried by the contract obligation.
    pub floor: f32,
    /// Encoder layer count, used to name the contracted components.
    pub layers: usize,
}

/// The contracted component a name belongs to.
///
/// Deliberately derived from [`FreezeGroup::matches`] rather than from a second
/// prefix table: ENC-04's four components ARE 01-07's four freeze groups, and
/// 01-07 asserts that partition covers every named parameter with no gaps. A
/// private copy here could drift from the mapping the freeze policy uses, and
/// then this gate and `frozen_gate` would be talking about different sets.
pub fn component_of(name: &str, layers: usize) -> Option<String> {
    if FreezeGroup::Embeddings.matches(name) {
        return Some("embeddings".to_string());
    }
    for n in 0..layers {
        if FreezeGroup::LayerAttention(n).matches(name) {
            return Some(format!("layer{n}.attention"));
        }
        if FreezeGroup::LayerFfn(n).matches(name) {
            return Some(format!("layer{n}.ffn"));
        }
        if FreezeGroup::LayerNorm(n).matches(name) {
            return Some(format!("layer{n}.norm"));
        }
    }
    None
}

/// **The canonical ENC-04 gate**, exactly as
/// `OBLIG-ENC-04-GRADIENT-AND-STEP-GATE` states it:
///
/// * (a) finiteness, every tensor, no exemptions;
/// * (b) non-zero AGGREGATE gradient L2 per contracted component;
/// * (c) per-tensor non-zero for names ABSENT from `exemptions`;
/// * (d)/(e) two-sided exemption: a name PRESENT in `exemptions` must itself
///   satisfy `max|g| <= floor`, so an unexpectedly LARGE key-bias gradient is a
///   failure rather than a pass — the list is a pinned prediction, not a mute
///   button;
/// * (f) when `deltas` is supplied, the same shape over post-step movement.
///
/// The earlier "non-zero gradient on every trainable tensor" phrasing is
/// unsatisfiable against a CORRECT implementation (`attention.self.key.bias` has
/// an analytically zero gradient by softmax shift invariance), which is why the
/// exemption is data-driven and two-sided.
///
/// `gradient_gate`, `frozen_gate` and `detach_negative` all call THIS function,
/// unmodified. That identity is what makes the D-24 negative evidence rather
/// than theater: a second implementation could be wrong in exactly the way that
/// lets both the positive and the negative pass.
///
/// # Errors
///
/// A report NAMING every offending parameter. `detach_negative` asserts on those
/// names, so the message content is part of the gate's contract.
pub fn assert_encoder_updates(input: &GateInput) -> Result<(), String> {
    let mut problems: Vec<String> = Vec::new();

    // ---- (a) finiteness, no exemptions ------------------------------------
    let mut missing: Vec<&str> = Vec::new();
    for (name, g) in input.grads {
        match g {
            None => missing.push(name),
            Some(values) => {
                if let Some(i) = values.iter().position(|v| !v.is_finite()) {
                    problems.push(format!(
                        "(a) `{name}` gradient element {i} is {} — not finite",
                        values[i]
                    ));
                }
            }
        }
    }
    if !missing.is_empty() {
        problems.push(format!(
            "(a) {} parameter(s) received NO gradient: {}",
            missing.len(),
            missing.join(", ")
        ));
    }

    // ---- (b) non-zero aggregate per contracted component ------------------
    let mut aggregates: BTreeMap<String, f64> = BTreeMap::new();
    let mut uncovered: Vec<&str> = Vec::new();
    for (name, g) in input.grads {
        match component_of(name, input.layers) {
            Some(c) => {
                let sq: f64 = g.as_ref().map_or(0.0, |v| {
                    v.iter().map(|x| f64::from(*x) * f64::from(*x)).sum()
                });
                *aggregates.entry(c).or_insert(0.0) += sq;
            }
            None => uncovered.push(name),
        }
    }
    if !uncovered.is_empty() {
        problems.push(format!(
            "the ENC-04 component partition does not cover: {} — the component mapping has \
             drifted from the parameter names",
            uncovered.join(", ")
        ));
    }
    for (component, sq) in &aggregates {
        if sq.sqrt() <= 0.0 {
            problems.push(format!(
                "(b) component `{component}` aggregate gradient L2 is {} — a severed graph \
                 anywhere in the body drives some component aggregate to exactly zero",
                sq.sqrt()
            ));
        }
    }

    // ---- (c) per-tensor non-zero, (d)/(e) two-sided exemption -------------
    for (name, g) in input.grads {
        let exempt = input.exemptions.iter().any(|e| e == name);
        let Some(values) = g else {
            continue; // already reported under (a)
        };
        if exempt {
            let m = max_abs(values);
            if m > input.floor {
                problems.push(format!(
                    "(e) `{name}` is on the analytically-zero list but max|g| = {m:e} > \
                     zero_grad_floor {:e}",
                    input.floor
                ));
            }
        } else if l2(values) <= 0.0 {
            problems.push(format!(
                "(c) `{name}` has a zero gradient and is NOT on the analytically-zero list"
            ));
        }
    }

    // ---- (f) post-step movement, same shape -------------------------------
    if let Some(deltas) = input.deltas {
        let mut moved: BTreeMap<String, f64> = BTreeMap::new();
        for (name, d) in deltas {
            if let Some(c) = component_of(name, input.layers) {
                *moved.entry(c).or_insert(0.0) += f64::from(*d);
            }
            if input.exemptions.iter().any(|e| e == name) {
                continue;
            }
            if *d <= 0.0 {
                problems.push(format!(
                    "(f) `{name}` did not move across the optimizer step and is not exempt"
                ));
                continue;
            }
            // D55 — `moved at all` is a very weak claim. At step 1 bias correction
            // gives m_hat = g and v_hat = g², so the update SATURATES to lr*sign(g):
            // every non-exempt tensor's max element must move by lr, not merely by
            // something positive. Measured across the fixture the band is
            // [1.00052, 1.00739] * lr — the residual is the decay term — so +/-10%
            // is loose by more than an order of magnitude while still catching a
            // wrong lr, a doubled step, or an update that is not sign-saturated.
            // Reference-free: it needs no fixture, only the configured lr.
            if let Some(lr) = input.step_lr {
                let ratio = *d / lr;
                if !(0.9..=1.1).contains(&ratio) {
                    problems.push(format!(
                        "(f) `{name}` moved {d:e}, which is {ratio:.4}x lr ({lr:e}); a step-1 \
                         AdamW update saturates to lr*sign(g) for any tensor carrying a real \
                         gradient, so this is outside [0.9, 1.1]x"
                    ));
                }
            }
        }
        for (component, total) in &moved {
            if *total <= 0.0 {
                problems.push(format!(
                    "(f) component `{component}` aggregate post-step delta is {total}"
                ));
            }
        }
    }

    if problems.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "ENC-04 gate FAILED with {} finding(s):\n  {}",
            problems.len(),
            problems.join("\n  ")
        ))
    }
}

/// `(name, gradient)` for the model's TRAINABLE partition.
///
/// **D42, and the single most important line in this harness.** The set comes
/// from `trainable_parameters_mut()` — never from `encoder().named_parameters_mut()`
/// filtered by `requires_grad`. 01-07's mutation D measured a `Linear` weight
/// with the flag cleared still receiving a gradient and still MOVING across an
/// optimizer step, because the ops consuming it register their edge on their
/// INPUT requiring grad and then produce gradients for both operands regardless
/// of the weight's own flag. Exclusion from this method is the load-bearing
/// mechanism; the flag protects only parameters whose consuming op checks it
/// (notably `embedding_gather`). Building the optimizer's parameter set the
/// other way would silently train frozen weights, and NO fixture parity gate in
/// this file could see it.
pub fn trainable_grads(model: &mut SetFitMiniLm) -> NamedGrads {
    model
        .trainable_parameters_mut()
        .into_iter()
        .map(|(name, t)| (name, get_grad(t.id()).map(|g| g.data().to_vec())))
        .collect()
}

/// Bitwise snapshot of every named parameter, in traversal order.
pub fn snapshot(model: &SetFitMiniLm) -> Vec<(String, Vec<u32>)> {
    model
        .encoder()
        .named_parameters()
        .into_iter()
        .map(|(n, t)| (n, t.data().iter().map(|v| v.to_bits()).collect()))
        .collect()
}

/// The recorded pair batch, rebuilt through the tokenizer, with the derived
/// fixture's `source` pointer asserted against what was actually loaded.
pub struct PairBatch {
    pub a: SentenceBatch,
    pub b: SentenceBatch,
    pub labels: Vec<f32>,
}

pub fn pair_batch(model: &SetFitMiniLm, source: &SourcePointer) -> PairBatch {
    let loss: LossFixture = read_fixture("loss_pair.json");
    assert_eq!(
        source.fixture, "loss_pair.json",
        "the derived fixture points at `{}`, not loss_pair.json — a regeneration moved the \
         pair batch and this gate would be comparing against the wrong reference",
        source.fixture
    );
    assert_eq!(source.a_case_id, loss.pair.a_case_id);
    assert_eq!(source.b_case_id, loss.pair.b_case_id);
    PairBatch {
        a: batch_from_case(model, &loss.pair.a_texts).expect("tokenize a"),
        b: batch_from_case(model, &loss.pair.b_texts).expect("tokenize b"),
        labels: loss.pair.labels,
    }
}

// ===========================================================================
// Source assertions — the structural half of the harness
// ===========================================================================

/// Every source file this harness is made of.
fn harness_sources() -> Vec<(PathBuf, String)> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");
    let mut files = vec![root.join("setfit_conformance.rs")];
    let dir = root.join("setfit_conformance");
    let entries = std::fs::read_dir(&dir).expect("tests/setfit_conformance/ must exist");
    let mut sub: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "rs"))
        .collect();
    sub.sort();
    files.extend(sub);
    files
        .into_iter()
        .map(|p| {
            let text = std::fs::read_to_string(&p)
                .unwrap_or_else(|e| panic!("{} unreadable: {e}", p.display()));
            (p, text)
        })
        .collect()
}

/// The part of `line` before the first `//`, i.e. everything the compiler sees.
fn code_of(line: &str) -> &str {
    line.split("//").next().unwrap_or("")
}

/// Needles are assembled at RUNTIME.
///
/// D40's lesson, one plan later: a source-scanning guard whose own needles are
/// contiguous literals inside the scanned tree trips its own gate. 01-07 met it
/// on the first run. Splitting each needle keeps the meaning while leaving the
/// source text clean for both this scan and the shell `grep` the SUMMARY quotes.
fn needle(head: &str, tail: &str) -> String {
    format!("{head}{tail}")
}

#[test]
fn conformance_the_tokenizer_boundary_is_the_only_batch_source() {
    let tokenize_call = needle(".token", "ize(");
    let batch_literal = needle("SentenceBatch ", "{");
    let mut tokenize_sites: Vec<String> = Vec::new();
    let mut literal_sites: Vec<String> = Vec::new();

    for (path, text) in harness_sources() {
        let mut current_fn = String::new();
        for (n, line) in text.lines().enumerate() {
            let code = code_of(line);
            if let Some(rest) = code.trim_start().strip_prefix("pub fn ") {
                current_fn = rest.split('(').next().unwrap_or("").to_string();
            } else if let Some(rest) = code.trim_start().strip_prefix("fn ") {
                current_fn = rest.split('(').next().unwrap_or("").to_string();
            }
            if code.contains(&tokenize_call) {
                tokenize_sites.push(format!(
                    "{}:{} (in fn `{current_fn}`)",
                    path.display(),
                    n + 1
                ));
            }
            if code.contains(&batch_literal) {
                literal_sites.push(format!("{}:{}", path.display(), n + 1));
            }
        }
    }

    assert!(
        literal_sites.is_empty(),
        "a batch literal was constructed outside the tokenizer boundary at: {literal_sites:?}"
    );
    assert_eq!(
        tokenize_sites.len(),
        1,
        "the tokenizer must be called exactly once in this harness, inside `batch_from_case`; \
         found {tokenize_sites:?}"
    );
    assert!(
        tokenize_sites[0].contains("batch_from_case"),
        "the single tokenize call is not inside `batch_from_case`: {}",
        tokenize_sites[0]
    );
}

#[test]
fn conformance_no_sealed_constructor_is_reached_from_this_harness() {
    // These are `pub(crate)` (01-05/01-06) so a call would not compile at all —
    // 01-07 recorded four E0624s (NOT E0603, D41) from an out-of-crate probe.
    // The scan catches an attempt to REOPEN the seal, which would compile and
    // would silently give this harness a construction path production lacks.
    let forbidden = [
        needle("MiniLmImport", "::"),
        needle("MiniLmTokenizer::", "from_bytes"),
        needle("BertSentenceEncoder::", "from_import"),
    ];
    let mut hits: Vec<String> = Vec::new();
    for (path, text) in harness_sources() {
        for (n, line) in text.lines().enumerate() {
            let code = code_of(line);
            for f in &forbidden {
                if code.contains(f.as_str()) {
                    hits.push(format!("{}:{} -> {f}", path.display(), n + 1));
                }
            }
        }
    }
    assert!(
        hits.is_empty(),
        "sealed construction path reached: {hits:?}"
    );
}

#[test]
fn conformance_the_harness_never_reads_a_pre_remapped_id_array() {
    // The remap must happen INSIDE the encoder. Not deserializing the fixtures'
    // pre-remapped arrays makes feeding them impossible rather than merely
    // forbidden — this scan proves the fields are absent from the schema.
    let forbidden = [
        needle("input_ids_", "slice"),
        needle("a_ids_", "slice"),
        needle("b_ids_", "slice"),
    ];
    let mut hits: Vec<String> = Vec::new();
    for (path, text) in harness_sources() {
        for (n, line) in text.lines().enumerate() {
            let code = code_of(line);
            for f in &forbidden {
                if code.contains(f.as_str()) {
                    hits.push(format!("{}:{} -> {f}", path.display(), n + 1));
                }
            }
        }
    }
    assert!(
        hits.is_empty(),
        "a pre-remapped slice-id array is referenced in harness CODE: {hits:?}"
    );
}

#[test]
fn conformance_no_hand_written_tolerance_literal_outside_the_generated_file() {
    // Every tolerance in this contract has the shape `<digits>e-<digits>`. A
    // scan for that shape in non-comment code catches the whole class, and the
    // case table below is re-RUN rather than the rule re-read (CLAUDE.md 7).
    //
    // The must-MATCH rows are ASSEMBLED AT RUNTIME. This file is inside the
    // scanned set, so contiguous literals would make the table trip its own
    // gate — D40, one plan later. It did, on the first run here too: the scan
    // reported exactly these three rows and nothing else, which is also the
    // evidence that this guard can turn red.
    let e = needle("e", "-");
    let table: [(String, bool); 8] = [
        (format!("    let tol = 1.5{e}5;"), true),
        (format!("    assert!(d <= 7.62939453{e}06);"), true),
        (format!("    let eps = 1{e}12;"), true),
        ("    let labels = [1.0, 0.0];".to_string(), false),
        ("    assert_eq!(shape.hidden, 64);".to_string(), false),
        (format!("    // measured 4.73{e}04 in 01-04"), false),
        ("    let n = counts[0] - 1;".to_string(), false),
        ("    let scaled = x * 2.0;".to_string(), false),
    ];
    for (row, want) in &table {
        assert_eq!(
            has_negative_exponent_literal(code_of(row)),
            *want,
            "case table row misclassified: {row:?}"
        );
    }

    let mut hits: Vec<String> = Vec::new();
    for (path, text) in harness_sources() {
        if path.ends_with("tolerances_generated.rs") {
            continue;
        }
        for (n, line) in text.lines().enumerate() {
            if has_negative_exponent_literal(code_of(line)) {
                hits.push(format!("{}:{} -> {}", path.display(), n + 1, line.trim()));
            }
        }
    }
    assert!(
        hits.is_empty(),
        "hand-written tolerance literal(s) outside tolerances_generated.rs: {hits:?}"
    );
}

/// `<digit>e-<digit>` — the shape every tolerance in this contract has.
fn has_negative_exponent_literal(code: &str) -> bool {
    let b: Vec<char> = code.chars().collect();
    for i in 0..b.len() {
        if b[i] != 'e' && b[i] != 'E' {
            continue;
        }
        let before = i > 0 && b[i - 1].is_ascii_digit();
        let after = i + 2 < b.len() && b[i + 1] == '-' && b[i + 2].is_ascii_digit();
        if before && after {
            return true;
        }
    }
    false
}

// ===========================================================================
// D-14: the tolerances originate ONLY in the contract
// ===========================================================================

/// `(constant name, generated value, obligation id it comes from)`.
///
/// One table, used by BOTH the agreement test and the generator, so the two can
/// never disagree about which obligation feeds which constant.
fn tolerance_bindings() -> Vec<(&'static str, f32, &'static str)> {
    vec![
        (
            "FORWARD_PER_LAYER",
            tol::FORWARD_PER_LAYER,
            "OBLIG-ENC-03-PER-LAYER-FORWARD-PARITY",
        ),
        (
            "POOLING_NORMALIZE",
            tol::POOLING_NORMALIZE,
            "OBLIG-ENC-03-POOLED-EMBEDDING-PARITY",
        ),
        (
            "ACTIVATION",
            tol::ACTIVATION,
            "OBLIG-ENC-03-ACTIVATION-PARITY",
        ),
        (
            "BATCH_INVARIANCE",
            tol::BATCH_INVARIANCE,
            "OBLIG-ENC-03-PADDING-INVARIANCE",
        ),
        (
            "GRADIENTS",
            tol::GRADIENTS,
            "OBLIG-ENC-04-NAMED-GRADIENT-PARITY",
        ),
        (
            "ZERO_GRAD_FLOOR",
            tol::ZERO_GRAD_FLOOR,
            "OBLIG-ENC-04-GRADIENT-AND-STEP-GATE",
        ),
        (
            "LOSS_PAIR",
            tol::LOSS_PAIR,
            "OBLIG-ENC-06-LOSS-FORWARD-PARITY",
        ),
        (
            "OPTIMIZER_STEP",
            tol::OPTIMIZER_STEP,
            "OBLIG-ENC-04-POST-STEP-PARAMETER-PARITY",
        ),
        (
            "OPTIMIZER_MULTISTEP",
            tol::OPTIMIZER_MULTISTEP,
            "OBLIG-ENC-04-MULTISTEP-TRAJECTORY-PARITY",
        ),
        (
            "FULL_MODEL_REFERENCE",
            tol::FULL_MODEL_REFERENCE,
            "OBLIG-ENC-01-FULL-MODEL-REFERENCE-PARITY",
        ),
    ]
}

/// Read `(obligation id -> tolerance)` from the contract with the SAME
/// deserializer `pv validate` uses.
///
/// Not `serde_yaml` by hand: re-implementing a reader the workspace already
/// owns is muda (CLAUDE.md) and would be a second interpretation of the schema
/// that could drift from pv's. `pv codegen` and `pv equations` were both checked
/// first and neither emits tolerances — `pv codegen` walks only
/// preconditions/postconditions/invariants, and `pv equations --format` accepts
/// `text|latex|ptx|asm`.
fn contract_tolerances(path: &Path) -> BTreeMap<String, f64> {
    let contract = provable_contracts::schema::parse_contract(path)
        .unwrap_or_else(|e| panic!("{} failed to parse: {e}", path.display()));
    let mut out = BTreeMap::new();
    for o in &contract.proof_obligations {
        let Some(t) = o.tolerance else { continue };
        // The obligation id is the first whitespace-delimited token of
        // `property`, followed by a colon.
        if let Some(id) = o.property.split_whitespace().next() {
            out.insert(id.trim_end_matches(':').to_string(), t);
        }
    }
    out
}

fn contract_version(path: &Path) -> String {
    provable_contracts::schema::parse_contract(path)
        .unwrap_or_else(|e| panic!("{} failed to parse: {e}", path.display()))
        .metadata
        .version
}

pub fn sha256_file(path: &Path) -> String {
    let bytes =
        std::fs::read(path).unwrap_or_else(|e| panic!("{} unreadable: {e}", path.display()));
    let mut h = Sha256::new();
    h.update(&bytes);
    format!("{:x}", h.finalize())
}

#[test]
fn conformance_tolerances_agree_with_the_contract() {
    let Some(path) = contract_path() else {
        // The ONLY skip in this suite, and it is documented: the workspace-root
        // `contracts/` directory is not part of the packaged crate, so inside an
        // unpacked `.crate` there is nothing to agree with. The digest is still
        // required to be well formed so a corrupted header cannot pass.
        assert_eq!(
            tol::CONTRACT_SHA256.len(),
            64,
            "CONTRACT_SHA256 is not a 64-character digest"
        );
        assert!(
            tol::CONTRACT_SHA256
                .chars()
                .all(|c: char| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "CONTRACT_SHA256 is not lowercase hex"
        );
        return;
    };

    let digest = sha256_file(&path);
    assert_eq!(
        digest,
        tol::CONTRACT_SHA256,
        "the contract has changed since tolerances_generated.rs was generated. \
         Regenerate it: {}",
        tol::REGENERATE_COMMAND
    );

    let from_contract = contract_tolerances(&path);
    for (name, generated, obligation) in tolerance_bindings() {
        let want = from_contract.get(obligation).unwrap_or_else(|| {
            panic!(
                "obligation `{obligation}` carries no tolerance in {}",
                path.display()
            )
        });
        #[allow(clippy::cast_possible_truncation)]
        let want32 = *want as f32;
        assert!(
            generated.to_bits() == want32.to_bits(),
            "tolerance drift: {name} is {generated:e} in tolerances_generated.rs but \
             {want32:e} in {obligation}. The generated file is DERIVED — regenerate it \
             ({}), never hand-edit it.",
            tol::REGENERATE_COMMAND
        );
    }

    assert_eq!(
        tol::CONTRACT_VERSION,
        contract_version(&path),
        "generated file records a stale contract version"
    );
}

/// The generator. `#[ignore]`d, and a no-op unless `APRENDER_REGEN_TOLERANCES`
/// is set — so a bulk `-- --ignored` run (the D-10 full-weight path) re-checks
/// agreement rather than silently rewriting a committed file.
///
/// ```text
/// APRENDER_REGEN_TOLERANCES=1 cargo test -p aprender-core \
///   --features setfit,conformance-fixtures --test setfit_conformance \
///   conformance_tolerances_regenerate -- --ignored
/// ```
#[test]
#[ignore = "generator: writes tolerances_generated.rs from the contract"]
fn conformance_tolerances_regenerate() {
    if std::env::var("APRENDER_REGEN_TOLERANCES").is_err() {
        conformance_tolerances_agree_with_the_contract();
        return;
    }
    let path = contract_path().expect("regeneration requires a workspace checkout");
    let from_contract = contract_tolerances(&path);
    let digest = sha256_file(&path);
    let version = contract_version(&path);

    let mut out = String::new();
    out.push_str("//! GENERATED FILE — DO NOT HAND-EDIT.\n//!\n");
    out.push_str("//! Tolerance constants for the SetFit conformance harness (plan 01-08).\n//!\n");
    out.push_str("//! Source contract: contracts/setfit-encoder-conformance-v1.yaml\n");
    out.push_str(&format!("//! Contract metadata.version: {version}\n"));
    out.push_str(&format!("//! Contract sha256: {digest}\n//!\n"));
    out.push_str("//! Regenerate with:\n//!\n");
    out.push_str("//! ```text\n");
    out.push_str("//! APRENDER_REGEN_TOLERANCES=1 cargo test -p aprender-core \\\n");
    out.push_str("//!   --features setfit,conformance-fixtures --test setfit_conformance \\\n");
    out.push_str("//!   conformance_tolerances_regenerate -- --ignored\n");
    out.push_str("//! ```\n//!\n");
    // D7's lesson applied to this generator: a generator whose output differs
    // from the committed, rustfmt'd file makes every future drift check 100%
    // noise. The emitter below is written to be rustfmt-STABLE, and `cargo fmt`
    // is named here so a future edit that breaks that has an obvious remedy.
    out.push_str("//! The emitter is rustfmt-stable; if a future edit breaks that, run\n");
    out.push_str("//! `cargo fmt -p aprender-core` after regenerating, so a drift check on\n");
    out.push_str("//! this file reports semantics rather than whitespace (D7).\n//!\n");
    out.push_str("//! D-14: these numbers exist in ONE place, the versioned contract. Widening\n");
    out.push_str("//! one requires a contract edit `pv diff` flags with a semver bump. The\n");
    out.push_str(
        "//! agreement test in tests/setfit_conformance.rs fails the build if this file\n",
    );
    out.push_str("//! and the contract ever disagree in a workspace checkout.\n\n");

    for (name, _, obligation) in tolerance_bindings() {
        let v = from_contract
            .get(obligation)
            .unwrap_or_else(|| panic!("obligation `{obligation}` carries no tolerance"));
        out.push_str(&format!("/// From `{obligation}`.\n"));
        out.push_str(&format!("pub const {name}: f32 = {v:.8e};\n"));
    }
    out.push_str("\n/// sha256 of the source contract at generation time.\n");
    // Pre-wrapped: unwrapped this line is 101 columns and rustfmt would break it,
    // making the generator's output differ from the committed file.
    out.push_str(&format!(
        "pub const CONTRACT_SHA256: &str =\n    \"{digest}\";\n"
    ));
    out.push_str("\n/// `metadata.version` of the source contract at generation time.\n");
    out.push_str(&format!(
        "pub const CONTRACT_VERSION: &str = \"{version}\";\n"
    ));
    out.push_str("\n/// The command that regenerates this file.\n");
    out.push_str(
        "pub const REGENERATE_COMMAND: &str = \"APRENDER_REGEN_TOLERANCES=1 cargo test \
         -p aprender-core --features setfit,conformance-fixtures --test setfit_conformance \
         conformance_tolerances_regenerate -- --ignored\";\n",
    );

    let dest = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/setfit_conformance/tolerances_generated.rs");
    std::fs::write(&dest, out).expect("write generated tolerances");
}

// ===========================================================================
// Fixture integrity and the B6 join
// ===========================================================================

#[test]
fn conformance_manifest_self_check() {
    // Catches a silent LOCAL edit of a frozen fixture. Without it every parity
    // gate below is comparing against whatever is on disk rather than against
    // what 01-04 froze.
    let dir = fixtures_dir();
    let manifest = std::fs::read_to_string(dir.join("manifest.sha256"))
        .expect("manifest.sha256 must be present");
    let mut checked = 0usize;
    for line in manifest.lines() {
        let mut parts = line.split_whitespace();
        let (Some(want), Some(name)) = (parts.next(), parts.next()) else {
            continue;
        };
        let got = sha256_file(&dir.join(name));
        assert_eq!(
            got, want,
            "fixture `{name}` has been modified since 01-04 froze it"
        );
        checked += 1;
    }
    assert!(checked >= 15, "manifest covered only {checked} files");
}

/// The triple equality: fixture ids == corpus-of-record ids == what the Rust
/// tokenizer produces from the recorded texts.
fn assert_case_join(
    model: &SetFitMiniLm,
    cases: &TokenizerCases,
    case_id: &str,
    texts: &[String],
    canonical: &[Vec<u32>],
) {
    let recorded = cases.get(case_id);
    assert_eq!(
        recorded.texts, texts,
        "`{case_id}`: the fixture's texts differ from tokenizer_cases.json"
    );
    assert_eq!(
        recorded.input_ids, canonical,
        "`{case_id}`: tokenizer_cases.json ids differ from the fixture's canonical ids"
    );
    let batch = batch_from_case(model, texts).expect("tokenize");
    let flat: Vec<u32> = canonical.iter().flatten().copied().collect();
    assert_eq!(
        batch.input_ids(),
        flat.as_slice(),
        "`{case_id}`: the Rust tokenizer disagrees with the frozen ids"
    );
}

#[test]
fn conformance_every_fixture_case_id_joins_the_corpus_of_record() {
    let model = slice_model();
    let cases: TokenizerCases = read_fixture("tokenizer_cases.json");

    let forward: ForwardFixture = read_fixture("forward_per_layer.json");
    for c in &forward.cases {
        assert_case_join(&model, &cases, &c.case_id, &c.texts, &c.input_ids_canonical);
    }
    let pooling: PoolingFixture = read_fixture("pooling_normalize.json");
    for c in &pooling.cases {
        assert_case_join(&model, &cases, &c.case_id, &c.texts, &c.input_ids_canonical);
    }
    let loss: LossFixture = read_fixture("loss_pair.json");
    assert_case_join(
        &model,
        &cases,
        &loss.pair.a_case_id,
        &loss.pair.a_texts,
        &loss.pair.a_ids_canonical,
    );
    assert_case_join(
        &model,
        &cases,
        &loss.pair.b_case_id,
        &loss.pair.b_texts,
        &loss.pair.b_ids_canonical,
    );
    let inv: InvarianceFixture = read_fixture("batch_invariance.json");
    assert_case_join(
        &model,
        &cases,
        &inv.single.case_id,
        &inv.single.texts,
        &inv.single.input_ids_canonical,
    );
    assert_case_join(
        &model,
        &cases,
        &inv.padded_batch.case_id,
        &inv.padded_batch.texts,
        &inv.padded_batch.input_ids_canonical,
    );
}

#[test]
fn conformance_the_slice_batch_carries_canonical_ids_the_encoder_must_remap() {
    // Without this the "remap happens inside the encoder" claim would be about
    // an unreachable branch: if every recorded id already fell inside the
    // 97-row slice vocabulary, a no-op remap would pass every gate in this file.
    let model = slice_model();
    let loss: LossFixture = read_fixture("loss_pair.json");
    let batch = batch_from_case(&model, &loss.pair.a_texts).expect("tokenize");
    let above = batch.input_ids().iter().filter(|id| **id >= 97).count();
    assert!(
        above > 0,
        "no canonical id in the pair batch exceeds the slice vocabulary, so the remap \
         inside the encoder is never exercised by these gates"
    );
}

/// `[B, H]` unit-norm embeddings through the production `encode` path.
pub fn encode(model: &SetFitMiniLm, batch: &SentenceBatch) -> Tensor {
    model.encoder().encode(batch).expect("encode")
}
