//! Stage one: the encoder-tuning loop, record-only.
//!
//! Contract: `setfit-train-lifecycle-v1` (authored in plan 03-06). Requirements: TRN-03
//! (SetFit identity evidence), TRN-06 (bitwise reproducibility).
//!
//! # This module RECORDS. It does not JUDGE.
//!
//! [`run_tuning`] is `pub(crate)` and returns a [`TuneOutput`]. It mints no
//! `SetFitRun<EncoderTuned>` and makes no pass/fail comparison, because the per-class
//! epsilon a verdict would compare against does not exist yet — plan 03-05 MEASURES the
//! distribution the epsilon is frozen from, and plan 03-06 freezes it into the contract
//! BEFORE any gate is armed. Arming a gate against a threshold derived from the same run it
//! judges is the Ph1 D-14 failure this ordering exists to prevent.
//!
//! # It records what it CONSUMED, never what its configuration implies
//!
//! Two streaming SHA-256 digests are fed at the point of consumption:
//! [`absorb_batch_digests`] is called once per pair, inside the pair-pull loop, at the moment
//! that pair is drawn; [`absorb_boundary_digest`] once per batch, at batch open. Neither is
//! ever recomputed from `(root_seed, epochs, batch_size)` afterwards. The difference is not
//! stylistic: a digest reconstructed from configuration reproduces perfectly across two runs
//! that consumed the SAME WRONG ORDER, so it certifies the intent instead of the execution.
//! `tune_digest_is_recorded_not_recomputed` reverses the intra-batch pull and asserts the
//! pair digest MOVES while the boundary digest does not.
//!
//! # Two measured facts about `aprender-core`'s autograd that shaped this loop
//!
//! Both were verified by reading and then by test, because the plan's context section stated
//! the opposite of each and either error would have produced a loop that looks right.
//!
//! **1. `aprender-train`'s `Tensor` is NOT `aprender-core`'s autograd `Tensor`.** They are
//! two unrelated types from two unrelated autograd engines — `aprender_train::autograd::
//! tensor::Tensor` wraps an `Array1<f32>` with an `Rc<RefCell<Option<Array1<f32>>>>`
//! gradient and an `Rc<dyn BackwardOp>`, while `aprender::autograd::Tensor` carries a
//! `Vector<f32>` plus a shape and is differentiated through a thread-local tape. The
//! encoder's parameters are the latter; `AdamW::step_refs` and `clip_grad_norm_refs` take
//! the former. [`ParamBridge`] is the explicit, tested adapter that lets this loop REUSE the
//! reference optimizer stack rather than grow a second AdamW.
//!
//! **2. Gradients do not live on the parameter tensors.** `ComputationGraph::backward`
//! writes them into the graph's own registry copies (`autograd/graph.rs:145`), so
//! `param.grad()` on an encoder parameter is `None` even immediately after a successful
//! backward; the gradient is reached with `autograd::get_grad(param.id())`. A loop that read
//! `param.grad()` would clip nothing, step nothing, and report a falling loss — PF-001 in a
//! new costume. `tune_gradients_reach_the_parameters_through_the_graph` pins the mechanism.
//!
//! # No `par_iter`, anywhere
//!
//! Every reduction goes through [`super::reduce`], in index order. See that module for why.

use std::collections::BTreeMap;

use aprender::autograd::{self, Tensor as CoreTensor};
use aprender::setfit::{dropout_rng, pair_cosine_mse, FreezeGroup, SetFitMiniLm};
use aprender_contrastive_data::pairs::PairSampler;
use aprender_contrastive_data::prepared::{Canonical, PreparedDataset};
use aprender_contrastive_data::select::Selection;
use ndarray::{Array1, ArrayView1};
use sha2::{Digest, Sha256};

use crate::optim::{
    clip_grad_norm_refs, warmup_steps_from_ratio, AdamW, LRScheduler, Optimizer,
    WarmupLinearDecayLR,
};
use crate::train::device::Device;
use crate::Tensor as TrainTensor;

use super::config::{
    ResolvedSetFitConfig, ADAMW_BETA1, ADAMW_BETA2, ADAMW_EPSILON, ADAMW_WEIGHT_DECAY,
};
use super::epoch::epoch_pair_order;
use super::evidence::{EvidenceRow, EvidenceSummary, ParameterClass, UpdateEvidence, Verdict};
use super::reduce;
use super::thresholds::{RegimeThresholds, Thresholds};
use super::SetFitTrainError;

/// The floor learning rate the reference schedule decays to.
const SCHEDULE_LR_MIN: f32 = 0.0;

/// Upper bound on the endpoint window `k`.
const ENDPOINT_K_CAP: usize = 5;

// ===========================================================================================
// Recorded output
// ===========================================================================================

/// What one parameter did, measured through [`super::reduce`] only.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ParamRecord {
    /// Parameter class is assigned in `evidence.rs`; this record is class-agnostic.
    pub(crate) element_count: u64,
    /// `||theta_init||_2` over the WHOLE initial tensor.
    pub(crate) init_norm: f64,
    /// `||theta_init||_2` restricted to the SUPPORT of the delta.
    pub(crate) init_norm_on_support: f64,
    /// `||theta_final - theta_init||_2`.
    pub(crate) delta_norm: f64,
    /// Elements whose delta is not exactly zero.
    pub(crate) delta_support_count: u64,
    /// Largest per-step PRE-clip gradient norm observed for this parameter.
    pub(crate) grad_norm_max: f64,
    /// Index-order mean of the per-step PRE-clip gradient norms.
    pub(crate) grad_norm_mean: f64,
    /// Steps at which this parameter had a gradient at all.
    pub(crate) steps_observed: u64,
}

/// One step's recorded observations, written BY the loop as it ran.
///
/// These are not reconstructions. `applied_lr` is the value handed to the optimizer at step
/// (a); `tape_len_at_step_top` is read after step (b) and before step (c). A test that
/// recomputed them from the scheduler afterwards would prove only that the scheduler is a
/// pure function of its step counter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct StepObservation {
    /// Global optimizer step, from 0.
    pub(crate) global_step: u64,
    /// `graph_tape_len()` at the top of the step, after the clear.
    pub(crate) tape_len_at_step_top: usize,
    /// Trainable parameters whose TENSOR-side `grad()` was `Some` at the top of the step.
    pub(crate) tensor_grads_at_step_top: usize,
    /// Trainable parameters whose GRAPH-side gradient was present at the top of the step.
    ///
    /// This is the one that matters: see the module doc's measured fact 2.
    pub(crate) graph_grads_at_step_top: usize,
    /// The learning rate actually applied to the optimizer for this step.
    pub(crate) applied_lr: f32,
    /// The two forward ordinals the two siamese branches drew their dropout masks at.
    pub(crate) forward_ordinals: (u64, u64),
    /// The PRE-clip global gradient norm `clip_grad_norm_refs` returned.
    pub(crate) pre_clip_norm: f32,
}

/// Everything stage one recorded. Record-only: no verdict, no threshold, no `Duration`.
///
// The recorded fields are the source plan 03-08 persists into the bundle and 03-09
// evaluates from. They are written here and read there, so they read as dead until that
// plan lands; drop this allow once 03-08 consumes them.
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct TuneOutput {
    /// The tuned encoder.
    pub(crate) encoder: SetFitMiniLm,
    /// The FULL initial trainable tensors. Norms alone cannot produce
    /// `||theta_final - theta_init||`, so the snapshot is the whole tensor.
    pub(crate) initial_snapshot: BTreeMap<String, Vec<f32>>,
    /// The snapshot's size in bytes — the memory accounting, recorded rather than discovered.
    pub(crate) snapshot_bytes: usize,
    /// Per-parameter measurements, in `BTreeMap` name order.
    pub(crate) per_name: BTreeMap<String, ParamRecord>,
    /// PRE-clip global gradient norm per step, in step order.
    pub(crate) pre_clip_norms: Vec<f32>,
    /// Scalar loss per step, in step order.
    pub(crate) loss_trace: Vec<f32>,
    /// SHA-256 over the loss trace's little-endian `f32` bit patterns, in step order.
    pub(crate) loss_trace_hash: [u8; 32],
    /// Optimizer steps taken.
    pub(crate) step_count: u64,
    /// SHA-256 over the ordered trainable parameter names.
    pub(crate) parameter_registry_hash: [u8; 32],
    /// In-band digest of every pair actually consumed, in consumption order.
    pub(crate) consumed_pair_digest: [u8; 32],
    /// In-band digest of every batch boundary actually opened.
    pub(crate) batch_boundary_digest: [u8; 32],
    /// `(epoch, start_ordinal, len)` per batch, in the order the batches opened.
    ///
    /// Retained as a readable list as well as a digest because plan 03-08's public
    /// `batch_boundaries()` accessor returns it: without a recorded source that accessor
    /// would have to RECOMPUTE the boundaries, which is precisely the false-green the in-band
    /// digests remove.
    pub(crate) batch_boundaries: Vec<(u32, u64, u32)>,
    /// The consumed `max_length`, carried so 03-08's bundle records it.
    pub(crate) max_length: u32,
    /// Mean of the first `k` losses. Computed, NOT judged here.
    pub(crate) first_k_mean: f64,
    /// Mean of the last `k` losses. Computed, NOT judged here.
    pub(crate) last_k_mean: f64,
    /// `k = min(5, step_count / 2)`.
    pub(crate) endpoint_k: usize,
    /// Eval-mode embeddings of the selected rows BEFORE tuning, in Selection order.
    pub(crate) embeddings_before: Vec<Vec<f32>>,
    /// Eval-mode embeddings of the selected rows AFTER tuning, in Selection order.
    pub(crate) embeddings_after: Vec<Vec<f32>>,
    /// Trainable parameter count after `apply_freeze`.
    pub(crate) trainable_count: usize,
    /// Frozen parameter count after `apply_freeze`.
    pub(crate) frozen_count: usize,
    /// Per-step observations, recorded in band.
    pub(crate) steps: Vec<StepObservation>,
    /// Tape length observed across the pre-tuning baseline encode: `(before, after)`.
    pub(crate) baseline_encode_tape: (usize, usize),
}

// ===========================================================================================
// Test probes
// ===========================================================================================

/// Deliberate defects a test can induce in the loop, to prove a step is load-bearing.
///
/// The fields are PRIVATE and the only non-default constructors are `#[cfg(test)]`, so a
/// production caller cannot build anything but [`TuningProbes::NONE`]. A probe that could
/// disable a correctness step in a shipped build would be worse than the bug it tests for.
///
/// # Why the clearing needs TWO probes
///
/// The loop clears twice — step (b) before the forward and step (l) after the step — and a
/// negative that removes only ONE of them is a negative that measures nothing, because the
/// other still leaves an empty tape. Measured: with only step (b) removed, steps 1 and 2 saw
/// tape length 0 and the first draft of the accumulation test failed for that reason rather
/// than for the reason it was written. The two probes separate the two properties:
/// [`Self::SKIP_STEP_TOP_CLEAR`] shows that (b) is what makes "the tape is empty when this
/// step's forward begins" true regardless of what ran BEFORE the loop, and
/// [`Self::SKIP_ALL_GRAPH_CLEARS`] shows the unbounded growth and the stale graph-side
/// gradient that the clearing as a mechanism prevents (T-3-55).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TuningProbes {
    skip_step_top_clear: bool,
    skip_post_step_clear: bool,
    reverse_intra_batch_pull: bool,
    poison_first_loss: bool,
}

impl TuningProbes {
    /// The only value production can construct.
    pub(crate) const NONE: Self = Self {
        skip_step_top_clear: false,
        skip_post_step_clear: false,
        reverse_intra_batch_pull: false,
        poison_first_loss: false,
    };

    /// Skip step (b) only: neither `zero_grad_` nor the pre-forward `clear_graph` runs.
    #[cfg(test)]
    pub(crate) const SKIP_STEP_TOP_CLEAR: Self = Self {
        skip_step_top_clear: true,
        skip_post_step_clear: false,
        reverse_intra_batch_pull: false,
        poison_first_loss: false,
    };

    /// Skip the clearing entirely — both step (b) and step (l).
    #[cfg(test)]
    pub(crate) const SKIP_ALL_GRAPH_CLEARS: Self = Self {
        skip_step_top_clear: true,
        skip_post_step_clear: true,
        reverse_intra_batch_pull: false,
        poison_first_loss: false,
    };

    /// Reverse the pair-pull order INSIDE `run_batch`, leaving batch structure identical.
    #[cfg(test)]
    pub(crate) const REVERSE_INTRA_BATCH_PULL: Self = Self {
        skip_step_top_clear: false,
        skip_post_step_clear: false,
        reverse_intra_batch_pull: true,
        poison_first_loss: false,
    };

    /// Force step 0's loss to `NaN`, leaving every other step untouched.
    ///
    /// The in-band negative for the `loss_trace_hash` finiteness precondition (REVIEW CR-03).
    /// A real divergence cannot be summoned from the pinned fixture on demand, and the
    /// alternative — unit-testing `f64::is_finite` — would prove the standard library works
    /// rather than that the LOOP refuses a diverged step. This injects at the one point the
    /// check guards, so removing the check turns the test red.
    #[cfg(test)]
    pub(crate) const POISON_FIRST_LOSS: Self = Self {
        skip_step_top_clear: false,
        skip_post_step_clear: false,
        reverse_intra_batch_pull: false,
        poison_first_loss: true,
    };
}

// ===========================================================================================
// The parameter bridge (measured fact 1)
// ===========================================================================================

/// Carries encoder parameters and their graph-side gradients across the two autograd engines.
///
/// One `TrainTensor` per trainable parameter, index-aligned with `trainable_parameters_mut()`
/// and allocated ONCE pre-loop — which is also what keeps `AdamW`'s positional moment state
/// paired with the right parameter, since the optimizer indexes `m`/`v` by slice position.
#[derive(Debug)]
struct ParamBridge {
    tensors: Vec<TrainTensor>,
}

impl ParamBridge {
    fn new(params: &[(String, &mut CoreTensor)]) -> Self {
        Self {
            tensors: params
                .iter()
                .map(|(_, t)| TrainTensor::from_vec(t.data().to_vec(), true))
                .collect(),
        }
    }

    /// Copy current values in, and the GRAPH-side gradients in as the bridge tensors' grads.
    ///
    /// A parameter with no graph-side gradient gets its bridge gradient cleared, so a stale
    /// gradient from the previous step cannot be stepped on twice.
    fn load(&mut self, params: &[(String, &mut CoreTensor)]) {
        for (slot, (_, param)) in self.tensors.iter_mut().zip(params.iter()) {
            slot.data_mut().assign(&ArrayView1::from(param.data()));
            match autograd::get_grad(param.id()) {
                Some(grad) => slot.set_grad(Array1::from_vec(grad.data().to_vec())),
                None => slot.zero_grad(),
            }
        }
    }

    /// Copy the optimizer's updated values back onto the encoder's parameters.
    fn store(&self, params: &mut [(String, &mut CoreTensor)]) {
        for (slot, (_, param)) in self.tensors.iter().zip(params.iter_mut()) {
            let updated = slot.data();
            param.data_mut().copy_from_slice(
                updated.as_slice().expect("a bridge tensor built from a Vec is contiguous"),
            );
        }
    }

    fn refs(&mut self) -> Vec<&mut TrainTensor> {
        self.tensors.iter_mut().collect()
    }
}

// ===========================================================================================
// In-band digest absorption (T-3-38)
// ===========================================================================================

/// Absorb ONE consumed pair, at the moment it is drawn.
///
/// Called from inside `run_batch`'s pair-pull loop and from nowhere else. Absorbing from a
/// returned `BatchRecord` would be absorbing a DESCRIPTION of the consumption rather than the
/// consumption itself, which is the recomputation this phase's structural fix removed.
fn absorb_batch_digests(
    hasher: &mut Sha256,
    epoch: u32,
    batch_index_in_epoch: u32,
    position_in_batch: u32,
    pair_ordinal: u64,
    a_selected_index: u32,
    b_selected_index: u32,
    target: f32,
) {
    hasher.update(epoch.to_le_bytes());
    hasher.update(batch_index_in_epoch.to_le_bytes());
    hasher.update(position_in_batch.to_le_bytes());
    hasher.update(pair_ordinal.to_le_bytes());
    hasher.update(a_selected_index.to_le_bytes());
    hasher.update(b_selected_index.to_le_bytes());
    hasher.update(target.to_bits().to_le_bytes());
}

/// Absorb ONE batch boundary, at batch open.
fn absorb_boundary_digest(
    hasher: &mut Sha256,
    epoch: u32,
    batch_index_in_epoch: u32,
    global_step: u64,
    batch_start_ordinal: u64,
    batch_len: u32,
) {
    hasher.update(epoch.to_le_bytes());
    hasher.update(batch_index_in_epoch.to_le_bytes());
    hasher.update(global_step.to_le_bytes());
    hasher.update(batch_start_ordinal.to_le_bytes());
    hasher.update(batch_len.to_le_bytes());
}

// ===========================================================================================
// Pre-loop
// ===========================================================================================

/// What the six pre-loop steps produced.
#[derive(Debug)]
struct Preflight {
    registry_hash: [u8; 32],
    snapshot: BTreeMap<String, Vec<f32>>,
    snapshot_bytes: usize,
    trainable_count: usize,
    frozen_count: usize,
}

/// Steps 1-5: device, `max_length`, freeze, registry hash, snapshot.
///
/// `apply_freeze` runs BEFORE the hash and BEFORE the snapshot, which is what makes its
/// zero-match typed error (T-3-19) fire before any evidence is captured rather than at the
/// first update.
///
/// # Why this takes three values instead of a `&ResolvedSetFitConfig`
///
/// The two rejections here are the CONSUMPTION of knobs that 03-03 validated and nobody read
/// — review fix 8. A `ResolvedSetFitConfig` is only mintable by `resolve()`, which probes the
/// host, so a CPU-only machine cannot construct a CUDA-resolved configuration and the device
/// rejection would have had no reachable test. Taking the three values directly makes both
/// negatives testable without a test-only constructor on the validated config type, which
/// would have been a wider door than the thing it tests.
fn preflight(
    encoder: &mut SetFitMiniLm,
    device: Device,
    requested_max_length: u32,
    freeze_policy: &[FreezeGroup],
) -> Result<Preflight, SetFitTrainError> {
    if device != Device::Cpu {
        return Err(SetFitTrainError::UnsupportedDeviceForPhase3 { resolved: device.tag() });
    }
    let pinned = super::config::pinned_max_length();
    if requested_max_length != pinned {
        return Err(SetFitTrainError::MaxLengthNotConsumable {
            requested: requested_max_length,
            pinned,
        });
    }

    encoder
        .apply_freeze(freeze_policy)
        .map_err(|e| SetFitTrainError::Encoder { reason: e.to_string() })?;

    let frozen_count = encoder.frozen_parameters().len();
    let params = encoder.trainable_parameters_mut();
    let registry_hash = registry_hash_of(&params);
    let snapshot = snapshot_initial(&params);
    let snapshot_bytes = snapshot.values().map(|v| v.len() * 4).sum();

    Ok(Preflight {
        registry_hash,
        snapshot,
        snapshot_bytes,
        trainable_count: params.len(),
        frozen_count,
    })
}

/// SHA-256 over the ordered dotted names, each length-prefixed.
///
/// Length-prefixed so `["ab", "c"]` and `["a", "bc"]` cannot collide (T-3-54).
fn registry_hash_of(params: &[(String, &mut CoreTensor)]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for (name, _) in params {
        let len = u32::try_from(name.len()).unwrap_or(u32::MAX);
        hasher.update(len.to_le_bytes());
        hasher.update(name.as_bytes());
    }
    hasher.finalize().into()
}

/// Clone the FULL initial trainable tensors.
///
/// Norms alone cannot produce `||theta_final - theta_init||`; only the values can. For the
/// 2-layer slice this is ~110 K parameters (~440 KB). For the full `all-MiniLM-L6-v2` it is
/// ~22.7 M parameters, ~91 MB, and peak tuning memory is that snapshot PLUS the live
/// parameters PLUS AdamW's two moment buffers — roughly four copies of the model. The
/// accounting is stated here and recorded in `TuneOutput::snapshot_bytes` so it is not
/// discovered in Phase 5.
fn snapshot_initial(params: &[(String, &mut CoreTensor)]) -> BTreeMap<String, Vec<f32>> {
    params.iter().map(|(name, t)| (name.clone(), t.data().to_vec())).collect()
}

/// Eval-mode, `no_grad` encode of the selected rows in pinned consecutive Selection-order
/// windows.
///
/// Returns the embeddings and the `(before, after)` tape lengths, so the caller can assert
/// the encode recorded nothing rather than assume `no_grad` engaged.
fn baseline_encode(
    encoder: &mut SetFitMiniLm,
    texts: &[&str],
    batch_size: usize,
) -> Result<(Vec<Vec<f32>>, usize, usize), SetFitTrainError> {
    encoder.set_training(false);
    let before = autograd::graph_tape_len();
    let mut out: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
    let result = autograd::no_grad(|| -> Result<(), SetFitTrainError> {
        for window in texts.chunks(batch_size) {
            let embedded = encoder
                .encode_texts(window)
                .map_err(|e| SetFitTrainError::Encoder { reason: e.to_string() })?;
            // The SAME checked split `head_input` uses. Indexing `shape()[1]` here would
            // panic on a non-2-D return and would accept a `B` that disagrees with the
            // window, which is the one way `embeddings_before` and the selection could
            // silently stop being row-aligned.
            super::head_input::push_rows(&mut out, &embedded, window.len())?;
        }
        Ok(())
    });
    result?;
    let after = autograd::graph_tape_len();
    Ok((out, before, after))
}

// ===========================================================================================
// The per-batch body
// ===========================================================================================

/// Everything one batch needs, held together so `run_batch` stays under the complexity cap.
struct TuneCtx<'a> {
    encoder: &'a mut SetFitMiniLm,
    sampler: &'a PairSampler<'a>,
    texts: &'a [&'a str],
    adamw: AdamW,
    scheduler: WarmupLinearDecayLR,
    bridge: ParamBridge,
    registry_hash: [u8; 32],
    grad_clip_max_norm: f32,
    probes: TuningProbes,
    pair_hasher: Sha256,
    boundary_hasher: Sha256,
    boundaries: Vec<(u32, u64, u32)>,
    loss_trace: Vec<f32>,
    pre_clip_norms: Vec<f32>,
    steps: Vec<StepObservation>,
    grad_norm_max: Vec<f64>,
    grad_norm_sum: Vec<f64>,
    grad_norm_steps: Vec<u64>,
    global_step: u64,
}

/// The materialized inputs of one batch, in consumption order.
struct BatchInputs {
    texts_a: Vec<usize>,
    texts_b: Vec<usize>,
    targets: Vec<f32>,
}

/// One optimizer step, in EXACTLY this order — a, b, c, d, e, f, g, h, i, j, k, l, m.
///
/// A reordering here is a silent correctness change, so it is pinned BEHAVIOURALLY rather
/// than by reading the source: `tune_step_order_is_pinned` observes cleared gradients, a
/// zero tape and the scheduled learning rate at step 0, all recorded in band by this
/// function.
/// THE PAIR-PULL LOOP (N-03), lifted out of [`run_batch`] as one whole stage.
///
/// `absorb_batch_digests` is called from HERE, once per pair, in the same iteration that DRAWS
/// it — never from `run_tuning`'s iteration and never from a record returned after this loop
/// finished. Absorbing from a returned record would absorb a DESCRIPTION of the consumption
/// instead of the consumption, and 03-05's reversed-consumption negative is what notices.
///
/// Extracted in plan 03-10 T3 to bring `run_batch` under the cyclomatic ceiling of 10
/// (measured 13). It is lifted as ONE stage rather than sliced: `run_batch`'s stages (a)-(m)
/// are a PINNED ORDER that `tune_`'s order-pin test asserts on, and fragmenting a pinned
/// sequence makes the property harder to audit rather than easier — which is the opposite of
/// what the ceiling exists for.
fn pull_batch_pairs(
    ctx: &mut TuneCtx<'_>,
    epoch: u32,
    batch_index: u32,
    ordinals: &[u64],
) -> Result<BatchInputs, SetFitTrainError> {
    let mut pull_order: Vec<u64> = ordinals.to_vec();
    if ctx.probes.reverse_intra_batch_pull {
        pull_order.reverse();
    }
    let sampler = ctx.sampler;
    let mut inputs = BatchInputs {
        texts_a: Vec::with_capacity(pull_order.len()),
        texts_b: Vec::with_capacity(pull_order.len()),
        targets: Vec::with_capacity(pull_order.len()),
    };
    for (position, drawn) in pull_order.iter().map(|&o| (o, sampler.pair_at(o))).enumerate() {
        let (pair_ordinal, labeled) = drawn;
        let labeled = labeled?;
        let a = labeled.pair.lo().ordinal();
        let b = labeled.pair.hi().ordinal();
        absorb_batch_digests(
            &mut ctx.pair_hasher,
            epoch,
            batch_index,
            u32::try_from(position).unwrap_or(u32::MAX),
            pair_ordinal,
            a,
            b,
            labeled.target,
        );
        inputs.texts_a.push(a as usize);
        inputs.texts_b.push(b as usize);
        inputs.targets.push(labeled.target);
    }
    Ok(inputs)
}

/// Stage (b) of the pinned step order: zero every trainable gradient, THEN clear the tape.
///
/// The order is the pin — clearing the tape first would leave `Tensor::grad` populated from the
/// previous step with no graph entry to match, which is the state 03-05's skip-zero_grad
/// negative exists to make visible. `probes.skip_step_top_clear` is that negative's door.
fn clear_grads_then_tape(ctx: &mut TuneCtx<'_>) {
    if ctx.probes.skip_step_top_clear {
        return;
    }
    for (_, param) in ctx.encoder.trainable_parameters_mut() {
        param.zero_grad_();
    }
    autograd::clear_graph();
}

/// Batch open: absorb the boundary digest and record the readable triple, once per batch.
///
/// Both happen HERE, at the open, and the digest commits the `global_step` in force — which the
/// triple does not carry. Two runs with identical boundary LISTS and a different step alignment
/// therefore differ in the digest and agree in the list, which is why 03-10's composite hash
/// reports both.
fn open_batch(ctx: &mut TuneCtx<'_>, epoch: u32, batch_index: u32, ordinals: &[u64]) {
    let batch_start_ordinal = ordinals.first().copied().unwrap_or(0);
    let batch_len = u32::try_from(ordinals.len()).unwrap_or(u32::MAX);
    absorb_boundary_digest(
        &mut ctx.boundary_hasher,
        epoch,
        batch_index,
        ctx.global_step,
        batch_start_ordinal,
        batch_len,
    );
    ctx.boundaries.push((epoch, batch_start_ordinal, batch_len));
}

fn run_batch(
    ctx: &mut TuneCtx<'_>,
    epoch: u32,
    batch_index: u32,
    ordinals: &[u64],
) -> Result<(), SetFitTrainError> {
    // (a) set the SCHEDULED learning rate for the step about to be taken.
    let applied_lr = ctx.scheduler.get_lr();
    ctx.scheduler.apply(&mut ctx.adamw);

    // (b) clear gradients, then the tape.
    clear_grads_then_tape(ctx);

    let observation = observe_step_top(ctx, applied_lr);

    // batch open — the boundary digest is absorbed HERE, once per batch.
    open_batch(ctx, epoch, batch_index, ordinals);

    let inputs = pull_batch_pairs(ctx, epoch, batch_index, ordinals)?;

    // (c) branch A, (d) branch B — distinct forward ordinals, so the two siamese branches
    // draw INDEPENDENT dropout masks (D-15 as amended).
    //
    // `dropout_rng::forward_ordinal(step, branch)` rather than an open-coded `2*s` / `2*s+1`:
    // `BertSentenceEncoder::set_forward_ordinal`'s own documentation asks callers to use it,
    // and it is the half that carries the CHECKED arithmetic — the open-coded form wraps a
    // large step into a SMALL ordinal and silently replays an early step's masks, which is
    // the one failure of this scheme that looks perfectly reproducible.
    let ordinal_a = u64::from(branch_ordinal(ctx.global_step, 0)?);
    let ordinal_b = u64::from(branch_ordinal(ctx.global_step, 1)?);
    ctx.encoder.set_training(true);
    let za = forward_branch(ctx.encoder, ctx.texts, &inputs.texts_a, ordinal_a)?;
    let zb = forward_branch(ctx.encoder, ctx.texts, &inputs.texts_b, ordinal_b)?;

    // (e) loss, (f) backward.
    let loss = pair_cosine_mse(&za, &zb, &inputs.targets)
        .map_err(|e| SetFitTrainError::Encoder { reason: e.to_string() })?;
    let loss_value = loss.data()[0];

    // The in-band negative for the check immediately below. Production cannot reach this arm:
    // `TuningProbes::NONE` is the only value it can construct, and `POISON_FIRST_LOSS` is
    // `#[cfg(test)]`.
    #[cfg(test)]
    let loss_value =
        if ctx.probes.poison_first_loss && ctx.global_step == 0 { f32::NAN } else { loss_value };

    // The loss_trace_hash precondition, enforced where the value ENTERS rather than where it is
    // hashed (REVIEW CR-03 — it had no implementation at all before). Checked BEFORE `backward()`
    // so a diverged step does not also propagate NaN through every gradient and every optimizer
    // moment on its way out: by the time a hash noticed, the encoder would already be poisoned.
    //
    // Why this cannot be left to the digest: serde_json renders +inf, -inf and every NaN payload
    // as `null`, so all three produce IDENTICAL canonical bytes. A digest is not a finiteness
    // check, and the bundle it seals fails its own reload (`null` is not an f64).
    if !loss_value.is_finite() {
        return Err(SetFitTrainError::NonFiniteLoss {
            step: ctx.global_step,
            value_bits: f64::from(loss_value).to_bits(),
        });
    }

    loss.backward();

    // (g) per-name PRE-clip gradient norms, (h) registry-hash assertion, (i) clip, (j) step.
    let pre_clip = apply_optimizer(ctx)?;

    // (k) advance the scheduler only AFTER the step, (l) clear the tape.
    ctx.scheduler.step();
    if !ctx.probes.skip_post_step_clear {
        autograd::clear_graph();
    }

    // (m) record.
    ctx.loss_trace.push(loss_value);
    ctx.pre_clip_norms.push(pre_clip);
    ctx.steps.push(StepObservation {
        pre_clip_norm: pre_clip,
        forward_ordinals: (ordinal_a, ordinal_b),
        ..observation
    });
    ctx.global_step += 1;
    Ok(())
}

/// Read the step-top observations the order-pin test asserts on.
fn observe_step_top(ctx: &mut TuneCtx<'_>, applied_lr: f32) -> StepObservation {
    let mut tensor_grads = 0_usize;
    let mut graph_grads = 0_usize;
    for (_, param) in ctx.encoder.trainable_parameters_mut() {
        if param.grad().is_some() {
            tensor_grads += 1;
        }
        if autograd::get_grad(param.id()).is_some() {
            graph_grads += 1;
        }
    }
    StepObservation {
        global_step: ctx.global_step,
        tape_len_at_step_top: autograd::graph_tape_len(),
        tensor_grads_at_step_top: tensor_grads,
        graph_grads_at_step_top: graph_grads,
        applied_lr,
        forward_ordinals: (0, 0),
        pre_clip_norm: 0.0,
    }
}

/// D-15's `block` for one branch of `global_step`, through the ONE definition of `2*s + b`.
///
/// The arithmetic lives in `aprender::setfit::dropout_rng`, beside the counter lane it has to
/// fit; a second spelling here is how the trainer and the encoder end up disagreeing about
/// which stream a branch draws from.
fn branch_ordinal(global_step: u64, branch: u32) -> Result<u32, SetFitTrainError> {
    dropout_rng::forward_ordinal(global_step, branch)
        .map_err(|e| SetFitTrainError::Encoder { reason: e.to_string() })
}

/// One siamese branch: point every dropout site at `forward_ordinal`, then encode.
fn forward_branch(
    encoder: &mut SetFitMiniLm,
    texts: &[&str],
    indices: &[usize],
    forward_ordinal: u64,
) -> Result<CoreTensor, SetFitTrainError> {
    encoder
        .set_forward_ordinal(forward_ordinal)
        .map_err(|e| SetFitTrainError::Encoder { reason: e.to_string() })?;
    let batch: Vec<&str> = indices.iter().map(|&i| texts[i]).collect();
    encoder.encode_texts(&batch).map_err(|e| SetFitTrainError::Encoder { reason: e.to_string() })
}

/// Steps (g) through (j): gradient norms, registry assertion, clip, optimizer step.
///
/// Returns the PRE-clip global gradient norm.
fn apply_optimizer(ctx: &mut TuneCtx<'_>) -> Result<f32, SetFitTrainError> {
    let TuneCtx {
        encoder,
        bridge,
        adamw,
        registry_hash,
        grad_clip_max_norm,
        grad_norm_max,
        grad_norm_sum,
        grad_norm_steps,
        ..
    } = ctx;
    let mut params = encoder.trainable_parameters_mut();

    // (g) per-name PRE-clip gradient norms, through reduce.rs only.
    for (index, (_, param)) in params.iter().enumerate() {
        let Some(grad) = autograd::get_grad(param.id()) else {
            continue;
        };
        let norm = reduce::l2_norm_in_index_order(grad.data());
        if norm > grad_norm_max[index] {
            grad_norm_max[index] = norm;
        }
        grad_norm_sum[index] += norm;
        grad_norm_steps[index] += 1;
    }

    // (h) the registry must not have moved: AdamW's moment state is POSITIONAL, so a
    // reordered registry silently pairs moments with the wrong parameters (T-3-54).
    if registry_hash_of(&params) != *registry_hash {
        return Err(SetFitTrainError::ParameterRegistryMoved);
    }

    bridge.load(&params);
    let mut refs = bridge.refs();
    // (i) clip — returns the PRE-clip global norm.
    let pre_clip = clip_grad_norm_refs(&mut refs, *grad_clip_max_norm);
    // (j) step.
    adamw.step_refs(&mut refs);
    drop(refs);
    bridge.store(&mut params);
    Ok(pre_clip)
}

// ===========================================================================================
// The loop
// ===========================================================================================

/// Run stage one and record everything it did.
///
/// # Errors
///
/// [`SetFitTrainError`] — a non-CPU resolved device, a `max_length` that is not the pinned
/// value, a freeze policy addressing zero parameters, a parameter registry that moved
/// mid-run, or anything the encoder / pair sampler rejects.
pub(crate) fn run_tuning(
    encoder: SetFitMiniLm,
    dataset: &PreparedDataset<Canonical>,
    selection: &Selection,
    config: &ResolvedSetFitConfig,
) -> Result<TuneOutput, SetFitTrainError> {
    tune_with_probes(encoder, dataset, selection, config, TuningProbes::NONE)
}

/// [`run_tuning`] with a probe set only tests can build.
///
/// `#[cfg(test)]` and `pub(crate)`: the reproducibility accessors of 03-08 claim
/// to report EXECUTION rather than intent, and the only way to falsify that claim
/// is to run a pipeline whose execution differs while its configuration does not.
/// That needs the probe to reach `SetFitRun`, and a production door with the same
/// shape would be a door whose only use is to make a run lie about itself.
#[cfg(test)]
pub(crate) fn run_tuning_with_probes(
    encoder: SetFitMiniLm,
    dataset: &PreparedDataset<Canonical>,
    selection: &Selection,
    config: &ResolvedSetFitConfig,
    probes: TuningProbes,
) -> Result<TuneOutput, SetFitTrainError> {
    tune_with_probes(encoder, dataset, selection, config, probes)
}

/// The loop body, with a probe knob only tests can set to anything but [`TuningProbes::NONE`].
#[allow(clippy::too_many_lines)]
fn tune_with_probes(
    mut encoder: SetFitMiniLm,
    dataset: &PreparedDataset<Canonical>,
    selection: &Selection,
    config: &ResolvedSetFitConfig,
    probes: TuningProbes,
) -> Result<TuneOutput, SetFitTrainError> {
    let requested = config.requested();
    let flight = preflight(
        &mut encoder,
        config.device(),
        requested.max_length(),
        requested.freeze_policy(),
    )?;

    let texts = selection_texts(dataset, selection)?;
    let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let batch_size = requested.batch_size() as usize;

    // (6) the pre-tuning baseline encode.
    let (embeddings_before, tape_before, tape_after) =
        baseline_encode(&mut encoder, &text_refs, batch_size)?;

    let sampler = PairSampler::new(selection, requested.pair_config())?;
    let n_pairs = sampler.budget();
    let per_epoch = n_pairs.div_ceil(u64::from(requested.batch_size()));
    let total_steps = u64::from(requested.epochs()) * per_epoch;
    let warmup_steps = warmup_steps_from_ratio(total_steps, requested.warmup_ratio());

    let param_count = flight.trainable_count;
    let mut ctx = TuneCtx {
        bridge: ParamBridge::new(&encoder.trainable_parameters_mut()),
        encoder: &mut encoder,
        sampler: &sampler,
        texts: &text_refs,
        #[allow(clippy::cast_possible_truncation)]
        adamw: AdamW::new(
            requested.encoder_lr() as f32,
            ADAMW_BETA1,
            ADAMW_BETA2,
            ADAMW_EPSILON,
            ADAMW_WEIGHT_DECAY,
        ),
        #[allow(clippy::cast_possible_truncation)]
        scheduler: WarmupLinearDecayLR::new(
            requested.encoder_lr() as f32,
            SCHEDULE_LR_MIN,
            usize::try_from(warmup_steps).unwrap_or(usize::MAX),
            usize::try_from(total_steps).unwrap_or(usize::MAX),
        ),
        registry_hash: flight.registry_hash,
        grad_clip_max_norm: requested.grad_clip_max_norm(),
        probes,
        pair_hasher: Sha256::new(),
        boundary_hasher: Sha256::new(),
        boundaries: Vec::new(),
        loss_trace: Vec::new(),
        pre_clip_norms: Vec::new(),
        steps: Vec::new(),
        grad_norm_max: vec![0.0; param_count],
        grad_norm_sum: vec![0.0; param_count],
        grad_norm_steps: vec![0; param_count],
        global_step: 0,
    };

    for epoch in 0..requested.epochs() {
        let order = epoch_pair_order(requested.root_seed(), epoch, n_pairs);
        for (batch_index, chunk) in order.chunks(batch_size).enumerate() {
            run_batch(&mut ctx, epoch, u32::try_from(batch_index).unwrap_or(u32::MAX), chunk)?;
        }
    }

    let TuneCtx {
        pair_hasher,
        boundary_hasher,
        boundaries,
        loss_trace,
        pre_clip_norms,
        steps,
        grad_norm_max,
        grad_norm_sum,
        grad_norm_steps,
        global_step,
        ..
    } = ctx;

    let (embeddings_after, _, _) = baseline_encode(&mut encoder, &text_refs, batch_size)?;

    let per_name = measure_parameters(
        &mut encoder,
        &flight.snapshot,
        &grad_norm_max,
        &grad_norm_sum,
        &grad_norm_steps,
    );
    let (first_k_mean, last_k_mean, endpoint_k) = endpoint_means(&loss_trace);

    Ok(TuneOutput {
        encoder,
        initial_snapshot: flight.snapshot,
        snapshot_bytes: flight.snapshot_bytes,
        per_name,
        pre_clip_norms,
        loss_trace_hash: loss_trace_hash_of(&loss_trace),
        loss_trace,
        step_count: global_step,
        parameter_registry_hash: flight.registry_hash,
        consumed_pair_digest: pair_hasher.finalize().into(),
        batch_boundary_digest: boundary_hasher.finalize().into(),
        batch_boundaries: boundaries,
        max_length: requested.max_length(),
        first_k_mean,
        last_k_mean,
        endpoint_k,
        embeddings_before,
        embeddings_after,
        trainable_count: flight.trainable_count,
        frozen_count: flight.frozen_count,
        steps,
        baseline_encode_tape: (tape_before, tape_after),
    })
}

/// The selected rows' text, indexed by `SelectedId::ordinal()`.
///
/// `SelectedExample` carries no text (Phase 2 `select.rs`), so the strings come from the
/// dataset's train split, matched by row id.
///
/// # A missing id is an ERROR, never an empty string
///
/// An earlier form substituted `String::new()` for an id the dataset does not carry. That is
/// silent corruption of the one input the whole run is measured on: the encoder would embed
/// an empty sentence, the pair objective would train on it, the loss trace and both execution
/// digests would be perfectly reproducible, and nothing would announce that a row went
/// missing. `SetFitRun::prepare` already rejects such a selection at the door — which is
/// exactly why the fallback was unreachable and therefore untestable as a behaviour, and why
/// it must not be a fallback at all.
///
/// # Errors
///
/// [`SetFitTrainError::SelectionRowMissing`] naming the first id that does not resolve.
pub(crate) fn selection_texts(
    dataset: &PreparedDataset<Canonical>,
    selection: &Selection,
) -> Result<Vec<String>, SetFitTrainError> {
    let by_id: BTreeMap<&str, &str> =
        dataset.train().rows().iter().map(|row| (row.id.as_str(), row.input.as_str())).collect();
    selection
        .examples()
        .iter()
        .map(|example| {
            by_id.get(example.id.as_str()).map_or_else(
                || Err(SetFitTrainError::SelectionRowMissing { id: example.id.clone() }),
                |t| Ok((*t).to_string()),
            )
        })
        .collect()
}

/// Final per-parameter measurements: delta, support, and the support-restricted init norm.
fn measure_parameters(
    encoder: &mut SetFitMiniLm,
    snapshot: &BTreeMap<String, Vec<f32>>,
    grad_norm_max: &[f64],
    grad_norm_sum: &[f64],
    grad_norm_steps: &[u64],
) -> BTreeMap<String, ParamRecord> {
    let mut out = BTreeMap::new();
    for (index, (name, param)) in encoder.trainable_parameters_mut().into_iter().enumerate() {
        let Some(initial) = snapshot.get(&name) else {
            continue;
        };
        let final_data = param.data();
        let delta: Vec<f32> = initial.iter().zip(final_data.iter()).map(|(i, f)| f - i).collect();
        let support: Vec<f32> =
            initial.iter().zip(delta.iter()).filter(|(_, d)| **d != 0.0).map(|(i, _)| *i).collect();
        let steps = grad_norm_steps.get(index).copied().unwrap_or(0);
        #[allow(clippy::cast_precision_loss)]
        let mean = if steps == 0 {
            0.0
        } else {
            grad_norm_sum.get(index).copied().unwrap_or(0.0) / steps as f64
        };
        out.insert(
            name.clone(),
            ParamRecord {
                element_count: initial.len() as u64,
                init_norm: reduce::l2_norm_in_index_order(initial),
                init_norm_on_support: reduce::l2_norm_in_index_order(&support),
                delta_norm: reduce::l2_norm_in_index_order(&delta),
                delta_support_count: support.len() as u64,
                grad_norm_max: grad_norm_max.get(index).copied().unwrap_or(0.0),
                grad_norm_mean: mean,
                steps_observed: steps,
            },
        );
    }
    out
}

/// SHA-256 over the little-endian `f32` bit patterns, in step order.
///
/// The TRACE is hashed, never an `OptimizationResult` or anything carrying a `Duration`.
fn loss_trace_hash_of(trace: &[f32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for value in trace {
        hasher.update(value.to_bits().to_le_bytes());
    }
    hasher.finalize().into()
}

/// `k = min(5, steps / 2)`, then the two endpoint means. Computed, NOT judged.
fn endpoint_means(trace: &[f32]) -> (f64, f64, usize) {
    let k = ENDPOINT_K_CAP.min(trace.len() / 2);
    if k == 0 {
        return (0.0, 0.0, 0);
    }
    let first = reduce::mean_in_index_order(&trace[..k]);
    let last = reduce::mean_in_index_order(&trace[trace.len() - k..]);
    (first, last, k)
}

// ===========================================================================================
// The evidence gate (plan 03-06) — THE single validating function
// ===========================================================================================

/// The one parameter a rejection blames, with everything needed to act on it.
#[derive(Debug, Clone, PartialEq)]
pub struct FailedParameter {
    /// The HF dotted name.
    pub name: String,
    /// The class whose epsilon it was compared against.
    pub class: String,
    /// Its measured relative delta.
    pub relative_delta: f64,
    /// The contracted epsilon for that class.
    pub eps: f64,
}

/// Proof that a run's evidence PASSED the gate.
///
/// Constructible only by [`validate_evidence`], and the only thing
/// `SetFitRun<EncoderTuned>` will accept as its evidence. Non-constructibility elsewhere is
/// what makes "the encoder demonstrably moved" a type-level fact rather than a convention.
///
/// # It OWNS the complete record
///
/// Not a verdict flag beside a discarded table. Every run-level field survives here —
/// `loss_trace_hash`, both execution digests, `batch_boundary_list`,
/// `parameter_registry_hash`, `step_count`, the endpoint means and `k`, the embedding-delta
/// aggregates, `pre_clip_norm_max` and `calibration_regime_id` — because every accessor a
/// later plan exposes must resolve to a field on this chain. Anything dropped here becomes
/// something a downstream executor RECOMPUTES, which is exactly the false-green the in-band
/// digests exist to remove.
#[derive(Debug, Clone, PartialEq)]
pub struct PassedEvidence {
    table: UpdateEvidence,
    summary: EvidenceSummary,
}

// Accessors for plan 03-08's persistence seam: it writes the table and the bound summary
// into the bundle. Drop this allow once that plan calls them.
#[allow(dead_code)]
impl PassedEvidence {
    /// The complete per-parameter table.
    #[must_use]
    pub(crate) fn table(&self) -> &UpdateEvidence {
        &self.table
    }

    /// The bound summary.
    #[must_use]
    pub(crate) fn summary(&self) -> &EvidenceSummary {
        &self.summary
    }
}

/// Validate a run's evidence against the frozen thresholds. THE gate.
///
/// Exactly one such function exists in this crate, and every future door calls it. A second
/// validating function would be a second policy, and the one nobody remembered to update is
/// the one a bad run would come through.
///
/// # Order is load-bearing
///
/// 1. The calibration-regime check, FIRST, before any number is compared. Thresholds measured
///    on one architecture are not evidence about another, so an unrecognised regime is refused
///    rather than judged leniently.
/// 2. The trainable set must be non-empty (SAFE-03 automatic under D-09).
/// 3. The GATED set must be non-empty. A run whose only trainable parameters are
///    analytically gradient-free has nothing that can testify, and passing it would be a
///    vacuous verdict rather than a lenient one.
/// 4. Per-parameter predicates: finite gradient norms, strict movement, and the class epsilon.
/// 5. The run-level embedding-delta floor, last, so a failure names a parameter first.
///
/// # Errors
///
/// [`SetFitTrainError::UncalibratedRegime`], [`SetFitTrainError::NoTrainableParameters`],
/// [`SetFitTrainError::NoTestifyingParameters`] or [`SetFitTrainError::EvidenceRejected`].
#[cfg_attr(
    feature = "setfit",
    provable_contracts_macros::contract("setfit-train-lifecycle-v1", equation = "evidence_gate")
)]
pub(crate) fn validate_evidence(
    evidence: &UpdateEvidence,
    thresholds: &Thresholds,
    trainable_count: usize,
    frozen_count: usize,
) -> Result<PassedEvidence, SetFitTrainError> {
    // (1) FAIL CLOSED OUTSIDE THE CALIBRATED REGIME — before any comparison.
    //
    // The refusal and the table are ONE lookup, deliberately. A membership check followed by a
    // separate table read is two decisions that can disagree: the second one has to pick a
    // table, and with more than one calibration in the constant the only thing it can pick
    // without being told the regime is "the first", which is how fixture-scale epsilons would
    // come to judge a production encoder. Here every threshold below is read out of
    // `regime` — the table measured at THESE coordinates — so there is no reachable state in
    // which a number is compared against a run the number was not measured on.
    let Some(regime) = thresholds.table_for(&evidence.calibration_regime_id) else {
        return Err(SetFitTrainError::UncalibratedRegime {
            observed: evidence.calibration_regime_id.clone(),
            calibrated: thresholds.calibrated_regimes().iter().map(|s| (*s).to_string()).collect(),
        });
    };

    let mut summary = EvidenceSummary::of(evidence, trainable_count, frozen_count)
        .map_err(|e| SetFitTrainError::Evidence { reason: e.to_string() })?;
    // Pessimistic until proven otherwise: every early return below leaves `Fail` stamped, so
    // a path that forgets to set it cannot emit a summary that claims to have passed.
    summary.verdict = Verdict::Fail;

    // (2) An empty trainable set is unpassable. SAFE-03 is structural, not a caller's habit.
    if trainable_count == 0 || evidence.rows.is_empty() {
        return Err(SetFitTrainError::NoTrainableParameters { trainable_count });
    }

    // (3) An all-ungated trainable set is unpassable too. Without this, freezing everything
    // EXCEPT the key biases would leave the gate with nothing to check and yield a pass.
    let gated: Vec<&EvidenceRow> =
        evidence.rows.values().filter(|r| regime.of(r.class).gated).collect();
    if gated.is_empty() {
        return Err(SetFitTrainError::NoTestifyingParameters {
            trainable_count,
            ungated_count: evidence.rows.len(),
        });
    }

    // (4) Per-parameter predicates.
    let worst = worst_failing_gated_parameter(&gated, regime, trainable_count, &evidence.rows)?;

    if let Some(offender) = worst {
        return Err(SetFitTrainError::EvidenceRejected {
            summary: Box::new(summary),
            table: Box::new(evidence.clone()),
            worst: offender,
        });
    }

    // (5) The run-level sparse aggregate, LAST. Non-finite fails closed for the same reason
    // the per-parameter guard above does: `+inf <= floor` is false, so a diverged embedding
    // class would otherwise clear the run-level floor by having blown up.
    let floor = regime.embedding_delta_floor();
    if !evidence.embedding_delta_median.is_finite() || evidence.embedding_delta_median <= floor {
        return Err(SetFitTrainError::EvidenceRejected {
            summary: Box::new(summary),
            table: Box::new(evidence.clone()),
            worst: FailedParameter {
                name: "<embedding class aggregate>".to_string(),
                class: ParameterClass::Embedding.tag().to_string(),
                relative_delta: evidence.embedding_delta_median,
                eps: floor,
            },
        });
    }

    summary.verdict = Verdict::Pass;
    Ok(PassedEvidence { table: evidence.clone(), summary })
}

/// Rung (4) of the evidence gate: the per-parameter predicates over the GATED rows.
///
/// Returns `Ok(None)` when every gated parameter passes, `Ok(Some(worst))` when at least one
/// fails, and `Err` only for the structural refusal a gated class with no epsilon represents.
///
/// # The worst offender is the SMALLEST margin, not the first name
///
/// Blame is assigned by `relative_delta / eps`, so it lands on the parameter that is furthest
/// from passing rather than on whichever name happens to sort first. `<=` in the comparison
/// keeps the FIRST parameter at a tied margin, which makes the choice deterministic over a
/// `HashMap`'s iteration order.
///
/// Extracted in plan 03-10 T3 to clear the cyclomatic ceiling of 10 (`validate_evidence`
/// measured 20). Behaviour-preserving: the loop, the finiteness conjunction and the tie rule
/// are moved verbatim.
fn worst_failing_gated_parameter(
    gated: &[&EvidenceRow],
    regime: &RegimeThresholds,
    trainable_count: usize,
    all_rows: &BTreeMap<String, EvidenceRow>,
) -> Result<Option<FailedParameter>, SetFitTrainError> {
    let mut worst: Option<(f64, FailedParameter)> = None;
    for row in gated {
        let entry = regime.of(row.class);
        let Some(eps) = entry.eps else {
            // Unreachable while `gated` and `eps.is_some()` agree, which thresholds.rs
            // asserts. Fail closed rather than skip: a gated class without a threshold must
            // never be silently waved through.
            //
            // `ungated_count` is the rows that are NOT gated, not the row total: `gated` is
            // non-empty on this path, so reporting `rows.len()` would render the message
            // "all N of the N trainable parameters are ungated", which is false and points a
            // diagnosis at the wrong mechanism.
            return Err(SetFitTrainError::NoTestifyingParameters {
                trainable_count,
                ungated_count: all_rows.len().saturating_sub(gated.len()),
            });
        };

        let Some(failure) = gated_row_failure(row, eps) else { continue };
        match &worst {
            Some((best_margin, _)) if *best_margin <= failure.0 => {}
            _ => worst = Some(failure),
        }
    }
    Ok(worst.map(|(_, offender)| offender))
}

/// One gated row's verdict: `None` when it passes, `Some((margin, failure))` when it does not.
///
/// # Every measured number the verdict rests on must be FINITE, not only the gradient norms
///
/// `relative_delta` is the value COMPARED against the epsilon, and `+inf > eps` is TRUE — so a
/// parameter that diverged to infinity while its recorded gradient norms stayed finite would
/// have passed the SetFit-identity gate outright. (The NaN side already failed closed, because
/// `moved` is `delta_norm > 0.0` and every NaN comparison is false; the infinity side did not.)
///
/// The margin is `relative_delta / eps`, or `+inf` at `eps == 0.0` so a zero-epsilon class
/// never wins the "worst offender" comparison by dividing by zero.
///
/// Extracted from the loop in plan 03-10 T3. The first split moved `validate_evidence` from
/// cyclomatic 20 to 10 but left the extracted helper at 12 — the complexity had MOVED rather
/// than dissolved, which is the failure mode of decomposing by cut-and-paste. This second cut
/// takes the per-row predicate out of the accumulation, which is what actually reduces both.
fn gated_row_failure(row: &EvidenceRow, eps: f64) -> Option<(f64, FailedParameter)> {
    let finite = row.grad_norm_max.is_finite()
        && row.grad_norm_mean.is_finite()
        && row.init_norm.is_finite()
        && row.delta_norm.is_finite()
        && row.relative_delta.is_finite();
    if finite && row.moved && row.relative_delta > eps {
        return None;
    }
    let margin = if eps > 0.0 { row.relative_delta / eps } else { f64::INFINITY };
    Some((
        margin,
        FailedParameter {
            name: row.name.clone(),
            class: row.class.tag().to_string(),
            relative_delta: row.relative_delta,
            eps,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::setfit::test_fixtures as fx;

    /// Run stage one on a fixture cell.
    fn tune(variant: fx::CalibrationVariant, probes: TuningProbes) -> TuneOutput {
        let (encoder, dataset, selection, config) = fx::prepared_run(variant, None).into_parts();
        tune_with_probes(encoder, &dataset, &selection, &config, probes)
            .expect("the fixture run must tune")
    }

    /// A diverged step is a TYPED FAILURE, not a hashed `null` (REVIEW CR-03).
    ///
    /// The contract's `loss_trace_hash` equation has carried the precondition "every loss value
    /// is finite; a NaN or infinite step is a typed failure before hashing" since 03-06, with no
    /// implementation: `run_batch` pushed the value unchecked. Because `serde_json` renders every
    /// non-finite `f64` as `null`, the consequence was not a loud failure but a QUIET one — the
    /// evidence digest could not tell `+inf` from `-inf` from `NaN`, and the bundle it sealed
    /// could not be reloaded at all.
    ///
    /// Measured, not assumed: `serde_json::to_string` on each of the three yields `null`, and
    /// `from_str::<f64>("null")` is `Err("invalid type: null, expected f64")`.
    #[test]
    fn tune_refuses_a_non_finite_loss_before_hashing() {
        let (encoder, dataset, selection, config) =
            fx::prepared_run(fx::default_variant(), None).into_parts();
        let err = tune_with_probes(
            encoder,
            &dataset,
            &selection,
            &config,
            TuningProbes::POISON_FIRST_LOSS,
        )
        .expect_err("a NaN loss at step 0 must fail the run, not be recorded");

        match err {
            SetFitTrainError::NonFiniteLoss { step, value_bits } => {
                assert_eq!(step, 0, "the poisoned step is step 0");
                // Compared through the BITS. `f64::NAN != f64::NAN`, so an assertion on the
                // value itself would pass for the wrong reason — or never pass at all.
                assert!(
                    f64::from_bits(value_bits).is_nan(),
                    "value_bits must carry the offending NaN, got {:?}",
                    f64::from_bits(value_bits),
                );
            }
            other => panic!("expected NonFiniteLoss, got {other:?}"),
        }
    }

    /// The same run WITHOUT the probe completes — so the test above fails for the injected
    /// reason, not because the fixture cell was broken.
    #[test]
    fn tune_control_for_the_non_finite_loss_probe_is_green() {
        let out = tune(fx::default_variant(), TuningProbes::NONE);
        assert_eq!(out.step_count, 3, "the control must still be the three-step run");
        assert!(
            out.loss_trace.iter().all(|v| v.is_finite()),
            "every recorded loss in the control run is finite",
        );
    }

    /// THE order pin — behavioural, not a source assertion.
    ///
    /// Reading the source would prove the statements appear in an order; these three
    /// assertions prove the EFFECTS of that order were observed by the loop itself.
    #[test]
    fn tune_step_order_is_pinned() {
        let variant = fx::default_variant();
        let out = tune(variant, TuningProbes::NONE);
        assert_eq!(out.step_count, 3, "the default cell is a three-step run");
        assert_eq!(out.steps.len(), 3);

        for step in &out.steps {
            // (i) gradients cleared at the top of every step — on BOTH sides.
            assert_eq!(
                step.tensor_grads_at_step_top, 0,
                "step {}: a trainable parameter still carried a tensor-side gradient",
                step.global_step,
            );
            assert_eq!(
                step.graph_grads_at_step_top, 0,
                "step {}: a trainable parameter still carried a GRAPH-side gradient — this is \
                 the side that matters, see the module doc",
                step.global_step,
            );
            // (iii) the tape was cleared.
            assert_eq!(
                step.tape_len_at_step_top, 0,
                "step {}: the tape was not cleared",
                step.global_step,
            );
        }

        // (ii) step 0 runs at the SCHEDULED rate, which with warmup_steps > 0 is exactly 0.0
        // and is NOT config.encoder_lr. HF's LambdaLR lambda(0) is 0; this is reference
        // fidelity, not an off-by-one.
        let config = fx::config_for(variant, None);
        #[allow(clippy::cast_possible_truncation)]
        let peak = config.encoder_lr() as f32;
        assert!(peak > 0.0, "the control value must be non-zero");
        assert_eq!(out.steps[0].applied_lr, 0.0, "step 0 must run at the warmup rate",);
        assert_ne!(out.steps[0].applied_lr, peak, "step 0 must NOT run at the peak rate",);
        // And the rate genuinely moves afterwards, so the assertion above is not satisfied by
        // a scheduler that returns 0.0 forever.
        assert!(out.steps.iter().any(|s| s.applied_lr > 0.0), "the schedule must leave warmup",);
    }

    /// Distinct forward ordinals per branch: `2*s` and `2*s+1` (D-15's `block`).
    #[test]
    fn tune_the_two_branches_use_distinct_forward_ordinals() {
        let out = tune(fx::default_variant(), TuningProbes::NONE);
        for step in &out.steps {
            assert_eq!(step.forward_ordinals.0, 2 * step.global_step);
            assert_eq!(step.forward_ordinals.1, 2 * step.global_step + 1);
            assert_ne!(step.forward_ordinals.0, step.forward_ordinals.1);
        }
    }

    /// The gradients reach the parameters through the GRAPH, not through `param.grad()`.
    ///
    /// Measured fact 2 in the module doc, pinned. If this ever inverts, the loop's
    /// `get_grad(id)` reads must invert with it or the optimizer will step on nothing.
    #[test]
    fn tune_gradients_reach_the_parameters_through_the_graph() {
        let (mut encoder, dataset, selection, config) =
            fx::prepared_run(fx::default_variant(), None).into_parts();
        let texts = selection_texts(&dataset, &selection).expect("every selected id resolves");
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();

        autograd::clear_graph();
        encoder.set_training(true);
        let za = forward_branch(&mut encoder, &refs, &[0, 1], 0).expect("branch a");
        let zb = forward_branch(&mut encoder, &refs, &[2, 3], 1).expect("branch b");
        let loss = pair_cosine_mse(&za, &zb, &[1.0, 0.0]).expect("loss");
        loss.backward();

        let mut tensor_side = 0;
        let mut graph_side = 0;
        for (_, param) in encoder.trainable_parameters_mut() {
            if param.grad().is_some() {
                tensor_side += 1;
            }
            if autograd::get_grad(param.id()).is_some() {
                graph_side += 1;
            }
        }
        assert_eq!(
            tensor_side, 0,
            "aprender-core writes gradients into the graph's registry copies, not onto the \
             model's tensors",
        );
        assert!(
            graph_side > 0,
            "at least one trainable parameter must have a graph-side gradient after backward",
        );
        assert_eq!(config.requested().max_length(), super::super::config::pinned_max_length(),);
        autograd::clear_graph();
    }

    /// Step (b) is LOAD-BEARING, and the measured reason is not the one the plan predicted.
    ///
    /// The plan expected step N's gradient to be the sum of steps 0..N without a
    /// `zero_grad_`. Measured: it is not, and the mechanism is that `register_tensor`
    /// REPLACES a leaf's entry on every forward (`autograd/graph.rs:67`) while a previous
    /// step's sub-tape has no seeded output gradient, so `backward` skips it entirely. What
    /// step (b) actually guarantees is the invariant "the tape is empty when THIS step's
    /// forward begins", and that invariant does not follow from step (l) alone, because
    /// step (l) has never run when step 0 starts.
    ///
    /// The fixture makes that reachable rather than hypothetical: loading the MiniLM slice
    /// leaves 24 operations on the tape before the loop ever begins (see
    /// `tune_baseline_encode_records_no_operations`). With step (b) removed, step 0's
    /// backward walks those 24 foreign operations.
    #[test]
    fn tune_step_top_clear_is_load_bearing() {
        let variant = fx::default_variant();
        let correct = tune(variant, TuningProbes::NONE);
        let skipped = tune(variant, TuningProbes::SKIP_STEP_TOP_CLEAR);

        assert!(
            correct.steps.iter().all(|s| s.tape_len_at_step_top == 0),
            "the correct run must start every step with an empty tape",
        );
        assert!(
            skipped.steps[0].tape_len_at_step_top > 0,
            "with step (b) removed, step 0 must inherit the loader's tape; observed {:?}",
            skipped.steps.iter().map(|s| s.tape_len_at_step_top).collect::<Vec<_>>(),
        );
        // Non-vacuity: the inherited tape is exactly what the baseline encode reported.
        assert_eq!(
            skipped.steps[0].tape_len_at_step_top, skipped.baseline_encode_tape.1,
            "the inherited tape must be the one the pre-loop encode observed",
        );
    }

    /// The clearing AS A MECHANISM prevents unbounded tape growth and stale gradients.
    ///
    /// Removing only one of the two clears is not observable past step 0 — the other still
    /// leaves an empty tape — so this probe removes both. Every backward re-walks every
    /// earlier step's operations, which is quadratic in the step count with monotone memory
    /// (T-3-55), and the previous step's graph-side gradient is still present when the next
    /// step begins.
    #[test]
    fn tune_removing_the_clearing_grows_the_tape_and_strands_gradients() {
        let variant = fx::default_variant();
        let correct = tune(variant, TuningProbes::NONE);
        let unclear = tune(variant, TuningProbes::SKIP_ALL_GRAPH_CLEARS);

        let lens: Vec<usize> = unclear.steps.iter().map(|s| s.tape_len_at_step_top).collect();
        assert!(
            lens.windows(2).all(|w| w[1] > w[0]),
            "the tape must grow monotonically once nothing clears it: {lens:?}",
        );
        assert!(
            unclear.steps.iter().skip(1).any(|s| s.graph_grads_at_step_top > 0),
            "a stale graph-side gradient must be visible at a later step's top",
        );
        assert!(
            correct.steps.iter().all(|s| s.graph_grads_at_step_top == 0),
            "the correct run must see none",
        );
    }

    /// Two in-process runs from identical inputs agree bitwise — the first TRN-06 signal.
    #[test]
    fn tune_two_runs_are_bitwise_identical() {
        let variant = fx::default_variant();
        let a = tune(variant, TuningProbes::NONE);
        let b = tune(variant, TuningProbes::NONE);

        assert_eq!(a.loss_trace, b.loss_trace, "loss trace");
        assert_eq!(a.loss_trace_hash, b.loss_trace_hash, "loss trace hash");
        assert_eq!(a.consumed_pair_digest, b.consumed_pair_digest, "pair digest");
        assert_eq!(a.batch_boundary_digest, b.batch_boundary_digest, "boundary digest",);
        assert_eq!(a.batch_boundaries, b.batch_boundaries, "boundary list");
        assert_eq!(a.step_count, b.step_count, "step count");
        assert_eq!(a.per_name, b.per_name, "per-parameter measurements");
        assert_eq!(a.embeddings_after, b.embeddings_after, "final embeddings");

        // Non-vacuity: the trace must actually contain something to compare.
        assert_eq!(a.loss_trace.len(), 3);
        assert!(a.loss_trace.iter().all(|v| v.is_finite()), "finite losses");
    }

    /// The digest is RECORDED, not RECOMPUTED.
    ///
    /// The probe reverses only the INTRA-BATCH pull order. Batch structure, batch count,
    /// batch start ordinals and batch lengths are all untouched, so a digest reconstructed
    /// from configuration would be identical for both runs — and this test would be green
    /// under an implementation that certifies intent rather than execution.
    #[test]
    fn tune_digest_is_recorded_not_recomputed() {
        let variant = fx::default_variant();
        let forward = tune(variant, TuningProbes::NONE);
        let reversed = tune(variant, TuningProbes::REVERSE_INTRA_BATCH_PULL);

        assert_eq!(
            forward.batch_boundaries, reversed.batch_boundaries,
            "the probe must leave the batch STRUCTURE alone, or the test is too coarse",
        );
        assert_eq!(
            forward.batch_boundary_digest, reversed.batch_boundary_digest,
            "the boundary digest must be unchanged by an intra-batch reordering",
        );
        assert_ne!(
            forward.consumed_pair_digest, reversed.consumed_pair_digest,
            "a pair digest that survives a real intra-batch reordering is a RECOMPUTED digest",
        );
    }

    /// Step count is exactly `epochs * ceil(n_pairs / batch_size)`, on both cells.
    #[test]
    fn tune_step_count_matches_the_closed_form() {
        for variant in fx::calibration_variants() {
            let out = tune(variant, TuningProbes::NONE);
            assert_eq!(
                out.step_count,
                variant.total_steps(),
                "{}: epochs * ceil(budget / batch_size)",
                variant.label,
            );
            assert_eq!(out.loss_trace.len() as u64, out.step_count);
            assert_eq!(out.batch_boundaries.len() as u64, out.step_count);
        }
    }

    /// The loss trace is finite at every step, on every cell.
    #[test]
    fn tune_loss_trace_is_finite() {
        for variant in fx::calibration_variants() {
            let out = tune(variant, TuningProbes::NONE);
            assert!(!out.loss_trace.is_empty(), "{}: empty trace", variant.label);
            for (step, value) in out.loss_trace.iter().enumerate() {
                assert!(
                    value.is_finite(),
                    "{} step {step}: loss {value} is not finite",
                    variant.label,
                );
            }
        }
    }

    /// The pre-tuning baseline encode records NOTHING — a DELTA, not an absolute.
    ///
    /// The distinction cost a red test and is worth keeping. `SetFitMiniLm::from_slice_fixture`
    /// leaves 24 operations on the thread-local tape before any of this code runs (measured;
    /// pre-existing Phase 1 loader behaviour, logged as a deferred item), so an absolute
    /// `== 0` assertion here would be a test of the LOADER wearing this test's name. The
    /// property that belongs to this function is that the `no_grad` encode adds nothing, and
    /// the non-zero entry value is what makes that assertion non-vacuous. The absolute claim
    /// about `no_grad` is pinned where it belongs, in
    /// `aprender::autograd::tests::autograd_graph_tape_len_stays_zero_under_no_grad`, which
    /// carries its own recording control.
    #[test]
    fn tune_baseline_encode_records_no_operations() {
        let out = tune(fx::default_variant(), TuningProbes::NONE);
        assert_eq!(
            out.baseline_encode_tape.0, out.baseline_encode_tape.1,
            "the no_grad baseline encode must not grow the tape",
        );
        assert!(
            out.baseline_encode_tape.0 > 0,
            "the loader is expected to leave a tape; if it stops doing so this test becomes \
             vacuous and the assertion above stops proving anything",
        );
        assert_eq!(out.embeddings_before.len(), out.embeddings_after.len());
        assert_eq!(out.embeddings_before.len(), 24, "8 shots x 3 classes");
    }

    /// Tuning MOVED the encoder — otherwise every measurement below is of a no-op.
    #[test]
    fn tune_actually_changes_the_encoder() {
        let out = tune(fx::default_variant(), TuningProbes::NONE);
        assert_ne!(
            out.embeddings_before, out.embeddings_after,
            "the tuned encoder must produce different embeddings",
        );
        let moved = out.per_name.values().filter(|r| r.delta_norm > 0.0).count();
        assert!(moved > 0, "at least one parameter must have a non-zero delta norm",);
        assert!(
            out.per_name.values().any(|r| r.grad_norm_max > 0.0),
            "at least one parameter must have had a gradient",
        );
        assert_eq!(out.trainable_count, 37);
        assert_eq!(out.frozen_count, 0, "the D-20 default freezes nothing");
        assert_eq!(out.initial_snapshot.len(), 37);
        assert!(out.snapshot_bytes > 0);

        // The tuned encoder and the consumed max_length ride out with the record, because
        // 03-06 mints `SetFitRun<EncoderTuned>` from the first and 03-08's bundle records the
        // second. Asserted here so neither is carried untested until then.
        assert_eq!(out.encoder.num_layers(), 2);
        assert_eq!(
            out.max_length,
            super::super::config::pinned_max_length(),
            "the consumed max_length must be the pinned one",
        );
    }

    /// The recorded snapshot accounting, and the full-encoder projection it implies.
    ///
    /// # Peak tuning memory is FIVE copies of the trainable parameters, not four
    ///
    /// The live parameters, the initial snapshot, [`ParamBridge`]'s mirror, and AdamW's two
    /// moment buffers. The bridge is the copy an earlier count missed: it exists because the
    /// two crates have separate `Tensor` types, so it is a real cost of reusing the reference
    /// optimizer rather than reimplementing it, and it should be named rather than absorbed
    /// into a round number.
    ///
    /// For `all-MiniLM-L6-v2` at ~22.7 M trainable parameters that is ~90.8 MB per copy and
    /// ~454 MB at peak. Stated here so Phase 5 does not discover it on a memory-constrained
    /// host.
    #[test]
    fn tune_snapshot_memory_accounting_is_recorded() {
        let out = tune(fx::default_variant(), TuningProbes::NONE);
        let elements: usize = out.initial_snapshot.values().map(Vec::len).sum();
        assert_eq!(
            out.snapshot_bytes,
            elements * 4,
            "the recorded byte count must be the f32 element count",
        );
        // The exact figure for the pinned slice, so the SUMMARY's number is asserted rather
        // than transcribed: 3 embedding tables (6208 + 4096 + 128) + embeddings LayerNorm
        // (128) + 2 layers x 49984.
        assert_eq!(elements, 110_528, "the pinned slice's trainable element count");
        assert_eq!(out.snapshot_bytes, 442_112, "and its snapshot in bytes");
    }

    /// A non-CPU resolved device is a typed error naming the device.
    #[test]
    fn tune_rejects_a_non_cpu_resolved_device() {
        let pinned = super::super::config::pinned_max_length();
        let mut encoder = fx::slice_encoder(fx::FIXTURE_SEED);
        let err = preflight(&mut encoder, Device::Cuda { index: 0 }, pinned, &[])
            .expect_err("a CUDA-resolved device must be refused");
        match err {
            SetFitTrainError::UnsupportedDeviceForPhase3 { ref resolved } => {
                assert_eq!(resolved, "cuda:0");
                assert!(err.to_string().contains("cuda:0"));
            }
            other => panic!("expected UnsupportedDeviceForPhase3, got {other:?}"),
        }
        // Control: the same call with a CPU device gets past the device rung.
        assert!(preflight(&mut encoder, Device::Cpu, pinned, &[]).is_ok());
    }

    /// A `max_length` other than the pinned value is a typed error naming BOTH values.
    #[test]
    fn tune_rejects_a_max_length_that_is_not_the_pinned_value() {
        let pinned = super::super::config::pinned_max_length();
        let mut encoder = fx::slice_encoder(fx::FIXTURE_SEED);
        let err = preflight(&mut encoder, Device::Cpu, pinned + 1, &[])
            .expect_err("a non-pinned max_length must be refused");
        match err {
            SetFitTrainError::MaxLengthNotConsumable { requested, pinned: p } => {
                assert_eq!(p, pinned);
                assert_eq!(requested, pinned + 1);
                let rendered = err.to_string();
                assert!(rendered.contains(&requested.to_string()), "{rendered}");
                assert!(rendered.contains(&p.to_string()), "{rendered}");
            }
            other => panic!("expected MaxLengthNotConsumable, got {other:?}"),
        }
        // Control, so the rejection is not simply "preflight always fails".
        assert!(preflight(&mut encoder, Device::Cpu, pinned, &[]).is_ok());
    }

    /// A freeze policy addressing zero parameters fails BEFORE the snapshot (T-3-19).
    #[test]
    fn tune_zero_match_freeze_policy_fails_before_any_evidence_is_captured() {
        let pinned = super::super::config::pinned_max_length();
        let mut encoder = fx::slice_encoder(fx::FIXTURE_SEED);
        // The slice has 2 layers, so layer 5 addresses nothing.
        let err = preflight(&mut encoder, Device::Cpu, pinned, &[FreezeGroup::LayerAttention(5)])
            .expect_err("a freeze group naming a layer that does not exist must be refused");
        assert!(matches!(err, SetFitTrainError::Encoder { .. }), "got {err:?}",);
    }

    /// The endpoint window is `min(5, steps / 2)`, and is COMPUTED rather than judged.
    #[test]
    fn tune_endpoint_window_follows_the_closed_form() {
        assert_eq!(endpoint_means(&[]), (0.0, 0.0, 0));
        assert_eq!(endpoint_means(&[1.0]), (0.0, 0.0, 0));
        let (first, last, k) = endpoint_means(&[1.0, 2.0, 3.0]);
        assert_eq!(k, 1);
        assert!((first - 1.0).abs() < f64::EPSILON);
        assert!((last - 3.0).abs() < f64::EPSILON);
        let (_, _, k) = endpoint_means(&[0.0; 20]);
        assert_eq!(k, ENDPOINT_K_CAP);
    }

    /// The loss-trace hash is order-sensitive and length-prefix-free by construction, so a
    /// permutation must move it.
    #[test]
    fn tune_loss_trace_hash_depends_on_order() {
        let a = loss_trace_hash_of(&[0.25, 0.5, 0.75]);
        let b = loss_trace_hash_of(&[0.75, 0.5, 0.25]);
        assert_ne!(a, b);
        assert_eq!(a, loss_trace_hash_of(&[0.25, 0.5, 0.75]));
    }

    /// The registry hash is length-prefixed, so a name-boundary shift cannot collide.
    #[test]
    fn tune_registry_hash_is_length_prefixed() {
        let mut ab = CoreTensor::from_slice(&[1.0]);
        let mut c = CoreTensor::from_slice(&[1.0]);
        let split_one: Vec<(String, &mut CoreTensor)> =
            vec![("ab".to_string(), &mut ab), ("c".to_string(), &mut c)];
        let one = registry_hash_of(&split_one);
        drop(split_one);
        let split_two: Vec<(String, &mut CoreTensor)> =
            vec![("a".to_string(), &mut ab), ("bc".to_string(), &mut c)];
        let two = registry_hash_of(&split_two);
        assert_ne!(one, two, "concatenated names must not collide");
    }
}
