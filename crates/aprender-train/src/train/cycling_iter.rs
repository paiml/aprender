//! Shard-stream exhaustion primitives for MODEL-2 pretrain
//! (GATE-TRAIN-EXHAUST / INV-TRAIN-011, task #141).
//!
//! Contract: `contracts/training-loop-pretrain-v1.yaml` v1.5.0.
//!
//! When an `LMBatch` iterator is exhausted mid-run the training loop
//! MUST do EXACTLY one of:
//!   (a) iterator-layer cycle with a single INFO log at the first
//!       cycle boundary, OR
//!   (b) hard-fail by panicking with a `GATE-TRAIN-EXHAUST` message
//!       that the driver surfaces as a nonzero exit.
//!
//! Returning a constant placeholder tuple `(1.0, 1.0)` from
//! `StepFn::step` is FORBIDDEN (this was the task #141 defect: it
//! silently converted a 45×-under-Chinchilla corpus into fake
//! convergence evidence — epochs 2..5 of the 10k MODEL-2 run emitted
//! `train_loss=1.0`, `train_ppl=e`, `grad_norm=1.0`, `wall<1s`
//! fingerprints).

use crate::train::transformer_trainer::LMBatch;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Panic message prefix every `StepFn::step` impl must use when the
/// underlying shard iterator returns `None` without an operator-opted
/// cycle wrapper. Downstream tests assert this prefix literally.
pub const EXHAUST_PANIC_PREFIX: &str = "GATE-TRAIN-EXHAUST";

/// Pull the next `LMBatch` or panic with a diagnostic message citing
/// GATE-TRAIN-EXHAUST. This is the hard-fail path (option b) of
/// INV-TRAIN-011. Callers that want cycling instead wrap the iterator
/// in [`CyclingBatchIter`].
///
/// # Panics
///
/// Panics with a message beginning `GATE-TRAIN-EXHAUST:` when
/// `iter.next()` returns `None`. The panic is the intended contract
/// signal — the driver either catches it via `catch_unwind` or the
/// process exits nonzero, both of which satisfy option (b).
pub fn next_batch_or_panic<I: Iterator<Item = LMBatch> + ?Sized>(iter: &mut I) -> LMBatch {
    if let Some(batch) = iter.next() {
        return batch;
    }
    panic!(
        "{}: shard stream exhausted mid-run. Wrap the iterator in \
         CyclingBatchIter to opt into cycling (INV-TRAIN-011 path a) \
         or enlarge the corpus so total_tokens covers the planned \
         (epochs × steps × batch_tokens) budget. See \
         contracts/training-loop-pretrain-v1.yaml v1.5.0 \
         GATE-TRAIN-EXHAUST and \
         contracts/pretraining-corpus-v1.yaml v2.0.0 \
         FALSIFY-CORPUS-004 for the pre-flight gate.",
        EXHAUST_PANIC_PREFIX
    );
}

/// Iterator wrapper that cycles back to the beginning of a factory-
/// produced source when exhausted. Emits a single INFO log at the
/// first cycle boundary so operators can correlate training logs
/// against under-provisioned corpora.
///
/// The factory closure `F` is invoked every time the inner iterator
/// exhausts — the caller is responsible for producing a fresh iterator
/// (typically by re-opening the shard manifest) that replays the same
/// token stream.
///
/// This is INV-TRAIN-011 path (a). It is opt-in; default MODEL-2
/// dispatches use the non-cycling path so under-provisioned corpora
/// surface as a visible panic rather than silent multi-epoch churn.
pub struct CyclingBatchIter<F>
where
    F: FnMut() -> Box<dyn Iterator<Item = LMBatch>>,
{
    inner: Box<dyn Iterator<Item = LMBatch>>,
    factory: F,
    has_cycled: Arc<AtomicBool>,
}

impl<F> CyclingBatchIter<F>
where
    F: FnMut() -> Box<dyn Iterator<Item = LMBatch>>,
{
    /// Build a cycling iterator by calling `factory` once to obtain
    /// the initial source.
    pub fn new(mut factory: F) -> Self {
        let inner = factory();
        Self { inner, factory, has_cycled: Arc::new(AtomicBool::new(false)) }
    }

    /// Shared flag that flips `true` on the first cycle boundary. The
    /// flag survives the iterator's lifetime so tests can observe it
    /// without racing the log emission.
    pub fn has_cycled_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.has_cycled)
    }
}

impl<F> Iterator for CyclingBatchIter<F>
where
    F: FnMut() -> Box<dyn Iterator<Item = LMBatch>>,
{
    type Item = LMBatch;

    fn next(&mut self) -> Option<LMBatch> {
        if let Some(batch) = self.inner.next() {
            return Some(batch);
        }
        if !self.has_cycled.swap(true, Ordering::SeqCst) {
            eprintln!(
                "INFO: GATE-TRAIN-EXHAUST cycle — shard stream exhausted, \
                 re-opening source via factory closure. Subsequent cycles \
                 are silent. INV-TRAIN-011 path (a) is active. Consider \
                 enlarging the corpus to cover the planned token budget \
                 (contracts/pretraining-corpus-v1.yaml FALSIFY-CORPUS-004)."
            );
        }
        self.inner = (self.factory)();
        self.inner.next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    fn tiny_batch(batch_size: usize) -> LMBatch {
        // Each sequence needs `seq_len + 1` tokens (target shifts by 1).
        let seq_len_plus_one = 5_usize;
        let sequences: Vec<Vec<u32>> = (0..batch_size)
            .map(|b| {
                (0..seq_len_plus_one).map(|i| ((b * seq_len_plus_one + i) % 256) as u32).collect()
            })
            .collect();
        LMBatch::from_sequences(&sequences, 0, 0)
    }

    #[test]
    fn next_batch_or_panic_returns_first_batch_when_nonempty() {
        let batches = vec![tiny_batch(2), tiny_batch(2)];
        let mut iter: Box<dyn Iterator<Item = LMBatch>> = Box::new(batches.into_iter());
        let b = next_batch_or_panic(iter.as_mut());
        assert_eq!(b.batch_size, 2);
    }

    #[test]
    fn next_batch_or_panic_panics_with_gate_exhaust_prefix_on_empty() {
        let mut iter: Box<dyn Iterator<Item = LMBatch>> = Box::new(std::iter::empty::<LMBatch>());
        let result = catch_unwind(AssertUnwindSafe(|| {
            let _ = next_batch_or_panic(iter.as_mut());
        }));
        let err = result.expect_err("next_batch_or_panic must panic on empty iterator");
        let msg = err
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| err.downcast_ref::<&str>().copied())
            .unwrap_or("");
        assert!(
            msg.starts_with(EXHAUST_PANIC_PREFIX),
            "panic message must begin with {EXHAUST_PANIC_PREFIX:?}, got: {msg:?}"
        );
    }

    #[test]
    fn cycling_batch_iter_keeps_emitting_after_exhaustion() {
        // Three batches, called six times — must never return None and
        // must flip the cycle flag exactly once at call 4 (where the
        // first factory re-invocation happens).
        let factory = || -> Box<dyn Iterator<Item = LMBatch>> {
            Box::new(vec![tiny_batch(1), tiny_batch(1), tiny_batch(1)].into_iter())
        };
        let mut cyc = CyclingBatchIter::new(factory);
        let flag = cyc.has_cycled_flag();
        for i in 0..6 {
            let b =
                cyc.next().unwrap_or_else(|| panic!("cycle iter must never exhaust (call {i})"));
            assert_eq!(b.batch_size, 1);
        }
        assert!(flag.load(Ordering::SeqCst), "cycle flag must be set after 4th call");
    }

    #[test]
    fn cycling_batch_iter_never_emits_the_placeholder_fingerprint() {
        // Guards against future regressions where the cycling wrapper
        // could be mis-wired to produce the (1.0, 1.0) tuple. We check
        // the INV-TRAIN-011 boundary condition: the iterator always
        // yields an LMBatch, not a (loss, grad_norm) tuple — so the
        // (1.0, 1.0) fingerprint cannot physically appear from
        // CyclingBatchIter. This test documents that invariant.
        let factory =
            || -> Box<dyn Iterator<Item = LMBatch>> { Box::new(vec![tiny_batch(1)].into_iter()) };
        let mut cyc = CyclingBatchIter::new(factory);
        for _ in 0..10 {
            // Every emission is a real LMBatch — StepFn downstream
            // computes a real (loss, grad_norm) from it; the
            // placeholder tuple cannot appear.
            assert!(cyc.next().is_some());
        }
    }
}
