//! PERF-034: partial selection + scratch reuse for the single-stream decode samplers.
//!
//! Every top-k sampler in this crate was written the same way:
//!
//! ```ignore
//! let mut indexed: Vec<(usize, f32)> = data.iter().copied().enumerate().collect();
//! indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
//! indexed.truncate(k);
//! ```
//!
//! On Qwen2.5 (`vocab = 151_936..=152_064`) with the default `top_k = 40` that is a
//! full `O(V log V)` sort — about 2.6M `partial_cmp` calls through a closure — to keep
//! 40 elements, plus a fresh `Vec<(usize, f32)>` of `V * 16 = 2.4 MB` **per token**,
//! plus the `n/2` scratch buffer that Rust's *stable* sort allocates (another ~1.2 MB).
//! `select_nth_unstable_by` does the same selection in `O(V)` with no allocation.
//!
//! # Why this is bit-exact
//!
//! The existing code calls `sort_by`, which is **stable**: elements the comparator
//! reports as `Equal` keep their original relative order. The vector is built with
//! `.enumerate()`, so "original order" is exactly "index ascending". The observable
//! order is therefore the total order *(value descending, then index ascending)* —
//! [`cmp_desc_then_index`].
//!
//! Because every index is distinct, that comparator has **no ties at all**, so the
//! sorted permutation is *unique*: an unstable sort must produce the same one. And
//! `select_nth_unstable_by(k - 1, cmp)` partitions on that same unique total order,
//! so `buf[..k]` is exactly the set the stable sort would have left after
//! `truncate(k)` — sorting just those `k` restores the order. Ties in the *value*
//! (which do occur, e.g. two masked `NEG_INFINITY` logits) are broken identically by
//! the explicit index term.
//!
//! # NaN
//!
//! `partial_cmp(..).unwrap_or(Equal)` is not a strict weak ordering when a NaN is
//! present, so the *pre-existing* behaviour was already unspecified (Rust guarantees
//! only memory safety for an inconsistent comparator, for both `sort_by` and
//! `select_nth_unstable_by`). This module does not change that: it keeps
//! `partial_cmp`, deliberately **not** `total_cmp`, because `total_cmp` orders
//! `-0.0 < 0.0` where `partial_cmp` calls them equal — switching would be a silent
//! behaviour change on real (finite) logits.

use std::cell::RefCell;
use std::cmp::Ordering;

/// The total order every `(index, value)` sampler in this crate sorts by:
/// value descending, ties broken by index ascending.
///
/// This is the order a *stable* `sort_by(|a, b| b.1.partial_cmp(&a.1))` already
/// produced on a `.enumerate()`-built vector; making it explicit is what lets the
/// unstable partial-selection routines below be bit-exact replacements.
#[inline]
#[must_use]
pub fn cmp_desc_then_index(a: &(usize, f32), b: &(usize, f32)) -> Ordering {
    b.1.partial_cmp(&a.1)
        .unwrap_or(Ordering::Equal)
        .then_with(|| a.0.cmp(&b.0))
}

/// Sort `buf` into the canonical descending order.
///
/// Uses `sort_unstable_by`, which allocates nothing, in place of the stable
/// `sort_by`, which allocates an `n/2` scratch buffer. Identical result because
/// [`cmp_desc_then_index`] admits no ties.
#[inline]
pub fn sort_desc_by_index(buf: &mut [(usize, f32)]) {
    buf.sort_unstable_by(cmp_desc_then_index);
}

/// Reduce `buf` to its top `k` entries, in the exact order a full stable sort
/// followed by `truncate(k)` would have produced.
///
/// `O(n)` selection plus `O(k log k)` sort instead of `O(n log n)`, with zero
/// allocation. `k >= buf.len()` degrades to a plain full sort.
pub fn retain_top_k_sorted(buf: &mut Vec<(usize, f32)>, k: usize) {
    if k == 0 {
        buf.clear();
        return;
    }
    if k < buf.len() {
        // Partition so that `buf[..k]` holds the k smallest under the total order
        // (= the k largest logits, ties to the lower index) in unspecified order.
        buf.select_nth_unstable_by(k - 1, cmp_desc_then_index);
        buf.truncate(k);
    }
    sort_desc_by_index(buf);
}

thread_local! {
    /// Per-thread `(index, logit)` scratch, reused across decode steps.
    ///
    /// A decode loop is single-stream and calls the sampler once per token, so one
    /// buffer per thread is grown once (to `V * 16` bytes) and then reused for the
    /// rest of the stream: steady-state heap traffic for the candidate vector drops
    /// to zero. Nothing is held across the callback, so this is not reentrant-hostile
    /// in the only way that matters — the sampler never calls itself.
    static CANDIDATE_SCRATCH: RefCell<Vec<(usize, f32)>> = const { RefCell::new(Vec::new()) };
}

/// Run `f` with the thread-local candidate scratch, cleared but with its capacity
/// intact.
///
/// Falls back to a fresh `Vec` if the scratch is already borrowed (it never is on the
/// decode path; this only keeps the function total instead of panicking).
pub fn with_candidate_scratch<R>(f: impl FnOnce(&mut Vec<(usize, f32)>) -> R) -> R {
    CANDIDATE_SCRATCH.with(|cell| match cell.try_borrow_mut() {
        Ok(mut buf) => {
            buf.clear();
            let out = f(&mut buf);
            // Drop the elements but keep the allocation for the next token.
            buf.clear();
            out
        },
        Err(_) => f(&mut Vec::new()),
    })
}

/// Fill `buf` with `(index, logit / temperature)` for every logit.
///
/// Fuses the temperature scaling into the candidate build, removing the separate
/// `scaled: Vec<f32>` (`V * 4` bytes per token) the samplers used to materialise.
#[inline]
pub fn fill_scaled(buf: &mut Vec<(usize, f32)>, logits: &[f32], temperature: f32) {
    buf.clear();
    buf.reserve(logits.len());
    buf.extend(
        logits
            .iter()
            .enumerate()
            .map(|(i, &x)| (i, x / temperature)),
    );
}

/// Index of the maximum of `values`, ties going to the lowest index.
///
/// Exactly the element a stable descending `sort_by` leaves at position 0, which is
/// what several "top-k" helpers were paying a full sort to read via `.first()`.
/// Returns `None` for an empty slice.
#[must_use]
pub fn argmax_first_wins(values: &[f32]) -> Option<usize> {
    let mut iter = values.iter().enumerate();
    let (mut best_i, mut best_v) = iter.next().map(|(i, &v)| (i, v))?;
    for (i, &v) in iter {
        if v > best_v {
            best_i = i;
            best_v = v;
        }
    }
    Some(best_i)
}

#[cfg(test)]
#[path = "sampling_select_tests.rs"]
mod sampling_select_tests;
