//! PERF-034 — measure, don't assert: how much the dense decode sampler allocates.
//!
//! This lives in its own integration-test binary because it installs a counting
//! `#[global_allocator]`, which is a per-binary singleton. It measures the *shipping*
//! sampler (`OwnedQuantizedModel::sample_topk_seeded`) against a verbatim copy of the
//! pre-PERF-034 implementation, **in the same process, on the same logits, through the
//! same allocator**, so the before/after numbers are a paired comparison rather than
//! two runs quoted at each other.
//!
//! Allocation counts are exact and deterministic, so they are asserted. Wall time is
//! *reported* (`PERF034_TIMING=1`) and never asserted: a timing gate in a checked-in
//! test is the class of gate that fails at random on a loaded runner.
//!
//! ```text
//! cargo test -p aprender-serve --release --test perf034_decode_sampler_alloc -- --nocapture
//! PERF034_TIMING=1 cargo test -p aprender-serve --release \
//!     --test perf034_decode_sampler_alloc -- --nocapture
//! ```
//!
//! No `--test-threads=1` is required, and that is load-bearing rather than tidy: the
//! first version of this harness counted into process-global atomics and was correct
//! ONLY under `--test-threads=1`. Run the default way, the nucleus test's
//! allocations were charged to the steady-state test's window and both tests
//! reported the same wrong total. Counting is now per-thread; see `Counting`.
//!
//! NOTE: the workspace CI job runs `--lib` only, so nothing here gates a merge. The
//! behavioural equivalence proof deliberately lives in the lib unit tests
//! (`gguf/inference/generate_quantized_perf034_tests.rs`), which do gate.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use realizar::gguf::OwnedQuantizedModel;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::time::Instant;

/// Qwen2.5's vocabulary. The whole point of PERF-034 is that this number is the
/// per-token work factor, so the harness must use the real one.
const VOCAB: usize = 152_064;
/// The shipped default `top_k`.
const TOP_K: usize = 40;

// ---------------------------------------------------------------------------
// Counting allocator
// ---------------------------------------------------------------------------

struct Counting;

thread_local! {
    /// Counting is per-THREAD, deliberately, and this is the second version.
    ///
    /// The first used process-global atomics, and it was WRONG in the only way that
    /// mattered: `libtest` runs `#[test]`s on concurrent threads by default, so the
    /// nucleus test's allocations landed inside the steady-state test's measurement
    /// window and *both* tests then reported the same polluted total (33 allocations
    /// / 5376 bytes -- byte-identical between two tests measuring different things).
    /// It only read correctly under `--test-threads=1`, which is not how anyone runs
    /// `cargo test`. A measurement harness that is silently wrong under the default
    /// invocation is worse than no harness.
    ///
    /// `Cell<u64>` / `Cell<bool>` with `const` initialisers have no `Drop` and no
    /// lazy initialiser, so reading them from inside `alloc` neither allocates nor
    /// registers a TLS destructor -- the two ways a thread-local can deadlock or
    /// recurse inside a global allocator.
    static COUNTING_ON: Cell<bool> = const { Cell::new(false) };
    static ALLOC_CALLS: Cell<u64> = const { Cell::new(0) };
    static ALLOC_BYTES: Cell<u64> = const { Cell::new(0) };
}

/// Charge one allocation of `size` bytes to the calling thread, if that thread is
/// currently measuring. Allocations on every other thread are ignored.
#[inline]
fn record(size: usize) {
    // `try_with` rather than `with`: during thread teardown the slot may be gone, and
    // an allocator must never panic. These types have no destructor so this should
    // not happen, but the allocator is the wrong place to be confident.
    let _ = COUNTING_ON.try_with(|on| {
        if on.get() {
            let _ = ALLOC_CALLS.try_with(|c| c.set(c.get().wrapping_add(1)));
            let _ = ALLOC_BYTES.try_with(|b| b.set(b.get().wrapping_add(size as u64)));
        }
    });
}

// SAFETY: `Counting` is a pure pass-through to the `System` allocator. It adds only
// relaxed atomic counter updates, which allocate nothing and cannot re-enter the
// allocator, and it forwards every pointer, layout and size unchanged. The
// `GlobalAlloc` contract is therefore upheld exactly as `System` upholds it.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        // SAFETY: `layout` is forwarded unmodified from a caller that already
        // satisfied `GlobalAlloc::alloc`'s preconditions.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` was returned by `System.alloc`/`System.realloc` above with
        // this same `layout`, because every allocation in this binary goes through
        // this allocator; both are forwarded unmodified.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record(new_size);
        // SAFETY: `ptr`/`layout` came from this allocator and `new_size` is the
        // caller's, all forwarded unmodified to the same `System` allocator.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AllocStats {
    calls: u64,
    bytes: u64,
}

/// Count every heap allocation `f` performs **on the calling thread**.
///
/// Concurrent tests cannot pollute this: another thread's allocations are charged to
/// that thread's own counters, which nobody reads.
fn measure_allocs<R>(f: impl FnOnce() -> R) -> (R, AllocStats) {
    ALLOC_CALLS.with(|c| c.set(0));
    ALLOC_BYTES.with(|b| b.set(0));
    COUNTING_ON.with(|on| on.set(true));
    let out = f();
    COUNTING_ON.with(|on| on.set(false));
    (
        out,
        AllocStats {
            calls: ALLOC_CALLS.with(Cell::get),
            bytes: ALLOC_BYTES.with(Cell::get),
        },
    )
}

// ---------------------------------------------------------------------------
// The pre-PERF-034 sampler, verbatim
// ---------------------------------------------------------------------------

/// `OwnedQuantizedModel::sample_topk_with_draw` exactly as it shipped at `62d23d8d1`.
fn legacy_sample_topk_with_draw(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    r: f32,
) -> u32 {
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();

    let mut indexed: Vec<(usize, f32)> = scaled.iter().copied().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    if top_k > 0 && top_k < indexed.len() {
        indexed.truncate(top_k);
    }

    if top_p > 0.0 && top_p < 1.0 {
        let max_val = indexed.first().map_or(0.0, |(_, v)| *v);
        let exp_vals: Vec<f32> = indexed.iter().map(|(_, v)| (v - max_val).exp()).collect();
        let total: f32 = exp_vals.iter().sum();
        if total > 0.0 {
            let mut cumulative = 0.0;
            let mut cutoff = indexed.len();
            for (i, &ev) in exp_vals.iter().enumerate() {
                cumulative += ev / total;
                if cumulative >= top_p {
                    cutoff = i + 1;
                    break;
                }
            }
            indexed.truncate(cutoff);
        }
    }

    let max_val = indexed.first().map_or(0.0, |(_, v)| *v);
    let exp_sum: f32 = indexed.iter().map(|(_, v)| (v - max_val).exp()).sum();
    let probs: Vec<(usize, f32)> = indexed
        .iter()
        .map(|(i, v)| (*i, (v - max_val).exp() / exp_sum))
        .collect();

    let mut cumulative = 0.0;
    for &(idx, prob) in &probs {
        cumulative += prob;
        if cumulative >= r {
            return idx as u32;
        }
    }

    probs.last().map_or(0, |(idx, _)| *idx as u32)
}

/// Reproducible pseudo-logits (LCG) so the harness measures the sampler, not the RNG.
fn pseudo_logits(vocab: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..vocab)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 33) as f32 / 4_294_967_296.0f32).mul_add(24.0, -12.0)
        })
        .collect()
}

/// The pre-PERF-034 sampler wrapped in the *same* entry-point shape as the shipping
/// one: it owns a seeded RNG and draws once per token, exactly as
/// `sample_topk_seeded` does. Both sides therefore pay for the identical RNG work,
/// so neither the allocation counts nor the timings attribute it to one of them.
fn legacy_sample_topk_seeded(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    rng: &mut StdRng,
) -> u32 {
    let r: f32 = rng.random();
    legacy_sample_topk_with_draw(logits, temperature, top_k, top_p, r)
}

// ---------------------------------------------------------------------------
// Allocation measurement (asserted — exact and deterministic)
// ---------------------------------------------------------------------------

#[test]
fn perf034_steady_state_decode_sampling_allocates_nothing() {
    const STEPS: usize = 64;
    let logits = pseudo_logits(VOCAB, 1);

    // Warm the thread-local scratch exactly as the first token of a stream does.
    let mut sink = 0u32;
    let mut warm = StdRng::seed_from_u64(5);
    sink ^= OwnedQuantizedModel::sample_topk_seeded(&logits, 0.7, TOP_K, 1.0, &mut warm);

    let mut rng_new = StdRng::seed_from_u64(17);
    let (out, new_stats) = measure_allocs(|| {
        let mut acc = Vec::with_capacity(STEPS);
        for _ in 0..STEPS {
            acc.push(OwnedQuantizedModel::sample_topk_seeded(
                &logits,
                0.7,
                TOP_K,
                1.0,
                &mut rng_new,
            ));
        }
        acc
    });

    let mut rng_legacy = StdRng::seed_from_u64(17);
    let (out2, legacy_stats) = measure_allocs(|| {
        let mut acc = Vec::with_capacity(STEPS);
        for _ in 0..STEPS {
            acc.push(legacy_sample_topk_seeded(
                &logits,
                0.7,
                TOP_K,
                1.0,
                &mut rng_legacy,
            ));
        }
        acc
    });
    sink ^= out.iter().chain(out2.iter()).fold(0u32, |a, b| a ^ b);

    // Same seed, same logits: the token streams must match, or the allocation
    // comparison below is between two different computations.
    assert_eq!(out, out2, "the two samplers must agree on every token");

    let n = STEPS as u64;
    // `Vec::with_capacity` inside the measured closure is one allocation each; charge
    // it back so the reported figure is the SAMPLER's cost, not the harness's.
    let harness_allocs = 1u64;
    let new_calls = new_stats.calls.saturating_sub(harness_allocs);
    let legacy_calls = legacy_stats.calls.saturating_sub(harness_allocs);

    println!(
        "\nPERF-034 allocation, vocab={VOCAB} top_k={TOP_K} top_p=1.0, {n} decode steps\n\
         \x20 before: {legacy_calls:>7} allocations ({:>11} bytes) = {:>6.2} allocs/token, {:>10.0} B/token\n\
         \x20 after:  {new_calls:>7} allocations ({:>11} bytes) = {:>6.2} allocs/token, {:>10.0} B/token",
        legacy_stats.bytes,
        legacy_calls as f64 / n as f64,
        legacy_stats.bytes as f64 / n as f64,
        new_stats.bytes,
        new_calls as f64 / n as f64,
        new_stats.bytes as f64 / n as f64,
    );

    // The claim PERF-034 makes: at the shipped default (top_p = 1.0) a steady-state
    // decode step allocates NOTHING. The scratch is grown once and reused.
    assert_eq!(
        new_calls, 0,
        "steady-state decode sampling must not allocate; got {new_stats:?}"
    );

    // Guard the guard: the legacy path must be measurably non-zero, or the allocator
    // hook is simply not firing and the assertion above proves nothing.
    assert!(
        legacy_calls >= n,
        "counting allocator appears inert: legacy path reported {legacy_stats:?} over \
         {n} steps, expected at least one allocation each"
    );
    assert!(
        legacy_stats.bytes as f64 / n as f64 > 1_000_000.0,
        "legacy path should allocate megabytes per token at vocab={VOCAB}, got {legacy_stats:?}"
    );
    assert_ne!(
        sink,
        u32::MAX,
        "keep the samplers from being optimised away"
    );
}

#[test]
fn perf034_nucleus_path_also_allocates_nothing() {
    // top_p < 1.0 takes the nucleus branch, which used to materialise one
    // `exp_vals: Vec<f32>` per token (measured at exactly 1.00 allocs/token,
    // 160 B/token = 40 f32 at the default top_k). That Vec is gone too, so this
    // configuration must also reach zero.
    const STEPS: usize = 32;
    let logits = pseudo_logits(VOCAB, 2);

    let mut warm = StdRng::seed_from_u64(5);
    let mut sink = OwnedQuantizedModel::sample_topk_seeded(&logits, 0.8, TOP_K, 0.9, &mut warm);

    let mut rng = StdRng::seed_from_u64(23);
    let (out, new_stats) = measure_allocs(|| {
        let mut acc = 0u32;
        for _ in 0..STEPS {
            acc ^= OwnedQuantizedModel::sample_topk_seeded(&logits, 0.8, TOP_K, 0.9, &mut rng);
        }
        acc
    });
    sink ^= out;

    let n = STEPS as u64;
    println!(
        "PERF-034 allocation, vocab={VOCAB} top_k={TOP_K} top_p=0.9, {n} decode steps\n\
         \x20 after:  {:>7} allocations ({:>11} bytes) = {:>6.2} allocs/token, {:>10.0} B/token",
        new_stats.calls,
        new_stats.bytes,
        new_stats.calls as f64 / n as f64,
        new_stats.bytes as f64 / n as f64,
    );

    assert_eq!(
        new_stats.calls, 0,
        "nucleus-path decode sampling must not allocate; got {new_stats:?}"
    );
    assert_ne!(sink, u32::MAX);
}

/// The harness must measure THIS thread only.
///
/// This is the regression test for the defect this file shipped with: process-global
/// counters, so a `#[test]` running concurrently on another thread was charged into
/// whichever measurement window happened to be open. It went unnoticed because the
/// harness was only ever run with `--test-threads=1`.
///
/// RED-on-regression: revert `record` to process-global atomics and this fails --
/// the spawned thread's megabytes land in the measured total.
#[test]
fn perf034_measurement_is_isolated_from_other_threads() {
    // Warm this thread's scratch so the measured region is genuinely allocation-free.
    let logits = pseudo_logits(4096, 9);
    let mut warm = StdRng::seed_from_u64(1);
    let sink = OwnedQuantizedModel::sample_topk_seeded(&logits, 0.7, TOP_K, 1.0, &mut warm);

    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop_thread = std::sync::Arc::clone(&stop);
    // A noisy neighbour doing exactly what a concurrent test does: allocate hard.
    let noisy = std::thread::spawn(move || {
        let mut churn = 0usize;
        while !stop_thread.load(std::sync::atomic::Ordering::Relaxed) {
            let v: Vec<u8> = vec![7u8; 64 * 1024];
            churn = churn.wrapping_add(v.len());
        }
        churn
    });

    let mut rng = StdRng::seed_from_u64(2);
    let (out, stats) = measure_allocs(|| {
        let mut acc = 0u32;
        for _ in 0..2048 {
            acc ^= OwnedQuantizedModel::sample_topk_seeded(&logits, 0.7, TOP_K, 1.0, &mut rng);
        }
        acc
    });

    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let churn = noisy.join().unwrap_or(0);

    assert!(
        churn > 0,
        "the noisy thread never allocated, so this proves nothing"
    );
    assert_eq!(
        stats.calls, 0,
        "another thread's allocations leaked into this measurement: {stats:?} \
         (neighbour churned {churn} bytes)"
    );
    assert_ne!(sink ^ out, u32::MAX);
}

// ---------------------------------------------------------------------------
// Timing report (printed, never asserted)
// ---------------------------------------------------------------------------

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = xs.len();
    if n.is_multiple_of(2) {
        (xs[n / 2 - 1] + xs[n / 2]) / 2.0
    } else {
        xs[n / 2]
    }
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx]
}

/// Report per-token sampling latency for both implementations.
///
/// Opt-in (`PERF034_TIMING=1`) and assertion-free. It reports n, median, and the
/// interquartile range plus min/max, because a bare median over a bimodal
/// distribution has already misreported a speedup by 2.4x on this fleet.
#[test]
fn perf034_timing_report() {
    if std::env::var("PERF034_TIMING").as_deref() != Ok("1") {
        println!("perf034_timing_report: set PERF034_TIMING=1 to run (measurement, not a gate)");
        return;
    }

    const REPS: usize = 25;
    const STEPS: usize = 32;
    let logits = pseudo_logits(VOCAB, 3);

    let mut sink = 0u32;
    let mut legacy_us = Vec::with_capacity(REPS);
    let mut new_us = Vec::with_capacity(REPS);
    let mut rng_legacy = StdRng::seed_from_u64(41);
    let mut rng_new = StdRng::seed_from_u64(41);

    // Interleave the two implementations so any thermal or scheduler drift over the
    // run hits both equally instead of being attributed to whichever ran second.
    for _ in 0..REPS {
        let t0 = Instant::now();
        for _ in 0..STEPS {
            sink ^= legacy_sample_topk_seeded(&logits, 0.7, TOP_K, 1.0, &mut rng_legacy);
        }
        legacy_us.push(t0.elapsed().as_secs_f64() * 1e6 / STEPS as f64);

        let t1 = Instant::now();
        for _ in 0..STEPS {
            sink ^= OwnedQuantizedModel::sample_topk_seeded(&logits, 0.7, TOP_K, 1.0, &mut rng_new);
        }
        new_us.push(t1.elapsed().as_secs_f64() * 1e6 / STEPS as f64);
    }

    let l_med = median(&mut legacy_us);
    let n_med = median(&mut new_us);
    println!(
        "\nPERF-034 timing, vocab={VOCAB} top_k={TOP_K} top_p=1.0, n={REPS} reps x {STEPS} steps\n\
         \x20 before: median {:8.1} us/token  IQR [{:8.1}, {:8.1}]  min {:8.1}  max {:8.1}\n\
         \x20 after:  median {:8.1} us/token  IQR [{:8.1}, {:8.1}]  min {:8.1}  max {:8.1}\n\
         \x20 median speedup: {:.2}x",
        l_med,
        quantile(&legacy_us, 0.25),
        quantile(&legacy_us, 0.75),
        legacy_us[0],
        legacy_us[REPS - 1],
        n_med,
        quantile(&new_us, 0.25),
        quantile(&new_us, 0.75),
        new_us[0],
        new_us[REPS - 1],
        l_med / n_med,
    );
    assert_ne!(sink, u32::MAX);
}
