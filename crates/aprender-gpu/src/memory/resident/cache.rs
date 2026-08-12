//! Kernel Cache (WAPR-PERF-004)
//!
//! Global kernel cache to eliminate PTX recompilation overhead.

#[cfg(feature = "cuda")]
use std::cell::Cell;

#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};

#[cfg(feature = "cuda")]
use crate::driver::{CudaContext, CudaModule, CudaStream, LaunchConfig};
#[cfg(feature = "cuda")]
use crate::error::Result;

/// Global kernel cache to eliminate PTX recompilation overhead.
///
/// Each unique kernel configuration (name + parameters) is compiled once
/// and cached for reuse. This eliminates the 24x recompilation per inference
/// that was previously observed.
///
/// ## Keying Strategy
///
/// Keys are strings of format: `"{kernel_name}:{config}"` where config
/// encodes all parameters that affect the PTX output.
///
/// ## Thread Safety
///
/// The cache uses double-locking:
/// - Outer Mutex guards the HashMap
/// - Inner Arc<Mutex<CudaModule>> allows concurrent kernel launches
///
/// ## Example Keys
///
/// - `"softmax:32"` - SoftmaxKernel for row_size=32
/// - `"softmax_long_row:1500"` - LongRowSoftmaxKernel for row_size=1500
/// - `"residual_add:1024"` - ResidualAddKernel for n=1024
#[cfg(feature = "cuda")]
static KERNEL_CACHE: OnceLock<Mutex<HashMap<String, Arc<Mutex<CudaModule>>>>> = OnceLock::new();

// Kernel cache hit/miss statistics, counted **per thread**.
//
// GPU-ORD-2: these were process-global atomics, and every caller uses them the
// same way — reset, do work, read. Sharing them across threads meant one
// caller's window measured every other caller's compilations. Attributing a hit
// or a miss to the thread that asked for the kernel makes cross-thread
// pollution unrepresentable, and matches how the numbers are read.
#[cfg(feature = "cuda")]
thread_local! {
    static KERNEL_CACHE_HITS: Cell<u64> = const { Cell::new(0) };
    static KERNEL_CACHE_MISSES: Cell<u64> = const { Cell::new(0) };
}

/// Serialises *destructive* access to the shared compiled-module map.
///
/// The map itself must stay process-global — it exists so a kernel is JIT
/// compiled once rather than 24 times per inference. What broke
/// `test_kernel_cache_stats_after_operations` was not sharing the map, it was
/// one caller wiping it while another was midway through observing that a
/// repeated kernel does *not* recompile. The tell was two identical
/// `[KERNEL-CACHE] Compiling: gelu:16` lines in a single test's output.
///
/// `clear_kernel_cache` is the only way to empty the map and it takes this
/// lock, so anyone who needs a compile/hit sequence to be atomic against every
/// clear in the process can hold `kernel_cache_exclusive()` across it.
#[cfg(feature = "cuda")]
static CACHE_CLEAR_LOCK: Mutex<()> = Mutex::new(());

/// Get the global kernel cache, initializing if needed
#[cfg(feature = "cuda")]
pub(crate) fn get_kernel_cache() -> &'static Mutex<HashMap<String, Arc<Mutex<CudaModule>>>> {
    KERNEL_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Acquire the global cache lock, mapping poison errors to `GpuError`.
///
/// Every production call-site that needs the cache HashMap goes through this
/// single helper, eliminating the repeated `.lock().map_err(…)` boilerplate.
#[cfg(feature = "cuda")]
fn lock_cache(
    cache: &Mutex<HashMap<String, Arc<Mutex<CudaModule>>>>,
) -> Result<std::sync::MutexGuard<'_, HashMap<String, Arc<Mutex<CudaModule>>>>> {
    cache
        .lock()
        .map_err(|e| crate::GpuError::KernelLaunch(format!("Cache lock poisoned: {e}")))
}

/// Acquire a `Mutex<CudaModule>` lock, mapping poison errors to `GpuError`.
#[cfg(feature = "cuda")]
fn lock_module(module: &Mutex<CudaModule>) -> Result<std::sync::MutexGuard<'_, CudaModule>> {
    module
        .lock()
        .map_err(|e| crate::GpuError::KernelLaunch(format!("Module lock poisoned: {e}")))
}

/// Get a cached kernel module, compiling if not present.
///
/// # Arguments
///
/// * `ctx` - CUDA context for compilation
/// * `key` - Cache key (kernel_name:config)
/// * `ptx` - PTX source to compile on cache miss
///
/// # Returns
///
/// Arc to the cached module, wrapped in Mutex for mutable access.
#[cfg(feature = "cuda")]
pub(crate) fn get_or_compile_kernel(
    ctx: &CudaContext,
    key: &str,
    ptx: &str,
) -> Result<Arc<Mutex<CudaModule>>> {
    let cache = get_kernel_cache();

    // Fast path: check if already cached
    {
        let cache_guard = lock_cache(cache)?;
        if let Some(module) = cache_guard.get(key) {
            KERNEL_CACHE_HITS.with(|c| c.set(c.get() + 1));
            return Ok(Arc::clone(module));
        }
    }

    // Cache miss: compile and store
    KERNEL_CACHE_MISSES.with(|c| c.set(c.get() + 1));
    eprintln!("[KERNEL-CACHE] Compiling: {key}");

    let module = CudaModule::from_ptx(ctx, ptx)?;
    let module_arc = Arc::new(Mutex::new(module));

    // Insert into cache
    lock_cache(cache)?.insert(key.to_string(), Arc::clone(&module_arc));

    Ok(module_arc)
}

/// Compile (or fetch from cache), lock the module, and launch the kernel.
///
/// This centralises the repeated resource-management boilerplate that every
/// CUDA operation needs:
///
/// 1. `get_or_compile_kernel` — cache lookup / PTX JIT compilation
/// 2. `module_arc.lock()` — acquire the `Mutex<CudaModule>`
/// 3. `stream.launch_kernel` — unsafe dispatch
///
/// By housing this in `cache.rs` the pattern is written once and all call
/// sites across `elementwise`, `gemm`, `norm_activation`, `linear_bias`,
/// `layout`, and `incremental` can delegate to it.
///
/// # Safety
///
/// The caller must guarantee that `args` contains valid device pointers whose
/// types and count match the kernel signature identified by `kernel_name`.
#[cfg(feature = "cuda")]
pub(crate) fn compile_lock_launch(
    ctx: &CudaContext,
    stream: &CudaStream,
    cache_key: &str,
    ptx: &str,
    kernel_name: &str,
    config: &LaunchConfig,
    args: &mut [*mut std::ffi::c_void],
) -> Result<()> {
    let module_arc = get_or_compile_kernel(ctx, cache_key, ptx)?;
    let mut module = lock_module(&module_arc)?;
    // SAFETY: Caller guarantees args are valid pointers matching kernel signature.
    unsafe {
        stream.launch_kernel(&mut module, kernel_name, config, args)?;
    }
    Ok(())
}

/// Get kernel cache hits attributed to the current thread since last reset
#[cfg(feature = "cuda")]
#[must_use]
pub fn kernel_cache_hits() -> u64 {
    KERNEL_CACHE_HITS.with(Cell::get)
}

/// Get kernel cache misses attributed to the current thread since last reset
#[cfg(feature = "cuda")]
#[must_use]
pub fn kernel_cache_misses() -> u64 {
    KERNEL_CACHE_MISSES.with(Cell::get)
}

/// Reset the current thread's kernel cache statistics
#[cfg(feature = "cuda")]
pub fn reset_kernel_cache_stats() {
    KERNEL_CACHE_HITS.with(|c| c.set(0));
    KERNEL_CACHE_MISSES.with(|c| c.set(0));
}

/// Empty the shared module map. Assumes `CACHE_CLEAR_LOCK` is already held.
#[cfg(feature = "cuda")]
fn clear_kernel_cache_locked() {
    if let Some(cache) = KERNEL_CACHE.get() {
        if let Ok(mut guard) = lock_cache(cache) {
            guard.clear();
        }
    }
    reset_kernel_cache_stats();
}

/// Exclusive access to the shared compiled-kernel cache.
///
/// While this guard is alive no `clear_kernel_cache()` anywhere in the process
/// can take effect, so a sequence like "compile a kernel, run the same kernel
/// again, assert the second run did not recompile" is atomic against every
/// clear. Dropping the guard releases it.
#[cfg(feature = "cuda")]
pub struct KernelCacheExclusive(MutexGuard<'static, ()>);

#[cfg(feature = "cuda")]
impl KernelCacheExclusive {
    /// Empty the cache while holding exclusivity.
    pub fn clear(&self) {
        clear_kernel_cache_locked();
    }
}

/// Acquire exclusive access to the shared compiled-kernel cache.
///
/// Blocks until no other holder and no in-flight `clear_kernel_cache()` call
/// is running. See [`KernelCacheExclusive`].
#[cfg(feature = "cuda")]
#[must_use]
pub fn kernel_cache_exclusive() -> KernelCacheExclusive {
    // A poisoned lock guards no invariant here — it only orders clears — so
    // recovering the guard is correct and keeps an unrelated panic from
    // cascading into every later cache user.
    KernelCacheExclusive(
        CACHE_CLEAR_LOCK
            .lock()
            .unwrap_or_else(PoisonError::into_inner),
    )
}

/// Clear the kernel cache (useful for testing)
///
/// Takes the same exclusivity lock as [`kernel_cache_exclusive`], so it cannot
/// land in the middle of another caller's compile/hit observation.
#[cfg(feature = "cuda")]
pub fn clear_kernel_cache() {
    let _exclusive = kernel_cache_exclusive();
    clear_kernel_cache_locked();
}

// Non-CUDA stubs for compilation without cuda feature

/// Get kernel cache hit count (stub when CUDA disabled)
#[cfg(not(feature = "cuda"))]
#[must_use]
pub fn kernel_cache_hits() -> u64 {
    0
}

/// Get kernel cache miss count (stub when CUDA disabled)
#[cfg(not(feature = "cuda"))]
#[must_use]
pub fn kernel_cache_misses() -> u64 {
    0
}

/// Reset kernel cache statistics (stub when CUDA disabled)
#[cfg(not(feature = "cuda"))]
pub fn reset_kernel_cache_stats() {}

/// Clear the kernel cache (stub when CUDA disabled)
#[cfg(not(feature = "cuda"))]
pub fn clear_kernel_cache() {}

/// Exclusive access to the kernel cache (stub when CUDA disabled)
#[cfg(not(feature = "cuda"))]
pub struct KernelCacheExclusive;

#[cfg(not(feature = "cuda"))]
impl KernelCacheExclusive {
    /// Empty the cache (stub when CUDA disabled)
    pub fn clear(&self) {}
}

/// Acquire exclusive access to the kernel cache (stub when CUDA disabled)
#[cfg(not(feature = "cuda"))]
#[must_use]
pub fn kernel_cache_exclusive() -> KernelCacheExclusive {
    KernelCacheExclusive
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reset stats and assert both hits and misses are zero.
    fn assert_clean_stats() {
        reset_kernel_cache_stats();
        assert_eq!(kernel_cache_hits(), 0);
        assert_eq!(kernel_cache_misses(), 0);
    }

    #[test]
    fn test_kernel_cache_stats_initial() {
        assert_clean_stats();
    }

    #[test]
    fn test_clear_kernel_cache() {
        // Just verify it doesn't panic
        clear_kernel_cache();
        assert_clean_stats();
    }

    /// Reset and clear are both idempotent and leave stats at zero.
    #[test]
    fn test_idempotent_operations() {
        for _ in 0..3 {
            reset_kernel_cache_stats();
            clear_kernel_cache();
        }
        assert_clean_stats();
    }
}

/// CUDA-specific tests that exercise the cache infrastructure
#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::*;

    /// Reset stats and assert both hits and misses are zero.
    fn assert_clean_stats() {
        reset_kernel_cache_stats();
        assert_eq!(kernel_cache_hits(), 0);
        assert_eq!(kernel_cache_misses(), 0);
    }

    /// Increment both counters by the given amounts, assert they match, then reset.
    fn assert_counter_round_trip(hits: u64, misses: u64) {
        assert_clean_stats();
        for _ in 0..hits {
            KERNEL_CACHE_HITS.with(|c| c.set(c.get() + 1));
        }
        for _ in 0..misses {
            KERNEL_CACHE_MISSES.with(|c| c.set(c.get() + 1));
        }
        assert_eq!(kernel_cache_hits(), hits);
        assert_eq!(kernel_cache_misses(), misses);
        assert_clean_stats();
    }

    /// GPU-ORD-2 falsifier (a): another thread's compilations must not land in
    /// this thread's hit/miss window.
    #[test]
    fn test_cache_stats_are_not_polluted_by_other_threads() {
        reset_kernel_cache_stats();
        KERNEL_CACHE_HITS.with(|c| c.set(c.get() + 1));

        std::thread::spawn(|| {
            reset_kernel_cache_stats();
            for _ in 0..500 {
                KERNEL_CACHE_HITS.with(|c| c.set(c.get() + 1));
                KERNEL_CACHE_MISSES.with(|c| c.set(c.get() + 1));
            }
        })
        .join()
        .expect("neighbour thread must not panic");

        assert_eq!(
            (kernel_cache_hits(), kernel_cache_misses()),
            (1, 0),
            "another thread's cache activity was attributed to this thread"
        );
    }

    /// GPU-ORD-2 falsifier (b): a clear cannot land inside someone else's
    /// compile/hit observation.
    ///
    /// This is the interference itself, expressed as behaviour and without
    /// needing a GPU: while exclusivity is held, a `clear_kernel_cache()` on
    /// another thread must not have completed. Before the fix the clear was a
    /// straight `HashMap::clear()` with nothing to wait on, so it landed
    /// immediately — which is exactly how a neighbour wiped `gelu:16` between
    /// the two halves of `test_kernel_cache_stats_after_operations`.
    #[test]
    fn test_clear_cannot_interleave_with_exclusive_holder() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc as StdArc;

        let cleared = StdArc::new(AtomicBool::new(false));
        let cleared_bg = StdArc::clone(&cleared);

        let exclusive = kernel_cache_exclusive();

        let bg = std::thread::spawn(move || {
            clear_kernel_cache();
            cleared_bg.store(true, Ordering::SeqCst);
        });

        // Give the background thread ample opportunity to run to completion.
        for _ in 0..50 {
            assert!(
                !cleared.load(Ordering::SeqCst),
                "clear_kernel_cache() completed while another caller held \
                 exclusivity — a clear can still land inside someone else's \
                 compile/hit observation"
            );
            std::thread::sleep(std::time::Duration::from_millis(2));
        }

        drop(exclusive);
        bg.join().expect("clearing thread must not panic");
        assert!(
            cleared.load(Ordering::SeqCst),
            "clear_kernel_cache() never completed after exclusivity was released"
        );
    }

    /// Clear the cache and assert it is empty with zeroed stats.
    ///
    /// Holds exclusivity across the clear *and* the assertion: otherwise a
    /// concurrent test compiling a kernel repopulates the map between the two
    /// and this reads a non-empty cache.
    fn clear_and_assert_empty() {
        let exclusive = kernel_cache_exclusive();
        exclusive.clear();
        let guard = lock_cache(get_kernel_cache()).expect("Cache lock should not be poisoned");
        assert!(guard.is_empty(), "Cache should be empty");
    }

    /// Test get_kernel_cache is idempotent (returns same static reference)
    /// and the lock can be acquired repeatedly.
    #[test]
    fn test_get_kernel_cache_static_and_reentrant() {
        let cache1 = get_kernel_cache();
        let cache2 = get_kernel_cache();
        assert!(std::ptr::eq(cache1, cache2));
        // Lock can be acquired and released multiple times
        for _ in 0..3 {
            let _guard = lock_cache(cache1).expect("lock");
        }
    }

    /// Test clear_kernel_cache empties the hashmap and resets stats.
    #[test]
    fn test_clear_kernel_cache_clears_hashmap() {
        clear_and_assert_empty();
    }

    /// Test atomic counter increment round-trips at multiple scales.
    #[test]
    fn test_atomic_counter_operations() {
        assert_counter_round_trip(5, 3);
        assert_counter_round_trip(100, 50);
    }

    /// Test clear_kernel_cache is safe even if the cache was never
    /// explicitly initialised (covers the `KERNEL_CACHE.get() == None` path).
    #[test]
    fn test_clear_uninitialized_cache() {
        clear_kernel_cache();
        assert_clean_stats();
    }
}
