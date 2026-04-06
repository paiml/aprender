//! Kernel Cache (WAPR-PERF-004)
//!
//! Global kernel cache to eliminate PTX recompilation overhead.

#[cfg(feature = "cuda")]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex, OnceLock};

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

/// Statistics for kernel cache performance
#[cfg(feature = "cuda")]
static KERNEL_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "cuda")]
static KERNEL_CACHE_MISSES: AtomicU64 = AtomicU64::new(0);

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
    cache.lock().map_err(|e| crate::GpuError::KernelLaunch(format!("Cache lock poisoned: {e}")))
}

/// Acquire a `Mutex<CudaModule>` lock, mapping poison errors to `GpuError`.
#[cfg(feature = "cuda")]
fn lock_module(module: &Mutex<CudaModule>) -> Result<std::sync::MutexGuard<'_, CudaModule>> {
    module.lock().map_err(|e| crate::GpuError::KernelLaunch(format!("Module lock poisoned: {e}")))
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
            KERNEL_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
            return Ok(Arc::clone(module));
        }
    }

    // Cache miss: compile and store
    KERNEL_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
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

/// Get kernel cache hit count
#[cfg(feature = "cuda")]
#[must_use]
pub fn kernel_cache_hits() -> u64 {
    KERNEL_CACHE_HITS.load(Ordering::Relaxed)
}

/// Get kernel cache miss count
#[cfg(feature = "cuda")]
#[must_use]
pub fn kernel_cache_misses() -> u64 {
    KERNEL_CACHE_MISSES.load(Ordering::Relaxed)
}

/// Reset kernel cache statistics
#[cfg(feature = "cuda")]
pub fn reset_kernel_cache_stats() {
    KERNEL_CACHE_HITS.store(0, Ordering::Relaxed);
    KERNEL_CACHE_MISSES.store(0, Ordering::Relaxed);
}

/// Clear the kernel cache (useful for testing)
#[cfg(feature = "cuda")]
pub fn clear_kernel_cache() {
    if let Some(cache) = KERNEL_CACHE.get() {
        if let Ok(mut guard) = lock_cache(cache) {
            guard.clear();
        }
    }
    reset_kernel_cache_stats();
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
            KERNEL_CACHE_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        for _ in 0..misses {
            KERNEL_CACHE_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        assert_eq!(kernel_cache_hits(), hits);
        assert_eq!(kernel_cache_misses(), misses);
        assert_clean_stats();
    }

    /// Clear the cache and assert it is empty with zeroed stats.
    fn clear_and_assert_empty() {
        clear_kernel_cache();
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
