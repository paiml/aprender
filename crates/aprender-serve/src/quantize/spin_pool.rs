// PMAT-303: Lightweight thread pool for GEMV — lower overhead than rayon.
// Uses crossbeam-style scoped threads with a pre-allocated thread pool.
// Each dispatch creates a scoped parallel region like OpenMP parallel-for.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Lightweight parallel dispatch for CPU GEMV.
/// Uses rayon's thread pool but with a single par_bridge to minimize overhead.
pub(crate) struct GemvPool {
    num_threads: usize,
}

impl GemvPool {
    pub fn new() -> Self {
        Self {
            num_threads: rayon::current_num_threads(),
        }
    }

    /// Parallel for-loop: calls `f(row)` for each row in 0..n.
    /// Uses contiguous chunks with exactly `num_threads` tasks.
    #[inline]
    pub fn parallel_rows<F: Fn(usize) + Sync>(&self, n: usize, f: F) {
        use rayon::prelude::*;
        let chunk = (n + self.num_threads - 1) / self.num_threads;
        (0..n)
            .into_par_iter()
            .with_min_len(chunk)
            .for_each(|row| f(row));
    }
}

static GEMV_POOL: std::sync::OnceLock<GemvPool> = std::sync::OnceLock::new();

pub(crate) fn get_gemv_pool() -> &'static GemvPool {
    GEMV_POOL.get_or_init(GemvPool::new)
}
