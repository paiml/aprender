//! Late materialization for browser WASM deployment.
//!
//! Implements the pattern from Abadi et al. (2008) to minimize memory usage
//! by deferring row reconstruction until absolutely necessary.
//!
//! # Key Insight
//!
//! Traditional (early materialization):
//! ```text
//! Read all rows → Filter → Aggregate
//!     |
//!     └─> High memory usage (all rows in memory)
//! ```
//!
//! Late materialization:
//! ```text
//! Read filter column → Get matching indices → Read only needed columns
//!     |
//!     └─> Low memory usage (only filtered results)
//! ```
//!
//! # Memory Savings Example
//!
//! Dataset: 1M rows × 100 columns × 8 bytes = 800MB
//! Filter selectivity: 1% (10K matching rows)
//!
//! Early materialization: 800MB loaded
//! Late materialization: 8MB loaded (100x reduction!)
//!
//! # References
//! - Abadi et al. 2008 (CIDR): "Materialization strategies in column-oriented DBMS"
//! - DOI: 10.1145/1376616.1376712

#![cfg(target_arch = "wasm32")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use wasm_bindgen::prelude::*;
use web_sys::console;

/// Memory budget for browser deployment (<2GB limit)
///
/// Tracks memory usage and prevents OOM by enforcing limits.
/// Uses atomic operations for thread-safe tracking.
///
/// # Toyota Way: Poka-Yoke (Mistake Proofing)
///
/// Prevents browser crashes from excessive memory allocation.
#[derive(Clone)]
pub struct MemoryBudget {
    /// Memory limit in bytes (default: 1.5GB, leaves 500MB headroom)
    limit: usize,

    /// Current memory usage in bytes
    current: Arc<AtomicUsize>,
}

impl MemoryBudget {
    /// Create new memory budget with default limit (1.5GB)
    ///
    /// Leaves 500MB headroom below browser's ~2GB limit.
    pub fn new() -> Self {
        Self::with_limit(1_500_000_000) // 1.5GB
    }

    /// Create memory budget with custom limit
    pub fn with_limit(limit: usize) -> Self {
        console::log_1(&format!("Memory budget: {} MB", limit / 1_000_000).into());

        Self { limit, current: Arc::new(AtomicUsize::new(0)) }
    }

    /// Try to allocate memory
    ///
    /// # Errors
    ///
    /// Returns error if allocation would exceed budget.
    ///
    /// # Toyota Way: Poka-Yoke
    ///
    /// Prevents memory allocation that would cause browser OOM.
    pub fn try_allocate(&self, size: usize) -> Result<MemoryAllocation, JsValue> {
        let old_total = self.current.fetch_add(size, Ordering::SeqCst);
        let new_total = old_total + size;

        if new_total > self.limit {
            // Rollback allocation
            self.current.fetch_sub(size, Ordering::SeqCst);

            return Err(JsValue::from_str(&format!(
                "Memory budget exceeded: {} MB used, tried to allocate {} MB, limit {} MB",
                old_total / 1_000_000,
                size / 1_000_000,
                self.limit / 1_000_000
            )));
        }

        console::log_1(
            &format!(
                "Memory: {} MB / {} MB ({:.1}%)",
                new_total / 1_000_000,
                self.limit / 1_000_000,
                (new_total as f64 / self.limit as f64) * 100.0
            )
            .into(),
        );

        Ok(MemoryAllocation { size, budget: self.clone() })
    }

    /// Get current memory usage in bytes
    pub fn current_usage(&self) -> usize {
        self.current.load(Ordering::SeqCst)
    }

    /// Get memory limit in bytes
    pub fn limit(&self) -> usize {
        self.limit
    }

    /// Get available memory in bytes
    pub fn available(&self) -> usize {
        self.limit.saturating_sub(self.current_usage())
    }
}

impl Default for MemoryBudget {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII guard for memory allocation
///
/// Automatically deallocates on drop (prevents memory leaks).
pub struct MemoryAllocation {
    size: usize,
    budget: MemoryBudget,
}

impl Drop for MemoryAllocation {
    fn drop(&mut self) {
        self.budget.current.fetch_sub(self.size, Ordering::SeqCst);
    }
}

/// Late materialization query executor
///
/// Defers row reconstruction until after filtering and aggregation.
///
/// # Strategy
///
/// 1. Read only filter columns
/// 2. Compute matching row indices
/// 3. Read only needed columns for matching rows
/// 4. Materialize results
///
/// # Memory Benefits
///
/// - Reduces data transfer (HTTP range requests)
/// - Reduces memory usage (only matching rows)
/// - Enables large dataset queries in browser
pub struct LateMaterializationExecutor {
    /// Memory budget enforcer
    budget: MemoryBudget,
}

impl LateMaterializationExecutor {
    /// Create new executor with memory budget
    pub fn new(budget: MemoryBudget) -> Self {
        Self { budget }
    }

    /// Execute query with late materialization
    ///
    /// # Example
    ///
    /// ```ignore
    /// let executor = LateMaterializationExecutor::new(MemoryBudget::new());
    ///
    /// // Query: SELECT name, revenue FROM sales WHERE price > 100
    /// let filter_col = reader.read_column("price").await?;
    /// let indices = executor.filter_indices(&filter_col, |&v| v > 100)?;
    ///
    /// // Only read needed columns for matching rows
    /// let results = executor.select_by_indices(
    ///     &reader,
    ///     &["name", "revenue"],
    ///     &indices
    /// ).await?;
    /// ```
    pub fn filter_indices<T, F>(&self, column: &[T], predicate: F) -> Result<Vec<usize>, JsValue>
    where
        F: Fn(&T) -> bool,
    {
        // Allocate memory for indices (worst case: all rows match)
        let max_indices_size = column.len() * std::mem::size_of::<usize>();
        let _allocation = self.budget.try_allocate(max_indices_size)?;

        let indices: Vec<usize> = column
            .iter()
            .enumerate()
            .filter_map(|(idx, val)| if predicate(val) { Some(idx) } else { None })
            .collect();

        console::log_1(
            &format!(
                "Filter selectivity: {:.2}% ({} / {} rows)",
                (indices.len() as f64 / column.len() as f64) * 100.0,
                indices.len(),
                column.len()
            )
            .into(),
        );

        Ok(indices)
    }

    /// Get current memory usage statistics
    pub fn memory_stats(&self) -> MemoryStats {
        MemoryStats {
            used_bytes: self.budget.current_usage(),
            limit_bytes: self.budget.limit(),
            available_bytes: self.budget.available(),
            usage_percent: (self.budget.current_usage() as f64 / self.budget.limit() as f64)
                * 100.0,
        }
    }
}

impl Default for LateMaterializationExecutor {
    fn default() -> Self {
        Self::new(MemoryBudget::default())
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Bytes currently used
    pub used_bytes: usize,

    /// Memory limit in bytes
    pub limit_bytes: usize,

    /// Available memory in bytes
    pub available_bytes: usize,

    /// Usage as percentage (0-100)
    pub usage_percent: f64,
}

impl MemoryStats {
    /// Format for human-readable display
    pub fn format(&self) -> String {
        format!(
            "{:.1} MB / {:.1} MB ({:.1}%)",
            self.used_bytes as f64 / 1_000_000.0,
            self.limit_bytes as f64 / 1_000_000.0,
            self.usage_percent
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_budget_new() {
        let budget = MemoryBudget::new();
        assert_eq!(budget.limit(), 1_500_000_000);
        assert_eq!(budget.current_usage(), 0);
        assert_eq!(budget.available(), 1_500_000_000);
    }

    #[test]
    fn test_memory_budget_with_limit() {
        let budget = MemoryBudget::with_limit(1_000_000);
        assert_eq!(budget.limit(), 1_000_000);
    }

    #[test]
    fn test_memory_allocation_success() {
        let budget = MemoryBudget::with_limit(1_000_000);
        let alloc = budget.try_allocate(500_000);
        assert!(alloc.is_ok());
        assert_eq!(budget.current_usage(), 500_000);
    }

    #[test]
    fn test_memory_allocation_exceeds_budget() {
        let budget = MemoryBudget::with_limit(1_000_000);
        let result = budget.try_allocate(1_500_000);
        assert!(result.is_err());
        assert_eq!(budget.current_usage(), 0); // Rolled back
    }

    #[test]
    fn test_memory_allocation_raii() {
        let budget = MemoryBudget::with_limit(1_000_000);
        {
            let _alloc = budget.try_allocate(500_000).unwrap();
            assert_eq!(budget.current_usage(), 500_000);
        } // alloc dropped here
        assert_eq!(budget.current_usage(), 0); // Deallocated
    }

    #[test]
    fn test_filter_indices_basic() {
        let executor = LateMaterializationExecutor::default();
        let data = vec![10, 20, 30, 40, 50];
        let indices = executor.filter_indices(&data, |&v| v > 25).unwrap();
        assert_eq!(indices, vec![2, 3, 4]);
    }

    #[test]
    fn test_filter_indices_selectivity() {
        let executor = LateMaterializationExecutor::default();
        let data: Vec<i32> = (0..1000).collect();
        let indices = executor.filter_indices(&data, |&v| v > 900).unwrap();
        assert_eq!(indices.len(), 99); // 901..999 inclusive
    }

    #[test]
    fn test_memory_stats() {
        let budget = MemoryBudget::with_limit(2_000_000_000);
        let _alloc = budget.try_allocate(500_000_000).unwrap();

        let executor = LateMaterializationExecutor::new(budget);
        let stats = executor.memory_stats();

        assert_eq!(stats.used_bytes, 500_000_000);
        assert_eq!(stats.limit_bytes, 2_000_000_000);
        assert_eq!(stats.available_bytes, 1_500_000_000);
        assert!((stats.usage_percent - 25.0).abs() < 0.1);
    }

    #[test]
    fn test_memory_stats_format() {
        let stats = MemoryStats {
            used_bytes: 750_000_000,
            limit_bytes: 1_500_000_000,
            available_bytes: 750_000_000,
            usage_percent: 50.0,
        };

        let formatted = stats.format();
        assert!(formatted.contains("750"));
        assert!(formatted.contains("1500"));
        assert!(formatted.contains("50"));
    }
}
