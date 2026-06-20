//! Shared test-support for env-mutating tests (PMAT-876).
//!
//! `std::env` is **process-global**: a single environment table is
//! shared by every thread in the process. `cargo test` / nextest run
//! `#[test]` functions in PARALLEL by default, so two tests that each
//! `set_var` / `remove_var` the *same* variable (or one mutates while
//! another reads) interleave non-deterministically and observe the
//! wrong value. This produced an intermittent `workspace-test` failure
//! (`agent::auto_memory::tests::root_uses_config_dir_when_env_unset`
//! and siblings) that repeatedly broke the merge queue.
//!
//! ## Why a single shared lock
//!
//! Historically `auto_memory`, `settings`, and `instructions` each
//! defined their *own* module-private `Mutex` and acquired it in their
//! own tests. That serializes tests *within* one module but NOT across
//! modules — yet all three mutate the **same** `APR_CONFIG` variable.
//! Two tests in different modules therefore still raced. The fix is one
//! crate-wide [`ENV_LOCK`]: every test that touches process env acquires
//! THIS lock, so all env-mutating tests across the whole crate are
//! serialized against each other while non-env tests stay parallel.
//!
//! ## Usage
//!
//! ```ignore
//! use crate::agent::env_test_support::env_lock;
//!
//! #[test]
//! fn my_env_test() {
//!     let _guard = env_lock();              // held for the whole test
//!     let _restore = ScopedEnv::set("APR_CONFIG", "/tmp/x");
//!     // ... assert against the env-dependent behavior ...
//! }   // _restore drops → prior value restored; _guard drops → lock freed
//! ```

#![cfg(test)]

use std::sync::{Mutex, MutexGuard};

/// Process-wide lock serializing every env-mutating test in the crate.
///
/// Poison-tolerant: a panicking test would poison the mutex, but the
/// lock itself protects no data we care about (it only serializes
/// access to the global env table), so we recover the guard via
/// `into_inner()` rather than propagating the poison.
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Acquire the crate-wide env lock for the duration of a test.
///
/// Hold the returned guard across the full set → assert → restore
/// sequence so no other env-mutating test can interleave.
pub fn env_lock() -> MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// RAII guard that sets an env var and restores its prior value (or
/// absence) on drop. Combined with [`env_lock`], this keeps the global
/// env table hermetic: a test never leaves a variable mutated for the
/// next test to observe.
#[must_use = "the prior env value is restored when this guard is dropped"]
pub struct ScopedEnv {
    key: String,
    prior: Option<String>,
}

impl ScopedEnv {
    /// Save the current value of `key`, then set it to `value`.
    pub fn set(key: &str, value: impl AsRef<std::ffi::OsStr>) -> Self {
        let prior = std::env::var(key).ok();
        std::env::set_var(key, value);
        Self { key: key.to_string(), prior }
    }

    /// Save the current value of `key`, then remove it.
    pub fn remove(key: &str) -> Self {
        let prior = std::env::var(key).ok();
        std::env::remove_var(key);
        Self { key: key.to_string(), prior }
    }
}

impl Drop for ScopedEnv {
    fn drop(&mut self) {
        match &self.prior {
            Some(v) => std::env::set_var(&self.key, v),
            None => std::env::remove_var(&self.key),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoped_env_set_restores_prior_absence() {
        let _guard = env_lock();
        // Pick a name unlikely to exist in CI.
        let key = "PMAT_876_SCOPED_PROBE_A";
        std::env::remove_var(key); // ensure absent baseline
        {
            let _s = ScopedEnv::set(key, "value");
            assert_eq!(std::env::var(key).as_deref(), Ok("value"));
        }
        // Restored to absence.
        assert!(std::env::var(key).is_err());
    }

    #[test]
    fn scoped_env_remove_restores_prior_value() {
        let _guard = env_lock();
        let key = "PMAT_876_SCOPED_PROBE_B";
        std::env::set_var(key, "original");
        {
            let _s = ScopedEnv::remove(key);
            assert!(std::env::var(key).is_err());
        }
        // Restored to the original value.
        assert_eq!(std::env::var(key).as_deref(), Ok("original"));
        std::env::remove_var(key); // clean up
    }

    #[test]
    fn env_lock_is_reentrant_safe_across_calls() {
        // Sequential acquisition must not deadlock (guard dropped between).
        {
            let _g = env_lock();
        }
        let _g2 = env_lock();
    }
}
