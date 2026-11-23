//! CUDA-independent lazy kernel-module cache (C-LPTX-001, KAIZEN-058).
//!
//! The forward, backward and optimizer kernel caches all share one policy:
//! a hit is a pure map lookup (no PTX generation, no compile), a miss compiles
//! once and every later call hits, and PTX is rejected when its `.target` does
//! not match the device (F-PTX-001). That policy does not need a GPU, so it
//! lives here, ungated, and runs in every `--lib` test. The actual CUDA module
//! load stays in the `cuda`-gated callers' closures and is NOT covered here.

use std::collections::hash_map::Entry;
use std::collections::HashMap;

/// Map of kernel key to compiled module, with a count of compiles (misses).
#[derive(Debug)]
pub struct LazyModuleCache<M> {
    modules: HashMap<String, M>,
    jit_compiles: u64,
}

impl<M> Default for LazyModuleCache<M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<M> LazyModuleCache<M> {
    /// Empty cache, zero compiles.
    pub fn new() -> Self {
        Self { modules: HashMap::new(), jit_compiles: 0 }
    }

    /// Hit path: pure `HashMap::get_mut`. Never compiles, never counts.
    pub fn get_cached(&mut self, name: &str) -> Option<&mut M> {
        self.modules.get_mut(name)
    }

    /// Return the cached module, or run `compile` once on a miss and insert it.
    ///
    /// A miss counts one compile before `compile` runs. If `compile` fails the
    /// error is returned and nothing is inserted, so the next call retries.
    pub fn get_or_compile_with<E>(
        &mut self,
        name: &str,
        compile: impl FnOnce() -> Result<M, E>,
    ) -> Result<&mut M, E> {
        match self.modules.entry(name.to_string()) {
            Entry::Occupied(e) => Ok(e.into_mut()),
            Entry::Vacant(e) => {
                self.jit_compiles += 1;
                Ok(e.insert(compile()?))
            }
        }
    }

    /// Compiles (misses) since construction or the last reset.
    pub fn jit_compiles(&self) -> u64 {
        self.jit_compiles
    }

    /// Zero the compile counter (the cached modules stay).
    pub fn reset_jit_counter(&mut self) {
        self.jit_compiles = 0;
    }
}

/// F-PTX-001: reject PTX whose `.target` directive differs from the device's.
///
/// PTX with no `.target` line is accepted (nothing to compare).
pub fn check_ptx_target(ptx: &str, sm_target: &str) -> Result<(), String> {
    let Some(line) = ptx.lines().find(|l| l.starts_with(".target ")) else {
        return Ok(());
    };
    let ptx_target = line.trim().trim_start_matches(".target ");
    if ptx_target == sm_target {
        Ok(())
    } else {
        Err(format!("F-PTX-001 violated: PTX target '{ptx_target}' != device target '{sm_target}'"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hit_does_not_compile() {
        let mut c: LazyModuleCache<u32> = LazyModuleCache::new();
        c.get_or_compile_with("k", || Ok::<_, String>(7)).expect("first compile");
        let mut called = false;
        let m = c
            .get_or_compile_with("k", || {
                called = true;
                Ok::<_, String>(99)
            })
            .expect("hit");
        assert_eq!(*m, 7);
        assert!(!called, "hit must not invoke compile");
        assert_eq!(c.jit_compiles(), 1);
    }

    #[test]
    fn get_cached_is_pure_lookup() {
        let mut c: LazyModuleCache<u32> = LazyModuleCache::new();
        assert!(c.get_cached("k").is_none());
        assert_eq!(c.jit_compiles(), 0);
        c.get_or_compile_with("k", || Ok::<_, String>(1)).expect("compile");
        assert_eq!(c.get_cached("k").copied(), Some(1));
        assert_eq!(c.jit_compiles(), 1);
    }

    #[test]
    fn miss_compiles_once_then_hits() {
        let mut c: LazyModuleCache<u32> = LazyModuleCache::new();
        let mut calls = 0;
        for _ in 0..3 {
            c.get_or_compile_with("k", || {
                calls += 1;
                Ok::<_, String>(5)
            })
            .expect("ok");
        }
        assert_eq!(calls, 1);
        assert_eq!(c.jit_compiles(), 1);
        c.reset_jit_counter();
        assert_eq!(c.jit_compiles(), 0);
        assert!(c.get_cached("k").is_some());
    }

    #[test]
    fn compile_error_propagates_and_inserts_nothing() {
        let mut c: LazyModuleCache<u32> = LazyModuleCache::new();
        let err = c.get_or_compile_with("k", || Err::<u32, _>("boom".to_string()));
        assert_eq!(err.err().as_deref(), Some("boom"));
        assert!(c.get_cached("k").is_none());
        c.get_or_compile_with("k", || Ok::<_, String>(2)).expect("retry compiles");
        assert_eq!(c.get_cached("k").copied(), Some(2));
    }

    #[test]
    fn target_mismatch_rejected() {
        let ptx = ".version 8.0\n.target sm_80\n.address_size 64\n";
        let e = check_ptx_target(ptx, "sm_89").expect_err("mismatch");
        assert!(e.contains("F-PTX-001") && e.contains("sm_80") && e.contains("sm_89"));
    }

    #[test]
    fn target_match_or_absent_accepted() {
        assert!(check_ptx_target(".version 8.0\n.target sm_89\n", "sm_89").is_ok());
        assert!(check_ptx_target(".version 8.0\n.address_size 64\n", "sm_89").is_ok());
    }
}
