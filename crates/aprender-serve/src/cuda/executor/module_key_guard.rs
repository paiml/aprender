//! #3759: one module-cache key, one PTX text.
//!
//! Most kernels bake their parameters into the PTX as immediates (epsilon, shapes, rope theta,
//! scales), and the executor caches the compiled module under a key the call site builds by
//! hand. A key that leaves out a baked parameter hands every later request with a different
//! value the first request's kernel. That happened twice in one day: the FP8 activation cache
//! (#3727), and RMSNorm keyed by shape alone while preload compiled it at a hardcoded 1e-5, so a
//! model with epsilon 1e-6 ran at 1e-5 and its special tokens came out of RMSNorm at 0.431x.
//!
//! `ensure_kernel_module` is the compile-on-miss step for a `KernelType`. In debug builds and in
//! every `cargo test` build (`cfg(test)`, so `--release` test runs too) it proves the key is complete. It records the hash of the PTX compiled under each key; a later
//! hit whose `KernelType` differs from every request already proven equivalent is regenerated
//! and must hash the same, or it panics naming both requests. Grid-only differences (the same
//! PTX) pass, so the check has no false positives and costs one string comparison per repeated
//! launch. A release build of the library (what ships) does only the lookup.

use super::CudaExecutor;
use crate::cuda::KernelType;
use trueno_gpu::GpuError;

/// Debug/test builds: what each key was compiled from, and which requests are proven equivalent.
#[cfg(any(debug_assertions, test))]
#[derive(Default)]
pub(crate) struct ModuleKeyLedger {
    /// key -> (hash of the PTX compiled under it, `Debug` of every request proven to produce it)
    entries: std::collections::HashMap<String, (u64, Vec<String>)>,
}

#[cfg(any(debug_assertions, test))]
fn ptx_hash(ptx: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    ptx.hash(&mut h);
    h.finish()
}

impl CudaExecutor {
    /// Compile `kernel_type` under `key` unless a module is already cached there.
    ///
    /// Debug/test builds panic if `key` already holds a module compiled from different PTX:
    /// the key is missing a parameter the kernel bakes in (#3759).
    pub(crate) fn ensure_kernel_module(
        &mut self,
        key: &str,
        kernel_type: &KernelType,
    ) -> Result<(), GpuError> {
        if self.modules.contains_key(key) {
            #[cfg(any(debug_assertions, test))]
            self.prove_module_key_complete(key, kernel_type);
            return Ok(());
        }
        let ptx = self.kernels.generate_ptx(kernel_type);
        #[cfg(any(debug_assertions, test))]
        self.module_key_ledger.entries.insert(
            key.to_string(),
            (ptx_hash(&ptx), vec![format!("{kernel_type:?}")]),
        );
        let module = self.compile_ptx(&ptx)?;
        self.modules.insert(key.to_string(), module);
        Ok(())
    }

    #[cfg(any(debug_assertions, test))]
    fn prove_module_key_complete(&mut self, key: &str, kernel_type: &KernelType) {
        let request = format!("{kernel_type:?}");
        // A key compiled outside this helper has no record: nothing to compare against.
        let Some((_, proven)) = self.module_key_ledger.entries.get(key) else {
            return;
        };
        if proven.contains(&request) {
            return;
        }
        let hash = ptx_hash(&self.kernels.generate_ptx(kernel_type));
        let (compiled, proven) = self
            .module_key_ledger
            .entries
            .get_mut(key)
            .expect("checked above");
        assert!(
            hash == *compiled,
            "module key `{key}` is incomplete: it holds a kernel compiled for {} and is now asked \
             for {request}, whose PTX differs. A parameter baked into the PTX is missing from the \
             key (#3759).",
            proven[0]
        );
        proven.push(request);
    }
}
