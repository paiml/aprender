//! Integration test root module.
//!
//! This file enables Cargo to discover tests in the integration/ subdirectory.

mod integration {
    mod lifecycle_chaos_tests;
    mod null_fuzzer_tests;
    mod ptx_poison_tests;
    mod quant_oracle_tests;
    mod shmem_prober_tests;
}
