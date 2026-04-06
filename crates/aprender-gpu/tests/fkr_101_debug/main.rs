//! FKR-101-DEBUG: Instrumented Non-Zero Page Test
//!
//! This test instruments the LZ4 kernel with debug markers to find
//! the exact crash location in the compression loop.
//!
//! Debug markers:
//! 0x01000000 - Loop entry
//! 0x02000000 - After state load
//! 0x03000000 - After hash compute
//! 0x04000000 - After hash table lookup
//! 0x05000000 - At L_found_match
//! 0x06000000 - At L_copy_lit_loop
//! 0x07000000 - At L_no_match
//! 0x08000000 - At L_emit_remaining

#[cfg(feature = "cuda")]
mod common;

#[cfg(feature = "cuda")]
mod minimal_tests;

#[cfg(feature = "cuda")]
mod param_tests;

#[cfg(feature = "cuda")]
mod setup_tests;

#[cfg(feature = "cuda")]
mod advanced_tests;

#[cfg(not(feature = "cuda"))]
mod no_cuda {
    #[test]
    fn fkr_101_debug_skip_no_cuda() {
        println!("FKR-101-DEBUG: Skipped - CUDA feature not enabled");
    }
}
