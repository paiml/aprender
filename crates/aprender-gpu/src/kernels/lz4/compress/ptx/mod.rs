//! PTX code generation for the LZ4 warp-cooperative compression kernel
//!
//! ## Submodules
//!
//! - [`build_ptx`]: Full PTX kernel generation with cooperative load,
//!   zero-page detection, LZ4 compression loop, and output phases

mod build_ptx;
