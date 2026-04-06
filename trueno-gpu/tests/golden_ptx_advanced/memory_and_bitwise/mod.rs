//! Global memory operations, type conversions (various widths), shift/bitwise
//! operations, select, inplace variants, register moves, comparisons,
//! warp shuffle, multiply variants, min/max, const helpers, and shared pointer.
//!
//! IMMUTABLE GUARDIAN - DO NOT MODIFY WITHOUT FALSIFICATION EVIDENCE

use trueno_gpu::ptx::{PtxComparison, PtxControl, PtxKernel, PtxReg, PtxType};

mod global_mem_and_types;
mod operations_and_helpers;
