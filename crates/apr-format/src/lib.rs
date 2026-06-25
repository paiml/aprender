// Tests legitimately use expect/unwrap/panic and exact-float asserts on known
// values; mirror the workspace convention of allowing these in test code only.
#![cfg_attr(
    test,
    allow(
        clippy::expect_used,
        clippy::unwrap_used,
        clippy::panic,
        clippy::float_cmp
    )
)]

//! # apr-format — sovereign `.apr` model container format
//!
//! Minimal, dependency-light read/write for the `.apr` model container (v1
//! `APRN` + v2 `APR\0`), extracted from `aprender-core` so that downstream
//! consumers (realizar inference, xpile, external tooling) can read and write
//! `.apr` files **without** pulling the full ML/GPU/tokenizer/quantization stack.
//! See issue #2231 — "depend on the *format*, not the *framework*."
//!
//! ## Status: Stage 1 foundation + cut-feasibility spike
//!
//! This crate currently ships:
//! - The sovereign error seam ([`error::AprFormatError`], wrapped by
//!   `aprender-core` via `impl From`).
//! - The single deduplicated [`crc32::crc32`] and [`f16`] conversions.
//! - A representative v1 (`APRN`) slice — [`types`] (header/metadata/flags) and
//!   [`core_io`] (save/load) — proving the error-seam compiles at a crate
//!   boundary.
//! - The byte-only structural validator split ([`validate`]), demonstrating the
//!   Structure-vs-Physics separation.
//!
//! The bulk `git-mv` of the rest of `format/` (v2 container, mmap, spec, …) lands
//! in Stage 2.
//!
//! ## Locked design decisions (issue #2231)
//! 1. The GGUF/SafeTensors/ONNX **converter stays in `aprender-core`** (it needs
//!    `f32` physics + the ML stack); only the container moves.
//! 2. **std-only** for v1 (no `no_std` yet — the std surface is kept thin).
//! 3. **Wrapper error seam**: the leaf owns `AprFormatError`; core From-wraps it.
//! 4. **mmap is feature-gated** (`mmap` feature, off by default).

pub mod core_io;
pub mod crc32;
pub mod error;
pub mod f16;
pub mod falsifiers;
pub mod types;
pub mod validate;

// --- Convenience re-exports (the public surface aprender-core re-exports) ---
pub use core_io::{load, load_from_bytes, save};
pub use crc32::crc32;
pub use error::{AprFormatError, Result};
pub use f16::{f16_to_f32, f32_to_f16};
pub use types::{
    Compression, Flags, Header, LicenseInfo, LicenseTier, Metadata, ModelType, SaveOptions,
    TrainingInfo, FORMAT_VERSION, HEADER_SIZE, MAGIC, MAX_UNCOMPRESSED_SIZE,
};
pub use validate::{validate_structure, StructureCheck};

#[cfg(test)]
mod sovereignty_tests {
    /// FALSIFY-APRF-SOV-STD-ONLY: the leaf is std-only (v1) — it links and runs
    /// against std (fs/io save+load round-trip works), and carries no `#![no_std]`.
    /// `no_std` is an explicit deferred decision; this pins that the std surface
    /// stays available so the deferral cannot silently regress into a half-`no_std`
    /// state. (A genuine `no_std` build would fail to link `std::fs::File` here.)
    #[test]
    fn test_std_only_surface_available() {
        use crate::types::{ModelType, SaveOptions};
        let dir = std::env::temp_dir();
        let path = dir.join("apr_format_std_probe.apr");
        // Uses std::fs via the leaf's save/load — proves the std surface links.
        crate::save(
            &vec![1.0_f32, 2.0],
            ModelType::LinearRegression,
            &path,
            SaveOptions::default(),
        )
        .expect("std-only save must work");
        let back: Vec<f32> =
            crate::load(&path, ModelType::LinearRegression).expect("std-only load must work");
        assert_eq!(back, vec![1.0, 2.0]);
        let _ = std::fs::remove_file(&path);
    }
}
