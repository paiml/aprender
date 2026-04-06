//! trueno-ublk - GPU-accelerated ZRAM replacement using ublk
//!
//! This crate provides a ublk-based block device with SIMD and GPU-accelerated
//! compression. It can be used as a drop-in replacement for Linux kernel ZRAM.
//!
//! # Features
//!
//! - **SIMD acceleration**: AVX2, AVX-512, and NEON support via trueno-zram-core
//! - **GPU offload**: CUDA-based batch compression for large workloads
//! - **Entropy routing**: Automatic algorithm selection based on data entropy
//! - **Zero-page deduplication**: Efficient handling of all-zero pages
//! - **zram-compatible stats**: Statistics format compatible with kernel zram
//!
//! # Example
//!
//! ```no_run
//! use trueno_ublk::BlockDevice;
//! use trueno_zram_core::{Algorithm, CompressorBuilder, PAGE_SIZE};
//!
//! // Create a compressor
//! let compressor = CompressorBuilder::new()
//!     .algorithm(Algorithm::Lz4)
//!     .build()
//!     .unwrap();
//!
//! // Create a 1GB block device
//! let mut device = BlockDevice::new(1 << 30, compressor);
//!
//! // Write data
//! let data = vec![0xAB; PAGE_SIZE];
//! device.write(0, &data).unwrap();
//!
//! // Read back
//! let mut buf = vec![0u8; PAGE_SIZE];
//! device.read(0, &mut buf).unwrap();
//! assert_eq!(data, buf);
//!
//! // Check stats
//! let stats = device.stats();
//! println!("Compression ratio: {:.2}x", stats.compression_ratio());
//! ```

#![allow(dead_code, unused_imports)]
#![allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#![allow(unexpected_cfgs)]
#![allow(clippy::borrow_as_ptr)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_ptr_alignment)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::default_trait_access)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::explicit_auto_deref)]
#![allow(clippy::explicit_iter_loop)]
#![allow(clippy::if_not_else)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::manual_is_variant_and)]
#![allow(clippy::map_entry)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::needless_bool)]
#![allow(clippy::needless_continue)]
#![allow(clippy::needless_lifetimes)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::option_if_let_else)]
#![allow(clippy::ptr_as_ptr)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::ref_as_ptr)]
#![allow(clippy::semicolon_if_nothing_returned)]
#![allow(clippy::similar_names)]
#![allow(clippy::single_match_else)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::unnested_or_patterns)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::unused_self)]
#![allow(clippy::used_underscore_binding)]
#![allow(clippy::used_underscore_items)]
#![allow(clippy::wildcard_imports)]
#![allow(clippy::elidable_lifetime_names)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::unnecessary_unwrap)]
#![allow(clippy::zero_sized_map_values)]

pub mod backend; // KERN-001: Kernel-cooperative tiered storage
pub mod cleanup;
pub mod daemon;
pub mod device;
pub mod duende_lifecycle; // DT-008: Duende daemon lifecycle management
pub mod perf;
pub mod stats;
pub mod ublk;
pub mod visualize; // VIZ-001: Renacer integration

// Re-export commonly used types
pub use backend::{BackendType, EntropyThresholds, StorageBackend, TieredStorageManager};
pub use device::{BlockDevice, BlockDeviceStats, DeviceConfig, DeviceStats, UblkDevice};
pub use ublk::{DaemonError, UblkCtrl, UblkDaemon};

// Re-export duende-mlock for swap deadlock prevention (DT-007)
pub use duende_mlock::{
    is_locked as is_memory_locked, lock_all as lock_daemon_memory, MlockStatus,
};

// Re-export duende-core lifecycle management (DT-008)
pub use duende_lifecycle::{create_daemon, setup_duende_signals, TruenoUblkDaemon};
#[cfg(not(test))]
pub use ublk::{run_daemon, run_daemon_batched, BatchedDaemonConfig};

// Re-export batched page store for direct use
pub use daemon::{spawn_flush_thread, BatchConfig, BatchedPageStore, BatchedPageStoreStats};

// Re-export performance optimization module (PERF-001)
// Note: Additional types (BatchCoalescer, NumaAllocator, etc.) are available
// in perf submodules for future PERF-001 integration
pub use perf::{HiPerfContext, PerfConfig, PollResult, PollingConfig, TenXConfig, TenXContext};

// Re-export visualization module (VIZ-001)
pub use visualize::TruenoCollector;
