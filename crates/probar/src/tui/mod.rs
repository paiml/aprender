//! TUI Testing Support (Feature 21 - EDD Compliance)
//!
//! Test terminal user interfaces with zero external dependencies.
//! Uses custom TextGrid and MockTty for 100% provable UX.
//!
//! ## EXTREME TDD: Tests written FIRST per spec
//!
//! ## Toyota Way Application
//!
//! - **Poka-Yoke**: Type-safe frame capture with TextGrid
//! - **Muda**: Direct buffer comparison without rendering overhead
//! - **Genchi Genbutsu**: MockTty reflects actual terminal behavior
//! - **Jidoka**: Fail-fast on frame mismatch
//!
//! ## ComputeBlock Testing (PROBAR-SPEC-009)
//!
//! With the `compute-blocks` feature, probar supports testing presentar-terminal
//! ComputeBlocks with SIMD verification and latency budget assertions.
//!
//! ```ignore
//! use jugar_probar::tui::{ComputeBlockAssertion, assert_brick_valid};
//! use presentar_terminal::SparklineBlock;
//!
//! let block = SparklineBlock::new(60);
//! assert_brick_valid(&block).unwrap();
//!
//! ComputeBlockAssertion::new(&block)
//!     .to_have_simd_support()
//!     .to_have_latency_under(100);
//! ```

mod assertions;
mod backend;
mod buffer;
mod snapshot;
mod tty;

// Brick and ComputeBlock testing (optional, requires presentar-terminal)
#[cfg(feature = "compute-blocks")]
mod brick;
#[cfg(feature = "compute-blocks")]
mod compute_block;

pub use assertions::{expect_frame, FrameAssertion, MultiValueTracker, ValueTracker};
pub use backend::{FrameDiff, LineDiff, TuiFrame, TuiTestBackend};
pub use buffer::TextGrid;
pub use snapshot::{FrameSequence, SnapshotManager, TuiSnapshot};
pub use tty::{AnsiCommand, ClearMode, MockTty};

// Re-export Brick testing utilities
#[cfg(feature = "compute-blocks")]
pub use brick::{
    assert_brick_budget, assert_brick_valid, brick_verification_score, BrickAssertionResult,
    BrickTestAssertion, BrickVerificationError, BudgetExceededError,
};

// Re-export ComputeBlock testing utilities
#[cfg(feature = "compute-blocks")]
pub use compute_block::{
    assert_compute_latency, assert_simd_available, detect_simd, simd_available,
    ComputeBlockAssertion, LatencyBudgetError, SimdNotAvailableError,
};

// Re-export presentar-terminal types for convenience
#[cfg(feature = "compute-blocks")]
pub use presentar_terminal::{
    ComputeBlock, ComputeBlockId, CpuFrequencyBlock, CpuGovernor, CpuGovernorBlock,
    FrequencyScalingState, GpuThermalBlock, GpuThermalState, GpuVramBlock, HugePagesBlock,
    LoadTrendBlock, MemPressureBlock, MemoryPressureLevel, SimdInstructionSet, SparklineBlock,
};

// Note: Brick, BrickAssertion, BrickBudget, BrickVerification are exported
// from the main jugar_probar::brick module (probar defines the trait, presentar uses it)
