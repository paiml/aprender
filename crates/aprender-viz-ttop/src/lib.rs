//! ttop v2: Terminal Top - Sovereign AI Stack System Monitor
//!
//! Built on presentar-terminal (no ratatui dependency).
//! Zero-allocation steady-state rendering via CellBuffer + DiffRenderer.
//!
//! ## Architecture
//!
//! ```text
//! ttop binary
//!   └── presentar-terminal::ptop (panels, app, ui)
//!         └── presentar-terminal::direct (CellBuffer, DiffRenderer)
//!               └── crossterm (terminal I/O)
//! ```
//!
//! ## Sovereign Stack
//!
//! All rendering is PAIML-owned. The only external terminal dependency is crossterm.
//! Contracts enforced via provable-contracts YAML definitions in `contracts/`.

#[macro_use]
#[allow(unused_macros)]
mod generated_contracts;

// =============================================================================
// SPEC-024 ARCHITECTURAL ENFORCEMENT
// =============================================================================
// These constants require test files to exist at compile time.
// If a test file is deleted, the build FAILS.
// Tests define the interface. Implementation follows.

/// ENFORCEMENT: Brick interface tests MUST exist
#[doc(hidden)]
pub const _ENFORCE_BRICK_TESTS: &str = include_str!("../tests/brick_interface.rs");

/// ENFORCEMENT: Falsification tests MUST exist
#[doc(hidden)]
pub const _ENFORCE_FALSIFICATION: &str = include_str!("../tests/panel_falsification.rs");

/// ENFORCEMENT: Pixel parity tests MUST exist
#[doc(hidden)]
pub const _ENFORCE_PIXEL_PARITY: &str = include_str!("../tests/pixel_parity.rs");

// Re-export presentar-terminal ptop types for testing and embedding
pub use presentar_terminal::direct::{CellBuffer, DiffRenderer};
pub use presentar_terminal::ptop::app::App;
pub use presentar_terminal::ptop::config::PtopConfig;
pub use presentar_terminal::ptop::ui;
pub use presentar_terminal::ptop::MetricsSnapshot;
pub use presentar_terminal::ptop::PanelType;
pub use presentar_terminal::ColorMode;
