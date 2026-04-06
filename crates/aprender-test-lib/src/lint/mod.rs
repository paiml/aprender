//! Linting Module for WASM State Synchronization and Panic Paths
//!
//! Implements static analysis for detecting anti-patterns per PROBAR specs.
//!
//! ## State Sync Rules (PROBAR-SPEC-WASM-001)
//!
//! | Rule ID | Description |
//! |---------|-------------|
//! | WASM-SS-001 | Local `Rc::new()` in method with closure |
//! | WASM-SS-002 | Potential state desync (self.field and local_clone both exist) |
//! | WASM-SS-003 | Closure captures local instead of self field |
//! | WASM-SS-004 | Duplicate state fields (RefCell + non-RefCell) |
//! | WASM-SS-005 | Missing `self.*.clone()` before closure |
//! | WASM-SS-006 | Type alias for Rc (including turbofish `Alias::<T>::new()`) |
//! | WASM-SS-007 | Function returning Rc captured in closure |
//! | WASM-SS-008 | Method chain returning Rc (`.to_rc()`, etc.) |
//!
//! ## Panic Path Rules (PROBAR-WASM-006)
//!
//! | Rule ID | Description | Severity |
//! |---------|-------------|----------|
//! | WASM-PANIC-001 | `unwrap()` call | Error |
//! | WASM-PANIC-002 | `expect()` call | Error |
//! | WASM-PANIC-003 | `panic!()` macro | Error |
//! | WASM-PANIC-004 | `unreachable!()` macro | Warning |
//! | WASM-PANIC-005 | `todo!()` macro | Error |
//! | WASM-PANIC-006 | `unimplemented!()` macro | Error |
//! | WASM-PANIC-007 | Index access without bounds check | Warning |
//!
//! ## AST vs Text-Based Analysis
//!
//! The linter supports two modes:
//! - **AST-based** (`ast_visitor`): Uses `syn` crate for accurate parsing
//! - **Text-based** (`state_sync`): Legacy fallback for edge cases
//!
//! AST-based analysis is preferred as it handles:
//! - Turbofish syntax (`Type::<T>::new()`)
//! - Alternative constructors (`Rc::default()`, `Rc::from()`)
//! - Method chains (`.to_rc()`)
//! - Unusual whitespace/formatting

pub mod ast_visitor;
pub mod panic_paths;
pub mod state_sync;

pub use ast_visitor::{lint_source_ast, AstStateSyncVisitor};
pub use panic_paths::{lint_panic_paths, PanicPathSummary, PanicPathVisitor};
pub use state_sync::{LintError, LintResult, LintSeverity, StateSyncLinter, StateSyncReport};
