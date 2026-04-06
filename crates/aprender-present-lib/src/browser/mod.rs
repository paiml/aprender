//! Browser runtime for Presentar WASM applications.
//!
//! This module provides the bridge between Presentar's widget system
//! and the browser's rendering APIs (Canvas2D, WebGPU).

// WASM-only modules
#[cfg(target_arch = "wasm32")]
pub mod app;
#[cfg(target_arch = "wasm32")]
pub mod canvas2d;
#[cfg(target_arch = "wasm32")]
pub mod events;
#[cfg(target_arch = "wasm32")]
pub mod showcase;
#[cfg(target_arch = "wasm32")]
pub mod websocket;

// Cross-platform modules (work on native for testing)
pub mod notebook;
pub mod router;
pub mod shell_autocomplete;
pub mod storage;

#[cfg(target_arch = "wasm32")]
pub use app::App;
#[cfg(target_arch = "wasm32")]
pub use canvas2d::Canvas2DRenderer;
pub use router::{BrowserRouter, RouteMatch, RouteMatcher};
pub use shell_autocomplete::ShellAutocomplete;
#[cfg(target_arch = "wasm32")]
pub use shell_autocomplete::ShellAutocompleteDemo;
#[cfg(target_arch = "wasm32")]
pub use showcase::ShowcaseDemo;
pub use storage::{ScopedStorage, Storage, StorageError, StorageType};
#[cfg(target_arch = "wasm32")]
pub use websocket::{WebSocketClient, WebSocketError};

// Notebook runtime
pub use notebook::{Cell, CellGraph, CellId, CellOutput, NotebookRuntime};
