//! WASM Frontend for Calculator
//!
//! This module provides the WebAssembly bindings for the calculator,
//! demonstrating 100% test coverage across platform boundaries.
//!
//! Probar: Balanced testing - Balanced testing across TUI and WASM

#[cfg(feature = "wasm")]
mod browser;
mod calculator;
mod dom;
mod driver;
mod keypad;

#[cfg(feature = "wasm")]
pub use browser::BrowserCalculator;
pub use calculator::WasmCalculator;
pub use dom::{DomElement, DomEvent, MockDom};
pub use driver::WasmDriver;
pub use keypad::{KeypadAction, KeypadButtonDef, MockDomKeypadExt, WasmKeypad};
