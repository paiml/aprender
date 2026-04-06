//! TUI Frontend for Calculator
//!
//! Probar: Visual feedback - Visual feedback through terminal interface

mod app;
mod input;
mod keypad;
mod ui;

pub use app::CalculatorApp;
pub use input::{InputHandler, KeyAction};
pub use keypad::{ButtonAction, Keypad, KeypadButton, KeypadWidget};
pub use ui::render;
