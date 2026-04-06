//! Terminal Monitor Callback (ENT-054)
//!
//! Real-time terminal monitoring callback for training loop integration.

mod monitor;
mod render;

#[cfg(test)]
mod tests;

pub use monitor::TerminalMonitorCallback;
