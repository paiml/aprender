//! Terminal Capability Detection (ENT-061)
//!
//! Detects terminal features for optimal rendering.

mod capabilities;
mod layout;
mod mode;

pub use capabilities::TerminalCapabilities;
pub use layout::DashboardLayout;
pub use mode::TerminalMode;

#[cfg(test)]
mod tests;
