//! Interactive REPL for HuggingFace model exploration and distillation.
//!
//! This crate provides an interactive shell for:
//! - Exploring model architectures and memory requirements
//! - Running distillation experiments interactively
//! - Managing model downloads and exports
//!
//! # Toyota Way Principles
//!
//! - **Genchi Genbutsu**: Direct interaction with the "gemba" (model tensors)
//! - **Kaizen**: Iterative experimentation for continuous improvement
//! - **Visual Control**: Immediate feedback on operations

pub mod commands;
pub mod repl;
pub mod state;

pub use repl::Repl;
pub use state::SessionState;

use entrenar_common::Result;

/// Start the interactive shell.
pub fn start() -> Result<()> {
    let mut repl = Repl::new()?;
    repl.run()
}

/// Start the shell with a pre-configured state.
pub fn start_with_state(state: SessionState) -> Result<()> {
    let mut repl = Repl::with_state(state)?;
    repl.run()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_state_default() {
        let state = SessionState::default();
        assert!(state.loaded_models().is_empty());
        assert!(state.history().is_empty());
    }
}
