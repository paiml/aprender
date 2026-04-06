//! REPL (Read-Eval-Print Loop) implementation.

use crate::commands::{execute, parse, Command};
use crate::state::SessionState;
use entrenar_common::{cli::styles, EntrenarError, Result};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::path::PathBuf;

/// Interactive REPL for entrenar-shell.
pub struct Repl {
    editor: DefaultEditor,
    state: SessionState,
    history_path: Option<PathBuf>,
}

impl Repl {
    /// Create a new REPL instance.
    pub fn new() -> Result<Self> {
        let editor = DefaultEditor::new().map_err(|e| EntrenarError::Internal {
            message: format!("Failed to create editor: {e}"),
        })?;

        let history_path = dirs::data_dir().map(|p| p.join("entrenar").join("shell_history"));

        let mut repl = Self { editor, state: SessionState::new(), history_path };

        // Load history if available
        if let Some(ref path) = repl.history_path {
            let _ = repl.editor.load_history(path);
        }

        Ok(repl)
    }

    /// Create a REPL with pre-configured state.
    pub fn with_state(state: SessionState) -> Result<Self> {
        let mut repl = Self::new()?;
        repl.state = state;
        Ok(repl)
    }

    /// Run the REPL main loop.
    pub fn run(&mut self) -> Result<()> {
        self.print_banner();

        loop {
            let prompt = self.format_prompt();

            match self.editor.readline(&prompt) {
                Ok(line) => {
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }

                    // Add to readline history
                    let _ = self.editor.add_history_entry(line);

                    // Parse and execute
                    match parse(line) {
                        Ok(cmd) => {
                            if matches!(cmd, Command::Quit) {
                                self.save_state();
                                println!("{}", styles::info("Session saved. Goodbye!"));
                                break;
                            }

                            if matches!(cmd, Command::Clear) {
                                print!("\x1B[2J\x1B[1;1H");
                                continue;
                            }

                            match execute(&cmd, &mut self.state) {
                                Ok(output) => {
                                    if !output.is_empty() {
                                        println!("{output}");
                                    }
                                }
                                Err(e) => {
                                    println!("{}", styles::error(&e.to_string()));
                                }
                            }
                        }
                        Err(e) => {
                            println!("{}", styles::error(&e.to_string()));
                        }
                    }
                }
                Err(ReadlineError::Interrupted) => {
                    println!("{}", styles::warning("Use 'quit' or Ctrl-D to exit"));
                }
                Err(ReadlineError::Eof) => {
                    self.save_state();
                    println!("\n{}", styles::info("Session saved. Goodbye!"));
                    break;
                }
                Err(e) => {
                    println!("{}", styles::error(&format!("Error: {e}")));
                }
            }
        }

        Ok(())
    }

    fn print_banner(&self) {
        println!("{}", styles::header("Entrenar Shell v0.1.0"));
        println!("Interactive Distillation Environment");
        println!("Type 'help' for commands, 'quit' to exit.\n");
    }

    fn format_prompt(&self) -> String {
        let model_count = self.state.loaded_models().len();
        if model_count > 0 {
            format!("entrenar ({model_count} models)> ")
        } else {
            "entrenar> ".to_string()
        }
    }

    fn save_state(&mut self) {
        // Save readline history
        if let Some(ref path) = self.history_path {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let _ = self.editor.save_history(path);
        }

        // Save session state
        if self.state.preferences().auto_save_history {
            if let Some(data_dir) = dirs::data_dir() {
                let state_path = data_dir.join("entrenar").join("session.json");
                if let Some(parent) = state_path.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                let _ = self.state.save(&state_path);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repl_creation() {
        // REPL creation should succeed
        let repl = Repl::new();
        assert!(repl.is_ok());
    }

    #[test]
    fn test_repl_with_state() {
        let mut state = SessionState::new();
        state.preferences_mut().default_batch_size = 64;

        let repl = Repl::with_state(state).expect("operation should succeed");
        assert_eq!(repl.state.preferences().default_batch_size, 64);
    }
}
