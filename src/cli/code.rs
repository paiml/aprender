//! `batuta code` — interactive AI coding assistant.
//!
//! Thin CLI wrapper that delegates to `batuta::agent::code::cmd_code()`.
//! The library module contains all the logic so that `apr-cli` can also
//! call it directly (PMAT-162: Phase 6).
//!
//! See: docs/specifications/components/apr-code.md

use std::path::PathBuf;

/// Entry point for `batuta code` (binary-side thin wrapper).
///
/// Delegates entirely to the library-level `agent::code::cmd_code`.
pub fn cmd_code(
    model: Option<PathBuf>,
    project: PathBuf,
    resume: Option<Option<String>>,
    prompt: Vec<String>,
    print: bool,
    max_turns: u32,
    manifest_path: Option<PathBuf>,
) -> anyhow::Result<()> {
    batuta::agent::code::cmd_code(model, project, resume, prompt, print, max_turns, manifest_path)
}

#[cfg(test)]
#[path = "code_tests.rs"]
mod tests;
