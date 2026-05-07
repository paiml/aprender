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
///
/// Note: the upstream `cmd_code` has gained additional parameters over time:
/// - 8th: `emit_trace: Option<PathBuf>` (M28 — ccpa-trace.jsonl)
/// - 9th: `output_format: &str` (PMAT-CODE-OUTPUT-FORMAT-001 — Claude-Code parity)
/// - 10th: `input_format: &str` (PMAT-CODE-INPUT-FORMAT-001 — Claude-Code parity)
///
/// The bin-side clap surface (`Commands::Code` in `main_dispatch.rs`) does
/// not yet expose them; this wrapper passes legacy defaults so behavior is
/// unchanged. Threading the new flags through the binary's clap is a future
/// enhancement (the apr-cli surface already exposes them via `apr code`).
pub fn cmd_code(
    model: Option<PathBuf>,
    project: PathBuf,
    resume: Option<Option<String>>,
    prompt: Vec<String>,
    print: bool,
    max_turns: u32,
    manifest_path: Option<PathBuf>,
) -> anyhow::Result<()> {
    batuta::agent::code::cmd_code(
        model,
        project,
        resume,
        prompt,
        print,
        max_turns,
        manifest_path,
        None,   // emit_trace: not yet plumbed through the binary's clap surface.
        "text", // output_format: legacy default
        "text", // input_format: legacy default
    )
}

#[cfg(test)]
#[path = "code_tests.rs"]
mod tests;
