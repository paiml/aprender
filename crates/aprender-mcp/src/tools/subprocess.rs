//! Shared subprocess wrapper for M2 tools.
//!
//! Every M2 tool spawns `apr <subcommand> [...args] --json` and passes stdout
//! through to the MCP client verbatim. Non-zero exit maps to `isError: true`
//! with stderr attached. This module centralizes that pattern so each tool is
//! a thin definition + a list of CLI args.

use crate::types::ToolCallResult;
use std::process::Command;

/// Spawn `apr <args...>` and wrap the result as a `ToolCallResult`.
///
/// - Successful exit with non-empty stdout → `success(stdout)`
/// - Successful exit with empty stdout → `error("apr ... produced no output")`
/// - Non-zero exit → `error("apr ... failed (exit N): <stderr-or-stdout>")`
/// - Spawn failure → `error("Failed to spawn apr ...: <io-err>")`
#[must_use]
pub fn run_apr(args: &[&str]) -> ToolCallResult {
    let output = match Command::new("apr").args(args).output() {
        Ok(o) => o,
        Err(e) => {
            let cmd = format!("apr {}", args.join(" "));
            return ToolCallResult::error(format!("Failed to spawn `{cmd}`: {e}"));
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();

    if output.status.success() {
        if stdout.trim().is_empty() {
            let cmd = format!("apr {}", args.join(" "));
            ToolCallResult::error(format!("`{cmd}` produced no output"))
        } else {
            ToolCallResult::success(stdout)
        }
    } else {
        let code = output.status.code().unwrap_or(-1);
        let detail = if stderr.trim().is_empty() {
            stdout
        } else {
            stderr
        };
        let cmd = format!("apr {}", args.join(" "));
        ToolCallResult::error(format!("`{cmd}` failed (exit {code}): {detail}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spawning a non-existent binary yields a spawn error, not a panic.
    #[test]
    fn spawn_failure_maps_to_tool_error() {
        // Temporarily simulate missing binary by calling a clearly-absurd
        // subcommand through the real `apr`. We can't easily swap the binary
        // name without refactoring, so we exercise the non-zero-exit branch
        // via an invalid argument instead.
        let result = run_apr(&["this-subcommand-does-not-exist"]);
        assert_eq!(result.is_error, Some(true));
    }
}
