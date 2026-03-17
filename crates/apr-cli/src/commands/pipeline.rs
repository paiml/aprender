//! ALB-028: Pipeline orchestration — wraps forjar DAG engine.
//!
//! `apr pipeline plan/apply/status/validate` shells out to the `forjar`
//! binary, keeping the sovereign stack tools decoupled. Each subcommand
//! maps to a forjar CLI command with appropriate flags.

use crate::CliError;
use std::path::Path;
use std::process::Command;

/// Check that the forjar binary is available.
fn find_forjar() -> Result<String, CliError> {
    which::which("forjar")
        .map(|p| p.to_string_lossy().to_string())
        .map_err(|_| {
            CliError::ValidationFailed(
                "forjar not found in PATH. Install with: cargo install forjar".to_string(),
            )
        })
}

/// Run a forjar subcommand, streaming stdout/stderr to the terminal.
fn run_forjar(args: &[&str], json: bool) -> Result<(), CliError> {
    let bin = find_forjar()?;
    let mut cmd = Command::new(&bin);
    cmd.args(args);
    if json {
        cmd.arg("--json");
    }

    let status = cmd
        .status()
        .map_err(|e| CliError::Aprender(format!("failed to run forjar: {}", e)))?;

    if status.success() {
        Ok(())
    } else {
        let code = status.code().unwrap_or(1);
        Err(CliError::Aprender(format!(
            "forjar exited with code {}",
            code
        )))
    }
}

/// `apr pipeline validate <manifest>` — validate manifest without connecting.
pub(crate) fn run_validate(manifest: &Path, json: bool) -> Result<(), CliError> {
    let manifest_str = manifest.to_string_lossy();
    run_forjar(&["validate", "-f", &manifest_str], json)
}

/// `apr pipeline plan <manifest>` — show execution plan.
pub(crate) fn run_plan(
    manifest: &Path,
    machine: Option<&str>,
    tag: Option<&str>,
    cost: bool,
    json: bool,
) -> Result<(), CliError> {
    let manifest_str = manifest.to_string_lossy();
    let mut args: Vec<&str> = vec!["plan", "-f", &manifest_str];

    if let Some(m) = machine {
        args.push("-m");
        args.push(m);
    }
    if let Some(t) = tag {
        args.push("-t");
        args.push(t);
    }
    if cost {
        args.push("--cost");
    }

    run_forjar(&args, json)
}

/// `apr pipeline apply <manifest>` — converge resources.
pub(crate) fn run_apply(
    manifest: &Path,
    machine: Option<&str>,
    tag: Option<&str>,
    parallel: Option<u32>,
    keep_going: bool,
    json: bool,
) -> Result<(), CliError> {
    let manifest_str = manifest.to_string_lossy();
    let mut args: Vec<&str> = vec!["apply", "-f", &manifest_str];

    if let Some(m) = machine {
        args.push("-m");
        args.push(m);
    }
    if let Some(t) = tag {
        args.push("-t");
        args.push(t);
    }

    // GH-506: --parallel removed — forjar has no parallelism flag
    // (-p is --param KEY=VALUE, not parallelism)
    if parallel.is_some() {
        return Err(CliError::ValidationFailed(
            "--parallel is not supported (forjar does not have a parallelism flag)".to_string(),
        ));
    }
    if keep_going {
        args.push("--keep-going");
    }

    run_forjar(&args, json)
}

/// `apr pipeline status <manifest>` — show resource state.
pub(crate) fn run_status(manifest: &Path, json: bool) -> Result<(), CliError> {
    let manifest_str = manifest.to_string_lossy();
    run_forjar(&["status", "-f", &manifest_str], json)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_forjar_returns_path_or_error() {
        let result = find_forjar();
        match result {
            Ok(path) => assert!(path.contains("forjar")),
            Err(e) => assert!(e.to_string().contains("not found")),
        }
    }

    #[test]
    fn test_validate_nonexistent_manifest() {
        let result = run_validate(Path::new("/nonexistent/manifest.yaml"), false);
        assert!(result.is_err());
    }

    #[test]
    fn test_plan_nonexistent_manifest() {
        let result = run_plan(
            Path::new("/nonexistent/manifest.yaml"),
            None,
            None,
            false,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_status_nonexistent_manifest() {
        let result = run_status(Path::new("/nonexistent/manifest.yaml"), false);
        assert!(result.is_err());
    }
}
