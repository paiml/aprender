//! Serve command handler - pure functions for dev server

use crate::error::CliResult;
use crate::TreeArgs;
use std::path::Path;

/// Validate module imports before serving
pub fn validate_imports(directory: &Path, exclude: &[String]) -> CliResult<()> {
    use crate::dev_server::ModuleValidator;

    let mut validator = ModuleValidator::new(directory);
    if !exclude.is_empty() {
        validator = validator.with_exclude(exclude.to_vec());
    }
    let result = validator.validate();
    validator.print_results(&result);

    if !result.is_ok() {
        return Err(crate::CliError::test_execution(format!(
            "Module validation failed: {} error(s) found. Fix imports before serving.",
            result.errors.len()
        )));
    }
    eprintln!("\n✓ All module imports validated successfully\n");
    Ok(())
}

/// Open browser at the given URL
pub fn open_browser(url: &str) {
    println!("Opening browser at {url}...");
    #[cfg(target_os = "macos")]
    let _ = std::process::Command::new("open").arg(url).spawn();
    #[cfg(target_os = "linux")]
    let _ = std::process::Command::new("xdg-open").arg(url).spawn();
    #[cfg(target_os = "windows")]
    let _ = std::process::Command::new("start").arg(url).spawn();
}

/// Build tree config from CLI args
#[must_use]
pub fn build_tree_config(args: &TreeArgs) -> crate::TreeConfig {
    crate::TreeConfig::default()
        .with_depth(args.depth)
        .with_filter(args.filter.as_deref())
        .with_sizes(args.sizes)
        .with_mime_types(args.mime_types)
}

/// Format a server URL from port
#[must_use]
pub fn format_server_url(port: u16) -> String {
    format!("http://localhost:{port}")
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_format_server_url() {
        assert_eq!(format_server_url(8080), "http://localhost:8080");
        assert_eq!(format_server_url(3000), "http://localhost:3000");
    }

    #[test]
    fn test_build_tree_config_defaults() {
        let args = TreeArgs {
            path: std::path::PathBuf::from("."),
            depth: None,
            filter: None,
            sizes: false,
            mime_types: false,
        };
        let config = build_tree_config(&args);
        // Config should be created without panic
        assert!(config.max_depth.is_none());
    }

    #[test]
    fn test_build_tree_config_with_options() {
        let args = TreeArgs {
            path: std::path::PathBuf::from("."),
            depth: Some(3),
            filter: Some("*.rs".to_string()),
            sizes: true,
            mime_types: true,
        };
        let config = build_tree_config(&args);
        assert_eq!(config.max_depth, Some(3));
    }

    #[test]
    fn test_validate_imports_empty_dir() {
        let temp = TempDir::new().unwrap();
        let result = validate_imports(temp.path(), &[]);
        // Empty dir should pass validation
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_imports_with_exclude() {
        let temp = TempDir::new().unwrap();
        let result = validate_imports(temp.path(), &["node_modules".to_string()]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_open_browser_does_not_panic() {
        // Just test that it doesn't panic - actual browser opening is platform-specific
        open_browser("http://localhost:8080");
    }
}
