//! Build command handler - pure functions for build validation

use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Find all HTML files in a directory recursively
#[must_use]
pub fn find_html_files(dir: &Path) -> Vec<PathBuf> {
    let mut html_files = Vec::new();
    scan_files_recursive(dir, "html", &mut html_files);
    html_files.sort();
    html_files
}

/// Find all files with given extension recursively
pub fn scan_files_recursive(dir: &Path, extension: &str, files: &mut Vec<PathBuf>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                // Skip node_modules and hidden directories
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if !name.starts_with('.') && name != "node_modules" && name != "target" {
                    scan_files_recursive(&path, extension, files);
                }
            } else if path.extension().is_some_and(|ext| ext == extension) {
                files.push(path);
            }
        }
    }
}

/// Check if an HTML file references WASM
#[must_use]
pub fn html_references_wasm(html_path: &Path) -> bool {
    std::fs::read_to_string(html_path)
        .map(|content| {
            content.contains(".wasm")
                || content.contains("WebAssembly")
                || content.contains("wasm-bindgen")
                || content.contains("wasm_bindgen")
        })
        .unwrap_or(false)
}

/// Find which HTML files reference WASM
#[must_use]
pub fn find_wasm_pages(html_files: &[PathBuf]) -> HashSet<PathBuf> {
    html_files
        .iter()
        .filter(|f| html_references_wasm(f))
        .cloned()
        .collect()
}

/// Report validation results
#[must_use]
pub fn format_validation_result(errors: &[String]) -> String {
    if errors.is_empty() {
        "RESULT: PASS (App works!)\n\nCould not prove the app is broken. All validation checks passed.".to_string()
    } else {
        let mut result = format!(
            "RESULT: FAIL (Grade: F)\n\nFound {} issue(s) that prove the app is broken:\n\n",
            errors.len()
        );
        for (i, error) in errors.iter().enumerate() {
            result.push_str(&format!("  {}. {}\n", i + 1, error));
        }
        result
    }
}

/// Select a free port for dev server
#[must_use]
pub fn find_free_port(start: u16) -> u16 {
    // Simple approach: try ports starting from start
    for port in start..start + 100 {
        if std::net::TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return port;
        }
    }
    start // Fallback to start if no free port found
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_find_html_files_empty() {
        let temp = TempDir::new().unwrap();
        let files = find_html_files(temp.path());
        assert!(files.is_empty());
    }

    #[test]
    fn test_find_html_files_with_files() {
        let temp = TempDir::new().unwrap();
        std::fs::write(temp.path().join("index.html"), "<html></html>").unwrap();
        std::fs::write(temp.path().join("about.html"), "<html></html>").unwrap();
        std::fs::write(temp.path().join("style.css"), "body {}").unwrap();

        let files = find_html_files(temp.path());
        assert_eq!(files.len(), 2);
    }

    #[test]
    fn test_find_html_files_nested() {
        let temp = TempDir::new().unwrap();
        std::fs::create_dir_all(temp.path().join("pages")).unwrap();
        std::fs::write(temp.path().join("index.html"), "<html></html>").unwrap();
        std::fs::write(
            temp.path().join("pages").join("about.html"),
            "<html></html>",
        )
        .unwrap();

        let files = find_html_files(temp.path());
        assert_eq!(files.len(), 2);
    }

    #[test]
    fn test_find_html_files_skips_node_modules() {
        let temp = TempDir::new().unwrap();
        std::fs::create_dir_all(temp.path().join("node_modules")).unwrap();
        std::fs::write(temp.path().join("index.html"), "<html></html>").unwrap();
        std::fs::write(
            temp.path().join("node_modules").join("lib.html"),
            "<html></html>",
        )
        .unwrap();

        let files = find_html_files(temp.path());
        assert_eq!(files.len(), 1);
    }

    #[test]
    fn test_find_html_files_skips_hidden() {
        let temp = TempDir::new().unwrap();
        std::fs::create_dir_all(temp.path().join(".hidden")).unwrap();
        std::fs::write(temp.path().join("index.html"), "<html></html>").unwrap();
        std::fs::write(
            temp.path().join(".hidden").join("secret.html"),
            "<html></html>",
        )
        .unwrap();

        let files = find_html_files(temp.path());
        assert_eq!(files.len(), 1);
    }

    #[test]
    fn test_html_references_wasm_true() {
        let temp = TempDir::new().unwrap();
        let html_path = temp.path().join("index.html");
        std::fs::write(&html_path, r#"<script src="app.wasm"></script>"#).unwrap();

        assert!(html_references_wasm(&html_path));
    }

    #[test]
    fn test_html_references_wasm_webassembly() {
        let temp = TempDir::new().unwrap();
        let html_path = temp.path().join("index.html");
        std::fs::write(&html_path, r"<script>WebAssembly.instantiate()</script>").unwrap();

        assert!(html_references_wasm(&html_path));
    }

    #[test]
    fn test_html_references_wasm_bindgen() {
        let temp = TempDir::new().unwrap();
        let html_path = temp.path().join("index.html");
        std::fs::write(
            &html_path,
            r"<script>import init from './pkg/wasm_bindgen.js'</script>",
        )
        .unwrap();

        assert!(html_references_wasm(&html_path));
    }

    #[test]
    fn test_html_references_wasm_false() {
        let temp = TempDir::new().unwrap();
        let html_path = temp.path().join("index.html");
        std::fs::write(&html_path, r"<html><body>Hello</body></html>").unwrap();

        assert!(!html_references_wasm(&html_path));
    }

    #[test]
    fn test_html_references_wasm_nonexistent() {
        assert!(!html_references_wasm(Path::new("/nonexistent.html")));
    }

    #[test]
    fn test_find_wasm_pages() {
        let temp = TempDir::new().unwrap();
        let html1 = temp.path().join("wasm.html");
        let html2 = temp.path().join("plain.html");
        std::fs::write(&html1, r#"<script src="app.wasm"></script>"#).unwrap();
        std::fs::write(&html2, r"<html></html>").unwrap();

        let files = vec![html1.clone(), html2];
        let wasm_pages = find_wasm_pages(&files);

        assert_eq!(wasm_pages.len(), 1);
        assert!(wasm_pages.contains(&html1));
    }

    #[test]
    fn test_format_validation_result_pass() {
        let result = format_validation_result(&[]);
        assert!(result.contains("PASS"));
        assert!(result.contains("Could not prove"));
    }

    #[test]
    fn test_format_validation_result_fail() {
        let errors = vec!["Error 1".to_string(), "Error 2".to_string()];
        let result = format_validation_result(&errors);
        assert!(result.contains("FAIL"));
        assert!(result.contains("2 issue(s)"));
        assert!(result.contains("Error 1"));
        assert!(result.contains("Error 2"));
    }

    #[test]
    fn test_find_free_port() {
        let port = find_free_port(50000);
        assert!(port >= 50000);
        assert!(port < 50100);
    }

    #[test]
    fn test_scan_files_recursive_skips_target() {
        let temp = TempDir::new().unwrap();
        std::fs::create_dir_all(temp.path().join("target")).unwrap();
        std::fs::write(temp.path().join("lib.rs"), "fn main() {}").unwrap();
        std::fs::write(temp.path().join("target").join("output.rs"), "").unwrap();

        let mut files = Vec::new();
        scan_files_recursive(temp.path(), "rs", &mut files);
        assert_eq!(files.len(), 1);
    }
}
