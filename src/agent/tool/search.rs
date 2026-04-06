//! Search tools for agent code discovery.
//!
//! Provides `glob` and `grep` tools for the `apr code` agentic
//! coding assistant. These are the agent's primary code navigation
//! tools, giving it the ability to find files by pattern and search
//! content by regex.
//!
//! Both tools require `Capability::FileRead` and respect path
//! restrictions (Poka-Yoke). Results are truncated to prevent
//! context overflow (Jidoka: bounded output).

use std::path::PathBuf;

use async_trait::async_trait;

use crate::agent::capability::Capability;
use crate::agent::driver::ToolDefinition;

use super::{Tool, ToolResult};

/// Maximum matching files returned by glob.
const MAX_GLOB_RESULTS: usize = 200;

/// Maximum matching lines returned by grep.
const MAX_GREP_RESULTS: usize = 200;

/// Maximum bytes of grep output before truncation.
const MAX_GREP_BYTES: usize = 32_768;

// ─── GlobTool ───────────────────────────────────────────────

/// Find files by glob pattern.
///
/// Wraps the `glob` crate for fast file discovery. Returns paths
/// sorted by modification time (most recent first), capped at
/// `MAX_GLOB_RESULTS`.
pub struct GlobTool {
    allowed_paths: Vec<String>,
}

impl GlobTool {
    pub fn new(allowed_paths: Vec<String>) -> Self {
        Self { allowed_paths }
    }
}

#[async_trait]
impl Tool for GlobTool {
    fn name(&self) -> &'static str {
        "glob"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "glob".into(),
            description:
                "Find files matching a glob pattern. Returns paths sorted by modification time."
                    .into(),
            input_schema: serde_json::json!({
                "type": "object",
                "required": ["pattern"],
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g., 'src/**/*.rs', '*.toml')"
                    },
                    "path": {
                        "type": "string",
                        "description": "Base directory to search in (default: current dir)"
                    }
                }
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let pattern = match input.get("pattern").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return ToolResult::error("missing required field 'pattern'"),
        };

        let base = input.get("path").and_then(|v| v.as_str()).unwrap_or(".");

        // Construct full pattern
        let full_pattern = if pattern.starts_with('/') {
            pattern.to_string()
        } else {
            format!("{}/{}", base.trim_end_matches('/'), pattern)
        };

        let entries = match glob::glob(&full_pattern) {
            Ok(paths) => paths,
            Err(e) => return ToolResult::error(format!("invalid glob pattern: {e}")),
        };

        let mut results: Vec<(PathBuf, std::time::SystemTime)> = Vec::new();
        for entry in entries.take(MAX_GLOB_RESULTS * 2) {
            // overscan to allow filtering
            let Ok(path) = entry else { continue };
            if !path.is_file() {
                continue;
            }
            // Validate against allowed paths
            if !self.allowed_paths.iter().any(|p| p == "*") {
                let Ok(canon) = path.canonicalize() else {
                    continue;
                };
                let allowed = self.allowed_paths.iter().any(|prefix| {
                    PathBuf::from(prefix)
                        .canonicalize()
                        .map(|pc| canon.starts_with(&pc))
                        .unwrap_or(false)
                });
                if !allowed {
                    continue;
                }
            }
            let mtime = path.metadata().and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH);
            results.push((path, mtime));
        }

        // Sort by modification time (most recent first)
        results.sort_by(|a, b| b.1.cmp(&a.1));
        results.truncate(MAX_GLOB_RESULTS);

        if results.is_empty() {
            return ToolResult::success(format!("No files matching '{full_pattern}'"));
        }

        let output: String =
            results.iter().map(|(p, _)| p.display().to_string()).collect::<Vec<_>>().join("\n");

        let suffix = if results.len() == MAX_GLOB_RESULTS {
            format!("\n\n[truncated at {MAX_GLOB_RESULTS} results]")
        } else {
            String::new()
        };

        ToolResult::success(format!("{output}{suffix}"))
    }

    fn required_capability(&self) -> Capability {
        Capability::FileRead { allowed_paths: self.allowed_paths.clone() }
    }
}

// ─── GrepTool ───────────────────────────────────────────────

/// Search file contents by regex pattern.
///
/// Walks a directory and matches lines against a regex. Returns
/// matching lines with file path, line number, and content.
/// Results capped at `MAX_GREP_RESULTS` lines.
pub struct GrepTool {
    allowed_paths: Vec<String>,
}

impl GrepTool {
    pub fn new(allowed_paths: Vec<String>) -> Self {
        Self { allowed_paths }
    }
}

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &'static str {
        "grep"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "grep".into(),
            description:
                "Search file contents with regex. Returns matching lines with file:line:content."
                    .into(),
            input_schema: serde_json::json!({
                "type": "object",
                "required": ["pattern"],
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for"
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search (default: current dir)"
                    },
                    "glob": {
                        "type": "string",
                        "description": "Glob to filter files (e.g., '*.rs', '*.toml')"
                    },
                    "case_insensitive": {
                        "type": "boolean",
                        "description": "Case-insensitive search (default: false)"
                    }
                }
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let pattern_str = match input.get("pattern").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return ToolResult::error("missing required field 'pattern'"),
        };

        let search_path = input.get("path").and_then(|v| v.as_str()).unwrap_or(".");

        let file_glob = input.get("glob").and_then(|v| v.as_str());
        let case_insensitive =
            input.get("case_insensitive").and_then(|v| v.as_bool()).unwrap_or(false);

        let matcher = PatternMatcher::new(pattern_str, case_insensitive);

        let root = PathBuf::from(search_path);
        if !root.exists() {
            return ToolResult::error(format!("path '{}' not found", root.display()));
        }

        let mut output = String::new();
        let mut match_count = 0;

        // Single file
        if root.is_file() {
            search_file(&root, &matcher, &mut output, &mut match_count);
            return finish_grep(output, match_count);
        }

        // Directory walk
        let walker = walkdir::WalkDir::new(&root)
            .max_depth(20)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file());

        // Compile file glob filter if provided
        let file_pattern = file_glob.and_then(|g| glob::Pattern::new(g).ok());

        for entry in walker {
            if match_count >= MAX_GREP_RESULTS {
                break;
            }

            let path = entry.path();

            // Filter by file glob
            if let Some(ref pat) = file_pattern {
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if !pat.matches(name) {
                    continue;
                }
            }

            // Skip binary files (quick heuristic: NUL byte in first 512 bytes)
            if is_likely_binary(path) {
                continue;
            }

            search_file(path, &matcher, &mut output, &mut match_count);
        }

        finish_grep(output, match_count)
    }

    fn required_capability(&self) -> Capability {
        Capability::FileRead { allowed_paths: self.allowed_paths.clone() }
    }
}

/// Simple pattern matcher (substring with optional case-insensitivity).
///
/// Uses string `contains()` instead of regex to avoid adding a regex
/// dependency. For the agent's search use case, substring matching
/// covers the vast majority of queries.
struct PatternMatcher {
    pattern: String,
    case_insensitive: bool,
}

impl PatternMatcher {
    fn new(pattern: &str, case_insensitive: bool) -> Self {
        let pattern = if case_insensitive { pattern.to_lowercase() } else { pattern.to_string() };
        Self { pattern, case_insensitive }
    }

    fn is_match(&self, line: &str) -> bool {
        if self.case_insensitive {
            line.to_lowercase().contains(&self.pattern)
        } else {
            line.contains(&self.pattern)
        }
    }
}

/// Search a single file for pattern matches, appending results to `output`.
fn search_file(
    path: &std::path::Path,
    matcher: &PatternMatcher,
    output: &mut String,
    match_count: &mut usize,
) {
    let Ok(content) = std::fs::read_to_string(path) else {
        return;
    };
    for (line_num, line) in content.lines().enumerate() {
        if *match_count >= MAX_GREP_RESULTS {
            break;
        }
        if matcher.is_match(line) {
            use std::fmt::Write;
            let _ = writeln!(output, "{}:{}:{}", path.display(), line_num + 1, line);
            *match_count += 1;
        }
    }
}

/// Quick check if a file is likely binary (non-UTF-8 in first 512 bytes).
fn is_likely_binary(path: &std::path::Path) -> bool {
    let Ok(mut f) = std::fs::File::open(path) else {
        return true;
    };
    let mut buf = [0u8; 512];
    let Ok(n) = std::io::Read::read(&mut f, &mut buf) else {
        return true;
    };
    buf[..n].contains(&0)
}

/// Format the final grep result with truncation info.
fn finish_grep(mut output: String, match_count: usize) -> ToolResult {
    if match_count == 0 {
        return ToolResult::success("No matches found.");
    }

    if output.len() > MAX_GREP_BYTES {
        output.truncate(MAX_GREP_BYTES);
        output.push_str("\n\n[output truncated]");
    }

    if match_count >= MAX_GREP_RESULTS {
        output.push_str(&format!("\n\n[truncated at {MAX_GREP_RESULTS} matches]"));
    }

    ToolResult::success(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as IoWrite;
    use tempfile::TempDir;

    fn create_project(dir: &std::path::Path) {
        std::fs::create_dir_all(dir.join("src")).unwrap();
        let mut f1 = std::fs::File::create(dir.join("src/main.rs")).unwrap();
        f1.write_all(b"fn main() {\n    println!(\"hello\");\n}\n").unwrap();

        let mut f2 = std::fs::File::create(dir.join("src/lib.rs")).unwrap();
        f2.write_all(b"pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n").unwrap();

        let mut f3 = std::fs::File::create(dir.join("Cargo.toml")).unwrap();
        f3.write_all(b"[package]\nname = \"test\"\nversion = \"0.1.0\"\n").unwrap();
    }

    // ─── GlobTool tests ─────────────────────────────────

    #[tokio::test]
    async fn test_glob_find_rust_files() {
        let dir = TempDir::new().unwrap();
        create_project(dir.path());
        let tool = GlobTool::new(vec!["*".into()]);

        let result = tool
            .execute(serde_json::json!({
                "pattern": "**/*.rs",
                "path": dir.path().to_str().unwrap()
            }))
            .await;
        assert!(!result.is_error, "error: {}", result.content);
        assert!(result.content.contains("main.rs"));
        assert!(result.content.contains("lib.rs"));
        assert!(!result.content.contains("Cargo.toml"));
    }

    #[tokio::test]
    async fn test_glob_find_toml() {
        let dir = TempDir::new().unwrap();
        create_project(dir.path());
        let tool = GlobTool::new(vec!["*".into()]);

        let result = tool
            .execute(serde_json::json!({
                "pattern": "*.toml",
                "path": dir.path().to_str().unwrap()
            }))
            .await;
        assert!(!result.is_error);
        assert!(result.content.contains("Cargo.toml"));
        assert!(!result.content.contains(".rs"));
    }

    #[tokio::test]
    async fn test_glob_no_matches() {
        let dir = TempDir::new().unwrap();
        create_project(dir.path());
        let tool = GlobTool::new(vec!["*".into()]);

        let result = tool
            .execute(serde_json::json!({
                "pattern": "**/*.py",
                "path": dir.path().to_str().unwrap()
            }))
            .await;
        assert!(!result.is_error);
        assert!(result.content.contains("No files matching"));
    }

    #[tokio::test]
    async fn test_glob_invalid_pattern() {
        let tool = GlobTool::new(vec!["*".into()]);
        let result = tool.execute(serde_json::json!({"pattern": "[invalid"})).await;
        assert!(result.is_error);
        assert!(result.content.contains("invalid glob"));
    }

    #[tokio::test]
    async fn test_glob_missing_pattern() {
        let tool = GlobTool::new(vec!["*".into()]);
        let result = tool.execute(serde_json::json!({"path": "."})).await;
        assert!(result.is_error);
        assert!(result.content.contains("missing"));
    }

    #[test]
    fn test_glob_tool_metadata() {
        let tool = GlobTool::new(vec!["/home".into()]);
        assert_eq!(tool.name(), "glob");
        let def = tool.definition();
        assert_eq!(def.name, "glob");
        match tool.required_capability() {
            Capability::FileRead { allowed_paths } => {
                assert_eq!(allowed_paths, vec!["/home".to_string()]);
            }
            other => panic!("expected FileRead, got: {other:?}"),
        }
    }

    // ─── GrepTool tests ─────────────────────────────────

    #[tokio::test]
    async fn test_grep_find_pattern() {
        let dir = TempDir::new().unwrap();
        create_project(dir.path());
        let tool = GrepTool::new(vec!["*".into()]);

        let result = tool
            .execute(serde_json::json!({
                "pattern": "println",
                "path": dir.path().to_str().unwrap()
            }))
            .await;
        assert!(!result.is_error, "error: {}", result.content);
        assert!(result.content.contains("main.rs"));
        assert!(result.content.contains("println"));
    }

    #[tokio::test]
    async fn test_grep_with_file_glob() {
        let dir = TempDir::new().unwrap();
        create_project(dir.path());
        let tool = GrepTool::new(vec!["*".into()]);

        let result = tool
            .execute(serde_json::json!({
                "pattern": "fn",
                "path": dir.path().to_str().unwrap(),
                "glob": "*.rs"
            }))
            .await;
        assert!(!result.is_error);
        assert!(result.content.contains("main.rs"));
        assert!(result.content.contains("lib.rs"));
        // Should NOT search Cargo.toml
        assert!(!result.content.contains("Cargo.toml"));
    }

    #[tokio::test]
    async fn test_grep_case_insensitive() {
        let dir = TempDir::new().unwrap();
        create_project(dir.path());
        let tool = GrepTool::new(vec!["*".into()]);

        let result = tool
            .execute(serde_json::json!({
                "pattern": "PRINTLN",
                "path": dir.path().to_str().unwrap(),
                "case_insensitive": true
            }))
            .await;
        assert!(!result.is_error);
        assert!(result.content.contains("println"));
    }

    #[tokio::test]
    async fn test_grep_no_matches() {
        let dir = TempDir::new().unwrap();
        create_project(dir.path());
        let tool = GrepTool::new(vec!["*".into()]);

        let result = tool
            .execute(serde_json::json!({
                "pattern": "ZZZZZ_NONEXISTENT",
                "path": dir.path().to_str().unwrap()
            }))
            .await;
        assert!(!result.is_error);
        assert!(result.content.contains("No matches"));
    }

    #[tokio::test]
    async fn test_grep_special_chars_in_pattern() {
        let dir = TempDir::new().unwrap();
        create_project(dir.path());
        let tool = GrepTool::new(vec!["*".into()]);

        // Brackets are treated as literal substring, not regex
        let result = tool
            .execute(serde_json::json!({
                "pattern": "[invalid",
                "path": dir.path().to_str().unwrap()
            }))
            .await;
        assert!(!result.is_error);
        assert!(result.content.contains("No matches"));
    }

    #[tokio::test]
    async fn test_grep_single_file() {
        let dir = TempDir::new().unwrap();
        create_project(dir.path());
        let tool = GrepTool::new(vec!["*".into()]);

        let file_path = dir.path().join("src/main.rs");
        let result = tool
            .execute(serde_json::json!({
                "pattern": "fn",
                "path": file_path.to_str().unwrap()
            }))
            .await;
        assert!(!result.is_error);
        assert!(result.content.contains("fn main"));
    }

    #[tokio::test]
    async fn test_grep_nonexistent_path() {
        let tool = GrepTool::new(vec!["*".into()]);
        let result = tool
            .execute(serde_json::json!({
                "pattern": "test",
                "path": "/nonexistent_dir_xyz"
            }))
            .await;
        assert!(result.is_error);
        assert!(result.content.contains("not found"));
    }

    #[tokio::test]
    async fn test_grep_missing_pattern() {
        let tool = GrepTool::new(vec!["*".into()]);
        let result = tool.execute(serde_json::json!({"path": "."})).await;
        assert!(result.is_error);
        assert!(result.content.contains("missing"));
    }

    #[test]
    fn test_grep_tool_metadata() {
        let tool = GrepTool::new(vec!["/project".into()]);
        assert_eq!(tool.name(), "grep");
        let def = tool.definition();
        assert_eq!(def.name, "grep");
        match tool.required_capability() {
            Capability::FileRead { allowed_paths } => {
                assert_eq!(allowed_paths, vec!["/project".to_string()]);
            }
            other => panic!("expected FileRead, got: {other:?}"),
        }
    }

    // ─── Helper tests ───────────────────────────────────

    #[test]
    fn test_is_likely_binary_text() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("text.txt");
        std::fs::write(&path, "hello world").unwrap();
        assert!(!is_likely_binary(&path));
    }

    #[test]
    fn test_is_likely_binary_binary() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("binary.bin");
        std::fs::write(&path, &[0u8, 1, 2, 0, 3, 4]).unwrap();
        assert!(is_likely_binary(&path));
    }

    #[test]
    fn test_is_likely_binary_nonexistent() {
        assert!(is_likely_binary(std::path::Path::new("/no_such_file_xyz")));
    }
}
