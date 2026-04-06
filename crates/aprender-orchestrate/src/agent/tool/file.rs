//! File system tools for agent code editing.
//!
//! Provides `file_read`, `file_write`, and `file_edit` tools for the
//! `apr code` agentic coding assistant. Each tool enforces path
//! restrictions via `Capability::FileRead` / `Capability::FileWrite`
//! (Poka-Yoke: mistake-proofing).
//!
//! Design follows Claude Code's tool semantics:
//! - `file_read`: Read file with optional line range
//! - `file_write`: Create or overwrite a file
//! - `file_edit`: Replace a unique string in a file
//!
//! Security: paths are canonicalized and checked against allowed
//! prefixes before any I/O. Symlink traversal is blocked.

use std::path::{Path, PathBuf};

use async_trait::async_trait;

use crate::agent::capability::Capability;
use crate::agent::driver::ToolDefinition;

use super::{Tool, ToolResult};

/// Maximum file size that `file_read` will return (128 KB).
const MAX_READ_BYTES: usize = 128 * 1024;

/// Maximum number of lines `file_read` returns per call.
const MAX_READ_LINES: usize = 2000;

// ─── Path validation (shared) ───────────────────────────────

/// Validate that a path is within allowed prefixes.
/// Returns the canonicalized path on success.
fn validate_path(raw: &str, allowed: &[String]) -> Result<PathBuf, String> {
    if raw.is_empty() {
        return Err("path is empty".into());
    }
    // Canonicalize to resolve symlinks (Poka-Yoke: block symlink traversal)
    let canonical = PathBuf::from(raw)
        .canonicalize()
        .map_err(|e| format!("cannot resolve path '{}': {}", raw, e))?;
    check_prefix(&canonical, &canonical, allowed)
}

/// Validate a path for writing. Parent directory must exist.
/// For new files, we validate the parent directory instead.
fn validate_write_path(raw: &str, allowed: &[String]) -> Result<PathBuf, String> {
    if raw.is_empty() {
        return Err("path is empty".into());
    }

    let path = PathBuf::from(raw);

    // For existing files, canonicalize normally
    if path.exists() {
        return validate_path(raw, allowed);
    }

    // For new files, validate parent directory exists and is allowed
    let parent = path.parent().ok_or_else(|| format!("cannot determine parent of '{}'", raw))?;

    let parent_canon = parent
        .canonicalize()
        .map_err(|e| format!("parent directory '{}' not found: {}", parent.display(), e))?;

    let target = parent_canon.join(path.file_name().unwrap_or_default());
    check_prefix(&target, &parent_canon, allowed)
}

/// Check that a canonical path is within at least one allowed prefix.
fn check_prefix(target: &Path, canonical: &Path, allowed: &[String]) -> Result<PathBuf, String> {
    if allowed.iter().any(|p| p == "*") {
        return Ok(target.to_path_buf());
    }
    for prefix in allowed {
        if let Ok(prefix_canon) = PathBuf::from(prefix).canonicalize() {
            if canonical.starts_with(&prefix_canon) {
                return Ok(target.to_path_buf());
            }
        }
    }
    Err(format!("path '{}' outside allowed prefixes: {:?}", target.display(), allowed))
}

// ─── FileReadTool ───────────────────────────────────────────

/// Read file contents with optional line range.
///
/// Returns numbered lines (like `cat -n`). Respects `MAX_READ_BYTES`
/// and `MAX_READ_LINES` limits to prevent context overflow.
pub struct FileReadTool {
    allowed_paths: Vec<String>,
}

impl FileReadTool {
    pub fn new(allowed_paths: Vec<String>) -> Self {
        Self { allowed_paths }
    }
}

#[async_trait]
impl Tool for FileReadTool {
    fn name(&self) -> &'static str {
        "file_read"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "file_read".into(),
            description: "Read a file's contents. Returns numbered lines.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start from (1-based, default 1)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum lines to read (default 2000)"
                    }
                }
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let path_str = match input.get("path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return ToolResult::error("missing required field 'path'"),
        };

        let offset = input.get("offset").and_then(|v| v.as_u64()).unwrap_or(1).max(1) as usize;
        let limit = input
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(MAX_READ_LINES as u64)
            .min(MAX_READ_LINES as u64) as usize;

        let path = match validate_path(path_str, &self.allowed_paths) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(e),
        };

        // Check file size (Jidoka: don't read huge files)
        match std::fs::metadata(&path) {
            Ok(meta) if meta.len() > MAX_READ_BYTES as u64 => {
                return ToolResult::error(format!(
                    "file too large ({} bytes, max {}). Use offset/limit to read a portion.",
                    meta.len(),
                    MAX_READ_BYTES
                ));
            }
            Err(e) => return ToolResult::error(format!("cannot stat '{}': {}", path.display(), e)),
            _ => {}
        }

        match std::fs::read_to_string(&path) {
            Ok(content) => {
                let lines: Vec<&str> = content.lines().collect();
                let start = (offset - 1).min(lines.len());
                let end = (start + limit).min(lines.len());
                let selected = &lines[start..end];

                let mut result = String::with_capacity(selected.len() * 80);
                for (i, line) in selected.iter().enumerate() {
                    let line_num = start + i + 1;
                    result.push_str(&format!("{line_num}\t{line}\n"));
                }

                if end < lines.len() {
                    result.push_str(&format!(
                        "\n[{} more lines, use offset={} to continue]",
                        lines.len() - end,
                        end + 1
                    ));
                }

                ToolResult::success(result)
            }
            Err(e) => ToolResult::error(format!("cannot read '{}': {}", path.display(), e)),
        }
    }

    fn required_capability(&self) -> Capability {
        Capability::FileRead { allowed_paths: self.allowed_paths.clone() }
    }
}

// ─── FileWriteTool ──────────────────────────────────────────

/// Write content to a file. Creates or overwrites.
///
/// Creates parent directories if they don't exist.
pub struct FileWriteTool {
    allowed_paths: Vec<String>,
}

impl FileWriteTool {
    pub fn new(allowed_paths: Vec<String>) -> Self {
        Self { allowed_paths }
    }
}

#[async_trait]
impl Tool for FileWriteTool {
    fn name(&self) -> &'static str {
        "file_write"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "file_write".into(),
            description: "Create or overwrite a file with the given content.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "required": ["path", "content"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file"
                    },
                    "content": {
                        "type": "string",
                        "description": "File content to write"
                    }
                }
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let path_str = match input.get("path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return ToolResult::error("missing required field 'path'"),
        };

        let content = match input.get("content").and_then(|v| v.as_str()) {
            Some(c) => c,
            None => return ToolResult::error("missing required field 'content'"),
        };

        let path = match validate_write_path(path_str, &self.allowed_paths) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(e),
        };

        // Create parent directories
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    return ToolResult::error(format!(
                        "cannot create directory '{}': {}",
                        parent.display(),
                        e
                    ));
                }
            }
        }

        match std::fs::write(&path, content) {
            Ok(()) => {
                ToolResult::success(format!("Wrote {} bytes to {}", content.len(), path.display()))
            }
            Err(e) => ToolResult::error(format!("cannot write '{}': {}", path.display(), e)),
        }
    }

    fn required_capability(&self) -> Capability {
        Capability::FileWrite { allowed_paths: self.allowed_paths.clone() }
    }
}

// ─── FileEditTool ───────────────────────────────────────────

/// Edit a file by replacing a unique string.
///
/// Semantics match Claude Code's Edit tool:
/// - `old_string` must appear exactly once in the file
/// - `new_string` replaces it
/// - If `old_string` appears 0 or >1 times, the edit fails
pub struct FileEditTool {
    allowed_paths: Vec<String>,
}

impl FileEditTool {
    pub fn new(allowed_paths: Vec<String>) -> Self {
        Self { allowed_paths }
    }
}

#[async_trait]
impl Tool for FileEditTool {
    fn name(&self) -> &'static str {
        "file_edit"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "file_edit".into(),
            description: "Replace a unique string in a file. old_string must appear exactly once."
                .into(),
            input_schema: serde_json::json!({
                "type": "object",
                "required": ["path", "old_string", "new_string"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file"
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Exact string to find (must be unique in the file)"
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement string"
                    }
                }
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let path_str = match input.get("path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return ToolResult::error("missing required field 'path'"),
        };

        let old_string = match input.get("old_string").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return ToolResult::error("missing required field 'old_string'"),
        };

        let new_string = match input.get("new_string").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return ToolResult::error("missing required field 'new_string'"),
        };

        if old_string == new_string {
            return ToolResult::error("old_string and new_string are identical");
        }

        let path = match validate_path(path_str, &self.allowed_paths) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(e),
        };

        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(e) => return ToolResult::error(format!("cannot read '{}': {}", path.display(), e)),
        };

        let count = content.matches(old_string).count();
        match count {
            0 => ToolResult::error(format!(
                "old_string not found in {}. Provide more context to match.",
                path.display()
            )),
            1 => {
                let new_content = content.replacen(old_string, new_string, 1);
                match std::fs::write(&path, &new_content) {
                    Ok(()) => ToolResult::success(format!(
                        "Edited {}. Replaced 1 occurrence ({} bytes → {} bytes).",
                        path.display(),
                        old_string.len(),
                        new_string.len()
                    )),
                    Err(e) => {
                        ToolResult::error(format!("cannot write '{}': {}", path.display(), e))
                    }
                }
            }
            n => ToolResult::error(format!(
                "old_string found {} times in {}. Provide more context to make it unique.",
                n,
                path.display()
            )),
        }
    }

    fn required_capability(&self) -> Capability {
        Capability::FileWrite { allowed_paths: self.allowed_paths.clone() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn temp_file(dir: &Path, name: &str, content: &str) -> PathBuf {
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        path
    }

    // ─── FileReadTool tests ─────────────────────────────

    #[tokio::test]
    async fn test_file_read_basic() {
        let dir = TempDir::new().unwrap();
        let path = temp_file(dir.path(), "test.txt", "line1\nline2\nline3\n");
        let tool = FileReadTool::new(vec!["*".into()]);

        let result = tool.execute(serde_json::json!({"path": path.to_str().unwrap()})).await;
        assert!(!result.is_error, "error: {}", result.content);
        assert!(result.content.contains("1\tline1"));
        assert!(result.content.contains("2\tline2"));
        assert!(result.content.contains("3\tline3"));
    }

    #[tokio::test]
    async fn test_file_read_with_offset_and_limit() {
        let dir = TempDir::new().unwrap();
        let content: String = (1..=100).map(|i| format!("line{i}\n")).collect();
        let path = temp_file(dir.path(), "big.txt", &content);
        let tool = FileReadTool::new(vec!["*".into()]);

        let result = tool
            .execute(serde_json::json!({"path": path.to_str().unwrap(), "offset": 50, "limit": 5}))
            .await;
        assert!(!result.is_error);
        assert!(result.content.contains("50\tline50"));
        assert!(result.content.contains("54\tline54"));
        assert!(!result.content.contains("55\tline55"));
    }

    #[tokio::test]
    async fn test_file_read_nonexistent() {
        let tool = FileReadTool::new(vec!["*".into()]);
        let result = tool.execute(serde_json::json!({"path": "/nonexistent_file_xyz"})).await;
        assert!(result.is_error);
        assert!(result.content.contains("cannot resolve"));
    }

    #[tokio::test]
    async fn test_file_read_missing_path_field() {
        let tool = FileReadTool::new(vec!["*".into()]);
        let result = tool.execute(serde_json::json!({"file": "test.txt"})).await;
        assert!(result.is_error);
        assert!(result.content.contains("missing"));
    }

    #[tokio::test]
    async fn test_file_read_path_restricted() {
        let dir = TempDir::new().unwrap();
        let path = temp_file(dir.path(), "secret.txt", "secret data");
        let tool = FileReadTool::new(vec!["/nonexistent_allowed_prefix".into()]);

        let result = tool.execute(serde_json::json!({"path": path.to_str().unwrap()})).await;
        assert!(result.is_error);
        assert!(result.content.contains("outside allowed"));
    }

    // ─── FileWriteTool tests ────────────────────────────

    #[tokio::test]
    async fn test_file_write_create() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("new_file.txt");
        let tool = FileWriteTool::new(vec!["*".into()]);

        let result = tool
            .execute(serde_json::json!({"path": path.to_str().unwrap(), "content": "hello world"}))
            .await;
        assert!(!result.is_error, "error: {}", result.content);
        assert!(result.content.contains("11 bytes"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello world");
    }

    #[tokio::test]
    async fn test_file_write_overwrite() {
        let dir = TempDir::new().unwrap();
        let path = temp_file(dir.path(), "existing.txt", "old content");
        let tool = FileWriteTool::new(vec!["*".into()]);

        let result = tool
            .execute(serde_json::json!({"path": path.to_str().unwrap(), "content": "new content"}))
            .await;
        assert!(!result.is_error);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "new content");
    }

    #[tokio::test]
    async fn test_file_write_path_restricted() {
        let tool = FileWriteTool::new(vec!["/nonexistent_allowed_prefix".into()]);
        let result =
            tool.execute(serde_json::json!({"path": "/tmp/evil.txt", "content": "bad"})).await;
        assert!(result.is_error);
        assert!(result.content.contains("outside allowed"));
    }

    #[tokio::test]
    async fn test_file_write_missing_content() {
        let tool = FileWriteTool::new(vec!["*".into()]);
        let result = tool.execute(serde_json::json!({"path": "/tmp/test.txt"})).await;
        assert!(result.is_error);
        assert!(result.content.contains("missing"));
    }

    // ─── FileEditTool tests ─────────────────────────────

    #[tokio::test]
    async fn test_file_edit_unique_match() {
        let dir = TempDir::new().unwrap();
        let path = temp_file(dir.path(), "code.rs", "fn main() {\n    println!(\"hello\");\n}\n");
        let tool = FileEditTool::new(vec!["*".into()]);

        let result = tool
            .execute(serde_json::json!({
                "path": path.to_str().unwrap(),
                "old_string": "println!(\"hello\")",
                "new_string": "println!(\"world\")"
            }))
            .await;
        assert!(!result.is_error, "error: {}", result.content);
        assert!(result.content.contains("Replaced 1 occurrence"));

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("println!(\"world\")"));
        assert!(!content.contains("println!(\"hello\")"));
    }

    #[tokio::test]
    async fn test_file_edit_no_match() {
        let dir = TempDir::new().unwrap();
        let path = temp_file(dir.path(), "code.rs", "fn main() {}\n");
        let tool = FileEditTool::new(vec!["*".into()]);

        let result = tool
            .execute(serde_json::json!({
                "path": path.to_str().unwrap(),
                "old_string": "nonexistent string",
                "new_string": "replacement"
            }))
            .await;
        assert!(result.is_error);
        assert!(result.content.contains("not found"));
    }

    #[tokio::test]
    async fn test_file_edit_multiple_matches() {
        let dir = TempDir::new().unwrap();
        let path = temp_file(dir.path(), "code.rs", "let x = 1;\nlet y = 1;\n");
        let tool = FileEditTool::new(vec!["*".into()]);

        let result = tool
            .execute(serde_json::json!({
                "path": path.to_str().unwrap(),
                "old_string": "= 1",
                "new_string": "= 2"
            }))
            .await;
        assert!(result.is_error);
        assert!(result.content.contains("2 times"));
    }

    #[tokio::test]
    async fn test_file_edit_identical_strings() {
        let dir = TempDir::new().unwrap();
        let path = temp_file(dir.path(), "code.rs", "hello\n");
        let tool = FileEditTool::new(vec!["*".into()]);

        let result = tool
            .execute(serde_json::json!({
                "path": path.to_str().unwrap(),
                "old_string": "hello",
                "new_string": "hello"
            }))
            .await;
        assert!(result.is_error);
        assert!(result.content.contains("identical"));
    }

    #[tokio::test]
    async fn test_file_edit_path_restricted() {
        let dir = TempDir::new().unwrap();
        let path = temp_file(dir.path(), "code.rs", "hello\n");
        let tool = FileEditTool::new(vec!["/nonexistent_allowed_prefix".into()]);

        let result = tool
            .execute(serde_json::json!({
                "path": path.to_str().unwrap(),
                "old_string": "hello",
                "new_string": "world"
            }))
            .await;
        assert!(result.is_error);
        assert!(result.content.contains("outside allowed"));
    }

    // ─── Capability tests ───────────────────────────────

    #[test]
    fn test_file_read_capability() {
        let tool = FileReadTool::new(vec!["/home".into()]);
        match tool.required_capability() {
            Capability::FileRead { allowed_paths } => {
                assert_eq!(allowed_paths, vec!["/home".to_string()]);
            }
            other => panic!("expected FileRead, got: {other:?}"),
        }
    }

    #[test]
    fn test_file_write_capability() {
        let tool = FileWriteTool::new(vec!["/tmp".into()]);
        match tool.required_capability() {
            Capability::FileWrite { allowed_paths } => {
                assert_eq!(allowed_paths, vec!["/tmp".to_string()]);
            }
            other => panic!("expected FileWrite, got: {other:?}"),
        }
    }

    #[test]
    fn test_file_edit_capability() {
        let tool = FileEditTool::new(vec!["/project".into()]);
        match tool.required_capability() {
            Capability::FileWrite { allowed_paths } => {
                assert_eq!(allowed_paths, vec!["/project".to_string()]);
            }
            other => panic!("expected FileWrite, got: {other:?}"),
        }
    }

    #[test]
    fn test_tool_names() {
        assert_eq!(FileReadTool::new(vec![]).name(), "file_read");
        assert_eq!(FileWriteTool::new(vec![]).name(), "file_write");
        assert_eq!(FileEditTool::new(vec![]).name(), "file_edit");
    }

    #[test]
    fn test_tool_schemas() {
        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(FileReadTool::new(vec![])),
            Box::new(FileWriteTool::new(vec![])),
            Box::new(FileEditTool::new(vec![])),
        ];
        for tool in &tools {
            let def = tool.definition();
            assert_eq!(def.input_schema["type"], "object");
            assert!(def.input_schema["required"].as_array().unwrap().iter().any(|v| v == "path"));
        }
    }
}
