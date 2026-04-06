//! Tests for SourceContext and SourceSnippet

use crate::content::*;
use std::path::PathBuf;

#[test]
#[allow(non_snake_case)]
fn test_SOURCE_001_source_context_new() {
    let ctx = SourceContext::new();
    assert!(ctx.paths.is_empty());
    assert!(ctx.snippets.is_empty());
    assert_eq!(ctx.total_tokens, 0);
}

#[test]
#[allow(non_snake_case)]
fn test_SOURCE_002_source_context_add_path() {
    let mut ctx = SourceContext::new();
    ctx.add_path(PathBuf::from("/src/lib.rs"));
    assert_eq!(ctx.paths.len(), 1);
}

#[test]
#[allow(non_snake_case)]
fn test_SOURCE_003_source_context_add_snippet() {
    let mut ctx = SourceContext::new();
    ctx.add_snippet(SourceSnippet {
        path: PathBuf::from("/src/lib.rs"),
        lines: Some((1, 10)),
        content: "fn main() {}".to_string(),
        tokens: 50,
    });
    assert_eq!(ctx.snippets.len(), 1);
    assert_eq!(ctx.total_tokens, 50);
}

#[test]
#[allow(non_snake_case)]
fn test_SOURCE_004_source_context_format_empty() {
    let ctx = SourceContext::new();
    assert!(ctx.format_for_prompt().is_empty());
}

#[test]
#[allow(non_snake_case)]
fn test_SOURCE_005_source_context_format_with_snippets() {
    let mut ctx = SourceContext::new();
    ctx.add_snippet(SourceSnippet {
        path: PathBuf::from("/src/lib.rs"),
        lines: Some((1, 10)),
        content: "fn main() {}".to_string(),
        tokens: 50,
    });
    let formatted = ctx.format_for_prompt();
    assert!(formatted.contains("Source Context"));
    assert!(formatted.contains("Genchi Genbutsu"));
    assert!(formatted.contains("lib.rs"));
    assert!(formatted.contains("fn main()"));
}
