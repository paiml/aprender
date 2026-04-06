//! Literate Document with Typst support (ENT-021)
//!
//! Provides literate programming document support with code block extraction.

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

/// Regex for extracting code blocks from Typst documents
static TYPST_CODE_BLOCK: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"```(\w*)\n([\s\S]*?)```").expect("Invalid Typst code block regex")
});

/// Regex for extracting code blocks from Markdown
static MARKDOWN_CODE_BLOCK: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"```(\w*)\n([\s\S]*?)```").expect("Invalid Markdown code block regex")
});

/// A code block extracted from a literate document
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodeBlock {
    /// Programming language (if specified)
    pub language: Option<String>,
    /// Code content
    pub content: String,
    /// Line number where block starts (1-indexed)
    pub line_number: usize,
}

impl CodeBlock {
    /// Create a new code block
    pub fn new(content: impl Into<String>, line_number: usize) -> Self {
        Self { language: None, content: content.into(), line_number }
    }

    /// Set the language
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        let lang = language.into();
        self.language = if lang.is_empty() { None } else { Some(lang) };
        self
    }
}

/// Literate document types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LiterateDocument {
    /// Typst document
    Typst(String),
    /// Markdown document
    Markdown(String),
    /// Raw text (no special parsing)
    RawText(String),
}

impl LiterateDocument {
    /// Parse a Typst document from string
    pub fn parse_typst(content: impl Into<String>) -> Self {
        Self::Typst(content.into())
    }

    /// Parse a Markdown document from string
    pub fn parse_markdown(content: impl Into<String>) -> Self {
        Self::Markdown(content.into())
    }

    /// Create a raw text document
    pub fn raw(content: impl Into<String>) -> Self {
        Self::RawText(content.into())
    }

    /// Get the raw content
    pub fn content(&self) -> &str {
        match self {
            Self::Typst(s) | Self::Markdown(s) | Self::RawText(s) => s,
        }
    }

    /// Extract code blocks from the document
    pub fn extract_code_blocks(&self) -> Vec<CodeBlock> {
        match self {
            Self::Typst(content) => extract_blocks_with_regex(content, &TYPST_CODE_BLOCK),
            Self::Markdown(content) => extract_blocks_with_regex(content, &MARKDOWN_CODE_BLOCK),
            Self::RawText(_) => Vec::new(),
        }
    }

    /// Convert to basic HTML representation
    pub fn to_html(&self) -> String {
        match self {
            Self::Typst(content) | Self::Markdown(content) => {
                let mut html = String::new();
                html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
                html.push_str("<meta charset=\"utf-8\">\n");
                html.push_str("<style>\n");
                html.push_str("body { font-family: system-ui, sans-serif; max-width: 800px; margin: 0 auto; padding: 2rem; }\n");
                html.push_str("pre { background: #f5f5f5; padding: 1rem; overflow-x: auto; }\n");
                html.push_str("code { font-family: monospace; }\n");
                html.push_str("</style>\n</head>\n<body>\n");

                // Simple conversion: paragraphs and code blocks
                let mut in_code_block = false;
                let mut code_lang = String::new();
                let mut code_content = String::new();

                for line in content.lines() {
                    if line.starts_with("```") {
                        if in_code_block {
                            // End code block
                            html.push_str("<pre><code");
                            if !code_lang.is_empty() {
                                html.push_str(&format!(" class=\"language-{code_lang}\""));
                            }
                            html.push('>');
                            html.push_str(&escape_html(&code_content));
                            html.push_str("</code></pre>\n");
                            code_content.clear();
                            code_lang.clear();
                            in_code_block = false;
                        } else {
                            // Start code block
                            code_lang = line.trim_start_matches('`').to_string();
                            in_code_block = true;
                        }
                    } else if in_code_block {
                        if !code_content.is_empty() {
                            code_content.push('\n');
                        }
                        code_content.push_str(line);
                    } else if line.starts_with('#') {
                        // Heading
                        let level = line.chars().take_while(|&c| c == '#').count().min(6);
                        let text = line.trim_start_matches('#').trim();
                        html.push_str(&format!("<h{level}>{}</h{level}>\n", escape_html(text)));
                    } else if line.is_empty() {
                        // Empty line
                    } else {
                        // Paragraph
                        html.push_str(&format!("<p>{}</p>\n", escape_html(line)));
                    }
                }

                html.push_str("</body>\n</html>");
                html
            }
            Self::RawText(content) => {
                format!(
                    "<!DOCTYPE html>\n<html>\n<body>\n<pre>{}</pre>\n</body>\n</html>",
                    escape_html(content)
                )
            }
        }
    }

    /// Check if this is a Typst document
    pub fn is_typst(&self) -> bool {
        matches!(self, Self::Typst(_))
    }

    /// Check if this is a Markdown document
    pub fn is_markdown(&self) -> bool {
        matches!(self, Self::Markdown(_))
    }

    /// Check if this is raw text
    pub fn is_raw(&self) -> bool {
        matches!(self, Self::RawText(_))
    }
}

/// Extract code blocks using a regex pattern
fn extract_blocks_with_regex(content: &str, pattern: &Regex) -> Vec<CodeBlock> {
    let mut blocks = Vec::new();

    for cap in pattern.captures_iter(content) {
        let full_match = cap.get(0).expect("capture group 0 always exists in a regex match");
        let lang = cap.get(1).map(|m| m.as_str().to_string());
        let code = cap.get(2).map(|m| m.as_str().to_string()).unwrap_or_default();

        // Calculate line number
        let line_number = content[..full_match.start()].chars().filter(|&c| c == '\n').count() + 1;

        let mut block = CodeBlock::new(code.trim_end(), line_number);
        if let Some(l) = lang {
            block = block.with_language(l);
        }
        blocks.push(block);
    }

    blocks
}

/// Escape HTML special characters
fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_typst_parsing() {
        let content = r#"
= Introduction

This is a Typst document.

```rust
fn main() {
    println!("Hello, world!");
}
```

More text here.
"#;

        let doc = LiterateDocument::parse_typst(content);
        assert!(doc.is_typst());
        assert!(doc.content().contains("Typst document"));
    }

    #[test]
    fn test_code_block_extraction() {
        let content = r#"
# My Document

Here's some code:

```python
def hello():
    print("Hello!")
```

And more:

```rust
fn main() {}
```
"#;

        let doc = LiterateDocument::parse_markdown(content);
        let blocks = doc.extract_code_blocks();

        assert_eq!(blocks.len(), 2);

        assert_eq!(blocks[0].language, Some("python".to_string()));
        assert!(blocks[0].content.contains("def hello()"));
        assert_eq!(blocks[0].line_number, 6);

        assert_eq!(blocks[1].language, Some("rust".to_string()));
        assert!(blocks[1].content.contains("fn main()"));
        assert_eq!(blocks[1].line_number, 13);
    }

    #[test]
    fn test_code_block_no_language() {
        let content = r"
```
plain code here
```
";

        let doc = LiterateDocument::parse_markdown(content);
        let blocks = doc.extract_code_blocks();

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].language, None);
        assert_eq!(blocks[0].content, "plain code here");
    }

    #[test]
    fn test_markdown_passthrough() {
        let content = "# Hello\n\nThis is markdown.";
        let doc = LiterateDocument::parse_markdown(content);

        assert!(doc.is_markdown());
        assert_eq!(doc.content(), content);
    }

    #[test]
    fn test_raw_text() {
        let content = "Just plain text";
        let doc = LiterateDocument::raw(content);

        assert!(doc.is_raw());
        assert_eq!(doc.content(), content);

        // Raw text should have no code blocks
        let blocks = doc.extract_code_blocks();
        assert!(blocks.is_empty());
    }

    #[test]
    fn test_to_html_basic() {
        let content = r"# Title

This is a paragraph.

```rust
fn main() {}
```
";

        let doc = LiterateDocument::parse_markdown(content);
        let html = doc.to_html();

        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("<h1>Title</h1>"));
        assert!(html.contains("<p>This is a paragraph.</p>"));
        assert!(html.contains("<pre><code class=\"language-rust\">"));
        assert!(html.contains("fn main()"));
    }

    #[test]
    fn test_to_html_escaping() {
        let content = "This has <script>alert('xss')</script> in it.";
        let doc = LiterateDocument::parse_markdown(content);
        let html = doc.to_html();

        assert!(!html.contains("<script>"));
        assert!(html.contains("&lt;script&gt;"));
    }

    #[test]
    fn test_raw_text_to_html() {
        let content = "Line 1\nLine 2";
        let doc = LiterateDocument::raw(content);
        let html = doc.to_html();

        assert!(html.contains("<pre>"));
        assert!(html.contains("Line 1\nLine 2"));
    }

    #[test]
    fn test_multiple_headings() {
        let content = "# H1\n## H2\n### H3";
        let doc = LiterateDocument::parse_markdown(content);
        let html = doc.to_html();

        assert!(html.contains("<h1>H1</h1>"));
        assert!(html.contains("<h2>H2</h2>"));
        assert!(html.contains("<h3>H3</h3>"));
    }

    #[test]
    fn test_code_block_struct() {
        let block = CodeBlock::new("let x = 1;", 10).with_language("rust");

        assert_eq!(block.language, Some("rust".to_string()));
        assert_eq!(block.content, "let x = 1;");
        assert_eq!(block.line_number, 10);
    }

    #[test]
    fn test_empty_language_becomes_none() {
        let block = CodeBlock::new("code", 1).with_language("");
        assert_eq!(block.language, None);
    }

    #[test]
    fn test_typst_code_extraction() {
        let content = r"
= Typst Document

#set text(size: 12pt)

```python
import numpy as np
x = np.array([1, 2, 3])
```

More content here.
";

        let doc = LiterateDocument::parse_typst(content);
        let blocks = doc.extract_code_blocks();

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].language, Some("python".to_string()));
        assert!(blocks[0].content.contains("import numpy"));
    }
}
