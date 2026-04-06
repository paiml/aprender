//! Source context (Genchi Genbutsu)
//!
//! Source context for grounding content in reality.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Source context for grounding content in reality
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SourceContext {
    /// Paths to source files/directories
    pub paths: Vec<PathBuf>,
    /// Extracted content snippets
    pub snippets: Vec<SourceSnippet>,
    /// Total tokens used
    pub total_tokens: usize,
}

/// A snippet extracted from source material
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceSnippet {
    /// Source file path
    pub path: PathBuf,
    /// Line range (start, end)
    pub lines: Option<(usize, usize)>,
    /// Content
    pub content: String,
    /// Estimated tokens
    pub tokens: usize,
}

impl SourceContext {
    /// Create new empty source context
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a path to include
    pub fn add_path(&mut self, path: PathBuf) {
        self.paths.push(path);
    }

    /// Add a snippet
    pub fn add_snippet(&mut self, snippet: SourceSnippet) {
        self.total_tokens += snippet.tokens;
        self.snippets.push(snippet);
    }

    /// Format for inclusion in prompt
    pub fn format_for_prompt(&self) -> String {
        if self.snippets.is_empty() {
            return String::new();
        }

        let mut output = String::new();
        output.push_str("## Source Context (Genchi Genbutsu)\n\n");
        output.push_str("The following source material must be referenced in your output:\n\n");

        for snippet in &self.snippets {
            output.push_str(&format!("### {}\n", snippet.path.display()));
            if let Some((start, end)) = snippet.lines {
                output.push_str(&format!("Lines {}-{}:\n", start, end));
            }
            output.push_str("```\n");
            output.push_str(&snippet.content);
            output.push_str("\n```\n\n");
        }

        output.push_str("**Requirement**: Quote or reference specific line numbers from the source material above.\n\n");
        output
    }
}
