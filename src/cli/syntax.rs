//! Syntax highlighting for code output using syntect.
//!
//! Provides ANSI-colored syntax highlighting for Rust, Python, Go, and TypeScript code.
//!
//! Note: These utilities are available for future CLI enhancements.

#![cfg(feature = "syntect")]
use std::sync::LazyLock;

use syntect::easy::HighlightLines;
use syntect::highlighting::{Style, ThemeSet};
use syntect::parsing::SyntaxSet;
use syntect::util::{as_24_bit_terminal_escaped, LinesWithEndings};

/// Global syntax set (loaded once, reused)
static SYNTAX_SET: LazyLock<SyntaxSet> = LazyLock::new(SyntaxSet::load_defaults_newlines);

/// Global theme set
static THEME_SET: LazyLock<ThemeSet> = LazyLock::new(ThemeSet::load_defaults);

/// Supported languages for highlighting
#[derive(Debug, Clone, Copy)]
pub enum Language {
    Rust,
    Python,
    Go,
    TypeScript,
    JavaScript,
    Markdown,
    Toml,
    Json,
    Shell,
}

impl Language {
    /// Get the syntect syntax name for this language
    fn syntax_name(&self) -> &'static str {
        match self {
            Language::Rust => "Rust",
            Language::Python => "Python",
            Language::Go => "Go",
            Language::TypeScript => "TypeScript",
            Language::JavaScript => "JavaScript",
            Language::Markdown => "Markdown",
            Language::Toml => "TOML",
            Language::Json => "JSON",
            Language::Shell => "Bourne Again Shell (bash)",
        }
    }

    /// Detect language from file extension
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "rs" => Some(Language::Rust),
            "py" => Some(Language::Python),
            "go" => Some(Language::Go),
            "ts" | "tsx" => Some(Language::TypeScript),
            "js" | "jsx" => Some(Language::JavaScript),
            "md" => Some(Language::Markdown),
            "toml" => Some(Language::Toml),
            "json" => Some(Language::Json),
            "sh" | "bash" => Some(Language::Shell),
            _ => None,
        }
    }
}

/// Highlight a code block and return ANSI-escaped string.
///
/// Uses the "base16-ocean.dark" theme which works well in terminals.
pub fn highlight_code(code: &str, lang: Language) -> String {
    let syntax = SYNTAX_SET
        .find_syntax_by_name(lang.syntax_name())
        .unwrap_or_else(|| SYNTAX_SET.find_syntax_plain_text());

    let theme = &THEME_SET.themes["base16-ocean.dark"];
    let mut highlighter = HighlightLines::new(syntax, theme);
    let mut output = String::new();

    for line in LinesWithEndings::from(code) {
        let ranges: Vec<(Style, &str)> =
            highlighter.highlight_line(line, &SYNTAX_SET).unwrap_or_default();
        let escaped = as_24_bit_terminal_escaped(&ranges, false);
        output.push_str(&escaped);
    }

    // Reset terminal colors at the end
    output.push_str("\x1b[0m");
    output
}

/// Highlight a single line of code and return ANSI-escaped string.
pub fn highlight_line(line: &str, lang: Language) -> String {
    let syntax = SYNTAX_SET
        .find_syntax_by_name(lang.syntax_name())
        .unwrap_or_else(|| SYNTAX_SET.find_syntax_plain_text());

    let theme = &THEME_SET.themes["base16-ocean.dark"];
    let mut highlighter = HighlightLines::new(syntax, theme);

    let ranges: Vec<(Style, &str)> =
        highlighter.highlight_line(line, &SYNTAX_SET).unwrap_or_default();

    let mut escaped = as_24_bit_terminal_escaped(&ranges, false);
    // Reset and remove trailing newline artifacts
    escaped.push_str("\x1b[0m");
    escaped.trim_end().to_string()
}

/// Highlight Rust code (convenience function).
pub fn highlight_rust(code: &str) -> String {
    highlight_code(code, Language::Rust)
}

/// Highlight a single line of Rust code (convenience function).
pub fn highlight_rust_line(line: &str) -> String {
    highlight_line(line, Language::Rust)
}

/// Print highlighted code with optional indentation.
pub fn print_highlighted(code: &str, lang: Language, indent: &str) {
    let syntax = SYNTAX_SET
        .find_syntax_by_name(lang.syntax_name())
        .unwrap_or_else(|| SYNTAX_SET.find_syntax_plain_text());

    let theme = &THEME_SET.themes["base16-ocean.dark"];
    let mut highlighter = HighlightLines::new(syntax, theme);

    for line in LinesWithEndings::from(code) {
        let ranges: Vec<(Style, &str)> =
            highlighter.highlight_line(line, &SYNTAX_SET).unwrap_or_default();
        let escaped = as_24_bit_terminal_escaped(&ranges, false);
        print!("{}{}", indent, escaped);
    }
    // Reset terminal colors
    print!("\x1b[0m");
}

/// Print a single highlighted line with optional indentation.
pub fn print_highlighted_line(line: &str, lang: Language, indent: &str) {
    let highlighted = highlight_line(line, lang);
    println!("{}{}", indent, highlighted);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_highlight_rust_code() {
        let code = r#"fn main() {
    println!("Hello, world!");
}"#;
        let highlighted = highlight_rust(code);
        // Should contain ANSI escape codes
        assert!(highlighted.contains("\x1b["));
        // Should end with reset
        assert!(highlighted.ends_with("\x1b[0m"));
    }

    #[test]
    fn test_highlight_rust_line() {
        let line = "let x = 42;";
        let highlighted = highlight_rust_line(line);
        assert!(highlighted.contains("\x1b["));
    }

    #[test]
    fn test_language_from_extension() {
        assert!(matches!(Language::from_extension("rs"), Some(Language::Rust)));
        assert!(matches!(Language::from_extension("py"), Some(Language::Python)));
        assert!(matches!(Language::from_extension("go"), Some(Language::Go)));
        assert!(matches!(Language::from_extension("ts"), Some(Language::TypeScript)));
        assert!(Language::from_extension("xyz").is_none());
    }
}
