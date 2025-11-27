//! Shell history file parsing

use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

/// Parser for shell history files
pub struct HistoryParser;

impl HistoryParser {
    pub fn new() -> Self {
        Self
    }

    /// Auto-detect the shell history file
    pub fn find_history_file() -> Option<PathBuf> {
        let home = dirs::home_dir()?;

        // Try in order of preference
        let candidates = [
            home.join(".zsh_history"),
            home.join(".bash_history"),
            home.join(".local/share/fish/fish_history"),
            home.join(".history"),
        ];

        candidates.into_iter().find(|p| p.exists())
    }

    /// Parse a history file into commands
    pub fn parse_file(&self, path: &PathBuf) -> std::io::Result<Vec<String>> {
        let mut file = File::open(path)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;

        // Convert to string, replacing invalid UTF-8 with replacement char
        let content = String::from_utf8_lossy(&bytes);

        let mut commands = Vec::new();

        for line in content.lines() {
            if let Some(cmd) = self.parse_line(line) {
                if !cmd.is_empty() && self.is_valid_command(&cmd) {
                    commands.push(cmd);
                }
            }
        }

        Ok(commands)
    }

    /// Parse a single history line (handles zsh extended format)
    ///
    /// Handles:
    /// - ZSH extended format (`: timestamp:0;command`)
    /// - Fish format (`- cmd: command`)
    /// - Plain format (bash)
    /// - Comment stripping
    /// - Shell no-op filtering
    fn parse_line(&self, line: &str) -> Option<String> {
        let line = line.trim();

        if line.is_empty() {
            return None;
        }

        // Skip comment-only lines (issue #91)
        if line.starts_with('#') {
            return None;
        }

        // Skip shell no-ops (issue #91)
        // `: ` is a valid ZSH timestamp prefix, but `: ` followed by non-numeric is a no-op
        if line == ":"
            || (line.starts_with(": ") && !line.chars().nth(2).is_some_and(|c| c.is_ascii_digit()))
        {
            // Check if it's NOT a ZSH timestamp (starts with digit after ": ")
            if !line.starts_with(": ") || !line.chars().nth(2).is_some_and(|c| c.is_ascii_digit()) {
                return None;
            }
        }

        // ZSH extended history format: ": timestamp:0;command"
        if line.starts_with(": ") {
            if let Some(pos) = line.find(';') {
                let cmd = &line[pos + 1..];
                return Some(self.strip_inline_comment(cmd));
            }
        }

        // Fish history format: "- cmd: command"
        if let Some(cmd) = line.strip_prefix("- cmd: ") {
            return Some(self.strip_inline_comment(cmd));
        }

        // Plain format (bash) - strip inline comments
        Some(self.strip_inline_comment(line))
    }

    /// Strip inline comments from commands while preserving quoted strings
    ///
    /// # Examples
    /// - `git status # check` -> `git status`
    /// - `echo "hello #world"` -> `echo "hello #world"` (preserved in quotes)
    /// - `gh issue view #123` -> `gh issue view #123` (preserved - issue number)
    fn strip_inline_comment(&self, cmd: &str) -> String {
        let mut result = String::with_capacity(cmd.len());
        let mut in_single_quote = false;
        let mut in_double_quote = false;
        let chars: Vec<char> = cmd.chars().collect();

        let mut i = 0;
        while i < chars.len() {
            let ch = chars[i];
            let prev_char = if i > 0 { chars[i - 1] } else { '\0' };
            let next_char = chars.get(i + 1).copied();

            // Handle quote state
            if ch == '\'' && !in_double_quote && prev_char != '\\' {
                in_single_quote = !in_single_quote;
            } else if ch == '"' && !in_single_quote && prev_char != '\\' {
                in_double_quote = !in_double_quote;
            }

            // Check for inline comment (# preceded by whitespace, not in quotes)
            // BUT preserve #123 style issue numbers (# followed by digit)
            if ch == '#' && !in_single_quote && !in_double_quote && prev_char.is_whitespace() {
                // Check if # is followed by a digit (issue number like #123)
                if let Some(next) = next_char {
                    if next.is_ascii_digit() {
                        // This is an issue number, preserve it
                        result.push(ch);
                        i += 1;
                        continue;
                    }
                }
                // Found inline comment, stop here and trim trailing whitespace
                return result.trim_end().to_string();
            }

            result.push(ch);
            i += 1;
        }

        result.trim().to_string()
    }

    /// Filter out commands we don't want to learn
    fn is_valid_command(&self, cmd: &str) -> bool {
        // Skip very short commands
        if cmd.len() < 2 {
            return false;
        }

        // Skip malformed/incomplete commands (multiline artifacts)
        if self.is_malformed(cmd) {
            return false;
        }

        // Skip corrupted commands (missing spaces before flags)
        if self.has_corrupted_tokens(cmd) {
            return false;
        }

        // Skip commands with sensitive patterns
        let sensitive = [
            "password",
            "passwd",
            "secret",
            "token",
            "api_key",
            "AWS_SECRET",
            "GITHUB_TOKEN",
            "Authorization:",
        ];

        let cmd_lower = cmd.to_lowercase();
        for pattern in sensitive {
            if cmd_lower.contains(&pattern.to_lowercase()) {
                return false;
            }
        }

        // Skip history manipulation
        if cmd.starts_with("history") || cmd.starts_with("fc ") {
            return false;
        }

        true
    }

    /// Check for malformed commands (incomplete multiline, etc.)
    fn is_malformed(&self, cmd: &str) -> bool {
        let trimmed = cmd.trim();

        // Lone backslash or backslash with whitespace
        if trimmed == "\\" || trimmed.ends_with("\\ ") {
            return true;
        }

        // Incomplete brace/bracket patterns
        if trimmed.starts_with('}') || trimmed.starts_with(')') || trimmed.starts_with(']') {
            return true;
        }

        false
    }

    /// Check for corrupted tokens like "commit-m" (missing space before flag)
    fn has_corrupted_tokens(&self, cmd: &str) -> bool {
        // Common subcommands that should never have flags directly attached
        let subcommands = [
            "commit", "checkout", "clone", "push", "pull", "merge", "rebase", "status", "add",
            "build", "run", "test", "install",
        ];

        for token in cmd.split_whitespace() {
            if let Some(dash_pos) = token.find('-') {
                if dash_pos > 0 && dash_pos < token.len() - 1 {
                    let before = &token[..dash_pos];
                    let after = &token[dash_pos + 1..];

                    // Pattern: subcommand-flag (e.g., "commit-m", "add-A")
                    if subcommands.contains(&before) && (after.len() <= 2 || after.starts_with('-'))
                    {
                        return true;
                    }
                }
            }
        }

        false
    }
}

impl Default for HistoryParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_zsh_extended() {
        let parser = HistoryParser::new();
        let line = ": 1699900000:0;git status";
        assert_eq!(parser.parse_line(line), Some("git status".to_string()));
    }

    #[test]
    fn test_parse_bash() {
        let parser = HistoryParser::new();
        let line = "ls -la";
        assert_eq!(parser.parse_line(line), Some("ls -la".to_string()));
    }

    #[test]
    fn test_filter_sensitive() {
        let parser = HistoryParser::new();
        assert!(!parser.is_valid_command("export API_KEY=secret123"));
        assert!(!parser.is_valid_command("echo $PASSWORD"));
        assert!(parser.is_valid_command("git push origin main"));
    }

    #[test]
    fn test_filter_short() {
        let parser = HistoryParser::new();
        assert!(!parser.is_valid_command("l"));
        assert!(parser.is_valid_command("ls"));
    }

    // ==================== EXTREME TDD: Corrupted Command Filtering ====================

    #[test]
    fn test_filter_corrupted_commands() {
        let parser = HistoryParser::new();

        // Corrupted: missing space before flag
        assert!(
            !parser.is_valid_command("git commit-m test"),
            "Should reject 'commit-m' (missing space)"
        );
        assert!(
            !parser.is_valid_command("git add-A"),
            "Should reject 'add-A' (missing space)"
        );
        assert!(
            !parser.is_valid_command("cargo build-r"),
            "Should reject 'build-r' (missing space)"
        );

        // Valid: proper spacing
        assert!(
            parser.is_valid_command("git commit -m test"),
            "Should accept 'commit -m' (proper spacing)"
        );
        assert!(
            parser.is_valid_command("git add -A"),
            "Should accept 'add -A' (proper spacing)"
        );

        // Valid: legitimate hyphenated words
        assert!(
            parser.is_valid_command("git checkout feature-branch"),
            "Should accept 'feature-branch' (legitimate hyphen)"
        );
        assert!(
            parser.is_valid_command("npm install lodash-es"),
            "Should accept 'lodash-es' (package name)"
        );
    }

    #[test]
    fn test_filter_malformed_multiline() {
        let parser = HistoryParser::new();

        // ZSH sometimes captures incomplete multiline commands
        assert!(
            !parser.is_valid_command("}\\ "),
            "Should reject incomplete multiline"
        );
        assert!(
            !parser.is_valid_command("\\"),
            "Should reject lone backslash"
        );
    }

    // ==================== Issue #91: History Parsing Fixes ====================

    #[test]
    fn test_skip_comment_lines() {
        let parser = HistoryParser::new();
        assert!(parser.parse_line("# this is a comment").is_none());
        assert!(parser.parse_line("  # indented comment").is_none());
        assert!(parser.parse_line("#").is_none());
    }

    #[test]
    fn test_skip_shell_noops() {
        let parser = HistoryParser::new();
        assert!(parser.parse_line(":").is_none());
        assert!(parser.parse_line(": ignored text").is_none());
        assert!(parser.parse_line(":  spaces after").is_none());
    }

    #[test]
    fn test_preserve_zsh_timestamp_format() {
        let parser = HistoryParser::new();
        // ZSH timestamp format should NOT be treated as no-op
        assert_eq!(
            parser.parse_line(": 1699900000:0;git status"),
            Some("git status".to_string())
        );
    }

    #[test]
    fn test_strip_inline_comments() {
        let parser = HistoryParser::new();
        assert_eq!(
            parser.parse_line("git status # check status"),
            Some("git status".to_string())
        );
        assert_eq!(
            parser.parse_line("cargo build --release # optimized"),
            Some("cargo build --release".to_string())
        );
    }

    #[test]
    fn test_preserve_hash_in_arguments() {
        let parser = HistoryParser::new();
        // Issue numbers should be preserved (no space before #)
        assert_eq!(
            parser.parse_line("gh issue view #123"),
            Some("gh issue view #123".to_string())
        );
        // Quoted strings should preserve #
        assert_eq!(
            parser.parse_line(r#"echo "hello #world""#),
            Some(r#"echo "hello #world""#.to_string())
        );
        assert_eq!(
            parser.parse_line("echo '#hashtag'"),
            Some("echo '#hashtag'".to_string())
        );
    }

    #[test]
    fn test_inline_comment_with_quotes() {
        let parser = HistoryParser::new();
        // Comment after quoted string
        assert_eq!(
            parser.parse_line(r#"echo "hello" # comment"#),
            Some(r#"echo "hello""#.to_string())
        );
    }
}
