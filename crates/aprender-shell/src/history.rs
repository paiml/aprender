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
    fn parse_line(&self, line: &str) -> Option<String> {
        let line = line.trim();

        if line.is_empty() {
            return None;
        }

        // ZSH extended history format: ": timestamp:0;command"
        if line.starts_with(": ") {
            if let Some(pos) = line.find(';') {
                return Some(line[pos + 1..].to_string());
            }
        }

        // Fish history format: "- cmd: command"
        if line.starts_with("- cmd: ") {
            return Some(line[7..].to_string());
        }

        // Plain format (bash)
        Some(line.to_string())
    }

    /// Filter out commands we don't want to learn
    fn is_valid_command(&self, cmd: &str) -> bool {
        // Skip very short commands
        if cmd.len() < 2 {
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
}
