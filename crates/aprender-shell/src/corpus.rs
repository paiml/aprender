//! Corpus management for synthetic shell command training
//!
//! Provides utilities for loading, validating, and managing training corpora
//! for shell completion models. Supports privacy-safe synthetic data generation.

use std::collections::HashSet;
use std::path::Path;

/// Patterns that indicate sensitive data - these should NEVER appear in a public corpus
const SENSITIVE_PATTERNS: &[&str] = &[
    // Credentials (specific patterns to avoid false positives)
    "password=",
    "password:",
    "passwd=",
    "passwd:",
    "PASSWORD",
    "PASSWD",
    // Secret patterns (avoid catching "kubectl create secret")
    "secret=",
    "secret:",
    "SECRET_",
    "_SECRET",
    // API keys
    "api_key",
    "apikey",
    "api-key",
    "bearer",
    // Auth patterns (more specific to avoid --author false positive)
    "auth_token",
    "auth=",
    "AUTH_",
    "authorization:",
    // AWS
    "AKIA",
    "aws_access",
    "aws_secret",
    // Private paths
    "/home/",
    "/Users/",
    "C:\\Users\\",
    // Hostnames
    ".internal",
    ".local",
    ".corp",
    // SSH
    "id_rsa",
    "id_ed25519",
    ".pem",
    // Environment
    "export ",
    "ENV=",
];

/// Result type for corpus operations
pub type CorpusResult<T> = Result<T, CorpusError>;

/// Errors that can occur during corpus operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CorpusError {
    /// Corpus file not found
    NotFound(String),
    /// Corpus contains sensitive patterns
    SensitiveData { line: usize, pattern: String },
    /// Corpus is empty
    Empty,
    /// Invalid format
    InvalidFormat(String),
    /// IO error
    IoError(String),
}

impl std::fmt::Display for CorpusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(path) => write!(f, "Corpus not found: {path}"),
            Self::SensitiveData { line, pattern } => {
                write!(f, "Sensitive pattern '{pattern}' found at line {line}")
            }
            Self::Empty => write!(f, "Corpus is empty"),
            Self::InvalidFormat(msg) => write!(f, "Invalid format: {msg}"),
            Self::IoError(msg) => write!(f, "IO error: {msg}"),
        }
    }
}

impl std::error::Error for CorpusError {}

/// A validated corpus of shell commands
#[derive(Debug, Clone)]
pub struct Corpus {
    /// Commands in the corpus
    commands: Vec<String>,
    /// Unique command prefixes for coverage analysis
    prefixes: HashSet<String>,
}

impl Corpus {
    /// Load corpus from a file, validating for sensitive data
    pub fn load<P: AsRef<Path>>(path: P) -> CorpusResult<Self> {
        let path = path.as_ref();
        let content =
            std::fs::read_to_string(path).map_err(|e| CorpusError::IoError(e.to_string()))?;

        Self::from_string(&content)
    }

    /// Create corpus from string content
    pub fn from_string(content: &str) -> CorpusResult<Self> {
        let mut commands = Vec::new();

        for (line_num, line) in content.lines().enumerate() {
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Validate for sensitive patterns
            Self::validate_line(line, line_num + 1)?;

            commands.push(line.to_string());
        }

        if commands.is_empty() {
            return Err(CorpusError::Empty);
        }

        // Build prefix set for coverage analysis
        let prefixes: HashSet<String> = commands
            .iter()
            .filter_map(|cmd| cmd.split_whitespace().next())
            .map(String::from)
            .collect();

        Ok(Self { commands, prefixes })
    }

    /// Validate a single line for sensitive patterns
    fn validate_line(line: &str, line_num: usize) -> CorpusResult<()> {
        let lower = line.to_lowercase();

        for pattern in SENSITIVE_PATTERNS {
            if lower.contains(&pattern.to_lowercase()) {
                return Err(CorpusError::SensitiveData {
                    line: line_num,
                    pattern: (*pattern).to_string(),
                });
            }
        }

        Ok(())
    }

    /// Get commands for training
    pub fn commands(&self) -> &[String] {
        &self.commands
    }

    /// Get number of commands
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    /// Check if corpus is empty
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Get unique command prefixes (first word of each command)
    pub fn prefixes(&self) -> &HashSet<String> {
        &self.prefixes
    }

    /// Get coverage statistics
    pub fn coverage_stats(&self) -> CorpusStats {
        let mut token_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        let mut total_tokens = 0;

        for cmd in &self.commands {
            for token in cmd.split_whitespace() {
                *token_counts.entry(token.to_string()).or_insert(0) += 1;
                total_tokens += 1;
            }
        }

        CorpusStats {
            total_commands: self.commands.len(),
            unique_prefixes: self.prefixes.len(),
            unique_tokens: token_counts.len(),
            total_tokens,
        }
    }
}

/// Statistics about a corpus
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorpusStats {
    /// Total number of commands
    pub total_commands: usize,
    /// Number of unique command prefixes
    pub unique_prefixes: usize,
    /// Number of unique tokens
    pub unique_tokens: usize,
    /// Total tokens across all commands
    pub total_tokens: usize,
}

// ============================================================================
// TESTS - EXTREME TDD
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Corpus Loading Tests
    // ========================================================================

    #[test]
    fn test_corpus_from_string_basic() {
        let content = "git status\ngit commit -m message\ncargo build";
        let corpus = Corpus::from_string(content).expect("should parse");

        assert_eq!(corpus.len(), 3);
        assert_eq!(corpus.commands()[0], "git status");
        assert_eq!(corpus.commands()[2], "cargo build");
    }

    #[test]
    fn test_corpus_skips_empty_lines() {
        let content = "git status\n\n\ncargo build\n";
        let corpus = Corpus::from_string(content).expect("should parse");

        assert_eq!(corpus.len(), 2);
    }

    #[test]
    fn test_corpus_skips_comments() {
        let content = "# This is a comment\ngit status\n# Another comment\ncargo build";
        let corpus = Corpus::from_string(content).expect("should parse");

        assert_eq!(corpus.len(), 2);
    }

    #[test]
    fn test_corpus_trims_whitespace() {
        let content = "  git status  \n\tcargo build\t";
        let corpus = Corpus::from_string(content).expect("should parse");

        assert_eq!(corpus.commands()[0], "git status");
        assert_eq!(corpus.commands()[1], "cargo build");
    }

    #[test]
    fn test_corpus_empty_returns_error() {
        let content = "\n\n# Only comments\n";
        let result = Corpus::from_string(content);

        assert!(matches!(result, Err(CorpusError::Empty)));
    }

    // ========================================================================
    // Sensitive Data Detection Tests (CRITICAL)
    // ========================================================================

    #[test]
    fn test_detects_password_pattern() {
        // PASSWORD in uppercase is detected
        let content = "mysql -u root PASSWORD=secret123";
        let result = Corpus::from_string(content);

        assert!(matches!(result, Err(CorpusError::SensitiveData { .. })));
    }

    #[test]
    fn test_detects_password_colon_pattern() {
        // password: pattern is detected
        let content = "curl -H 'password: secret123'";
        let result = Corpus::from_string(content);

        assert!(matches!(result, Err(CorpusError::SensitiveData { .. })));
    }

    #[test]
    fn test_detects_api_key_pattern() {
        let content = "curl -H 'api_key: secret123'";
        let result = Corpus::from_string(content);

        assert!(matches!(result, Err(CorpusError::SensitiveData { .. })));
    }

    #[test]
    fn test_detects_aws_key_pattern() {
        let content = "aws configure set aws_access_key_id AKIA123";
        let result = Corpus::from_string(content);

        assert!(matches!(result, Err(CorpusError::SensitiveData { .. })));
    }

    #[test]
    fn test_detects_home_path() {
        let content = "cd /home/username/projects";
        let result = Corpus::from_string(content);

        assert!(matches!(result, Err(CorpusError::SensitiveData { .. })));
    }

    #[test]
    fn test_detects_users_path_mac() {
        let content = "ls /Users/john/Documents";
        let result = Corpus::from_string(content);

        assert!(matches!(result, Err(CorpusError::SensitiveData { .. })));
    }

    #[test]
    fn test_detects_internal_hostname() {
        let content = "ssh server.internal";
        let result = Corpus::from_string(content);

        assert!(matches!(result, Err(CorpusError::SensitiveData { .. })));
    }

    #[test]
    fn test_detects_ssh_key_path() {
        let content = "ssh -i id_rsa user@server";
        let result = Corpus::from_string(content);

        assert!(matches!(result, Err(CorpusError::SensitiveData { .. })));
    }

    #[test]
    fn test_detects_export_statement() {
        let content = "export API_KEY=secret";
        let result = Corpus::from_string(content);

        assert!(matches!(result, Err(CorpusError::SensitiveData { .. })));
    }

    #[test]
    fn test_sensitive_detection_case_insensitive() {
        let content = "curl -H 'PASSWORD: test'";
        let result = Corpus::from_string(content);

        assert!(matches!(result, Err(CorpusError::SensitiveData { .. })));
    }

    #[test]
    fn test_reports_correct_line_number() {
        let content = "git status\ncargo build\ncurl -H 'api_key: x'";
        let result = Corpus::from_string(content);

        match result {
            Err(CorpusError::SensitiveData { line, .. }) => {
                assert_eq!(line, 3);
            }
            _ => panic!("Expected SensitiveData error"),
        }
    }

    // ========================================================================
    // Safe Commands Tests
    // ========================================================================

    #[test]
    fn test_allows_safe_git_commands() {
        let content = r#"
git status
git commit -m "feat: add feature"
git push origin main
git pull --rebase
git checkout -b feature/new
git log --oneline -10
git diff HEAD~1
git stash pop
"#;
        let corpus = Corpus::from_string(content).expect("should parse");
        assert!(corpus.len() >= 8);
    }

    #[test]
    fn test_allows_safe_docker_commands() {
        let content = r#"
docker build -t myapp .
docker run -d -p 8080:80 nginx
docker compose up -d
docker ps -a
docker logs container_name
docker exec -it container_name bash
"#;
        let corpus = Corpus::from_string(content).expect("should parse");
        assert!(corpus.len() >= 6);
    }

    #[test]
    fn test_allows_safe_cargo_commands() {
        let content = r#"
cargo build --release
cargo test --all-features
cargo clippy -- -D warnings
cargo fmt --check
cargo run --example demo
cargo doc --open
"#;
        let corpus = Corpus::from_string(content).expect("should parse");
        assert!(corpus.len() >= 6);
    }

    // ========================================================================
    // Coverage Statistics Tests
    // ========================================================================

    #[test]
    fn test_prefixes_extraction() {
        let content = "git status\ngit commit\ncargo build\ncargo test";
        let corpus = Corpus::from_string(content).expect("should parse");

        assert_eq!(corpus.prefixes().len(), 2);
        assert!(corpus.prefixes().contains("git"));
        assert!(corpus.prefixes().contains("cargo"));
    }

    #[test]
    fn test_coverage_stats() {
        let content = "git status\ngit commit -m msg";
        let corpus = Corpus::from_string(content).expect("should parse");
        let stats = corpus.coverage_stats();

        assert_eq!(stats.total_commands, 2);
        assert_eq!(stats.unique_prefixes, 1); // just "git"
        assert_eq!(stats.total_tokens, 6); // git, status, git, commit, -m, msg
    }

    // ========================================================================
    // Error Display Tests
    // ========================================================================

    #[test]
    fn test_error_display_not_found() {
        let err = CorpusError::NotFound("/path/to/file".into());
        assert!(err.to_string().contains("/path/to/file"));
    }

    #[test]
    fn test_error_display_sensitive() {
        let err = CorpusError::SensitiveData {
            line: 42,
            pattern: "password".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("password"));
        assert!(msg.contains("42"));
    }
}
