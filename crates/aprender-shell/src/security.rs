//! Security filtering for aprender-shell
//!
//! Follows Toyota Way principle *Andon* (Stop the line):
//! Halt immediately when quality issues arise.
//!
//! Prevents sensitive commands (passwords, tokens, keys) from being
//! suggested or learned into the model.

/// Patterns that indicate sensitive commands.
///
/// These patterns match credentials, authentication commands,
/// and environment variable exports with secrets.
const SENSITIVE_PATTERNS: &[&str] = &[
    // Credentials in arguments
    "password=",
    "passwd=",
    "pwd=",
    "secret=",
    "token=",
    "api_key=",
    "apikey=",
    // Environment variables (uppercase)
    "AWS_SECRET",
    "GITHUB_TOKEN",
    "API_KEY",
    "PRIVATE_KEY",
    "ACCESS_TOKEN",
    "AUTH_TOKEN",
    "SECRET_KEY",
    // Authentication commands
    "curl -u",
    "curl --user",
    "wget --password",
    // Key generation (might expose paths)
    "ssh-keygen",
    "gpg --gen-key",
    // Database credentials
    "mysql -p",
    "psql -W",
    "mongo --password",
    // Cloud credentials
    "aws configure",
    "gcloud auth",
    "az login",
];

/// Check if a command contains sensitive information.
///
/// Returns `true` if the command should be filtered out.
///
/// # Arguments
/// * `cmd` - The command to check
///
/// # Example
/// ```
/// use aprender_shell::security::is_sensitive_command;
///
/// assert!(is_sensitive_command("export AWS_SECRET_ACCESS_KEY=abc123"));
/// assert!(is_sensitive_command("curl -u admin:password123 https://api.example.com"));
/// assert!(!is_sensitive_command("git status"));
/// ```
pub fn is_sensitive_command(cmd: &str) -> bool {
    let upper = cmd.to_uppercase();

    // Check against known patterns
    for pattern in SENSITIVE_PATTERNS {
        let pattern_upper = pattern.to_uppercase();
        if upper.contains(&pattern_upper) {
            return true;
        }
    }

    // Check for export commands with secret-like variable names
    if upper.contains("EXPORT") {
        let secret_keywords = ["SECRET", "TOKEN", "KEY", "PASSWORD", "CREDENTIAL", "AUTH"];
        for keyword in secret_keywords {
            if upper.contains(keyword) && upper.contains('=') {
                return true;
            }
        }
    }

    // Check for mysql/psql inline passwords (e.g., mysql -pMyPassword)
    // MySQL allows -p immediately followed by password without space
    let lower = cmd.to_lowercase();
    if lower.contains("mysql") || lower.contains("psql") || lower.contains("mongo") {
        // Check for -p followed directly by non-space characters
        if let Some(pos) = lower.find("-p") {
            let after_p = &lower[pos + 2..];
            // If -p is followed by non-space, non-empty content, it's likely a password
            if !after_p.is_empty() && !after_p.starts_with(' ') && !after_p.starts_with('\t') {
                return true;
            }
        }
    }

    // Check for inline key=value patterns with suspicious keys
    let suspicious_keys = [
        "password",
        "passwd",
        "secret",
        "token",
        "key",
        "credential",
        "auth",
    ];

    for key in suspicious_keys {
        // Match patterns like "password=something" or "password = something"
        let pattern1 = format!("{key}=");
        let pattern2 = format!("{key} =");
        if cmd.to_lowercase().contains(&pattern1) || cmd.to_lowercase().contains(&pattern2) {
            // Exclude false positives like "git config user.password" without value
            // Check if there's an actual value after the =
            if let Some(pos) = cmd.to_lowercase().find(&pattern1) {
                let after_eq = &cmd[pos + pattern1.len()..];
                if !after_eq.trim().is_empty()
                    && !after_eq.trim().starts_with('-')
                    && after_eq.trim() != "\""
                    && after_eq.trim() != "'"
                {
                    return true;
                }
            }
        }
    }

    false
}

/// Filter sensitive commands from a list.
///
/// Returns a new vector with sensitive commands removed.
pub fn filter_sensitive_commands(commands: &[String]) -> Vec<String> {
    commands
        .iter()
        .filter(|cmd| !is_sensitive_command(cmd))
        .cloned()
        .collect()
}

/// Filter sensitive suggestions before display.
pub fn filter_sensitive_suggestions(suggestions: Vec<(String, f32)>) -> Vec<(String, f32)> {
    suggestions
        .into_iter()
        .filter(|(cmd, _)| !is_sensitive_command(cmd))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Credential Detection Tests
    // =========================================================================

    #[test]
    fn test_detect_password_in_curl() {
        assert!(is_sensitive_command(
            "curl -u admin:password123 https://api.example.com"
        ));
        assert!(is_sensitive_command(
            "curl --user admin:pass https://api.example.com"
        ));
    }

    #[test]
    fn test_detect_env_export() {
        assert!(is_sensitive_command("export AWS_SECRET_ACCESS_KEY=abc123"));
        assert!(is_sensitive_command("export GITHUB_TOKEN=ghp_xxxx"));
        assert!(is_sensitive_command("export API_KEY=sk-xxxxx"));
        assert!(is_sensitive_command("export AUTH_TOKEN=bearer_xxx"));
    }

    #[test]
    fn test_detect_mysql_password() {
        assert!(is_sensitive_command("mysql -u root -pMyPassword"));
        assert!(is_sensitive_command("mysql -p'secret'"));
    }

    #[test]
    fn test_detect_psql_password() {
        assert!(is_sensitive_command("psql -W"));
    }

    #[test]
    fn test_detect_inline_secrets() {
        assert!(is_sensitive_command("password=hunter2"));
        assert!(is_sensitive_command("secret=mysecret"));
        assert!(is_sensitive_command("token=abc123"));
        assert!(is_sensitive_command("api_key=sk-xxxx"));
    }

    #[test]
    fn test_detect_cloud_auth() {
        assert!(is_sensitive_command("aws configure"));
        assert!(is_sensitive_command("gcloud auth login"));
        assert!(is_sensitive_command("az login"));
    }

    #[test]
    fn test_detect_key_generation() {
        assert!(is_sensitive_command("ssh-keygen -t rsa"));
        assert!(is_sensitive_command("gpg --gen-key"));
    }

    // =========================================================================
    // Non-Sensitive Command Tests (False Positive Prevention)
    // =========================================================================

    #[test]
    fn test_allow_normal_commands() {
        assert!(!is_sensitive_command("git status"));
        assert!(!is_sensitive_command("git commit -m 'message'"));
        assert!(!is_sensitive_command("docker ps"));
        assert!(!is_sensitive_command("cargo build --release"));
        assert!(!is_sensitive_command("kubectl get pods"));
        assert!(!is_sensitive_command("npm install"));
    }

    #[test]
    fn test_allow_curl_without_auth() {
        assert!(!is_sensitive_command("curl https://api.example.com"));
        assert!(!is_sensitive_command(
            "curl -X POST https://api.example.com"
        ));
        assert!(!is_sensitive_command(
            "curl -H 'Content-Type: application/json' https://api.example.com"
        ));
    }

    #[test]
    fn test_allow_git_config() {
        // These don't contain actual secrets
        assert!(!is_sensitive_command("git config user.name"));
        assert!(!is_sensitive_command("git config user.email"));
    }

    #[test]
    fn test_allow_export_without_secrets() {
        assert!(!is_sensitive_command("export PATH=/usr/bin:$PATH"));
        assert!(!is_sensitive_command("export EDITOR=vim"));
        assert!(!is_sensitive_command("export LANG=en_US.UTF-8"));
    }

    #[test]
    fn test_case_insensitive() {
        assert!(is_sensitive_command("PASSWORD=secret"));
        assert!(is_sensitive_command("Token=abc123"));
        assert!(is_sensitive_command("EXPORT github_token=xxx"));
    }

    // =========================================================================
    // Filter Function Tests
    // =========================================================================

    #[test]
    fn test_filter_sensitive_commands() {
        let commands = vec![
            "git status".to_string(),
            "export SECRET=xxx".to_string(),
            "cargo build".to_string(),
            "curl -u user:pass http://localhost".to_string(),
        ];

        let filtered = filter_sensitive_commands(&commands);

        assert_eq!(filtered.len(), 2);
        assert!(filtered.contains(&"git status".to_string()));
        assert!(filtered.contains(&"cargo build".to_string()));
        assert!(!filtered.contains(&"export SECRET=xxx".to_string()));
    }

    #[test]
    fn test_filter_sensitive_suggestions() {
        let suggestions = vec![
            ("git status".to_string(), 0.9),
            ("export TOKEN=abc".to_string(), 0.8),
            ("docker ps".to_string(), 0.7),
        ];

        let filtered = filter_sensitive_suggestions(suggestions);

        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].0, "git status");
        assert_eq!(filtered[1].0, "docker ps");
    }
}
