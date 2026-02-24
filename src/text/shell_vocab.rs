//! Shell-aware vocabulary for tokenizing bash/shell scripts.
//!
//! Provides a specialized vocabulary that understands shell syntax:
//! builtins, operators, variables, control flow, and common patterns.
//! Designed for use with the neural encoder architecture in `citl::neural`.
//!
//! # Example
//!
//! ```
//! use aprender::text::shell_vocab::ShellVocabulary;
//!
//! let vocab = ShellVocabulary::new();
//! let tokens = vocab.tokenize("#!/bin/bash\necho $HOME");
//! assert!(!tokens.is_empty());
//! assert_eq!(vocab.cls_token(), 2);
//! ```

use std::collections::HashMap;

/// Shell-aware vocabulary for tokenizing shell scripts.
///
/// Maps shell tokens (builtins, operators, variables, control flow keywords)
/// to integer IDs suitable for embedding layers. Follows the same pattern
/// as `citl::neural::Vocabulary` but specialized for shell script analysis.
#[derive(Debug, Clone)]
pub struct ShellVocabulary {
    /// Token to ID mapping
    token_to_id: HashMap<String, usize>,
    /// ID to token mapping (for decoding)
    id_to_token: HashMap<usize, String>,
    /// Special token IDs
    pad_id: usize,
    unk_id: usize,
    cls_id: usize,
    sep_id: usize,
    eos_id: usize,
    /// Safety class labels
    label_names: Vec<&'static str>,
}

/// The 5 safety classes for shell script classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SafetyClass {
    /// Passes all checks (lint clean, deterministic, idempotent)
    Safe = 0,
    /// Variable quoting issues detected
    NeedsQuoting = 1,
    /// Contains non-deterministic constructs ($RANDOM, $$, timestamps)
    NonDeterministic = 2,
    /// Missing idempotency flags (-p, -f)
    NonIdempotent = 3,
    /// Security rule violations (SEC001-SEC008)
    Unsafe = 4,
}

impl SafetyClass {
    /// All safety class variants in order.
    #[must_use]
    pub const fn all() -> [SafetyClass; 5] {
        [
            SafetyClass::Safe,
            SafetyClass::NeedsQuoting,
            SafetyClass::NonDeterministic,
            SafetyClass::NonIdempotent,
            SafetyClass::Unsafe,
        ]
    }

    /// Number of classes.
    #[must_use]
    pub const fn num_classes() -> usize {
        5
    }

    /// Human-readable label for this class.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            SafetyClass::Safe => "safe",
            SafetyClass::NeedsQuoting => "needs-quoting",
            SafetyClass::NonDeterministic => "non-deterministic",
            SafetyClass::NonIdempotent => "non-idempotent",
            SafetyClass::Unsafe => "unsafe",
        }
    }

    /// Create from class index.
    #[must_use]
    pub const fn from_index(idx: usize) -> Option<SafetyClass> {
        match idx {
            0 => Some(SafetyClass::Safe),
            1 => Some(SafetyClass::NeedsQuoting),
            2 => Some(SafetyClass::NonDeterministic),
            3 => Some(SafetyClass::NonIdempotent),
            4 => Some(SafetyClass::Unsafe),
            _ => None,
        }
    }
}

impl std::fmt::Display for SafetyClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

impl ShellVocabulary {
    /// Create a new shell-aware vocabulary with comprehensive shell token coverage.
    #[must_use]
    pub fn new() -> Self {
        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();
        let mut id = 0usize;

        let mut insert = |token: &str, current_id: &mut usize| {
            token_to_id.insert(token.to_string(), *current_id);
            id_to_token.insert(*current_id, token.to_string());
            *current_id += 1;
        };

        // Special tokens (0-4)
        insert("[PAD]", &mut id);
        let pad_id = 0;
        insert("[UNK]", &mut id);
        let unk_id = 1;
        insert("[CLS]", &mut id);
        let cls_id = 2;
        insert("[SEP]", &mut id);
        let sep_id = 3;
        insert("[EOS]", &mut id);
        let eos_id = 4;

        // Shebangs
        for shebang in &["#!/bin/bash", "#!/bin/sh", "#!/usr/bin/env"] {
            insert(shebang, &mut id);
        }

        // Shell builtins
        for builtin in &[
            "echo", "printf", "read", "cd", "pwd", "export", "unset", "set", "source", "eval",
            "exec", "exit", "return", "shift", "test", "true", "false", ":", "type", "hash",
            "alias", "unalias", "trap", "wait", "kill", "jobs", "bg", "fg", "umask", "getopts",
            "local", "declare", "typeset", "readonly", "let", "command", "builtin", "enable",
        ] {
            insert(builtin, &mut id);
        }

        // Common external commands
        for cmd in &[
            "mkdir", "rm", "cp", "mv", "ln", "chmod", "chown", "cat", "grep", "sed", "awk", "find",
            "sort", "uniq", "wc", "head", "tail", "cut", "tr", "tee", "xargs", "curl", "wget",
            "tar", "gzip", "gunzip", "zip", "unzip", "touch", "ls", "stat", "file", "basename",
            "dirname", "mktemp", "date", "sleep", "install",
        ] {
            insert(cmd, &mut id);
        }

        // Control flow keywords
        for kw in &[
            "if", "then", "else", "elif", "fi", "for", "in", "do", "done", "while", "until",
            "case", "esac", "function",
        ] {
            insert(kw, &mut id);
        }

        // Shell operators
        for op in &[
            "|", "||", "&&", ";", ";;", "&", ">", ">>", "<", "<<", "<<<", "2>", "2>>", "2>&1",
            "&>", "(", ")", "((", "))", "[", "]", "[[", "]]", "{", "}", "$", "${", "$(", "$((",
            "`", "=", "==", "!=", "-eq", "-ne", "-lt", "-gt", "-le", "-ge", "-z", "-n", "-f", "-d",
            "-e", "-r", "-w", "-x", "-s", "!", "~",
        ] {
            insert(op, &mut id);
        }

        // Common shell variables
        for var in &[
            "$HOME",
            "$USER",
            "$PATH",
            "$PWD",
            "$SHELL",
            "$RANDOM",
            "$BASHPID",
            "$$",
            "$!",
            "$?",
            "$#",
            "$@",
            "$*",
            "$0",
            "$1",
            "$2",
            "$TMPDIR",
            "$IFS",
            "$LANG",
            "$TERM",
            "$HOSTNAME",
            "$OSTYPE",
            "$MACHTYPE",
        ] {
            insert(var, &mut id);
        }

        // Common flags (idempotency/safety related)
        for flag in &[
            "-p",
            "-f",
            "-r",
            "-rf",
            "-n",
            "-e",
            "-i",
            "-v",
            "-q",
            "-m",
            "-s",
            "-t",
            "-o",
            "-a",
            "-b",
            "-c",
            "-g",
            "-h",
            "-k",
            "-l",
            "-u",
            "-x",
            "--force",
            "--recursive",
            "--verbose",
            "--quiet",
            "--parents",
            "--no-clobber",
        ] {
            insert(flag, &mut id);
        }

        // String/quoting tokens
        for q in &["\"", "'", "\\", "\n", "\t"] {
            insert(q, &mut id);
        }

        // Numeric literals (common)
        for n in &[
            "0", "1", "2", "3", "4", "5", "10", "100", "255", "644", "755",
        ] {
            insert(n, &mut id);
        }

        // Common words in shell scripts
        for word in &[
            "file", "dir", "path", "name", "tmp", "log", "err", "out", "input", "output", "config",
            "data", "src", "dest", "root", "user", "home", "bin", "lib", "etc", "var", "dev",
            "null", "proc", "sys", "run", "opt", "ok", "error", "warning", "fatal", "info",
            "debug", "start", "stop", "restart", "status", "check", "install", "update", "remove",
            "clean", "build",
        ] {
            insert(word, &mut id);
        }

        let label_names = vec![
            "safe",
            "needs-quoting",
            "non-deterministic",
            "non-idempotent",
            "unsafe",
        ];

        Self {
            token_to_id,
            id_to_token,
            pad_id,
            unk_id,
            cls_id,
            sep_id,
            eos_id,
            label_names,
        }
    }

    /// Get vocabulary size.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.token_to_id.len()
    }

    /// Get the PAD token ID.
    #[must_use]
    pub fn pad_token(&self) -> usize {
        self.pad_id
    }

    /// Get the UNK token ID.
    #[must_use]
    pub fn unk_token(&self) -> usize {
        self.unk_id
    }

    /// Get the CLS token ID.
    #[must_use]
    pub fn cls_token(&self) -> usize {
        self.cls_id
    }

    /// Get the SEP token ID.
    #[must_use]
    pub fn sep_token(&self) -> usize {
        self.sep_id
    }

    /// Get the EOS token ID.
    #[must_use]
    pub fn eos_token(&self) -> usize {
        self.eos_id
    }

    /// Get the safety class labels.
    #[must_use]
    pub fn label_names(&self) -> &[&str] {
        &self.label_names
    }

    /// Tokenize a shell script into token IDs.
    ///
    /// Performs shell-aware tokenization that preserves:
    /// - Shebangs as single tokens
    /// - Variable references ($VAR, ${VAR})
    /// - Operators (||, &&, >>, etc.)
    /// - Quoted strings
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::text::shell_vocab::ShellVocabulary;
    ///
    /// let vocab = ShellVocabulary::new();
    /// let tokens = vocab.tokenize("echo hello");
    /// assert!(tokens.len() >= 2);
    /// ```
    #[must_use]
    pub fn tokenize(&self, script: &str) -> Vec<usize> {
        let raw_tokens = self.shell_split(script);
        raw_tokens.iter().map(|t| self.get_token_id(t)).collect()
    }

    /// Tokenize and prepend CLS, append EOS, truncate to max_len.
    #[must_use]
    pub fn encode(&self, script: &str, max_len: usize) -> Vec<usize> {
        let mut ids = Vec::with_capacity(max_len);
        ids.push(self.cls_id);

        let raw_tokens = self.shell_split(script);
        for token in &raw_tokens {
            if ids.len() >= max_len - 1 {
                break;
            }
            ids.push(self.get_token_id(token));
        }

        ids.push(self.eos_id);

        // Pad to max_len
        while ids.len() < max_len {
            ids.push(self.pad_id);
        }

        ids
    }

    /// Decode token IDs back to tokens (for debugging).
    #[must_use]
    pub fn decode(&self, ids: &[usize]) -> Vec<String> {
        ids.iter()
            .map(|&id| {
                self.id_to_token
                    .get(&id)
                    .cloned()
                    .unwrap_or_else(|| format!("[ID:{id}]"))
            })
            .collect()
    }

    /// Export vocabulary as JSON (token -> id mapping).
    ///
    /// # Errors
    ///
    /// Returns error if JSON serialization fails.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.token_to_id)
    }

    /// Split a shell script into tokens using shell-aware rules.
    fn shell_split(&self, script: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let chars: Vec<char> = script.chars().collect();
        let len = chars.len();
        let mut i = 0;

        while i < len {
            let c = chars[i];

            // Skip whitespace
            if c.is_whitespace() {
                i += 1;
                continue;
            }

            // Check for shebang at start of line
            if c == '#' && i + 1 < len && chars[i + 1] == '!' {
                let start = i;
                while i < len && chars[i] != '\n' {
                    i += 1;
                }
                let shebang = chars[start..i].iter().collect::<String>();
                tokens.push(shebang);
                continue;
            }

            // Comments
            if c == '#' {
                while i < len && chars[i] != '\n' {
                    i += 1;
                }
                continue;
            }

            // Dollar-prefixed tokens ($VAR, ${...}, $(...), $((..)))
            if c == '$' {
                let start = i;
                i += 1;
                if i < len {
                    match chars[i] {
                        '{' => {
                            // ${...}
                            while i < len && chars[i] != '}' {
                                i += 1;
                            }
                            if i < len {
                                i += 1; // skip '}'
                            }
                        }
                        '(' => {
                            if i + 1 < len && chars[i + 1] == '(' {
                                // $((...))
                                tokens.push("$((".to_string());
                                i += 2;
                                continue;
                            }
                            // $(...)
                            tokens.push("$(".to_string());
                            i += 1;
                            continue;
                        }
                        c2 if c2.is_alphanumeric()
                            || c2 == '_'
                            || c2 == '?'
                            || c2 == '#'
                            || c2 == '@'
                            || c2 == '*'
                            || c2 == '!'
                            || c2 == '$' =>
                        {
                            while i < len && (chars[i].is_alphanumeric() || chars[i] == '_') {
                                i += 1;
                            }
                        }
                        _ => {}
                    }
                }
                let token = chars[start..i].iter().collect::<String>();
                tokens.push(token);
                continue;
            }

            // Multi-character operators
            if i + 1 < len {
                let two: String = chars[i..i + 2].iter().collect();
                if matches!(
                    two.as_str(),
                    "||" | "&&"
                        | ">>"
                        | "<<"
                        | "!="
                        | "=="
                        | ";;"
                        | "(("
                        | "))"
                        | "[["
                        | "]]"
                        | "2>"
                ) {
                    tokens.push(two);
                    i += 2;
                    continue;
                }
            }

            // Single-character operators
            if matches!(
                c,
                '|' | '&' | ';' | '>' | '<' | '(' | ')' | '[' | ']' | '{' | '}' | '!' | '=' | '~'
            ) {
                tokens.push(c.to_string());
                i += 1;
                continue;
            }

            // Quoted strings — tokenize the quotes and contents separately
            if c == '"' || c == '\'' {
                tokens.push(c.to_string());
                let quote = c;
                i += 1;
                let start = i;
                while i < len && chars[i] != quote {
                    if chars[i] == '\\' && quote == '"' {
                        i += 1; // skip escaped char
                    }
                    i += 1;
                }
                if i > start {
                    let content: String = chars[start..i].iter().collect();
                    // Split content into sub-tokens
                    for word in content.split_whitespace() {
                        tokens.push(word.to_string());
                    }
                }
                if i < len {
                    tokens.push(quote.to_string());
                    i += 1;
                }
                continue;
            }

            // Backtick command substitution
            if c == '`' {
                tokens.push("`".to_string());
                i += 1;
                continue;
            }

            // Words (identifiers, commands, arguments)
            if c.is_alphanumeric() || c == '_' || c == '-' || c == '/' || c == '.' {
                let start = i;
                while i < len
                    && (chars[i].is_alphanumeric()
                        || chars[i] == '_'
                        || chars[i] == '-'
                        || chars[i] == '/'
                        || chars[i] == '.')
                {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                tokens.push(word);
                continue;
            }

            // Backslash
            if c == '\\' {
                tokens.push("\\".to_string());
                i += 1;
                if i < len {
                    i += 1; // skip escaped char
                }
                continue;
            }

            // Anything else — skip
            i += 1;
        }

        tokens
    }

    fn get_token_id(&self, token: &str) -> usize {
        self.token_to_id.get(token).copied().unwrap_or(self.unk_id)
    }
}

impl Default for ShellVocabulary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocabulary_creation() {
        let vocab = ShellVocabulary::new();
        assert!(vocab.vocab_size() > 100);
        assert_eq!(vocab.pad_token(), 0);
        assert_eq!(vocab.unk_token(), 1);
        assert_eq!(vocab.cls_token(), 2);
    }

    #[test]
    fn test_special_tokens() {
        let vocab = ShellVocabulary::new();
        assert_eq!(vocab.pad_token(), 0);
        assert_eq!(vocab.unk_token(), 1);
        assert_eq!(vocab.cls_token(), 2);
        assert_eq!(vocab.sep_token(), 3);
        assert_eq!(vocab.eos_token(), 4);
    }

    #[test]
    fn test_tokenize_simple_echo() {
        let vocab = ShellVocabulary::new();
        let tokens = vocab.tokenize("echo hello");
        assert_eq!(tokens.len(), 2);
        // "echo" should be a known builtin
        assert_ne!(tokens[0], vocab.unk_token());
    }

    #[test]
    fn test_tokenize_shebang() {
        let vocab = ShellVocabulary::new();
        let tokens = vocab.tokenize("#!/bin/bash\necho hello");
        assert!(tokens.len() >= 2);
        // Shebang should be a known token
        assert_ne!(tokens[0], vocab.unk_token());
    }

    #[test]
    fn test_tokenize_variable() {
        let vocab = ShellVocabulary::new();
        let tokens = vocab.tokenize("echo $HOME");
        assert_eq!(tokens.len(), 2);
        assert_ne!(tokens[1], vocab.unk_token()); // $HOME is known
    }

    #[test]
    fn test_tokenize_pipe() {
        let vocab = ShellVocabulary::new();
        let tokens = vocab.tokenize("ls | grep foo");
        assert!(tokens.len() >= 3);
    }

    #[test]
    fn test_tokenize_operators() {
        let vocab = ShellVocabulary::new();
        let tokens = vocab.tokenize("cmd1 && cmd2 || cmd3");
        // Should find && and || as known tokens
        assert!(tokens.len() >= 3);
    }

    #[test]
    fn test_encode_with_padding() {
        let vocab = ShellVocabulary::new();
        let encoded = vocab.encode("echo hello", 32);
        assert_eq!(encoded.len(), 32);
        assert_eq!(encoded[0], vocab.cls_token());
        // Last non-pad should be EOS
        let last_non_pad = encoded
            .iter()
            .rposition(|&id| id != vocab.pad_token())
            .expect("should have non-pad token");
        assert_eq!(encoded[last_non_pad], vocab.eos_token());
    }

    #[test]
    fn test_encode_truncation() {
        let vocab = ShellVocabulary::new();
        let long_script = "echo a; echo b; echo c; echo d; echo e; echo f; echo g";
        let encoded = vocab.encode(long_script, 8);
        assert_eq!(encoded.len(), 8);
        assert_eq!(encoded[0], vocab.cls_token());
    }

    #[test]
    fn test_decode_roundtrip() {
        let vocab = ShellVocabulary::new();
        let tokens = vocab.tokenize("echo hello");
        let decoded = vocab.decode(&tokens);
        assert_eq!(decoded[0], "echo");
    }

    #[test]
    fn test_safety_class_labels() {
        assert_eq!(SafetyClass::Safe.label(), "safe");
        assert_eq!(SafetyClass::Unsafe.label(), "unsafe");
        assert_eq!(SafetyClass::num_classes(), 5);
    }

    #[test]
    fn test_safety_class_from_index() {
        assert_eq!(SafetyClass::from_index(0), Some(SafetyClass::Safe));
        assert_eq!(SafetyClass::from_index(4), Some(SafetyClass::Unsafe));
        assert_eq!(SafetyClass::from_index(5), None);
    }

    #[test]
    fn test_to_json() {
        let vocab = ShellVocabulary::new();
        let json = vocab.to_json().expect("JSON export should succeed");
        assert!(json.contains("echo"));
        assert!(json.contains("[PAD]"));
    }

    #[test]
    fn test_comment_handling() {
        let vocab = ShellVocabulary::new();
        let tokens = vocab.tokenize("echo hello # this is a comment\necho world");
        // Comments should be stripped
        let decoded = vocab.decode(&tokens);
        assert!(!decoded.contains(&"#".to_string()));
        assert!(!decoded.contains(&"comment".to_string()));
    }

    #[test]
    fn test_tokenize_mkdir_flags() {
        let vocab = ShellVocabulary::new();
        let tokens = vocab.tokenize("mkdir -p /tmp/foo");
        assert!(tokens.len() >= 3);
        // "mkdir" and "-p" should be known tokens
        assert_ne!(tokens[0], vocab.unk_token());
        assert_ne!(tokens[1], vocab.unk_token());
    }
}
