//! Syscall filtering for -e trace= expressions
//!
//! Sprint 9-10: Implement strace-compatible filtering
//! Supports:
//! - Individual syscalls: -e trace=open,read,write
//! - Syscall classes: -e trace=file, -e trace=network, -e trace=process
//!
//! Sprint 15: Advanced filtering with negation
//! Supports:
//! - Negation: -e trace=!close, -e trace=!file
//! - Mixed: -e trace=file,!close
//!
//! Sprint 16: Advanced filtering with regex patterns
//! Supports:
//! - Regex patterns: -e trace=/^open.*/, -e trace=/.*at$/
//! - Mixed: -e trace=/^open.*/,close, -e trace=/read|write/

use anyhow::{bail, Result};
use regex::Regex;
use std::collections::HashSet;

/// Syscall filter that determines which syscalls to trace
#[derive(Debug, Clone)]
pub struct SyscallFilter {
    /// Set of syscall names to include (None = all syscalls)
    include: Option<HashSet<String>>,
    /// Set of syscall names to exclude (always applied)
    exclude: HashSet<String>,
    /// Regex patterns to include (Sprint 16)
    include_regex: Vec<Regex>,
    /// Regex patterns to exclude (Sprint 16)
    exclude_regex: Vec<Regex>,
}

impl SyscallFilter {
    /// Create a filter that includes all syscalls
    pub fn all() -> Self {
        Self {
            include: None,
            exclude: HashSet::new(),
            include_regex: Vec::new(),
            exclude_regex: Vec::new(),
        }
    }

    /// Parse a filter expression like "trace=open,read,write" or "trace=file"
    pub fn from_expr(expr: &str) -> Result<Self> {
        // Parse trace=SPEC format (e.g., trace=file, trace=open,read)
        if let Some(trace_spec) = expr.strip_prefix("trace=") {
            Self::from_trace_spec(trace_spec)
        } else {
            bail!("Invalid filter expression: {expr}. Expected format: trace=SPEC");
        }
    }

    /// Parse a trace specification (the part after "trace=")
    fn from_trace_spec(spec: &str) -> Result<Self> {
        // Sprint 15: Validate spec
        validate_trace_spec(spec)?;

        // Sprint 16: Parse include/exclude sets and regex patterns
        let (include_syscalls, exclude_syscalls, include_regex, exclude_regex, has_includes) =
            parse_syscall_sets(spec)?;

        // Sprint 15: If only negations, include all syscalls except excluded
        let include = if has_includes {
            Some(include_syscalls)
        } else {
            None // Trace all except excluded
        };

        Ok(Self { include, exclude: exclude_syscalls, include_regex, exclude_regex })
    }

    /// Check if a syscall should be traced
    pub fn should_trace(&self, syscall_name: &str) -> bool {
        contract_pre_error_handling!(syscall_name);
        contract_post_error_handling!(&"ok");
        // Sprint 15: First check exclusions (highest priority)
        if self.exclude.contains(syscall_name) {
            return false;
        }

        // Sprint 16: Check exclude regex patterns
        for pattern in &self.exclude_regex {
            if pattern.is_match(syscall_name) {
                return false;
            }
        }

        // Then check inclusions
        match &self.include {
            None => {
                // No filter = trace all (except excluded)
                // Sprint 16: But if we have include_regex, check those too
                if self.include_regex.is_empty() {
                    true
                } else {
                    // If we have include regex, syscall must match at least one
                    self.include_regex.iter().any(|pattern| pattern.is_match(syscall_name))
                }
            }
            Some(set) => {
                // Sprint 16: Match if in literal set OR matches include regex
                set.contains(syscall_name)
                    || self.include_regex.iter().any(|pattern| pattern.is_match(syscall_name))
            }
        }
    }
}

/// Validate trace specification syntax
/// Sprint 15: Extracted to reduce complexity
fn validate_trace_spec(spec: &str) -> Result<()> {
    if spec.is_empty() {
        return Ok(());
    }

    if spec.trim() == "!" {
        bail!("Invalid negation syntax: '!' must be followed by syscall name or class");
    }

    Ok(())
}

/// Parse result for syscall sets
/// Sprint 16: Type alias to satisfy `clippy::type_complexity`
type ParseResult = (
    HashSet<String>, // include_set
    HashSet<String>, // exclude_set
    Vec<Regex>,      // include_regex
    Vec<Regex>,      // exclude_regex
    bool,            // has_includes
);

/// Parse syscall sets from trace specification
/// Sprint 15: Extracted to reduce complexity
/// Sprint 16: Extended to support regex patterns
/// Returns (`include_set`, `exclude_set`, `include_regex`, `exclude_regex`, `has_includes`)
fn parse_syscall_sets(spec: &str) -> Result<ParseResult> {
    let mut include_syscalls = HashSet::new();
    let mut exclude_syscalls = HashSet::new();
    let mut include_regex = Vec::new();
    let mut exclude_regex = Vec::new();
    let mut has_includes = false;

    if spec.is_empty() {
        return Ok((include_syscalls, exclude_syscalls, include_regex, exclude_regex, true));
    }

    for part in spec.split(',') {
        let part = part.trim();

        // Check for negation prefix
        let (is_negation, syscall_part) = if let Some(s) = part.strip_prefix('!') {
            (true, s)
        } else {
            has_includes = true;
            (false, part)
        };

        // Sprint 16: Check if this is a regex pattern /pattern/
        if let Some(pattern) = parse_regex_pattern(syscall_part)? {
            if is_negation {
                exclude_regex.push(pattern);
            } else {
                include_regex.push(pattern);
            }
        } else {
            // Expand syscall classes or add individual syscall
            let syscalls_to_add = expand_syscall_class(syscall_part);

            if is_negation {
                exclude_syscalls.extend(syscalls_to_add);
            } else {
                include_syscalls.extend(syscalls_to_add);
            }
        }
    }

    Ok((include_syscalls, exclude_syscalls, include_regex, exclude_regex, has_includes))
}

/// Parse a regex pattern from /pattern/ syntax
/// Sprint 16: Extracted to reduce complexity
/// Returns Some(Regex) if input is /pattern/, None otherwise
fn parse_regex_pattern(input: &str) -> Result<Option<Regex>> {
    // Check if input is wrapped in forward slashes
    // Allow empty regex (//) as valid pattern (len >= 2, not > 2)
    if input.starts_with('/') && input.ends_with('/') && input.len() >= 2 {
        // Extract pattern between slashes
        let pattern = &input[1..input.len() - 1];

        // Compile regex, propagating errors
        match Regex::new(pattern) {
            Ok(regex) => Ok(Some(regex)),
            Err(e) => bail!("Invalid regex pattern '{pattern}': {e}"),
        }
    } else {
        Ok(None)
    }
}

/// Expand a syscall class or return a single syscall name
/// Sprint 15: Extracted to reduce complexity
fn expand_syscall_class(name: &str) -> Vec<String> {
    match name {
        "file" => vec![
            "open",
            "openat",
            "close",
            "read",
            "write",
            "lseek",
            "stat",
            "fstat",
            "newfstatat",
            "access",
            "mkdir",
            "rmdir",
            "unlink",
        ]
        .iter()
        .map(std::string::ToString::to_string)
        .collect(),
        "network" => [
            "socket",
            "connect",
            "accept",
            "bind",
            "listen",
            "send",
            "recv",
            "sendto",
            "recvfrom",
            "setsockopt",
            "getsockopt",
        ]
        .iter()
        .map(std::string::ToString::to_string)
        .collect(),
        "process" => [
            "fork",
            "vfork",
            "clone",
            "execve",
            "exit",
            "exit_group",
            "wait4",
            "waitid",
            "kill",
            "tkill",
            "tgkill",
        ]
        .iter()
        .map(std::string::ToString::to_string)
        .collect(),
        "memory" => ["mmap", "munmap", "mprotect", "mremap", "brk", "sbrk"]
            .iter()
            .map(std::string::ToString::to_string)
            .collect(),
        _ => vec![name.to_string()],
    }
}

// Compile-time thread-safety verification (Sprint 59)
static_assertions::assert_impl_all!(SyscallFilter: Send, Sync);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_all_traces_everything() {
        let filter = SyscallFilter::all();
        assert!(filter.should_trace("open"));
        assert!(filter.should_trace("write"));
        assert!(filter.should_trace("anything"));
    }

    #[test]
    fn test_filter_individual_syscalls() {
        let filter = SyscallFilter::from_expr("trace=open,read,write").expect("test");
        assert!(filter.should_trace("open"));
        assert!(filter.should_trace("read"));
        assert!(filter.should_trace("write"));
        assert!(!filter.should_trace("close"));
    }

    #[test]
    fn test_filter_file_class() {
        let filter = SyscallFilter::from_expr("trace=file").expect("test");
        assert!(filter.should_trace("open"));
        assert!(filter.should_trace("openat"));
        assert!(filter.should_trace("read"));
        assert!(filter.should_trace("write"));
        assert!(!filter.should_trace("socket"));
    }

    #[test]
    fn test_filter_network_class() {
        let filter = SyscallFilter::from_expr("trace=network").expect("test");
        assert!(filter.should_trace("socket"));
        assert!(filter.should_trace("connect"));
        assert!(!filter.should_trace("open"));
    }

    #[test]
    fn test_filter_mixed() {
        let filter = SyscallFilter::from_expr("trace=file,socket").expect("test");
        assert!(filter.should_trace("open"));
        assert!(filter.should_trace("socket"));
        assert!(!filter.should_trace("clone"));
    }

    #[test]
    fn test_invalid_expression() {
        let result = SyscallFilter::from_expr("invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_filter_process_class() {
        let filter = SyscallFilter::from_expr("trace=process").expect("test");
        assert!(filter.should_trace("fork"));
        assert!(filter.should_trace("clone"));
        assert!(filter.should_trace("execve"));
        assert!(filter.should_trace("exit"));
        assert!(!filter.should_trace("open"));
        assert!(!filter.should_trace("socket"));
    }

    #[test]
    fn test_filter_memory_class() {
        let filter = SyscallFilter::from_expr("trace=memory").expect("test");
        assert!(filter.should_trace("mmap"));
        assert!(filter.should_trace("munmap"));
        assert!(filter.should_trace("mprotect"));
        assert!(filter.should_trace("brk"));
        assert!(!filter.should_trace("open"));
        assert!(!filter.should_trace("fork"));
    }

    #[test]
    fn test_filter_multiple_classes() {
        let filter = SyscallFilter::from_expr("trace=file,network,process").expect("test");
        // File class
        assert!(filter.should_trace("open"));
        assert!(filter.should_trace("read"));
        // Network class
        assert!(filter.should_trace("socket"));
        assert!(filter.should_trace("connect"));
        // Process class
        assert!(filter.should_trace("fork"));
        assert!(filter.should_trace("execve"));
        // Not included
        assert!(!filter.should_trace("mmap"));
    }

    #[test]
    fn test_filter_clone() {
        let filter1 = SyscallFilter::from_expr("trace=open,read").expect("test");
        let filter2 = filter1.clone();
        assert!(filter2.should_trace("open"));
        assert!(filter2.should_trace("read"));
        assert!(!filter2.should_trace("write"));
    }

    #[test]
    fn test_filter_debug() {
        let filter = SyscallFilter::all();
        let debug_str = format!("{:?}", filter);
        assert!(debug_str.contains("SyscallFilter"));
    }

    #[test]
    fn test_filter_empty_trace_spec() {
        // Empty spec should create filter with no syscalls
        let filter = SyscallFilter::from_expr("trace=").expect("test");
        // Empty filter should not trace anything (empty HashSet)
        assert!(!filter.should_trace("open"));
        assert!(!filter.should_trace("read"));
    }

    #[test]
    fn test_filter_whitespace_handling() {
        let filter = SyscallFilter::from_expr("trace=open, read , write").expect("test");
        assert!(filter.should_trace("open"));
        assert!(filter.should_trace("read"));
        assert!(filter.should_trace("write"));
        assert!(!filter.should_trace("close"));
    }

    // Sprint 15: Negation operator tests
    #[test]
    fn test_negation_single_syscall() {
        let filter = SyscallFilter::from_expr("trace=!close").expect("test");
        assert!(filter.should_trace("open"));
        assert!(filter.should_trace("read"));
        assert!(!filter.should_trace("close"));
    }

    #[test]
    fn test_negation_multiple_syscalls() {
        let filter = SyscallFilter::from_expr("trace=!open,!close").expect("test");
        assert!(!filter.should_trace("open"));
        assert!(!filter.should_trace("close"));
        assert!(filter.should_trace("read"));
        assert!(filter.should_trace("write"));
    }

    #[test]
    fn test_negation_syscall_class() {
        let filter = SyscallFilter::from_expr("trace=!file").expect("test");
        assert!(!filter.should_trace("open"));
        assert!(!filter.should_trace("read"));
        assert!(!filter.should_trace("write"));
        assert!(filter.should_trace("socket"));
        assert!(filter.should_trace("fork"));
    }

    #[test]
    fn test_mixed_positive_negative() {
        let filter = SyscallFilter::from_expr("trace=file,!close").expect("test");
        assert!(filter.should_trace("open"));
        assert!(filter.should_trace("read"));
        assert!(!filter.should_trace("close")); // Explicitly excluded
        assert!(!filter.should_trace("socket")); // Not in file class
    }

    #[test]
    fn test_negation_invalid_syntax() {
        let result = SyscallFilter::from_expr("trace=!");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid negation syntax"));
    }

    #[test]
    fn test_negation_preserves_original_behavior() {
        // Ensure positive-only filters still work
        let filter = SyscallFilter::from_expr("trace=open,read").expect("test");
        assert!(filter.should_trace("open"));
        assert!(filter.should_trace("read"));
        assert!(!filter.should_trace("write"));
    }

    #[test]
    fn test_expand_syscall_class_file() {
        let syscalls = expand_syscall_class("file");
        assert!(syscalls.contains(&"open".to_string()));
        assert!(syscalls.contains(&"close".to_string()));
        assert!(syscalls.contains(&"read".to_string()));
    }

    #[test]
    fn test_expand_syscall_class_network() {
        let syscalls = expand_syscall_class("network");
        assert!(syscalls.contains(&"socket".to_string()));
        assert!(syscalls.contains(&"connect".to_string()));
    }

    #[test]
    fn test_expand_syscall_class_individual() {
        let syscalls = expand_syscall_class("custom_syscall");
        assert_eq!(syscalls, vec!["custom_syscall".to_string()]);
    }

    // Sprint 16: Regex pattern tests
    #[test]
    fn test_regex_pattern_basic() {
        let filter = SyscallFilter::from_expr("trace=/^open.*/").expect("test");
        assert!(filter.should_trace("open"));
        assert!(filter.should_trace("openat"));
        assert!(!filter.should_trace("close"));
        assert!(!filter.should_trace("read"));
    }

    #[test]
    fn test_regex_pattern_suffix() {
        let filter = SyscallFilter::from_expr("trace=/.*at$/").expect("test");
        assert!(filter.should_trace("openat"));
        assert!(filter.should_trace("newfstatat"));
        assert!(!filter.should_trace("open"));
        assert!(!filter.should_trace("close"));
    }

    #[test]
    fn test_regex_pattern_or() {
        let filter = SyscallFilter::from_expr("trace=/read|write/").expect("test");
        assert!(filter.should_trace("read"));
        assert!(filter.should_trace("write"));
        assert!(!filter.should_trace("open"));
        assert!(!filter.should_trace("close"));
    }

    #[test]
    fn test_regex_pattern_case_insensitive() {
        let filter = SyscallFilter::from_expr("trace=/(?i)OPEN/").expect("test");
        assert!(filter.should_trace("open"));
        assert!(filter.should_trace("OPEN"));
        assert!(!filter.should_trace("close"));
    }

    #[test]
    fn test_regex_mixed_with_literal() {
        let filter = SyscallFilter::from_expr("trace=/^open.*/,close").expect("test");
        assert!(filter.should_trace("open"));
        assert!(filter.should_trace("openat"));
        assert!(filter.should_trace("close"));
        assert!(!filter.should_trace("read"));
    }

    #[test]
    fn test_regex_mixed_with_negation() {
        let filter = SyscallFilter::from_expr("trace=/^open.*/,!/openat/").expect("test");
        assert!(filter.should_trace("open"));
        assert!(!filter.should_trace("openat")); // Excluded by negation
        assert!(!filter.should_trace("close"));
    }

    #[test]
    fn test_regex_negation_pattern() {
        let filter = SyscallFilter::from_expr("trace=!/close/").expect("test");
        assert!(filter.should_trace("open"));
        assert!(filter.should_trace("read"));
        assert!(!filter.should_trace("close")); // Excluded by regex
    }

    #[test]
    fn test_regex_invalid_pattern() {
        let result = SyscallFilter::from_expr("trace=/[invalid/");
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("regex") || err_msg.contains("invalid"));
    }

    #[test]
    fn test_parse_regex_pattern_valid() {
        let result = parse_regex_pattern("/^test.*/");
        assert!(result.is_ok());
        let pattern = result.expect("test");
        assert!(pattern.is_some());
        let regex = pattern.expect("test");
        assert!(regex.is_match("test123"));
        assert!(!regex.is_match("other"));
    }

    #[test]
    fn test_parse_regex_pattern_not_regex() {
        let result = parse_regex_pattern("open");
        assert!(result.is_ok());
        assert!(result.expect("test").is_none());
    }

    #[test]
    fn test_parse_regex_pattern_empty() {
        let result = parse_regex_pattern("//");
        assert!(result.is_ok());
        // Empty regex is valid but matches nothing useful
        let pattern = result.expect("test");
        assert!(pattern.is_some());
    }

    #[test]
    fn test_regex_with_syscall_class() {
        let filter = SyscallFilter::from_expr("trace=file,/socket|connect/").expect("test");
        // File class
        assert!(filter.should_trace("open"));
        assert!(filter.should_trace("read"));
        // Regex patterns
        assert!(filter.should_trace("socket"));
        assert!(filter.should_trace("connect"));
        // Not included
        assert!(!filter.should_trace("fork"));
    }

    #[test]
    fn test_regex_exclude_with_include_class() {
        let filter = SyscallFilter::from_expr("trace=file,!/.*at$/").expect("test");
        assert!(filter.should_trace("open"));
        assert!(filter.should_trace("read"));
        assert!(!filter.should_trace("openat")); // Excluded by regex
        assert!(!filter.should_trace("newfstatat")); // Excluded by regex
    }
}
