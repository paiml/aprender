//! PMAT Work Ticket Integration (BH-12)
//!
//! Parses PMAT work tickets to scope bug hunting to active development areas.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// A parsed PMAT work ticket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PmatTicket {
    /// Ticket identifier (e.g., "PMAT-1234")
    pub id: String,
    /// Ticket title
    pub title: String,
    /// Description
    pub description: String,
    /// Affected files/paths
    pub affected_paths: Vec<PathBuf>,
    /// Expected behavior description
    pub expected_behavior: Option<String>,
    /// Acceptance criteria
    pub acceptance_criteria: Vec<String>,
    /// Priority
    pub priority: TicketPriority,
}

/// Ticket priority levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TicketPriority {
    Critical,
    High,
    #[default]
    Medium,
    Low,
}

impl PmatTicket {
    /// Parse a ticket from file path or ID.
    pub fn parse(ticket_ref: &str, project_path: &Path) -> Result<Self, String> {
        // Check if it's a file path
        let ticket_path = if ticket_ref.ends_with(".md") || ticket_ref.ends_with(".yaml") {
            PathBuf::from(ticket_ref)
        } else {
            // Try to find ticket by ID in .pmat/tickets/
            let pmat_dir = project_path.join(".pmat/tickets");
            let md_path = pmat_dir.join(format!("{}.md", ticket_ref));
            let yaml_path = pmat_dir.join(format!("{}.yaml", ticket_ref));

            if md_path.exists() {
                md_path
            } else if yaml_path.exists() {
                yaml_path
            } else {
                // Try GitHub issue
                return Self::from_github_issue(ticket_ref);
            }
        };

        if ticket_path.extension().map(|e| e == "yaml").unwrap_or(false) {
            Self::from_yaml(&ticket_path)
        } else {
            Self::from_markdown(&ticket_path)
        }
    }

    /// Parse from YAML file.
    fn from_yaml(path: &Path) -> Result<Self, String> {
        let content =
            fs::read_to_string(path).map_err(|e| format!("Failed to read ticket: {}", e))?;

        serde_yaml_ng::from_str(&content).map_err(|e| format!("Failed to parse YAML ticket: {}", e))
    }

    /// Parse from Markdown file.
    fn from_markdown(path: &Path) -> Result<Self, String> {
        let content =
            fs::read_to_string(path).map_err(|e| format!("Failed to read ticket: {}", e))?;

        parse_markdown_ticket(&content, path)
    }

    /// Parse from GitHub issue (placeholder - would use gh CLI).
    fn from_github_issue(issue_ref: &str) -> Result<Self, String> {
        // Extract issue number
        let issue_num: u32 = issue_ref
            .trim_start_matches("PMAT-")
            .trim_start_matches('#')
            .parse()
            .map_err(|_| format!("Invalid issue reference: {}", issue_ref))?;

        // For now, return a placeholder - in production would call `gh issue view`
        Ok(Self {
            id: format!("PMAT-{}", issue_num),
            title: format!("GitHub Issue #{}", issue_num),
            description: "Loaded from GitHub".to_string(),
            affected_paths: vec![PathBuf::from("src")],
            expected_behavior: None,
            acceptance_criteria: Vec::new(),
            priority: TicketPriority::Medium,
        })
    }

    /// Get target paths for scoped analysis.
    pub fn target_paths(&self) -> Vec<PathBuf> {
        if self.affected_paths.is_empty() {
            vec![PathBuf::from("src")]
        } else {
            self.affected_paths.clone()
        }
    }
}

/// Parse a markdown ticket file.
fn parse_markdown_ticket(content: &str, path: &Path) -> Result<PmatTicket, String> {
    let mut ticket = PmatTicket {
        id: path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "UNKNOWN".to_string()),
        title: String::new(),
        description: String::new(),
        affected_paths: Vec::new(),
        expected_behavior: None,
        acceptance_criteria: Vec::new(),
        priority: TicketPriority::Medium,
    };

    let mut current_section = "";
    let mut description_lines: Vec<&str> = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim();

        // Parse title from first # header
        if trimmed.starts_with("# ") && ticket.title.is_empty() {
            ticket.title = trimmed.get(2..).unwrap_or("").to_string();
            continue;
        }

        // Track sections
        if let Some(section) = trimmed.strip_prefix("## ") {
            current_section = section.trim();
            continue;
        }

        parse_section_line(current_section, trimmed, &mut ticket, &mut description_lines);
    }

    ticket.description = description_lines.join(" ");

    Ok(ticket)
}

fn parse_section_line<'a>(
    section: &str,
    trimmed: &'a str,
    ticket: &mut PmatTicket,
    description_lines: &mut Vec<&'a str>,
) {
    fn strip_list_marker(s: &str) -> &str {
        s.get(2..).unwrap_or("")
    }

    match section.to_lowercase().as_str() {
        "description" | "summary" => {
            if !trimmed.is_empty() {
                description_lines.push(trimmed);
            }
        }
        "affected files" | "files" | "paths" | "scope" => {
            if trimmed.starts_with("- ") || trimmed.starts_with("* ") {
                let path_str = strip_list_marker(trimmed).trim().trim_matches('`');
                ticket.affected_paths.push(PathBuf::from(path_str));
            }
        }
        "expected behavior" | "expected" => {
            if !trimmed.is_empty() {
                ticket.expected_behavior = Some(trimmed.to_string());
            }
        }
        "acceptance criteria" | "criteria" => {
            if trimmed.starts_with("- ") || trimmed.starts_with("* ") {
                ticket.acceptance_criteria.push(strip_list_marker(trimmed).to_string());
            }
        }
        "priority" => {
            ticket.priority = parse_priority(trimmed);
        }
        _ => {}
    }
}

fn parse_priority(s: &str) -> TicketPriority {
    match s.to_lowercase().as_str() {
        "critical" => TicketPriority::Critical,
        "high" => TicketPriority::High,
        "medium" => TicketPriority::Medium,
        "low" => TicketPriority::Low,
        _ => TicketPriority::Medium,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_markdown_ticket() {
        let content = r#"
# Fix Authentication Bug

## Description

Users cannot login when using special characters.

## Affected Files

- `src/auth/login.rs`
- `src/auth/token.rs`

## Expected Behavior

Login should work with any valid password.

## Acceptance Criteria

- [ ] Special characters handled correctly
- [ ] No regression in existing tests

## Priority

High
"#;

        let ticket =
            parse_markdown_ticket(content, Path::new("PMAT-123.md")).expect("unexpected failure");
        assert_eq!(ticket.id, "PMAT-123");
        assert_eq!(ticket.title, "Fix Authentication Bug");
        assert_eq!(ticket.affected_paths.len(), 2);
        assert_eq!(ticket.priority, TicketPriority::High);
    }

    #[test]
    fn test_ticket_priority_default() {
        assert_eq!(TicketPriority::default(), TicketPriority::Medium);
    }

    #[test]
    fn test_target_paths_empty() {
        let ticket = PmatTicket {
            id: "TEST".to_string(),
            title: "Test".to_string(),
            description: String::new(),
            affected_paths: Vec::new(),
            expected_behavior: None,
            acceptance_criteria: Vec::new(),
            priority: TicketPriority::Medium,
        };
        let paths = ticket.target_paths();
        assert_eq!(paths, vec![PathBuf::from("src")]);
    }

    #[test]
    fn test_target_paths_with_paths() {
        let ticket = PmatTicket {
            id: "TEST".to_string(),
            title: "Test".to_string(),
            description: String::new(),
            affected_paths: vec![PathBuf::from("src/lib.rs"), PathBuf::from("src/main.rs")],
            expected_behavior: None,
            acceptance_criteria: Vec::new(),
            priority: TicketPriority::Medium,
        };
        let paths = ticket.target_paths();
        assert_eq!(paths.len(), 2);
        assert!(paths.contains(&PathBuf::from("src/lib.rs")));
    }

    #[test]
    fn test_from_github_issue_with_pmat_prefix() {
        let ticket = PmatTicket::from_github_issue("PMAT-1234").expect("unexpected failure");
        assert_eq!(ticket.id, "PMAT-1234");
        assert!(ticket.description.contains("GitHub"));
        assert_eq!(ticket.priority, TicketPriority::Medium);
    }

    #[test]
    fn test_from_github_issue_with_hash() {
        let ticket = PmatTicket::from_github_issue("#5678").expect("unexpected failure");
        assert_eq!(ticket.id, "PMAT-5678");
    }

    #[test]
    fn test_from_github_issue_number_only() {
        let ticket = PmatTicket::from_github_issue("42").expect("unexpected failure");
        assert_eq!(ticket.id, "PMAT-42");
    }

    #[test]
    fn test_from_github_issue_invalid() {
        let result = PmatTicket::from_github_issue("invalid-ref");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid issue reference"));
    }

    #[test]
    fn test_parse_markdown_summary_section() {
        let content = r#"
# Test Ticket

## Summary

This is the summary text.
"#;
        let ticket =
            parse_markdown_ticket(content, Path::new("TEST.md")).expect("unexpected failure");
        assert_eq!(ticket.description, "This is the summary text.");
    }

    #[test]
    fn test_parse_markdown_files_section() {
        let content = r#"
# Test

## Files

* src/foo.rs
* src/bar.rs
"#;
        let ticket = parse_markdown_ticket(content, Path::new("T.md")).expect("unexpected failure");
        assert_eq!(ticket.affected_paths.len(), 2);
    }

    #[test]
    fn test_parse_markdown_paths_section() {
        let content = r#"
# Test

## Paths

- lib/
"#;
        let ticket = parse_markdown_ticket(content, Path::new("T.md")).expect("unexpected failure");
        assert_eq!(ticket.affected_paths, vec![PathBuf::from("lib/")]);
    }

    #[test]
    fn test_parse_markdown_scope_section() {
        let content = r#"
# Test

## Scope

- module/
"#;
        let ticket = parse_markdown_ticket(content, Path::new("T.md")).expect("unexpected failure");
        assert_eq!(ticket.affected_paths, vec![PathBuf::from("module/")]);
    }

    #[test]
    fn test_parse_markdown_expected_section() {
        let content = r#"
# Test

## Expected

It should work correctly.
"#;
        let ticket = parse_markdown_ticket(content, Path::new("T.md")).expect("unexpected failure");
        assert_eq!(ticket.expected_behavior, Some("It should work correctly.".to_string()));
    }

    #[test]
    fn test_parse_markdown_criteria_section() {
        let content = r#"
# Test

## Criteria

- First criterion
- Second criterion
"#;
        let ticket = parse_markdown_ticket(content, Path::new("T.md")).expect("unexpected failure");
        assert_eq!(ticket.acceptance_criteria.len(), 2);
        assert!(ticket.acceptance_criteria.contains(&"First criterion".to_string()));
    }

    #[test]
    fn test_parse_markdown_priority_critical() {
        let content = r#"
# Test

## Priority

Critical
"#;
        let ticket = parse_markdown_ticket(content, Path::new("T.md")).expect("unexpected failure");
        assert_eq!(ticket.priority, TicketPriority::Critical);
    }

    #[test]
    fn test_parse_markdown_priority_low() {
        let content = r#"
# Test

## Priority

Low
"#;
        let ticket = parse_markdown_ticket(content, Path::new("T.md")).expect("unexpected failure");
        assert_eq!(ticket.priority, TicketPriority::Low);
    }

    #[test]
    fn test_parse_markdown_priority_medium() {
        let content = r#"
# Test

## Priority

Medium
"#;
        let ticket = parse_markdown_ticket(content, Path::new("T.md")).expect("unexpected failure");
        assert_eq!(ticket.priority, TicketPriority::Medium);
    }

    #[test]
    fn test_parse_markdown_priority_invalid() {
        let content = r#"
# Test

## Priority

Unknown
"#;
        let ticket = parse_markdown_ticket(content, Path::new("T.md")).expect("unexpected failure");
        assert_eq!(ticket.priority, TicketPriority::Medium); // defaults to Medium
    }

    #[test]
    fn test_parse_markdown_no_title() {
        let content = "Just some content without a title.";
        let ticket = parse_markdown_ticket(content, Path::new("T.md")).expect("unexpected failure");
        assert_eq!(ticket.title, "");
    }

    #[test]
    fn test_ticket_serialization() {
        let ticket = PmatTicket {
            id: "PMAT-1".to_string(),
            title: "Test".to_string(),
            description: "Desc".to_string(),
            affected_paths: vec![PathBuf::from("src/")],
            expected_behavior: Some("Works".to_string()),
            acceptance_criteria: vec!["Done".to_string()],
            priority: TicketPriority::High,
        };
        let json = serde_json::to_string(&ticket).expect("json serialize failed");
        let deserialized: PmatTicket =
            serde_json::from_str(&json).expect("json deserialize failed");
        assert_eq!(ticket.id, deserialized.id);
        assert_eq!(ticket.priority, deserialized.priority);
    }

    #[test]
    fn test_priority_equality() {
        assert_eq!(TicketPriority::Critical, TicketPriority::Critical);
        assert_ne!(TicketPriority::High, TicketPriority::Low);
    }

    #[test]
    fn test_priority_copy() {
        let p = TicketPriority::High;
        let p2 = p;
        assert_eq!(p, p2);
    }

    // ========================================================================
    // Coverage: PmatTicket::parse() — file path dispatch and lookup branches
    // ========================================================================

    #[test]
    fn test_parse_md_extension_routes_to_from_markdown() {
        // Passing a .md path that doesn't exist should return a read error
        let result = PmatTicket::parse("nonexistent_ticket.md", Path::new("/tmp"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Failed to read ticket"));
    }

    #[test]
    fn test_parse_yaml_extension_routes_to_from_yaml() {
        // Passing a .yaml path that doesn't exist should return a read error
        let result = PmatTicket::parse("nonexistent_ticket.yaml", Path::new("/tmp"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Failed to read ticket"));
    }

    #[test]
    fn test_parse_ticket_id_no_pmat_dir_falls_to_github() {
        // When neither .md nor .yaml exists in .pmat/tickets/, falls through to from_github_issue
        let result = PmatTicket::parse("42", Path::new("/tmp/nonexistent_project"));
        assert!(result.is_ok());
        let ticket = result.expect("operation failed");
        assert_eq!(ticket.id, "PMAT-42");
    }

    #[test]
    fn test_parse_ticket_id_invalid_falls_to_github_error() {
        // Invalid issue ref that can't parse as u32
        let result = PmatTicket::parse("not-a-number", Path::new("/tmp/nonexistent_project"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid issue reference"));
    }

    #[test]
    fn test_parse_finds_md_ticket_in_pmat_dir() {
        // Create a temp directory with .pmat/tickets/<id>.md
        let tmp = std::env::temp_dir().join("batuta_test_parse_md");
        let tickets_dir = tmp.join(".pmat/tickets");
        let _ = fs::create_dir_all(&tickets_dir);
        let md_content = "# Test Ticket\n\n## Description\n\nA test.\n";
        let md_path = tickets_dir.join("PMAT-999.md");
        fs::write(&md_path, md_content).expect("fs write failed");

        let result = PmatTicket::parse("PMAT-999", &tmp);
        assert!(result.is_ok());
        let ticket = result.expect("operation failed");
        assert_eq!(ticket.title, "Test Ticket");

        // Cleanup
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_parse_finds_yaml_ticket_in_pmat_dir() {
        // Create a temp directory with .pmat/tickets/<id>.yaml
        let tmp = std::env::temp_dir().join("batuta_test_parse_yaml");
        let tickets_dir = tmp.join(".pmat/tickets");
        let _ = fs::create_dir_all(&tickets_dir);
        let yaml_content = r#"id: "PMAT-888"
title: "YAML Ticket"
description: "From YAML"
affected_paths:
  - "src/lib.rs"
expected_behavior: null
acceptance_criteria: []
priority: High
"#;
        let yaml_path = tickets_dir.join("PMAT-888.yaml");
        fs::write(&yaml_path, yaml_content).expect("fs write failed");

        let result = PmatTicket::parse("PMAT-888", &tmp);
        assert!(result.is_ok());
        let ticket = result.expect("operation failed");
        assert_eq!(ticket.title, "YAML Ticket");

        // Cleanup
        let _ = fs::remove_dir_all(&tmp);
    }

    // ========================================================================
    // Coverage: from_yaml() and from_markdown() direct calls
    // ========================================================================

    #[test]
    fn test_from_yaml_valid_file() {
        let tmp = std::env::temp_dir().join("batuta_test_from_yaml_valid");
        let _ = fs::create_dir_all(&tmp);
        let yaml_content = r#"id: "TK-1"
title: "YAML Direct"
description: "Direct YAML parse"
affected_paths: []
expected_behavior: "Works"
acceptance_criteria:
  - "Test passes"
priority: Low
"#;
        let path = tmp.join("ticket.yaml");
        fs::write(&path, yaml_content).expect("fs write failed");

        let ticket = PmatTicket::from_yaml(&path).expect("unexpected failure");
        assert_eq!(ticket.id, "TK-1");
        assert_eq!(ticket.title, "YAML Direct");
        assert_eq!(ticket.priority, TicketPriority::Low);

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_from_yaml_invalid_content() {
        let tmp = std::env::temp_dir().join("batuta_test_from_yaml_invalid");
        let _ = fs::create_dir_all(&tmp);
        let path = tmp.join("bad.yaml");
        fs::write(&path, "not: valid: yaml: [[[").expect("fs write failed");

        let result = PmatTicket::from_yaml(&path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Failed to parse YAML ticket"));

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_from_yaml_nonexistent_file() {
        let result = PmatTicket::from_yaml(Path::new("/tmp/does_not_exist_at_all.yaml"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Failed to read ticket"));
    }

    #[test]
    fn test_from_markdown_valid_file() {
        let tmp = std::env::temp_dir().join("batuta_test_from_md_valid");
        let _ = fs::create_dir_all(&tmp);
        let md_content = "# MD Direct Test\n\n## Description\n\nA direct test.\n";
        let path = tmp.join("DIRECT-1.md");
        fs::write(&path, md_content).expect("fs write failed");

        let ticket = PmatTicket::from_markdown(&path).expect("unexpected failure");
        assert_eq!(ticket.id, "DIRECT-1");
        assert_eq!(ticket.title, "MD Direct Test");

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_from_markdown_nonexistent_file() {
        let result = PmatTicket::from_markdown(Path::new("/tmp/does_not_exist_at_all.md"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Failed to read ticket"));
    }

    #[test]
    fn test_parse_yaml_extension_with_real_yaml_file() {
        // Test the parse() path where extension is .yaml and file exists
        let tmp = std::env::temp_dir().join("batuta_test_parse_yaml_ext");
        let _ = fs::create_dir_all(&tmp);
        let yaml_content = r#"id: "EXT-1"
title: "Extension Test"
description: "Test yaml extension detection in parse()"
affected_paths: []
expected_behavior: null
acceptance_criteria: []
priority: Medium
"#;
        let path = tmp.join("ticket.yaml");
        fs::write(&path, yaml_content).expect("fs write failed");

        let result = PmatTicket::parse(path.to_str().expect("path to_str failed"), &tmp);
        assert!(result.is_ok());
        assert_eq!(result.expect("operation failed").title, "Extension Test");

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_parse_md_extension_with_real_md_file() {
        // Test the parse() path where extension is .md and file exists
        let tmp = std::env::temp_dir().join("batuta_test_parse_md_ext");
        let _ = fs::create_dir_all(&tmp);
        let md_content = "# MD Extension Test\n\n## Description\n\nParse route to markdown.\n";
        let path = tmp.join("ticket.md");
        fs::write(&path, md_content).expect("fs write failed");

        let result = PmatTicket::parse(path.to_str().expect("path to_str failed"), &tmp);
        assert!(result.is_ok());
        assert_eq!(result.expect("operation failed").title, "MD Extension Test");

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_parse_markdown_no_file_stem() {
        // Path with no file stem (edge case)
        let content = "Just content";
        let ticket = parse_markdown_ticket(content, Path::new("")).expect("unexpected failure");
        // Empty path -> file_stem returns None -> defaults to "UNKNOWN"
        assert_eq!(ticket.id, "UNKNOWN");
    }
}
