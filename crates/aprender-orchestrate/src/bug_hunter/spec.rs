//! Spec-Driven Bug Hunting Module (BH-11, BH-14)
//!
//! Parses specification files to extract claims, maps them to code,
//! and supports bidirectional linking between specs and findings.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::{Finding, FindingSeverity};

/// A claim extracted from a specification file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecClaim {
    /// Claim identifier (e.g., "BH-01", "AUTH-01")
    pub id: String,
    /// Claim title
    pub title: String,
    /// Line number in spec file
    pub line: usize,
    /// Section hierarchy (e.g., ["Section 11", "BH-01"])
    pub section_path: Vec<String>,
    /// Implementation locations found
    pub implementations: Vec<CodeLocation>,
    /// Bug findings linked to this claim
    pub findings: Vec<String>,
    /// Status: Verified, Warning, Failed
    pub status: ClaimStatus,
}

/// A location in code.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeLocation {
    pub file: PathBuf,
    pub line: usize,
    pub context: String,
}

/// Status of a spec claim after bug hunting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClaimStatus {
    /// No issues found
    Verified,
    /// Some findings, not critical
    Warning,
    /// Critical findings or implementation missing
    Failed,
    /// Not yet analyzed
    Pending,
}

impl std::fmt::Display for ClaimStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Verified => write!(f, "✓ Verified"),
            Self::Warning => write!(f, "⚠️ Warning"),
            Self::Failed => write!(f, "✗ Failed"),
            Self::Pending => write!(f, "○ Pending"),
        }
    }
}

/// Parsed specification with claims.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedSpec {
    /// Path to the spec file
    pub path: PathBuf,
    /// All claims extracted
    pub claims: Vec<SpecClaim>,
    /// Original content (for updates)
    pub original_content: String,
}

impl ParsedSpec {
    /// Parse a specification file.
    pub fn parse(spec_path: &Path) -> Result<Self, String> {
        let content = fs::read_to_string(spec_path)
            .map_err(|e| format!("Failed to read spec file: {}", e))?;

        let claims = parse_claims(&content);

        Ok(Self { path: spec_path.to_path_buf(), claims, original_content: content })
    }

    /// Get claims matching a section filter.
    pub fn claims_for_section(&self, section: &str) -> Vec<&SpecClaim> {
        self.claims
            .iter()
            .filter(|c| {
                c.section_path.iter().any(|s| s.contains(section))
                    || c.id.contains(section)
                    || c.title.contains(section)
            })
            .collect()
    }

    /// Update spec file with findings (BH-14).
    pub fn update_with_findings(
        &mut self,
        findings: &[(String, Vec<Finding>)], // (claim_id, findings)
    ) -> Result<String, String> {
        // First, remove all existing bug-hunter-status blocks
        let mut updated = remove_existing_status_blocks(&self.original_content);

        for (claim_id, claim_findings) in findings {
            // Find the claim in our parsed claims
            if let Some(claim) = self.claims.iter_mut().find(|c| c.id == *claim_id) {
                // Update claim status
                claim.status = if claim_findings.is_empty() {
                    ClaimStatus::Verified
                } else if claim_findings.iter().any(|f| {
                    matches!(f.severity, FindingSeverity::Critical | FindingSeverity::High)
                }) {
                    ClaimStatus::Failed
                } else {
                    ClaimStatus::Warning
                };

                // Generate the status block to insert
                let status_block = generate_status_block(claim, claim_findings);

                // Find where to insert (after the claim header) in the cleaned content
                // We need to re-find the line since we removed status blocks
                if let Some(insert_pos) = find_claim_end(&updated, &claim.id) {
                    updated.insert_str(insert_pos, &status_block);
                }
            }
        }

        Ok(updated)
    }

    /// Write updated spec to file (with backup).
    pub fn write_updated(&self, updated_content: &str) -> Result<(), String> {
        // Create backup
        let backup_path = self.path.with_extension("md.bak");
        fs::copy(&self.path, &backup_path)
            .map_err(|e| format!("Failed to create backup: {}", e))?;

        // Write updated content
        fs::write(&self.path, updated_content)
            .map_err(|e| format!("Failed to write spec: {}", e))?;

        Ok(())
    }
}

/// Parse claims from markdown content.
fn parse_claims(content: &str) -> Vec<SpecClaim> {
    let mut claims = Vec::new();
    let mut current_sections: Vec<String> = Vec::new();

    for (line_num, line) in content.lines().enumerate() {
        let line_num = line_num + 1;
        let trimmed = line.trim();

        // Track section hierarchy
        if let Some(section) = trimmed.strip_prefix("## ") {
            current_sections.clear();
            current_sections.push(section.to_string());
        } else if let Some(subsection) = trimmed.strip_prefix("### ") {
            // Keep parent section, replace subsection
            current_sections.truncate(1);
            current_sections.push(subsection.to_string());
        }

        // Extract claim IDs from headers like "### BH-01: Title" or "### AUTH-01: Title"
        if trimmed.starts_with("### ") {
            if let Some((id, title)) = parse_claim_header(trimmed) {
                claims.push(SpecClaim {
                    id,
                    title,
                    line: line_num,
                    section_path: current_sections.clone(),
                    implementations: Vec::new(),
                    findings: Vec::new(),
                    status: ClaimStatus::Pending,
                });
            }
        }
    }

    claims
}

/// Parse a claim header like "### BH-01: Mutation-Based Invariant Falsification"
fn parse_claim_header(header: &str) -> Option<(String, String)> {
    let text = header.trim_start_matches('#').trim();

    // Look for pattern: ID: Title
    // ID can be: BH-01, AUTH-01, CB-020, etc.
    // Pattern: 1-4 uppercase letters, dash, 1-4 digits, colon, title

    let colon_pos = text.find(':')?;
    let potential_id = &text[..colon_pos];
    let title = text[colon_pos + 1..].trim();

    // Validate ID format: letters-digits
    let dash_pos = potential_id.find('-')?;
    let prefix = &potential_id[..dash_pos];
    let suffix = &potential_id[dash_pos + 1..];

    // Prefix should be 1-4 uppercase letters
    if prefix.is_empty() || prefix.len() > 4 || !prefix.chars().all(|c| c.is_ascii_uppercase()) {
        return None;
    }

    // Suffix should be 1-4 digits
    if suffix.is_empty() || suffix.len() > 4 || !suffix.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }

    Some((potential_id.to_string(), title.to_string()))
}

/// Generate status block to insert after a claim header.
fn generate_status_block(claim: &SpecClaim, findings: &[Finding]) -> String {
    let mut block = String::new();
    block.push_str("\n\n<!-- bug-hunter-status -->\n");
    block.push_str(&format!("**Bug Hunter Status:** {}\n", claim.status));

    if !claim.implementations.is_empty() {
        block.push_str("**Implementations:**\n");
        for loc in &claim.implementations {
            block.push_str(&format!("- `{}:{}` - {}\n", loc.file.display(), loc.line, loc.context));
        }
    }

    if findings.is_empty() {
        block.push_str("**Findings:** None ✓\n");
    } else {
        block.push_str(&format!("**Findings:** {} issue(s)\n", findings.len()));
        for finding in findings.iter().take(5) {
            block.push_str(&format!(
                "- [{}]({}) - {}\n",
                finding.id,
                finding.location(),
                finding.title
            ));
        }
        if findings.len() > 5 {
            block.push_str(&format!("- ... and {} more\n", findings.len() - 5));
        }
    }

    block.push_str("<!-- /bug-hunter-status -->\n");
    block
}

/// Remove all existing bug-hunter-status blocks from content.
fn remove_existing_status_blocks(content: &str) -> String {
    let mut result = String::new();
    let mut in_status_block = false;

    for line in content.lines() {
        if line.contains("<!-- bug-hunter-status -->") {
            in_status_block = true;
            continue;
        }
        if line.contains("<!-- /bug-hunter-status -->") {
            in_status_block = false;
            continue;
        }
        if !in_status_block {
            result.push_str(line);
            result.push('\n');
        }
    }

    result
}

/// Find the byte position after a claim header line.
fn find_claim_end(content: &str, claim_id: &str) -> Option<usize> {
    let mut offset = 0;

    for line in content.lines() {
        offset += line.len() + 1; // +1 for newline
                                  // Found the claim header
        if line.contains("###") && line.contains(claim_id) {
            return Some(offset);
        }
    }

    None
}

/// Find code implementing a spec claim by searching for the claim ID in comments.
pub fn find_implementations(claim: &SpecClaim, project_path: &Path) -> Vec<CodeLocation> {
    let mut locations = Vec::new();

    // Search for claim ID in source files
    let pattern = &claim.id;

    if let Ok(entries) = glob::glob(&format!("{}/**/*.rs", project_path.display())) {
        for entry in entries.flatten() {
            if let Ok(content) = fs::read_to_string(&entry) {
                for (line_num, line) in content.lines().enumerate() {
                    if line.contains(pattern) {
                        // Extract context (the line content)
                        let context = line.trim().chars().take(60).collect::<String>();
                        locations.push(CodeLocation {
                            file: entry.clone(),
                            line: line_num + 1,
                            context,
                        });
                    }
                }
            }
        }
    }

    locations
}

/// Map findings to spec claims based on file paths and content.
pub fn map_findings_to_claims(
    claims: &[SpecClaim],
    findings: &[Finding],
    project_path: &Path,
) -> HashMap<String, Vec<Finding>> {
    let mut mapping: HashMap<String, Vec<Finding>> = HashMap::new();

    // Initialize all claims
    for claim in claims {
        mapping.insert(claim.id.clone(), Vec::new());
    }

    // For each finding, try to associate with a claim
    for finding in findings {
        // Check if finding is in a file that implements any claim
        for claim in claims {
            let implementations = find_implementations(claim, project_path);
            for impl_loc in &implementations {
                // If finding is in same file and near the implementation
                if finding.file == impl_loc.file {
                    let distance = (finding.line as i64 - impl_loc.line as i64).unsigned_abs();
                    if distance < 50 {
                        // Within 50 lines
                        mapping.entry(claim.id.clone()).or_default().push(finding.clone());
                        break;
                    }
                }
            }
        }
    }

    mapping
}

#[cfg(test)]
#[path = "spec_tests.rs"]
mod tests;
