//! Custom subagent discovery from project-level markdown files.
//!
//! PMAT-CODE-CUSTOM-AGENTS-001: Claude-Code parity for user-defined
//! subagents. Claude scans `.claude/agents/<name>/AGENT.md` (and
//! `~/.claude/agents/<name>/AGENT.md` at user scope); `apr code`
//! mirrors the same pattern at `.apr/agents/` and `~/.config/apr/agents/`.
//!
//! The frontmatter format is markdown `---`-fenced key: value lines,
//! deliberately NOT full YAML so that the loader has no external
//! dependency and can't accept adversarial nested documents.
//!
//! # Example
//!
//! `.apr/agents/code-reviewer.md`:
//!
//! ```markdown
//! ---
//! name: code-reviewer
//! description: Reviews code for bugs, style, and security issues
//! max_iterations: 8
//! ---
//!
//! You are a code-reviewing subagent. Focus on bugs, security issues,
//! and style violations. Return a structured list of findings.
//! ```
//!
//! After loading, `registry.resolve("code-reviewer")` returns a
//! [`SubagentSpec`] usable by [`super::task_tool::TaskTool`].

use std::fs;
use std::path::{Path, PathBuf};

use super::task_tool::{SubagentRegistry, SubagentSpec};

/// Default project-scope directory for custom agents (relative to cwd).
pub const DEFAULT_PROJECT_DIR: &str = ".apr/agents";

/// Legacy / cross-compat directory — Claude Code projects that also
/// want to work with `apr code` can share a single agents directory.
pub const CLAUDE_COMPAT_DIR: &str = ".claude/agents";

/// Error shapes from parsing or loading a custom agent.
#[derive(Debug)]
pub enum CustomAgentError {
    /// Frontmatter fence missing or malformed.
    MissingFrontmatter,
    /// Required field `name` was absent or empty.
    MissingName,
    /// Required field `description` was absent or empty.
    MissingDescription,
    /// Body after frontmatter was empty (no system prompt).
    EmptyBody,
    /// Filesystem error reading the file.
    Io(String),
}

impl std::fmt::Display for CustomAgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingFrontmatter => write!(f, "missing `---`-fenced frontmatter"),
            Self::MissingName => write!(f, "required field `name` missing or empty"),
            Self::MissingDescription => {
                write!(f, "required field `description` missing or empty")
            }
            Self::EmptyBody => write!(f, "body (system prompt) is empty"),
            Self::Io(msg) => write!(f, "I/O error: {msg}"),
        }
    }
}

impl std::error::Error for CustomAgentError {}

/// Parse a custom-agent markdown document into a [`SubagentSpec`].
///
/// Frontmatter format: leading `---\n`, then lines of `key: value`,
/// then `---\n`, then the body (used as the system prompt).
pub fn parse_agent_md(source: &str) -> Result<SubagentSpec, CustomAgentError> {
    let trimmed = source.trim_start_matches('\u{feff}');
    let rest = trimmed
        .strip_prefix("---\n")
        .or_else(|| trimmed.strip_prefix("---\r\n"))
        .ok_or(CustomAgentError::MissingFrontmatter)?;

    let (front, body) = split_at_fence(rest).ok_or(CustomAgentError::MissingFrontmatter)?;

    let mut name = String::new();
    let mut description = String::new();
    let mut max_iterations: u32 = 8;

    for line in front.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim();
        let value = value.trim().trim_matches('"').trim_matches('\'');
        match key {
            "name" => name = value.to_string(),
            "description" => description = value.to_string(),
            "max_iterations" => {
                if let Ok(n) = value.parse::<u32>() {
                    if n > 0 {
                        max_iterations = n;
                    }
                }
            }
            // tools / model fields tolerated but ignored (Claude-compat).
            _ => {}
        }
    }

    if name.is_empty() {
        return Err(CustomAgentError::MissingName);
    }
    if description.is_empty() {
        return Err(CustomAgentError::MissingDescription);
    }
    let system_prompt = body.trim().to_string();
    if system_prompt.is_empty() {
        return Err(CustomAgentError::EmptyBody);
    }

    Ok(SubagentSpec { name, description, system_prompt, max_iterations })
}

/// Scan a directory for custom-agent `.md` files.
///
/// Supports two layouts (both Claude-compatible):
///   * `dir/<name>.md` — flat file per agent.
///   * `dir/<name>/AGENT.md` — subdirectory per agent (Claude's default).
///
/// Silently skips files that fail to parse (a malformed agent should
/// not break the entire load).
pub fn load_custom_agents_from(dir: &Path) -> Vec<SubagentSpec> {
    let mut specs = Vec::new();
    let Ok(entries) = fs::read_dir(dir) else {
        return specs;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() {
            if path.extension().is_some_and(|e| e == "md") {
                if let Some(spec) = try_parse_file(&path) {
                    specs.push(spec);
                }
            }
        } else if path.is_dir() {
            let agent_md = path.join("AGENT.md");
            if agent_md.is_file() {
                if let Some(spec) = try_parse_file(&agent_md) {
                    specs.push(spec);
                }
            }
        }
    }
    specs
}

/// Discover custom agents from standard locations (project + user).
///
/// Returns the merged list with project-scope taking precedence on
/// name collision. Returns an empty `Vec` when no discovery dir exists.
pub fn discover_standard_locations(cwd: &Path) -> Vec<SubagentSpec> {
    let mut merged: Vec<SubagentSpec> = Vec::new();

    let user_dir = user_agents_dir();
    if let Some(u) = user_dir.as_deref() {
        merged.extend(load_custom_agents_from(u));
    }

    for dir_rel in [DEFAULT_PROJECT_DIR, CLAUDE_COMPAT_DIR] {
        let project_dir = cwd.join(dir_rel);
        if project_dir.is_dir() {
            let project_specs = load_custom_agents_from(&project_dir);
            for spec in project_specs {
                merged.retain(|s| s.name != spec.name);
                merged.push(spec);
            }
            break;
        }
    }

    merged
}

/// Register discovered agents into a [`SubagentRegistry`]. Returns the
/// number of specs registered.
pub fn register_discovered_into(registry: &mut SubagentRegistry, cwd: &Path) -> usize {
    let specs = discover_standard_locations(cwd);
    let n = specs.len();
    for spec in specs {
        registry.register(spec);
    }
    n
}

fn try_parse_file(path: &Path) -> Option<SubagentSpec> {
    let content = fs::read_to_string(path).ok()?;
    parse_agent_md(&content).ok()
}

fn split_at_fence(after_open: &str) -> Option<(&str, &str)> {
    for (idx, line_start) in line_starts(after_open) {
        let rest_at = &after_open[line_start..];
        if let Some(line_end) = rest_at.find('\n') {
            let line = &rest_at[..line_end];
            if line.trim_end_matches('\r') == "---" {
                let front_end = line_start;
                let body_start = line_start + line_end + 1;
                let _ = idx;
                return Some((&after_open[..front_end], &after_open[body_start..]));
            }
        } else if rest_at.trim_end_matches('\r') == "---" {
            return Some((&after_open[..line_start], ""));
        }
    }
    None
}

fn line_starts(s: &str) -> impl Iterator<Item = (usize, usize)> + '_ {
    std::iter::once((0usize, 0usize))
        .chain(s.match_indices('\n').enumerate().map(|(i, (pos, _))| (i + 1, pos + 1)))
}

fn user_agents_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let home = PathBuf::from(home);
    let candidate = home.join(".config").join("apr").join("agents");
    if candidate.is_dir() {
        Some(candidate)
    } else {
        None
    }
}

#[cfg(test)]
mod tests;
