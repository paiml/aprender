//! User-invocable + auto-loadable skill definitions (Claude-Code parity).
//!
//! PMAT-CODE-SKILLS-001: Claude Code loads `.claude/skills/<name>/SKILL.md`
//! (and `~/.claude/skills/` at user scope). Each `SKILL.md` carries
//! markdown frontmatter declaring a `name`, `description`, optional
//! `when_to_use` heuristic, and optional `allowed-tools` allowlist. The
//! body is treated as the skill's instructions — injected into the
//! model's system context when the skill is invoked either
//! (a) explicitly by the user via `/<skill-name>` in the REPL, or
//! (b) automatically by matching `when_to_use` against the pending turn.
//!
//! `apr code` mirrors the same pattern at `.apr/skills/` with identical
//! frontmatter schema so Claude-native projects can share skills with
//! `apr code` unchanged.
//!
//! # Example — `.apr/skills/rust-test.md`
//!
//! ```markdown
//! ---
//! name: rust-test
//! description: Run the Rust test suite and interpret failures
//! when_to_use: User asks about failing cargo tests
//! allowed-tools: shell, grep
//! ---
//!
//! Run `cargo test -p <crate> --lib <pattern>` from the repo root.
//! When a test fails, read the span around the assertion and report
//! the minimal reproduction.
//! ```
//!
//! After discovery, `registry.resolve("rust-test")` returns the
//! [`Skill`] whose `instructions` field can be appended to the system
//! prompt of the active agent turn.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Default project-scope skills directory (relative to cwd).
pub const DEFAULT_PROJECT_DIR: &str = ".apr/skills";

/// Legacy / cross-compat directory — share skills tree with Claude Code.
pub const CLAUDE_COMPAT_DIR: &str = ".claude/skills";

/// Errors that can arise while parsing or loading a skill file.
#[derive(Debug)]
pub enum SkillError {
    /// No `---`-fenced frontmatter at the top of the file.
    MissingFrontmatter,
    /// Required field `name` absent or empty.
    MissingName,
    /// Required field `description` absent or empty.
    MissingDescription,
    /// Body (instructions) after frontmatter was empty.
    EmptyBody,
    /// Filesystem error.
    Io(String),
}

impl std::fmt::Display for SkillError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingFrontmatter => write!(f, "missing `---`-fenced frontmatter"),
            Self::MissingName => write!(f, "required field `name` missing or empty"),
            Self::MissingDescription => {
                write!(f, "required field `description` missing or empty")
            }
            Self::EmptyBody => write!(f, "body (instructions) is empty"),
            Self::Io(msg) => write!(f, "I/O error: {msg}"),
        }
    }
}

impl std::error::Error for SkillError {}

/// A parsed Skill definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skill {
    /// Identifier used by `/<name>` slash invocation.
    pub name: String,
    /// One-line purpose, shown in UI listings.
    pub description: String,
    /// Heuristic the auto-loader matches against the current turn.
    pub when_to_use: Option<String>,
    /// Tool-allowlist enforcement when the skill runs. Empty → inherit.
    pub allowed_tools: Vec<String>,
    /// Markdown body — the actual skill text injected into context.
    pub instructions: String,
}

/// Registry of discovered skills, keyed by `name` (alphabetical iteration).
#[derive(Debug, Clone, Default)]
pub struct SkillRegistry {
    by_name: BTreeMap<String, Skill>,
}

impl SkillRegistry {
    /// Empty registry.
    pub fn new() -> Self {
        Self { by_name: BTreeMap::new() }
    }

    /// Register (or replace) a skill.
    pub fn register(&mut self, skill: Skill) {
        self.by_name.insert(skill.name.clone(), skill);
    }

    /// Resolve by skill name.
    pub fn resolve(&self, name: &str) -> Option<&Skill> {
        self.by_name.get(name)
    }

    /// Total number of registered skills.
    pub fn len(&self) -> usize {
        self.by_name.len()
    }

    /// True when no skills are registered.
    pub fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }

    /// Alphabetical list of registered skill names.
    pub fn names(&self) -> Vec<String> {
        self.by_name.keys().cloned().collect()
    }

    /// Find the first skill whose `when_to_use` heuristic lexically
    /// overlaps with the given turn. A skill matches when **at least
    /// two** whitespace-separated tokens of length ≥ 4 in its
    /// `when_to_use` appear (case-insensitive substring) in the turn.
    /// Two-token threshold keeps single-word false positives
    /// (e.g. "about", "tests") from spuriously triggering. Callers
    /// inject the resolved skill's `instructions` into the active
    /// system prompt.
    pub fn auto_match(&self, turn: &str) -> Option<&Skill> {
        let hay = turn.to_ascii_lowercase();
        self.by_name.values().find(|s| {
            let Some(needle) = s.when_to_use.as_ref() else {
                return false;
            };
            let hits = needle
                .split_whitespace()
                .filter(|t| t.len() >= 4)
                .map(|t| t.to_ascii_lowercase())
                .filter(|t| hay.contains(t))
                .count();
            hits >= 2
        })
    }
}

/// Parse a `SKILL.md` document into a [`Skill`].
///
/// Frontmatter format: leading `---\n` (or `---\r\n`), then lines of
/// `key: value`, then `---\n`, then the body (skill instructions).
/// `allowed-tools` (or `allowed_tools`) may be comma- or space-separated.
pub fn parse_skill_md(source: &str) -> Result<Skill, SkillError> {
    let trimmed = source.trim_start_matches('\u{feff}');
    let rest = trimmed
        .strip_prefix("---\n")
        .or_else(|| trimmed.strip_prefix("---\r\n"))
        .ok_or(SkillError::MissingFrontmatter)?;

    let (front, body) = split_at_fence(rest).ok_or(SkillError::MissingFrontmatter)?;

    let mut name = String::new();
    let mut description = String::new();
    let mut when_to_use: Option<String> = None;
    let mut allowed_tools: Vec<String> = Vec::new();

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
            "when_to_use" | "when-to-use" if !value.is_empty() => {
                when_to_use = Some(value.to_string());
            }
            "allowed-tools" | "allowed_tools" => {
                allowed_tools = value
                    .split([',', ' '])
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .collect();
            }
            // Claude-compat: `context: fork` etc. silently tolerated.
            _ => {}
        }
    }

    if name.is_empty() {
        return Err(SkillError::MissingName);
    }
    if description.is_empty() {
        return Err(SkillError::MissingDescription);
    }
    let instructions = body.trim().to_string();
    if instructions.is_empty() {
        return Err(SkillError::EmptyBody);
    }

    Ok(Skill { name, description, when_to_use, allowed_tools, instructions })
}

/// Scan a directory for skill `.md` files. Both layouts supported:
///   * `dir/<name>.md` — flat file per skill.
///   * `dir/<name>/SKILL.md` — subdirectory per skill (Claude default).
///
/// Silently skips files that fail to parse (malformed skill should not
/// disable all skills).
pub fn load_skills_from(dir: &Path) -> Vec<Skill> {
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir(dir) else {
        return out;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() {
            if path.extension().is_some_and(|e| e == "md") {
                if let Some(s) = try_parse(&path) {
                    out.push(s);
                }
            }
        } else if path.is_dir() {
            let skill_md = path.join("SKILL.md");
            if skill_md.is_file() {
                if let Some(s) = try_parse(&skill_md) {
                    out.push(s);
                }
            }
        }
    }

    out
}

/// Discover skills from the standard locations: user scope
/// (`~/.config/apr/skills/`) then project scope (`.apr/skills/` or,
/// as fallback, `.claude/skills/` — cross-compat). Project scope wins
/// on name collision.
pub fn discover_skills(cwd: &Path) -> Vec<Skill> {
    let mut merged: Vec<Skill> = Vec::new();

    if let Some(u) = user_skills_dir().as_deref() {
        merged.extend(load_skills_from(u));
    }

    for rel in [DEFAULT_PROJECT_DIR, CLAUDE_COMPAT_DIR] {
        let project_dir = cwd.join(rel);
        if project_dir.is_dir() {
            let project_skills = load_skills_from(&project_dir);
            for s in project_skills {
                merged.retain(|existing| existing.name != s.name);
                merged.push(s);
            }
            break;
        }
    }

    merged
}

/// Register all discovered skills into the given [`SkillRegistry`].
/// Returns the number of skills added.
pub fn register_discovered_skills_into(registry: &mut SkillRegistry, cwd: &Path) -> usize {
    let skills = discover_skills(cwd);
    let n = skills.len();
    for s in skills {
        registry.register(s);
    }
    n
}

fn try_parse(path: &Path) -> Option<Skill> {
    let src = fs::read_to_string(path).ok()?;
    parse_skill_md(&src).ok()
}

fn split_at_fence(after_open: &str) -> Option<(&str, &str)> {
    for line_start in line_starts(after_open) {
        let rest_at = &after_open[line_start..];
        if let Some(line_end) = rest_at.find('\n') {
            let line = &rest_at[..line_end];
            if line.trim_end_matches('\r') == "---" {
                let body_start = line_start + line_end + 1;
                return Some((&after_open[..line_start], &after_open[body_start..]));
            }
        } else if rest_at.trim_end_matches('\r') == "---" {
            return Some((&after_open[..line_start], ""));
        }
    }
    None
}

fn line_starts(s: &str) -> impl Iterator<Item = usize> + '_ {
    std::iter::once(0usize).chain(s.match_indices('\n').map(|(pos, _)| pos + 1))
}

fn user_skills_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let candidate = PathBuf::from(home).join(".config").join("apr").join("skills");
    candidate.is_dir().then_some(candidate)
}

#[cfg(test)]
mod tests;
