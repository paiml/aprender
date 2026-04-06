//! Cross-project contract usage index.
//!
//! Discovers sibling projects that consume provable-contracts and indexes
//! contract references: `#[contract(...)]` annotations, binding.yaml entries,
//! and KAIZEN/contract-ID patterns in code and git commits.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// A discovered sibling project that consumes contracts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectEntry {
    pub name: String,
    pub path: PathBuf,
    pub has_cargo_toml: bool,
    pub has_binding: bool,
    pub binding_path: Option<PathBuf>,
}

/// A `#[contract("...", equation = "...")]` annotation in consumer code.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallSite {
    pub project: String,
    pub file: String,
    pub line: u32,
    pub contract_stem: String,
    pub equation: Option<String>,
}

/// A binding.yaml reference from a consumer project.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingRef {
    pub project: String,
    pub binding_path: String,
    pub contract: String,
    pub equation: String,
    pub status: String,
}

/// A KAIZEN-NNN or C-UPPER-DIGITS reference in code or commits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KaizenRef {
    pub project: String,
    pub file: String,
    pub line: u32,
    pub pattern: String,
}

/// A contract or KAIZEN reference found in git commit messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitRef {
    pub project: String,
    pub commit_hash: String,
    pub pattern: String,
}

/// Cross-project index aggregating all contract references across the stack.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossProjectIndex {
    pub projects: Vec<ProjectEntry>,
    pub call_sites: HashMap<String, Vec<CallSite>>,
    pub binding_refs: HashMap<String, Vec<BindingRef>>,
    pub kaizen_refs: HashMap<String, Vec<KaizenRef>>,
    pub commit_refs: HashMap<String, Vec<CommitRef>>,
}

impl CrossProjectIndex {
    /// Build a cross-project index by discovering sibling projects.
    ///
    /// Starts from `contracts_repo_root` (the provable-contracts repo),
    /// walks `../` for sibling projects with `Cargo.toml`, and scans them
    /// for contract references.
    pub fn build(contracts_repo_root: &Path) -> Self {
        Self::build_with_extra(contracts_repo_root, None)
    }

    /// Build with an optional extra project path to include.
    pub fn build_with_extra(contracts_repo_root: &Path, extra_path: Option<&Path>) -> Self {
        let root = contracts_repo_root
            .canonicalize()
            .unwrap_or_else(|_| contracts_repo_root.to_path_buf());
        let parent = root.parent().unwrap_or(&root);

        let mut projects = discover_projects(parent, &root);

        // Add explicit extra project if provided and not already discovered
        if let Some(extra) = extra_path {
            let extra_canon = extra.canonicalize().unwrap_or_else(|_| extra.to_path_buf());
            if extra_canon.is_dir() && !projects.iter().any(|p| p.path == extra_canon) {
                let name = extra_canon
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string();
                let binding_path = find_binding_path(&extra_canon, &name);
                projects.push(ProjectEntry {
                    name,
                    has_cargo_toml: extra_canon.join("Cargo.toml").exists(),
                    has_binding: binding_path.is_some(),
                    binding_path,
                    path: extra_canon,
                });
            }
        }

        let mut call_sites: HashMap<String, Vec<CallSite>> = HashMap::new();
        let mut binding_refs: HashMap<String, Vec<BindingRef>> = HashMap::new();
        let mut kaizen_refs: HashMap<String, Vec<KaizenRef>> = HashMap::new();
        let mut commit_refs: HashMap<String, Vec<CommitRef>> = HashMap::new();

        for project in &projects {
            scan_contract_annotations(project, &mut call_sites);
            scan_binding_refs(project, &mut binding_refs);
            scan_kaizen_refs(project, &mut kaizen_refs);
            scan_commit_refs(project, &mut commit_refs);
        }

        Self {
            projects,
            call_sites,
            binding_refs,
            kaizen_refs,
            commit_refs,
        }
    }

    /// Get all call sites for a given contract stem.
    pub fn call_sites_for(&self, stem: &str) -> &[CallSite] {
        self.call_sites.get(stem).map_or(&[], Vec::as_slice)
    }

    /// Get all binding references for a given contract stem.
    pub fn binding_refs_for(&self, stem: &str) -> &[BindingRef] {
        self.binding_refs.get(stem).map_or(&[], Vec::as_slice)
    }

    /// Get all KAIZEN references for a given pattern.
    pub fn kaizen_refs_for(&self, pattern: &str) -> &[KaizenRef] {
        self.kaizen_refs.get(pattern).map_or(&[], Vec::as_slice)
    }

    /// Get all commit message references for a given pattern.
    pub fn commit_refs_for(&self, pattern: &str) -> &[CommitRef] {
        self.commit_refs.get(pattern).map_or(&[], Vec::as_slice)
    }

    /// Total call sites across all contracts.
    pub fn total_call_sites(&self) -> usize {
        self.call_sites.values().map(Vec::len).sum()
    }

    /// Total projects discovered.
    pub fn project_count(&self) -> usize {
        self.projects.len()
    }
}

/// Discover sibling projects that might consume contracts.
fn discover_projects(parent_dir: &Path, self_dir: &Path) -> Vec<ProjectEntry> {
    let mut projects = Vec::new();
    let Ok(entries) = std::fs::read_dir(parent_dir) else {
        return projects;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() || path == self_dir {
            continue;
        }

        let has_cargo_toml = path.join("Cargo.toml").exists();
        if !has_cargo_toml {
            continue;
        }

        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        // Look for binding.yaml in contracts/<project>/ or at the root
        let binding_path = find_binding_path(&path, &name);

        projects.push(ProjectEntry {
            name,
            has_cargo_toml,
            has_binding: binding_path.is_some(),
            binding_path,
            path,
        });
    }

    projects.sort_by(|a, b| a.name.cmp(&b.name));
    projects
}

/// Find binding.yaml for a project — check common locations.
fn find_binding_path(project_dir: &Path, name: &str) -> Option<PathBuf> {
    // Check contracts/<name>/binding.yaml (in provable-contracts repo)
    let sibling_binding = project_dir
        .parent()?
        .join("provable-contracts/contracts")
        .join(name)
        .join("binding.yaml");
    if sibling_binding.exists() {
        return Some(sibling_binding);
    }

    // Check binding.yaml at project root
    let root_binding = project_dir.join("binding.yaml");
    if root_binding.exists() {
        return Some(root_binding);
    }

    None
}

/// Scan .rs files for `#[contract("...", equation = "...")]` annotations.
fn scan_contract_annotations(
    project: &ProjectEntry,
    call_sites: &mut HashMap<String, Vec<CallSite>>,
) {
    let src_dir = project.path.join("src");
    if !src_dir.exists() {
        return;
    }

    let output = std::process::Command::new("grep")
        .args(["-rn", "contract(\"", "--include=*.rs"])
        .arg(&src_dir)
        .output();

    let Ok(output) = output else { return };
    if !output.status.success() {
        return;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if let Some(site) = parse_contract_annotation(line, &project.name, &project.path) {
            call_sites
                .entry(site.contract_stem.clone())
                .or_default()
                .push(site);
        }
    }
}

/// Parse a grep output line into a `CallSite`.
fn parse_contract_annotation(
    line: &str,
    project_name: &str,
    project_path: &Path,
) -> Option<CallSite> {
    // grep output: /abs/path/file.rs:42:content
    let parts: Vec<&str> = line.splitn(3, ':').collect();
    if parts.len() < 3 {
        return None;
    }
    let file_path = parts[0];
    let line_no: u32 = parts[1].parse().ok()?;
    let content = parts[2];

    // Extract contract stem from content like: contract("stem-v1", ...)
    let stem = extract_contract_stem(content)?;

    // Extract equation if present
    let equation = extract_equation(content);

    // Make file path relative to project
    let relative = file_path
        .strip_prefix(project_path.to_string_lossy().as_ref())
        .unwrap_or(file_path)
        .trim_start_matches('/');

    Some(CallSite {
        project: project_name.to_string(),
        file: relative.to_string(),
        line: line_no,
        contract_stem: stem,
        equation,
    })
}

/// Extract contract stem from annotation content.
fn extract_contract_stem(content: &str) -> Option<String> {
    // Look for contract("stem-v1" or contract("stem-v1.yaml"
    let idx = content.find("contract(\"")?;
    let start = idx + "contract(\"".len();
    let rest = &content[start..];
    let end = rest.find('"')?;
    let stem = &rest[..end];
    // Strip .yaml extension if present
    Some(stem.strip_suffix(".yaml").unwrap_or(stem).to_string())
}

/// Extract equation from annotation content.
fn extract_equation(content: &str) -> Option<String> {
    // Look for equation = "eq_name"
    let idx = content.find("equation")?;
    let rest = &content[idx..];
    let q_start = rest.find('"')? + 1;
    let q_rest = &rest[q_start..];
    let q_end = q_rest.find('"')?;
    Some(q_rest[..q_end].to_string())
}

/// Scan binding.yaml files for contract references.
fn scan_binding_refs(project: &ProjectEntry, refs: &mut HashMap<String, Vec<BindingRef>>) {
    let Some(binding_path) = &project.binding_path else {
        return;
    };
    let Ok(content) = std::fs::read_to_string(binding_path) else {
        return;
    };
    let Ok(registry) = crate::binding::parse_binding_str(&content) else {
        return;
    };

    for binding in &registry.bindings {
        let stem = binding
            .contract
            .strip_suffix(".yaml")
            .unwrap_or(&binding.contract)
            .to_string();

        refs.entry(stem).or_default().push(BindingRef {
            project: project.name.clone(),
            binding_path: binding_path.display().to_string(),
            contract: binding.contract.clone(),
            equation: binding.equation.clone(),
            status: binding.status.to_string(),
        });
    }
}

/// Scan .rs files for KAIZEN-NNN and C-UPPER-DIGITS patterns.
fn scan_kaizen_refs(project: &ProjectEntry, refs: &mut HashMap<String, Vec<KaizenRef>>) {
    let src_dir = project.path.join("src");
    if !src_dir.exists() {
        return;
    }

    let output = std::process::Command::new("grep")
        .args([
            "-rn",
            "-E",
            r"KAIZEN-[0-9]+|C-[A-Z]+-[0-9]+",
            "--include=*.rs",
        ])
        .arg(&src_dir)
        .output();

    let Ok(output) = output else { return };
    if !output.status.success() {
        return;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        let parts: Vec<&str> = line.splitn(3, ':').collect();
        if parts.len() < 3 {
            continue;
        }
        let line_no: u32 = match parts[1].parse() {
            Ok(n) => n,
            Err(_) => continue,
        };
        let content = parts[2];

        let relative = parts[0]
            .strip_prefix(project.path.to_string_lossy().as_ref())
            .unwrap_or(parts[0])
            .trim_start_matches('/');

        // Extract all KAIZEN-NNN and C-UPPER-DIGITS patterns
        for pattern in extract_patterns(content) {
            refs.entry(pattern.clone()).or_default().push(KaizenRef {
                project: project.name.clone(),
                file: relative.to_string(),
                line: line_no,
                pattern,
            });
        }
    }
}

/// Extract KAIZEN-NNN and C-UPPER-DIGITS patterns from a line.
fn extract_patterns(content: &str) -> Vec<String> {
    let mut patterns = Vec::new();
    let mut rest = content;
    while let Some(idx) = rest.find("KAIZEN-") {
        let start = idx;
        let after = &rest[idx + 7..];
        let end = after
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(after.len());
        if end > 0 {
            patterns.push(rest[start..start + 7 + end].to_string());
        }
        rest = &rest[start + 7 + end..];
    }

    rest = content;
    while let Some(idx) = rest.find("C-") {
        let after_c = &rest[idx + 2..];
        // Must match C-<UPPER>+-<DIGIT>+
        let alpha_end = after_c.find(|c: char| !c.is_ascii_uppercase()).unwrap_or(0);
        if alpha_end > 0 && after_c.get(alpha_end..=alpha_end) == Some("-") {
            let digit_start = alpha_end + 1;
            let digit_rest = &after_c[digit_start..];
            let digit_end = digit_rest
                .find(|c: char| !c.is_ascii_digit())
                .unwrap_or(digit_rest.len());
            if digit_end > 0 {
                let full_end = idx + 2 + digit_start + digit_end;
                patterns.push(rest[idx..full_end].to_string());
                rest = &rest[full_end..];
                continue;
            }
        }
        rest = &rest[idx + 2..];
    }

    patterns
}

/// Scan recent git commit messages for KAIZEN-NNN and C-UPPER-DIGITS patterns.
fn scan_commit_refs(project: &ProjectEntry, refs: &mut HashMap<String, Vec<CommitRef>>) {
    if !project.path.join(".git").exists() {
        return;
    }
    // Get recent commit messages (last 200 commits)
    let output = std::process::Command::new("git")
        .args(["log", "--oneline", "-200", "--format=%H %s"])
        .current_dir(&project.path)
        .output();

    let Ok(output) = output else { return };
    if !output.status.success() {
        return;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        let Some((hash, subject)) = line.split_once(' ') else {
            continue;
        };
        let short_hash = &hash[..hash.len().min(12)];
        for pattern in extract_patterns(subject) {
            refs.entry(pattern.clone()).or_default().push(CommitRef {
                project: project.name.clone(),
                commit_hash: short_hash.to_string(),
                pattern,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    include!("cross_project_tests.rs");
}
