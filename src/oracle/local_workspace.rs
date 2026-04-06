//! Local Workspace Oracle - Multi-project development intelligence
//!
//! Provides discovery and analysis of local PAIML projects for intelligent
//! orchestration across 10+ active development projects.
//!
//! ## Features
//!
//! - Auto-discover PAIML projects in ~/src
//! - Track git status across all projects
//! - Build cross-project dependency graph
//! - Detect version drift (local vs crates.io)
//! - Suggest publish order for dependent crates
//!
//! ## Toyota Way Principles
//!
//! - **Genchi Genbutsu**: Go and see the actual local state
//! - **Jidoka**: Stop on version conflicts before publishing
//! - **Just-in-Time**: Pull-based publish ordering

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::stack::{is_paiml_crate, CratesIoClient};

/// A discovered local project
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalProject {
    /// Project name (from Cargo.toml)
    pub name: String,
    /// Path to project root
    pub path: PathBuf,
    /// Local version from Cargo.toml
    pub local_version: String,
    /// Published version on crates.io (if any)
    pub published_version: Option<String>,
    /// Git status
    pub git_status: GitStatus,
    /// Development state (clean/dirty/unpushed)
    pub dev_state: DevState,
    /// Dependencies on other PAIML crates
    pub paiml_dependencies: Vec<DependencyInfo>,
    /// Whether this is a workspace
    pub is_workspace: bool,
    /// Workspace members (if workspace)
    pub workspace_members: Vec<String>,
}
impl LocalProject {
    /// Get the effective version for dependency resolution
    /// - Dirty projects: use crates.io version (local is WIP)
    /// - Clean projects: use local version
    pub fn effective_version(&self) -> &str {
        if self.dev_state.use_local_version() {
            &self.local_version
        } else {
            self.published_version.as_deref().unwrap_or(&self.local_version)
        }
    }

    /// Is this project blocking the stack?
    /// Only clean projects with version drift block the stack
    pub fn is_blocking(&self) -> bool {
        if !self.dev_state.use_local_version() {
            return false; // Dirty projects don't block
        }
        // Clean project - check if version is behind
        match &self.published_version {
            Some(pub_v) => compare_versions(&self.local_version, pub_v) == std::cmp::Ordering::Less,
            None => false,
        }
    }
}

/// Git repository status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitStatus {
    /// Current branch
    pub branch: String,
    /// Whether there are uncommitted changes
    pub has_changes: bool,
    /// Number of modified files
    pub modified_count: usize,
    /// Number of unpushed commits
    pub unpushed_commits: usize,
    /// Whether the branch is up to date with remote
    pub up_to_date: bool,
}

/// Information about a PAIML dependency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyInfo {
    /// Dependency name
    pub name: String,
    /// Required version (from Cargo.toml)
    pub required_version: String,
    /// Whether this is a path dependency
    pub is_path_dep: bool,
    /// Whether the local version satisfies the requirement
    pub version_satisfied: Option<bool>,
}

/// Version drift information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionDrift {
    /// Crate name
    pub name: String,
    /// Local version
    pub local_version: String,
    /// Published version
    pub published_version: String,
    /// Drift type
    pub drift_type: DriftType,
}

/// Type of version drift
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DriftType {
    /// Local is ahead of published (ready to publish)
    LocalAhead,
    /// Local is behind published (need to update)
    LocalBehind,
    /// Versions match
    InSync,
    /// Not published yet
    NotPublished,
}

/// Development state of a project
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DevState {
    /// Clean - no uncommitted changes, safe to use local version
    Clean,
    /// Dirty - active development, use crates.io version for deps
    Dirty,
    /// Unpushed - clean but has unpushed commits
    Unpushed,
}
impl DevState {
    /// Should this project's local version be used for dependency resolution?
    pub fn use_local_version(&self) -> bool {
        matches!(self, DevState::Clean)
    }

    /// Is this project safe to release?
    pub fn safe_to_release(&self) -> bool {
        matches!(self, DevState::Clean | DevState::Unpushed)
    }
}

/// Publish order for coordinated releases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublishOrder {
    /// Ordered list of crates to publish
    pub order: Vec<PublishStep>,
    /// Detected cycles (if any)
    pub cycles: Vec<Vec<String>>,
}

/// A step in the publish order
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublishStep {
    /// Crate name
    pub name: String,
    /// Current local version
    pub version: String,
    /// Dependencies that must be published first
    pub blocked_by: Vec<String>,
    /// Whether this crate has unpublished changes
    pub needs_publish: bool,
}

/// Local workspace oracle for multi-project intelligence
pub struct LocalWorkspaceOracle {
    /// Base directory to scan (typically ~/src)
    base_dir: PathBuf,
    /// Crates.io client for version checks
    crates_io: CratesIoClient,
    /// Discovered projects
    projects: HashMap<String, LocalProject>,
}

impl LocalWorkspaceOracle {
    /// Create a new oracle with default base directory (~//src)
    pub fn new() -> Result<Self> {
        let home = dirs::home_dir().context("Could not find home directory")?;
        let base_dir = home.join("src");
        Self::with_base_dir(base_dir)
    }

    /// Create with a specific base directory
    pub fn with_base_dir(base_dir: PathBuf) -> Result<Self> {
        Ok(Self { base_dir, crates_io: CratesIoClient::new(), projects: HashMap::new() })
    }

    /// Discover all PAIML projects in the base directory
    pub fn discover_projects(&mut self) -> Result<&HashMap<String, LocalProject>> {
        self.projects.clear();

        if !self.base_dir.exists() {
            return Ok(&self.projects);
        }

        // Scan for Cargo.toml files
        for entry in std::fs::read_dir(&self.base_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                let cargo_toml = path.join("Cargo.toml");
                if cargo_toml.exists() {
                    if let Ok(project) = self.analyze_project(&path) {
                        // Only include PAIML projects
                        if is_paiml_crate(&project.name) || self.has_paiml_deps(&project) {
                            self.projects.insert(project.name.clone(), project);
                        }
                    }
                }
            }
        }

        Ok(&self.projects)
    }

    /// Check if project has PAIML dependencies
    fn has_paiml_deps(&self, project: &LocalProject) -> bool {
        !project.paiml_dependencies.is_empty()
    }

    /// Analyze a single project
    fn analyze_project(&self, path: &Path) -> Result<LocalProject> {
        let cargo_toml = path.join("Cargo.toml");
        let content = std::fs::read_to_string(&cargo_toml)?;
        let parsed: toml::Value = toml::from_str(&content)?;

        // Get package info (handle workspaces)
        let (name, local_version, is_workspace, workspace_members) = if let Some(package) =
            parsed.get("package")
        {
            let name =
                package.get("name").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();

            let version = Self::extract_version(package, &parsed);

            (name, version, false, vec![])
        } else if let Some(workspace) = parsed.get("workspace") {
            // Workspace - use directory name
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("unknown").to_string();

            let members = workspace
                .get("members")
                .and_then(|m| m.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default();

            let version = workspace
                .get("package")
                .and_then(|p| p.get("version"))
                .and_then(|v| v.as_str())
                .unwrap_or("0.0.0")
                .to_string();

            (name, version, true, members)
        } else {
            anyhow::bail!("No [package] or [workspace] section");
        };

        // Get dependencies
        let paiml_dependencies = self.extract_paiml_deps(&parsed);

        // Get git status
        let git_status = self.get_git_status(path);

        // Determine development state
        let dev_state = if git_status.has_changes {
            DevState::Dirty
        } else if git_status.unpushed_commits > 0 {
            DevState::Unpushed
        } else {
            DevState::Clean
        };

        Ok(LocalProject {
            name,
            path: path.to_path_buf(),
            local_version,
            published_version: None, // Filled in later
            git_status,
            dev_state,
            paiml_dependencies,
            is_workspace,
            workspace_members,
        })
    }

    /// Extract version handling workspace inheritance
    fn extract_version(package: &toml::Value, root: &toml::Value) -> String {
        if let Some(version) = package.get("version") {
            if let Some(v) = version.as_str() {
                return v.to_string();
            }
            // Check for workspace = true
            if let Some(table) = version.as_table() {
                if table.get("workspace").and_then(|v| v.as_bool()) == Some(true) {
                    // Get from workspace.package.version
                    if let Some(ws_version) = root
                        .get("workspace")
                        .and_then(|w| w.get("package"))
                        .and_then(|p| p.get("version"))
                        .and_then(|v| v.as_str())
                    {
                        return ws_version.to_string();
                    }
                }
            }
        }
        "0.0.0".to_string()
    }

    /// Extract PAIML dependencies from Cargo.toml
    fn extract_paiml_deps(&self, parsed: &toml::Value) -> Vec<DependencyInfo> {
        let mut deps = Vec::new();

        // Check [dependencies]
        if let Some(dependencies) = parsed.get("dependencies") {
            self.collect_paiml_deps(dependencies, &mut deps);
        }

        // Check [dev-dependencies]
        if let Some(dev_deps) = parsed.get("dev-dependencies") {
            self.collect_paiml_deps(dev_deps, &mut deps);
        }

        // Check workspace dependencies
        if let Some(workspace) = parsed.get("workspace") {
            if let Some(ws_deps) = workspace.get("dependencies") {
                self.collect_paiml_deps(ws_deps, &mut deps);
            }
        }

        deps
    }

    fn collect_paiml_deps(&self, deps: &toml::Value, result: &mut Vec<DependencyInfo>) {
        if let Some(table) = deps.as_table() {
            for (name, value) in table {
                if !is_paiml_crate(name) {
                    continue;
                }

                let (version, is_path) = match value {
                    toml::Value::String(v) => (v.clone(), false),
                    toml::Value::Table(t) => {
                        let version =
                            t.get("version").and_then(|v| v.as_str()).unwrap_or("*").to_string();
                        let is_path = t.contains_key("path");
                        (version, is_path)
                    }
                    _ => continue,
                };

                result.push(DependencyInfo {
                    name: name.clone(),
                    required_version: version,
                    is_path_dep: is_path,
                    version_satisfied: None,
                });
            }
        }
    }

    /// Get git status for a project
    fn get_git_status(&self, path: &Path) -> GitStatus {
        let branch = Command::new("git")
            .args(["branch", "--show-current"])
            .current_dir(path)
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        let status_output = Command::new("git")
            .args(["status", "--porcelain"])
            .current_dir(path)
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .unwrap_or_default();

        let modified_count = status_output.lines().count();
        let has_changes = modified_count > 0;

        // Check unpushed commits
        let unpushed = Command::new("git")
            .args(["log", "@{u}..HEAD", "--oneline"])
            .current_dir(path)
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.lines().count())
            .unwrap_or(0);

        let up_to_date = unpushed == 0 && !has_changes;

        GitStatus { branch, has_changes, modified_count, unpushed_commits: unpushed, up_to_date }
    }

    /// Fetch published versions from crates.io
    pub async fn fetch_published_versions(&mut self) -> Result<()> {
        // Collect project names first to avoid borrow issues
        let names: Vec<String> = self.projects.keys().cloned().collect();

        for name in names {
            if let Ok(response) = self.crates_io.get_crate(&name).await {
                if let Some(project) = self.projects.get_mut(&name) {
                    project.published_version = Some(response.krate.max_version.clone());
                }
            }
        }
        Ok(())
    }

    /// Detect version drift between local and published
    pub fn detect_drift(&self) -> Vec<VersionDrift> {
        let mut drifts = Vec::new();

        for project in self.projects.values() {
            let drift_type = match &project.published_version {
                None => DriftType::NotPublished,
                Some(published) => {
                    use std::cmp::Ordering;
                    match compare_versions(&project.local_version, published) {
                        Ordering::Greater => DriftType::LocalAhead,
                        Ordering::Less => DriftType::LocalBehind,
                        Ordering::Equal => DriftType::InSync,
                    }
                }
            };

            if drift_type != DriftType::InSync {
                drifts.push(VersionDrift {
                    name: project.name.clone(),
                    local_version: project.local_version.clone(),
                    published_version: project
                        .published_version
                        .clone()
                        .unwrap_or_else(|| "not published".to_string()),
                    drift_type,
                });
            }
        }

        drifts
    }

    /// Build cross-project dependency graph and suggest publish order
    pub fn suggest_publish_order(&self) -> PublishOrder {
        // Build dependency graph
        let mut graph: HashMap<String, HashSet<String>> = HashMap::new();
        let mut in_degree: HashMap<String, usize> = HashMap::new();

        // Initialize all projects
        for name in self.projects.keys() {
            graph.entry(name.clone()).or_default();
            in_degree.entry(name.clone()).or_insert(0);
        }

        // Add edges for dependencies
        for project in self.projects.values() {
            for dep in &project.paiml_dependencies {
                if self.projects.contains_key(&dep.name) && !dep.is_path_dep {
                    graph.entry(dep.name.clone()).or_default().insert(project.name.clone());
                    *in_degree.entry(project.name.clone()).or_insert(0) += 1;
                }
            }
        }

        // Topological sort (Kahn's algorithm)
        let mut order = Vec::new();
        let mut queue: Vec<String> = in_degree
            .iter()
            .filter(|(_, &degree)| degree == 0)
            .map(|(name, _)| name.clone())
            .collect();

        queue.sort(); // Deterministic ordering

        while let Some(name) = queue.pop() {
            if let Some(project) = self.projects.get(&name) {
                let blocked_by: Vec<String> = project
                    .paiml_dependencies
                    .iter()
                    .filter(|d| self.projects.contains_key(&d.name) && !d.is_path_dep)
                    .map(|d| d.name.clone())
                    .collect();

                let needs_publish = project.git_status.has_changes
                    || project.git_status.unpushed_commits > 0
                    || matches!(
                        self.detect_drift().iter().find(|d| d.name == name).map(|d| d.drift_type),
                        Some(DriftType::LocalAhead | DriftType::NotPublished)
                    );

                order.push(PublishStep {
                    name: name.clone(),
                    version: project.local_version.clone(),
                    blocked_by,
                    needs_publish,
                });
            }

            // Decrease in-degree of dependents
            if let Some(dependents) = graph.get(&name) {
                for dependent in dependents {
                    if let Some(degree) = in_degree.get_mut(dependent) {
                        *degree -= 1;
                        if *degree == 0 {
                            queue.push(dependent.clone());
                            queue.sort();
                        }
                    }
                }
            }
        }

        // Detect cycles (remaining nodes with non-zero in-degree)
        let cycles: Vec<Vec<String>> = in_degree
            .iter()
            .filter(|(_, &degree)| degree > 0)
            .map(|(name, _)| vec![name.clone()])
            .collect();

        PublishOrder { order, cycles }
    }

    /// Get all discovered projects
    pub fn projects(&self) -> &HashMap<String, LocalProject> {
        &self.projects
    }

    /// Get summary statistics
    pub fn summary(&self) -> WorkspaceSummary {
        let total = self.projects.len();
        let with_changes = self.projects.values().filter(|p| p.git_status.has_changes).count();
        let with_unpushed =
            self.projects.values().filter(|p| p.git_status.unpushed_commits > 0).count();
        let workspaces = self.projects.values().filter(|p| p.is_workspace).count();

        WorkspaceSummary {
            total_projects: total,
            projects_with_changes: with_changes,
            projects_with_unpushed: with_unpushed,
            workspace_count: workspaces,
        }
    }
}

/// Summary of workspace state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceSummary {
    pub total_projects: usize,
    pub projects_with_changes: usize,
    pub projects_with_unpushed: usize,
    pub workspace_count: usize,
}

/// Compare semver versions
fn compare_versions(a: &str, b: &str) -> std::cmp::Ordering {
    let parse = |s: &str| -> (u32, u32, u32) {
        let parts: Vec<u32> = s.split('.').take(3).map(|p| p.parse().unwrap_or(0)).collect();
        (*parts.first().unwrap_or(&0), *parts.get(1).unwrap_or(&0), *parts.get(2).unwrap_or(&0))
    };

    parse(a).cmp(&parse(b))
}

#[cfg(test)]
#[path = "local_workspace_tests.rs"]
mod tests;
