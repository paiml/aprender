use crate::types::*;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::process::Command;

#[cfg(feature = "native")]
use tracing::{debug, info, warn};

#[cfg(feature = "native")]
use walkdir::WalkDir;

// Stub macros for WASM build
#[cfg(not(feature = "native"))]
macro_rules! info {
    ($($arg:tt)*) => {{}};
}

#[cfg(not(feature = "native"))]
macro_rules! debug {
    ($($arg:tt)*) => {{}};
}

#[cfg(not(feature = "native"))]
macro_rules! warn {
    ($($arg:tt)*) => {{}};
}

/// Analyze a project directory
#[allow(clippy::cognitive_complexity)]
pub fn analyze_project(
    path: &Path,
    include_tdg: bool,
    include_languages: bool,
    include_dependencies: bool,
) -> Result<ProjectAnalysis> {
    contract_pre_analyze!(path);
    info!("Starting project analysis at {:?}", path);

    let mut analysis = ProjectAnalysis::new(path.to_path_buf());

    // Always detect languages for file/line counts (needed by TDG display)
    if include_languages || include_tdg {
        info!("Detecting languages...");
        let stats = detect_languages(path)?;
        analysis.languages = stats;

        // Determine primary language (most lines of code)
        if let Some(primary) = analysis.languages.first() {
            analysis.primary_language = Some(primary.language.clone());
        }

        // Calculate total files and lines
        analysis.total_files = analysis.languages.iter().map(|s| s.file_count).sum();
        analysis.total_lines = analysis.languages.iter().map(|s| s.line_count).sum();
    }

    if include_dependencies {
        info!("Analyzing dependencies...");
        analysis.dependencies = detect_dependencies(path)?;
    }

    if include_tdg {
        info!("Calculating TDG score...");
        analysis.tdg_score = calculate_tdg_score(path);
    }

    Ok(analysis)
}

/// Detect programming languages in the project (stub when native disabled)
#[cfg(not(feature = "native"))]
fn detect_languages(_path: &Path) -> Result<Vec<LanguageStats>> {
    Ok(Vec::new())
}

/// Detect programming languages in the project
#[cfg(feature = "native")]
fn detect_languages(path: &Path) -> Result<Vec<LanguageStats>> {
    let mut language_stats: HashMap<Language, (usize, usize)> = HashMap::new();

    for entry in
        WalkDir::new(path).follow_links(false).into_iter().filter_entry(|e| !is_ignored(e.path()))
    {
        let entry = entry?;
        if !entry.file_type().is_file() {
            continue;
        }

        if let Some(lang) = detect_language_from_path(entry.path()) {
            let line_count = count_lines(entry.path()).unwrap_or(0);
            let stats = language_stats.entry(lang).or_insert((0, 0));
            stats.0 += 1; // file count
            stats.1 += line_count; // line count
        }
    }

    // Convert to LanguageStats and sort by line count (descending)
    let total_lines: usize = language_stats.values().map(|(_, lines)| lines).sum();
    let mut stats: Vec<LanguageStats> = language_stats
        .into_iter()
        .map(|(language, (file_count, line_count))| LanguageStats {
            language,
            file_count,
            line_count,
            percentage: if total_lines > 0 {
                (line_count as f64 / total_lines as f64) * 100.0
            } else {
                0.0
            },
        })
        .collect();

    stats.sort_by(|a, b| b.line_count.cmp(&a.line_count));

    Ok(stats)
}

/// Detect language from file extension
fn detect_language_from_path(path: &Path) -> Option<Language> {
    let extension = path.extension()?.to_str()?;

    match extension {
        "py" | "pyx" | "pyi" => Some(Language::Python),
        "c" | "h" => Some(Language::C),
        "cpp" | "cc" | "cxx" | "hpp" | "hxx" | "hh" => Some(Language::Cpp),
        "rs" => Some(Language::Rust),
        "sh" | "bash" | "zsh" => Some(Language::Shell),
        "js" | "jsx" | "mjs" => Some(Language::JavaScript),
        "ts" | "tsx" => Some(Language::TypeScript),
        "go" => Some(Language::Go),
        "java" => Some(Language::Java),
        _ => None,
    }
}

/// Count non-empty lines in a file
fn count_lines(path: &Path) -> Result<usize> {
    let content = fs::read_to_string(path).context("Failed to read file")?;
    Ok(content.lines().filter(|line| !line.trim().is_empty()).count())
}

/// Check if path should be ignored (common directories to skip)
fn is_ignored(path: &Path) -> bool {
    let ignore_names = [
        ".git",
        ".svn",
        ".hg",
        "node_modules",
        "target",
        "build",
        "dist",
        "__pycache__",
        ".pytest_cache",
        ".venv",
        "venv",
        ".idea",
        ".vscode",
    ];

    path.components().any(|c| {
        if let Some(name) = c.as_os_str().to_str() {
            ignore_names.contains(&name)
        } else {
            false
        }
    })
}

/// Detect dependency managers and files
fn detect_dependencies(path: &Path) -> Result<Vec<DependencyInfo>> {
    let mut deps = Vec::new();

    // Python
    if let Some(info) = check_dependency_file(path, "requirements.txt", DependencyManager::Pip) {
        deps.push(info);
    }
    if let Some(info) = check_dependency_file(path, "Pipfile", DependencyManager::Pipenv) {
        deps.push(info);
    }
    if let Some(info) = check_poetry_deps(path) {
        deps.push(info);
    }
    if let Some(info) = check_dependency_file(path, "environment.yml", DependencyManager::Conda) {
        deps.push(info);
    }

    // Rust
    if let Some(info) = check_dependency_file(path, "Cargo.toml", DependencyManager::Cargo) {
        deps.push(info);
    }

    // JavaScript/Node
    if let Some(info) = check_dependency_file(path, "package.json", DependencyManager::Npm) {
        deps.push(info);
    }
    if let Some(info) = check_dependency_file(path, "yarn.lock", DependencyManager::Yarn) {
        deps.push(info);
    }

    // Go
    if let Some(info) = check_dependency_file(path, "go.mod", DependencyManager::GoMod) {
        deps.push(info);
    }

    // Java
    if let Some(info) = check_dependency_file(path, "pom.xml", DependencyManager::Maven) {
        deps.push(info);
    }
    if let Some(info) = check_dependency_file(path, "build.gradle", DependencyManager::Gradle) {
        deps.push(info);
    }

    // C/C++
    if let Some(info) = check_dependency_file(path, "Makefile", DependencyManager::Make) {
        deps.push(info);
    }

    Ok(deps)
}

fn check_dependency_file(
    base_path: &Path,
    filename: &str,
    manager: DependencyManager,
) -> Option<DependencyInfo> {
    let file_path = base_path.join(filename);
    if file_path.exists() {
        debug!("Found dependency file: {:?}", file_path);
        let count = count_dependencies(&file_path, &manager);
        Some(DependencyInfo { manager, file_path, count })
    } else {
        None
    }
}

fn check_poetry_deps(base_path: &Path) -> Option<DependencyInfo> {
    let file_path = base_path.join("pyproject.toml");
    if file_path.exists() {
        if let Ok(content) = fs::read_to_string(&file_path) {
            if content.contains("[tool.poetry]") {
                debug!("Found Poetry project: {:?}", file_path);
                let count = count_dependencies(&file_path, &DependencyManager::Poetry);
                return Some(DependencyInfo {
                    manager: DependencyManager::Poetry,
                    file_path,
                    count,
                });
            }
        }
    }
    None
}

fn count_pip_dependencies(content: &str) -> usize {
    content
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            !trimmed.is_empty() && !trimmed.starts_with('#')
        })
        .count()
}

fn count_cargo_dependencies(content: &str) -> usize {
    let mut in_deps = false;
    let mut count = 0;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed == "[dependencies]" || trimmed == "[dev-dependencies]" {
            in_deps = true;
        } else if trimmed.starts_with('[') {
            in_deps = false;
        } else if in_deps && !trimmed.is_empty() && !trimmed.starts_with('#') {
            count += 1;
        }
    }
    count
}

fn count_npm_dependencies(content: &str) -> Option<usize> {
    let json: serde_json::Value = serde_json::from_str(content).ok()?;
    let deps = json.get("dependencies").and_then(|d| d.as_object());
    let dev_deps = json.get("devDependencies").and_then(|d| d.as_object());
    Some(deps.map(|d| d.len()).unwrap_or(0) + dev_deps.map(|d| d.len()).unwrap_or(0))
}

fn count_dependencies(path: &Path, manager: &DependencyManager) -> Option<usize> {
    let content = fs::read_to_string(path).ok()?;

    match manager {
        DependencyManager::Pip => Some(count_pip_dependencies(&content)),
        DependencyManager::Cargo => Some(count_cargo_dependencies(&content)),
        DependencyManager::Npm => count_npm_dependencies(&content),
        _ => None,
    }
}

/// Calculate TDG score using PMAT, with fallback for when PMAT is unavailable
fn calculate_tdg_score(path: &Path) -> Option<f64> {
    debug!("Running PMAT TDG analysis...");

    // Try to use PMAT first
    if let Some(score) = calculate_tdg_with_pmat(path) {
        return Some(score);
    }

    // Fallback: basic heuristic TDG score when PMAT unavailable
    debug!("PMAT unavailable, using fallback TDG calculation");
    calculate_tdg_fallback(path)
}

/// Parse PMAT output for score line: "Overall Score: 100.0/100 (A+)"
fn parse_pmat_score_line(line: &str) -> Option<f64> {
    if !line.contains("Overall Score:") {
        return None;
    }
    let score_str = line.split(':').nth(1)?;
    let score = score_str.trim().split('/').next()?;
    score.trim().parse::<f64>().ok()
}

/// Calculate TDG using external PMAT tool
fn calculate_tdg_with_pmat(path: &Path) -> Option<f64> {
    let output = Command::new("pmat").arg("tdg").arg(path).output().ok()?;

    if !output.status.success() {
        warn!("PMAT TDG command failed");
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout.lines().find_map(parse_pmat_score_line)
}

/// Fallback TDG calculation (stub when native disabled)
#[cfg(not(feature = "native"))]
fn calculate_tdg_fallback(_path: &Path) -> Option<f64> {
    None
}

/// Fallback TDG calculation using basic heuristics
/// This provides a reasonable estimate when PMAT is not available
#[cfg(feature = "native")]
fn calculate_tdg_fallback(path: &Path) -> Option<f64> {
    let mut score: f64 = 100.0;

    // Check for tests directory or #[test] presence
    let has_tests = path.join("tests").exists()
        || WalkDir::new(path)
            .max_depth(3)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "rs"))
            .take(10)
            .any(|e| {
                std::fs::read_to_string(e.path())
                    .ok()
                    .is_some_and(|content| content.contains("#[test]"))
            });

    if !has_tests {
        score -= 10.0; // Deduct for no tests
    }

    // Check for README
    if !path.join("README.md").exists() && !path.join("README").exists() {
        score -= 5.0;
    }

    // Check for CI configuration
    let has_ci = path.join(".github/workflows").exists()
        || path.join(".gitlab-ci.yml").exists()
        || path.join(".circleci").exists();
    if !has_ci {
        score -= 5.0;
    }

    // Check for license
    let has_license = path.join("LICENSE").exists()
        || path.join("LICENSE.md").exists()
        || path.join("LICENSE.txt").exists();
    if !has_license {
        score -= 5.0;
    }

    Some(score.max(0.0))
}

#[cfg(test)]
#[path = "analyzer_tests.rs"]
mod tests;
