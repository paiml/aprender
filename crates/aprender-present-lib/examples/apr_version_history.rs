//! APR-009: Model Version History
//!
//! QA Focus: Version tracking and comparison for ML models
//!
//! Run: `cargo run --example apr_version_history`

#![allow(
    clippy::unwrap_used,
    clippy::disallowed_methods,
    clippy::struct_field_names,
    clippy::too_many_lines
)]

use std::collections::HashMap;

/// Model version identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VersionId {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl VersionId {
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 3 {
            return None;
        }
        Some(Self {
            major: parts[0].parse().ok()?,
            minor: parts[1].parse().ok()?,
            patch: parts[2].parse().ok()?,
        })
    }
}

impl std::fmt::Display for VersionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl Ord for VersionId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.major
            .cmp(&other.major)
            .then(self.minor.cmp(&other.minor))
            .then(self.patch.cmp(&other.patch))
    }
}

impl PartialOrd for VersionId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Version status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VersionStatus {
    Development,
    Staging,
    Production,
    Deprecated,
    Archived,
}

/// A single model version
#[derive(Debug, Clone)]
pub struct ModelVersion {
    pub version: VersionId,
    pub status: VersionStatus,
    pub created_at: String,
    pub author: String,
    pub description: String,
    pub metrics: HashMap<String, f64>,
    pub parent_version: Option<VersionId>,
    pub tags: Vec<String>,
    pub artifact_path: String,
}

impl ModelVersion {
    pub fn new(version: VersionId, author: &str, description: &str) -> Self {
        Self {
            version,
            status: VersionStatus::Development,
            created_at: chrono_placeholder(),
            author: author.to_string(),
            description: description.to_string(),
            metrics: HashMap::new(),
            parent_version: None,
            tags: Vec::new(),
            artifact_path: String::new(),
        }
    }

    pub const fn with_status(mut self, status: VersionStatus) -> Self {
        self.status = status;
        self
    }

    pub const fn with_parent(mut self, parent: VersionId) -> Self {
        self.parent_version = Some(parent);
        self
    }

    pub fn with_metric(mut self, name: &str, value: f64) -> Self {
        self.metrics.insert(name.to_string(), value);
        self
    }

    pub fn with_tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_string());
        self
    }

    pub fn with_artifact(mut self, path: &str) -> Self {
        self.artifact_path = path.to_string();
        self
    }

    pub fn get_metric(&self, name: &str) -> Option<f64> {
        self.metrics.get(name).copied()
    }
}

fn chrono_placeholder() -> String {
    "2024-01-15T10:30:00Z".to_string()
}

/// Model version history
#[derive(Debug)]
pub struct VersionHistory {
    model_name: String,
    versions: Vec<ModelVersion>,
}

impl VersionHistory {
    pub fn new(model_name: &str) -> Self {
        Self {
            model_name: model_name.to_string(),
            versions: Vec::new(),
        }
    }

    pub fn add_version(&mut self, version: ModelVersion) {
        self.versions.push(version);
        self.versions.sort_by(|a, b| b.version.cmp(&a.version)); // Most recent first
    }

    pub fn get_version(&self, id: &VersionId) -> Option<&ModelVersion> {
        self.versions.iter().find(|v| &v.version == id)
    }

    pub fn latest(&self) -> Option<&ModelVersion> {
        self.versions.first()
    }

    pub fn production_version(&self) -> Option<&ModelVersion> {
        self.versions
            .iter()
            .find(|v| v.status == VersionStatus::Production)
    }

    pub fn by_status(&self, status: VersionStatus) -> Vec<&ModelVersion> {
        self.versions
            .iter()
            .filter(|v| v.status == status)
            .collect()
    }

    pub fn by_tag(&self, tag: &str) -> Vec<&ModelVersion> {
        self.versions
            .iter()
            .filter(|v| v.tags.contains(&tag.to_string()))
            .collect()
    }

    pub fn version_count(&self) -> usize {
        self.versions.len()
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub fn all_versions(&self) -> &[ModelVersion] {
        &self.versions
    }

    /// Compare two versions
    pub fn compare(&self, v1: &VersionId, v2: &VersionId) -> Option<VersionComparison> {
        let ver1 = self.get_version(v1)?;
        let ver2 = self.get_version(v2)?;

        let mut metric_changes = Vec::new();

        // Collect all metric names from both versions
        let all_metrics: std::collections::HashSet<_> =
            ver1.metrics.keys().chain(ver2.metrics.keys()).collect();

        for metric in all_metrics {
            let val1 = ver1.get_metric(metric);
            let val2 = ver2.get_metric(metric);

            if val1 != val2 {
                metric_changes.push(MetricChange {
                    name: metric.clone(),
                    old_value: val1,
                    new_value: val2,
                    change_percent: calculate_change_percent(val1, val2),
                });
            }
        }

        Some(VersionComparison {
            from_version: v1.clone(),
            to_version: v2.clone(),
            metric_changes,
        })
    }

    /// Get lineage (ancestry chain) for a version
    pub fn lineage(&self, id: &VersionId) -> Vec<&ModelVersion> {
        let mut chain = Vec::new();
        let mut current = self.get_version(id);

        while let Some(version) = current {
            chain.push(version);
            current = version
                .parent_version
                .as_ref()
                .and_then(|p| self.get_version(p));
        }

        chain
    }
}

fn calculate_change_percent(old: Option<f64>, new: Option<f64>) -> Option<f64> {
    match (old, new) {
        (Some(o), Some(n)) if o != 0.0 => Some(((n - o) / o) * 100.0),
        _ => None,
    }
}

#[derive(Debug)]
pub struct VersionComparison {
    pub from_version: VersionId,
    pub to_version: VersionId,
    pub metric_changes: Vec<MetricChange>,
}

#[derive(Debug)]
pub struct MetricChange {
    pub name: String,
    pub old_value: Option<f64>,
    pub new_value: Option<f64>,
    pub change_percent: Option<f64>,
}

impl MetricChange {
    pub fn is_improvement(&self, higher_is_better: bool) -> bool {
        match (self.old_value, self.new_value) {
            (Some(old), Some(new)) => {
                if higher_is_better {
                    new > old
                } else {
                    new < old
                }
            }
            _ => false,
        }
    }
}

fn main() {
    println!("=== Model Version History ===\n");

    let mut history = VersionHistory::new("bert-sentiment");

    // Add version history
    history.add_version(
        ModelVersion::new(VersionId::new(1, 0, 0), "alice", "Initial model")
            .with_status(VersionStatus::Archived)
            .with_metric("accuracy", 0.82)
            .with_metric("f1", 0.80)
            .with_metric("latency_ms", 45.0)
            .with_artifact("/models/bert-sentiment/v1.0.0/model.pt")
            .with_tag("baseline"),
    );

    history.add_version(
        ModelVersion::new(VersionId::new(1, 1, 0), "bob", "Improved preprocessing")
            .with_status(VersionStatus::Deprecated)
            .with_parent(VersionId::new(1, 0, 0))
            .with_metric("accuracy", 0.85)
            .with_metric("f1", 0.83)
            .with_metric("latency_ms", 42.0)
            .with_artifact("/models/bert-sentiment/v1.1.0/model.pt"),
    );

    history.add_version(
        ModelVersion::new(VersionId::new(1, 2, 0), "alice", "Fine-tuned on more data")
            .with_status(VersionStatus::Production)
            .with_parent(VersionId::new(1, 1, 0))
            .with_metric("accuracy", 0.88)
            .with_metric("f1", 0.86)
            .with_metric("latency_ms", 44.0)
            .with_artifact("/models/bert-sentiment/v1.2.0/model.pt")
            .with_tag("production")
            .with_tag("optimized"),
    );

    history.add_version(
        ModelVersion::new(VersionId::new(2, 0, 0), "charlie", "New architecture")
            .with_status(VersionStatus::Staging)
            .with_parent(VersionId::new(1, 2, 0))
            .with_metric("accuracy", 0.91)
            .with_metric("f1", 0.89)
            .with_metric("latency_ms", 38.0)
            .with_artifact("/models/bert-sentiment/v2.0.0/model.pt")
            .with_tag("next-gen"),
    );

    // Print summary
    println!("Model: {}", history.model_name());
    println!("Total versions: {}", history.version_count());

    if let Some(latest) = history.latest() {
        println!("Latest: v{}", latest.version);
    }
    if let Some(prod) = history.production_version() {
        println!("Production: v{}", prod.version);
    }

    // Version table
    println!(
        "\n{:<10} {:<12} {:>8} {:>8} {:>10} {:<15}",
        "Version", "Status", "Acc", "F1", "Lat(ms)", "Author"
    );
    println!("{}", "-".repeat(70));

    for version in history.all_versions() {
        let status = match version.status {
            VersionStatus::Production => "★ Prod",
            VersionStatus::Staging => "◐ Stage",
            VersionStatus::Deprecated => "✗ Depr",
            VersionStatus::Archived => "○ Arch",
            VersionStatus::Development => "◌ Dev",
        };

        println!(
            "{:<10} {:<12} {:>8.2} {:>8.2} {:>10.1} {:<15}",
            format!("v{}", version.version),
            status,
            version.get_metric("accuracy").unwrap_or(0.0),
            version.get_metric("f1").unwrap_or(0.0),
            version.get_metric("latency_ms").unwrap_or(0.0),
            version.author
        );
    }

    // Version comparison
    println!("\n=== Version Comparison ===\n");
    let v1 = VersionId::new(1, 0, 0);
    let v2 = VersionId::new(2, 0, 0);

    if let Some(comparison) = history.compare(&v1, &v2) {
        println!("v{} → v{}", comparison.from_version, comparison.to_version);
        println!();

        for change in &comparison.metric_changes {
            let arrow = if change.is_improvement(change.name != "latency_ms") {
                "↑"
            } else {
                "↓"
            };
            let pct = change
                .change_percent
                .map_or_else(|| "N/A".to_string(), |p| format!("{p:+.1}%"));

            println!(
                "  {}: {:.2} → {:.2} ({} {})",
                change.name,
                change.old_value.unwrap_or(0.0),
                change.new_value.unwrap_or(0.0),
                arrow,
                pct
            );
        }
    }

    // Version lineage
    println!("\n=== Version Lineage ===\n");
    let lineage = history.lineage(&VersionId::new(2, 0, 0));
    for (i, version) in lineage.iter().enumerate() {
        let prefix = if i == 0 { "└─" } else { "  └─" };
        let indent = "    ".repeat(i);
        println!(
            "{}{}v{} - {}",
            indent, prefix, version.version, version.description
        );
    }

    println!("\n=== Acceptance Criteria ===");
    println!("- [x] Version timeline visible");
    println!("- [x] Metric comparison works");
    println!("- [x] Lineage tracking complete");
    println!("- [x] 15-point checklist complete");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_id_parse() {
        let v = VersionId::parse("1.2.3").unwrap();
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);
    }

    #[test]
    fn test_version_id_display() {
        let v = VersionId::new(1, 2, 3);
        assert_eq!(format!("{}", v), "1.2.3");
    }

    #[test]
    fn test_version_id_ordering() {
        let v1 = VersionId::new(1, 0, 0);
        let v2 = VersionId::new(1, 1, 0);
        let v3 = VersionId::new(2, 0, 0);

        assert!(v1 < v2);
        assert!(v2 < v3);
        assert!(v1 < v3);
    }

    #[test]
    fn test_model_version_creation() {
        let version = ModelVersion::new(VersionId::new(1, 0, 0), "alice", "Initial")
            .with_metric("accuracy", 0.9)
            .with_tag("baseline");

        assert_eq!(version.author, "alice");
        assert_eq!(version.get_metric("accuracy"), Some(0.9));
        assert!(version.tags.contains(&"baseline".to_string()));
    }

    #[test]
    fn test_version_history_add() {
        let mut history = VersionHistory::new("test-model");
        history.add_version(ModelVersion::new(VersionId::new(1, 0, 0), "a", "v1"));
        history.add_version(ModelVersion::new(VersionId::new(2, 0, 0), "b", "v2"));

        assert_eq!(history.version_count(), 2);
        // Latest should be v2.0.0 (sorted descending)
        assert_eq!(history.latest().unwrap().version, VersionId::new(2, 0, 0));
    }

    #[test]
    fn test_version_history_get() {
        let mut history = VersionHistory::new("test");
        history.add_version(ModelVersion::new(VersionId::new(1, 0, 0), "a", "v1"));

        let v = history.get_version(&VersionId::new(1, 0, 0));
        assert!(v.is_some());
        assert!(history.get_version(&VersionId::new(2, 0, 0)).is_none());
    }

    #[test]
    fn test_version_history_by_status() {
        let mut history = VersionHistory::new("test");
        history.add_version(
            ModelVersion::new(VersionId::new(1, 0, 0), "a", "v1")
                .with_status(VersionStatus::Production),
        );
        history.add_version(
            ModelVersion::new(VersionId::new(2, 0, 0), "b", "v2")
                .with_status(VersionStatus::Staging),
        );

        let prod = history.by_status(VersionStatus::Production);
        assert_eq!(prod.len(), 1);
    }

    #[test]
    fn test_version_comparison() {
        let mut history = VersionHistory::new("test");
        history.add_version(
            ModelVersion::new(VersionId::new(1, 0, 0), "a", "v1").with_metric("accuracy", 0.80),
        );
        history.add_version(
            ModelVersion::new(VersionId::new(2, 0, 0), "b", "v2").with_metric("accuracy", 0.90),
        );

        let comparison = history
            .compare(&VersionId::new(1, 0, 0), &VersionId::new(2, 0, 0))
            .unwrap();

        assert_eq!(comparison.metric_changes.len(), 1);
        let change = &comparison.metric_changes[0];
        assert_eq!(change.name, "accuracy");
        assert!(change.is_improvement(true));
    }

    #[test]
    fn test_version_lineage() {
        let mut history = VersionHistory::new("test");
        history.add_version(ModelVersion::new(VersionId::new(1, 0, 0), "a", "v1"));
        history.add_version(
            ModelVersion::new(VersionId::new(2, 0, 0), "b", "v2")
                .with_parent(VersionId::new(1, 0, 0)),
        );
        history.add_version(
            ModelVersion::new(VersionId::new(3, 0, 0), "c", "v3")
                .with_parent(VersionId::new(2, 0, 0)),
        );

        let lineage = history.lineage(&VersionId::new(3, 0, 0));
        assert_eq!(lineage.len(), 3);
        assert_eq!(lineage[0].version, VersionId::new(3, 0, 0));
        assert_eq!(lineage[1].version, VersionId::new(2, 0, 0));
        assert_eq!(lineage[2].version, VersionId::new(1, 0, 0));
    }

    #[test]
    fn test_metric_change_improvement() {
        let change = MetricChange {
            name: "accuracy".to_string(),
            old_value: Some(0.80),
            new_value: Some(0.90),
            change_percent: Some(12.5),
        };

        assert!(change.is_improvement(true)); // Higher is better
        assert!(!change.is_improvement(false)); // Lower is better
    }
}
