//! Stack Drift Detection
//!
//! Detects when PAIML stack crates are using outdated versions of other
//! stack crates. This ensures stack coherence and prevents version drift.
//!
//! ## Toyota Way Principles
//!
//! - **Jidoka**: Blocks operations when drift detected
//! - **Genchi Genbutsu**: Real-time crates.io dependency verification
//! - **Kaizen**: Continuous improvement through enforced updates

use super::crates_io::{CratesIoClient, DependencyData};
use super::{is_paiml_crate, PAIML_CRATES};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Severity of version drift
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DriftSeverity {
    /// Major version difference (e.g., 0.10 vs 0.11) - likely breaking
    Major,
    /// Minor version difference within same major (e.g., 0.10.1 vs 0.10.5)
    Minor,
    /// Patch version difference - negligible
    Patch,
}

impl DriftSeverity {
    /// Get display string for severity
    pub fn as_str(&self) -> &'static str {
        match self {
            DriftSeverity::Major => "MAJOR",
            DriftSeverity::Minor => "MINOR",
            DriftSeverity::Patch => "PATCH",
        }
    }
}

impl std::fmt::Display for DriftSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A single drift issue detected in the stack
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftReport {
    /// The crate that has the outdated dependency
    pub crate_name: String,
    /// Version of the crate with the drift
    pub crate_version: String,
    /// The dependency that is behind
    pub dependency: String,
    /// Version requirement the crate uses
    pub uses_version: String,
    /// Latest version available on crates.io
    pub latest_version: String,
    /// Severity of the drift
    pub severity: DriftSeverity,
}

impl DriftReport {
    /// Format for display
    pub fn display(&self) -> String {
        format!(
            "{} {}: {} {} → {} ({})",
            self.crate_name,
            self.crate_version,
            self.dependency,
            self.uses_version,
            self.latest_version,
            self.severity
        )
    }
}

/// Drift checker for the PAIML stack
pub struct DriftChecker {
    /// Latest versions of PAIML crates (cached)
    latest_versions: HashMap<String, semver::Version>,
}

impl DriftChecker {
    /// Create a new drift checker
    pub fn new() -> Self {
        Self { latest_versions: HashMap::new() }
    }

    /// Fetch latest versions of all PAIML crates
    #[cfg(feature = "native")]
    pub async fn fetch_latest_versions(&mut self, client: &mut CratesIoClient) -> Result<()> {
        for crate_name in PAIML_CRATES {
            match client.get_latest_version(crate_name).await {
                Ok(version) => {
                    self.latest_versions.insert((*crate_name).to_string(), version);
                }
                Err(_) => {
                    // Crate not published yet, skip
                    continue;
                }
            }
        }
        Ok(())
    }

    /// Detect drift for batuta's own dependencies only.
    ///
    /// Checks whether the published batuta crate uses the latest versions
    /// of other PAIML stack crates. This is the startup check — it only
    /// reports on batuta itself, not the entire ecosystem.
    #[cfg(feature = "native")]
    pub async fn detect_self_drift(
        &mut self,
        client: &mut CratesIoClient,
    ) -> Result<Vec<DriftReport>> {
        if self.latest_versions.is_empty() {
            self.fetch_latest_versions(client).await?;
        }

        let mut drifts = Vec::new();

        let crate_version = match self.latest_versions.get("batuta") {
            Some(v) => v.to_string(),
            None => return Ok(drifts),
        };

        let deps = match client.get_dependencies("batuta", &crate_version).await {
            Ok(d) => d,
            Err(_) => return Ok(drifts),
        };

        self.check_deps_for_drift("batuta", &crate_version, &deps, &mut drifts);
        Self::sort_drifts(&mut drifts);
        Ok(drifts)
    }

    /// Detect drift across the entire PAIML stack (maintainer tool).
    ///
    /// For each published PAIML crate, checks if its dependencies on other
    /// PAIML crates are using the latest versions.
    #[cfg(feature = "native")]
    pub async fn detect_drift(&mut self, client: &mut CratesIoClient) -> Result<Vec<DriftReport>> {
        // First, ensure we have latest versions
        if self.latest_versions.is_empty() {
            self.fetch_latest_versions(client).await?;
        }

        let mut drifts = Vec::new();

        // Check each published PAIML crate
        for crate_name in PAIML_CRATES {
            let crate_version = match self.latest_versions.get(*crate_name) {
                Some(v) => v.to_string(),
                None => continue, // Not published
            };

            // Get dependencies for this crate version
            let deps = match client.get_dependencies(crate_name, &crate_version).await {
                Ok(d) => d,
                Err(_) => continue, // Skip if can't fetch deps
            };

            self.check_deps_for_drift(crate_name, &crate_version, &deps, &mut drifts);
        }

        Self::sort_drifts(&mut drifts);
        Ok(drifts)
    }

    /// Check dependencies of a single crate for drift against latest versions
    fn check_deps_for_drift(
        &self,
        crate_name: &str,
        crate_version: &str,
        deps: &[DependencyData],
        drifts: &mut Vec<DriftReport>,
    ) {
        for dep in deps {
            // Only check PAIML dependencies
            if !is_paiml_crate(&dep.crate_id) {
                continue;
            }

            // Skip dev dependencies
            if dep.kind == "dev" {
                continue;
            }

            // Get latest version of this dependency
            let latest = match self.latest_versions.get(&dep.crate_id) {
                Some(v) => v,
                None => continue, // Dependency not published
            };

            // Check if behind
            if let Some(drift) = self.check_drift(crate_name, crate_version, dep, latest) {
                drifts.push(drift);
            }
        }
    }

    /// Sort drift reports by severity (major first) then alphabetically
    fn sort_drifts(drifts: &mut [DriftReport]) {
        drifts.sort_by(|a, b| match (&a.severity, &b.severity) {
            (DriftSeverity::Major, DriftSeverity::Major) => a.crate_name.cmp(&b.crate_name),
            (DriftSeverity::Major, _) => std::cmp::Ordering::Less,
            (_, DriftSeverity::Major) => std::cmp::Ordering::Greater,
            (DriftSeverity::Minor, DriftSeverity::Minor) => a.crate_name.cmp(&b.crate_name),
            (DriftSeverity::Minor, _) => std::cmp::Ordering::Less,
            (_, DriftSeverity::Minor) => std::cmp::Ordering::Greater,
            _ => a.crate_name.cmp(&b.crate_name),
        });
    }

    /// Check if a dependency version is behind the latest
    fn check_drift(
        &self,
        crate_name: &str,
        crate_version: &str,
        dep: &DependencyData,
        latest: &semver::Version,
    ) -> Option<DriftReport> {
        // Parse the version requirement to extract the base version
        let uses_version = &dep.version_req;

        // Try to parse a semver from the requirement
        // Handle common patterns: "0.11", "0.11.0", "^0.11", "~0.11"
        let version_str = uses_version
            .trim_start_matches('^')
            .trim_start_matches('~')
            .trim_start_matches('=')
            .trim_start_matches('>')
            .trim_start_matches('<')
            .trim();

        // Parse as version or partial version
        let (uses_major, uses_minor) = Self::parse_version_parts(version_str);

        // Compare with latest
        let severity = if uses_major < latest.major as u32 {
            // Major version behind (rare for 0.x crates)
            Some(DriftSeverity::Major)
        } else if uses_major == latest.major as u32 && uses_minor < latest.minor as u32 {
            // Minor version behind within same major
            Some(DriftSeverity::Major) // For 0.x, minor is effectively major
        } else {
            // Up to date or ahead
            None
        };

        severity.map(|sev| DriftReport {
            crate_name: crate_name.to_string(),
            crate_version: crate_version.to_string(),
            dependency: dep.crate_id.clone(),
            uses_version: uses_version.clone(),
            latest_version: latest.to_string(),
            severity: sev,
        })
    }

    /// Parse version string into (major, minor) parts
    fn parse_version_parts(version_str: &str) -> (u32, u32) {
        let parts: Vec<&str> = version_str.split('.').collect();
        let major = parts.first().and_then(|s| s.parse().ok()).unwrap_or(0);
        let minor = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
        (major, minor)
    }

    /// Get cached latest versions
    pub fn latest_versions(&self) -> &HashMap<String, semver::Version> {
        &self.latest_versions
    }
}

impl Default for DriftChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Format drift reports for display (blocking error style)
pub fn format_drift_errors(drifts: &[DriftReport]) -> String {
    if drifts.is_empty() {
        return String::new();
    }

    let mut output = String::new();
    output.push_str("🔴 Stack Drift Detected - Cannot Proceed\n\n");

    for drift in drifts {
        output.push_str(&format!("   {}\n", drift.display()));
    }

    output.push_str("\nStack drift detected. Fix dependencies before proceeding.\n");
    output.push_str("Run: batuta stack drift --fix\n");
    output.push_str("Or use --allow-drift to bypass (for local development).\n");

    output
}

/// Format drift reports as JSON
pub fn format_drift_json(drifts: &[DriftReport]) -> Result<String> {
    Ok(serde_json::to_string_pretty(drifts)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dep(crate_id: &str, version_req: &str, kind: &str) -> DependencyData {
        DependencyData {
            crate_id: crate_id.to_string(),
            version_req: version_req.to_string(),
            kind: kind.to_string(),
            optional: false,
        }
    }

    fn make_report(severity: DriftSeverity) -> DriftReport {
        DriftReport {
            crate_name: "trueno-rag".to_string(),
            crate_version: "0.1.5".to_string(),
            dependency: "trueno".to_string(),
            uses_version: "0.10.1".to_string(),
            latest_version: "0.11.0".to_string(),
            severity,
        }
    }

    // ===== DriftSeverity =====

    #[test]
    fn test_drift_severity_as_str() {
        assert_eq!(DriftSeverity::Major.as_str(), "MAJOR");
        assert_eq!(DriftSeverity::Minor.as_str(), "MINOR");
        assert_eq!(DriftSeverity::Patch.as_str(), "PATCH");
    }

    #[test]
    fn test_drift_severity_display_trait() {
        assert_eq!(format!("{}", DriftSeverity::Major), "MAJOR");
        assert_eq!(format!("{}", DriftSeverity::Minor), "MINOR");
        assert_eq!(format!("{}", DriftSeverity::Patch), "PATCH");
    }

    #[test]
    fn test_drift_severity_clone_eq() {
        let a = DriftSeverity::Major;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn test_drift_severity_debug() {
        let dbg = format!("{:?}", DriftSeverity::Patch);
        assert_eq!(dbg, "Patch");
    }

    #[test]
    fn test_drift_severity_serde_roundtrip() {
        let json = serde_json::to_string(&DriftSeverity::Minor).expect("json serialize failed");
        let back: DriftSeverity = serde_json::from_str(&json).expect("json deserialize failed");
        assert_eq!(back, DriftSeverity::Minor);
    }

    // ===== DriftReport =====

    #[test]
    fn test_drift_report_display() {
        let report = make_report(DriftSeverity::Major);
        let display = report.display();
        assert!(display.contains("trueno-rag"));
        assert!(display.contains("0.1.5"));
        assert!(display.contains("trueno"));
        assert!(display.contains("0.10.1"));
        assert!(display.contains("0.11.0"));
        assert!(display.contains("MAJOR"));
    }

    #[test]
    fn test_drift_report_serde_roundtrip() {
        let report = make_report(DriftSeverity::Major);
        let json = serde_json::to_string(&report).expect("json serialize failed");
        let back: DriftReport = serde_json::from_str(&json).expect("json deserialize failed");
        assert_eq!(back.crate_name, "trueno-rag");
        assert_eq!(back.severity, DriftSeverity::Major);
    }

    #[test]
    fn test_drift_report_clone() {
        let report = make_report(DriftSeverity::Minor);
        let cloned = report.clone();
        assert_eq!(cloned.crate_name, report.crate_name);
        assert_eq!(cloned.severity, report.severity);
    }

    // ===== DriftChecker =====

    #[test]
    fn test_drift_checker_new() {
        let checker = DriftChecker::new();
        assert!(checker.latest_versions.is_empty());
    }

    #[test]
    fn test_drift_checker_default() {
        let checker = DriftChecker::default();
        assert!(checker.latest_versions().is_empty());
    }

    #[test]
    fn test_drift_checker_latest_versions_accessor() {
        let mut checker = DriftChecker::new();
        checker.latest_versions.insert("trueno".to_string(), semver::Version::new(0, 14, 0));
        assert_eq!(checker.latest_versions().len(), 1);
        assert_eq!(checker.latest_versions()["trueno"], semver::Version::new(0, 14, 0));
    }

    // ===== parse_version_parts =====

    #[test]
    fn test_parse_version_parts_full() {
        assert_eq!(DriftChecker::parse_version_parts("0.11.0"), (0, 11));
    }

    #[test]
    fn test_parse_version_parts_two() {
        assert_eq!(DriftChecker::parse_version_parts("0.11"), (0, 11));
    }

    #[test]
    fn test_parse_version_parts_three() {
        assert_eq!(DriftChecker::parse_version_parts("1.2.3"), (1, 2));
    }

    #[test]
    fn test_parse_version_parts_single() {
        assert_eq!(DriftChecker::parse_version_parts("2"), (2, 0));
    }

    #[test]
    fn test_parse_version_parts_empty() {
        assert_eq!(DriftChecker::parse_version_parts(""), (0, 0));
    }

    #[test]
    fn test_parse_version_parts_garbage() {
        assert_eq!(DriftChecker::parse_version_parts("abc.def"), (0, 0));
    }

    // ===== check_drift =====

    #[test]
    fn test_check_drift_behind_minor() {
        let mut checker = DriftChecker::new();
        checker.latest_versions.insert("trueno".to_string(), semver::Version::new(0, 14, 0));

        let dep = make_dep("trueno", "^0.11", "normal");
        let latest = &semver::Version::new(0, 14, 0);
        let result = checker.check_drift("aprender", "0.24.0", &dep, latest);

        assert!(result.is_some());
        let report = result.expect("operation failed");
        assert_eq!(report.dependency, "trueno");
        assert_eq!(report.severity, DriftSeverity::Major);
    }

    #[test]
    fn test_check_drift_up_to_date() {
        let checker = DriftChecker::new();
        let dep = make_dep("trueno", "^0.14", "normal");
        let latest = &semver::Version::new(0, 14, 0);
        let result = checker.check_drift("aprender", "0.24.0", &dep, latest);
        assert!(result.is_none());
    }

    #[test]
    fn test_check_drift_ahead() {
        let checker = DriftChecker::new();
        let dep = make_dep("trueno", "0.15", "normal");
        let latest = &semver::Version::new(0, 14, 0);
        let result = checker.check_drift("aprender", "0.24.0", &dep, latest);
        assert!(result.is_none());
    }

    #[test]
    fn test_check_drift_major_behind() {
        let checker = DriftChecker::new();
        let dep = make_dep("repartir", "1.0", "normal");
        let latest = &semver::Version::new(2, 0, 0);
        let result = checker.check_drift("batuta", "0.6.0", &dep, latest);

        assert!(result.is_some());
        let report = result.expect("operation failed");
        assert_eq!(report.severity, DriftSeverity::Major);
    }

    #[test]
    fn test_check_drift_strips_caret() {
        let checker = DriftChecker::new();
        let dep = make_dep("trueno", "^0.11.0", "normal");
        let latest = &semver::Version::new(0, 14, 0);
        let result = checker.check_drift("test", "1.0.0", &dep, latest);
        assert!(result.is_some());
    }

    #[test]
    fn test_check_drift_strips_tilde() {
        let checker = DriftChecker::new();
        let dep = make_dep("trueno", "~0.11", "normal");
        let latest = &semver::Version::new(0, 14, 0);
        let result = checker.check_drift("test", "1.0.0", &dep, latest);
        assert!(result.is_some());
    }

    #[test]
    fn test_check_drift_strips_eq() {
        let checker = DriftChecker::new();
        let dep = make_dep("trueno", "=0.11.0", "normal");
        let latest = &semver::Version::new(0, 14, 0);
        let result = checker.check_drift("test", "1.0.0", &dep, latest);
        assert!(result.is_some());
    }

    #[test]
    fn test_check_drift_strips_gt() {
        let checker = DriftChecker::new();
        let dep = make_dep("trueno", ">0.11", "normal");
        let latest = &semver::Version::new(0, 14, 0);
        let result = checker.check_drift("test", "1.0.0", &dep, latest);
        assert!(result.is_some());
    }

    // ===== format_drift_errors =====

    #[test]
    fn test_format_drift_errors_empty() {
        let output = format_drift_errors(&[]);
        assert!(output.is_empty());
    }

    #[test]
    fn test_format_drift_errors_with_drifts() {
        let drifts = vec![make_report(DriftSeverity::Major)];
        let output = format_drift_errors(&drifts);
        assert!(output.contains("Stack Drift Detected"));
        assert!(output.contains("trueno-rag"));
        assert!(output.contains("batuta stack drift --fix"));
        assert!(output.contains("--allow-drift"));
    }

    #[test]
    fn test_format_drift_errors_multiple() {
        let drifts = vec![
            make_report(DriftSeverity::Major),
            DriftReport {
                crate_name: "aprender".to_string(),
                crate_version: "0.24.0".to_string(),
                dependency: "trueno".to_string(),
                uses_version: "0.12".to_string(),
                latest_version: "0.14.0".to_string(),
                severity: DriftSeverity::Major,
            },
        ];
        let output = format_drift_errors(&drifts);
        assert!(output.contains("trueno-rag"));
        assert!(output.contains("aprender"));
    }

    // ===== format_drift_json =====

    #[test]
    fn test_format_drift_json_empty() {
        let json = format_drift_json(&[]).expect("unexpected failure");
        assert_eq!(json, "[]");
    }

    #[test]
    fn test_format_drift_json_single() {
        let drifts = vec![make_report(DriftSeverity::Major)];
        let json = format_drift_json(&drifts).expect("unexpected failure");
        assert!(json.contains("trueno-rag"));
        assert!(json.contains("\"Major\""));
    }

    #[test]
    fn test_format_drift_json_roundtrip() {
        let drifts = vec![
            make_report(DriftSeverity::Major),
            DriftReport {
                crate_name: "aprender".to_string(),
                crate_version: "0.24.0".to_string(),
                dependency: "trueno".to_string(),
                uses_version: "0.12".to_string(),
                latest_version: "0.14.0".to_string(),
                severity: DriftSeverity::Minor,
            },
        ];
        let json = format_drift_json(&drifts).expect("unexpected failure");
        let back: Vec<DriftReport> = serde_json::from_str(&json).expect("json deserialize failed");
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].crate_name, "trueno-rag");
        assert_eq!(back[1].severity, DriftSeverity::Minor);
    }

    // ===== Drift sorting (exercises the comparator from detect_drift) =====

    fn make_drift(name: &str, severity: DriftSeverity) -> DriftReport {
        DriftReport {
            crate_name: name.to_string(),
            crate_version: "0.1.0".to_string(),
            dependency: "trueno".to_string(),
            uses_version: "0.10".to_string(),
            latest_version: "0.14.0".to_string(),
            severity,
        }
    }

    #[test]
    fn test_drift_sort_major_first() {
        let mut drifts = vec![
            make_drift("zeta", DriftSeverity::Minor),
            make_drift("alpha", DriftSeverity::Major),
        ];
        DriftChecker::sort_drifts(&mut drifts);
        assert_eq!(drifts[0].crate_name, "alpha");
        assert_eq!(drifts[0].severity, DriftSeverity::Major);
    }

    #[test]
    fn test_drift_sort_major_alpha() {
        let mut drifts = vec![
            make_drift("beta", DriftSeverity::Major),
            make_drift("alpha", DriftSeverity::Major),
        ];
        DriftChecker::sort_drifts(&mut drifts);
        assert_eq!(drifts[0].crate_name, "alpha");
        assert_eq!(drifts[1].crate_name, "beta");
    }

    #[test]
    fn test_drift_sort_minor_alpha() {
        let mut drifts = vec![
            make_drift("beta", DriftSeverity::Minor),
            make_drift("alpha", DriftSeverity::Minor),
        ];
        DriftChecker::sort_drifts(&mut drifts);
        assert_eq!(drifts[0].crate_name, "alpha");
    }

    #[test]
    fn test_drift_sort_patch_alpha() {
        let mut drifts = vec![
            make_drift("beta", DriftSeverity::Patch),
            make_drift("alpha", DriftSeverity::Patch),
        ];
        DriftChecker::sort_drifts(&mut drifts);
        assert_eq!(drifts[0].crate_name, "alpha");
    }

    #[test]
    fn test_drift_sort_minor_before_patch() {
        let mut drifts = vec![
            make_drift("alpha", DriftSeverity::Patch),
            make_drift("beta", DriftSeverity::Minor),
        ];
        DriftChecker::sort_drifts(&mut drifts);
        assert_eq!(drifts[0].severity, DriftSeverity::Minor);
    }

    #[test]
    fn test_drift_sort_all_severities() {
        let mut drifts = vec![
            make_drift("alpha", DriftSeverity::Patch),
            make_drift("beta", DriftSeverity::Minor),
            make_drift("gamma", DriftSeverity::Major),
        ];
        DriftChecker::sort_drifts(&mut drifts);
        assert_eq!(drifts[0].severity, DriftSeverity::Major);
        assert_eq!(drifts[1].severity, DriftSeverity::Minor);
        assert_eq!(drifts[2].severity, DriftSeverity::Patch);
    }

    // ===== check_deps_for_drift =====

    #[test]
    fn test_check_deps_for_drift_with_paiml_dep() {
        let mut checker = DriftChecker::new();
        checker.latest_versions.insert("trueno".to_string(), semver::Version::new(0, 14, 0));

        let deps = vec![make_dep("trueno", "^0.11", "normal")];
        let mut drifts = Vec::new();
        checker.check_deps_for_drift("aprender", "0.24.0", &deps, &mut drifts);
        assert_eq!(drifts.len(), 1);
        assert_eq!(drifts[0].dependency, "trueno");
    }

    #[test]
    fn test_check_deps_for_drift_skips_non_paiml() {
        let mut checker = DriftChecker::new();
        checker.latest_versions.insert("trueno".to_string(), semver::Version::new(0, 14, 0));

        let deps = vec![make_dep("serde", "1.0", "normal")];
        let mut drifts = Vec::new();
        checker.check_deps_for_drift("aprender", "0.24.0", &deps, &mut drifts);
        assert!(drifts.is_empty());
    }

    #[test]
    fn test_check_deps_for_drift_skips_dev_deps() {
        let mut checker = DriftChecker::new();
        checker.latest_versions.insert("trueno".to_string(), semver::Version::new(0, 14, 0));

        let deps = vec![make_dep("trueno", "^0.11", "dev")];
        let mut drifts = Vec::new();
        checker.check_deps_for_drift("aprender", "0.24.0", &deps, &mut drifts);
        assert!(drifts.is_empty());
    }

    #[test]
    fn test_check_deps_for_drift_skips_unpublished() {
        let checker = DriftChecker::new(); // no versions cached
        let deps = vec![make_dep("trueno", "^0.11", "normal")];
        let mut drifts = Vec::new();
        checker.check_deps_for_drift("aprender", "0.24.0", &deps, &mut drifts);
        assert!(drifts.is_empty());
    }

    #[test]
    fn test_check_deps_for_drift_up_to_date() {
        let mut checker = DriftChecker::new();
        checker.latest_versions.insert("trueno".to_string(), semver::Version::new(0, 14, 0));

        let deps = vec![make_dep("trueno", "^0.14", "normal")];
        let mut drifts = Vec::new();
        checker.check_deps_for_drift("aprender", "0.24.0", &deps, &mut drifts);
        assert!(drifts.is_empty());
    }

    #[test]
    fn test_check_deps_for_drift_mixed() {
        let mut checker = DriftChecker::new();
        checker.latest_versions.insert("trueno".to_string(), semver::Version::new(0, 14, 0));
        checker.latest_versions.insert("aprender".to_string(), semver::Version::new(0, 25, 0));

        let deps = vec![
            make_dep("trueno", "^0.11", "normal"),   // behind
            make_dep("serde", "1.0", "normal"),      // non-paiml, skip
            make_dep("aprender", "^0.25", "normal"), // up to date
            make_dep("trueno", "^0.12", "dev"),      // dev dep, skip
        ];
        let mut drifts = Vec::new();
        checker.check_deps_for_drift("realizar", "0.6.0", &deps, &mut drifts);
        assert_eq!(drifts.len(), 1);
        assert_eq!(drifts[0].dependency, "trueno");
    }

    // ===== check_drift edge cases =====

    #[test]
    fn test_check_drift_equal_major_different_minor() {
        let checker = DriftChecker::new();
        // same major (0), uses minor 14, latest minor 14 → up to date
        let dep = make_dep("trueno", "0.14.0", "normal");
        let latest = &semver::Version::new(0, 14, 5);
        assert!(checker.check_drift("test", "1.0.0", &dep, latest).is_none());
    }

    #[test]
    fn test_check_drift_report_fields() {
        let checker = DriftChecker::new();
        let dep = make_dep("trueno", "^0.11", "normal");
        let latest = &semver::Version::new(0, 14, 0);
        let report =
            checker.check_drift("aprender", "0.24.0", &dep, latest).expect("unexpected failure");
        assert_eq!(report.crate_name, "aprender");
        assert_eq!(report.crate_version, "0.24.0");
        assert_eq!(report.dependency, "trueno");
        assert_eq!(report.uses_version, "^0.11");
        assert_eq!(report.latest_version, "0.14.0");
    }

    #[test]
    fn test_check_drift_strips_lt() {
        let checker = DriftChecker::new();
        let dep = make_dep("trueno", "<0.11", "normal");
        let latest = &semver::Version::new(0, 14, 0);
        assert!(checker.check_drift("test", "1.0.0", &dep, latest).is_some());
    }

    // ===== async detect_drift with offline client =====

    #[cfg(feature = "native")]
    #[tokio::test]
    async fn test_detect_self_drift_offline_client() {
        let mut client = CratesIoClient::new();
        client.set_offline(true);

        let mut checker = DriftChecker::new();
        // Pre-populate so fetch_latest_versions is skipped
        checker.latest_versions.insert("batuta".to_string(), semver::Version::new(0, 7, 2));
        checker.latest_versions.insert("trueno".to_string(), semver::Version::new(0, 16, 0));

        // Offline client can't get_dependencies → empty result
        let drifts = checker.detect_self_drift(&mut client).await.expect("async operation failed");
        assert!(drifts.is_empty());
    }

    #[cfg(feature = "native")]
    #[tokio::test]
    async fn test_detect_self_drift_no_batuta_published() {
        let mut client = CratesIoClient::new();
        client.set_offline(true);

        let mut checker = DriftChecker::new();
        // Only trueno published, not batuta → should return empty
        checker.latest_versions.insert("trueno".to_string(), semver::Version::new(0, 16, 0));

        let drifts = checker.detect_self_drift(&mut client).await.expect("async operation failed");
        assert!(drifts.is_empty());
    }

    /// Falsification: if batuta's deps are current, self-drift must return empty.
    #[test]
    fn test_detect_self_drift_all_current() {
        let mut checker = DriftChecker::new();
        checker.latest_versions.insert("batuta".to_string(), semver::Version::new(0, 7, 2));
        checker.latest_versions.insert("trueno".to_string(), semver::Version::new(0, 16, 0));
        checker.latest_versions.insert("aprender".to_string(), semver::Version::new(0, 27, 0));

        // Simulate batuta depending on current versions
        let deps = vec![
            make_dep("trueno", "^0.16", "normal"),
            make_dep("aprender", "^0.27", "normal"),
            make_dep("serde", "1.0", "normal"), // non-PAIML, ignored
        ];
        let mut drifts = Vec::new();
        checker.check_deps_for_drift("batuta", "0.7.2", &deps, &mut drifts);
        assert!(drifts.is_empty(), "No drift expected when deps are current");
    }

    /// Falsification: self-drift must never produce reports for non-batuta crates.
    #[test]
    fn test_detect_self_drift_never_reports_other_crates() {
        let mut checker = DriftChecker::new();
        checker.latest_versions.insert("batuta".to_string(), semver::Version::new(0, 7, 2));
        checker.latest_versions.insert("trueno".to_string(), semver::Version::new(0, 16, 0));

        // Even if we fabricate deps for batuta, all reports must have crate_name == "batuta"
        let deps = vec![make_dep("trueno", "^0.11", "normal")]; // stale
        let mut drifts = Vec::new();
        checker.check_deps_for_drift("batuta", "0.7.2", &deps, &mut drifts);

        for d in &drifts {
            assert_eq!(
                d.crate_name, "batuta",
                "Self-drift must only report batuta, got: {}",
                d.crate_name
            );
        }
        assert_eq!(drifts.len(), 1);
    }

    #[cfg(feature = "native")]
    #[tokio::test]
    async fn test_detect_drift_offline_client() {
        let mut client = CratesIoClient::new();
        client.set_offline(true);

        let mut checker = DriftChecker::new();
        // Pre-populate so fetch_latest_versions is skipped
        checker.latest_versions.insert("trueno".to_string(), semver::Version::new(0, 14, 0));
        checker.latest_versions.insert("aprender".to_string(), semver::Version::new(0, 25, 0));

        // Offline client can't get_dependencies → all crates skip → empty result
        let drifts = checker.detect_drift(&mut client).await.expect("async operation failed");
        assert!(drifts.is_empty());
    }

    #[cfg(feature = "native")]
    #[tokio::test]
    async fn test_detect_drift_no_versions_cached() {
        let mut client = CratesIoClient::new();
        client.set_offline(true);

        let mut checker = DriftChecker::new();
        // latest_versions is empty → fetch_latest_versions called → offline errors → remains empty
        // Then detect_drift loops but no crate has a version → all skip
        let drifts = checker.detect_drift(&mut client).await.expect("async operation failed");
        assert!(drifts.is_empty());
    }

    #[cfg(feature = "native")]
    #[tokio::test]
    async fn test_fetch_latest_versions_offline() {
        let mut client = CratesIoClient::new();
        client.set_offline(true);

        let mut checker = DriftChecker::new();
        // Should not error, just skip crates that can't be fetched
        checker.fetch_latest_versions(&mut client).await.expect("async operation failed");
        // All fetches fail in offline mode → no versions cached
        assert!(checker.latest_versions.is_empty());
    }
}
