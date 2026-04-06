//! Lint rule catalog — every `pv lint` rule has an ID, default severity, and description.
//!
//! Rule ID format: `PV-<CATEGORY>-NNN`
//! Categories: VAL (validation), AUD (audit), SCR (score), PRV (provability), TRD (trend)
//!
//! Spec: `docs/specifications/sub/lint.md` Section 12

use serde::{Deserialize, Serialize};

/// A lint rule definition.
#[derive(Debug, Clone, Serialize)]
pub struct LintRule {
    pub id: &'static str,
    pub category: RuleCategory,
    pub default_severity: RuleSeverity,
    pub description: &'static str,
    /// Estimated remediation effort in minutes.
    pub effort_minutes: u32,
}

/// Rule categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum RuleCategory {
    Validate,
    Audit,
    Score,
    Provability,
    Trend,
    Suppression,
    Enforcement,
}

/// Configurable severity levels per rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum RuleSeverity {
    Off,
    Info,
    Warning,
    Error,
}

impl RuleSeverity {
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s {
            "off" => Some(Self::Off),
            "info" => Some(Self::Info),
            "warning" | "warn" => Some(Self::Warning),
            "error" => Some(Self::Error),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Info => "info",
            Self::Warning => "warning",
            Self::Error => "error",
        }
    }

    pub fn sarif_level(self) -> &'static str {
        match self {
            Self::Off => "none",
            Self::Info => "note",
            Self::Warning => "warning",
            Self::Error => "error",
        }
    }
}

/// All lint rules in the catalog.
pub static RULES: &[LintRule] = &[
    // Validation rules (5 min each — fix YAML syntax)
    LintRule {
        id: "PV-VAL-001",
        category: RuleCategory::Validate,
        default_severity: RuleSeverity::Error,
        description: "Schema validation error (parse failure or missing section)",
        effort_minutes: 5,
    },
    LintRule {
        id: "PV-VAL-002",
        category: RuleCategory::Validate,
        default_severity: RuleSeverity::Error,
        description: "Missing metadata.version",
        effort_minutes: 5,
    },
    LintRule {
        id: "PV-VAL-003",
        category: RuleCategory::Validate,
        default_severity: RuleSeverity::Warning,
        description: "Missing metadata.created",
        effort_minutes: 5,
    },
    LintRule {
        id: "PV-VAL-004",
        category: RuleCategory::Validate,
        default_severity: RuleSeverity::Error,
        description: "Empty equation formula",
        effort_minutes: 5,
    },
    LintRule {
        id: "PV-VAL-005",
        category: RuleCategory::Validate,
        default_severity: RuleSeverity::Error,
        description: "Empty proof obligation property",
        effort_minutes: 5,
    },
    LintRule {
        id: "PV-VAL-006",
        category: RuleCategory::Validate,
        default_severity: RuleSeverity::Warning,
        description: "Duplicate formal predicate in proof obligations",
        effort_minutes: 5,
    },
    // Audit rules (15 min each — add test/reference/domain)
    LintRule {
        id: "PV-AUD-001",
        category: RuleCategory::Audit,
        default_severity: RuleSeverity::Warning,
        description: "Obligation without falsification test",
        effort_minutes: 15,
    },
    LintRule {
        id: "PV-AUD-002",
        category: RuleCategory::Audit,
        default_severity: RuleSeverity::Info,
        description: "Missing paper reference",
        effort_minutes: 15,
    },
    LintRule {
        id: "PV-AUD-003",
        category: RuleCategory::Audit,
        default_severity: RuleSeverity::Warning,
        description: "Obligation ID not referenced by any test",
        effort_minutes: 15,
    },
    LintRule {
        id: "PV-AUD-004",
        category: RuleCategory::Audit,
        default_severity: RuleSeverity::Warning,
        description: "Equation without domain specification",
        effort_minutes: 15,
    },
    LintRule {
        id: "PV-AUD-005",
        category: RuleCategory::Audit,
        default_severity: RuleSeverity::Warning,
        description: "Missing tolerance for numerical obligation",
        effort_minutes: 15,
    },
    // Score rules (30 min each — improve score dimension)
    LintRule {
        id: "PV-SCR-001",
        category: RuleCategory::Score,
        default_severity: RuleSeverity::Error,
        description: "Composite score below threshold",
        effort_minutes: 30,
    },
    LintRule {
        id: "PV-SCR-002",
        category: RuleCategory::Score,
        default_severity: RuleSeverity::Warning,
        description: "Missing binding entry for equation",
        effort_minutes: 30,
    },
    LintRule {
        id: "PV-SCR-003",
        category: RuleCategory::Score,
        default_severity: RuleSeverity::Info,
        description: "Kani coverage below 50%",
        effort_minutes: 30,
    },
    LintRule {
        id: "PV-SCR-004",
        category: RuleCategory::Score,
        default_severity: RuleSeverity::Info,
        description: "Lean coverage at 0%",
        effort_minutes: 30,
    },
    // Provability rules (60 min each — add Kani harness or falsification test)
    LintRule {
        id: "PV-PRV-001",
        category: RuleCategory::Provability,
        default_severity: RuleSeverity::Error,
        description: "Kernel contract without Kani harnesses",
        effort_minutes: 60,
    },
    LintRule {
        id: "PV-PRV-002",
        category: RuleCategory::Provability,
        default_severity: RuleSeverity::Error,
        description: "Kernel contract without falsification tests",
        effort_minutes: 60,
    },
    LintRule {
        id: "PV-PRV-003",
        category: RuleCategory::Provability,
        default_severity: RuleSeverity::Warning,
        description: "Kani harness count < obligation count",
        effort_minutes: 60,
    },
    LintRule {
        id: "PV-PRV-004",
        category: RuleCategory::Provability,
        default_severity: RuleSeverity::Info,
        description: "No Lean theorems (L5 not attempted)",
        effort_minutes: 60,
    },
    // Trend rules (10 min each — review trend)
    LintRule {
        id: "PV-TRD-001",
        category: RuleCategory::Trend,
        default_severity: RuleSeverity::Warning,
        description: "Mean score dropped >5% from 7-day rolling average",
        effort_minutes: 10,
    },
    LintRule {
        id: "PV-TRD-002",
        category: RuleCategory::Trend,
        default_severity: RuleSeverity::Info,
        description: "Error count increased from previous snapshot",
        effort_minutes: 10,
    },
    // Suppression rules (2 min each — remove stale suppression)
    LintRule {
        id: "PV-SUP-001",
        category: RuleCategory::Suppression,
        default_severity: RuleSeverity::Warning,
        description: "Stale suppression — the finding no longer fires",
        effort_minutes: 2,
    },
    // Enforcement rules (Section 17, Gaps 1 + 5)
    LintRule {
        id: "PV-ENF-001",
        category: RuleCategory::Enforcement,
        default_severity: RuleSeverity::Warning,
        description: "Equation without preconditions",
        effort_minutes: 20,
    },
    LintRule {
        id: "PV-ENF-002",
        category: RuleCategory::Enforcement,
        default_severity: RuleSeverity::Warning,
        description: "Equation without lean_theorem",
        effort_minutes: 20,
    },
    LintRule {
        id: "PV-LCK-001",
        category: RuleCategory::Enforcement,
        default_severity: RuleSeverity::Error,
        description: "Contract regressed below locked verification level",
        effort_minutes: 5,
    },
];

/// Look up a rule by ID.
pub fn find_rule(id: &str) -> Option<&'static LintRule> {
    RULES.iter().find(|r| r.id == id)
}

/// Get all rules for a category.
pub fn rules_for_category(cat: RuleCategory) -> Vec<&'static LintRule> {
    RULES.iter().filter(|r| r.category == cat).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rule_catalog_not_empty() {
        assert!(!RULES.is_empty());
        assert!(RULES.len() >= 21, "Expected 21+ rules, got {}", RULES.len());
    }

    #[test]
    fn all_rule_ids_unique() {
        let mut seen = std::collections::HashSet::new();
        for rule in RULES {
            assert!(seen.insert(rule.id), "Duplicate rule ID: {}", rule.id);
        }
    }

    #[test]
    fn find_rule_by_id() {
        let rule = find_rule("PV-VAL-001").unwrap();
        assert_eq!(rule.category, RuleCategory::Validate);
        assert_eq!(rule.default_severity, RuleSeverity::Error);
    }

    #[test]
    fn find_rule_missing() {
        assert!(find_rule("PV-FAKE-999").is_none());
    }

    #[test]
    fn rules_for_category_validate() {
        let vals = rules_for_category(RuleCategory::Validate);
        assert!(vals.len() >= 6);
        assert!(vals.iter().all(|r| r.category == RuleCategory::Validate));
    }

    #[test]
    fn severity_from_str() {
        assert_eq!(
            RuleSeverity::from_str_opt("error"),
            Some(RuleSeverity::Error)
        );
        assert_eq!(
            RuleSeverity::from_str_opt("warning"),
            Some(RuleSeverity::Warning)
        );
        assert_eq!(
            RuleSeverity::from_str_opt("warn"),
            Some(RuleSeverity::Warning)
        );
        assert_eq!(RuleSeverity::from_str_opt("info"), Some(RuleSeverity::Info));
        assert_eq!(RuleSeverity::from_str_opt("off"), Some(RuleSeverity::Off));
        assert_eq!(RuleSeverity::from_str_opt("bogus"), None);
    }

    #[test]
    fn severity_as_str_roundtrip() {
        for sev in [
            RuleSeverity::Off,
            RuleSeverity::Info,
            RuleSeverity::Warning,
            RuleSeverity::Error,
        ] {
            let s = sev.as_str();
            assert_eq!(RuleSeverity::from_str_opt(s), Some(sev));
        }
    }

    #[test]
    fn severity_sarif_level() {
        assert_eq!(RuleSeverity::Error.sarif_level(), "error");
        assert_eq!(RuleSeverity::Warning.sarif_level(), "warning");
        assert_eq!(RuleSeverity::Info.sarif_level(), "note");
        assert_eq!(RuleSeverity::Off.sarif_level(), "none");
    }

    #[test]
    fn severity_ordering() {
        assert!(RuleSeverity::Off < RuleSeverity::Info);
        assert!(RuleSeverity::Info < RuleSeverity::Warning);
        assert!(RuleSeverity::Warning < RuleSeverity::Error);
    }

    #[test]
    fn rule_serializes() {
        let json = serde_json::to_string(&RULES[0]).unwrap();
        assert!(json.contains("PV-VAL-001"));
        assert!(json.contains("validate"));
    }
}
