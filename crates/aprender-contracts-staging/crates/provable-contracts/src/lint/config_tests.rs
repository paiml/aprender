use super::*;

#[test]
fn parse_empty_config() {
    let config = toml_parse("");
    assert!(config.lint.min_score.is_none());
    assert!(!config.lint.strict);
}

#[test]
fn parse_lint_section() {
    let toml = r#"
[lint]
min_score = 0.60
severity = "warning"
strict = true
contracts_dir = "contracts/"
binding = "contracts/aprender/binding.yaml"
"#;
    let config = toml_parse(toml);
    assert_eq!(config.lint.min_score, Some(0.60));
    assert_eq!(config.lint.severity.as_deref(), Some("warning"));
    assert!(config.lint.strict);
    assert_eq!(config.lint.contracts_dir.as_deref(), Some("contracts/"));
    assert!(config.lint.binding.as_deref().unwrap().contains("aprender"));
}

#[test]
fn parse_rules_section() {
    let toml = r#"
[lint.rules]
PV-VAL-001 = "error"
PV-AUD-001 = "info"
PV-SCR-001 = "warning"
"#;
    let config = toml_parse(toml);
    assert_eq!(config.lint.rules.get("PV-VAL-001").unwrap(), "error");
    assert_eq!(config.lint.rules.get("PV-AUD-001").unwrap(), "info");
}

#[test]
fn parse_suppress_section() {
    let toml = r#"
[lint.suppress]
findings = ["SM-INV-001", "KANI-SM-002"]
rules = ["PV-AUD-002"]
files = ["contracts/arch-constraints-v1.yaml"]
"#;
    let config = toml_parse(toml);
    assert_eq!(config.lint.suppress.findings.len(), 2);
    assert_eq!(config.lint.suppress.rules, vec!["PV-AUD-002"]);
    assert_eq!(config.lint.suppress.files.len(), 1);
}

#[test]
fn parse_output_section() {
    let toml = r#"
[output]
format = "sarif"
color = "auto"
"#;
    let config = toml_parse(toml);
    assert_eq!(config.output.format.as_deref(), Some("sarif"));
    assert_eq!(config.output.color.as_deref(), Some("auto"));
}

#[test]
fn parse_comments_and_blank_lines() {
    let toml = r"
# This is a comment

[lint]
# Another comment
min_score = 0.75

";
    let config = toml_parse(toml);
    assert_eq!(config.lint.min_score, Some(0.75));
}

#[test]
fn resolve_defaults() {
    let config = PvConfig::default();
    let map = resolve_rule_severities(&config, &[], false);
    assert_eq!(map.get("PV-VAL-001"), Some(&RuleSeverity::Error));
    assert_eq!(map.get("PV-AUD-001"), Some(&RuleSeverity::Warning));
    assert_eq!(map.get("PV-AUD-002"), Some(&RuleSeverity::Info));
}

#[test]
fn resolve_config_override() {
    let mut config = PvConfig::default();
    config
        .lint
        .rules
        .insert("PV-AUD-001".into(), "error".into());
    let map = resolve_rule_severities(&config, &[], false);
    assert_eq!(map.get("PV-AUD-001"), Some(&RuleSeverity::Error));
}

#[test]
fn resolve_cli_override_wins() {
    let mut config = PvConfig::default();
    config
        .lint
        .rules
        .insert("PV-AUD-001".into(), "error".into());
    let cli = vec![("PV-AUD-001".into(), "off".into())];
    let map = resolve_rule_severities(&config, &cli, false);
    assert_eq!(map.get("PV-AUD-001"), Some(&RuleSeverity::Off));
}

#[test]
fn resolve_strict_mode() {
    let config = PvConfig::default();
    let map = resolve_rule_severities(&config, &[], true);
    // Warnings should become errors in strict mode
    assert_eq!(map.get("PV-AUD-001"), Some(&RuleSeverity::Error));
    // Errors stay errors
    assert_eq!(map.get("PV-VAL-001"), Some(&RuleSeverity::Error));
    // Info stays info
    assert_eq!(map.get("PV-AUD-002"), Some(&RuleSeverity::Info));
}

#[test]
fn find_config_nonexistent() {
    let tmp = tempfile::tempdir().unwrap();
    assert!(find_config(tmp.path()).is_none());
}

#[test]
fn find_config_local() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join(".pv.toml"), "[lint]\nmin_score = 0.5\n").unwrap();
    let found = find_config(tmp.path());
    assert!(found.is_some());
}

#[test]
fn load_config_file() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join(".pv.toml");
    std::fs::write(
        &path,
        "[lint]\nmin_score = 0.75\nstrict = true\n\n[lint.rules]\nPV-VAL-001 = \"error\"\n",
    )
    .unwrap();
    let config = load_config(&path).unwrap();
    assert_eq!(config.lint.min_score, Some(0.75));
    assert!(config.lint.strict);
}

#[test]
fn load_config_missing_file() {
    let result = load_config(Path::new("/nonexistent/.pv.toml"));
    assert!(result.is_err());
}

#[test]
fn trend_defaults() {
    let t = TrendSection::default();
    assert!(t.enabled);
    assert_eq!(t.retention_days, 90);
    assert!((t.drift_threshold - 0.05).abs() < f64::EPSILON);
}

#[test]
fn cache_defaults() {
    let c = CacheSection::default();
    assert!(c.enabled);
    assert!(c.dir.is_none());
}

#[test]
fn config_serializes() {
    let config = PvConfig::default();
    let json = serde_json::to_string(&config).unwrap();
    assert!(json.contains("\"lint\""));
    assert!(json.contains("\"output\""));
}

#[test]
fn parse_diff_section() {
    let toml = r#"
[lint.diff]
base_ref = "main"
"#;
    let config = toml_parse(toml);
    assert_eq!(config.lint.diff.base_ref.as_deref(), Some("main"));
}

#[test]
fn parse_trend_section() {
    let toml = r"
[lint.trend]
enabled = true
retention_days = 30
drift_threshold = 0.10
";
    let config = toml_parse(toml);
    assert!(config.lint.trend.enabled);
    assert_eq!(config.lint.trend.retention_days, 30);
    assert!((config.lint.trend.drift_threshold - 0.10).abs() < f64::EPSILON);
}

#[test]
fn parse_cache_section() {
    let toml = r#"
[lint.cache]
enabled = false
dir = "/tmp/cache"
"#;
    let config = toml_parse(toml);
    assert!(!config.lint.cache.enabled);
    assert_eq!(config.lint.cache.dir.as_deref(), Some("/tmp/cache"));
}

#[test]
fn parse_unknown_section_ignored() {
    let toml = r#"
[unknown]
key = "value"

[lint]
min_score = 0.5
"#;
    let config = toml_parse(toml);
    assert_eq!(config.lint.min_score, Some(0.5));
}

#[test]
fn parse_unknown_keys_in_sections_ignored() {
    let toml = r#"
[lint]
unknown_key = "whatever"
min_score = 0.5

[lint.suppress]
unknown_suppress = "nope"

[lint.trend]
unknown_trend = "nope"

[lint.cache]
unknown_cache = "nope"

[output]
unknown_output = "nope"
format = "json"
"#;
    let config = toml_parse(toml);
    assert_eq!(config.lint.min_score, Some(0.5));
    assert_eq!(config.output.format.as_deref(), Some("json"));
}

#[test]
fn find_config_not_in_tempdir() {
    // tempdir has no git repo root and no HOME config
    let tmp = tempfile::tempdir().unwrap();
    assert!(find_config(tmp.path()).is_none());
}

#[test]
fn default_functions_return_expected_values() {
    assert!(default_true());
    assert_eq!(default_retention_days(), 90);
    assert!((default_drift_threshold() - 0.05).abs() < f64::EPSILON);
}

#[test]
fn resolve_strict_from_config() {
    let mut config = PvConfig::default();
    config.lint.strict = true;
    let map = resolve_rule_severities(&config, &[], false);
    // Warnings should become errors in strict mode (from config)
    assert_eq!(map.get("PV-AUD-001"), Some(&RuleSeverity::Error));
    // Info stays info
    assert_eq!(map.get("PV-AUD-002"), Some(&RuleSeverity::Info));
}
