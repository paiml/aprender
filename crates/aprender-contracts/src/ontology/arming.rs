//! ONT-001 §3.9 — per-repo arming (ONT-6, PMAT-3451).
//!
//! `armed_gates` in `contracts/lint-baseline.json` names the gates that enter a repo's meet. Every gate is
//! computed and printed everywhere; an unarmed gate prints `Unknown{NotArmed}` and is excluded. The list is
//! monotone against a committed comparand: dropping a gate is `ArmedGatesShrank` (exit 3). An explicitly
//! empty list is a decline, never an accept (R-2). An absent file or an absent key means the default set.

use std::fmt;

use super::verdict::{Reason, Verdict};

/// The default armed set (operator ruling 2026-09-17, verbatim: "8: drop reverse-coverage (Recommended)").
/// `reverse-coverage` is Skipped unless `--binding` and `--crate-dir` are both given; `strict-test-binding` is
/// opt-in; `shapes` is appended by ONT-4b's own change.
pub const DEFAULT_ARMED: [&str; 8] = [
    "validate",
    "audit",
    "score",
    "verify",
    "enforce",
    "enforcement-level",
    "duplicate-stems",
    "composition",
];

/// The gates that enter this repo's meet, in declared order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArmedGates {
    names: Vec<String>,
}

impl ArmedGates {
    /// Exactly these names, in this order.
    #[must_use]
    pub fn new(names: Vec<String>) -> Self {
        Self { names }
    }

    /// [`DEFAULT_ARMED`].
    #[must_use]
    pub fn default_set() -> Self {
        // RED (PMAT-3451): deliberately wrong until GREEN.
        Self { names: Vec::new() }
    }

    #[must_use]
    pub fn names(&self) -> &[String] {
        &self.names
    }

    #[must_use]
    pub fn is_armed(&self, gate: &str) -> bool {
        // RED (PMAT-3451): deliberately wrong until GREEN.
        let _ = gate;
        false
    }

    /// Parse `contracts/lint-baseline.json`. `None` (no file) or no `armed_gates` key → the default set; an
    /// array of strings → exactly that list (possibly empty); anything else → [`BaselineError`].
    pub fn from_baseline(text: Option<&str>) -> Result<Self, BaselineError> {
        // RED (PMAT-3451): deliberately wrong until GREEN.
        let _ = text;
        Ok(Self::default_set())
    }
}

/// `contracts/lint-baseline.json` could not be read as an arming declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaselineError(pub String);

impl fmt::Display for BaselineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "lint-baseline.json: {}", self.0)
    }
}

impl std::error::Error for BaselineError {}

/// Gates present in the comparand and absent now. Exit 3, `error: armed_gates shrank: <names>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArmedGatesShrank {
    pub dropped: Vec<String>,
}

impl fmt::Display for ArmedGatesShrank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "armed_gates shrank: {}", self.dropped.join(", "))
    }
}

impl std::error::Error for ArmedGatesShrank {}

/// `current` may add or reorder gates; it may not drop one the comparand armed.
pub fn check_monotone(
    comparand: &ArmedGates,
    current: &ArmedGates,
) -> Result<(), ArmedGatesShrank> {
    // RED (PMAT-3451): deliberately wrong until GREEN.
    let _ = (comparand, current);
    Ok(())
}

/// The armed meet over one run's gate verdicts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArmedMeet {
    /// The meet over the armed gates (Pass when every armed gate passed).
    pub verdict: Verdict,
    /// Each armed gate, in armed order, with the verdict it contributed (`Unknown{NotRun}` if it did not run).
    pub armed: Vec<(String, Verdict)>,
    /// Gates that ran but are not armed, in run order; each prints `Unknown{NotArmed}` and is excluded.
    pub not_armed: Vec<String>,
}

/// Meet over the armed gates. An armed gate that did not run is `Unknown{NotRun}` in the meet; an empty
/// armed set is `Unknown{NotArmed}` (R-2: zero is a decline, never an accept).
#[must_use]
pub fn meet_armed(results: &[(String, Verdict)], armed: &ArmedGates) -> ArmedMeet {
    // RED (PMAT-3451): deliberately wrong until GREEN.
    let _ = (results, armed, Reason::NotArmed);
    ArmedMeet {
        verdict: Verdict::Pass,
        armed: Vec::new(),
        not_armed: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gates(names: &[&str]) -> ArmedGates {
        ArmedGates::new(names.iter().map(|s| (*s).to_string()).collect())
    }

    fn run(pairs: &[(&str, Verdict)]) -> Vec<(String, Verdict)> {
        pairs.iter().map(|(n, v)| ((*n).to_string(), *v)).collect()
    }

    fn all_default_pass() -> Vec<(String, Verdict)> {
        DEFAULT_ARMED
            .iter()
            .map(|n| ((*n).to_string(), Verdict::Pass))
            .collect()
    }

    #[test]
    fn default_set_is_the_eight_ruled_gates() {
        let d = ArmedGates::default_set();
        assert_eq!(d.names(), &DEFAULT_ARMED.map(String::from));
        for g in DEFAULT_ARMED {
            assert!(d.is_armed(g), "{g} armed by default");
        }
        assert!(!d.is_armed("reverse-coverage"), "ruled out");
        assert!(!d.is_armed("strict-test-binding"), "opt-in");
        assert!(!d.is_armed("shapes"), "ONT-4b arms it");
    }

    #[test]
    fn baseline_parsing() {
        assert_eq!(
            ArmedGates::from_baseline(None),
            Ok(ArmedGates::default_set()),
            "no file"
        );
        assert_eq!(
            ArmedGates::from_baseline(Some(r#"{"_spec": "x", "ont": {}}"#)),
            Ok(ArmedGates::default_set()),
            "no armed_gates key"
        );
        assert_eq!(
            ArmedGates::from_baseline(Some(r#"{"armed_gates": ["validate", "audit"]}"#)),
            Ok(gates(&["validate", "audit"]))
        );
        assert_eq!(
            ArmedGates::from_baseline(Some(r#"{"armed_gates": []}"#)),
            Ok(gates(&[])),
            "explicit empty kept"
        );
        assert!(
            ArmedGates::from_baseline(Some(r#"{"armed_gates": "validate"}"#)).is_err(),
            "not an array"
        );
        assert!(
            ArmedGates::from_baseline(Some(r#"{"armed_gates": ["validate", 3]}"#)).is_err(),
            "non-string entry"
        );
        assert!(
            ArmedGates::from_baseline(Some("not json")).is_err(),
            "malformed"
        );
    }

    #[test]
    fn a_dropped_gate_is_a_shrink_naming_it() {
        let current = gates(&[
            "validate",
            "audit",
            "score",
            "verify",
            "enforce",
            "enforcement-level",
            "duplicate-stems",
        ]);
        let err = check_monotone(&ArmedGates::default_set(), &current).unwrap_err();
        assert_eq!(err.dropped, vec!["composition".to_string()]);
        assert_eq!(err.to_string(), "armed_gates shrank: composition");
    }

    #[test]
    fn growth_and_reorder_are_monotone() {
        let d = ArmedGates::default_set();
        assert_eq!(check_monotone(&d, &d), Ok(()));
        let mut grown = DEFAULT_ARMED.to_vec();
        grown.push("shapes");
        assert_eq!(check_monotone(&d, &gates(&grown)), Ok(()));
        let mut reordered = DEFAULT_ARMED.to_vec();
        reordered.reverse();
        assert_eq!(check_monotone(&d, &gates(&reordered)), Ok(()));
    }

    #[test]
    fn meet_over_armed_gates() {
        let d = ArmedGates::default_set();
        let mut r = all_default_pass();
        r.push(("reverse-coverage".into(), Verdict::Unknown(Reason::Skip)));
        let m = meet_armed(&r, &d);
        assert_eq!(
            m.verdict,
            Verdict::Pass,
            "an unarmed Skipped gate is excluded"
        );
        assert_eq!(m.not_armed, vec!["reverse-coverage".to_string()]);
        assert_eq!(m.armed.len(), 8);
        assert_eq!(
            m.armed.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
            DEFAULT_ARMED.to_vec()
        );

        let mut failing = all_default_pass();
        failing[7].1 = Verdict::Fail;
        assert_eq!(
            meet_armed(&failing, &d).verdict,
            Verdict::Fail,
            "armed composition Fail"
        );

        let missing: Vec<_> = all_default_pass()
            .into_iter()
            .filter(|(n, _)| n != "enforce")
            .collect();
        let m = meet_armed(&missing, &d);
        assert_eq!(
            m.verdict,
            Verdict::Unknown(Reason::NotRun),
            "armed gate that did not run"
        );
        assert!(m
            .armed
            .contains(&("enforce".to_string(), Verdict::Unknown(Reason::NotRun))));

        let mut unarmed_fail = all_default_pass();
        unarmed_fail.push(("strict-test-binding".into(), Verdict::Fail));
        assert_eq!(
            meet_armed(&unarmed_fail, &d).verdict,
            Verdict::Pass,
            "unarmed Fail excluded"
        );
    }

    #[test]
    fn empty_armed_set_declines_r2() {
        let m = meet_armed(
            &run(&[("validate", Verdict::Pass), ("audit", Verdict::Pass)]),
            &gates(&[]),
        );
        assert_eq!(
            m.verdict,
            Verdict::Unknown(Reason::NotArmed),
            "zero armed gates is a decline, never an accept"
        );
        assert_eq!(
            m.not_armed,
            vec!["validate".to_string(), "audit".to_string()]
        );
        assert!(m.armed.is_empty());
    }
}
