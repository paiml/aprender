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
        Self {
            names: DEFAULT_ARMED.iter().map(|s| (*s).to_string()).collect(),
        }
    }

    #[must_use]
    pub fn names(&self) -> &[String] {
        &self.names
    }

    #[must_use]
    pub fn is_armed(&self, gate: &str) -> bool {
        self.names.iter().any(|n| n == gate)
    }

    /// Parse `contracts/lint-baseline.json`. `None` (no file) or no `armed_gates` key → the default set; an
    /// array of strings → exactly that list (possibly empty); anything else → [`BaselineError`].
    pub fn from_baseline(text: Option<&str>) -> Result<Self, BaselineError> {
        let Some(text) = text else {
            return Ok(Self::default_set());
        };
        let parsed: serde_json::Value = match serde_json::from_str(text) {
            Ok(v) => v,
            Err(e) => return Err(BaselineError(e.to_string())),
        };
        let Some(armed_gates_val) = parsed.get("armed_gates") else {
            return Ok(Self::default_set());
        };
        let Some(armed_gates_arr) = armed_gates_val.as_array() else {
            return Err(BaselineError("armed_gates is not an array".to_string()));
        };
        let mut names = Vec::new();
        for val in armed_gates_arr {
            if let Some(s) = val.as_str() {
                names.push(s.to_string());
            } else {
                return Err(BaselineError(
                    "armed_gates element is not a string".to_string(),
                ));
            }
        }
        Ok(Self::new(names))
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
    let mut dropped = Vec::new();
    for name in comparand.names() {
        if !current.is_armed(name) {
            dropped.push(name.clone());
        }
    }
    if dropped.is_empty() {
        Ok(())
    } else {
        Err(ArmedGatesShrank { dropped })
    }
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
    if armed.names().is_empty() {
        return ArmedMeet {
            verdict: Verdict::Unknown(Reason::NotArmed),
            armed: Vec::new(),
            not_armed: results.iter().map(|(n, _)| n.clone()).collect(),
        };
    }

    let mut armed_results = Vec::new();
    let mut verdict = Verdict::Pass;

    for name in armed.names() {
        let v = results
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| *v)
            .unwrap_or(Verdict::Unknown(Reason::NotRun));
        armed_results.push((name.clone(), v));
        verdict = verdict.meet(v);
    }

    let not_armed = results
        .iter()
        .filter_map(|(n, _)| {
            if !armed.is_armed(n) {
                Some(n.clone())
            } else {
                None
            }
        })
        .collect();

    ArmedMeet {
        verdict,
        armed: armed_results,
        not_armed,
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

// ── ONT-4c1: arming one level down — per shape ───────────────────────────────────────────────────────────────

/// Which shapes feed the `shapes` gate's verdict (ONT-001 v4.6 §3.9). `All` when `lint-baseline.json` carries no
/// `armed_shapes` key (ONT-4b's behaviour, unchanged); `Listed` when it does — a shape not listed is computed
/// and reported under `not_armed_shapes`, and does not feed the meet. Monotone under [`check_shapes_monotone`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArmedShapes {
    All,
    Listed(Vec<String>),
}

impl ArmedShapes {
    #[must_use]
    pub fn is_armed(&self, shape: &str) -> bool {
        match self {
            Self::All => true,
            Self::Listed(names) => names.iter().any(|n| n == shape),
        }
    }

    /// The declared list, empty for `All` (which arms everything without naming anything).
    #[must_use]
    pub fn names(&self) -> &[String] {
        match self {
            Self::All => &[],
            Self::Listed(names) => names,
        }
    }

    /// Parse `contracts/lint-baseline.json`. No file, or no `armed_shapes` key → `All`; an array of strings →
    /// `Listed` (possibly empty: a repo that arms no shape); anything else → [`BaselineError`].
    pub fn from_baseline(text: Option<&str>) -> Result<Self, BaselineError> {
        let Some(text) = text else {
            return Ok(Self::All);
        };
        let parsed: serde_json::Value =
            serde_json::from_str(text).map_err(|e| BaselineError(e.to_string()))?;
        let Some(val) = parsed.get("armed_shapes") else {
            return Ok(Self::All);
        };
        let Some(arr) = val.as_array() else {
            return Err(BaselineError("armed_shapes is not an array".to_string()));
        };
        let mut names = Vec::new();
        for v in arr {
            match v.as_str() {
                Some(s) => names.push(s.to_string()),
                None => {
                    return Err(BaselineError(
                        "armed_shapes element is not a string".to_string(),
                    ))
                }
            }
        }
        Ok(Self::Listed(names))
    }
}

/// Shapes the comparand armed and the current declaration does not. Exit 3, `error: armed_shapes shrank: <names>`.
/// `All` → `Listed` is a shrink of every shape the comparand's corpus carried that the list omits; that case is
/// judged by name against the shapes the current corpus declares, so the caller passes them in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArmedShapesShrank {
    pub dropped: Vec<String>,
}

impl fmt::Display for ArmedShapesShrank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "armed_shapes shrank: {}", self.dropped.join(", "))
    }
}

impl std::error::Error for ArmedShapesShrank {}

/// `current` may add shapes; it may not drop one the comparand armed. `corpus_shapes` names every shape the
/// corpus declares today, so a comparand of `All` is compared as "every declared shape".
pub fn check_shapes_monotone(
    comparand: &ArmedShapes,
    current: &ArmedShapes,
    corpus_shapes: &[String],
) -> Result<(), ArmedShapesShrank> {
    let committed: Vec<String> = match comparand {
        ArmedShapes::All => corpus_shapes.to_vec(),
        ArmedShapes::Listed(names) => names.clone(),
    };
    let dropped: Vec<String> = committed
        .into_iter()
        .filter(|n| !current.is_armed(n))
        .collect();
    if dropped.is_empty() {
        Ok(())
    } else {
        Err(ArmedShapesShrank { dropped })
    }
}

#[cfg(test)]
mod shape_arming_tests {
    use super::*;

    #[test]
    fn no_key_arms_every_shape_and_a_list_arms_exactly_the_list() {
        let all = ArmedShapes::from_baseline(None).expect("ok");
        assert_eq!(all, ArmedShapes::All);
        assert!(all.is_armed("anything"));
        let none_key =
            ArmedShapes::from_baseline(Some(r#"{"armed_gates":["validate"]}"#)).expect("ok");
        assert_eq!(none_key, ArmedShapes::All);
        let listed = ArmedShapes::from_baseline(Some(
            r#"{"armed_shapes":["ont-shapes-v1","ladder-measured"]}"#,
        ))
        .expect("ok");
        assert!(listed.is_armed("ladder-measured"));
        assert!(!listed.is_armed("ladder-green"));
        assert!(ArmedShapes::from_baseline(Some(r#"{"armed_shapes":"x"}"#)).is_err());
        assert!(ArmedShapes::from_baseline(Some(r#"{"armed_shapes":[1]}"#)).is_err());
    }

    #[test]
    fn dropping_an_armed_shape_is_a_shrink_and_adding_one_is_not() {
        let committed = ArmedShapes::Listed(vec!["a".into(), "b".into()]);
        let grown = ArmedShapes::Listed(vec!["a".into(), "b".into(), "c".into()]);
        let shrunk = ArmedShapes::Listed(vec!["a".into()]);
        let corpus = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        assert!(check_shapes_monotone(&committed, &grown, &corpus).is_ok());
        let e = check_shapes_monotone(&committed, &shrunk, &corpus).expect_err("shrank");
        assert_eq!(e.to_string(), "armed_shapes shrank: b");
        // All → Listed is judged against the corpus: listing fewer than the corpus declares is a shrink
        let e = check_shapes_monotone(&ArmedShapes::All, &shrunk, &corpus).expect_err("shrank");
        assert_eq!(e.dropped, vec!["b".to_string(), "c".to_string()]);
        assert!(check_shapes_monotone(&ArmedShapes::All, &ArmedShapes::All, &corpus).is_ok());
    }
}
