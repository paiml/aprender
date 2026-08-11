//! Shared vacuity guard for the observation-file lint family.
//!
//! Every `apr *-lint` command reads a captured JSON observation and dispatches
//! a set of pure classifiers over it. Each classifier is optional: a section
//! that is not present in the observation is skipped. That is the right
//! behaviour for an *absent* section and the wrong behaviour for every other
//! reason a classifier can fail to run.
//!
//! Two failure modes fell through the gap and made these commands
//! unfalsifiable when wired into CI:
//!
//! 1. **Nothing was recognised.** `{}`, `{"typo": {...}}` or a JSON scalar
//!    engages no classifier at all, so no failure reason is produced and the
//!    command exits 0 having checked nothing.
//! 2. **A section is present but unusable.** `{"range": {"p": "1.7"}}` has the
//!    field the classifier wants; it is merely the wrong JSON type, so the
//!    `?`-chain that reads it yields `None` and the gate is silently dropped —
//!    including gates whose input carries a *real, detectable* violation.
//!
//! A lint that asserts nothing must not report success. [`assert_not_vacuous`]
//! turns both cases into a hard error, and [`skipped_label`] stops the report
//! from describing a wrong-typed field as "missing".

use serde_json::Value;

/// What a gate actually established about the observation.
///
/// `Vacuous` is deliberately distinct from `Pass`. The `[PASS]`/`[FAIL]`
/// binary is what let 0.63.0 print `[PASS] offline … expected_count_ok=false`
/// and `"passed": true` for a gate that compared nothing. All three verdicts
/// are reported; only `Pass` exits 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Verdict {
    Pass,
    Fail,
    Vacuous,
}

impl Verdict {
    /// Report tag. `VACUOUS` is spelled out so an operator scanning CI logs
    /// cannot mistake it for a pass.
    pub(crate) fn tag(self) -> &'static str {
        match self {
            Verdict::Pass => "PASS",
            Verdict::Fail => "FAIL",
            Verdict::Vacuous => "VACUOUS",
        }
    }

    /// Classify a gate result. A reason opening with `VACUOUS` means the gate
    /// had nothing to check, as opposed to checking something and rejecting it.
    pub(crate) fn of(result: &Result<String, String>) -> Verdict {
        match result {
            Ok(_) => Verdict::Pass,
            Err(msg) if msg.starts_with("VACUOUS") => Verdict::Vacuous,
            Err(_) => Verdict::Fail,
        }
    }
}

/// Report tag for a gate whose outcome string is already rendered.
///
/// Lets a command that stores only `passed: bool` still distinguish the third
/// verdict, by recognising the `VACUOUS` prefix its own gate wrote.
pub(crate) fn verdict_tag(passed: bool, outcome: &str) -> &'static str {
    if passed {
        Verdict::Pass.tag()
    } else if outcome.starts_with("VACUOUS") {
        Verdict::Vacuous.tag()
    } else {
        Verdict::Fail.tag()
    }
}

/// JSON type name, for schema-error messages.
pub(crate) fn json_type(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// One classifier's participation in a run.
pub(crate) struct SectionRun {
    /// Gate name as printed in the report (e.g. `"masking"`).
    pub name: &'static str,
    /// Top-level observation keys the classifier reads. The section counts as
    /// *engaged* when the observation carries any of them.
    pub keys: &'static [&'static str],
    /// Whether the classifier actually produced an outcome.
    pub ran: bool,
}

/// True when the observation carries at least one of `keys` (non-null).
pub(crate) fn any_key_present(obs: &Value, keys: &[&str]) -> bool {
    keys.iter()
        .any(|k| !matches!(obs.get(k), None | Some(Value::Null)))
}

/// Report label for a classifier that produced no outcome.
///
/// Distinguishes "you did not supply this section" from "you supplied it and
/// it is unusable" — the shipped binary printed the former for both, so a
/// wrong-typed field read as an intentional omission.
pub(crate) fn skipped_label(obs: &Value, keys: &[&str]) -> &'static str {
    if any_key_present(obs, keys) {
        "(PRESENT BUT UNUSABLE — missing or wrong-typed fields; NOT checked)"
    } else {
        "(section absent — not checked)"
    }
}

/// Fail a run in which no gate could reach a verdict.
///
/// `Ok(())` only when at least one section ran and every *engaged* section ran.
/// The returned string is the operator-facing reason.
pub(crate) fn assert_not_vacuous(
    falsify_prefix: &str,
    obs: &Value,
    sections: &[SectionRun],
) -> Result<(), String> {
    let all_names = sections
        .iter()
        .map(|s| s.name)
        .collect::<Vec<_>>()
        .join("/");

    if !obs.is_object() {
        return Err(format!(
            "{falsify_prefix}: observation is not a JSON object, so none of {all_names} could be \
             read — nothing was checked"
        ));
    }

    let unusable: Vec<&str> = sections
        .iter()
        .filter(|s| !s.ran && any_key_present(obs, s.keys))
        .map(|s| s.name)
        .collect();
    if !unusable.is_empty() {
        return Err(format!(
            "{falsify_prefix}: section(s) {} are present but unusable (a required field is \
             missing or has the wrong JSON type), so those gates did not run — a present-but-\
             malformed section is a schema error, not a skip",
            unusable.join(", ")
        ));
    }

    if !sections.iter().any(|s| s.ran) {
        return Err(format!(
            "{falsify_prefix}: observation has none of {all_names} — no gate ran, so nothing was \
             checked (a lint that asserts nothing must not pass)"
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const SECTIONS: &[&str] = &["alpha", "beta"];

    fn sections(alpha_ran: bool, beta_ran: bool) -> Vec<SectionRun> {
        vec![
            SectionRun {
                name: "alpha",
                keys: &["alpha"],
                ran: alpha_ran,
            },
            SectionRun {
                name: "beta",
                keys: &["beta"],
                ran: beta_ran,
            },
        ]
    }

    #[test]
    fn falsifier_empty_observation_is_vacuous() {
        let err = assert_not_vacuous("X", &json!({}), &sections(false, false)).unwrap_err();
        assert!(err.contains("has none of alpha/beta"), "{err}");
        assert!(err.contains("nothing was checked"), "{err}");
    }

    #[test]
    fn falsifier_unrecognised_keys_are_vacuous() {
        let err =
            assert_not_vacuous("X", &json!({"typo": 1}), &sections(false, false)).unwrap_err();
        assert!(err.contains("has none of alpha/beta"), "{err}");
    }

    #[test]
    fn falsifier_scalar_observation_is_vacuous() {
        let err = assert_not_vacuous("X", &json!(42), &sections(false, false)).unwrap_err();
        assert!(err.contains("not a JSON object"), "{err}");
    }

    #[test]
    fn falsifier_present_but_unrunnable_section_is_an_error() {
        // `alpha` is supplied; its classifier could not run. That is a schema
        // error, not an omission.
        let err = assert_not_vacuous(
            "X",
            &json!({"alpha": {"p": "1.7"}}),
            &sections(false, false),
        )
        .unwrap_err();
        assert!(err.contains("present but unusable"), "{err}");
        assert!(err.contains("alpha"), "{err}");
    }

    #[test]
    fn falsifier_unrunnable_section_fails_even_when_a_sibling_ran() {
        let err = assert_not_vacuous(
            "X",
            &json!({"alpha": {"p": "1.7"}, "beta": {}}),
            &sections(false, true),
        )
        .unwrap_err();
        assert!(err.contains("present but unusable"), "{err}");
    }

    #[test]
    fn control_one_section_that_ran_is_not_vacuous() {
        assert!(assert_not_vacuous("X", &json!({"beta": {}}), &sections(false, true)).is_ok());
    }

    #[test]
    fn skipped_label_distinguishes_absent_from_unusable() {
        assert!(skipped_label(&json!({}), &["alpha"]).contains("absent"));
        assert!(skipped_label(&json!({"alpha": "x"}), &["alpha"]).contains("UNUSABLE"));
        // A null value reads as "not supplied", not as a malformed section.
        assert!(skipped_label(&json!({"alpha": null}), &["alpha"]).contains("absent"));
    }

    #[test]
    fn any_key_present_is_true_when_any_alias_is_supplied() {
        assert!(any_key_present(&json!({"beta": 1}), SECTIONS));
        assert!(!any_key_present(&json!({"gamma": 1}), SECTIONS));
    }
}
