//! Per-field form accessibility rules (WCAG 1.3.1, 1.3.5, 2.4.6, 3.3.1, 3.3.2),
//! driven through the public `FormA11yChecker::check` entry point (#5551).

use presentar_test::{
    AutocompleteValue, FormA11yChecker, FormA11yRule, FormAccessibility, FormFieldA11y,
    FormViolation, Impact, InputType,
};

fn violations(field: FormFieldA11y) -> Vec<FormViolation> {
    let form = FormAccessibility::new().with_name("f").field(field);
    FormA11yChecker::check(&form).violations
}

fn rules(field: FormFieldA11y) -> Vec<FormA11yRule> {
    violations(field).iter().map(|v| v.rule).collect()
}

#[test]
fn labelled_plain_field_has_no_violations() {
    assert!(rules(FormFieldA11y::new("n").with_label("Name")).is_empty());
}

#[test]
fn missing_label_is_critical_and_names_the_field() {
    let v = violations(FormFieldA11y::new("zip"));
    assert_eq!(v.len(), 1);
    assert_eq!(v[0].rule, FormA11yRule::MissingLabel);
    assert_eq!(v[0].field_id, "zip");
    assert_eq!(v[0].impact, Impact::Critical);
    assert_eq!(v[0].wcag, "1.3.1, 2.4.6");
    assert!(v[0].message.contains("'zip'"));
}

#[test]
fn any_one_label_source_satisfies_the_label_rule() {
    assert!(rules(FormFieldA11y::new("a").with_aria_label("A")).is_empty());
    let mut f = FormFieldA11y::new("b");
    f.aria_labelledby = Some("lbl".into());
    assert!(rules(f).is_empty());
}

#[test]
fn required_without_indicators_reports_both() {
    let f = FormFieldA11y::new("r")
        .with_label("R")
        .with_required(false, false);
    assert_eq!(
        rules(f),
        vec![
            FormA11yRule::MissingRequiredIndicator,
            FormA11yRule::MissingVisualRequired
        ]
    );
}

#[test]
fn required_reports_only_the_missing_indicator() {
    let base = || FormFieldA11y::new("r").with_label("R");
    assert_eq!(
        rules(base().with_required(true, false)),
        vec![FormA11yRule::MissingRequiredIndicator]
    );
    assert_eq!(
        rules(base().with_required(false, true)),
        vec![FormA11yRule::MissingVisualRequired]
    );
    assert!(rules(base().with_required(true, true)).is_empty());
}

#[test]
fn required_violation_impacts_and_criterion() {
    let v = violations(
        FormFieldA11y::new("r")
            .with_label("R")
            .with_required(false, false),
    );
    assert_eq!(v[0].impact, Impact::Serious);
    assert_eq!(v[1].impact, Impact::Moderate);
    assert!(v.iter().all(|x| x.wcag == "3.3.2"));
}

#[test]
fn error_state_branches_are_independent() {
    let mut f = FormFieldA11y::new("e").with_label("E");
    f.has_error = true;
    assert_eq!(
        rules(f.clone()),
        vec![
            FormA11yRule::MissingErrorState,
            FormA11yRule::MissingErrorMessage,
            FormA11yRule::ErrorNotAssociated
        ]
    );
    f.aria_invalid = true;
    assert_eq!(
        rules(f.clone()),
        vec![
            FormA11yRule::MissingErrorMessage,
            FormA11yRule::ErrorNotAssociated
        ]
    );
    f.error_message = Some("bad".into());
    assert_eq!(rules(f.clone()), vec![FormA11yRule::ErrorNotAssociated]);
    f.aria_errormessage = Some("x".into());
    assert!(rules(f).is_empty());
}

#[test]
fn describedby_also_associates_the_error() {
    let f = FormFieldA11y::new("e")
        .with_label("E")
        .with_error("bad", true);
    assert!(rules(f).is_empty());
    let f = FormFieldA11y::new("e")
        .with_label("E")
        .with_error("bad", false);
    assert_eq!(rules(f), vec![FormA11yRule::ErrorNotAssociated]);
}

#[test]
fn autocomplete_required_only_for_personal_data_types() {
    for t in [
        InputType::Text,
        InputType::Email,
        InputType::Password,
        InputType::Tel,
        InputType::Url,
        InputType::Number,
    ] {
        let f = FormFieldA11y::new("a").with_label("A").with_type(t);
        assert_eq!(rules(f), vec![FormA11yRule::MissingAutocomplete], "{t:?}");
    }
    for t in [
        InputType::Date,
        InputType::Radio,
        InputType::Checkbox,
        InputType::Hidden,
    ] {
        let f = FormFieldA11y::new("a").with_label("A").with_type(t);
        assert!(rules(f).is_empty(), "{t:?}");
    }
}

#[test]
fn autocomplete_present_clears_the_violation() {
    let f = FormFieldA11y::new("a")
        .with_label("A")
        .with_type(InputType::Email)
        .with_autocomplete(AutocompleteValue::Email);
    assert!(rules(f).is_empty());
}

#[test]
fn placeholder_only_labeling_is_flagged() {
    assert_eq!(
        rules(FormFieldA11y::new("p").with_placeholder("Name")),
        vec![FormA11yRule::MissingLabel, FormA11yRule::PlaceholderAsLabel]
    );
}

#[test]
fn placeholder_with_label_or_aria_label_is_fine() {
    let f = FormFieldA11y::new("p")
        .with_label("N")
        .with_placeholder("x");
    assert!(rules(f).is_empty());
    let f = FormFieldA11y::new("p")
        .with_aria_label("N")
        .with_placeholder("x");
    assert!(rules(f).is_empty());
}

#[test]
fn check_attributes_violations_to_the_right_field() {
    let form = FormAccessibility::new()
        .with_name("f")
        .field(FormFieldA11y::new("ok").with_label("OK"))
        .field(FormFieldA11y::new("bad"));
    let report = FormA11yChecker::check(&form);
    assert!(!report.violations_for_field("bad").is_empty());
    assert!(report.violations_for_field("ok").is_empty());
    assert!(!report.is_acceptable());
}
