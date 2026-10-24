use super::*;

const H: &str = "roadmap_version: '1.0'\nroadmap:\n";

fn run(base: &str, head: &str) -> (i32, String) {
    let (rc, lines) = run_check(&format!("{H}{base}"), &format!("{H}{head}")).unwrap();
    (rc, join_lines(&lines))
}

const E1: &str = "- id: A-1\n  title: one\n  status: planned\n";
const E2: &str = "- id: A-2\n  title: two\n  status: planned\n";

#[test]
fn identical_files_pass() {
    assert_eq!(run(E1, E1).0, 0);
}

#[test]
fn pure_addition_passes() {
    assert_eq!(run(E1, &format!("{E1}{E2}")).0, 0);
}

#[test]
fn deletion_is_a_violation() {
    let (rc, out) = run(&format!("{E1}{E2}"), E1);
    assert_eq!(rc, 1);
    assert!(out.contains("A-2"), "{out}");
}

#[test]
fn editing_a_core_field_is_a_violation() {
    let edited = "- id: A-1\n  title: CHANGED\n  status: planned\n";
    let (rc, out) = run(E1, edited);
    assert_eq!(rc, 1);
    assert!(out.contains("A-1") && out.contains("title"), "{out}");
}

#[test]
fn lifecycle_change_is_allowed() {
    let done = "- id: A-1\n  title: one\n  status: completed\n";
    assert_eq!(run(E1, done).0, 0);
}

#[test]
fn reserialisation_without_a_field_change_is_flagged_not_silent() {
    let reflowed = "- id: A-1\n  status: planned\n  title: one\n";
    let (rc, out) = run(E1, reflowed);
    assert_eq!(rc, 1, "a re-serialisation must be a violation: {out}");
    assert!(out.contains("A-1"), "{out}");
}

#[test]
fn unparsable_base_is_a_usage_error() {
    assert!(run_check("a: [\n", &format!("{H}{E1}")).is_err());
}

#[test]
fn trim_restores_base_bytes_for_a_reserialised_entry() {
    let base = format!("{H}{E1}{E2}");
    let reflowed = format!("{H}- id: A-1\n  status: planned\n  title: one\n{E2}");
    assert_eq!(render_trim(&base, &reflowed), base);
}

#[test]
fn trim_keeps_new_entries_and_restores_deleted_ones() {
    let base = format!("{H}{E1}{E2}");
    let head = format!("{H}{E1}- id: A-3\n  title: three\n  status: planned\n");
    let t = render_trim(&base, &head);
    assert!(t.contains("A-2") && t.contains("A-3"), "{t}");
}

#[test]
fn join_lines_terminates_each_line() {
    assert_eq!(join_lines(&["a".into(), "b".into()]), "a\nb\n");
}
