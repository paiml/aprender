//! `pv inert` — classify contracts by whether their claims can be refuted,
//! and ratchet the count that cannot.
//!
//! See `provable_contracts::inert` for the classification rule and why the
//! bare count of zero-falsification contracts (759 of 1726) is NOT the number
//! to ratchet.

use std::path::Path;

use provable_contracts::inert::{classify_tree, self_test, InertReport, Verdict, WALK_FLOOR};

/// Run the `inert` subcommand.
///
/// # Errors
/// Returns `Err` when the case table fails, when the measurement is vacuous
/// (nothing walked), or when `--max` is exceeded. Each is a distinct message
/// so an operator can tell "the guard broke" from "the tree regressed".
pub fn run(
    contract_dir: &Path,
    format: &str,
    max: Option<usize>,
    list: bool,
    run_self_test: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if run_self_test {
        return match self_test() {
            Ok(n) => {
                println!("pv inert --self-test: {n}/{n} classifier cases pass");
                Ok(())
            }
            Err(failures) => Err(format!(
                "pv inert --self-test FAILED ({} case(s)):\n  {}",
                failures.len(),
                failures.join("\n  ")
            )
            .into()),
        };
    }

    let report = classify_tree(contract_dir);

    // VACUOUS guard, before any verdict is printed. A gate that passes on n=0
    // is a fail mode, and this one would otherwise report "0 inert — PASS" for
    // a mistyped directory.
    if report.walked == 0 {
        return Err(format!(
            "VACUOUS: pv inert walked 0 contracts under {} — refusing to report a pass \
             on an empty measurement",
            contract_dir.display()
        )
        .into());
    }

    match format {
        "json" => print_json(&report),
        _ => print_text(&report, list),
    }

    // A file the raw probe could not read is a file whose lost falsification
    // tests are invisible to the count above. That must never pass quietly.
    if !report.probe_failures.is_empty() {
        return Err(format!(
            "pv inert: raw-YAML probe failed on {} file(s); their dropped falsification \
             blocks cannot be seen, so the count above is an undercount:\n  {}",
            report.probe_failures.len(),
            report
                .probe_failures
                .iter()
                .map(|(p, e)| format!("{}: {e}", p.display()))
                .collect::<Vec<_>>()
                .join("\n  ")
        )
        .into());
    }

    if report.walked < WALK_FLOOR && max.is_some() {
        return Err(format!(
            "VACUOUS: --max was given but only {} contracts were walked (floor {WALK_FLOOR}); \
             a ratchet must not pass on a truncated tree",
            report.walked
        )
        .into());
    }

    if let Some(limit) = max {
        let inert = report.inert_count();
        if inert > limit {
            return Err(format!(
                "INERT RATCHET: {inert} contract(s) assert something with no way to refute it, \
                 above the pinned ceiling of {limit}. Give the new contract a \
                 `falsification_tests:` entry, or drop the claim field if it is a catalog."
            )
            .into());
        }
        println!("\nRatchet: {inert} inert <= {limit} pinned — PASS");
    }

    Ok(())
}

fn print_text(report: &InertReport, list: bool) {
    let falsifiable = report.count(Verdict::Falsifiable);
    let catalog = report.count(Verdict::Catalog);
    let inert = report.count(Verdict::Inert);

    println!("Inert Contract Report");
    println!("=====================");
    println!();
    println!("  Walked            : {}", report.walked);
    println!("  Parse failures    : {}", report.parse_failures);
    println!("  Probe failures    : {}", report.probe_failures.len());
    println!("  Classified        : {}", report.contracts.len());
    println!();
    println!("  falsifiable       : {falsifiable}  (>=1 falsification_tests entry)");
    println!("  catalog           : {catalog}  (asserts nothing a test could refute)");
    println!("  inert             : {inert}  (asserts something, cannot be refuted)");

    if inert > 0 {
        println!();
        println!("  Inert by trigger:");
        let mut by_reason: std::collections::BTreeMap<&str, usize> =
            std::collections::BTreeMap::new();
        for c in report.inert() {
            for r in &c.reasons {
                *by_reason.entry(r.as_str()).or_default() += 1;
            }
        }
        for (reason, n) in by_reason {
            println!("    {reason:<28} {n}");
        }
    }

    if list {
        println!();
        println!("  Inert contracts:");
        for c in report.inert() {
            println!(
                "    {} [{}] {}",
                c.path.display(),
                c.kind,
                c.reasons.join(",")
            );
        }
    }
}

fn print_json(report: &InertReport) {
    let entries: Vec<String> = report
        .inert()
        .iter()
        .map(|c| {
            format!(
                r#"{{"path":{},"stem":{},"kind":"{}","reasons":[{}]}}"#,
                json_str(&c.path.display().to_string()),
                json_str(&c.stem),
                c.kind,
                c.reasons
                    .iter()
                    .map(|r| json_str(r))
                    .collect::<Vec<_>>()
                    .join(",")
            )
        })
        .collect();
    println!(
        r#"{{"walked":{},"parse_failures":{},"probe_failures":{},"classified":{},"falsifiable":{},"catalog":{},"inert":{},"inert_contracts":[{}]}}"#,
        report.walked,
        report.parse_failures,
        report.probe_failures.len(),
        report.contracts.len(),
        report.count(Verdict::Falsifiable),
        report.count(Verdict::Catalog),
        report.count(Verdict::Inert),
        entries.join(",")
    );
}

fn json_str(s: &str) -> String {
    let escaped = s.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{escaped}\"")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_tree() -> tempfile::TempDir {
        let tmp = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            tmp.path().join("inert-v1.yaml"),
            "metadata:\n  version: \"1.0.0\"\n  description: d\n  kind: schema\nfalsification:\n  - id: F\n",
        )
        .expect("write");
        std::fs::write(
            tmp.path().join("good-v1.yaml"),
            "metadata:\n  version: \"1.0.0\"\n  description: d\n  kind: schema\nfalsification_tests:\n  - id: F\n",
        )
        .expect("write");
        tmp
    }

    #[test]
    fn self_test_mode_passes() {
        assert!(run(Path::new("."), "text", None, false, true).is_ok());
    }

    #[test]
    fn empty_dir_is_vacuous_not_a_pass() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let err = run(tmp.path(), "text", Some(0), false, false)
            .expect_err("an empty tree must not report PASS");
        assert!(err.to_string().contains("VACUOUS"), "got: {err}");
    }

    #[test]
    fn max_below_actual_fails_and_above_passes() {
        // Small tree: --max is refused as vacuous, which is itself the point.
        let tmp = tmp_tree();
        let err = run(tmp.path(), "text", Some(99), false, false).expect_err("small tree");
        assert!(err.to_string().contains("VACUOUS"), "got: {err}");
        // Without --max it reports and succeeds.
        assert!(run(tmp.path(), "text", None, false, false).is_ok());
        assert!(run(tmp.path(), "json", None, true, false).is_ok());
    }

    #[test]
    fn json_escaping_is_applied() {
        assert_eq!(json_str(r#"a"b\c"#), r#""a\"b\\c""#);
    }
}
