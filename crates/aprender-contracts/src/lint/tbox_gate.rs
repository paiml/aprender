//! ONT-001 §3.8, row ONT-2c — the `tbox` gate: Σ's classification is ADVISORY, never a verdict that arms.
//!
//! What it answers, in order:
//!
//! - no `ontology.yaml` → [`TboxOutcome::NoSigma`] (decline: nothing measured, R-2);
//! - Σ malformed, or not writable as OWL → [`TboxOutcome::Malformed`] (exit 3, the declaration's fault);
//! - the tracked `ontology.ofn` or `tbox-report.json` is missing or differs from a fresh computation →
//!   [`TboxOutcome::Stale`] (exit 3, R-18: files are derived, and CI asserts fresh == tracked);
//! - the told-closure precondition fails → [`TboxOutcome::PreconditionFailed`] (exit 3: the method's
//!   equivalence to EL classification is gone, so no claim about subsumption is made);
//! - otherwise → [`TboxOutcome::Advisory`], which the CLI reports as `Unknown{Advisory}`.
//!
//! **There is no Pass arm.** R-7: "no inferred fact arms a merge". Even a clean classification is
//! `Unknown{Advisory}`, so this gate can never enter `armed_gates` as a green. The independent oracle (ELK,
//! `tests/oracle/`) must agree with the report at the RELEASE gate only, never per PR.

use std::path::Path;

use crate::ontology::owl::{self, TboxReport};
use crate::ontology::sigma::Sigma;

#[derive(Debug)]
pub enum TboxOutcome {
    NoSigma,
    Malformed(String),
    Stale(String),
    PreconditionFailed(Vec<String>),
    Advisory(Box<TboxReport>),
}

#[must_use]
pub fn run_tbox_gate(contract_dir: &Path) -> TboxOutcome {
    let Ok(text) = std::fs::read_to_string(contract_dir.join("ontology.yaml")) else {
        return TboxOutcome::NoSigma;
    };
    let sigma = match Sigma::from_yaml(&text) {
        Ok(s) => s,
        Err(e) => return TboxOutcome::Malformed(format!("Σ: {e}")),
    };
    if let Err(e) = sigma.check_integrity() {
        return TboxOutcome::Malformed(format!("Σ: {e}"));
    }
    let export = match owl::export(&sigma) {
        Ok(x) => x,
        Err(e) => return TboxOutcome::Malformed(e.to_string()),
    };
    let report = owl::tbox(&export);
    let fresh = [
        (
            "ontology.ofn",
            owl::to_ofn(&export),
            "pv ontology export --owl --write",
        ),
        (
            "tbox-report.json",
            owl::report_json(&report),
            "pv ontology tbox --write",
        ),
    ];
    for (name, want, fix) in fresh {
        match std::fs::read_to_string(contract_dir.join(name)) {
            Ok(got) if got == want => {}
            Ok(_) => {
                return TboxOutcome::Stale(format!(
                    "{name} differs from a fresh computation; run `{fix}`"
                ))
            }
            Err(_) => return TboxOutcome::Stale(format!("{name} is missing; run `{fix}`")),
        }
    }
    if !report.precondition.holds {
        return TboxOutcome::PreconditionFailed(report.precondition.refused.clone());
    }
    TboxOutcome::Advisory(Box::new(report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn repo() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn corpus(files: &[(&str, &str)]) -> tempfile::TempDir {
        let d = tempfile::tempdir().expect("tempdir");
        for (n, body) in files {
            std::fs::write(d.path().join(n), body).expect("write");
        }
        d
    }

    fn fixture() -> (String, String, String) {
        let sigma = std::fs::read_to_string(repo().join("tests/fixtures/ont/owl/ontology.yaml"))
            .expect("Σ");
        let e = owl::export(&Sigma::from_yaml(&sigma).expect("Σ")).expect("export");
        (sigma, owl::to_ofn(&e), owl::report_json(&owl::tbox(&e)))
    }

    #[test]
    fn ont2c_real_corpus_is_advisory_never_pass() {
        match run_tbox_gate(&repo().join("contracts")) {
            TboxOutcome::Advisory(r) => assert!(r.advisory && r.consistent),
            other => panic!("the repository's tbox gate must answer Advisory: {other:?}"),
        }
    }

    #[test]
    fn ont2c_fresh_fixture_is_advisory() {
        let (s, o, r) = fixture();
        let d = corpus(&[
            ("ontology.yaml", &s),
            ("ontology.ofn", &o),
            ("tbox-report.json", &r),
        ]);
        assert!(matches!(run_tbox_gate(d.path()), TboxOutcome::Advisory(_)));
    }

    #[test]
    fn ont2c_stale_report_is_refused() {
        let (s, o, r) = fixture();
        let stale = r.replace("\"classes\": 3", "\"classes\": 4");
        assert_ne!(stale, r, "the mutation must change the report");
        let d = corpus(&[
            ("ontology.yaml", &s),
            ("ontology.ofn", &o),
            ("tbox-report.json", &stale),
        ]);
        assert!(
            matches!(run_tbox_gate(d.path()), TboxOutcome::Stale(m) if m.contains("tbox-report.json"))
        );
    }

    #[test]
    fn ont2c_missing_ofn_is_refused() {
        let (s, _, r) = fixture();
        let d = corpus(&[("ontology.yaml", &s), ("tbox-report.json", &r)]);
        assert!(
            matches!(run_tbox_gate(d.path()), TboxOutcome::Stale(m) if m.contains("ontology.ofn"))
        );
    }

    #[test]
    fn ont2c_no_sigma_declines() {
        let d = corpus(&[]);
        assert!(matches!(run_tbox_gate(d.path()), TboxOutcome::NoSigma));
    }
}
