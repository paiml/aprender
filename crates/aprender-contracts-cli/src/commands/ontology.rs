//! `pv ontology export --owl` / `pv ontology tbox` — ONT-001 §3.8, row ONT-2c.
//!
//! `export --owl` writes Σ as OWL 2 functional syntax with the in-house writer, byte-deterministic, so CI can
//! `cmp` a fresh export against the tracked `contracts/ontology.ofn` (R-18). `tbox` writes the told-closure
//! classification, `contracts/tbox-report.json`, which is ADVISORY: the `tbox` lint gate maps it to
//! `Unknown{Advisory}` and it never arms (R-7: no inferred fact arms a merge).
//!
//! Exit codes follow `pv lint`: 3 `error:` when Σ is malformed, cannot be written as OWL, or the TBox
//! precondition fails. In that last case the report is still printed, so the refusal names its axiom.

use std::path::Path;

use provable_contracts::ontology::owl;
use provable_contracts::ontology::sigma::Sigma;

use crate::cli::OntologyCommand;

pub fn run(command: &OntologyCommand) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        OntologyCommand::Export {
            sigma,
            owl: as_owl,
            write,
        } => {
            if !as_owl {
                eprintln!("error: `pv ontology export` writes one format today; pass --owl");
                std::process::exit(2);
            }
            let export = load_export(sigma);
            emit(sigma, "ontology.ofn", &owl::to_ofn(&export), *write)
        }
        OntologyCommand::Tbox { sigma, write } => {
            let export = load_export(sigma);
            let report = owl::tbox(&export);
            emit(
                sigma,
                "tbox-report.json",
                &owl::report_json(&report),
                *write,
            )?;
            if !report.precondition.holds {
                for r in &report.precondition.refused {
                    eprintln!("error: told-closure precondition fails: {r}");
                }
                std::process::exit(3);
            }
            Ok(())
        }
    }
}

fn load_export(sigma_path: &Path) -> owl::OwlExport {
    let fail = |msg: String| -> ! {
        eprintln!("error: {msg}");
        std::process::exit(3);
    };
    let text = std::fs::read_to_string(sigma_path)
        .unwrap_or_else(|e| fail(format!("cannot read Σ {}: {e}", sigma_path.display())));
    let sigma = Sigma::from_yaml(&text).unwrap_or_else(|e| fail(format!("Σ: {e}")));
    if let Err(e) = sigma.check_integrity() {
        fail(format!("Σ: {e}"));
    }
    owl::export(&sigma).unwrap_or_else(|e| fail(e.to_string()))
}

fn emit(
    sigma_path: &Path,
    name: &str,
    body: &str,
    write: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if write {
        let dir = sigma_path.parent().unwrap_or_else(|| Path::new("."));
        let out = dir.join(name);
        std::fs::write(&out, body)?;
        eprintln!("wrote {}", out.display());
    } else {
        print!("{body}");
    }
    Ok(())
}
