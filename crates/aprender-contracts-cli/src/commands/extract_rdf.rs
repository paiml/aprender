//! `pv extract` — ONT-001 §5 ONT-4b, R-15, R-18: the corpus as RDF, written deterministically.
//!
//! Writes `<contract_dir>/contracts.nt` (sorted N-Triples, no blank nodes) and `<contract_dir>/shapes.ttl` (every
//! `shape:` block as SHACL Turtle), and prints `{triples, sha256, shapes_n, written}`. `--check` writes nothing and
//! exits 1 when either tracked file differs from a fresh extraction — R-18: files are canonical, the graph is
//! derived, and CI asserts fresh == tracked. The sha256 is over the N-Triples bytes, so it is the graph's content
//! address (R-15).

use std::path::Path;

use provable_contracts::lint::shapes_gate::collect_shapes;
use provable_contracts::ontology::extract;
use provable_contracts::ontology::shapes::to_turtle;
use sha2::{Digest, Sha256};

#[derive(serde::Serialize)]
struct ExtractReport<'a> {
    triples: usize,
    sha256: String,
    shapes_n: usize,
    written: Vec<&'a str>,
    check: Option<Vec<String>>,
}

/// `check == true` → compare and report drift, write nothing.
pub fn run(contract_dir: &Path, check: bool) -> Result<(), Box<dyn std::error::Error>> {
    let extraction = match extract::all(contract_dir) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(3);
        }
    };
    for w in &extraction.warnings {
        eprintln!("warning: {w}");
    }
    let graph = extraction.graph;
    if graph.is_empty() {
        eprintln!("decline: no contracts under {}", contract_dir.display());
        std::process::exit(2);
    }
    let nt = graph.to_ntriples();
    let (shapes, _) = match collect_shapes(contract_dir) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(3);
        }
    };
    let shapes: Vec<_> = shapes.into_iter().map(|(s, _)| s).collect();
    let ttl = to_turtle(&shapes);
    let mut hasher = Sha256::new();
    hasher.update(nt.as_bytes());
    let sha256 = format!("{:x}", hasher.finalize());
    let nt_path = contract_dir.join("contracts.nt");
    let ttl_path = contract_dir.join("shapes.ttl");

    let mut written = Vec::new();
    let mut drift = Vec::new();
    for (path, fresh) in [(&nt_path, &nt), (&ttl_path, &ttl)] {
        let tracked = std::fs::read_to_string(path).unwrap_or_default();
        if check {
            if tracked != *fresh {
                drift.push(format!(
                    "{} differs from a fresh extraction",
                    path.display()
                ));
            }
        } else if tracked != *fresh {
            std::fs::write(path, fresh)?;
            written.push(path.file_name().and_then(|n| n.to_str()).unwrap_or("?"));
        }
    }
    let report = ExtractReport {
        triples: graph.len(),
        sha256,
        shapes_n: shapes.len(),
        written,
        check: check.then(|| drift.clone()),
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    if check && !drift.is_empty() {
        for d in &drift {
            eprintln!(
                "reject: {d} — run `pv extract {}` and commit the result (R-18)",
                contract_dir.display()
            );
        }
        std::process::exit(1);
    }
    Ok(())
}
