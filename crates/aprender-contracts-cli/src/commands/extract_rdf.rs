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
use provable_contracts::ontology::extract::release_inputs::Subject;

use crate::contract_walk::ReleaseArgsRefused;
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

/// `--out` and `--cells-out` describe a release; without a release subject they are refused.
fn refuse_release_only_args(
    out: Option<&Path>,
    cells_out: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    if cells_out.is_some() {
        return Err(ReleaseArgsRefused(
            "--cells-out lists a release's derived cells; it needs --release-version, --release-commit and --surface"
                .into(),
        )
        .into());
    }
    if out.is_some() {
        return Err(ReleaseArgsRefused(
            "--out writes the release evidence graph; it needs --release-version and --release-commit".into(),
        )
        .into());
    }
    Ok(())
}

/// `check == true` → compare and report drift, write nothing. With a release `subject` (aprender#3715) the
/// release evidence joins the graph, and the result goes ONLY to `out`: the tracked `contracts.nt` is the corpus,
/// and a release's receipts written into it would be a release baked into every later PR's baseline.
pub fn run(
    contract_dir: &Path,
    check: bool,
    subject: Option<&Subject>,
    out: Option<&Path>,
    cells_out: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(subject) = subject {
        return run_release(contract_dir, check, subject, out, cells_out);
    }
    refuse_release_only_args(out, cells_out)?;
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
    if !extraction.unparsed.is_empty() {
        eprintln!(
            "skipped: {} crate-local YAML file(s) are not typed contracts, counted by neither Σ nor the census: {}",
            extraction.unparsed.len(),
            extraction.unparsed.join(", ")
        );
    }
    for r in &extraction.refused {
        eprintln!(
            "refused: PV-DUP-001 crate contract `{}` differs from another copy, not unioned: {}",
            r.stem,
            r.paths.join(", ")
        );
    }
    for b in &extraction.code.refused_bindings {
        eprintln!("refused: PV-DUP-001 binding names a refused crate copy: {b}");
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

#[derive(serde::Serialize)]
struct ReleaseExtractReport<'a> {
    triples: usize,
    sha256: String,
    out: &'a str,
    release: provable_contracts::ontology::extract::release_evidence::ReleaseStats,
}

/// `pv extract --release-version V --release-commit MC … --out FILE`: the corpus graph plus the release evidence,
/// written to FILE (never to the tracked files), with its content address and what was derived.
fn run_release(
    contract_dir: &Path,
    check: bool,
    subject: &Subject,
    out: Option<&Path>,
    cells_out: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    if check {
        return Err(ReleaseArgsRefused(
            "--check compares the TRACKED corpus files; a release graph is never tracked, so --check with \
             --release-* has nothing to compare"
                .into(),
        )
        .into());
    }
    let Some(out) = out else {
        return Err(ReleaseArgsRefused(
            "--release-* needs --out FILE: the release graph is never written into the tracked contracts.nt".into(),
        )
        .into());
    };
    let extraction = match extract::all_with(contract_dir, Some(subject)) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(3);
        }
    };
    let nt = extraction.graph.to_ntriples();
    let mut hasher = Sha256::new();
    hasher.update(nt.as_bytes());
    std::fs::write(out, &nt)?;
    if let Some(path) = cells_out {
        write_cells(path, subject, extraction.release.as_ref())?;
    }
    let report = ReleaseExtractReport {
        triples: extraction.graph.len(),
        sha256: format!("{:x}", hasher.finalize()),
        out: out.to_str().unwrap_or("?"),
        release: extraction.release.unwrap_or_default(),
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

/// `--cells-out`: every derived cell — the producer's work list, keyed by `cell_id` (aprender#3745 S2).
fn write_cells(
    path: &Path,
    subject: &Subject,
    release: Option<&provable_contracts::ontology::extract::release_evidence::ReleaseStats>,
) -> Result<(), Box<dyn std::error::Error>> {
    let derived = release.map(|r| r.derived.as_slice()).unwrap_or_default();
    if derived.is_empty() {
        return Err(ReleaseArgsRefused(
            "--cells-out: no cell was derived (no --surface, or a surface with no leaf command)"
                .into(),
        )
        .into());
    }
    let mut doc = serde_json::Map::new();
    doc.insert("schema".into(), "apr-release-cells/v1".into());
    doc.insert("version".into(), subject.version.clone().into());
    doc.insert("release_commit".into(), subject.commit.clone().into());
    doc.insert("cells".into(), serde_json::to_value(derived)?);
    std::fs::write(path, serde_json::to_string_pretty(&doc)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn subject() -> Subject {
        Subject::new("0.69.1", "1111111111111111111111111111111111111111").expect("subject")
    }

    fn refused(r: Result<(), Box<dyn std::error::Error>>) -> String {
        r.expect_err("refused").to_string()
    }

    #[test]
    fn release_only_args_are_refused_by_name_without_a_subject() {
        let p = Path::new("x");
        assert!(refuse_release_only_args(None, None).is_ok());
        assert!(refused(refuse_release_only_args(None, Some(p))).starts_with("--cells-out"));
        assert!(refused(refuse_release_only_args(Some(p), None)).starts_with("--out"));
        let dir = tempfile::tempdir().expect("tempdir");
        assert!(refused(run(dir.path(), false, None, Some(p), None)).starts_with("--out"));
    }

    #[test]
    fn a_release_run_refuses_check_and_a_missing_out() {
        let dir = tempfile::tempdir().expect("tempdir");
        let s = subject();
        assert!(refused(run(dir.path(), true, Some(&s), None, None)).starts_with("--check"));
        assert!(refused(run_release(dir.path(), false, &s, None, None)).starts_with("--release-*"));
    }

    #[test]
    fn cells_out_refuses_a_release_with_no_derived_cell() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("cells.json");
        let msg = refused(write_cells(&path, &subject(), None));
        assert!(msg.starts_with("--cells-out: no cell was derived"), "{msg}");
        let empty =
            provable_contracts::ontology::extract::release_evidence::ReleaseStats::default();
        assert!(write_cells(&path, &subject(), Some(&empty)).is_err());
        assert!(!path.exists(), "nothing is written on a refusal");
    }
}
