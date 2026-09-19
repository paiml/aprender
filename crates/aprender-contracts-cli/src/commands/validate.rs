use std::collections::HashSet;
use std::path::{Path, PathBuf};

use provable_contracts::error::Severity;
use provable_contracts::lint::collect_yaml_files;
use provable_contracts::schema::{parse_contract_str, validate_artifact, ArtifactKind};

use crate::contract_walk::{has_contract_files, ZeroContracts};

/// `pv validate <path>` — validate whatever artifact `path` holds.
///
/// Dispatches on what the file IS, not on the assumption that everything under
/// `contracts/` is a `Contract`. Five files in the corpus are not: two pv
/// binding registries and three publish manifests, all of which failed here
/// with ``missing field `metadata` `` while the directory walkers that lint
/// them already knew to treat them differently. See
/// `provable_contracts::schema::artifact`.
///
/// PVL-1 (PMAT-1099): a path that does not exist, or a directory without a
/// contract file, is a DECLINE — exit 2, [`ZeroContracts`] — never an OS error
/// at exit 1. Measured 2026-09-11 by the third review quorum on #3093, after
/// two PASS quorums: `pv validate <empty dir>` answered `Is a directory (os
/// error 21)` and `pv validate /nonexistent` answered `No such file or
/// directory (os error 2)`, both exit 1 — validate reads one artifact and never
/// went through the walker every other reporting command was routed through.
/// A directory WITH contract files validates every file lint would walk
/// (the one definition of the corpus) and fails if any of them fails:
/// measured, and named.
pub fn run(path: &Path, check_ids: bool) -> Result<(), Box<dyn std::error::Error>> {
    if !path.exists() || (path.is_dir() && !has_contract_files(path)) {
        return Err(ZeroContracts {
            path: path.to_path_buf(),
            filter: None,
        }
        .into());
    }
    if check_ids {
        return run_check_ids(path);
    }
    if path.is_dir() {
        return run_dir(path);
    }
    run_file(path)
}

/// `pv validate <path> --check-ids` — report the obligation-id denominators.
///
/// This exists to be RUN ONCE, as the evidence that a consumer reads the key.
/// `ProofObligation` had no `id` field until #3314's follow-up: `id:` was
/// written to disk and silently dropped on parse, so 3,612 generated ids were
/// decoration. Counting them from YAML would have proved nothing -- the count
/// has to come through the same deserialization every other pv command uses,
/// which is why this walks parsed contracts and not the text.
///
/// Reports, deliberately, three numbers and not a verdict:
///   obligations  every proof obligation in the corpus
///   with id      how many carry an `id:` AFTER deserialization
///   referenced   kani harnesses whose `obligation:` RESOLVES to one of them
///
/// `referenced` is the second half of the same question: an id nothing cites is
/// still not a citation. Requiring these (ids mandatory, references resolving,
/// `N evaluated` instead of `0 errors`) is the NEXT change; this one only
/// counts, so it can land without red-lining a corpus that is still being named.
fn run_check_ids(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut files: Vec<PathBuf> = Vec::new();
    if path.is_dir() {
        collect_yaml_files(path, &mut files);
    } else {
        files.push(path.to_path_buf());
    }
    files.sort();

    let (mut obligations, mut with_id, mut referenced, mut contracts) =
        (0usize, 0usize, 0usize, 0usize);
    for file in &files {
        let Ok(content) = std::fs::read_to_string(file) else {
            continue;
        };
        // Not every YAML under contracts/ is a Contract -- binding registries
        // and publish manifests live there too. A parse failure here is "not a
        // contract", not an error: `pv validate` already judges those.
        let Ok(contract) = parse_contract_str(&content) else {
            continue;
        };
        contracts += 1;
        let ids: HashSet<&str> = contract
            .proof_obligations
            .iter()
            .filter_map(|ob| ob.id.as_deref())
            .collect();
        obligations += contract.proof_obligations.len();
        with_id += ids.len();
        referenced += contract
            .kani_harnesses
            .iter()
            .filter(|h| ids.contains(h.obligation.as_str()))
            .count();
    }

    println!(
        "{} obligations, {} with id, {} referenced   ({} contract(s) under {})",
        obligations,
        with_id,
        referenced,
        contracts,
        path.display()
    );
    Ok(())
}

/// Validate every contract file under `dir`, accumulating failures instead of
/// stopping at the first, so the verdict names all of them.
fn run_dir(dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut files: Vec<PathBuf> = Vec::new();
    collect_yaml_files(dir, &mut files);
    files.sort();
    let mut failed: Vec<PathBuf> = Vec::new();
    for file in &files {
        println!("== {}", file.display());
        if let Err(e) = run_file(file) {
            println!("{e}");
            failed.push(file.clone());
        }
    }
    println!(
        "\n{} artifact(s) under {}, {} failed",
        files.len(),
        dir.display(),
        failed.len()
    );
    if failed.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{} of {} artifacts under {} failed validation",
            failed.len(),
            files.len(),
            dir.display()
        )
        .into())
    }
}

fn run_file(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let (kind, violations) = validate_artifact(path)?;

    let errors: Vec<_> = violations
        .iter()
        .filter(|v| v.severity == Severity::Error)
        .collect();
    let warnings: Vec<_> = violations
        .iter()
        .filter(|v| v.severity == Severity::Warning)
        .collect();

    for v in &violations {
        println!("{v}");
    }

    println!("\n{} error(s), {} warning(s)", errors.len(), warnings.len());

    if errors.is_empty() {
        println!("{} is valid.", noun(kind));
        Ok(())
    } else {
        Err(format!("{} has {} validation error(s)", noun(kind), errors.len()).into())
    }
}

/// How to name the artifact in the verdict line. Naming the kind is the point:
/// a reader who runs `pv validate contracts/binding.yaml` and is told
/// "Contract is valid." has been told something false about which rules ran.
fn noun(kind: ArtifactKind) -> &'static str {
    match kind {
        ArtifactKind::Contract => "Contract",
        ArtifactKind::Binding => "Binding registry (kind: binding)",
        ArtifactKind::PublishManifest => "Publish manifest (kind: publish-manifest)",
        ArtifactKind::ExternalCorpora => "External-corpora declaration (kind: external-corpora)",
    }
}
