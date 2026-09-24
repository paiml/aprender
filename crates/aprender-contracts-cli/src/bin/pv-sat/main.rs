//! `pv-sat` — ONT-5's reasoner. Decides whether a corpus's typed relations are jointly satisfiable and writes the
//! certificate the `ont-consistency` gate re-checks: `contracts/witness/<relations_sha256>.json`.
//!
//! ```text
//! pv-sat [CONTRACT_DIR]     write (or confirm) the witness; default `contracts`
//! pv-sat --self-test        the reasoner's plant, its satisfiable twin, and the checker's corrupt core
//!                           (and ONT-4e's: `pc_liskov_reasoner`, `pc_liskov_checker`)
//! ```
//!
//! Private to this bin target (F-7): the library exports the graph and the checker, and nothing in it can reach
//! the reasoner. Before it writes, pv-sat runs its plant (`pc_reasoner`); a reasoner that cannot find the planted
//! core never writes a witness. A witness already on disk that names the same graph and still checks is left
//! untouched, so `make contracts` leaves a clean tree clean. Other `<sha>.json` files in `witness/` are pruned —
//! one graph, one witness.
//!
//! ONT-4e: once the consistency witness stands, pv-sat also certifies every checkable `A refines B` (R-20) in
//! `contracts/witness/liskov/<liskov_sha256>.json` — see `liskov.rs`.
//!
//! Exit: 0 written or confirmed · 1 a control failed · 2 nothing to reason over (no Σ, no typed relation) · 3 Σ
//! malformed or the witness could not be written.

mod liskov;
mod plant;
mod sat;

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use provable_contracts::lint::relations_gate::{typed_graph, TypedGraph};
use provable_contracts::ontology::witness::{
    census_id_set_sha256, check, pc_checker, relations_sha256, witness_path, ClauseSet, Witness,
    FIRED,
};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [flag] if flag == "--self-test" => self_test(),
        [flag] if flag == "-h" || flag == "--help" => {
            println!("usage: pv-sat [CONTRACT_DIR] | pv-sat --self-test");
            ExitCode::SUCCESS
        }
        [] => write_witness(Path::new("contracts")),
        [dir] => write_witness(Path::new(dir)),
        _ => {
            eprintln!("usage: pv-sat [CONTRACT_DIR] | pv-sat --self-test");
            ExitCode::from(3)
        }
    }
}

fn self_test() -> ExitCode {
    let checks: [(&str, Result<(), String>); 5] = [
        ("pc_reasoner", plant::pc_reasoner().map(|_| ())),
        ("pc_model", plant::pc_model()),
        ("pc_checker", pc_checker().map(|_| ())),
        ("pc_liskov_reasoner", liskov::pc_reasoner().map(|_| ())),
        (
            "pc_liskov_checker",
            provable_contracts::ontology::liskov::pc_checker().map(|_| ()),
        ),
    ];
    let mut ok = true;
    for (name, r) in &checks {
        match r {
            Ok(()) => println!("pv-sat self-test: {name} {FIRED}"),
            Err(e) => {
                ok = false;
                eprintln!("pv-sat self-test: {name} FAILED: {e}");
            }
        }
    }
    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

/// The consistency witness, then (ONT-4e) the Liskov witness for every checkable `refines` pair.
fn write_witness(dir: &Path) -> ExitCode {
    let code = write_consistency(dir);
    if code != ExitCode::SUCCESS {
        return code;
    }
    let TypedGraph::Read { edges, .. } = typed_graph(dir) else {
        return code;
    };
    let pairs = liskov::checkable_pairs(dir, &edges);
    liskov::write(dir, &pairs, git_head(dir)).map_or_else(ExitCode::from, |()| ExitCode::SUCCESS)
}

fn write_consistency(dir: &Path) -> ExitCode {
    let pc = match plant::pc_reasoner() {
        Ok(fired) => fired,
        Err(e) => {
            eprintln!("pv-sat: pc_reasoner did not fire, no witness written: {e}");
            return ExitCode::from(1);
        }
    };
    let (ids, edges) = match typed_graph(dir) {
        TypedGraph::NoSigma => {
            eprintln!(
                "pv-sat: no {}/ontology.yaml — nothing to reason over",
                dir.display()
            );
            return ExitCode::from(2);
        }
        TypedGraph::Malformed(e) => {
            eprintln!("pv-sat: Σ is malformed: {e}");
            return ExitCode::from(3);
        }
        TypedGraph::Read { ids, edges } => (ids, edges),
    };
    let cs = ClauseSet::from_graph(&ids, &edges);
    if cs.checkable_n() == 0 {
        eprintln!(
            "pv-sat: no typed relation clause in {} contracts — nothing to reason over",
            ids.len()
        );
        return ExitCode::from(2);
    }
    let census_sha = census_id_set_sha256(&ids);
    let relations_sha = relations_sha256(&edges);
    let path = witness_path(dir, &relations_sha);

    if let Some(w) = fresh_witness(&path, &cs, &census_sha, &relations_sha) {
        println!(
            "pv-sat: {} is fresh ({}; checkable_n {})",
            path.display(),
            kind(&w),
            w.checkable_n
        );
        return prune(&path).map_or(ExitCode::from(3), |()| ExitCode::SUCCESS);
    }

    let start = Instant::now();
    let result = sat::solve(&cs);
    let cpu_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
    if let Err(e) = check(&cs, &result) {
        eprintln!("pv-sat: the reasoner's own answer does not check, no witness written: {e}");
        return ExitCode::from(1);
    }
    let witness = Witness {
        census_id_set_sha256: census_sha,
        relations_sha256: relations_sha,
        reasoner_git_sha: git_head(dir),
        checkable_n: cs.checkable_n(),
        result,
        pc_reasoner: pc.to_string(),
        cpu_ms,
    };
    let written = path
        .parent()
        .map_or(Ok(()), std::fs::create_dir_all)
        .and_then(|()| {
            let mut text = serde_json::to_string_pretty(&witness).map_err(std::io::Error::other)?;
            text.push('\n');
            std::fs::write(&path, text)
        });
    if let Err(e) = written {
        eprintln!("pv-sat: cannot write {}: {e}", path.display());
        return ExitCode::from(3);
    }
    println!(
        "pv-sat: wrote {} ({}; checkable_n {}; {} ms)",
        path.display(),
        kind(&witness),
        witness.checkable_n,
        cpu_ms
    );
    prune(&path).map_or(ExitCode::from(3), |()| ExitCode::SUCCESS)
}

fn kind(w: &Witness) -> &'static str {
    match w.result {
        provable_contracts::ontology::witness::WitnessResult::UnsatCore(_) => "unsat_core",
        provable_contracts::ontology::witness::WitnessResult::Model(_) => "model",
    }
}

/// The witness on disk, when it names this graph, its reasoner's plant fired, and it still checks.
fn fresh_witness(path: &Path, cs: &ClauseSet, census: &str, relations: &str) -> Option<Witness> {
    let w: Witness = serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    let fresh = w.census_id_set_sha256 == census
        && w.relations_sha256 == relations
        && w.pc_reasoner == FIRED
        && w.checkable_n == cs.checkable_n()
        && check(cs, &w.result).is_ok();
    fresh.then_some(w)
}

/// Remove every other `<64 hex>.json` beside `keep`.
fn prune(keep: &Path) -> Result<(), ()> {
    let Some(dir) = keep.parent() else {
        return Ok(());
    };
    let entries = std::fs::read_dir(dir)
        .map_err(|e| eprintln!("pv-sat: cannot list {}: {e}", dir.display()))?;
    for entry in entries.flatten() {
        let p: PathBuf = entry.path();
        let is_witness = p.extension().is_some_and(|e| e == "json")
            && p.file_stem().and_then(|s| s.to_str()).is_some_and(|s| {
                s.len() == 64
                    && s.bytes()
                        .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
            });
        if is_witness && p != keep {
            std::fs::remove_file(&p)
                .map_err(|e| eprintln!("pv-sat: cannot prune {}: {e}", p.display()))?;
            println!("pv-sat: pruned {}", p.display());
        }
    }
    Ok(())
}

fn git_head(dir: &Path) -> Option<String> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    let sha = String::from_utf8(out.stdout).ok()?.trim().to_string();
    (out.status.success() && sha.len() == 40).then_some(sha)
}

#[cfg(test)]
mod tests;
