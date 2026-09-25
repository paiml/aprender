//! `cargo run -p aprender-review-experiment --example rex -- <command>`
//!
//! An example, not a `[[bin]]`: the workspace binary register only shrinks
//! (monorepo_invariants::test_no_unauthorized_binaries).
//!
//! Commands:
//! - `prereg`        print the prereg lock the tree implies
//! - `prereg-check`  exit 1 unless the committed lock matches the tree
//! - `corpus-build PRS_JSON CUTOFF ITEMS_DIR MUTANTS_JSON...`
//!   build corpus v1 (REX-02): writes `ITEMS_DIR/<id>.diff`, and in the repo
//!   `docs/audits/review-corpus/corpus-v1.jsonl` + `test-manifest-v1.txt`.
//!   Prints counts only; it never prints a test item.

use aprender_review_experiment::build_corpus::{
    choose, g_candidates, is_green, p_candidates, r_candidates, seal, Mutant, Pr, PER_CLASS,
};
use aprender_review_experiment::corpus::{assign_splits, corpus_version, render_manifest, Item};
use aprender_review_experiment::prereg;
use std::collections::BTreeMap;
use std::process::ExitCode;

/// Seed for every draw (the epic number, analysis plan §Seeds).
const SEED: u64 = 4354;
const CORPUS_DIR: &str = "docs/audits/review-corpus";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("prereg") => prereg_cmd(false),
        Some("prereg-check") => prereg_cmd(true),
        Some("corpus-build") if args.len() >= 5 => match corpus_build(&args[1..]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("rex corpus-build: {e}");
                ExitCode::from(1)
            }
        },
        _ => {
            eprintln!(
                "usage: rex <prereg|prereg-check|corpus-build PRS CUTOFF ITEMS_DIR MUTANTS...>"
            );
            ExitCode::from(2)
        }
    }
}

fn prereg_cmd(check: bool) -> ExitCode {
    let Some(c) = prereg::Components::in_tree() else {
        eprintln!("rex: spec lacks a §2..§6 span");
        return ExitCode::from(2);
    };
    if !check {
        print!("{}", c.render_lock());
        return ExitCode::SUCCESS;
    }
    let bad = prereg::verify(prereg::LOCK, &c);
    if bad.is_empty() {
        println!("rex-prereg-v1 OK prereg_sha={}", c.prereg_sha());
        return ExitCode::SUCCESS;
    }
    for b in &bad {
        eprintln!("rex-prereg-v1 DRIFT {b}");
    }
    ExitCode::from(1)
}

fn read_json<T: serde::de::DeserializeOwned>(path: &str) -> Result<T, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))
}

fn corpus_build(a: &[String]) -> Result<(), String> {
    let (prs_path, cutoff, items_dir) = (&a[0], &a[1], &a[2]);
    let prs: Vec<Pr> = read_json(prs_path)?;
    let mut mutants: Vec<Mutant> = Vec::new();
    for m in &a[3..] {
        mutants.extend(read_json::<Vec<Mutant>>(m)?);
    }
    let base = String::from_utf8_lossy(
        &std::process::Command::new("git")
            .args(["rev-parse", "--short=9", "HEAD"])
            .output()
            .map_err(|e| e.to_string())?
            .stdout,
    )
    .trim()
    .to_string();

    let r = choose(r_candidates(&prs), PER_CLASS, SEED);
    let g_pool: Vec<_> = choose(g_candidates(&prs, cutoff), PER_CLASS * 2, SEED)
        .into_iter()
        .filter(|(i, _)| i.id[4..].parse().is_ok_and(is_green))
        .collect();
    let g = choose(g_pool, PER_CLASS, SEED);
    let p = p_candidates(&mutants, &base, PER_CLASS, SEED);

    let mut all: Vec<(Item, String)> = p.into_iter().chain(r).chain(g).collect();
    let mut items: Vec<Item> = all.iter().map(|(i, _)| i.clone()).collect();
    assign_splits(&mut items, SEED);
    let split: BTreeMap<String, _> = items.iter().map(|i| (i.id.clone(), i.split)).collect();
    for (i, _) in &mut all {
        i.split = split[&i.id];
    }
    all.sort_by(|x, y| x.0.id.cmp(&y.0.id));
    write_outputs(&all, items_dir)
}

fn write_outputs(all: &[(Item, String)], items_dir: &str) -> Result<(), String> {
    std::fs::create_dir_all(items_dir).map_err(|e| e.to_string())?;
    let mut jsonl = String::new();
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for (i, d) in all {
        std::fs::write(format!("{items_dir}/{}.diff", i.id), d).map_err(|e| e.to_string())?;
        jsonl.push_str(&serde_json::to_string(i).map_err(|e| e.to_string())?);
        jsonl.push('\n');
        *counts
            .entry(format!("{:?} {:?} {:?}", i.class, i.stratum, i.split))
            .or_default() += 1;
    }
    let manifest = render_manifest(&seal(all));
    std::fs::write(format!("{CORPUS_DIR}/corpus-v1.jsonl"), &jsonl).map_err(|e| e.to_string())?;
    std::fs::write(format!("{CORPUS_DIR}/test-manifest-v1.txt"), &manifest)
        .map_err(|e| e.to_string())?;
    for (k, n) in &counts {
        println!("{k} {n}");
    }
    println!(
        "items {} corpus_version {}",
        all.len(),
        corpus_version(&manifest)
    );
    Ok(())
}
