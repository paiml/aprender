//! `pv challenge gen | check` (PVL-001 EV-7a, #4200). The rendering lives in
//! [`provable_contracts::discharge::challenge`]; this module writes or compares, and prints.
//!
//! Exit: 0 fresh · 1 reject (a statement could not be lifted, or `Challenge/` differs from its regeneration)
//! · 2 decline (the tree does not load, or zero challenges: nothing is pinned, which must not read as fresh).

use std::path::Path;

use provable_contracts::discharge::challenge::{self, Rendered, CHALLENGE_DIR};

use super::discharge::{DischargeDeclined, DischargeRejected};
use crate::cli::ChallengeAction;

type Res = Result<(), Box<dyn std::error::Error>>;

pub fn run(action: ChallengeAction) -> Res {
    match action {
        ChallengeAction::Gen {
            contracts,
            lean_dir,
        } => gen(&contracts, &lean_dir),
        ChallengeAction::Check {
            contracts,
            lean_dir,
        } => check(&contracts, &lean_dir),
    }
}

fn load(contracts: &Path, lean_dir: &Path) -> Result<Rendered, DischargeDeclined> {
    let r = challenge::render(lean_dir, contracts).map_err(DischargeDeclined)?;
    if r.files.is_empty() && r.unrestated.is_empty() {
        return Err(DischargeDeclined(format!(
            "zero challenges: no contract under {} binds a theorem of {}",
            contracts.display(),
            lean_dir.display()
        )));
    }
    Ok(r)
}

/// One FAIL line per root whose statement could not be lifted; the count.
fn report_unrestated(r: &Rendered) -> usize {
    for (stem, fqn, why) in &r.unrestated {
        println!("FAIL UNRESTATED {stem}: {fqn}: {why}");
    }
    r.unrestated.len()
}

fn gen(contracts: &Path, lean_dir: &Path) -> Res {
    let r = load(contracts, lean_dir)?;
    let written = challenge::write(lean_dir, &r)
        .map_err(|e| DischargeDeclined(format!("{}/{CHALLENGE_DIR}: {e}", lean_dir.display())))?;
    for rel in &written {
        println!("wrote {rel}");
    }
    println!(
        "{} challenge file(s) under {}/{CHALLENGE_DIR} ({} rewritten)",
        r.files.len(),
        lean_dir.display(),
        written.len()
    );
    match report_unrestated(&r) {
        0 => Ok(()),
        n => Err(DischargeRejected(format!("{n} bound theorem(s) have no challenge")).into()),
    }
}

fn check(contracts: &Path, lean_dir: &Path) -> Res {
    let r = load(contracts, lean_dir)?;
    let diffs = challenge::diff(lean_dir, &r);
    for d in &diffs {
        println!("FAIL STALE {d}");
    }
    let n = diffs.len() + report_unrestated(&r);
    if n > 0 {
        return Err(DischargeRejected(format!(
            "{n} failure(s); regenerate with `pv challenge gen {} {}`",
            contracts.display(),
            lean_dir.display()
        ))
        .into());
    }
    println!(
        "challenge-fresh: {} file(s) under {}/{CHALLENGE_DIR} match their regeneration",
        r.files.len(),
        lean_dir.display()
    );
    Ok(())
}
