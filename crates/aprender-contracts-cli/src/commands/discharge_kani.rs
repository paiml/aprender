//! `pv discharge check --kani <dir> --baseline <file>` and `pv discharge kani-ratchet` (PVL-001 EV-6c, #4197).
//! The counting lives in [`provable_contracts::kani_assume`]; this module reads and writes the baseline.
//!
//! Exit: 0 accept (no file rose; a fall is reported, never written) · 1 reject (a file rose, or the tree or the
//! baseline cannot be read — the gate fails closed, it never accepts on a count it could not take).

use std::path::Path;

use provable_contracts::kani_assume::{self, Baseline, Count};

use super::discharge::DischargeRejected;

type Res = Result<(), Box<dyn std::error::Error>>;

/// What `make kani-ratchet` records in the baseline's `command`, so a reader can reproduce it.
pub const RATCHET_COMMAND: &str = "make kani-ratchet";

fn measure(dir: &Path) -> Result<Count, DischargeRejected> {
    kani_assume::count_tree(dir).map_err(|e| DischargeRejected(format!("kani: {e}")))
}

fn load(baseline: &Path) -> Result<Baseline, DischargeRejected> {
    let text = std::fs::read_to_string(baseline)
        .map_err(|e| DischargeRejected(format!("kani: baseline {}: {e}", baseline.display())))?;
    let b: Baseline = serde_json::from_str(&text)
        .map_err(|e| DischargeRejected(format!("kani: baseline {}: {e}", baseline.display())))?;
    b.consistent()
        .map_err(|e| DischargeRejected(format!("kani: {}: {e}", baseline.display())))?;
    Ok(b)
}

/// Print one line per rise; the rejection names how many.
fn judge(measured: &Count, b: &Baseline) -> Res {
    let rises = kani_assume::rises(measured, b);
    for r in &rises {
        println!("reject: {}: kani::assume {} -> {}", r.path, r.was, r.now);
    }
    if rises.is_empty() {
        Ok(())
    } else {
        Err(DischargeRejected(format!(
            "kani: {} file(s) rose above the baseline -- remove the assume, or edit the baseline in review",
            rises.len()
        ))
        .into())
    }
}

/// `check --kani`: never writes.
pub fn check(dir: &Path, baseline: Option<&Path>) -> Res {
    let baseline =
        baseline.ok_or_else(|| DischargeRejected("kani: --baseline is required".into()))?;
    let b = load(baseline)?;
    let m = measure(dir)?;
    println!(
        "kani::assume: {} in {} file(s) under {} (baseline {} in {})",
        m.total,
        m.files.len(),
        dir.display(),
        b.total,
        b.files.len()
    );
    judge(&m, &b)?;
    println!("accept: no file's kani::assume count rose");
    Ok(())
}

/// `kani-ratchet`: seed a missing baseline, else rewrite it downward. Rises are reported (rc 1) after the write.
pub fn ratchet(dir: &Path, baseline: &Path) -> Res {
    let m = measure(dir)?;
    let seeded = !baseline.exists();
    let next = if seeded {
        kani_assume::baseline_of(&m, RATCHET_COMMAND)
    } else {
        kani_assume::ratchet_down(&m, &load(baseline)?, RATCHET_COMMAND)
    };
    let mut text = serde_json::to_string_pretty(&next)?;
    text.push('\n');
    std::fs::write(baseline, text).map_err(|e| {
        DischargeRejected(format!("kani: cannot write {}: {e}", baseline.display()))
    })?;
    println!(
        "wrote {} ({} kani::assume in {} file(s))",
        baseline.display(),
        next.total,
        next.files.len()
    );
    if seeded {
        Ok(())
    } else {
        judge(&m, &next)
    }
}
