//! #3569 part 2 — a shrink-only debt count is judged against the SAME count MEASURED at the comparand tree,
//! never against a number stored in `contracts/lint-baseline.json`.
//!
//! A stored number is a comparand the pull request under test can rewrite: restamp it in the same commit
//! and the rise check compares the branch with itself. Every PR that moved a count also had to restamp,
//! and two such PRs conflicted on the same line. So the comparand is now a TREE — the merge-base (or, on a
//! push to main, the first parent), extracted by `scripts/lib/comparand_tree.sh` — named by
//! [`ENV`] as that tree's contract dir, and measured by the one instrument that measures HEAD: the gate
//! itself, run over the comparand. The operator's rule (2026-09-27): never-worse is head vs base, same
//! scanner, same run; no stored limits.
//!
//! No comparand named → `None`, exactly as an unrecorded key was: each gate keeps its own no-baseline
//! verdict (the EV-11 ratchets are `Unknown(Report)`, never a pass). The CI step that names the comparand
//! refuses to run when it cannot name one, so a missing comparand is never a silent green there.

use std::cell::{Cell, RefCell};
use std::path::{Path, PathBuf};

use super::GateResult;

/// The environment variable naming the comparand tree's contract dir (the counterpart of `contracts/`).
pub const ENV: &str = "PV_LINT_COMPARAND";

thread_local! {
    /// Set while a gate is measuring the comparand, so the comparand's own run asks for no comparand.
    static MEASURING: Cell<bool> = const { Cell::new(false) };
    /// A comparand named for this thread only ([`with_comparand`]); it wins over [`ENV`].
    static OVERRIDE: RefCell<Option<PathBuf>> = const { RefCell::new(None) };
}

/// Clears [`MEASURING`] on every exit from [`baseline_at`], a panic included.
struct Measuring;

impl Drop for Measuring {
    fn drop(&mut self) {
        MEASURING.with(|m| m.set(false));
    }
}

/// The comparand contract dir named by [`ENV`]; `None` when unset or empty, or while the comparand is
/// itself being measured.
#[must_use]
pub fn comparand_dir() -> Option<PathBuf> {
    if MEASURING.with(Cell::get) {
        return None;
    }
    if let Some(dir) = OVERRIDE.with(|o| o.borrow().clone()) {
        return Some(dir);
    }
    std::env::var_os(ENV)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

/// `measure` run over `dir`, with no comparand visible inside it. `None` dir → `None`.
pub fn baseline_at(
    dir: Option<&Path>,
    measure: impl FnOnce(&Path) -> Option<usize>,
) -> Option<usize> {
    let dir = dir?;
    if MEASURING.with(Cell::get) {
        return None;
    }
    MEASURING.with(|m| m.set(true));
    let _clear = Measuring;
    measure(dir)
}

/// Run `f` with `dir` as this thread's comparand. The unit-test seam: [`ENV`] is process-wide and tests run
/// on parallel threads, so a test that set it would name a comparand for every other test too.
pub fn with_comparand<T>(dir: &Path, f: impl FnOnce() -> T) -> T {
    struct Restore(Option<PathBuf>);
    impl Drop for Restore {
        fn drop(&mut self) {
            let prev = self.0.take();
            OVERRIDE.with(|o| *o.borrow_mut() = prev);
        }
    }
    let _restore = Restore(OVERRIDE.with(|o| o.borrow_mut().replace(dir.to_path_buf())));
    f()
}

/// The debt count measured at the comparand named by [`ENV`] (see [`baseline_at`]).
pub fn baseline(measure: impl FnOnce(&Path) -> Option<usize>) -> Option<usize> {
    baseline_at(comparand_dir().as_deref(), measure)
}

/// The integer `key` of a gate result's `extra` payload — how a gate reads its own count off its run
/// over the comparand. Absent, or not a non-negative integer → `None`.
#[must_use]
pub fn count_in(result: &GateResult, key: &str) -> Option<usize> {
    let extra = serde_json::to_value(result.extra.as_ref()?).ok()?;
    extra
        .get(key)?
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
}

/// What a rise message names as the comparand.
pub const WHERE: &str =
    "the comparand tree (PV_LINT_COMPARAND: the merge-base, or the first parent on a push to main)";

#[cfg(test)]
#[path = "comparand_tests.rs"]
mod tests;
