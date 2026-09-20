//! ONT-001 §3.9 for `pv lint` (ONT-6, PMAT-3451): which gates this run arms, and whether the corpus's
//! `armed_gates` shrank against the committed comparand.
//!
//! The comparand is resolved the way `scripts/lib_baseline_ratchet.sh` resolves one — merge-base(HEAD,
//! origin/main), else the origin/main tip — or is the commit `--armed-baseline-ref` names. The two paths
//! fail differently, on purpose:
//!
//! - an EXPLICIT ref that does not resolve, or a corpus absent at it, is an error. The caller asked for a
//!   check; answering "not checked" would report a result nobody measured.
//! - the DEFAULT path with nothing to compare against (no git work tree, no origin/main, a corpus that
//!   is not tracked there) prints `NOT CHECKED (no comparand)` and leaves the exit alone — plan v3 open
//!   point 1, ruled by the grill: every non-git consumer of `pv lint` would otherwise exit 2.
//!
//! A committed `lint-baseline.json` that does not parse is an error on both paths: the committed value
//! exists, it just cannot be read, and skipping it would disarm the check for exactly that commit.

use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::Command;

use provable_contracts::ontology::arming::{
    check_monotone, check_shapes_monotone, ArmedGates, ArmedShapes,
};

const BASELINE: &str = "lint-baseline.json";
const NO_COMPARAND: &str = "NOT CHECKED (no comparand)";

/// What one `pv lint` run arms, and the monotone lines it prints.
pub struct Arming {
    pub armed: ArmedGates,
    pub monotone: String,
    /// ONT-4c1 (§3.9): the per-shape arming monotone line (the gate reads the declaration itself).
    pub shapes_monotone: String,
}

/// The corpus's declared arming — the declaration alone, never the command line (§3.9) — with the monotone
/// check against the comparand. `Err(ArmedGatesShrank)` when a committed gate was dropped;
/// `Err(ArmedShapesShrank)` when a committed shape was.
pub fn resolve(contract_dir: &Path, explicit_ref: Option<&str>) -> Result<Arming, Box<dyn Error>> {
    let baseline = read_baseline(contract_dir)?;
    let declared = ArmedGates::from_baseline(baseline.as_deref())?;
    let declared_shapes = ArmedShapes::from_baseline(baseline.as_deref())?;
    let (monotone, shapes_monotone) = match comparand(contract_dir, explicit_ref)? {
        Comparand::Absent(why) => (
            format!("{NO_COMPARAND} — {why}"),
            format!("{NO_COMPARAND} — {why}"),
        ),
        Comparand::At {
            label,
            text,
            top,
            commit,
            rel,
        } => {
            let committed = ArmedGates::from_baseline(text.as_deref())
                .map_err(|e| format!("armed_gates comparand {label}: {e}"))?;
            check_monotone(&committed, &declared)?;
            let committed_shapes = ArmedShapes::from_baseline(text.as_deref())
                .map_err(|e| format!("armed_shapes comparand {label}: {e}"))?;
            // An `All` comparand armed every shape its corpus carried; "its corpus" is every shape the
            // current corpus declares in a contract file that already existed at the comparand — a shape
            // in a NEW file was not armed there, and may ship reported-first.
            let existed_then: Vec<String> = match &committed_shapes {
                ArmedShapes::All => declared_shape_files(contract_dir)?
                    .into_iter()
                    .filter(|(_, file)| {
                        // `file` is relative to the corpus dir's PARENT; the comparand wants a path
                        // relative to the work-tree root
                        let parent_rel = rel.rsplit_once('/').map_or("", |(p, _)| p);
                        let path = if parent_rel.is_empty() {
                            file.clone()
                        } else {
                            format!("{parent_rel}/{file}")
                        };
                        exists_at(&top, &commit, &path)
                    })
                    .map(|(id, _)| id)
                    .collect(),
                ArmedShapes::Listed(_) => Vec::new(),
            };
            check_shapes_monotone(&committed_shapes, &declared_shapes, &existed_then)?;
            (
                format!(
                    "OK against {label} ({} committed, {} declared)",
                    committed.names().len(),
                    declared.names().len()
                ),
                match (&committed_shapes, &declared_shapes) {
                    (ArmedShapes::All, ArmedShapes::All) => {
                        format!("OK against {label} (all shapes armed there and here)")
                    }
                    (ArmedShapes::All, ArmedShapes::Listed(now)) => format!(
                        "OK against {label} (all {} pre-existing shape(s) armed there, {} declared here)",
                        existed_then.len(),
                        now.len()
                    ),
                    (ArmedShapes::Listed(then), ArmedShapes::Listed(now)) => format!(
                        "OK against {label} ({} committed, {} declared)",
                        then.len(),
                        now.len()
                    ),
                    (ArmedShapes::Listed(then), ArmedShapes::All) => format!(
                        "OK against {label} ({} committed, all armed here)",
                        then.len()
                    ),
                },
            )
        }
    };
    Ok(Arming {
        armed: declared,
        monotone,
        shapes_monotone,
    })
}

/// `(shape id, file relative to the repository root)` for every shape the corpus declares today.
fn declared_shape_files(contract_dir: &Path) -> Result<Vec<(String, String)>, Box<dyn Error>> {
    provable_contracts::lint::shapes_gate::declared_shapes(contract_dir)
        .map_err(|e| format!("shapes: {e}").into())
}

/// The declared arming without the git check (watch mode re-reads it every tick).
pub fn declared(contract_dir: &Path) -> Result<ArmedGates, Box<dyn Error>> {
    Ok(ArmedGates::from_baseline(
        read_baseline(contract_dir)?.as_deref(),
    )?)
}

/// `<contract_dir>/lint-baseline.json`, or `None` when the corpus has none (a single-file corpus never does).
fn read_baseline(contract_dir: &Path) -> Result<Option<String>, Box<dyn Error>> {
    if !contract_dir.is_dir() {
        return Ok(None);
    }
    let path = contract_dir.join(BASELINE);
    match std::fs::read_to_string(&path) {
        Ok(text) => Ok(Some(text)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("{}: {e}", path.display()).into()),
    }
}

enum Comparand {
    /// Default path only: nothing committed to compare against, and why.
    Absent(String),
    /// The committed baseline text at `label` (`None`: the corpus is tracked there without a baseline,
    /// which arms the default set).
    At {
        label: String,
        text: Option<String>,
        /// The work-tree root, the commit, and the corpus dir relative to the root — so a caller can ask
        /// whether a given file existed at the comparand (ONT-4c1: which shapes were armed by `All`).
        top: PathBuf,
        commit: String,
        rel: String,
    },
}

fn comparand(contract_dir: &Path, explicit_ref: Option<&str>) -> Result<Comparand, Box<dyn Error>> {
    let Some((top, rel)) = repo_path(contract_dir) else {
        let why = format!(
            "{} is not a directory in a git work tree",
            contract_dir.display()
        );
        return match explicit_ref {
            Some(r) => Err(format!("--armed-baseline-ref {r}: {why}").into()),
            None => Ok(Comparand::Absent(why)),
        };
    };
    let (commit, label) = match explicit_ref {
        Some(r) => {
            let spec = format!("{r}^{{commit}}");
            let commit =
                git(&top, &["rev-parse", "--verify", "--quiet", &spec]).ok_or_else(|| {
                    format!(
                        "--armed-baseline-ref {r}: not a commit in {}",
                        top.display()
                    )
                })?;
            let label = format!("{r} {}", short(&commit));
            (commit, label)
        }
        None => match default_commit(&top) {
            Some(found) => found,
            None => {
                return Ok(Comparand::Absent(
                    "neither merge-base(HEAD, origin/main) nor origin/main resolves".into(),
                ))
            }
        },
    };
    if !rel.is_empty() && !exists_at(&top, &commit, &rel) {
        let why = format!("{rel} is not tracked at {label}");
        return match explicit_ref {
            Some(r) => Err(format!("--armed-baseline-ref {r}: {why}").into()),
            None => Ok(Comparand::Absent(why)),
        };
    }
    let path = if rel.is_empty() {
        BASELINE.to_string()
    } else {
        format!("{rel}/{BASELINE}")
    };
    let text = if exists_at(&top, &commit, &path) {
        let spec = format!("{commit}:{path}");
        Some(git(&top, &["show", &spec]).ok_or_else(|| format!("git show {spec} failed"))?)
    } else {
        None
    };
    Ok(Comparand::At {
        label,
        text,
        top,
        commit,
        rel,
    })
}

/// merge-base(HEAD, origin/main), else the origin/main tip — `lib_baseline_ratchet.sh`'s order.
fn default_commit(top: &Path) -> Option<(String, String)> {
    if let Some(commit) = git(top, &["merge-base", "HEAD", "origin/main"]) {
        let label = format!("merge-base(HEAD, origin/main) {}", short(&commit));
        return Some((commit, label));
    }
    let commit = git(
        top,
        &["rev-parse", "--verify", "--quiet", "origin/main^{commit}"],
    )?;
    let label = format!("origin/main {}", short(&commit));
    Some((commit, label))
}

/// The work-tree root holding `dir`, and `dir` relative to it with `/` separators (`""` for the root).
fn repo_path(dir: &Path) -> Option<(PathBuf, String)> {
    if !dir.is_dir() {
        return None;
    }
    let top = PathBuf::from(git(dir, &["rev-parse", "--show-toplevel"])?)
        .canonicalize()
        .ok()?;
    let canonical = dir.canonicalize().ok()?;
    let rel = canonical.strip_prefix(&top).ok()?;
    let parts: Option<Vec<&str>> = rel.components().map(|c| c.as_os_str().to_str()).collect();
    Some((top, parts?.join("/")))
}

fn exists_at(top: &Path, commit: &str, path: &str) -> bool {
    let spec = format!("{commit}:{path}");
    git(top, &["cat-file", "-e", &spec]).is_some()
}

fn short(commit: &str) -> &str {
    commit.get(..12).unwrap_or(commit)
}

/// `git -C dir args…` → trimmed stdout on success. `GIT_DIR`/`GIT_WORK_TREE`/`GIT_INDEX_FILE` are
/// dropped: a git hook exports them, and inherited they would point every query at the hook's repository
/// instead of the one holding the corpus.
fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_INDEX_FILE")
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(
        String::from_utf8_lossy(&out.stdout)
            .trim_end_matches('\n')
            .to_string(),
    )
}
