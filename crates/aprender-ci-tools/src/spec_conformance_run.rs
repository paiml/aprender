//! The `spec-conformance <root>` driver (PMAT-7534, slice 2d): the Rust port of `check()` and
//! `main()` in `scripts/lib/spec_conformance.py`. It composes the four rule modules and prints
//! their TSV in the Python's order: SPEC, the §6 rows and cases with C0 and C5, the ledger's L
//! records, the §12 DAG's D records (with D5 against the committed derivation), then PARSED,
//! the §6 row count, last.
//!
//! Flags as the Python took them: `--spec <md>` and `--ledger <md>` override the default paths,
//! `--out <json>` (default `<root>/evidence/parity/derived_expiries.json`) or `--no-out`, and
//! `--write`. `--write` skips the D5 comparison and writes the derivation to `--out`; it exits 0
//! when the derivation landed and 1 when the §12 table could not be derived. Without an `--out`
//! there is nothing to write, so `--write` is a plain run.
//!
//! Otherwise the exit code is 1 on any VIOLATION, else 0. The shell caller decides vacuity
//! (MIN_ROWS) from the PARSED line, as it did against the Python.

use crate::pystr::{py_path_str, py_read_text};
use crate::spec_conformance_andon::{
    check_dag, derived_text, read_optional, today_default, Derived,
};
use crate::spec_conformance_dag::Violation;
use crate::spec_conformance_ledger::check_ledger;
use crate::spec_conformance_sec6::{check_id_contiguity, check_section_6, Tree};
use std::path::{Path, PathBuf};

/// The glob the Python joined (`SPEC_GLOB`), spelled out: the one master document.
const MASTER_GLOB: &str = "docs/specifications/PP-LLAMA-001-MASTER*.md";
const MASTER_DIR: &str = "docs/specifications";
const MASTER_PREFIX: &str = "PP-LLAMA-001-MASTER";
const MASTER_SUFFIX: &str = ".md";
const LEDGER_REL: &str = "evidence/parity/LEDGER.md";
const DERIVED_REL: &str = "evidence/parity/derived_expiries.json";

/// The driver's arguments, after clap has read them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Options {
    pub root: PathBuf,
    pub spec: Option<PathBuf>,
    pub ledger: Option<PathBuf>,
    pub out: Option<PathBuf>,
    pub no_out: bool,
    pub write: bool,
}

/// The paths the run reads and writes, every one absolute.
struct Paths {
    root: PathBuf,
    ledger: PathBuf,
    /// The derivation file: read for D5, or written by `--write`. None under `--no-out`.
    out: Option<PathBuf>,
}

/// What the §6/L/D checks of one master produced.
struct Joined {
    /// The §6 row count (PARSED).
    rows: usize,
    /// Whether `--write` landed the derivation.
    written: bool,
}

/// The driver's output: the TSV text and the exit code.
///
/// # Errors
///
/// The reason an input cannot be read or the derivation cannot be written (exit 2 at the caller).
pub fn run(opts: &Options) -> Result<(String, u8), String> {
    let paths = resolve(opts)?;
    let spec_override = opts.spec.as_deref().map(absolute).transpose()?;
    let specs = find_specs(&paths.root, spec_override.as_deref())?;
    let mut lines = Vec::new();
    let (parsed, written) = match specs.as_slice() {
        [spec] => {
            let joined = check_spec(&paths, spec, opts.write, &mut lines)?;
            (joined.rows, joined.written)
        }
        _ => {
            lines.push(c0_line(spec_override.as_deref(), specs.len()));
            (0, false)
        }
    };
    lines.push(format!("PARSED\t{parsed}"));
    let any_violation = lines.iter().any(|l| l.starts_with("VIOLATION\t"));
    let code = if opts.write && paths.out.is_some() {
        u8::from(!written)
    } else {
        u8::from(any_violation)
    };
    let text = lines.iter().map(|l| format!("{l}\n")).collect();
    Ok((text, code))
}

/// The absolute paths for this run. The derivation defaults under the root, as the Python's did.
fn resolve(opts: &Options) -> Result<Paths, String> {
    let root = absolute(&opts.root)?;
    let ledger = match &opts.ledger {
        Some(path) => absolute(path)?,
        None => root.join(LEDGER_REL),
    };
    let out = if opts.no_out {
        None
    } else {
        Some(match &opts.out {
            Some(path) => absolute(path)?,
            None => root.join(DERIVED_REL),
        })
    };
    Ok(Paths { root, ledger, out })
}

/// The C0 record when the master is not exactly one file.
fn c0_line(spec_override: Option<&Path>, found: usize) -> String {
    let what = spec_override.map_or_else(|| MASTER_GLOB.to_owned(), py_path_str);
    Violation {
        rule: "C0",
        row: "-".to_owned(),
        message: format!(
            "found {found} file(s) matching {what} -- the master is the ONE document this \
             guard joins; zero means the scan is broken and more than one means two specs \
             disagree"
        ),
    }
    .line()
}

/// Every master the glob (or the `--spec` override) names, in sorted order.
fn find_specs(root: &Path, spec: Option<&Path>) -> Result<Vec<PathBuf>, String> {
    if let Some(spec) = spec {
        return Ok(if spec.exists() {
            vec![spec.to_path_buf()]
        } else {
            Vec::new()
        });
    }
    let dir = root.join(MASTER_DIR);
    let entries = match std::fs::read_dir(&dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(format!("{}: {e}", dir.display())),
    };
    let mut found: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| is_master(path))
        .collect();
    found.sort();
    Ok(found)
}

fn is_master(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with(MASTER_PREFIX) && name.ends_with(MASTER_SUFFIX))
}

/// Joins the one master to the tree: SPEC, §6 and C5, the ledger, then the DAG.
fn check_spec(
    paths: &Paths,
    spec: &Path,
    write: bool,
    lines: &mut Vec<String>,
) -> Result<Joined, String> {
    let rel = relative(&paths.root, spec);
    lines.push(format!("SPEC\t{rel}"));
    let bytes = std::fs::read(spec).map_err(|e| format!("{}: {e}", spec.display()))?;
    let text = py_read_text(&bytes);
    let section = check_section_6(&rel, &text, &Tree::new(&paths.root));
    lines.extend(section.records.iter().map(|r| r.line()));
    lines.extend(check_id_contiguity(&text).iter().map(|r| r.line()));
    let ledger_label = relative(&paths.root, &paths.ledger);
    let ledger_text = read_optional(&paths.ledger)?;
    lines.extend(
        check_ledger(&ledger_label, ledger_text.as_deref())
            .iter()
            .map(|r| r.line()),
    );
    let written = check_dag_section(paths, &text, &rel, write, lines)?;
    Ok(Joined {
        rows: section.rows,
        written,
    })
}

/// The D records. A plain run compares against the committed derivation (D5); `--write` skips
/// the comparison and writes the derivation instead. Returns whether a derivation was written.
fn check_dag_section(
    paths: &Paths,
    text: &str,
    rel: &str,
    write: bool,
    lines: &mut Vec<String>,
) -> Result<bool, String> {
    let today = today_default()?;
    if write {
        return write_derivation(paths, text, rel, &today, lines);
    }
    let committed = match &paths.out {
        Some(path) => Some((relative(&paths.root, path), read_optional(path)?)),
        None => None,
    };
    let derived = committed.as_ref().map(|(rel, committed)| Derived {
        rel: rel.as_str(),
        committed: committed.as_deref(),
    });
    lines.extend(
        check_dag(text, rel, derived, &today)
            .iter()
            .map(|r| r.line()),
    );
    Ok(false)
}

/// `--write`: the DAG records without D5, then the derivation to the `--out` path when the §12
/// table derives. Returns whether it was written.
fn write_derivation(
    paths: &Paths,
    text: &str,
    rel: &str,
    today: &str,
    lines: &mut Vec<String>,
) -> Result<bool, String> {
    lines.extend(check_dag(text, rel, None, today).iter().map(|r| r.line()));
    let (Some(path), Ok(doc)) = (&paths.out, derived_text(text, rel)) else {
        return Ok(false);
    };
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    std::fs::write(path, doc).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(true)
}

/// `path` relative to `root` as the report names it, else the path as given.
fn relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .map_or_else(|_| py_path_str(path), py_path_str)
}

/// The path made absolute against the working directory, as Python's `abspath` does.
fn absolute(path: &Path) -> Result<PathBuf, String> {
    std::path::absolute(path).map_err(|e| format!("{}: {e}", path.display()))
}
