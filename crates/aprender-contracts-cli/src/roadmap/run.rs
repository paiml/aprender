//! The `pv-roadmap` command line: `fragments`, `diff`, `merge`, `trim`. Stdout and exit codes
//! match the Python scripts this replaces (`roadmap_{fragments,diff,merge}.py`, `roadmap_trim.py`).

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use super::diff::{join_lines, read_source, render_trim, run_check, UsageError};
use super::fragments::{
    self, adopt, aggregate, changed_entries, entries_dir, read_fragments, roadmap_path,
};
use super::fragments_selftest::selftest;
use super::merge::merge;
use super::merge_selftest::run_selftest;
use super::{repo_root, split_entries};

const DEFAULT_FILE: &str = "docs/roadmaps/roadmap.yaml";

/// Parsed `--flag value` / `--flag` / positional arguments.
#[derive(Default)]
struct Args {
    pos: Vec<String>,
    opts: Vec<(String, String)>,
    flags: Vec<String>,
}

impl Args {
    fn parse(argv: &[String], valued: &[&str], bools: &[&str]) -> Result<Self, String> {
        let mut a = Self::default();
        let mut it = argv.iter();
        while let Some(x) = it.next() {
            if let Some((k, v)) = x.split_once('=').filter(|(k, _)| valued.contains(k)) {
                a.opts.push((k.to_string(), v.to_string()));
            } else if valued.contains(&x.as_str()) {
                let v = it
                    .next()
                    .ok_or_else(|| format!("argument {x}: expected one argument"))?;
                a.opts.push((x.clone(), v.clone()));
            } else if bools.contains(&x.as_str()) {
                a.flags.push(x.clone());
            } else if x.starts_with("--") {
                return Err(format!("unrecognized arguments: {x}"));
            } else {
                a.pos.push(x.clone());
            }
        }
        Ok(a)
    }

    fn opt(&self, k: &str) -> Option<&str> {
        self.opts
            .iter()
            .rev()
            .find(|(n, _)| n == k)
            .map(|(_, v)| v.as_str())
    }

    fn flag(&self, k: &str) -> bool {
        self.flags.iter().any(|f| f == k)
    }
}

fn usage_err(prog: &str, msg: &str) -> i32 {
    eprintln!("usage: {prog} ...\n{prog}: error: {msg}");
    2
}

fn read(path: &Path, what: &str) -> Option<String> {
    match std::fs::read_to_string(path) {
        Ok(t) => Some(t),
        Err(e) => {
            eprintln!("FAIL {what} is unreadable ({e}) -- this box cannot judge");
            None
        }
    }
}

fn paths(a: &Args) -> (PathBuf, PathBuf) {
    let roadmap = a.opt("--roadmap").map_or_else(roadmap_path, PathBuf::from);
    if let Some(e) = a.opt("--entries") {
        return (roadmap, PathBuf::from(e));
    }
    if a.opt("--roadmap").is_some() {
        let abs = std::path::absolute(&roadmap).unwrap_or_else(|_| roadmap.clone());
        let dir = abs
            .parent()
            .map_or_else(|| PathBuf::from("entries"), |p| p.join("entries"));
        return (roadmap, dir);
    }
    (roadmap, entries_dir())
}

fn emit_aggregate(
    a: &Args,
    base: &str,
    out: &str,
    frags: &[(String, String)],
    roadmap: &Path,
) -> i32 {
    if a.flag("--check") {
        let (rc, msg) = fragments::check(base, out, frags);
        eprint!("{msg}");
        return rc;
    }
    if a.flag("--write") {
        if let Err(e) = std::fs::write(roadmap, out) {
            eprintln!("FAIL {e}");
            return 1;
        }
        eprintln!(
            "aggregate: {} base + {} fragment(s) -> {}",
            split_entries(base).1.len(),
            frags.len(),
            roadmap.display()
        );
        return 0;
    }
    let _ = std::io::stdout().write_all(out.as_bytes());
    0
}

fn cmd_aggregate(a: &Args, roadmap: &Path, dir: &Path) -> i32 {
    let Some(base) = read(roadmap, &roadmap.display().to_string()) else {
        return 2;
    };
    let frags = read_fragments(dir);
    match aggregate(&base, &frags) {
        Ok(out) => emit_aggregate(a, &base, &out, &frags, roadmap),
        Err(e) => {
            eprintln!("FAIL {e}");
            1
        }
    }
}

fn cmd_changed(a: &Args) -> i32 {
    let (Some(b), Some(h)) = (a.opt("--base"), a.opt("--head")) else {
        eprintln!("changed: --base and --head are both required");
        return 2;
    };
    let (Some(base), Some(head)) = (read(Path::new(b), b), read(Path::new(h), h)) else {
        return 2;
    };
    let (pre, verdicts) = changed_entries(&base, &head);
    let mut out = String::new();
    if pre {
        out.push_str("PREAMBLE\t(the header outside every entry)\n");
    }
    for (v, e) in verdicts {
        out.push_str(&format!("{v}\t{e}\n"));
    }
    let _ = std::io::stdout().write_all(out.as_bytes());
    0
}

fn cmd_adopt(ids: &[String], roadmap: &Path, dir: &Path) -> i32 {
    if ids.is_empty() {
        eprintln!("adopt: name at least one entry id");
        return 2;
    }
    let result = adopt(ids, roadmap, dir).and_then(|written| {
        let base = std::fs::read_to_string(roadmap).map_err(|e| e.to_string())?;
        let frags = read_fragments(dir);
        let out = aggregate(&base, &frags)?;
        Ok((written, frags.len(), out))
    });
    match result {
        Ok((written, n, out)) => {
            if let Err(e) = std::fs::write(roadmap, out) {
                eprintln!("FAIL {e}");
                return 1;
            }
            eprintln!(
                "adopt: {} -> {}/ ; {} regenerated from {n} fragment(s)",
                written.join(", "),
                dir.display(),
                roadmap.display()
            );
            0
        }
        Err(e) => {
            eprintln!("FAIL {e}");
            1
        }
    }
}

fn fragments_main(argv: &[String]) -> i32 {
    let valued = ["--roadmap", "--entries", "--base", "--head"];
    let bools = ["--write", "--check", "--selftest"];
    let a = match Args::parse(argv, &valued, &bools) {
        Ok(a) => a,
        Err(e) => return usage_err("pv-roadmap fragments", &e),
    };
    if a.flag("--selftest") {
        return selftest();
    }
    let (roadmap, dir) = paths(&a);
    let cmd = a.pos.first().map(String::as_str);
    match cmd {
        Some("changed") => cmd_changed(&a),
        Some("adopt") => cmd_adopt(&a.pos[1..], &roadmap, &dir),
        Some("aggregate") => cmd_aggregate(&a, &roadmap, &dir),
        Some(other) => usage_err(
            "pv-roadmap fragments",
            &format!("argument cmd: invalid choice: '{other}'"),
        ),
        None => {
            eprintln!("usage: pv-roadmap fragments {{aggregate,changed,adopt}} ...");
            2
        }
    }
}

fn diff_check(a: &Args) -> Result<i32, UsageError> {
    let file = a.opt("--file").unwrap_or(DEFAULT_FILE);
    let (Some(b), Some(h)) = (a.opt("--base"), a.opt("--head")) else {
        return Ok(usage_err(
            "pv-roadmap diff check",
            "the following arguments are required: --base, --head",
        ));
    };
    let (base, head) = (read_source(b, file)?, read_source(h, file)?);
    let (code, lines) = run_check(&base, &head)?;
    let _ = std::io::stdout().write_all(join_lines(&lines).as_bytes());
    Ok(code)
}

fn diff_trim(a: &Args) -> Result<i32, UsageError> {
    let file = a.opt("--file").unwrap_or(DEFAULT_FILE);
    let Some(b) = a.opt("--base") else {
        return Ok(usage_err(
            "pv-roadmap diff trim",
            "the following arguments are required: --base",
        ));
    };
    let base = read_source(b, file)?;
    if !Path::new(file).is_file() {
        return Err(UsageError(format!(
            "trim's head is the working file, not found: {file}"
        )));
    }
    let head = std::fs::read_to_string(file)
        .map_err(|e| UsageError(format!("cannot read {file}: {e}")))?;
    let result = render_trim(&base, &head);
    if a.flag("--write") {
        std::fs::write(file, result)
            .map_err(|e| UsageError(format!("cannot write {file}: {e}")))?;
    } else {
        let _ = std::io::stdout().write_all(result.as_bytes());
    }
    Ok(0)
}

fn diff_main(argv: &[String]) -> i32 {
    let Some((sub, rest)) = argv.split_first() else {
        return usage_err(
            "pv-roadmap diff",
            "the following arguments are required: cmd",
        );
    };
    let a = match Args::parse(rest, &["--base", "--head", "--file"], &["--write"]) {
        Ok(a) => a,
        Err(e) => return usage_err("pv-roadmap diff", &e),
    };
    let r = match sub.as_str() {
        "check" => diff_check(&a),
        "trim" => diff_trim(&a),
        other => {
            return usage_err(
                "pv-roadmap diff",
                &format!("argument cmd: invalid choice: '{other}'"),
            )
        }
    };
    r.unwrap_or_else(|e| {
        eprintln!("USAGE ERROR: {}", e.0);
        2
    })
}

fn trim_main(argv: &[String]) -> i32 {
    let a = match Args::parse(argv, &["--base", "--file"], &[]) {
        Ok(a) => a,
        Err(e) => return usage_err("pv-roadmap trim", &e),
    };
    let base = match a.opt("--base") {
        Some(b) => b.to_string(),
        None => match default_base() {
            Ok(b) => b,
            Err(e) => {
                eprintln!("USAGE ERROR: {}", e.0);
                return 2;
            }
        },
    };
    let file = a.opt("--file").unwrap_or(DEFAULT_FILE);
    let args = ["trim", "--base", base.as_str(), "--file", file, "--write"].map(String::from);
    diff_main(&args)
}

fn default_base() -> Result<String, UsageError> {
    let out = Command::new("git")
        .args(["merge-base", "origin/main", "HEAD"])
        .output()
        .map_err(|e| UsageError(format!("git merge-base failed to run: {e}")))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(UsageError(format!(
            "git merge-base origin/main HEAD failed: {}",
            err.trim()
        )));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn merge_main(argv: &[String]) -> i32 {
    let a = match Args::parse(argv, &["--out"], &["--selftest", "--self-test"]) {
        Ok(a) => a,
        Err(e) => return usage_err("pv-roadmap merge", &e),
    };
    if a.flag("--selftest") || a.flag("--self-test") {
        return run_selftest(&repo_root().join("scripts").join("check_roadmap_sorted.sh"));
    }
    let [b, o, t] = a.pos.as_slice() else {
        eprintln!("usage: pv-roadmap merge BASE OURS THEIRS [--out FILE] | --selftest");
        return 2;
    };
    let mut txts = Vec::new();
    for f in [b, o, t] {
        match std::fs::read_to_string(f) {
            Ok(s) => txts.push(s),
            Err(e) => {
                eprintln!("{f}: {e}");
                return 1;
            }
        }
    }
    let mutate = std::env::var("ROADMAP_MERGE_MUTATE").as_deref() == Ok("1");
    match merge(&txts[0], &txts[1], &txts[2], mutate) {
        Ok(out) => match std::fs::write(a.opt("--out").unwrap_or(o), out) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("{e}");
                1
            }
        },
        Err(c) => {
            eprintln!("conflict: {}", c.0);
            1
        }
    }
}

/// Entry point: `pv-roadmap <fragments|diff|merge|trim> ...`.
#[must_use]
pub fn main_with(argv: &[String]) -> i32 {
    let Some((sub, rest)) = argv.split_first() else {
        eprintln!("usage: pv-roadmap {{fragments|diff|merge|trim}} ...");
        return 2;
    };
    match sub.as_str() {
        "fragments" => fragments_main(rest),
        "diff" => diff_main(rest),
        "merge" => merge_main(rest),
        "trim" => trim_main(rest),
        other => {
            eprintln!("pv-roadmap: unknown subcommand '{other}'");
            2
        }
    }
}
