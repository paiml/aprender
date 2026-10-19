//! `aprender-rollup plan` prints the rollup it would build (read-only).
//! `aprender-rollup build <dir>` assembles it locally in worktree `<dir>`; it never pushes or closes.

use aprender_rollup::mg::{self, Verdict};
use aprender_rollup::{build, close, gh_json, outcome, report, select, Config};
use std::process::{Command, ExitCode};

fn candidates(cfg: &Config) -> Result<Vec<aprender_rollup::Candidate>, String> {
    let o = Command::new("gh")
        .args([
            "pr",
            "list",
            "--state",
            "open",
            "--limit",
            "100",
            "--json",
            gh_json::GH_FIELDS,
        ])
        .args(cfg.repo_args())
        .output()
        .map_err(|e| format!("cannot run gh: {e}"))?;
    if !o.status.success() {
        return Err(format!(
            "gh failed: {}",
            String::from_utf8_lossy(&o.stderr).trim()
        ));
    }
    gh_json::parse_prs(&String::from_utf8_lossy(&o.stdout))
}

/// Trimmed stdout of a successful `git` run in `dir`.
fn git_out(dir: &str, args: &[String]) -> Option<String> {
    let o = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .ok()?;
    o.status
        .success()
        .then(|| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

fn git(dir: &str, args: &[String]) -> bool {
    Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .status()
        .is_ok_and(|s| s.success())
}

/// Merge each picked PR; a conflicting PR is aborted and ejected (stays open).
fn assemble(base: &str, dir: &str, picked: &[u64], heads: &[String]) -> Vec<u64> {
    let mut merged = Vec::new();
    for (n, want) in picked.iter().zip(heads) {
        let one = build::steps(base, &[*n]);
        // steps 2..4 are this PR's fetch + merge; branch setup already ran for the full set
        if !git(dir, &one[2]) || git_out(dir, &build::head_step()).as_deref() != Some(want.as_str())
        {
            eprintln!("eject #{n}: head moved since it was judged green and approved");
        } else if git(dir, &one[3]) {
            merged.push(*n);
        } else {
            let _ = git(dir, &build::abort_step());
            eprintln!("eject #{n}: conflicts with the rollup so far");
        }
    }
    merged
}

fn run(cfg: &Config, cmd: &str, dir: Option<&str>) -> Result<(), String> {
    let sel = select(cfg, &candidates(cfg)?);
    if cmd == "plan" {
        print!("{}", report::plan_text(cfg, &sel));
        return Ok(());
    }
    let dir = dir.ok_or("usage: aprender-rollup build <dir>")?;
    let setup = build::steps(&cfg.base, &sel.picked);
    if !git(dir, &setup[0]) || !git(dir, &setup[1]) {
        return Err("cannot create rollup branch".to_string());
    }
    let merged = assemble(&cfg.base, dir, &sel.picked, &sel.heads);
    println!(
        "{}\n{}",
        build::branch_name(&sel.picked),
        build::message(&merged)
    );
    Ok(())
}

fn gh(args: &[String]) -> Result<String, String> {
    let o = Command::new("gh")
        .args(args)
        .output()
        .map_err(|e| format!("cannot run gh: {e}"))?;
    if !o.status.success() {
        return Err(format!(
            "gh failed: {}",
            String::from_utf8_lossy(&o.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&o.stdout).into_owned())
}

/// `verify <branch> <head-sha> <sha|-> <n,n,..> [--apply]`: read the merge-group verdict for the
/// rollup branch. Green + SHA on main closes each constituent with the SHA; red prints the
/// two halves to rebuild and test; pending touches nothing. Without `--apply`, dry run.
fn verify(cfg: &Config, a: &[String]) -> Result<(), String> {
    let [branch, head, sha, list, rest @ ..] = a else {
        return Err("usage: aprender-rollup verify <branch> <sha|-> <n,n,..> [--apply]".into());
    };
    let picked: Vec<u64> = list
        .split(',')
        .map(|s| {
            s.trim_start_matches('#')
                .parse()
                .map_err(|_| format!("bad PR {s}"))
        })
        .collect::<Result<_, _>>()?;
    let runs = gh(&[mg::run_list_args(branch), cfg.repo_args()].concat())?;
    let v = mg::parse_verdict(&runs, head)?;
    println!("verdict {v:?}");
    let on_main = |s: &str| git_ok(&mg::ancestor_args(s));
    let sha = (sha != "-" && on_main(sha)).then_some(sha.as_str());
    match v {
        Verdict::Pending => {}
        Verdict::Red => match aprender_rollup::split(&picked) {
            Some((x, y)) => println!("bisect: rebuild halves {x:?} and {y:?}"),
            None => println!("culprit #{}", picked[0]),
        },
        Verdict::Green if sha.is_none() => {
            println!("green, waiting for the rollup commit on origin/main; nothing posted");
        }
        Verdict::Green => {
            for n in &picked {
                let args = [close::gh_args(*n, &outcome(true, sha)), cfg.repo_args()].concat();
                println!("#{n}: gh {}", args.join(" "));
                if rest.iter().any(|r| r == "--apply") {
                    gh(&args)?;
                }
            }
        }
    }
    Ok(())
}

/// Run `git` in the current checkout; true on exit 0.
fn git_ok(args: &[String]) -> bool {
    Command::new("git")
        .args(args)
        .status()
        .is_ok_and(|s| s.success())
}

/// Strip `--config FILE` and `--repo NAME` (the table to read) from the arguments.
fn config(args: &mut Vec<String>) -> Result<Config, String> {
    let mut take = |flag: &str| -> Option<String> {
        let i = args.iter().position(|a| a == flag)?;
        let v = args.get(i + 1)?.clone();
        args.drain(i..=i + 1);
        Some(v)
    };
    match (take("--config"), take("--repo")) {
        (Some(file), Some(name)) => {
            let text = std::fs::read_to_string(&file).map_err(|e| format!("{file}: {e}"))?;
            Config::from_toml(&text, &name)
        }
        (None, None) => Ok(Config::default()),
        _ => Err("--config FILE and --repo NAME go together".to_string()),
    }
}

fn main() -> ExitCode {
    let mut args: Vec<String> = std::env::args().collect();
    let cfg = match config(&mut args) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("not_measured: {e}");
            return ExitCode::FAILURE;
        }
    };
    let cmd = args.get(1).map(String::as_str).unwrap_or("");
    let res = match cmd {
        "plan" | "build" => run(&cfg, cmd, args.get(2).map(String::as_str)),
        "verify" => verify(&cfg, &args[2..]),
        _ => {
            eprintln!("usage: aprender-rollup [--config FILE --repo NAME] plan | build <dir> | verify <branch> <head-sha> <sha|-> <n,..> [--apply]");
            return ExitCode::from(2);
        }
    };
    match res {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("not_measured: {e}");
            ExitCode::FAILURE
        }
    }
}
