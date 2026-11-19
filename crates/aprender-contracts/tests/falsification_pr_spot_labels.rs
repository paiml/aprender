//! Spot (burst) pool routing, judged on the PARSED workflow documents, never on raw lines.
//!
//! Every `.yml` and `.yaml` under `.github/workflows/` is parsed with the workspace's own YAML
//! crate (`serde_yaml`, which rejects duplicate keys), so quoted keys, escapes, flow style and
//! anchors resolve before anything is judged. Rules, for EVERY job of EVERY workflow, whatever
//! its trigger:
//!  (a) GitHub routes a job to a seat when the job's labels are a SUBSET of the seat's labels, so
//!      a job's resolved `runs-on` must hold a label no spot seat carries (or a runner group on the
//!      committed non-spot list). `[self-hosted]`, `[self-hosted, Linux]`, `[self-hosted, Linux,
//!      pool2, ffmpeg]`, `{labels: [...]}` and matrix options are all judged by that rule.
//!  (b) the only jobs that may route to the spot pool are the three ci.yml aggregators, and only
//!      with their one canonical expression. They must have permissions exactly `contents: read`,
//!      no scalar naming secrets, a token or id-token, and no checkout, cache or artifact step that
//!      is not skipped on `pull_request`.
//!  (c) a `cond && fromJSON('A') || fromJSON('B')` expression is judged on BOTH branches; any other
//!      expression, and any reusable call this file cannot read or does not allowlist, is red.
//!  (d) ci.yml's top-level `permissions` may be at most `contents: read`, and it names no secrets.
//!  (e) cannot judge = red: parse failure, a duplicate key, a merge key or tag, a second document,
//!      a non-mapping document or `jobs`, a job with neither `runs-on` nor `uses`, an unmodelled
//!      `runs-on` shape, an empty or missing directory.
//!  (f) the guard is wired, unmasked, into `mg-reuse-self-test`, the ci.yml job that runs on
//!      pull_request, merge_group AND push, and that job is a need of `gate`.

use serde_yaml::{Mapping, Value};
use std::fs;
use std::path::{Path, PathBuf};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

const SPOT: &str = r#"["self-hosted","burst-x64","burst-public"]"#;
const WT: &str = r#"["self-hosted","Linux","clean-room"]"#;
const X64: &str = r#"["self-hosted","Linux","X64","clean-room"]"#;

fn canonical(job: &str) -> Option<String> {
    let clean = match job {
        "workspace-test" => WT,
        "ci-gate" | "gate" => X64,
        _ => return None,
    };
    Some(format!(
        "${{{{ (github.event_name == 'pull_request' && vars.PR_SPOT_OPTIN == '1') && fromJSON('{SPOT}') || fromJSON('{clean}') }}}}"
    ))
}

/// Labels a spot seat carries, as a committed dated snapshot (2026-11-19), not a live read: re-measure
/// before trusting it for a new seat family. `x64` and `burst-public` are added on purpose (an arch
/// label may appear on a seat, and `burst-public` is the PR-only spot label). Compared lower-case.
/// Any label of the generic pool-class form (letters then digits, e.g. `pool2`) also counts as seat-carried, so a
/// class label never makes a job look safe.
const SPOT_SEAT_LABELS: [&str; 6] = [
    "self-hosted",
    "linux",
    "x64",
    "ffmpeg",
    "burst-x64",
    "burst-public",
];

fn seat_carried(l: &str) -> bool {
    let l = l.trim().to_lowercase();
    let digits = l.trim_start_matches(|c: char| c.is_ascii_lowercase());
    let class_label =
        digits.len() < l.len() && !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit());
    class_label || SPOT_SEAT_LABELS.contains(&l.as_str())
}

/// Runner groups known to hold no spot seat (same dated snapshot). Any other group is red (cannot judge).
const NON_SPOT_GROUPS: [&str; 1] = ["Default"];

/// EXTERNAL reusable calls this repo makes: the exact `uses` string. Anything else is red.
const EXTERNAL_CALLS: [&str; 3] = [
    "paiml/.github/.github/workflows/runner-smells.yml@d768b8ac5abfa7d0038470fd38023a15323ac0aa",
    "paiml/.github/.github/workflows/pr-gate.yml@8289382dfd2852714a7fa14868a7545c91cd98e6",
    "paiml/.github/.github/workflows/fork-pr-closer.yml@830147e62523b292823e3134943de10265c4569e",
];
/// The one `with.runs_on` an external call may pass, and which call may pass it.
const SMELLS_RUNS_ON: &str = X64;

/// Workflow keys are matched case-insensitively, as Actions does (`If:` is `if:`). Job ids use `Mapping::get`.
fn key(m: &Mapping, k: &str) -> Option<Value> {
    m.iter()
        .find(|(n, _)| n.as_str().is_some_and(|n| n.eq_ignore_ascii_case(k)))
        .map(|(_, v)| v.clone())
}

/// Every scalar (keys and values) under `v`; a merge key or a tag cannot be judged.
fn scalars(v: &Value, out: &mut Vec<String>, why: &mut Vec<String>) {
    match v {
        Value::String(s) => out.push(s.clone()),
        Value::Number(n) => out.push(n.to_string()),
        Value::Bool(b) => out.push(b.to_string()),
        Value::Null => {}
        Value::Sequence(s) => s.iter().for_each(|x| scalars(x, out, why)),
        Value::Mapping(m) => {
            for (k, x) in m {
                if k.as_str() == Some("<<") {
                    why.push("merge key `<<` is not modeled".into());
                }
                scalars(k, out, why);
                scalars(x, out, why);
            }
        }
        Value::Tagged(_) => why.push("tagged value is not modeled".into()),
    }
}

fn all_scalars(v: &Value, why: &mut Vec<String>) -> Vec<String> {
    let mut out = Vec::new();
    scalars(v, &mut out, why);
    out
}

/// A JSON-looking string is the list or map it spells; anything else stays a string.
fn as_json(v: &Value) -> Value {
    match v {
        Value::String(s) if matches!(s.trim_start().chars().next(), Some('[' | '{')) => {
            serde_yaml::from_str(s).unwrap_or_else(|_| v.clone())
        }
        other => other.clone(),
    }
}

/// Ok when this label list holds one literal label no spot seat carries (a job runs only on a seat
/// whose labels include ALL of the job's labels). An unresolved `${{` element proves nothing.
fn leaves_spot(v: &Value) -> Result<(), String> {
    let ls: Vec<String> = match v {
        Value::String(s) => vec![s.clone()],
        Value::Sequence(q) => q
            .iter()
            .map(|x| x.as_str().map(str::to_string))
            .collect::<Option<Vec<_>>>()
            .ok_or("a label is not a plain string")?,
        _ => return Err("runs-on shape is not modeled".into()),
    };
    if ls.is_empty() {
        return Err("empty label list".into());
    }
    let outside = ls.iter().any(|l| !l.contains("${{") && !seat_carried(l));
    if outside {
        Ok(())
    } else {
        Err(format!(
            "labels {ls:?} are a subset of the spot seat labels, or unresolved"
        ))
    }
}

fn matrix_options(j: &Mapping, k: &str) -> Result<Vec<Value>, String> {
    let Some(Value::Mapping(strategy)) = key(j, "strategy") else {
        return Err("matrix reference without a literal strategy".into());
    };
    let Some(Value::Mapping(matrix)) = key(&strategy, "matrix") else {
        return Err("strategy.matrix is not a literal mapping".into());
    };
    let mut out = Vec::new();
    match key(&matrix, k) {
        None => {}
        Some(Value::Sequence(v)) => out.extend(v),
        Some(_) => return Err(format!("matrix.{k} is not a literal list")),
    }
    match key(&matrix, "include") {
        None => {}
        Some(Value::Sequence(items)) => {
            for it in items {
                match it {
                    Value::Mapping(m) => out.extend(key(&m, k)),
                    _ => return Err("matrix include entry is not a mapping".into()),
                }
            }
        }
        Some(_) => return Err("matrix include is not a literal list".into()),
    }
    if out.is_empty() {
        return Err(format!("matrix.{k} has no literal options"));
    }
    Ok(out)
}

fn ident(k: &str) -> bool {
    !k.is_empty() && k.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// `cond && fromJSON('A') || fromJSON('B')` -> (A, B), both literal JSON without a quote or `${{`.
fn ternary(inner: &str) -> Option<(&str, &str)> {
    let (left, right) = inner.rsplit_once(" || fromJSON('")?;
    let b = right.strip_suffix("')")?;
    let (cond, a) = left.rsplit_once(" && fromJSON('")?;
    let a = a.strip_suffix("')")?;
    let plain = |s: &str| !s.contains('\'') && !s.contains("${{");
    (!cond.trim().is_empty() && plain(a) && plain(b)).then_some((a, b))
}

fn resolve_expr(j: &Mapping, s: &str) -> Result<(), String> {
    let inner = s
        .trim()
        .strip_prefix("${{")
        .and_then(|r| r.strip_suffix("}}"))
        .map(str::trim)
        .ok_or_else(|| format!("runs-on expression cannot be resolved: {s}"))?;
    let m = inner
        .strip_prefix("fromJSON(matrix.")
        .and_then(|r| r.strip_suffix(')'))
        .or_else(|| inner.strip_prefix("matrix."))
        .filter(|k| ident(k));
    if let Some(k) = m {
        return matrix_options(j, k)?
            .iter()
            .try_for_each(|o| leaves_spot(&as_json(o)));
    }
    if let Some((a, b)) = ternary(inner) {
        for br in [a, b] {
            leaves_spot(&as_json(&Value::String(br.to_string())))?;
        }
        return Ok(());
    }
    Err(format!("runs-on expression cannot be resolved: {s}"))
}

/// Ok when this job can NOT land on a spot seat.
fn reaches_spot(j: &Mapping, ro: &Value) -> Result<(), String> {
    match ro {
        Value::Mapping(m) => {
            if let Some(k) = m
                .keys()
                .find(|k| !matches!(k.as_str(), Some("group" | "labels")))
            {
                return Err(format!("runs-on key {k:?} is not modeled"));
            }
            match (key(m, "group"), key(m, "labels")) {
                (Some(Value::String(g)), _) if NON_SPOT_GROUPS.contains(&g.as_str()) => Ok(()),
                (Some(g), _) => Err(format!("runner group {g:?} is not on the non-spot list")),
                (None, Some(l)) => leaves_spot(&as_json(&l)),
                (None, None) => Err("empty runs-on mapping".into()),
            }
        }
        Value::String(s) if s.contains("${{") && !s.trim().starts_with('[') => resolve_expr(j, s),
        other => leaves_spot(other),
    }
}

fn is_inherit(j: &Mapping) -> bool {
    matches!(key(j, "secrets"), Some(Value::String(s)) if s.trim().eq_ignore_ascii_case("inherit"))
}

fn judge_external(ctx: &str, u: &str, j: &Mapping, why: &mut Vec<String>) {
    let bare = u.split(" #").next().unwrap_or(u).trim();
    if !EXTERNAL_CALLS.contains(&bare) {
        return why.push(format!(
            "{ctx}: external or unresolvable reusable call is not on the allowlist: {u}"
        ));
    }
    if key(j, "secrets").is_some() {
        why.push(format!("{ctx}: passes secrets to an external call: {u}"));
    }
    if let Some(Value::Mapping(w)) = key(j, "with") {
        for (k, x) in &w {
            let k = k.as_str().unwrap_or("");
            let l = k.to_lowercase();
            let routing = ["runs", "label", "runner", "group", "pool"]
                .iter()
                .any(|n| l.contains(n));
            let exact =
                bare == EXTERNAL_CALLS[0] && l == "runs_on" && x.as_str() == Some(SMELLS_RUNS_ON);
            if routing && !exact {
                why.push(format!(
                    "{ctx}: reusable call passes a runs-on or pool value: {k}"
                ));
            }
        }
    }
}

fn judge_local(ctx: &str, dir: &Path, rel: &str, j: &Mapping, depth: usize, why: &mut Vec<String>) {
    let ctx = format!("{ctx} via local call ./{rel}");
    if is_inherit(j) {
        why.push(format!(
            "{ctx}: secrets: inherit to a call not on an allowlist"
        ));
    }
    let name = rel.strip_prefix(".github/workflows/").unwrap_or("");
    let plain = !name.is_empty()
        && !name.contains('/')
        && !name.contains("..")
        && (name.ends_with(".yml") || name.ends_with(".yaml"));
    if !plain || depth >= 3 {
        return why.push(format!(
            "{ctx}: not a plain workflow file here (or nested too deep), cannot judge"
        ));
    }
    let mut inner = Vec::new();
    let top = parse_file(&dir.join(name), name, &mut inner);
    why.extend(inner.into_iter().map(|w| format!("{ctx}: {w}")));
    let Some(Value::Mapping(top)) = top else {
        return why.push(format!(
            "{ctx}: callee is missing or not a mapping, cannot judge"
        ));
    };
    let Some(Value::Mapping(jobs)) = key(&top, "jobs") else {
        return why.push(format!("{ctx}: callee jobs missing, cannot judge"));
    };
    let with = match key(j, "with") {
        Some(Value::Mapping(w)) => w,
        _ => Mapping::new(),
    };
    for (n, cj) in &jobs {
        let cctx = format!("{ctx}: job {}", n.as_str().unwrap_or("?"));
        let Value::Mapping(cm) = cj else {
            why.push(format!("{cctx} is not a mapping, cannot judge"));
            continue;
        };
        match key(cm, "runs-on") {
            None if key(cm, "uses").is_some() => judge_call(&cctx, dir, cm, depth + 1, why),
            None => why.push(format!("{cctx}: no runs-on and no uses, cannot judge")),
            Some(ro) => {
                let input = ro.as_str().and_then(|s| {
                    s.trim()
                        .strip_prefix("${{ inputs.")
                        .and_then(|r| r.strip_suffix(" }}"))
                });
                let resolved = match input {
                    Some(k) => key(&with, k).map_or_else(
                        || Value::String(format!("${{{{ inputs.{k} }}}}")),
                        |v| as_json(&v),
                    ),
                    None => ro,
                };
                if let Err(e) = reaches_spot(cm, &resolved) {
                    why.push(format!("{cctx}: can land on a spot seat: {e}"));
                }
            }
        }
    }
}

fn judge_call(ctx: &str, dir: &Path, j: &Mapping, depth: usize, why: &mut Vec<String>) {
    let Some(uses) = key(j, "uses") else { return };
    match uses.as_str() {
        None => why.push(format!("{ctx}: `uses` is not a string, cannot judge")),
        Some(u) => match u.strip_prefix("./") {
            Some(local) => judge_local(ctx, dir, local, j, depth, why),
            None => judge_external(ctx, u, j, why),
        },
    }
}

/// A step that touches the repo, a cache or an artifact needs a token: on a moved job it must be
/// skipped on `pull_request`.
fn step_needs_pr_skip(step: &Mapping) -> bool {
    let uses = key(step, "uses")
        .and_then(|u| u.as_str().map(str::to_lowercase))
        .unwrap_or_default();
    [
        "actions/checkout",
        "actions/cache",
        "actions/download-artifact",
        "actions/upload-artifact",
    ]
    .iter()
    .any(|p| uses.starts_with(p))
}

fn check_moved(ctx: &str, j: &Mapping, canon: &str, all: &[String], why: &mut Vec<String>) {
    match key(j, "runs-on") {
        Some(Value::String(s)) if s == canon => {}
        other => why.push(format!(
            "{ctx}: runs-on is not its one canonical value: {other:?}"
        )),
    }
    let want: Value = serde_yaml::from_str("contents: read").unwrap();
    if key(j, "permissions").as_ref() != Some(&want) {
        why.push(format!("{ctx}: permissions are not exactly contents: read"));
    }
    for s in all {
        let l = s.to_lowercase();
        if ["secrets", "id-token", "token", "write"]
            .iter()
            .any(|n| l.contains(n))
        {
            why.push(format!(
                "{ctx}: names secrets, a token, id-token or write: {s}"
            ));
        }
    }
    let steps = match key(j, "steps") {
        Some(Value::Sequence(s)) => s,
        _ => return why.push(format!("{ctx}: steps are not a literal list, cannot judge")),
    };
    for st in steps {
        let Value::Mapping(st) = st else {
            why.push(format!("{ctx}: a step is not a mapping, cannot judge"));
            continue;
        };
        if step_needs_pr_skip(&st) {
            let cond = key(&st, "if")
                .and_then(|c| c.as_str().map(str::to_string))
                .unwrap_or_default();
            if !cond.contains("github.event_name != 'pull_request'") || cond.contains("||") {
                why.push(format!(
                    "{ctx}: a checkout, cache or artifact step runs on pull_request"
                ));
            }
        }
    }
}

fn judge_job(file: &str, dir: &Path, name: &str, job: &Value, is_ci: bool, why: &mut Vec<String>) {
    let ctx = format!("{file}: job {name}");
    let Value::Mapping(j) = job else {
        return why.push(format!("{ctx} is not a mapping, cannot judge"));
    };
    let canon = if is_ci { canonical(name) } else { None };
    let all = all_scalars(job, why);
    for s in &all {
        if s.contains("PR_SPOT_OPTIN") && canon.as_deref() != Some(s.as_str()) {
            why.push(format!(
                "{ctx}: opt-in variable outside its canonical runs-on: {s}"
            ));
        }
    }
    let uses = key(j, "uses").is_some();
    match (&canon, key(j, "runs-on")) {
        (Some(c), _) => check_moved(&ctx, j, c, &all, why),
        (None, Some(ro)) => {
            if let Err(e) = reaches_spot(j, &ro) {
                why.push(format!("{ctx}: can land on a spot seat: {e}"));
            }
        }
        (None, None) if !uses => why.push(format!("{ctx}: no runs-on and no uses, cannot judge")),
        (None, None) => {}
    }
    if uses {
        judge_call(&ctx, dir, j, 0, why);
    }
}

fn permissions_ok(v: &Value) -> bool {
    match v {
        Value::Mapping(m) => m.iter().all(|(pk, pv)| {
            let pv = pv.as_str().unwrap_or("");
            pv == "none" || (pk.as_str() == Some("contents") && pv == "read")
        }),
        _ => false,
    }
}

fn check_workflow_level(base: &str, top: &Mapping, why: &mut Vec<String>) {
    for (k, v) in top {
        let ks = k.as_str().unwrap_or("");
        if ks.eq_ignore_ascii_case("jobs") {
            continue;
        }
        let mut names = all_scalars(v, &mut Vec::new());
        names.push(ks.to_string());
        for s in names
            .iter()
            .filter(|s| s.to_lowercase().contains("secrets"))
        {
            why.push(format!("{base}: workflow level names secrets: {s}"));
        }
        if ks.eq_ignore_ascii_case("permissions") && !permissions_ok(v) {
            why.push(format!(
                "{base}: workflow permissions wider than contents: read"
            ));
        }
    }
}

/// The only keys the guard step may carry. Any other (`shell`, `if`, `continue-on-error`, `uses`, `with`, `env`,
/// in any case) can skip or mask it.
const GUARD_STEP_KEYS: [&str; 3] = ["name", "run", "timeout-minutes"];

/// What a `run` script is judged on. Unlike bash, this reader cannot see two things, so a run block that has either is
/// refused outright (no row can match it): a `${{ }}` expression (GitHub expands it BEFORE bash runs, so even inside a
/// `#` comment it can inject lines), and a line ending in a backslash (bash joins it with no space and keeps the next
/// line's indent; a comment line is never joined, and heredocs and quotes are not modeled).
const UNJUDGEABLE_RUN: &str = "<run text this guard cannot judge: `${{` or a trailing backslash>";

fn run_text_judgeable(run: &str) -> bool {
    // Bash's blanks are ASCII space and tab only. Rust's `trim` also strips U+00A0, U+3000, \x0b, \x0c and \r, which bash
    // reads as word characters, so a line this reader drops as a `#` comment could be live bash. So the run text may hold
    // printable ASCII, newline and tab and nothing else, and only space and tab are ever trimmed.
    run.chars()
        .all(|c| c == '\n' || c == '\t' || c == ' ' || c.is_ascii_graphic())
        && !run.contains("${{")
        && !run.lines().any(|l| blank_trim_end(l).ends_with('\\'))
}

fn is_blank(c: char) -> bool {
    c == ' ' || c == '\t'
}

fn blank_trim_end(l: &str) -> &str {
    l.trim_end_matches(is_blank)
}

/// A `run` script as lines: blank and `#` comment lines dropped, each line trimmed.
fn logical_lines(run: &str) -> Vec<String> {
    if !run_text_judgeable(run) {
        return vec![UNJUDGEABLE_RUN.to_string()];
    }
    run.lines()
        .map(|l| l.trim_matches(is_blank))
        .filter(|t| !t.is_empty() && !t.starts_with('#'))
        .map(str::to_string)
        .collect()
}

fn run_lines(s: &Mapping) -> Vec<String> {
    key(s, "run")
        .and_then(|r| r.as_str().map(logical_lines))
        .unwrap_or_default()
}

/// The guard step's script is an ALLOWLIST: after `set -euo pipefail` only these lines and the one docker command.
/// Any other line (a wrapper, a function, a `set +`, an `exit`, a redefined `reap`) is red.
const GUARD_LINES: [&str; 6] = [
    "cname=\"spot-labels-${GITHUB_RUN_ID}\"",
    "reap() { if docker rm -f \"$cname\" >/dev/null 2>&1; then :; fi; }",
    "reap",
    "trap reap EXIT INT TERM",
    "base=\"/mnt/nvme-raid0/targets/aprender-ci/spot-labels/run-${GITHUB_RUN_ID}\"",
    "mkdir -p \"$base/registry\" \"$base/target\"",
];
/// The one docker command, EXACT up to the image digest: every option, mount and env var is fixed text, so a
/// second image, `-d`, `--entrypoint` (quoted or not), a cargo runner variable or any other extra word is red.
/// Only the 64-hex digest may change (a pin bump). Nothing may follow the test name.
const GUARD_CMD_PREFIX: &str = concat!(
    "timeout --foreground -k 30 900 docker run --rm --init --name \"$cname\" ",
    "--cpus 4 --memory 8g --memory-swap 8g --pids-limit 2048 ",
    "-v \"$GITHUB_WORKSPACE:/w:ro\" -v \"$base/registry:/usr/local/cargo/registry\" -v \"$base/target:/t\" ",
    "-e CARGO_TARGET_DIR=/t -e CARGO_BUILD_JOBS=\"${RUNNER_THREAD_CAP:-4}\" ",
    "-e RUST_TEST_THREADS=\"${RUNNER_THREAD_CAP:-4}\" -e CARGO_INCREMENTAL=0 ",
    "-e CARGO_PROFILE_DEV_DEBUG=0 -e CARGO_PROFILE_TEST_DEBUG=0 -w /w ",
    "localhost:5000/sovereign-ci:stable@sha256:"
);
const GUARD_CMD_TAIL: &str =
    " cargo test --locked -p aprender-contracts --test falsification_pr_spot_labels";

fn guard_cmd_ok(l: &str) -> bool {
    l.strip_prefix(GUARD_CMD_PREFIX).is_some_and(|rest| {
        rest.len() == 64 + GUARD_CMD_TAIL.len()
            && rest
                .get(..64)
                .is_some_and(|d| d.chars().all(|c| c.is_ascii_hexdigit()))
            && rest.get(64..) == Some(GUARD_CMD_TAIL)
    })
}

fn guard_step_ok(s: &Mapping) -> bool {
    let keys_ok = s.iter().all(|(k, _)| {
        k.as_str()
            .is_some_and(|k| GUARD_STEP_KEYS.iter().any(|a| a.eq_ignore_ascii_case(k)))
    });
    let lines = run_lines(s);
    keys_ok
        && lines.first().map(String::as_str) == Some("set -euo pipefail")
        && lines.iter().filter(|l| guard_cmd_ok(l)).count() == 1
        && lines
            .iter()
            .skip(1)
            .all(|l| GUARD_LINES.contains(&l.as_str()) || guard_cmd_ok(l))
}

/// An `env` that can change how bash runs a step (`SHELLOPTS`, `BASH_ENV`, an exported function) is red.
fn env_masks(m: &Mapping) -> bool {
    key(m, "env")
        .and_then(|e| e.as_mapping().cloned())
        .is_some_and(|e| {
            e.iter().any(|(k, _)| {
                k.as_str().is_some_and(|k| {
                    let k = k.to_ascii_uppercase();
                    ["SHELLOPTS", "BASHOPTS", "BASH_ENV", "ENV", "PS4"].contains(&k.as_str())
                        || k.starts_with("BASH_FUNC")
                })
            })
        })
}

/// The `gate` line that fails on a red `mg-reuse-self-test`, whole (so `exit 1` cannot be dropped).
const MST_LINE: &str =
    "[ \"$MST\" = success ] || { echo \"mg-reuse-self-test failed: $MST\"; exit 1; }";

/// The only lines that may run in `gate`'s step before the MST line (comments dropped). An allowlist, so no
/// wrapper, `trap`, `declare`/`export`/`readonly MST=`, `exec`, shadowing function or early `exit` can precede it.
const GATE_PRE_MST_LINES: [&str; 4] = [
    "set -euo pipefail",
    "[ \"${DRAFT:-}\" != true ] || { echo \"gate: draft PR, full CI is not owed until ready_for_review\"; exit 0; }",
    "res() { printf '%s' \"${1:-null}\" | SEC=\"$2\" jq -r '(. // {})[env.SEC].result // \"missing\"'; }",
    "[ \"$WT\" = success ] || { echo \"workspace-test failed: $WT\"; exit 1; }",
];
/// `gate`'s step env, EXACT: these names bound to exactly these expressions, nothing else (a different `DRAFT`, an
/// extra `PATH`, a constant `MST` are all red).
const GATE_STEP_ENV: [(&str, &str); 14] = [
    ("X86", "${{ needs.x86-main.outputs.results }}"),
    ("GRD", "${{ needs.guards.outputs.results }}"),
    ("GRDR", "${{ needs.guards.result }}"),
    ("DET", "${{ needs.determinism.outputs.results }}"),
    ("WT", "${{ needs.workspace-test.result }}"),
    ("EVT", "${{ github.event_name }}"),
    ("MG", "${{ needs.mg-reuse.result }}"),
    ("MG_X86", "${{ needs.mg-reuse.outputs.x86 }}"),
    ("MG_DET", "${{ needs.mg-reuse.outputs.det }}"),
    ("MG_CITE", "${{ needs.mg-reuse.outputs.cite }}"),
    ("ADV", "${{ needs.x86-main-advisories.result }}"),
    ("MST", "${{ needs.mg-reuse-self-test.result }}"),
    ("PRL", "${{ needs.pr-light.result }}"),
    ("DRAFT", "${{ github.event.pull_request.draft }}"),
];
const GATE_STEP_KEYS: [&str; 4] = ["name", "env", "run", "timeout-minutes"];
const GATE_JOB_KEYS: [&str; 6] = [
    "runs-on",
    "permissions",
    "timeout-minutes",
    "needs",
    "if",
    "steps",
];

/// `gate` runs even when a need failed (`if: always()`), has no key outside the job allowlist, and its FIRST step
/// (nothing runs before it) has only allowlisted keys and env names, runs exactly the allowlisted lines and then the
/// whole MST line, and binds `MST` to the job's result.
fn gate_enforces_mst(gate: &Mapping) -> bool {
    let job_ok = gate
        .iter()
        .all(|(k, _)| k.as_str().is_some_and(|k| GATE_JOB_KEYS.contains(&k)))
        && key(gate, "if").as_ref().and_then(Value::as_str) == Some("always()");
    let Some(Value::Sequence(steps)) = key(gate, "steps") else {
        return false;
    };
    let Some(Value::Mapping(s)) = steps.first() else {
        return false;
    };
    let lines = run_lines(s);
    let env = key(s, "env").and_then(|e| e.as_mapping().cloned());
    let env_ok = env.as_ref().is_some_and(|e| {
        e.len() == GATE_STEP_ENV.len()
            && GATE_STEP_ENV
                .iter()
                .all(|(k, v)| e.get(*k).and_then(Value::as_str) == Some(*v))
    });
    let keys_ok = s
        .iter()
        .all(|(k, _)| k.as_str().is_some_and(|k| GATE_STEP_KEYS.contains(&k)));
    let at = lines.iter().position(|l| l == MST_LINE);
    job_ok
        && env_ok
        && keys_ok
        && at == Some(GATE_PRE_MST_LINES.len())
        && lines
            .iter()
            .zip(GATE_PRE_MST_LINES)
            .all(|(l, want)| l == want)
}

/// Steps that may run before the guard step in `mg-reuse-self-test`: checkout, and the two scripts' exact `run` text.
const EARLIER_RUNS: [&str; 3] = [
    "bash scripts/ci_mg_reuse.sh --self-test",
    "setsid --wait bash scripts/check_ci_wait_cancel.sh --self-test",
    "setsid --wait bash scripts/check_ci_wait_cancel.sh",
];
const GUARD_JOB_KEYS: [&str; 5] = ["if", "runs-on", "timeout-minutes", "permissions", "steps"];

fn earlier_step_ok(s: &Mapping) -> bool {
    let keys_ok = s.iter().all(|(k, _)| {
        k.as_str()
            .is_some_and(|k| ["name", "uses", "run"].contains(&k))
    });
    let uses = key(s, "uses");
    let run = key(s, "run");
    keys_ok
        && match (uses, run) {
            (Some(Value::String(u)), None) => u == "actions/checkout@v7",
            (None, Some(Value::String(_))) => {
                let l = run_lines(s);
                !l.is_empty() && l.iter().all(|x| EARLIER_RUNS.contains(&x.as_str()))
            }
            _ => false,
        }
}

/// The `on:` key (`on` is a string here, but a YAML 1.1 reader makes it `true`, so accept both).
fn on_block(top: &Mapping) -> Option<Mapping> {
    top.iter()
        .find(|(k, _)| {
            k.as_bool() == Some(true) || k.as_str().is_some_and(|k| k.eq_ignore_ascii_case("on"))
        })
        .and_then(|(_, v)| v.as_mapping().cloned())
}

/// The whole `on:` block, pinned. The guard job only runs when the workflow does, so any filter (branches, tags, paths,
/// ignore lists), a narrowed `types` (which replaces GitHub's default opened/synchronize/reopened), a `merge_group` with
/// a body, or another trigger changes when the guard runs. The block is compared, normalised (maps and lists sorted, so
/// order and flow-versus-block style do not matter), against one expected value.
const EXPECTED_ON: &str = "push:
  branches: [main, master]
  tags: ['v*']
pull_request:
  branches: [main, master]
  types: [opened, synchronize, reopened, ready_for_review]
merge_group:
workflow_dispatch:
";

fn norm(v: &Value) -> String {
    match v {
        Value::Null => "null".into(),
        Value::Bool(b) => format!("b:{b}"),
        Value::Number(n) => format!("n:{n}"),
        Value::String(s) => format!("s:{s:?}"),
        Value::Sequence(s) => {
            let mut x: Vec<String> = s.iter().map(norm).collect();
            x.sort();
            format!("[{}]", x.join(","))
        }
        Value::Mapping(m) => {
            let mut x: Vec<String> = m
                .iter()
                .map(|(k, v)| format!("{}=>{}", norm(k), norm(v)))
                .collect();
            x.sort();
            format!("{{{}}}", x.join(","))
        }
        Value::Tagged(_) => "<tagged>".into(),
    }
}

fn check_triggers(top: &Mapping, why: &mut Vec<String>) {
    let Some(on) = on_block(top) else {
        return why.push("wiring: `on` is missing or not a mapping".into());
    };
    let want: Value = serde_yaml::from_str(EXPECTED_ON).expect("expected on block parses");
    if norm(&Value::Mapping(on)) != norm(&want) {
        why.push(
            "wiring: `on` lacks the exact pinned trigger block (push main/master + v* tags, pull_request main/master with opened/synchronize/reopened/ready_for_review, merge_group, workflow_dispatch; no filter, no other trigger)".into(),
        );
    }
}

/// Part (f): the guard test runs, unmasked, in the job every event runs, and `gate` needs it AND enforces its result.
fn check_wiring(top: &Mapping, why: &mut Vec<String>) {
    const JOB: &str = "mg-reuse-self-test";
    const IF: &str =
        "${{ github.event_name != 'pull_request' || !github.event.pull_request.draft }}";
    let Some(Value::Mapping(jobs)) = key(top, "jobs") else {
        return;
    };
    let Some(Value::Mapping(job)) = jobs.get(JOB).cloned() else {
        return why.push(format!("wiring: job {JOB} is missing"));
    };
    // Workflow-level env is inherited by both jobs (`PATH`, `LD_PRELOAD`, `DRAFT`...), so any of it is red.
    if key(top, "defaults").is_some() || key(top, "env").is_some() || env_masks(top) {
        why.push("wiring: workflow level sets defaults or an env".into());
    }
    check_triggers(top, why);
    if key(&job, "if").as_ref().and_then(Value::as_str) != Some(IF) {
        why.push(format!(
            "wiring: {JOB} `if` is not the all-events condition"
        ));
    }
    if key(&job, "continue-on-error").is_some() {
        why.push(format!("wiring: {JOB} has continue-on-error"));
    }
    if key(&job, "defaults").is_some() || key(&job, "env").is_some() {
        why.push(format!(
            "wiring: {JOB} sets defaults or env (a shell default or env can mask)"
        ));
    }
    let steps = match key(&job, "steps") {
        Some(Value::Sequence(s)) => s,
        _ => return why.push(format!("wiring: {JOB} steps are not a list")),
    };
    let wired = steps
        .iter()
        .any(|s| matches!(s, Value::Mapping(s) if guard_step_ok(s)));
    // Anything that runs before the guard step can poison its environment (`>> "$GITHUB_ENV"`, `>> "$GITHUB_PATH"`, an
    // action's `core.exportVariable`). So every earlier step is an exact allowlisted one, and the job has no other key.
    let guard_at = steps
        .iter()
        .position(|s| matches!(s, Value::Mapping(s) if guard_step_ok(s)))
        .unwrap_or(0);
    if !steps[..guard_at]
        .iter()
        .all(|s| s.as_mapping().is_some_and(earlier_step_ok))
    {
        why.push(format!(
            "wiring: {JOB} has an earlier step outside the allowlist (it could write GITHUB_ENV or GITHUB_PATH)"
        ));
    }
    // The guard must test the checked-out tree of the run: `actions/checkout@v7` (no `with`, which earlier_step_ok
    // enforces) has to come before the guard step in this same job, or a stale warm clone is what gets tested.
    if !steps[..guard_at].iter().any(
        |s| matches!(s, Value::Mapping(m) if key(m, "uses").as_ref().and_then(Value::as_str) == Some("actions/checkout@v7")),
    ) {
        why.push(format!(
            "wiring: {JOB} has no actions/checkout@v7 step before the guard step (it would test a stale workspace)"
        ));
    }
    if !job
        .iter()
        .all(|(k, _)| k.as_str().is_some_and(|k| GUARD_JOB_KEYS.contains(&k)))
    {
        why.push(format!("wiring: {JOB} has a key outside the allowlist"));
    }
    if !wired {
        why.push(format!(
            "wiring: {JOB} has no unmasked step running this test"
        ));
    }
    let gate = jobs.get("gate").and_then(Value::as_mapping);
    let needs = gate
        .and_then(|g| key(g, "needs"))
        .map(|n| all_scalars(&n, &mut Vec::new()))
        .unwrap_or_default();
    if !needs.iter().any(|n| n == JOB) {
        why.push(format!("wiring: `gate` does not need {JOB}"));
    }
    // Needing the job is not enough: `gate` must also fail on its result.
    if !gate.is_some_and(gate_enforces_mst)
        || gate.is_some_and(|g| {
            key(g, "continue-on-error").is_some() || key(g, "defaults").is_some() || env_masks(g)
        })
    {
        why.push(format!(
            "wiring: `gate` does not enforce {JOB}'s result ($MST)"
        ));
    }
}

fn parse_file(path: &Path, base: &str, why: &mut Vec<String>) -> Option<Value> {
    let text = fs::read_to_string(path)
        .map_err(|e| why.push(format!("{base}: unreadable: {e}")))
        .ok()?;
    let doc: Value = serde_yaml::from_str(&text)
        .map_err(|e| why.push(format!("{base}: does not parse (cannot judge): {e}")))
        .ok()?;
    let mut unmodeled = Vec::new();
    all_scalars(&doc, &mut unmodeled);
    unmodeled
        .iter()
        .for_each(|w| why.push(format!("{base}: {w}")));
    Some(doc)
}

fn judge_file(path: &Path, why: &mut Vec<String>) {
    let base = path.file_name().unwrap().to_string_lossy().to_string();
    let Some(doc) = parse_file(path, &base, why) else {
        return;
    };
    let Value::Mapping(top) = &doc else {
        return why.push(format!("{base}: document is not a mapping, cannot judge"));
    };
    let Some(Value::Mapping(jobs)) = key(top, "jobs") else {
        return why.push(format!(
            "{base}: jobs is missing or not a mapping, cannot judge"
        ));
    };
    let is_ci = base == "ci.yml";
    let dir = path.parent().unwrap_or(Path::new("."));
    for (n, j) in &jobs {
        match n.as_str() {
            Some(n) => judge_job(&base, dir, n, j, is_ci, why),
            None => why.push(format!("{base}: non-string job id, cannot judge")),
        }
    }
    if is_ci {
        check_workflow_level(&base, top, why);
        check_wiring(top, why);
    }
}

/// Ok(files judged) when the directory passes; Err carries every reason it does not. `dir` is
/// opened as given, so `.github/workflows/` and `.github/workflows` are the same directory.
fn judge(dir: &Path) -> Result<usize, Vec<String>> {
    let mut files: Vec<PathBuf> = match fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.is_file()
                    && matches!(p.extension().and_then(|e| e.to_str()), Some("yml" | "yaml"))
            })
            .collect(),
        Err(e) => return Err(vec![format!("{}: unreadable: {e}", dir.display())]),
    };
    files.sort();
    if files.is_empty() {
        return Err(vec![format!(
            "no workflow files in {}, nothing was measured",
            dir.display()
        )]);
    }
    let mut why = Vec::new();
    files.iter().for_each(|f| judge_file(f, &mut why));
    if why.is_empty() {
        Ok(files.len())
    } else {
        Err(why)
    }
}

// ---------------------------------------------------------------------------------------------
// tests

#[test]
fn only_ascii_printable_newline_and_tab_are_judgeable() {
    for bad in [
        "\u{a0}#x",
        "\u{3000}#x",
        "\u{b}#x",
        "\u{c}#x",
        "a\r\nb",
        "\u{1}",
        "caf\u{e9}",
    ] {
        assert_eq!(
            logical_lines(bad),
            vec![UNJUDGEABLE_RUN.to_string()],
            "{bad:?}"
        );
    }
    assert_eq!(
        logical_lines("\t # c\n  a b\t\n\n"),
        vec!["a b".to_string()]
    );
}

#[test]
fn trigger_order_and_style_do_not_matter() {
    let ci = real_mut(
        "branches: [main, master]\n    # Drafts",
        "branches: [master, main]\n    # Drafts",
    );
    let d = dir_with(&[("ci.yml", &ci)]);
    if let Err(w) = judge(d.path()) {
        assert!(!w.iter().any(|x| x.contains("`on` lacks")), "{w:?}");
    }
}

fn ci_text() -> String {
    fs::read_to_string(repo().join(".github/workflows/ci.yml")).unwrap()
}

fn dir_with(files: &[(&str, &str)]) -> tempfile::TempDir {
    let d = tempfile::tempdir().unwrap();
    for (n, t) in files {
        fs::write(d.path().join(n), t).unwrap();
    }
    d
}

/// Red for the NAMED reason, so a mutant cannot pass by tripping some other rule.
fn red_in(d: &tempfile::TempDir, name: &str, needle: &str) {
    match judge(d.path()) {
        Ok(_) => panic!("mutant {name} was accepted"),
        Err(w) => assert!(
            w.iter().any(|x| x.contains(needle)),
            "mutant {name} refused for the wrong reason (want `{needle}`): {w:?}"
        ),
    }
}

fn red(name: &str, ci: &str, needle: &str) {
    red_in(&dir_with(&[("ci.yml", ci)]), name, needle);
}

/// Replace the first `from` inside job `job` of the real ci.yml; the mutation must apply.
fn job_mut(job: &str, from: &str, to: &str) -> String {
    let ci = ci_text();
    let at = ci.find(&format!("\n  {job}:\n")).expect("job present");
    let (a, b) = ci.split_at(at);
    let m = b.replacen(from, to, 1);
    assert_ne!(b, m, "mutation must apply in {job}: {from}");
    format!("{a}{m}")
}

fn real_mut(from: &str, to: &str) -> String {
    let ci = ci_text();
    let m = ci.replacen(from, to, 1);
    assert_ne!(ci, m, "mutation must apply: {from}");
    m
}

fn wf(runs_on: &str) -> String {
    format!("on: push\njobs:\n  t:\n    runs-on: {runs_on}\n    steps: []\n")
}

#[test]
fn real_tree_passes() {
    let n = judge(&repo().join(".github/workflows")).unwrap_or_else(|w| panic!("{w:#?}"));
    assert!(n >= 40, "guard measured only {n} workflow files");
}

#[test]
fn trailing_slash_is_the_same_directory() {
    let p = format!("{}/", repo().join(".github/workflows").display());
    assert!(
        judge(Path::new(&p)).is_ok(),
        "trailing slash changed the verdict"
    );
    let d = dir_with(&[(
        "ci.yml",
        &real_mut(
            "runs-on: [self-hosted, Linux, X64, clean-room]",
            "runs-on: [self-hosted]",
        ),
    )]);
    let slashed = format!("{}/", d.path().display());
    assert!(
        judge(Path::new(&slashed)).is_err(),
        "a mutant passed with a trailing slash"
    );
}

#[test]
fn empty_and_missing_dir_fail() {
    let d = tempfile::tempdir().unwrap();
    assert!(judge(d.path()).is_err());
    assert!(judge(&d.path().join("absent")).is_err());
}

// label SUBSET routing, on the real x86-main job and on scratch workflows
#[test]
fn subset_label_lists_are_red() {
    let real = "runs-on: [self-hosted, Linux, X64, clean-room]";
    for (i, ro) in [
        "[self-hosted]",
        "[self-hosted, Linux]",
        "[self-hosted, Linux, X64]",
        "[self-hosted, Linux, pool2, ffmpeg]",
        "[self-hosted, Linux, burst-x64]",
        "[SELF-HOSTED, linux]",
        "self-hosted",
        "{labels: [self-hosted, Linux]}",
        "{group: spot-pool}",
        "{group: pool2, labels: [self-hosted]}",
        "[]",
    ]
    .iter()
    .enumerate()
    {
        let m = job_mut("x86-main", real, &format!("runs-on: {ro}"));
        red(
            &format!("x86-main-subset-{i}"),
            &m,
            "can land on a spot seat",
        );
        red(
            &format!("scratch-subset-{i}"),
            &wf(ro),
            "can land on a spot seat",
        );
    }
    // a group on the non-spot list, and a label no seat carries, are fine
    for ok in [
        "{group: Default}",
        "[self-hosted, Linux, clean-room]",
        "[self-hosted, other-pool]",
    ] {
        assert!(
            judge(dir_with(&[("a.yml", &wf(ok))]).path()).is_ok(),
            "false red: {ok}"
        );
    }
}

#[test]
fn pr_light_pool_expression_is_judged_on_both_branches() {
    let m = job_mut(
        "pr-light",
        r#"fromJSON('["self-hosted","Linux","X64","pr-x64"]')"#,
        r#"fromJSON('["self-hosted","Linux"]')"#,
    );
    red("pr-light-branch", &m, "can land on a spot seat");
    // the same expression copied onto a release job with a subset branch
    let rel = "on: push\njobs:\n  release:\n    permissions:\n      id-token: write\n    runs-on: ${{ vars.PR_LIGHT_POOL == 'pr-x64' && fromJSON('[\"self-hosted\",\"Linux\",\"X64\"]') || fromJSON('[\"self-hosted\",\"Linux\",\"X64\",\"clean-room\"]') }}\n    steps: []\n";
    red("release-pool-copy", rel, "can land on a spot seat");
}

#[test]
fn unresolvable_expressions_are_red() {
    for (i, ro) in [
        "${{ vars.POOL }}",
        "${{ fromJSON(vars.LABELS) }}",
        "${{ github.event_name == 'push' && 'a' || 'b' }}",
        "[self-hosted, \"${{ vars.POOL }}\"]",
        "${{ matrix.missing }}",
    ]
    .iter()
    .enumerate()
    {
        let d = dir_with(&[(
            "a.yml",
            &wf(&format!("'{}'", ro.replace('\'', "''")))
                .replace("'[", "[")
                .replace("]'", "]"),
        )]);
        red_in(&d, &format!("unresolved-{i}"), "spot seat");
    }
    // the macOS list: a literal label no seat carries beside one unresolved element is fine
    let mac = wf("[self-hosted, macOS, ARM64, apple-silicon, \"${{ inputs.runner_label }}\"]");
    assert!(judge(dir_with(&[("a.yml", &mac)]).path()).is_ok());
}

#[test]
fn matrix_labels_are_judged_per_option() {
    let m = |labels: &str| {
        format!(
            "on: push\njobs:\n  t:\n    runs-on: ${{{{ fromJSON(matrix.labels) }}}}\n    strategy:\n      matrix:\n        include:\n          - labels: '{labels}'\n          - labels: '[\"self-hosted\",\"Linux\",\"clean-room\"]'\n    steps: []\n"
        )
    };
    assert!(
        judge(dir_with(&[("a.yml", &m(r#"["self-hosted","Linux","ARM64","cuda"]"#))]).path())
            .is_ok()
    );
    for (i, l) in [
        r#"["self-hosted","Linux","X64"]"#,
        r#"["self-hosted"]"#,
        r#"${{ vars.X }}"#,
    ]
    .iter()
    .enumerate()
    {
        red_in(
            &dir_with(&[("a.yml", &m(l))]),
            &format!("matrix-{i}"),
            "spot seat",
        );
    }
    let no_strategy =
        "on: push\njobs:\n  t:\n    runs-on: ${{ fromJSON(matrix.labels) }}\n    steps: []\n";
    red("matrix-no-strategy", no_strategy, "literal strategy");
}

// non-PR triggers carry the same rule
#[test]
fn every_trigger_is_judged() {
    for trig in [
        "push",
        "schedule:\n    - cron: '0 0 * * *'",
        "workflow_dispatch",
        "issue_comment",
        "release",
    ] {
        let w = format!("on:\n  {trig}\njobs:\n  rel:\n    permissions:\n      contents: write\n      id-token: write\n    runs-on: [self-hosted, Linux, pool2, ffmpeg]\n    steps:\n      - run: echo ${{{{ secrets.CARGO_REGISTRY_TOKEN }}}}\n");
        red(&format!("trigger-{trig}"), &w, "can land on a spot seat");
    }
}

// duplicate keys, case variants, tabs, escapes are all resolved by the parser
#[test]
fn parser_level_bypasses_are_red() {
    let sh = "[self-hosted, Linux, X64, clean-room]";
    let cases: Vec<(&str, String, &str)> = vec![
        ("dup-runs-on", format!("on: push\njobs:\n  t:\n    runs-on: [self-hosted, burst-x64]\n    runs-on: {sh}\n    steps: []\n"), "does not parse"),
        ("dup-jobs", format!("on: push\njobs:\n  a:\n    runs-on: [self-hosted]\n    steps: []\njobs:\n  b:\n    runs-on: {sh}\n    steps: []\n"), "does not parse"),
        ("dup-permissions", real_mut("  gate:\n", "  gate:\n    permissions:\n      contents: read\n"), "does not parse"),
        ("case-variant-key", "on: push\njobs:\n  t:\n    Runs-On: ${{ vars.POOL }}\n    steps: []\n".to_string(), "runs-on expression cannot be resolved"),
        ("tab-job-name", "on: push\njobs:\n  \"a\tb\tc\":\n    runs-on: [self-hosted, Linux]\n    steps: []\n".to_string(), "spot seat"),
        ("tab-in-label", "on: push\njobs:\n  t:\n    runs-on: \"a\tb\t${{ vars.POOL }}\"\n    steps: []\n".to_string(), "spot seat"),
        ("escape-key", "on: push\njobs:\n  t:\n    \"runs\\x2Don\": [self-hosted]\n    steps: []\n".to_string(), "spot seat"),
        ("merge-key", format!("on: push\nx: &a\n  runs-on: {sh}\njobs:\n  t:\n    <<: *a\n    steps: []\n"), "merge key"),
        ("tag", "on: push\njobs:\n  t:\n    runs-on: !custom [self-hosted]\n    steps: []\n".to_string(), "tagged"),
        ("two-documents", format!("on: push\njobs: {{}}\n---\non: push\njobs:\n  t:\n    runs-on: {sh}\n    steps: []\n"), "does not parse"),
        ("not-a-map", "- a\n- b\n".to_string(), "not a mapping"),
        ("jobs-not-a-map", "on: push\njobs: [a]\n".to_string(), "jobs is missing or not a mapping"),
        ("job-not-a-map", "on: push\njobs:\n  t: nope\n".to_string(), "not a mapping"),
        ("no-runs-on-no-uses", "on: push\njobs:\n  t:\n    steps: []\n".to_string(), "no runs-on and no uses"),
        ("unparseable", "on: [\n".to_string(), "does not parse"),
    ];
    for (n, text, needle) in cases {
        red(n, &text, needle);
    }
}

// reusable calls are judged, not trusted
#[test]
fn reusable_calls_are_judged() {
    let call = |body: &str| format!("on: push\njobs:\n  c:\n{body}");
    let cases = [
        ("unlisted-external", call("    uses: o/r/.github/workflows/x.yml@v1\n"), "not on the allowlist"),
        ("unpinned-lookalike", call("    uses: paiml/.github/.github/workflows/pr-gate.yml@v1\n"), "not on the allowlist"),
        ("external-with-runs-on-hyphen", call("    uses: paiml/.github/.github/workflows/pr-gate.yml@8289382dfd2852714a7fa14868a7545c91cd98e6\n    with:\n      runs-on: '${{ vars.POOL }}'\n"), "passes a runs-on"),
        ("external-with-runner-label", call("    uses: paiml/.github/.github/workflows/pr-gate.yml@8289382dfd2852714a7fa14868a7545c91cd98e6\n    with:\n      runner_label: x\n"), "passes a runs-on"),
        ("external-secrets", call("    uses: paiml/.github/.github/workflows/pr-gate.yml@8289382dfd2852714a7fa14868a7545c91cd98e6\n    secrets: inherit\n"), "passes secrets"),
        ("external-smells-wrong-runs-on", call("    uses: paiml/.github/.github/workflows/runner-smells.yml@d768b8ac5abfa7d0038470fd38023a15323ac0aa\n    with:\n      runs_on: '[\"self-hosted\",\"Linux\"]'\n"), "passes a runs-on"),
        ("local-missing", call("    uses: ./.github/workflows/absent.yml\n"), "unreadable"),
        ("local-escape", call("    uses: ./.github/workflows/../../x.yml\n"), "cannot judge"),
        ("local-inherit", call("    uses: ./.github/workflows/callee.yml\n    secrets: inherit\n"), "secrets: inherit"),
        ("local-not-string", call("    uses: [a]\n"), "not a string"),
    ];
    let callee =
        "on: workflow_call\njobs:\n  j:\n    runs-on: [self-hosted, Linux]\n    steps: []\n";
    for (n, text, needle) in cases {
        red_in(
            &dir_with(&[("a.yml", &text), ("callee.yml", callee)]),
            n,
            needle,
        );
    }
    // a local callee whose runs-on comes from `with` is judged with that input resolved
    let by_input = "on: workflow_call\njobs:\n  j:\n    runs-on: ${{ inputs.ro }}\n    steps: []\n";
    let bad = call("    uses: ./.github/workflows/callee.yml\n    with:\n      ro: '[\"self-hosted\",\"Linux\"]'\n");
    red_in(
        &dir_with(&[("a.yml", &bad), ("callee.yml", by_input)]),
        "local-input-subset",
        "can land on a spot seat",
    );
    let unset = call("    uses: ./.github/workflows/callee.yml\n");
    red_in(
        &dir_with(&[("a.yml", &unset), ("callee.yml", by_input)]),
        "local-input-unset",
        "can land on a spot seat",
    );
    let ok = call("    uses: ./.github/workflows/callee.yml\n    with:\n      ro: '[\"self-hosted\",\"Linux\",\"clean-room\"]'\n");
    let w = judge(dir_with(&[("a.yml", &ok), ("callee.yml", by_input)]).path()).unwrap_err();
    assert!(w.iter().all(|x| x.starts_with("callee.yml")), "{w:?}");
}

// the three moved jobs: canonical expression only, and nowhere else
#[test]
fn only_the_three_aggregators_may_take_the_spot_expression() {
    let canon = canonical("gate").unwrap();
    let on_x86 = job_mut(
        "x86-main",
        "runs-on: [self-hosted, Linux, X64, clean-room]",
        &format!("runs-on: {canon}"),
    );
    red("canonical-on-x86-main", &on_x86, "opt-in variable outside");
    // `gate` carrying workspace-test's expression is not its one canonical value
    let swapped = job_mut("gate", &canon, &canonical("workspace-test").unwrap());
    red(
        "gate-with-workspace-test-expr",
        &swapped,
        "not its one canonical value",
    );
    // an unconditional spot route
    let spot = job_mut("gate", &canon, "[self-hosted, burst-x64, burst-public]");
    red(
        "gate-unconditional-spot",
        &spot,
        "not its one canonical value",
    );
    let var = job_mut("gate", "PR_SPOT_OPTIN == '1'", "PR_SPOT_OPTIN != '0'");
    red(
        "gate-changed-condition",
        &var,
        "not its one canonical value",
    );
}

#[test]
fn moved_jobs_carry_no_checkout_cache_secret_or_write() {
    // workspace-test's checkout is skipped on a pull_request; drop that skip
    let no_skip = job_mut(
        "workspace-test",
        "github.event.pull_request.draft != true && github.event_name != 'pull_request' }}",
        "github.event.pull_request.draft != true }}",
    );
    red(
        "workspace-test-checkout-on-pr",
        &no_skip,
        "runs on pull_request",
    );
    let or_skip = job_mut(
        "workspace-test",
        "github.event.pull_request.draft != true && github.event_name != 'pull_request' }}",
        "github.event_name != 'pull_request' || true }}",
    );
    red("workspace-test-or-true", &or_skip, "runs on pull_request");
    for job in ["ci-gate", "gate"] {
        for (n, step) in [
            ("checkout", "      - uses: actions/checkout@v7\n"),
            ("cache", "      - uses: actions/cache@v4\n"),
            ("download", "      - uses: actions/download-artifact@v4\n"),
        ] {
            let m = job_mut(job, "    steps:\n", &format!("    steps:\n{step}"));
            red(&format!("{job}-{n}"), &m, "runs on pull_request");
        }
        let sec = job_mut(
            job,
            "    steps:\n",
            "    env:\n      T: ${{ secrets.PR_REVIEW_SIGNING_KEY_B64 }}\n    steps:\n",
        );
        red(&format!("{job}-secret"), &sec, "names secrets");
        let tok = job_mut(
            job,
            "    steps:\n",
            "    env:\n      T: ${{ github.token }}\n    steps:\n",
        );
        red(&format!("{job}-token"), &tok, "token");
        let wide = job_mut(job, "      contents: read\n", "      contents: write\n");
        red(
            &format!("{job}-write"),
            &wide,
            "permissions are not exactly",
        );
        let idt = job_mut(
            job,
            "      contents: read\n",
            "      contents: read\n      id-token: write\n",
        );
        red(
            &format!("{job}-id-token"),
            &idt,
            "permissions are not exactly",
        );
        let wa = job_mut(
            job,
            "    permissions:\n      contents: read\n",
            "    permissions: write-all\n",
        );
        red(
            &format!("{job}-write-all"),
            &wa,
            "permissions are not exactly",
        );
        let none = job_mut(job, "    permissions:\n      contents: read\n", "");
        red(
            &format!("{job}-no-permissions"),
            &none,
            "permissions are not exactly",
        );
    }
}

// top-level permissions
#[test]
fn workflow_level_permissions_are_judged() {
    for (i, p) in [
        "read-all",
        "write-all",
        "{contents: write}",
        "{contents: read, id-token: write}",
    ]
    .iter()
    .enumerate()
    {
        let m = real_mut("\njobs:\n", &format!("\npermissions: {p}\njobs:\n"));
        red(&format!("top-permissions-{i}"), &m, "permissions wider");
    }
    let secrets = real_mut("\njobs:\n", "\nenv:\n  K: ${{ secrets.X }}\njobs:\n");
    red("top-secrets", &secrets, "names secrets");
    let ok = real_mut("\njobs:\n", "\npermissions:\n  contents: read\njobs:\n");
    assert!(judge(dir_with(&[("ci.yml", &ok)]).path()).is_ok());
}

// the guard must run on pull_request, merge_group AND push, unmasked
#[test]
fn guard_is_wired_unmasked_into_every_event() {
    const DIGEST: &str = "70f2cdc9e537fe4447306d33c317fea33017f78f1ffa3b7af99095391419ef36";
    let job = "mg-reuse-self-test";
    let step = "--test falsification_pr_spot_labels";
    let cases = [
        ("step-removed", job_mut(job, step, "--test other_test"), "no unmasked step"),
        ("or-true", job_mut(job, step, &format!("{step} || true")), "no unmasked step"),
        ("set-plus-e", job_mut(job, step, &format!("{step}; set +e")), "no unmasked step"),
        ("step-if", job_mut(job, "      - name: pr spot labels guard", "      - if: false\n        name: pr spot labels guard"), "no unmasked step"),
        ("step-continue", job_mut(job, "      - name: pr spot labels guard", "      - continue-on-error: true\n        name: pr spot labels guard"), "no unmasked step"),
        ("job-continue", job_mut(job, "    runs-on:", "    continue-on-error: true\n    runs-on:"), "continue-on-error"),
        ("job-if-merge-group-only", job_mut(job, "if: ${{ github.event_name != 'pull_request' || !github.event.pull_request.draft }}", "if: ${{ github.event_name == 'merge_group' }}"), "all-events condition"),
        ("job-if-pr-only", job_mut(job, "if: ${{ github.event_name != 'pull_request' || !github.event.pull_request.draft }}", "if: ${{ github.event_name == 'pull_request' }}"), "all-events condition"),
        ("skip-suffix", job_mut(job, step, &format!("{step} -- --skip ''")), "no unmasked step"),
        ("list-suffix", job_mut(job, step, &format!("{step} -- --list")), "no unmasked step"),
        ("semicolon-suffix", job_mut(job, step, &format!("{step}; true")), "no unmasked step"),
        ("step-shell-true", job_mut(job, "      - name: pr spot labels guard", "      - shell: \"true {0}\"\n        name: pr spot labels guard"), "no unmasked step"),
        ("step-If-case", job_mut(job, "      - name: pr spot labels guard", "      - If: false\n        name: pr spot labels guard"), "no unmasked step"),
        ("step-Continue-On-Error-case", job_mut(job, "      - name: pr spot labels guard", "      - Continue-On-Error: true\n        name: pr spot labels guard"), "no unmasked step"),
        ("if-wrapped", job_mut(job, "          timeout --foreground", "          if timeout --foreground"), "no unmasked step"),
        ("negated", job_mut(job, "          timeout --foreground", "          ! timeout --foreground"), "no unmasked step"),
        ("commented-out", job_mut(job, "cargo test --locked -p aprender-contracts --test falsification_pr_spot_labels", "# cargo test --locked -p aprender-contracts --test falsification_pr_spot_labels"), "no unmasked step"),
        ("errexit-off", job_mut(job, "set -euo pipefail", "set -euo pipefail\n          set +o errexit"), "no unmasked step"),
        ("no-errexit", job_mut(job, "set -euo pipefail", "true"), "no unmasked step"),
        ("job-defaults-shell", job_mut(job, "    runs-on:", "    defaults:\n      run:\n        shell: \"true {0}\"\n    runs-on:"), "sets defaults"),
        ("job-Continue-On-Error-case", job_mut(job, "    runs-on:", "    Continue-On-Error: true\n    runs-on:"), "has continue-on-error"),
        ("job-If-case", job_mut(job, "if: ${{ github.event_name != 'pull_request' || !github.event.pull_request.draft }}", "If: ${{ github.event_name == 'merge_group' }}"), "all-events condition"),
        ("gate-drops-mst-line", real_mut("          [ \"$MST\" = success ] || { echo \"mg-reuse-self-test failed: $MST\"; exit 1; }\n", ""), "does not enforce"),
        ("gate-mst-env-constant", real_mut("MST: ${{ needs.mg-reuse-self-test.result }}", "MST: success"), "does not enforce"),
        ("multiline-if-wrapper", job_mut(job, "          timeout --foreground", "          if false; then\n          timeout --foreground"), "no unmasked step"),
        ("function-shadow", job_mut(job, "          mkdir -p", "          timeout() { :; }\n          mkdir -p"), "no unmasked step"),
        ("reap-exit-two-spaces", job_mut(job, "reap() { if docker rm -f \"$cname\" >/dev/null 2>&1; then :; fi; }", "reap() { exit  0; }"), "no unmasked step"),
        ("set-two-spaces", job_mut(job, "set -euo pipefail", "set -euo pipefail\n          set  +e"), "no unmasked step"),
        ("exec-true", job_mut(job, "          mkdir -p", "          exec true\n          mkdir -p"), "no unmasked step"),
        ("no-run-before-test", job_mut(job, " cargo test --locked", " cargo test --no-run --locked"), "no unmasked step"),
        ("entrypoint-true", job_mut(job, "docker run --rm --init", "docker run --rm --entrypoint true --init"), "no unmasked step"),
        ("job-env-shellopts", job_mut(job, "    runs-on:", "    env:\n      SHELLOPTS: noexec\n    runs-on:"), "sets defaults or env"),
        ("top-defaults-shell", real_mut("\njobs:\n", "\ndefaults:\n  run:\n    shell: \"true {0}\"\njobs:\n"), "workflow level sets defaults"),
        ("top-Permissions-case", real_mut("\njobs:\n", "\nPermissions: write-all\njobs:\n"), "permissions wider"),
        ("gate-continue-on-error", job_mut("gate", "    runs-on:", "    continue-on-error: true\n    runs-on:"), "does not enforce"),
        ("gate-mst-no-exit", real_mut("echo \"mg-reuse-self-test failed: $MST\"; exit 1; }", "echo \"mg-reuse-self-test failed: $MST\"; }"), "does not enforce"),
        ("gate-mst-assigned", real_mut("          [ \"$MST\" = success ] ||", "          MST=success\n          [ \"$MST\" = success ] ||"), "does not enforce"),
        ("not-a-need-of-gate", real_mut("needs: [x86-main, guards, workspace-test, determinism, mg-reuse, x86-main-advisories, mg-reuse-self-test, pr-light]", "needs: [x86-main, guards, workspace-test, determinism, mg-reuse, x86-main-advisories, pr-light]"), "does not need"),
        ("duplicate-image", job_mut(job, "docker run --rm --init --name \"$cname\"", &format!("docker run --rm --init --name \"$cname\" localhost:5000/sovereign-ci:stable@sha256:{DIGEST} true")), "no unmasked step"),
        ("detach-flag", job_mut(job, "docker run --rm --init --name \"$cname\"", "docker run --rm --init --name \"$cname\" -d"), "no unmasked step"),
        ("quoted-entrypoint", job_mut(job, "docker run --rm --init --name \"$cname\"", "docker run --rm --init --name \"$cname\" --entry\"\"point true"), "no unmasked step"),
        ("cargo-runner-env", job_mut(job, "-e CARGO_TARGET_DIR=/t", "-e CARGO_TARGET_DIR=/t -e CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER=true"), "no unmasked step"),
        ("guard-step-env", job_mut(job, "      - name: pr spot labels guard", "      - env:\n          RUSTC_WRAPPER: true\n        name: pr spot labels guard"), "no unmasked step"),
        ("earlier-step-github-env", job_mut(job, "        run: bash scripts/ci_mg_reuse.sh --self-test", "        run: |\n          bash scripts/ci_mg_reuse.sh --self-test\n          echo CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER=true >> \"$GITHUB_ENV\""), "earlier step"),
        ("earlier-step-github-path", job_mut(job, "        run: bash scripts/ci_mg_reuse.sh --self-test", "        run: |\n          bash scripts/ci_mg_reuse.sh --self-test\n          echo /tmp/evil >> \"$GITHUB_PATH\""), "earlier step"),
        ("earlier-step-uses-action", job_mut(job, "      - name: pr spot labels guard", "      - uses: some/action@v1\n      - name: pr spot labels guard"), "earlier step"),
        ("job-extra-key", job_mut(job, "    runs-on:", "    container: alpine\n    runs-on:"), "key outside the allowlist"),
        ("gate-step-env-shellopts", job_mut("gate", "MST: ${{ needs.mg-reuse-self-test.result }}", "MST: ${{ needs.mg-reuse-self-test.result }}\n          SHELLOPTS: noexec"), "does not enforce"),
        ("gate-step-env-bash-func", job_mut("gate", "MST: ${{ needs.mg-reuse-self-test.result }}", "MST: ${{ needs.mg-reuse-self-test.result }}\n          \"BASH_FUNC_[%%\": \"() { return 0; }\""), "does not enforce"),
        ("gate-step-env-bash-env", job_mut("gate", "MST: ${{ needs.mg-reuse-self-test.result }}", "MST: ${{ needs.mg-reuse-self-test.result }}\n          bash_env: /dev/null"), "does not enforce"),
        ("gate-job-if-false", job_mut("gate", "    if: always()", "    if: false"), "does not enforce"),
        ("gate-job-if-success", job_mut("gate", "    if: always()", "    if: ${{ success() }}"), "does not enforce"),
        ("gate-job-if-dropped", job_mut("gate", "    if: always()\n", ""), "does not enforce"),
        ("gate-if-false-wrapper", job_mut("gate", &format!("          {MST_LINE}"), &format!("          if false; then\n          {MST_LINE}\n          fi")), "does not enforce"),
        ("gate-early-exit-0", job_mut("gate", &format!("          {MST_LINE}"), &format!("          exit 0\n          {MST_LINE}")), "does not enforce"),
        ("gate-declare-mst", job_mut("gate", &format!("          {MST_LINE}"), &format!("          declare MST=success\n          {MST_LINE}")), "does not enforce"),
        ("gate-readonly-mst", job_mut("gate", &format!("          {MST_LINE}"), &format!("          readonly MST=success\n          {MST_LINE}")), "does not enforce"),
        ("gate-export-mst", job_mut("gate", &format!("          {MST_LINE}"), &format!("          export MST=success\n          {MST_LINE}")), "does not enforce"),
        ("gate-bracket-shadow", job_mut("gate", &format!("          {MST_LINE}"), &format!("          [() {{ return 0; }}\n          {MST_LINE}")), "does not enforce"),
        ("gate-trap-exit-0", job_mut("gate", &format!("          {MST_LINE}"), &format!("          trap 'exit 0' EXIT\n          {MST_LINE}")), "does not enforce"),
        ("gate-exec-wrapper", job_mut("gate", &format!("          {MST_LINE}"), &format!("          exec bash -c 'exit 0'\n          {MST_LINE}")), "does not enforce"),
        ("gate-earlier-step-github-env", job_mut("gate", "      - name: Check required sections", "      - run: echo SHELLOPTS=noexec >> \"$GITHUB_ENV\"\n      - name: Check required sections"), "does not enforce"),
        ("gate-draft-constant", job_mut("gate", "DRAFT: ${{ github.event.pull_request.draft }}", "DRAFT: \"true\""), "does not enforce"),
        ("gate-draft-env-dropped", job_mut("gate", "          DRAFT: ${{ github.event.pull_request.draft }}\n", ""), "does not enforce"),
        ("gate-env-path", job_mut("gate", "MST: ${{ needs.mg-reuse-self-test.result }}", "MST: ${{ needs.mg-reuse-self-test.result }}\n          PATH: ./scripts:/usr/bin"), "does not enforce"),
        ("gate-job-env-path", job_mut("gate", "    runs-on:", "    env:\n      PATH: ./scripts:/usr/bin\n    runs-on:"), "does not enforce"),
        ("gate-job-env-draft", job_mut("gate", "    runs-on:", "    env:\n      DRAFT: \"true\"\n    runs-on:"), "does not enforce"),
        ("gate-continuation-split-operator", job_mut("gate", "[ \"${DRAFT:-}\" != true ]", "[ \"${DRAFT:-}\" !=\\\n          true ]"), "does not enforce"),
        ("gate-comment-expression", job_mut("gate", "          set -euo pipefail\n", "          set -euo pipefail\n          # ${{ '\\nexit 0\\n' }}\n"), "does not enforce"),
        ("gate-comment-continuation", job_mut("gate", "          set -euo pipefail\n", "          set -euo pipefail\n          # note \\\n"), "does not enforce"),
        ("guard-comment-expression", job_mut(job, "          reap\n", "          reap\n          # ${{ '\\nexit 0\\n' }}\n"), "no unmasked step"),
        ("guard-continuation", job_mut(job, " --cpus 4 --memory 8g", " \\\n            --cpus 4 --memory 8g"), "no unmasked step"),
        ("guard-comment-continuation", job_mut(job, "          reap\n", "          reap\n          # note \\\n"), "no unmasked step"),
        ("earlier-step-expression", job_mut(job, "        run: bash scripts/ci_mg_reuse.sh --self-test", "        run: |\n          # ${{ '\\ntrue >> \"$GITHUB_PATH\"\\n' }}\n          bash scripts/ci_mg_reuse.sh --self-test"), "earlier step"),
        ("earlier-step-continuation", job_mut(job, "        run: bash scripts/ci_mg_reuse.sh --self-test", "        run: |\n          bash scripts/ci_mg_reuse.sh \\\n            --self-test"), "earlier step"),
        ("top-env-path", real_mut("\njobs:\n", "\nenv:\n  PATH: ./scripts:/usr/bin\njobs:\n"), "workflow level sets defaults or an env"),
        ("top-env-draft", real_mut("\njobs:\n", "\nenv:\n  DRAFT: \"true\"\njobs:\n"), "workflow level sets defaults or an env"),
        ("top-Env-case", real_mut("\njobs:\n", "\nEnv:\n  LD_PRELOAD: /tmp/x.so\njobs:\n"), "workflow level sets defaults or an env"),
        ("job-env-path", job_mut(job, "    runs-on:", "    env:\n      PATH: ./scripts:/usr/bin\n    runs-on:"), "sets defaults or env"),
        ("on-no-ready-for-review", real_mut("types: [opened, synchronize, reopened, ready_for_review]", "types: [opened, synchronize, reopened]"), "`on` lacks"),
        ("on-no-push", real_mut("  push:\n    branches: [main, master]", "  pushx:\n    branches: [main, master]"), "`on` lacks"),
        ("on-push-paths", real_mut("    branches: [main, master]\n    #", "    branches: [main, master]\n    paths: [nonexistent]\n    #"), "`on` lacks"),
        ("on-no-merge-group", real_mut("  merge_group:\n", "  merge_groupx:\n"), "`on` lacks"),
        ("nbsp-comment-in-guard", job_mut(job, "set -euo pipefail", "set -euo pipefail\n          \u{a0}#() { :; }; exit 0"), "no unmasked step"),
        ("ideographic-space-comment-in-guard", job_mut(job, "set -euo pipefail", "set -euo pipefail\n          \u{3000}#() { :; }; exit 0"), "no unmasked step"),
        ("nbsp-comment-earlier-step", job_mut(job, "        run: bash scripts/ci_mg_reuse.sh --self-test", "        run: |\n          \u{a0}#() { :; }; echo SHELLOPTS=noexec >> \"$GITHUB_ENV\"\n          bash scripts/ci_mg_reuse.sh --self-test"), "earlier step"),
        ("nbsp-comment-in-gate", job_mut("gate", "set -euo pipefail", "set -euo pipefail\n          \u{a0}#() { :; }; exit 0"), "does not enforce"),
        ("nbsp-before-bracket-in-gate", job_mut("gate", "          [ \"${DRAFT:-}\" != true ]", "          \u{a0}[ \"${DRAFT:-}\" != true ]"), "does not enforce"),
        ("on-push-branches-narrowed", real_mut("  push:\n    branches: [main, master]", "  push:\n    branches: [nonexistent]"), "`on` lacks"),
        ("on-push-no-tags", real_mut("    tags: ['v*']\n", ""), "`on` lacks"),
        ("on-pr-types-narrowed", real_mut("types: [opened, synchronize, reopened, ready_for_review]", "types: [ready_for_review]"), "`on` lacks"),
        ("on-pr-branches-ignore", real_mut("  pull_request:\n    branches: [main, master]", "  pull_request:\n    branches-ignore: ['**']"), "`on` lacks"),
        ("on-merge-group-types-empty", real_mut("  merge_group:\n", "  merge_group:\n    types: []\n"), "`on` lacks"),
        ("on-extra-trigger", real_mut("  workflow_dispatch:\n", "  workflow_dispatch:\n  schedule:\n    - cron: '0 0 * * *'\n"), "`on` lacks"),
        ("no-checkout-before-guard", job_mut(job, "      - uses: actions/checkout@v7\n", ""), "actions/checkout@v7"),
        ("gate-step-if-false", job_mut("gate", "      - name: Check required sections", "      - if: false\n        name: Check required sections"), "does not enforce"),
    ];
    for (n, text, needle) in cases {
        red(n, &text, needle);
    }
    let gone = real_mut("  mg-reuse-self-test:\n", "  mg-reuse-self-test-renamed:\n");
    red("job-renamed", &gone, "mg-reuse-self-test");
}
