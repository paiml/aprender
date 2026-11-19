//! Spot (burst) pool routing, judged on the PARSED workflow documents, never on raw lines.
//!
//! Every `.yml` and `.yaml` under `.github/workflows/` is parsed with the workspace's own YAML
//! crate (`serde_yaml`, which rejects duplicate keys), so quoted keys, escapes, flow style and
//! anchors resolve before anything is judged. Rules, for EVERY job of EVERY workflow, whatever
//! its trigger:
//!  (a) GitHub routes a job to a seat when the job's labels are a SUBSET of the seat's labels, so
//!      a job's resolved `runs-on` must hold a label no spot seat carries (or a runner group on the
//!      committed non-spot list). `[self-hosted]`, `[self-hosted, Linux]`, `[self-hosted, Linux,
//!      tier2, ffmpeg]`, `{labels: [...]}` and matrix options are all judged by that rule.
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

/// MEASURED SNAPSHOT, 2026-11-19, read-only REST (`orgs/paiml/actions/runners`, 631 burst seats):
/// every burst seat carries exactly `Linux, burst-x64, ffmpeg, self-hosted, tier2`. `x64` and
/// `burst-public` are added on purpose (an arch label may appear on a seat, and `burst-public` is
/// the PR-only spot label). Compared lower-case. A snapshot, not a live read: re-measure before
/// trusting it for a new seat family.
const SPOT_SEAT_LABELS: [&str; 7] = [
    "self-hosted",
    "linux",
    "x64",
    "tier2",
    "ffmpeg",
    "burst-x64",
    "burst-public",
];

/// MEASURED SNAPSHOT, same read: org runner groups holding ZERO burst seats.
const NON_SPOT_GROUPS: [&str; 15] = [
    "Default",
    "gpu-nodes",
    "gpu-x86",
    "aprender-reserved",
    "gate-path-slot",
    "rmedia-loan",
    "mini-mac",
    "public-build",
    "lambda-night",
    "aprender-only",
    "infra-first",
    "gate-light-shared",
    "release-lane",
    "demo-record",
    "dotgithub-gate",
];

/// EXTERNAL reusable calls this repo makes: the exact `uses` string. Anything else is red.
const EXTERNAL_CALLS: [&str; 3] = [
    "paiml/.github/.github/workflows/runner-smells.yml@d768b8ac5abfa7d0038470fd38023a15323ac0aa",
    "paiml/.github/.github/workflows/pr-gate.yml@main",
    "paiml/.github/.github/workflows/fork-pr-closer.yml@830147e62523b292823e3134943de10265c4569e",
];
/// The one `with.runs_on` an external call may pass, and which call may pass it.
const SMELLS_RUNS_ON: &str = X64;

fn key(m: &Mapping, k: &str) -> Option<Value> {
    m.get(k).cloned()
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
    let outside = ls.iter().any(|l| {
        !l.contains("${{") && !SPOT_SEAT_LABELS.contains(&l.trim().to_lowercase().as_str())
    });
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
        if ks == "jobs" {
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
        if ks == "permissions" && !permissions_ok(v) {
            why.push(format!(
                "{base}: workflow permissions wider than contents: read"
            ));
        }
    }
}

/// Part (f): the guard test runs, unmasked, in the job every event runs, and `gate` needs it.
fn check_wiring(top: &Mapping, why: &mut Vec<String>) {
    const JOB: &str = "mg-reuse-self-test";
    const IF: &str =
        "${{ github.event_name != 'pull_request' || !github.event.pull_request.draft }}";
    let Some(Value::Mapping(jobs)) = key(top, "jobs") else {
        return;
    };
    let Some(Value::Mapping(job)) = key(&jobs, JOB) else {
        return why.push(format!("wiring: job {JOB} is missing"));
    };
    if key(&job, "if").as_ref().and_then(Value::as_str) != Some(IF) {
        why.push(format!(
            "wiring: {JOB} `if` is not the all-events condition"
        ));
    }
    if key(&job, "continue-on-error").is_some() {
        why.push(format!("wiring: {JOB} has continue-on-error"));
    }
    let steps = match key(&job, "steps") {
        Some(Value::Sequence(s)) => s,
        _ => return why.push(format!("wiring: {JOB} steps are not a list")),
    };
    let wired = steps.iter().any(|s| {
        let Value::Mapping(s) = s else { return false };
        let run = key(s, "run")
            .and_then(|r| r.as_str().map(str::to_string))
            .unwrap_or_default();
        run.contains("--test falsification_pr_spot_labels")
            && run.contains("cargo test")
            && !["||", "set +e", "exit 0"].iter().any(|m| run.contains(m))
            && key(s, "if").is_none()
            && key(s, "continue-on-error").is_none()
    });
    if !wired {
        why.push(format!(
            "wiring: {JOB} has no unmasked step running this test"
        ));
    }
    let needs = key(&jobs, "gate")
        .and_then(|g| g.as_mapping().and_then(|g| key(g, "needs")))
        .map(|n| all_scalars(&n, &mut Vec::new()))
        .unwrap_or_default();
    if !needs.iter().any(|n| n == JOB) {
        why.push(format!("wiring: `gate` does not need {JOB}"));
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

// lane 1+2 finding 1: label SUBSET routing, on the real x86-main job and on scratch workflows
#[test]
fn subset_label_lists_are_red() {
    let real = "runs-on: [self-hosted, Linux, X64, clean-room]";
    for (i, ro) in [
        "[self-hosted]",
        "[self-hosted, Linux]",
        "[self-hosted, Linux, X64]",
        "[self-hosted, Linux, tier2, ffmpeg]",
        "[self-hosted, Linux, burst-x64]",
        "[SELF-HOSTED, linux]",
        "self-hosted",
        "{labels: [self-hosted, Linux]}",
        "{group: spot-pool}",
        "{group: tier2, labels: [self-hosted]}",
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
        "{group: gpu-nodes}",
        "[self-hosted, Linux, clean-room]",
        "[self-hosted, intel]",
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

// lane 1 finding 2: non-PR triggers carry the same rule
#[test]
fn every_trigger_is_judged() {
    for trig in [
        "push",
        "schedule:\n    - cron: '0 0 * * *'",
        "workflow_dispatch",
        "issue_comment",
        "release",
    ] {
        let w = format!("on:\n  {trig}\njobs:\n  rel:\n    permissions:\n      contents: write\n      id-token: write\n    runs-on: [self-hosted, Linux, tier2, ffmpeg]\n    steps:\n      - run: echo ${{{{ secrets.CARGO_REGISTRY_TOKEN }}}}\n");
        red(&format!("trigger-{trig}"), &w, "can land on a spot seat");
    }
}

// lane 1+2: duplicate keys, case variants, tabs, escapes are all resolved by the parser
#[test]
fn parser_level_bypasses_are_red() {
    let sh = "[self-hosted, Linux, X64, clean-room]";
    let cases: Vec<(&str, String, &str)> = vec![
        ("dup-runs-on", format!("on: push\njobs:\n  t:\n    runs-on: [self-hosted, burst-x64]\n    runs-on: {sh}\n    steps: []\n"), "does not parse"),
        ("dup-jobs", format!("on: push\njobs:\n  a:\n    runs-on: [self-hosted]\n    steps: []\njobs:\n  b:\n    runs-on: {sh}\n    steps: []\n"), "does not parse"),
        ("dup-permissions", real_mut("  gate:\n", "  gate:\n    permissions:\n      contents: read\n"), "does not parse"),
        ("case-variant-key", "on: push\njobs:\n  t:\n    Runs-On: ${{ vars.POOL }}\n    steps: []\n".to_string(), "no runs-on and no uses"),
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

// lane 1 finding 3 / lane 2 finding 4: reusable calls are judged, not trusted
#[test]
fn reusable_calls_are_judged() {
    let call = |body: &str| format!("on: push\njobs:\n  c:\n{body}");
    let cases = [
        ("unlisted-external", call("    uses: o/r/.github/workflows/x.yml@v1\n"), "not on the allowlist"),
        ("unpinned-lookalike", call("    uses: paiml/.github/.github/workflows/pr-gate.yml@v1\n"), "not on the allowlist"),
        ("external-with-runs-on-hyphen", call("    uses: paiml/.github/.github/workflows/pr-gate.yml@main\n    with:\n      runs-on: '${{ vars.POOL }}'\n"), "passes a runs-on"),
        ("external-with-runner-label", call("    uses: paiml/.github/.github/workflows/pr-gate.yml@main\n    with:\n      runner_label: x\n"), "passes a runs-on"),
        ("external-secrets", call("    uses: paiml/.github/.github/workflows/pr-gate.yml@main\n    secrets: inherit\n"), "passes secrets"),
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
    // lane 2 finding 5: workspace-test's checkout is skipped on a pull_request; drop that skip
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

// lane 2 finding 6: top-level permissions
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

// lane 2 finding 3: the guard must run on pull_request, merge_group AND push, unmasked
#[test]
fn guard_is_wired_unmasked_into_every_event() {
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
        ("not-a-need-of-gate", real_mut("needs: [x86-main, guards, workspace-test, determinism, mg-reuse, x86-main-advisories, mg-reuse-self-test, pr-light]", "needs: [x86-main, guards, workspace-test, determinism, mg-reuse, x86-main-advisories, pr-light]"), "does not need"),
    ];
    for (n, text, needle) in cases {
        red(n, &text, needle);
    }
    let gone = real_mut("  mg-reuse-self-test:\n", "  mg-reuse-self-test-renamed:\n");
    red("job-renamed", &gone, "mg-reuse-self-test");
}
