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
//!   `docs/audits/review-corpus/corpus-v1.jsonl` + `test-manifest-v1.txt` +
//!   `test-sketch-v1.txt` (PRA-001 T7 near-dup sketches of the test items).
//!   Prints counts only; it never prints a test item.
//! - `review --url U --cell C --host H --backend B --arm A --model-id M
//!   --weights-sha W --apr-tag T --apr-sha S --items DIR --out DIR --split dev|test
//!   [--limit N] [--rerun] [--cold-first]`
//!   REX-03: drive a resident `apr serve`; append receipts to
//!   `OUT/receipts.jsonl` and raw outputs under `OUT/raw/`. Prints a count line.
//! - `not-run --why NoDeclaredExecutor|Refused (identity flags as review)`
//!   write explicit `NotRun` receipts for a cell that cannot run.
//! - `score --receipts F --cell C --arm A --split dev|test [--rerun]
//!   [--pubkey P]` the §2.3 metrics as JSON. Without `--pubkey` the result
//!   is labelled `unsigned` (exploratory); with it, `minisign -V` must pass.
//!   `--pilot` (with `--split dev`) adds the REX-05 projection: warm/cold
//!   wall-clock spread, projected test runtime, and the §2.2 sample-size rule.
//! - `admit --cell C|all --why NoDeclaredExecutor --model-id M --weights-sha W
//!   --apr-tag T --apr-sha S --out FILE` append `NotRun` admission rows (REX-04);
//!   `admit --cell C --removed-by R ...` appends a `Refused` row.
//! - `admission-check --file F` every §2.1 cell resolved exactly once; prints the
//!   summary JSON. Exit 1 if inadmissible, 10 if admissible but S-7 (no cell admitted).
//! - `ledger REPO OUT_JSONL QUORUM_RECEIPT...` REX-07: write `review-ledger-v2`
//!   rows from quorum receipts and print shadow coverage. Exit 10 unless every
//!   receipt carries an uncounted shadow row that leaves the width alone.
//! - `ladder REPORT_JSON` REX-09: the lane's rung (shadow/tripwire/vote) from a
//!   `rex-001-report-v2` report, with every reason it stopped below vote. Exit 0
//!   whatever the rung; the gates read `.mode`.
//! - `ratchet record --file F --tag T --cell C --receipts R [--llama-receipts L
//!   --llama-cell LC]` REX-10: append one `review-lane-perf-ratchet-v1` entry
//!   from the tag's warm, non-rerun receipts on the cell (and llama.cpp's p95 on
//!   the same items). A tag with no timings is refused: it did not run.
//! - `ratchet check --file F` the andon JSON (arming/green/red). Exit 10 on RED:
//!   a paired Harrell–Davis p95 rise over the best earlier tag, or a violation.
//! - `challenge --ledger F --test-version V --id ID --tier B1a --weights-sha W
//!   --prompt-sha P [--adapter-sha A] --initial-weights-sha W0 --initial-prompt-sha P0
//!   --cell C --pubkey K --challenger-receipts R --challenger-arm A
//!   --champion-receipts R --champion-arm A [--h1 holds|fails] [--h2 holds|fails]
//!   [--contamination-hits N] [--p95-s X] [--budget-p95-s Y]` REX-11: the §5.4
//!   decision on the sealed test split, appended to the
//!   `review-champion-challenger-v1` ledger. Both receipt files must verify under
//!   `--pubkey` (unsigned data cannot promote, R-1); an omitted gate fails.
//!   Exit 0 promoted, 10 rejected, 11 refused (no evaluation spent).
//! - `b2 status [--refusals F] [--accepted verb=receipt,...]` REX-12: one
//!   `review-b2-loop-v1` row per B2 tier (ready, verb_refused, unledgered).
//!   Exit 11 while any row is NotRun.
//! - `b2 teacher-receipt --logits F --teacher-sha W --k K` the teacher-logit
//!   dataset receipt (sha, count, 0 test hashes) or every reason it is refused.
//! - `lane-kappa ROWS --split val [--shadow qwen-shadow] (--min-n N | --manifest M
//!   --candidate ROWS2)` PRM-C13 (was PRA-001 T13) (contract lane-independence-v1): κ_err of the
//!   shadow lane against every counted lane on matured gold rows of the split, as
//!   JSON; a thin pair prints `insufficient`, never a number. With `--candidate`
//!   the gate runs: δ and min_n come only from the manifest's `lane_independence`
//!   block. Exit 12 (andon, S-14) on a refused candidate, an unusable manifest,
//!   or `--candidate` without `--manifest`.

use aprender_review_experiment::b2;
use aprender_review_experiment::build_corpus::{
    choose, g_candidates, is_green, p_candidates, r_candidates, seal, Mutant, Pr, PER_CLASS,
};
use aprender_review_experiment::champion;
use aprender_review_experiment::cluster::{parse_sketches, render_sketches, sketch};
use aprender_review_experiment::contamination::Index;
use aprender_review_experiment::corpus::{
    assign_splits, corpus_version, parse_manifest, render_manifest, sha256_hex, Item, Split,
};
use aprender_review_experiment::harness::{
    classify, post, request_body, rerun_subset, run_order, utc_now, Run,
};
use aprender_review_experiment::pilot::{project, Projection};
use aprender_review_experiment::prereg;
use aprender_review_experiment::ratchet;
use aprender_review_experiment::receipt::{admissible, Arm, Expect, NotRun, Receipt};
use aprender_review_experiment::score::{collect, score, Scored};
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
        Some("admission-check") => admission_check(&args[1..]),
        Some("ledger") => ledger_cmd(&args[1..]),
        Some("ladder") => ladder_cmd(&args[1..]),
        Some("ratchet") => ratchet_cmd(&args[1..]),
        Some("challenge") => challenge_cmd(&args[1..]),
        Some("b2") => b2_cmd(&args[1..]),
        Some("lane-kappa") => lane_kappa_cmd(&args[1..]),
        Some(c @ ("review" | "not-run" | "score" | "admit")) => {
            match flags(&args[1..]).and_then(|f| match c {
                "review" => review(&f),
                "not-run" => not_run(&f),
                "admit" => admit(&f),
                _ => score_cmd(&f),
            }) {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("rex {c}: {e}");
                    ExitCode::from(1)
                }
            }
        }
        Some("corpus-build") if args.len() >= 5 => match corpus_build(&args[1..]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("rex corpus-build: {e}");
                ExitCode::from(1)
            }
        },
        _ => {
            eprintln!(
                "usage: rex <prereg|prereg-check|corpus-build|review|not-run|score|admit|admission-check|ledger|ladder|ratchet|challenge|b2|lane-kappa> (see the example docs)"
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
        println!("{} OK prereg_sha={}", prereg::SCHEME, c.prereg_sha());
        return ExitCode::SUCCESS;
    }
    for b in &bad {
        eprintln!("{} DRIFT {b}", prereg::SCHEME);
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
    let sketches: Vec<(String, Vec<u64>)> = all
        .iter()
        .filter(|(i, _)| i.split == Split::Test)
        .filter_map(|(i, d)| Some((i.id.clone(), sketch(d)?)))
        .collect();
    std::fs::write(
        format!("{CORPUS_DIR}/test-sketch-v1.txt"),
        render_sketches(&sketches),
    )
    .map_err(|e| e.to_string())?;
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

type Flags = BTreeMap<String, String>;

/// `--key value` pairs; a flag followed by another flag (or nothing) is `"1"`.
fn flags(a: &[String]) -> Result<Flags, String> {
    let mut f = Flags::new();
    let mut i = 0;
    while i < a.len() {
        let k = a[i]
            .strip_prefix("--")
            .ok_or_else(|| format!("expected --flag, got {:?}", a[i]))?;
        match a.get(i + 1).filter(|v| !v.starts_with("--")) {
            Some(v) => {
                f.insert(k.into(), v.clone());
                i += 2;
            }
            None => {
                f.insert(k.into(), "1".into());
                i += 1;
            }
        }
    }
    Ok(f)
}

fn need<'a>(f: &'a Flags, k: &str) -> Result<&'a str, String> {
    f.get(k)
        .map(String::as_str)
        .ok_or_else(|| format!("--{k} is required"))
}

fn split_of(f: &Flags) -> Result<Split, String> {
    match need(f, "split")? {
        "dev" => Ok(Split::Dev),
        "test" => Ok(Split::Test),
        s => Err(format!("--split {s}: dev or test")),
    }
}

fn arm_of(s: &str) -> Result<Arm, String> {
    serde_json::from_value(serde_json::Value::String(s.into()))
        .map_err(|_| format!("--arm {s}: apr-4b|apr-9b|haiku|agy"))
}

/// The corpus items and the analysis identity (prereg sha, corpus version).
fn corpus() -> Result<(Vec<Item>, String, String), String> {
    let jsonl = std::fs::read_to_string(format!("{CORPUS_DIR}/corpus-v1.jsonl"))
        .map_err(|e| e.to_string())?;
    let items = jsonl
        .lines()
        .map(|l| serde_json::from_str(l).map_err(|e| e.to_string()))
        .collect::<Result<Vec<Item>, _>>()?;
    let manifest = std::fs::read_to_string(format!("{CORPUS_DIR}/test-manifest-v1.txt"))
        .map_err(|e| e.to_string())?;
    let prereg_sha = prereg::locked_prereg_sha()
        .ok_or("no locked prereg sha")?
        .to_string();
    Ok((items, prereg_sha, corpus_version(&manifest)))
}

fn run_of(f: &Flags, prereg_sha: String, corpus_version: String) -> Result<Run, String> {
    Ok(Run {
        cell: need(f, "cell")?.into(),
        host: need(f, "host")?.into(),
        backend: need(f, "backend")?.into(),
        arm: arm_of(need(f, "arm")?)?,
        apr_tag: need(f, "apr-tag")?.into(),
        apr_sha256: need(f, "apr-sha")?.into(),
        model_id: need(f, "model-id")?.into(),
        weights_sha256: need(f, "weights-sha")?.into(),
        prompt: prereg::PROMPT_V1.into(),
        corpus_version,
        prereg_sha,
    })
}

/// Items of the split in the seeded run order (the rerun subset with --rerun).
fn ordered(items: &[Item], split: Split, rerun: bool) -> Vec<&Item> {
    let by_id: BTreeMap<&str, &Item> = items
        .iter()
        .filter(|i| i.split == split)
        .map(|i| (i.id.as_str(), i))
        .collect();
    let ids: Vec<String> = by_id.keys().map(|k| (*k).to_string()).collect();
    let mut order = run_order(&ids, SEED);
    if rerun {
        order = rerun_subset(&order);
    }
    order.iter().map(|id| by_id[id.as_str()]).collect()
}

fn append(path: &str, r: &Receipt) -> Result<(), String> {
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    writeln!(
        f,
        "{}",
        serde_json::to_string(r).map_err(|e| e.to_string())?
    )
    .map_err(|e| e.to_string())
}

fn review(f: &Flags) -> Result<(), String> {
    let (items, prereg_sha, cv) = corpus()?;
    let run = run_of(f, prereg_sha, cv)?;
    let (url, items_dir, out) = (need(f, "url")?, need(f, "items")?, need(f, "out")?);
    let rerun = f.contains_key("rerun");
    let mut todo = ordered(&items, split_of(f)?, rerun);
    if let Some(n) = f.get("limit") {
        todo.truncate(n.parse().map_err(|_| "--limit N")?);
    }
    let raw_dir = format!("raw/{}/{}", run.cell, need(f, "arm")?);
    std::fs::create_dir_all(format!("{out}/{raw_dir}")).map_err(|e| e.to_string())?;
    let mut tally: BTreeMap<String, usize> = BTreeMap::new();
    for (n, item) in todo.iter().enumerate() {
        let diff = std::fs::read_to_string(format!("{items_dir}/{}.diff", item.id))
            .map_err(|e| format!("{}: {e}", item.id))?;
        if sha256_hex(diff.as_bytes()) != item.diff_sha256 {
            return Err(format!(
                "{}: diff sha differs from the corpus (wrong items tar?)",
                item.id
            ));
        }
        let body = request_body(&run.model_id, &run.prompt, &diff);
        let when = utc_now();
        let reply = post(url, &body)?;
        let parsed = classify(&reply);
        let raw = format!(
            "{raw_dir}/{}{}.txt",
            item.id,
            if rerun { ".rerun" } else { "" }
        );
        std::fs::write(format!("{out}/{raw}"), &parsed.text).map_err(|e| e.to_string())?;
        let mut r = run.receipt(item, &body, &parsed, reply.wall_ms, &raw, &when);
        r.cold = n == 0 && f.contains_key("cold-first");
        r.rerun = rerun;
        append(&format!("{out}/receipts.jsonl"), &r)?;
        *tally.entry(format!("{:?}", r.verdict)).or_default() += 1;
    }
    println!("reviewed {} {tally:?}", todo.len());
    Ok(())
}

fn not_run(f: &Flags) -> Result<(), String> {
    let (items, prereg_sha, cv) = corpus()?;
    let run = run_of(f, prereg_sha, cv)?;
    let why: NotRun = serde_json::from_value(serde_json::Value::String(need(f, "why")?.into()))
        .map_err(|_| "--why NoDeclaredExecutor|Refused|ContextOverflow|ServeError")?;
    let out = need(f, "out")?;
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let when = utc_now();
    let todo = ordered(&items, split_of(f)?, false);
    for item in &todo {
        append(
            &format!("{out}/receipts.jsonl"),
            &run.not_run(item, why, &when),
        )?;
    }
    println!("not-run {} {why:?}", todo.len());
    Ok(())
}

/// What `score` prints.
#[derive(serde::Serialize)]
struct Report<'a> {
    cell: &'a str,
    arm: Arm,
    provenance: &'a str,
    prereg_sha: &'a str,
    corpus_version: &'a str,
    rejected_rows: usize,
    score: aprender_review_experiment::score::LaneScore,
    #[serde(skip_serializing_if = "Option::is_none")]
    pilot: Option<Projection>,
}

fn score_cmd(f: &Flags) -> Result<(), String> {
    let (items, prereg_sha, cv) = corpus()?;
    let path = need(f, "receipts")?;
    let signed = match f.get("pubkey") {
        Some(pk) => {
            verify_signed(path, pk)?;
            "signed"
        }
        None => "unsigned (exploratory)",
    };
    let (cell, arm, rerun) = (
        need(f, "cell")?.to_string(),
        arm_of(need(f, "arm")?)?,
        f.contains_key("rerun"),
    );
    let expected = ordered(&items, split_of(f)?, rerun);
    let lines = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let root = std::path::Path::new(path)
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_default();
    let expect = Expect {
        prereg_sha: &prereg_sha,
        corpus_version: &cv,
    };
    let (rows, rejected) = collect(
        &lines,
        expect,
        &expected,
        |r| r.cell == cell && r.arm == arm && r.rerun == rerun,
        |p| std::fs::read_to_string(root.join(p)).ok(),
    );
    let sc = score(&rows);
    let pilot = if f.contains_key("pilot") {
        if split_of(f)? != Split::Dev {
            return Err("--pilot projects from dev items: pass --split dev".into());
        }
        let ran: std::collections::BTreeSet<&str> = rows
            .iter()
            .filter(|s| s.verdict.executed())
            .map(|s| s.id.as_str())
            .collect();
        let (mut warm, mut cold) = (Vec::new(), Vec::new());
        for r in lines.lines().filter_map(|l| admissible(l, expect).ok()) {
            let mine = r.cell == cell && r.arm == arm && !r.rerun;
            if let (true, Some(t)) = (mine && ran.contains(r.item_id.as_str()), r.timings) {
                if r.cold {
                    cold.push(t.wall_ms);
                } else {
                    warm.push(t.wall_ms);
                }
            }
        }
        let test: Vec<&Item> = items.iter().filter(|i| i.split == Split::Test).collect();
        let defects = test.iter().filter(|i| i.class.is_defect()).count() as u64;
        Some(project(sc.recall, &warm, &cold, defects, test.len() as u64))
    } else {
        None
    };
    let out = Report {
        cell: &cell,
        arm,
        provenance: signed,
        prereg_sha: &prereg_sha,
        corpus_version: &cv,
        rejected_rows: rejected.len(),
        score: sc,
        pilot,
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?
    );
    for r in rejected.iter().take(5) {
        eprintln!("rejected line {}: {}", r.line, r.reasons.join("; "));
    }
    Ok(())
}

/// REX-04: append admission rows for one cell or all six.
fn admit(f: &Flags) -> Result<(), String> {
    use aprender_review_experiment::admission::{Row, Status, CELLS, SCHEME};
    let prereg_sha = prereg::locked_prereg_sha()
        .ok_or("no locked prereg sha")?
        .to_string();
    let status = match f.get("removed-by") {
        Some(r) => Status::Refused {
            removed_by: r.clone(),
        },
        None => Status::NotRun {
            reason: serde_json::from_value(serde_json::Value::String(need(f, "why")?.into()))
                .map_err(|_| "--why NoDeclaredExecutor|ServeError|…")?,
        },
    };
    let which = need(f, "cell")?;
    let cells: Vec<_> = CELLS
        .iter()
        .filter(|c| which == "all" || c.cell == which)
        .collect();
    if cells.is_empty() {
        return Err(format!("--cell {which}: not a §2.1 cell"));
    }
    let out = need(f, "out")?;
    let at = utc_now();
    let mut text = String::new();
    for c in &cells {
        let row = Row {
            schema: SCHEME.into(),
            cell: c.cell.into(),
            host: c.host.into(),
            backend: c.backend.into(),
            apr_tag: need(f, "apr-tag")?.into(),
            apr_sha256: need(f, "apr-sha")?.into(),
            model_id: need(f, "model-id")?.into(),
            weights_sha256: need(f, "weights-sha")?.into(),
            prereg_sha: prereg_sha.clone(),
            at: at.clone(),
            status: status.clone(),
        };
        text += &(serde_json::to_string(&row).map_err(|e| e.to_string())? + "\n");
    }
    use std::io::Write;
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out)
        .and_then(|mut h| h.write_all(text.as_bytes()))
        .map_err(|e| e.to_string())?;
    println!("admit {} row(s)", cells.len());
    Ok(())
}

/// REX-04: resolve an admission file. Exit 10 (an alarm, never a crash code)
/// when it is admissible but no cell is admitted (S-7).
fn admission_check(a: &[String]) -> ExitCode {
    let run = || -> Result<aprender_review_experiment::admission::Summary, Vec<String>> {
        let f = flags(a).map_err(|e| vec![e])?;
        let path = need(&f, "file").map_err(|e| vec![e])?;
        let text = std::fs::read_to_string(path).map_err(|e| vec![e.to_string()])?;
        let lock =
            prereg::locked_prereg_sha().ok_or_else(|| vec!["no locked prereg sha".into()])?;
        aprender_review_experiment::admission::check(&text, lock)
    };
    match run() {
        Ok(s) => {
            match serde_json::to_string(&s) {
                Ok(j) => println!("{j}"),
                Err(e) => eprintln!("rex admission-check: {e}"),
            }
            if s.s7 {
                eprintln!("S-7: no cell is admitted — no admissible cell for (A)");
                ExitCode::from(10)
            } else {
                ExitCode::SUCCESS
            }
        }
        Err(es) => {
            for e in es {
                eprintln!("rex admission-check: {e}");
            }
            ExitCode::from(1)
        }
    }
}

fn ledger_cmd(a: &[String]) -> ExitCode {
    let (Some(repo), Some(out), receipts) = (a.first(), a.get(1), a.get(2..).unwrap_or(&[])) else {
        eprintln!("usage: rex ledger REPO OUT_JSONL QUORUM_RECEIPT...");
        return ExitCode::from(2);
    };
    let mut read = Vec::new();
    for p in receipts {
        match std::fs::read_to_string(p) {
            Ok(t) => read.push((p.clone(), t)),
            Err(e) => {
                eprintln!("rex ledger: {p}: {e}");
                return ExitCode::from(1);
            }
        }
    }
    let (rows, c) = aprender_review_experiment::ledger::build(&read, repo);
    let lines: Result<String, _> = rows
        .iter()
        .map(|r| serde_json::to_string(r).map(|j| j + "\n"))
        .collect();
    match lines
        .map_err(|e| e.to_string())
        .and_then(|l| std::fs::write(out, l).map_err(|e| format!("{out}: {e}")))
    {
        Ok(()) => {}
        Err(e) => {
            eprintln!("rex ledger: {e}");
            return ExitCode::from(1);
        }
    }
    match serde_json::to_string(&c) {
        Ok(j) => println!("{j}"),
        Err(e) => eprintln!("rex ledger: {e}"),
    }
    if c.holds() {
        ExitCode::SUCCESS
    } else {
        eprintln!(
            "REX-07: {} of {} quorums carry a shadow row, {} violations",
            c.carried,
            c.quorums,
            c.violations.len()
        );
        ExitCode::from(10)
    }
}

fn ladder_cmd(a: &[String]) -> ExitCode {
    let Some(path) = a.first() else {
        eprintln!("usage: rex ladder REPORT_JSON");
        return ExitCode::from(2);
    };
    let Some(lock) = prereg::locked_prereg_sha() else {
        eprintln!("rex ladder: no locked prereg sha");
        return ExitCode::from(1);
    };
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("rex ladder: {path}: {e}");
            return ExitCode::from(1);
        }
    };
    let d = aprender_review_experiment::ladder::decide(&text, lock);
    match serde_json::to_string(&d) {
        Ok(j) => println!("{j}"),
        Err(e) => eprintln!("rex ladder: {e}"),
    }
    ExitCode::SUCCESS
}

/// Exit code for a lane-independence andon (S-14).
const ANDON_KAPPA: u8 = 12;

fn lane_kappa_cmd(a: &[String]) -> ExitCode {
    use aprender_review_experiment::kappa_probe::{gate, parse_manifest, parse_rows, probe};
    const USAGE: &str = "usage: rex lane-kappa ROWS --split val [--shadow L] (--min-n N | --manifest M --candidate ROWS2)";
    let (Some(path), Ok(f)) = (a.first(), flags(a.get(1..).unwrap_or_default())) else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    let Some(split) = f.get("split") else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    let shadow = f.get("shadow").map_or("qwen-shadow", String::as_str);
    let load = |p: &str| {
        std::fs::read_to_string(p)
            .map_err(|e| format!("{p}: {e}"))
            .and_then(|t| parse_rows(&t).map_err(|e| format!("{p}: {e}")))
    };
    let andon = |why: String| {
        eprintln!("rex lane-kappa: ANDON {why}");
        ExitCode::from(ANDON_KAPPA)
    };
    let manifest = match f.get("manifest").map(|m| {
        std::fs::read_to_string(m)
            .map_err(|e| format!("{m}: {e}"))
            .and_then(|t| parse_manifest(&t))
    }) {
        None => None,
        Some(Ok(m)) => Some(m),
        Some(Err(e)) => return andon(format!("S-14 manifest: {e}")),
    };
    if f.contains_key("candidate") && manifest.is_none() {
        return andon("S-14 --candidate needs --manifest: δ is not pre-registered".into());
    }
    let min_n = match (&manifest, f.get("min-n").map(|n| n.parse::<usize>())) {
        (Some(m), _) => m.min_n,
        (None, Some(Ok(n))) if n > 0 => n,
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    let baseline = match load(path) {
        Ok(r) => probe(&r, split, shadow, min_n),
        Err(e) => {
            eprintln!("rex lane-kappa: {e}");
            return ExitCode::from(1);
        }
    };
    let mut out = serde_json::Map::new();
    out.insert("baseline".into(), to_json(&baseline));
    let mut code = ExitCode::SUCCESS;
    if let (Some(m), Some(c)) = (&manifest, f.get("candidate")) {
        let candidate = match load(c) {
            Ok(r) => probe(&r, split, shadow, min_n),
            Err(e) => {
                eprintln!("rex lane-kappa: {e}");
                return ExitCode::from(1);
            }
        };
        let v = match gate(&baseline, &candidate, m) {
            Ok(v) => v,
            Err(e) => return andon(e),
        };
        if !v.pass {
            eprintln!("rex lane-kappa: ANDON {}", v.refusals.join("; "));
            code = ExitCode::from(ANDON_KAPPA);
        }
        out.insert("candidate".into(), to_json(&candidate));
        out.insert("manifest".into(), to_json(m));
        out.insert("gate".into(), to_json(&v));
    }
    println!("{}", serde_json::Value::Object(out));
    code
}

fn to_json<T: serde::Serialize>(x: &T) -> serde_json::Value {
    serde_json::to_value(x).unwrap_or(serde_json::Value::Null)
}

/// Warm, non-rerun, admissible timings of one cell (and tag, when given).
fn warm_timings(
    path: &str,
    cell: &str,
    tag: Option<&str>,
) -> Result<(Vec<Receipt>, String), String> {
    let (_, prereg_sha, cv) = corpus()?;
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let expect = Expect {
        prereg_sha: &prereg_sha,
        corpus_version: &cv,
    };
    let rows = text
        .lines()
        .filter_map(|l| admissible(l, expect).ok())
        .filter(|r| {
            r.cell == cell
                && tag.is_none_or(|t| r.apr_tag == t)
                && !r.cold
                && !r.rerun
                && r.timings.is_some()
        })
        .collect();
    Ok((rows, sha256_hex(text.as_bytes())))
}

fn samples(rows: &[Receipt]) -> Vec<ratchet::Sample<'_>> {
    rows.iter()
        .filter_map(|r| {
            r.timings.map(|t| ratchet::Sample {
                item_id: &r.item_id,
                apr_sha256: &r.apr_sha256,
                wall_ms: t.wall_ms,
            })
        })
        .collect()
}

fn ratchet_entries(file: &str) -> Result<Vec<ratchet::Entry>, String> {
    match std::fs::read_to_string(file) {
        Ok(t) => t
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).map_err(|e| format!("{file}: {e}")))
            .collect(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(format!("{file}: {e}")),
    }
}

fn ratchet_record(f: &Flags) -> Result<(), String> {
    let (file, tag, cell) = (need(f, "file")?, need(f, "tag")?, need(f, "cell")?);
    let (rows, sha) = warm_timings(need(f, "receipts")?, cell, Some(tag))?;
    let llama = match f.get("llama-receipts") {
        Some(p) => warm_timings(p, need(f, "llama-cell")?, None)?.0,
        None => Vec::new(),
    };
    let entry = ratchet::record(tag, cell, &samples(&rows), &samples(&llama), &sha)?;
    let mut all = ratchet_entries(file)?;
    all.push(entry.clone());
    let line = serde_json::to_string(&entry).map_err(|e| e.to_string())?;
    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(file)
        .map_err(|e| format!("{file}: {e}"))?;
    std::io::Write::write_all(&mut out, format!("{line}\n").as_bytes())
        .map_err(|e| e.to_string())?;
    println!(
        "recorded {tag} on {cell}: {} items, p95 {:.0} ms; andon {:?}",
        entry.items.len(),
        entry.p95_ms,
        ratchet::check(&all).andon
    );
    Ok(())
}

fn ratchet_cmd(a: &[String]) -> ExitCode {
    let run = |sub: &str| -> Result<ExitCode, String> {
        let f = flags(&a[1..])?;
        if sub == "record" {
            return ratchet_record(&f).map(|()| ExitCode::SUCCESS);
        }
        let v = ratchet::check(&ratchet_entries(need(&f, "file")?)?);
        println!(
            "{}",
            serde_json::to_string_pretty(&v).map_err(|e| e.to_string())?
        );
        Ok(if v.andon == ratchet::Andon::Red {
            ExitCode::from(10)
        } else {
            ExitCode::SUCCESS
        })
    };
    match a.first().map(String::as_str) {
        Some(s @ ("record" | "check")) => run(s).unwrap_or_else(|e| {
            eprintln!("rex ratchet {s}: {e}");
            ExitCode::from(1)
        }),
        _ => {
            eprintln!("usage: rex ratchet record|check --file F ... (see the example docs)");
            ExitCode::from(2)
        }
    }
}

/// REX-12: `b2 status` (exit 11 while any B2 row is NotRun) and
/// `b2 teacher-receipt`.
fn b2_cmd(a: &[String]) -> ExitCode {
    let run = |sub: &str| -> Result<ExitCode, String> {
        let f = flags(&a[1..])?;
        if sub == "status" {
            let path = f
                .get("refusals")
                .map_or("evidence/verbs/refusals.json", String::as_str);
            let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
            let ledger = b2::parse_ledger(&text).map_err(|e| format!("{path}: {e}"))?;
            let mut accepted = BTreeMap::new();
            for kv in f.get("accepted").map_or("", String::as_str).split(',') {
                if kv.is_empty() {
                    continue;
                }
                let (verb, receipt) = kv
                    .split_once('=')
                    .ok_or_else(|| format!("--accepted {kv}: verb=receipt"))?;
                accepted.insert(verb.to_string(), receipt.to_string());
            }
            let rows = b2::status(&ledger, &accepted)?;
            for r in &rows {
                println!("{}", serde_json::to_string(r).map_err(|e| e.to_string())?);
            }
            let ready = rows
                .iter()
                .all(|r| matches!(r.state, b2::State::Ready { .. }));
            return Ok(if ready {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(11)
            });
        }
        let (items, _, _) = corpus()?;
        let manifest = std::fs::read_to_string(format!("{CORPUS_DIR}/test-manifest-v1.txt"))
            .map_err(|e| e.to_string())?;
        let sealed = parse_manifest(&manifest).ok_or("test manifest does not parse")?;
        let index = match std::fs::read_to_string(format!("{CORPUS_DIR}/test-sketch-v1.txt")) {
            Ok(t) => Index::new(&sealed)
                .with_sketches(parse_sketches(&t).ok_or("test sketch file does not parse")?),
            Err(_) => {
                eprintln!(
                    "no test-sketch-v1.txt: exact-hash contamination only (no cluster check)"
                );
                Index::new(&sealed)
            }
        };
        let path = need(&f, "logits")?;
        let jsonl = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let k = need(&f, "k")?.parse().map_err(|e| format!("--k: {e}"))?;
        match b2::teacher_receipt(&jsonl, need(&f, "teacher-sha")?, k, &items, &index) {
            Ok(r) => {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&r).map_err(|e| e.to_string())?
                );
                Ok(ExitCode::SUCCESS)
            }
            Err(e) => Err(e.join("\n")),
        }
    };
    match a.first().map(String::as_str) {
        Some(s @ ("status" | "teacher-receipt")) => run(s).unwrap_or_else(|e| {
            eprintln!("rex b2 {s}: {e}");
            ExitCode::from(1)
        }),
        _ => {
            eprintln!("usage: rex b2 status|teacher-receipt ... (see the example docs)");
            ExitCode::from(2)
        }
    }
}

/// `minisign -V` must pass on `path` under `pk`.
fn verify_signed(path: &str, pk: &str) -> Result<(), String> {
    let ok = std::process::Command::new("minisign")
        .args(["-Vqm", path, "-p", pk])
        .status()
        .map_err(|e| format!("minisign: {e}"))?
        .success();
    if ok {
        Ok(())
    } else {
        Err(format!("{path}: signature does not verify under {pk}"))
    }
}

/// One lane's scored rows on the sealed test split (first runs only).
fn test_rows(path: &str, pk: &str, cell: &str, arm: Arm) -> Result<Vec<Scored>, String> {
    verify_signed(path, pk)?;
    let (items, prereg_sha, cv) = corpus()?;
    let lines = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let root = std::path::Path::new(path)
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_default();
    let expect = Expect {
        prereg_sha: &prereg_sha,
        corpus_version: &cv,
    };
    let (rows, rejected) = collect(
        &lines,
        expect,
        &ordered(&items, Split::Test, false),
        |r| r.cell == cell && r.arm == arm && !r.rerun,
        |p| std::fs::read_to_string(root.join(p)).ok(),
    );
    if !rejected.is_empty() {
        eprintln!("{path}: {} rejected rows", rejected.len());
    }
    Ok(rows)
}

fn opt_bool(f: &Flags, k: &str) -> Result<Option<bool>, String> {
    match f.get(k).map(String::as_str) {
        None => Ok(None),
        Some("holds") => Ok(Some(true)),
        Some("fails") => Ok(Some(false)),
        Some(v) => Err(format!("--{k} {v}: holds|fails")),
    }
}

fn opt_num<T: std::str::FromStr>(f: &Flags, k: &str) -> Result<Option<T>, String> {
    f.get(k)
        .map(|v| v.parse().map_err(|_| format!("--{k} {v}: not a number")))
        .transpose()
}

fn challenge(f: &Flags) -> Result<champion::Outcome, String> {
    let file = need(f, "ledger")?;
    let ledger: Vec<champion::Decision> = match std::fs::read_to_string(file) {
        Ok(t) => t
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).map_err(|e| format!("{file}: {e}")))
            .collect::<Result<_, _>>()?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(e) => return Err(format!("{file}: {e}")),
    };
    let pin = |w: &str, p: &str, a: &str| -> Result<champion::Pin, String> {
        Ok(champion::Pin {
            weights_sha256: need(f, w)?.into(),
            prompt_sha256: need(f, p)?.into(),
            adapter_sha256: f.get(a).cloned(),
        })
    };
    let initial = pin(
        "initial-weights-sha",
        "initial-prompt-sha",
        "initial-adapter-sha",
    )?;
    let (pk, cell) = (need(f, "pubkey")?, need(f, "cell")?);
    let ch = test_rows(
        need(f, "challenger-receipts")?,
        pk,
        cell,
        arm_of(need(f, "challenger-arm")?)?,
    )?;
    let cm = test_rows(
        need(f, "champion-receipts")?,
        pk,
        cell,
        arm_of(need(f, "champion-arm")?)?,
    )?;
    let e = champion::Evidence {
        id: need(f, "id")?,
        tier: need(f, "tier")?,
        pin: pin("weights-sha", "prompt-sha", "adapter-sha")?,
        test_version: need(f, "test-version")?,
        challenger: &ch,
        champion: &cm,
        h1_holds: opt_bool(f, "h1")?,
        h2_holds: opt_bool(f, "h2")?,
        contamination_hits: opt_num(f, "contamination-hits")?,
        p95_s: opt_num(f, "p95-s")?,
        queue_budget_p95_s: opt_num(f, "budget-p95-s")?,
    };
    let d = champion::decide(&ledger, &initial, &e);
    let line = serde_json::to_string(&d).map_err(|e| e.to_string())?;
    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(file)
        .map_err(|e| format!("{file}: {e}"))?;
    std::io::Write::write_all(&mut out, format!("{line}\n").as_bytes())
        .map_err(|e| e.to_string())?;
    println!(
        "{}",
        serde_json::to_string_pretty(&d).map_err(|e| e.to_string())?
    );
    Ok(d.outcome)
}

fn challenge_cmd(a: &[String]) -> ExitCode {
    match flags(a).and_then(|f| challenge(&f)) {
        Ok(champion::Outcome::Promoted) => ExitCode::SUCCESS,
        Ok(champion::Outcome::Rejected) => ExitCode::from(10),
        Ok(champion::Outcome::Refused) => ExitCode::from(11),
        Err(e) => {
            eprintln!("rex challenge: {e}");
            ExitCode::from(1)
        }
    }
}
