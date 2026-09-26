//! `cargo run -p aprender-review-experiment --example crux_perf -- <command>`
//!
//! CRUX perf history (contract `crux-perf-receipt-v1`, library `crux_perf.rs`; Refs infra#1057).
//! `scripts/perf_trace_receipt.sh` drives it; an example, not a `[[bin]]` (see `rex.rs`).
//!
//! Commands:
//! - `receipt --rows F --tag T --rc-sha S --host H --gpu G --driver-cuda D --model M
//!   --gguf-sha G --quant Q --ctx N --batch N --competitor NAME --competitor-version V
//!   --competitor-sha S [--t1-sha X] [--t2-sha X] [--trace-overhead-pct P]
//!   [--tag-asset-sha S]` folds one PRM-S1 replay run (`rows-both.jsonl`: apr and the
//!   same-session competitor) into one receipt, printed as JSON. The prompt set is the rows'
//!   `set_sha`. With `--tag-asset-sha`, exit 1 unless the receipt holds against it.
//! - `gate --cur F [--prev F] [--released F] [--last3 F --last3 F --last3 F]
//!   --tag-asset-sha S` prints the gate; exit 1 when RED or refused.
//! - `coverage --tag T --tag-asset-sha S --cell ID... --receipt F...` (G3): exit 1 when any
//!   admitted cell lacks T0 or T1 at the tag.

use std::collections::BTreeMap;
use std::process::ExitCode;

use aprender_review_experiment::crux_perf::{
    coverage_gaps, gate, Cell, Competitor, CruxPerfReceipt, Phases,
};
use aprender_review_experiment::replay::{Engine, Row};

const SEED: u64 = 1057;

type Flags = BTreeMap<String, Vec<String>>;

fn main() -> ExitCode {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let r = match a.first().map(String::as_str) {
        Some("receipt") => flags(&a[1..]).and_then(|f| cmd_receipt(&f)),
        Some("gate") => flags(&a[1..]).and_then(|f| cmd_gate(&f)),
        Some("coverage") => flags(&a[1..]).and_then(|f| cmd_coverage(&f)),
        _ => Err("usage: crux_perf receipt|gate|coverage (see the example's docs)".into()),
    };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("crux_perf: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `--key value` pairs; a key may repeat.
fn flags(a: &[String]) -> Result<Flags, String> {
    let mut f = Flags::new();
    let mut it = a.iter();
    while let Some(k) = it.next() {
        let k = k
            .strip_prefix("--")
            .ok_or_else(|| format!("expected --flag, got {k:?}"))?;
        let v = it.next().ok_or_else(|| format!("--{k} needs a value"))?;
        f.entry(k.to_string()).or_default().push(v.clone());
    }
    Ok(f)
}

fn need<'a>(f: &'a Flags, k: &str) -> Result<&'a str, String> {
    opt(f, k).ok_or_else(|| format!("--{k} is required"))
}

fn opt<'a>(f: &'a Flags, k: &str) -> Option<&'a str> {
    f.get(k).and_then(|v| v.last()).map(String::as_str)
}

fn num<T: std::str::FromStr>(f: &Flags, k: &str) -> Result<T, String> {
    need(f, k)?
        .parse()
        .map_err(|_| format!("--{k} is not a number"))
}

fn read(path: &str) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))
}

fn load(path: &str) -> Result<CruxPerfReceipt, String> {
    serde_json::from_str(&read(path)?).map_err(|e| format!("{path}: {e}"))
}

fn cmd_receipt(f: &Flags) -> Result<(), String> {
    let rows: Vec<Row> = read(need(f, "rows")?)?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(|e| format!("row: {e}")))
        .collect::<Result<_, _>>()?;
    let set_sha = rows.first().map(|r| r.set_sha.clone()).ok_or("no rows")?;
    let apr = Phases::from_replay_rows(&rows, Engine::Apr, SEED)
        .map_err(|e| format!("apr phases: {e:?}"))?;
    let competitor_phases = Phases::from_replay_rows(&rows, Engine::LlamaCpp, SEED)
        .map_err(|e| format!("competitor phases: {e:?}"))?;
    let cell = Cell {
        host: need(f, "host")?.to_string(),
        gpu: need(f, "gpu")?.to_string(),
        driver_cuda: need(f, "driver-cuda")?.to_string(),
        model: need(f, "model")?.to_string(),
        gguf_sha256: need(f, "gguf-sha")?.to_string(),
        quant: need(f, "quant")?.to_string(),
        ctx: num(f, "ctx")?,
        batch: num(f, "batch")?,
        prompt_set_sha: set_sha,
    };
    let mut r = CruxPerfReceipt::new(
        need(f, "tag")?.to_string(),
        need(f, "rc-sha")?.to_string(),
        cell,
        Competitor {
            name: need(f, "competitor")?.to_string(),
            version: opt(f, "competitor-version").map(str::to_string),
            binary_sha256: need(f, "competitor-sha")?.to_string(),
        },
        apr,
        competitor_phases,
    );
    r.t1_topk_sha = opt(f, "t1-sha").map(str::to_string);
    r.t2_blob_sha = opt(f, "t2-sha").map(str::to_string);
    r.trace_overhead_pct = opt(f, "trace-overhead-pct")
        .map(|p| {
            p.parse()
                .map_err(|_| "--trace-overhead-pct is not a number")
        })
        .transpose()?;
    println!(
        "{}",
        serde_json::to_string_pretty(&r).map_err(|e| e.to_string())?
    );
    eprintln!("crux_perf: t2 {:?}", r.t2_status());
    if let Some(sha) = opt(f, "tag-asset-sha") {
        r.check(sha)
            .map_err(|e| format!("receipt refused: {e:?}"))?;
    }
    Ok(())
}

fn cmd_gate(f: &Flags) -> Result<(), String> {
    let sha = need(f, "tag-asset-sha")?;
    let cur = load(need(f, "cur")?)?;
    cur.check(sha)
        .map_err(|e| format!("current receipt refused: {e:?}"))?;
    let prev = opt(f, "prev").map(load).transpose()?;
    let released = opt(f, "released").map(load).transpose()?;
    let last3: Vec<CruxPerfReceipt> = f
        .get("last3")
        .into_iter()
        .flatten()
        .map(|p| load(p))
        .collect::<Result<_, _>>()?;
    let last3: Vec<&CruxPerfReceipt> = last3.iter().collect();
    let g = gate(&cur, prev.as_ref(), released.as_ref(), &last3)
        .map_err(|e| format!("gate refused: {e:?}"))?;
    for r in &g.reasons {
        println!("RED {:?} {:?} +{:.1}%", r.phase, r.rule, r.delta * 100.0);
    }
    if g.red() {
        return Err(format!("{} RED at {}", cur.cell_id, cur.tag));
    }
    println!("GREEN {} at {}", cur.cell_id, cur.tag);
    Ok(())
}

fn cmd_coverage(f: &Flags) -> Result<(), String> {
    let cells: Vec<String> = f.get("cell").cloned().unwrap_or_default();
    if cells.is_empty() {
        return Err("no admitted --cell: coverage over nothing is not GREEN".into());
    }
    let receipts: Vec<CruxPerfReceipt> = f
        .get("receipt")
        .into_iter()
        .flatten()
        .map(|p| load(p))
        .collect::<Result<_, _>>()?;
    let tag = need(f, "tag")?;
    let gaps = coverage_gaps(tag, need(f, "tag-asset-sha")?, &cells, &receipts);
    for g in &gaps {
        println!("RED {g:?}");
    }
    if gaps.is_empty() {
        println!("GREEN {tag}: T0+T1 for {} admitted cell(s)", cells.len());
        Ok(())
    } else {
        Err(format!("{tag}: {} admitted cell gap(s)", gaps.len()))
    }
}
