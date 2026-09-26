//! `cargo run -p aprender-review-experiment --example replay -- <command>`
//!
//! PRM-S1: the `review-replay-v1` speed benchmark (contract `review-replay-v1`,
//! library `replay.rs`). An example, not a `[[bin]]` (see `rex.rs`).
//!
//! Commands:
//! - `build --version V --candidates F --tokenize-url U --diffs-out DIR --out SET
//!   [--per-stratum N]` F is JSONL `{"group":"owner/repo#N","diff":"path"}`, one
//!   line per quorum diff (inputs only). Each diff is composed with the review
//!   prompt, templated by `U/apply-template` and counted by `U/tokenize` (a
//!   `llama-server` on the replay GGUF). The sealed test manifest and its sketch
//!   file are required. Writes the set and `DIR/<diff_sha>.diff` for the
//!   chosen items, and prints counts and the set sha only.
//! - `run --set SET --diffs DIR --url U --engine apr|llama_cpp --engine-version X
//!   --apr-tag T --cell C --gguf-sha G --model M --out ROWS [--server-pid P]`
//!   replays every item once through the chat-completions endpoint `U` and
//!   appends one `review-replay-receipt-v1` row per item. `--server-pid` adds
//!   the serve's VmHWM (peak RSS so far) to each row.
//! - `sketch --items DIR` writes `test-sketch-v1.txt` (the near-dup sketches
//!   `build` needs) from the corpus items in DIR, e.g. the unpacked
//!   `items-v1.tar`. Every test-split diff must hash to its corpus
//!   `diff_sha256`, or nothing is written. Prints counts only.
//! - `voters DIR` prints `lane seconds` for every
//!   `predicate.consultations.<lane>.duration_seconds` in `DIR/*/*/receipt.intoto.jsonl`
//!   (quorum receipts under `evidence/pr-review/`).
//! - `summary --rows F --voter LANE=FILE... [--prev F]` the §9 speed block as
//!   JSON (valid YAML). A voter FILE holds one latency in seconds per line.
//!   Exit 1 when the run is refused, and the error names the reason.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::process::ExitCode;

use aprender_review_experiment::cluster::{parse_sketches, render_sketches, sketch};
use aprender_review_experiment::contamination::Index;
use aprender_review_experiment::corpus::{parse_manifest, sha256_hex, Item, Split};
use aprender_review_experiment::harness::{classify, compose, post, request_body};
use aprender_review_experiment::prereg::PROMPT_V1;
use aprender_review_experiment::replay::{
    build, summarize, BuildError, Candidate, Engine, Row, Set, PER_STRATUM, ROW_SCHEMA,
};
use serde_json::Value;

const SEED: u64 = 4354;
const CORPUS_DIR: &str = "docs/audits/review-corpus";

type Flags = BTreeMap<String, Vec<String>>;

fn main() -> ExitCode {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let r = match a.first().map(String::as_str) {
        Some("build") => flags(&a[1..]).and_then(|f| cmd_build(&f)),
        Some("run") => flags(&a[1..]).and_then(|f| cmd_run(&f)),
        Some("sketch") => flags(&a[1..]).and_then(|f| cmd_sketch(&f)),
        Some("voters") => a
            .get(1)
            .ok_or_else(|| "voters DIR".to_string())
            .and_then(|d| cmd_voters(d)),
        Some("summary") => flags(&a[1..]).and_then(|f| cmd_summary(&f)),
        _ => Err("usage: replay build|run|sketch|voters|summary (see the example's docs)".into()),
    };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("replay: {e}");
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
    f.get(k)
        .and_then(|v| v.last())
        .map(String::as_str)
        .ok_or_else(|| format!("--{k} is required"))
}

fn read(path: &str) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))
}

fn sealed_index() -> Result<Index, String> {
    let manifest = parse_manifest(&read(&format!("{CORPUS_DIR}/test-manifest-v1.txt"))?)
        .ok_or("test manifest does not parse")?;
    let sketches = parse_sketches(&read(&format!("{CORPUS_DIR}/test-sketch-v1.txt"))?)
        .ok_or("test sketch file does not parse")?;
    Ok(Index::new(&manifest).with_sketches(sketches))
}

fn post_json(url: &str, body: &Value) -> Result<Value, String> {
    let reply = post(url, body)?;
    if reply.status != 200 {
        return Err(format!("{url}: HTTP {}", reply.status));
    }
    serde_json::from_str(&reply.body).map_err(|e| format!("{url}: {e}"))
}

/// Composed-prompt tokens under the serve's chat template.
fn count_tokens(base: &str, diff: &str) -> Result<u64, String> {
    let msgs = Value::Array(vec![Value::Object(
        [
            ("role".to_string(), Value::from("user")),
            ("content".to_string(), Value::from(compose(PROMPT_V1, diff))),
        ]
        .into_iter()
        .collect(),
    )]);
    let body = Value::Object([("messages".to_string(), msgs)].into_iter().collect());
    let templated = post_json(&format!("{base}/apply-template"), &body)?;
    let prompt = templated["prompt"]
        .as_str()
        .ok_or("apply-template: no prompt")?;
    let body = Value::Object(
        [
            ("content".to_string(), Value::from(prompt)),
            ("add_special".to_string(), Value::from(true)),
        ]
        .into_iter()
        .collect(),
    );
    let toks = post_json(&format!("{base}/tokenize"), &body)?;
    toks["tokens"]
        .as_array()
        .map(|t| t.len() as u64)
        .ok_or_else(|| "tokenize: no tokens".to_string())
}

fn cmd_build(f: &Flags) -> Result<(), String> {
    let index = sealed_index()?;
    let base = need(f, "tokenize-url")?.trim_end_matches('/');
    let per = match f.get("per-stratum") {
        Some(_) => need(f, "per-stratum")?
            .parse()
            .map_err(|e| format!("--per-stratum: {e}"))?,
        None => PER_STRATUM,
    };
    let mut cands = Vec::new();
    for (n, line) in read(need(f, "candidates")?)?.lines().enumerate() {
        let v: Value =
            serde_json::from_str(line).map_err(|e| format!("candidates:{}: {e}", n + 1))?;
        let (Some(group), Some(path)) = (v["group"].as_str(), v["diff"].as_str()) else {
            return Err(format!("candidates:{}: needs group and diff", n + 1));
        };
        let diff = read(path)?;
        cands.push(Candidate {
            group: group.to_string(),
            diff_sha256: sha256_hex(diff.as_bytes()),
            input_tokens: count_tokens(base, &diff)?,
            diff,
        });
    }
    let set = match build(need(f, "version")?, SEED, per, &cands, &index) {
        Ok(s) => s,
        Err(BuildError::EmptySealedIndex) => return Err("sealed test manifest is empty".into()),
        Err(BuildError::Short(s)) => {
            let s: Vec<String> = s
                .iter()
                .map(|x| format!("{} {}/{}", x.stratum.as_str(), x.have, x.need))
                .collect();
            return Err(format!("short strata (never padded): {}", s.join(", ")));
        }
    };
    let dir = need(f, "diffs-out")?;
    std::fs::create_dir_all(dir).map_err(|e| format!("{dir}: {e}"))?;
    for it in &set.items {
        let c = cands
            .iter()
            .find(|c| c.diff_sha256 == it.diff_sha256)
            .ok_or("lost a diff")?;
        let p = format!("{dir}/{}.diff", it.diff_sha256);
        std::fs::write(&p, &c.diff).map_err(|e| format!("{p}: {e}"))?;
    }
    let out = need(f, "out")?;
    std::fs::write(out, set.render()).map_err(|e| format!("{out}: {e}"))?;
    println!(
        "{{\"candidates\":{},\"items\":{},\"set_sha\":\"{}\"}}",
        cands.len(),
        set.items.len(),
        set.sha()
    );
    Ok(())
}

fn vm_hwm_mb(pid: &str) -> Option<f64> {
    let s = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    let kb: f64 = s
        .lines()
        .find_map(|l| l.strip_prefix("VmHWM:"))?
        .trim()
        .trim_end_matches("kB")
        .trim()
        .parse()
        .ok()?;
    Some(kb / 1024.0)
}

fn cmd_run(f: &Flags) -> Result<(), String> {
    let text = read(need(f, "set")?)?;
    let set = Set::parse(&text)?;
    let set_sha = sha256_hex(text.as_bytes());
    let engine = match need(f, "engine")? {
        "apr" => Engine::Apr,
        "llama_cpp" => Engine::LlamaCpp,
        e => return Err(format!("--engine {e}: want apr|llama_cpp")),
    };
    let (url, model, dir) = (need(f, "url")?, need(f, "model")?, need(f, "diffs")?);
    let pid = f.get("server-pid").and_then(|v| v.last());
    let out_path = need(f, "out")?;
    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out_path)
        .map_err(|e| format!("{out_path}: {e}"))?;
    for it in &set.items {
        let diff = read(&format!("{dir}/{}.diff", it.diff_sha256))?;
        if sha256_hex(diff.as_bytes()) != it.diff_sha256 {
            return Err(format!(
                "{dir}/{}.diff does not hash to its name",
                it.diff_sha256
            ));
        }
        let reply = post(url, &request_body(model, PROMPT_V1, &diff))?;
        let p = classify(&reply);
        let row = Row {
            schema: ROW_SCHEMA.into(),
            replay_version: set.version.clone(),
            set_sha: set_sha.clone(),
            diff_sha256: it.diff_sha256.clone(),
            stratum: it.stratum,
            engine,
            engine_version: need(f, "engine-version")?.into(),
            apr_tag: need(f, "apr-tag")?.into(),
            cell: need(f, "cell")?.into(),
            gguf_sha256: need(f, "gguf-sha")?.into(),
            wall_ms: reply.wall_ms,
            prompt_ms: p.server.map(|s| s.prompt_ms),
            prompt_tps: p.server.and_then(|s| s.prompt_per_second),
            decode_tps: p.server.and_then(|s| s.predicted_per_second),
            input_tokens: p.tokens.map(|t| t.prompt),
            output_tokens: p.tokens.map(|t| t.completion),
            peak_rss_mb: pid.and_then(|p| vm_hwm_mb(p)),
            verdict: p.verdict,
        };
        let line = serde_json::to_string(&row).map_err(|e| e.to_string())?;
        writeln!(out, "{line}").map_err(|e| format!("{out_path}: {e}"))?;
    }
    println!("{{\"rows\":{},\"set_sha\":\"{set_sha}\"}}", set.items.len());
    Ok(())
}

fn cmd_voters(dir: &str) -> Result<(), String> {
    let mut n = 0;
    for pr in std::fs::read_dir(dir)
        .map_err(|e| format!("{dir}: {e}"))?
        .flatten()
    {
        for head in std::fs::read_dir(pr.path()).into_iter().flatten().flatten() {
            let Ok(text) = std::fs::read_to_string(head.path().join("receipt.intoto.jsonl")) else {
                continue;
            };
            for line in text.lines() {
                let Ok(v) = serde_json::from_str::<Value>(line) else {
                    continue;
                };
                let Some(c) = v["predicate"]["consultations"].as_object() else {
                    continue;
                };
                for (lane, body) in c {
                    if let Some(s) = body["duration_seconds"].as_f64() {
                        println!("{lane} {s}");
                        n += 1;
                    }
                }
            }
        }
    }
    if n == 0 {
        return Err(format!("{dir}: no consultation latencies"));
    }
    Ok(())
}

fn rows_of(path: &str) -> Result<Vec<Row>, String> {
    read(path)?
        .lines()
        .enumerate()
        .map(|(i, l)| serde_json::from_str(l).map_err(|e| format!("{path}:{}: {e}", i + 1)))
        .collect()
}

fn cmd_summary(f: &Flags) -> Result<(), String> {
    let rows = rows_of(need(f, "rows")?)?;
    let prev = match f.get("prev") {
        Some(_) => Some(rows_of(need(f, "prev")?)?),
        None => None,
    };
    let mut voters = Vec::new();
    for v in f.get("voter").into_iter().flatten() {
        let (lane, path) = v
            .split_once('=')
            .ok_or_else(|| format!("--voter {v}: want LANE=FILE"))?;
        let xs = read(path)?
            .split_whitespace()
            .map(|x| x.parse::<f64>().map_err(|e| format!("{path}: {x}: {e}")))
            .collect::<Result<Vec<_>, _>>()?;
        voters.push((lane.to_string(), xs));
    }
    let speed =
        summarize(&rows, &voters, prev.as_deref(), SEED).map_err(|e| format!("refused: {e:?}"))?;
    println!(
        "{}",
        serde_json::to_string_pretty(&speed).map_err(|e| e.to_string())?
    );
    Ok(())
}

/// `sketch`: the test items' near-dup sketches, written only when every
/// test-split diff in DIR is byte-identical to the one the corpus sealed.
fn cmd_sketch(f: &Flags) -> Result<(), String> {
    let dir = need(f, "items")?;
    let mut out: Vec<(String, Vec<u64>)> = Vec::new();
    let mut tests = 0usize;
    for line in read(&format!("{CORPUS_DIR}/corpus-v1.jsonl"))?.lines() {
        let item: Item = serde_json::from_str(line).map_err(|e| e.to_string())?;
        if item.split != Split::Test {
            continue;
        }
        tests += 1;
        let diff = read(&format!("{dir}/{}.diff", item.id))?;
        if sha256_hex(diff.as_bytes()) != item.diff_sha256 {
            return Err(format!("{}: diff sha differs from the corpus", item.id));
        }
        if let Some(s) = sketch(&diff) {
            out.push((item.id, s));
        }
    }
    if tests == 0 {
        return Err("the corpus has no test items".into());
    }
    let path = format!("{CORPUS_DIR}/test-sketch-v1.txt");
    std::fs::write(&path, render_sketches(&out)).map_err(|e| format!("{path}: {e}"))?;
    println!("test_items {tests} sketched {}", out.len());
    Ok(())
}
