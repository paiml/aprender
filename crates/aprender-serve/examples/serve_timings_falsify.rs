//! SRV-TIM-001 live falsifier: FALSIFY-SRV-TIM-001/002/003/005/006/007 (#4490, APR-OBS-001 OBS-03).
//!
//! Starts `apr serve run <model> --timings-log <out>/timings.jsonl`, sends chat and completions
//! requests (stream and non-stream, 1 and 32 tokens), then checks every response against
//! `contracts/apr-serve-timings-v1.yaml`:
//!
//! - F1: the response (or the SSE terminal chunk) carries a non-null `timings`.
//! - F2: `timings.prompt_n / predicted_n == usage.prompt_tokens / completion_tokens`.
//! - F3: `prompt_ms > 0`, `predicted_ms > 0` (unless `predicted_n == 0`), and
//!   `prompt_ms + predicted_ms <= wall_ms + 1`.
//! - F5: the server's `[request] {json}` stderr line for that request id carries the same
//!   `prefill_ms / decode_ms / prompt_n / predicted_n`.
//! - F6: the `--timings-log` JSONL line for that request id carries the same values, plus
//!   `build` and `host`.
//! - F7: the X-Request-ID the client sent is echoed on the response and appears as
//!   `client_request_id` in both the stderr line and the JSONL line.
//! - BE: both log lines name `--expect-backend` (the device this run was started on) as
//!   `backend`, or `unreported` where the server cannot tell; naming the other device is RED.
//!
//! Exits 0 iff every cell is GREEN. A cell is never skipped: a missing line is RED.
//!
//! ```text
//! cargo run -p aprender-serve --example serve_timings_falsify -- \
//!     --apr "$APR" --model M.gguf --out /tmp/srvtim --expect-backend gpu -- --gpu-layers all
//! # GPU hosts: --prefix "flock /tmp/apr-gpu.lock"
//! cargo test -p aprender-serve --example serve_timings_falsify   # the checks' case table
//! ```

use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitCode, Stdio};
use std::time::{Duration, Instant};

const LOG_PREFIX: &str = "[request] ";
const CHAT_PROMPT: &str = "Explain in two sentences why the sky is blue.";
const COMPLETION_PROMPT: &str = "The sky is blue because";

struct Args {
    apr: String,
    model: String,
    out: PathBuf,
    port: u16,
    prefix: Vec<String>,
    health_seconds: u64,
    expect_backend: String,
    serve_args: Vec<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let (mut apr, mut model, mut out) = (None, None, None);
    let (mut port, mut prefix, mut health_seconds) = (18431u16, Vec::new(), 600u64);
    let mut expect_backend = None;
    let mut serve_args = Vec::new();
    while let Some(a) = it.next() {
        let mut val = |name: &str| it.next().ok_or(format!("{name} needs a value"));
        match a.as_str() {
            "--apr" => apr = Some(val("--apr")?),
            "--model" => model = Some(val("--model")?),
            "--out" => out = Some(PathBuf::from(val("--out")?)),
            "--port" => port = val("--port")?.parse().map_err(|e| format!("--port: {e}"))?,
            "--prefix" => {
                prefix = val("--prefix")?
                    .split_whitespace()
                    .map(String::from)
                    .collect()
            },
            "--health-seconds" => {
                health_seconds = val("--health-seconds")?
                    .parse()
                    .map_err(|e| format!("--health-seconds: {e}"))?;
            },
            "--expect-backend" => expect_backend = Some(val("--expect-backend")?),
            "--" => serve_args.extend(it.by_ref()),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(Args {
        apr: apr.ok_or("--apr is required (the pinned $APR)")?,
        model: model.ok_or("--model is required")?,
        out: out.ok_or("--out is required")?,
        port,
        prefix,
        health_seconds,
        expect_backend: expect_backend
            .filter(|b| b == "cpu" || b == "gpu")
            .ok_or("--expect-backend cpu|gpu is required: the device this run was started on")?,
        serve_args,
    })
}

/// `(id, timings, usage)` from a JSON body or from an SSE stream's chunks; a later chunk's
/// non-null field wins, as the terminal chunk carries `timings` and `usage`.
fn parse_body(raw: &str) -> (Option<String>, Option<Value>, Option<Value>) {
    let field = |o: &Value, k: &str| o.get(k).filter(|v| !v.is_null()).cloned();
    if raw.trim_start().starts_with('{') {
        let o: Value = serde_json::from_str(raw).unwrap_or(Value::Null);
        let id = o.get("id").and_then(Value::as_str).map(String::from);
        return (id, field(&o, "timings"), field(&o, "usage"));
    }
    let (mut id, mut timings, mut usage) = (None, None, None);
    for line in raw.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data.trim() == "[DONE]" {
            continue;
        }
        let Ok(o) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        if id.is_none() {
            id = o.get("id").and_then(Value::as_str).map(String::from);
        }
        timings = field(&o, "timings").or(timings);
        usage = field(&o, "usage").or(usage);
    }
    (id, timings, usage)
}

/// `request_id -> record`, from `[request] ` stderr lines (`strip_prefix`) or JSONL lines.
/// A line that is not JSON is skipped; the missing record then turns its cell RED.
fn records(lines: impl Iterator<Item = String>, strip_prefix: bool) -> HashMap<String, Value> {
    let mut out = HashMap::new();
    for line in lines {
        let body = if strip_prefix {
            match line.strip_prefix(LOG_PREFIX) {
                Some(b) => b,
                None => continue,
            }
        } else {
            line.as_str()
        };
        if let Ok(rec) = serde_json::from_str::<Value>(body.trim()) {
            if let Some(id) = rec.get("request_id").and_then(Value::as_str) {
                out.insert(id.to_string(), rec);
            }
        }
    }
    out
}

/// A log record agrees with the response's timings on all four measured values.
fn same(rec: Option<&Value>, t: Option<&Value>) -> bool {
    let (Some(rec), Some(t)) = (rec, t) else {
        return false;
    };
    let pairs = [
        ("prefill_ms", "prompt_ms"),
        ("decode_ms", "predicted_ms"),
        ("prompt_n", "prompt_n"),
        ("predicted_n", "predicted_n"),
    ];
    pairs
        .iter()
        .all(|(r, k)| rec.get(*r).is_some_and(|v| !v.is_null()) && rec.get(*r) == t.get(*k))
}

struct Response {
    name: String,
    wall_ms: f64,
    client_id: String,
    echoed: Option<String>,
    id: Option<String>,
    timings: Option<Value>,
    usage: Option<Value>,
}

/// The six cells F1 F2 F3 F5 F6 F7 for one response.
fn cells(
    r: &Response,
    logged: &HashMap<String, Value>,
    filed: &HashMap<String, Value>,
    expect_backend: &str,
) -> [bool; 7] {
    let num = |v: &Value, k: &str| v.get(k).and_then(Value::as_f64);
    let t = r.timings.as_ref();
    let f1 = t.is_some();
    let f2 = match (t, r.usage.as_ref()) {
        (Some(t), Some(u)) => {
            num(t, "prompt_n").is_some()
                && num(t, "prompt_n") == num(u, "prompt_tokens")
                && num(t, "predicted_n").is_some()
                && num(t, "predicted_n") == num(u, "completion_tokens")
        },
        _ => false,
    };
    let f3 = t.is_some_and(|t| {
        match (
            num(t, "prompt_ms"),
            num(t, "predicted_ms"),
            num(t, "predicted_n"),
        ) {
            (Some(p), Some(d), Some(n)) => {
                p > 0.0 && (d > 0.0 || n == 0.0) && p + d <= r.wall_ms + 1.0
            },
            _ => false,
        }
    });
    let lrec = r.id.as_ref().and_then(|id| logged.get(id));
    let frec = r.id.as_ref().and_then(|id| filed.get(id));
    let f5 = same(lrec, t);
    let nonempty = |rec: &Value, k: &str| {
        rec.get(k)
            .and_then(Value::as_str)
            .is_some_and(|s| !s.is_empty())
    };
    let f6 = same(frec, t) && frec.is_some_and(|f| nonempty(f, "build") && nonempty(f, "host"));
    let echoes = |rec: Option<&Value>| {
        rec.and_then(|x| x.get("client_request_id"))
            .and_then(Value::as_str)
            == Some(r.client_id.as_str())
    };
    let f7 = r.echoed.as_deref() == Some(r.client_id.as_str()) && echoes(lrec) && echoes(frec);
    // BE: a log line names the device the run was started on, or says it cannot tell
    // (`unreported`, the documented SSE-chat gap). Naming the OTHER device is a false label.
    let label_ok = |rec: Option<&Value>| {
        rec.and_then(|x| x.get("backend"))
            .and_then(Value::as_str)
            .is_some_and(|b| b == expect_backend || b == "unreported")
    };
    let be = label_ok(lrec) && label_ok(frec);
    [f1, f2, f3, f5, f6, f7, be]
}

fn cases() -> Vec<(String, &'static str, Value)> {
    let mut out = Vec::new();
    for tokens in [1u32, 32] {
        for stream in [false, true] {
            let tag = format!("{}{tokens}", if stream { "s" } else { "ns" });
            out.push((
                format!("chat-{tag}"),
                "/v1/chat/completions",
                json!({
                    "model": "default",
                    "messages": [{"role": "user", "content": CHAT_PROMPT}],
                    "temperature": 0, "seed": 4354, "max_tokens": tokens, "stream": stream,
                    "chat_template_kwargs": {"enable_thinking": false},
                }),
            ));
            out.push((
                format!("completions-{tag}"),
                "/v1/completions",
                json!({
                    "model": "default", "prompt": COMPLETION_PROMPT,
                    "temperature": 0, "max_tokens": tokens, "stream": stream,
                }),
            ));
        }
    }
    out
}

fn wait_healthy(
    client: &reqwest::blocking::Client,
    port: u16,
    child: &mut Child,
    secs: u64,
) -> bool {
    let url = format!("http://127.0.0.1:{port}/health");
    for _ in 0..secs {
        if !matches!(child.try_wait(), Ok(None)) {
            return false;
        }
        if client
            .get(&url)
            .timeout(Duration::from_secs(2))
            .send()
            .is_ok()
        {
            return true;
        }
        std::thread::sleep(Duration::from_secs(1));
    }
    false
}

fn read_lines(path: &Path) -> Vec<String> {
    File::open(path).map_or_else(
        |_| Vec::new(),
        |f| BufReader::new(f).lines().map_while(Result::ok).collect(),
    )
}

/// SIGTERM the server's whole process group (it was started as its own group leader, so a
/// `--prefix` such as flock goes with it), then reap it.
fn stop(child: &mut Child) {
    let group = format!("-{}", child.id());
    let _ = Command::new("kill").args(["-TERM", "--", &group]).status();
    let _ = child.wait();
}

fn run(a: &Args) -> Result<bool, String> {
    std::fs::create_dir_all(&a.out).map_err(|e| format!("{}: {e}", a.out.display()))?;
    let tlog = a.out.join("timings.jsonl");
    let _ = std::fs::remove_file(&tlog);
    let slog = a.out.join("serve.log");
    let err = File::create(&slog).map_err(|e| format!("{}: {e}", slog.display()))?;
    let out = err.try_clone().map_err(|e| e.to_string())?;

    let mut argv: Vec<String> = a.prefix.clone();
    argv.extend([a.apr.clone(), "serve".into(), "run".into(), a.model.clone()]);
    argv.extend(["--port".into(), a.port.to_string(), "--timings-log".into()]);
    argv.push(tlog.display().to_string());
    argv.extend(a.serve_args.iter().cloned());
    let mut child = Command::new(&argv[0])
        .args(&argv[1..])
        .stdin(Stdio::null())
        .stdout(out)
        .stderr(err)
        .process_group(0)
        .spawn()
        .map_err(|e| format!("spawn {}: {e}", argv[0]))?;

    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()
        .map_err(|e| e.to_string())?;
    let mut responses = Vec::new();
    let sent = (|| {
        if !wait_healthy(&client, a.port, &mut child, a.health_seconds) {
            return Err(format!(
                "RED: server never became healthy; see {}",
                slog.display()
            ));
        }
        for (name, route, body) in cases() {
            let client_id = uuid::Uuid::new_v4().to_string();
            let t0 = Instant::now();
            let resp = client
                .post(format!("http://127.0.0.1:{}{route}", a.port))
                .header("X-Request-ID", &client_id)
                .json(&body)
                .send()
                .map_err(|e| format!("{name}: {e}"))?;
            let echoed = resp
                .headers()
                .get("X-Request-ID")
                .and_then(|v| v.to_str().ok())
                .map(String::from);
            let raw = resp.text().map_err(|e| format!("{name}: {e}"))?;
            let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let _ = std::fs::write(a.out.join(format!("{name}.resp")), &raw);
            let (id, timings, usage) = parse_body(&raw);
            responses.push(Response {
                name,
                wall_ms,
                client_id,
                echoed,
                id,
                timings,
                usage,
            });
        }
        std::thread::sleep(Duration::from_millis(500)); // the terminal-chunk emit precedes the client's EOF
        Ok(())
    })();
    stop(&mut child);
    sent?;

    let logged = records(read_lines(&slog).into_iter(), true);
    let filed = records(read_lines(&tlog).into_iter(), false);
    let mut red = 0;
    println!(
        "{:<20} F1    F2    F3    F5    F6    F7    BE    prompt_ms/predicted_ms/wall_ms",
        "case"
    );
    for r in &responses {
        let c = cells(r, &logged, &filed, &a.expect_backend);
        red += c.iter().filter(|ok| !**ok).count();
        let ms = r.timings.as_ref().map_or_else(
            || "-".to_string(),
            |t| {
                let f = |k: &str| t.get(k).and_then(Value::as_f64).unwrap_or(f64::NAN);
                format!(
                    "{:.1}/{:.1}/{:.1}",
                    f("prompt_ms"),
                    f("predicted_ms"),
                    r.wall_ms
                )
            },
        );
        let row: Vec<String> = c
            .iter()
            .map(|ok| format!("{:<5}", if *ok { "ok" } else { "RED" }))
            .collect();
        println!("{:<20} {} {ms}", r.name, row.join(" "));
    }
    println!(
        "{}: {red} red cell(s) over {} cases",
        if red == 0 { "GREEN" } else { "RED" },
        responses.len()
    );
    Ok(red == 0)
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("serve_timings_falsify: {e}");
            return ExitCode::from(2);
        },
    };
    match run(&args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            println!("{e}");
            ExitCode::FAILURE
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ID: &str = "chatcmpl-1";
    const CID: &str = "c-1";

    fn timings() -> Value {
        json!({"prompt_n": 12, "prompt_ms": 30.5, "predicted_n": 4, "predicted_ms": 20.0})
    }

    fn rec() -> Value {
        json!({"request_id": ID, "client_request_id": CID, "prefill_ms": 30.5, "decode_ms": 20.0,
               "prompt_n": 12, "predicted_n": 4, "build": "0.70.0 abc123def", "host": "lambda", "backend": "gpu"})
    }

    fn resp() -> Response {
        Response {
            name: "chat-ns1".into(),
            wall_ms: 60.0,
            client_id: CID.into(),
            echoed: Some(CID.into()),
            id: Some(ID.into()),
            timings: Some(timings()),
            usage: Some(json!({"prompt_tokens": 12, "completion_tokens": 4})),
        }
    }

    fn logs() -> (HashMap<String, Value>, HashMap<String, Value>) {
        let l = records(std::iter::once(format!("{LOG_PREFIX}{}", rec())), true);
        let f = records(std::iter::once(rec().to_string()), false);
        (l, f)
    }

    #[test]
    fn a_consistent_response_is_all_green() {
        let (l, f) = logs();
        assert_eq!(cells(&resp(), &l, &f, "gpu"), [true; 7]);
    }

    /// Each row plants one defect and names EXACTLY the cells it turns RED (0=F1 … 5=F7, 6=BE).
    /// Null timings reddens every cell that compares against them; a missing JSONL line also
    /// loses F7 and BE, as the client id and the label ride on it. Every other plant stays in
    /// its own column.
    #[test]
    fn each_planted_defect_turns_exactly_its_cells_red() {
        type Plant = fn(&mut Response, &mut HashMap<String, Value>, &mut HashMap<String, Value>);
        let table: [(&str, Plant, &[usize]); 11] = [
            (
                "F1 null timings",
                |r, _, _| r.timings = None,
                &[0, 1, 2, 3, 4],
            ),
            (
                "F2 usage disagrees",
                |r, _, _| r.usage = Some(json!({"prompt_tokens": 12, "completion_tokens": 5})),
                &[1],
            ),
            (
                "F3 zero prefill",
                |r, l, f| {
                    r.timings.as_mut().expect("fixture")["prompt_ms"] = json!(0.0);
                    l.get_mut(ID).expect("fixture")["prefill_ms"] = json!(0.0);
                    f.get_mut(ID).expect("fixture")["prefill_ms"] = json!(0.0);
                },
                &[2],
            ),
            ("F3 exceeds wall", |r, _, _| r.wall_ms = 40.0, &[2]),
            (
                "F5 stderr line differs",
                |_, l, _| l.get_mut(ID).expect("fixture")["decode_ms"] = json!(21.0),
                &[3],
            ),
            ("F6 jsonl missing", |_, _, f| f.clear(), &[4, 5, 6]),
            (
                "F6 no host",
                |_, _, f| f.get_mut(ID).expect("fixture")["host"] = json!(""),
                &[4],
            ),
            ("F7 not echoed", |r, _, _| r.echoed = None, &[5]),
            (
                "F7 log lacks client id",
                |_, l, _| {
                    l.get_mut(ID)
                        .expect("fixture")
                        .as_object_mut()
                        .expect("fixture")
                        .remove("client_request_id");
                },
                &[5],
            ),
            (
                "BE stderr names the other device",
                |_, l, _| l.get_mut(ID).expect("fixture")["backend"] = json!("cpu"),
                &[6],
            ),
            (
                "BE jsonl says unreported (documented gap)",
                |_, _, f| f.get_mut(ID).expect("fixture")["backend"] = json!("unreported"),
                &[],
            ),
        ];
        for (name, plant, want) in table {
            let (mut l, mut f) = logs();
            let mut r = resp();
            plant(&mut r, &mut l, &mut f);
            let c = cells(&r, &l, &f, "gpu");
            let reds: Vec<usize> = (0..7).filter(|i| !c[*i]).collect();
            assert_eq!(reds, want, "{name}: RED cells {reds:?}, want {want:?}");
        }
    }

    #[test]
    fn f3_allows_zero_decode_only_when_nothing_was_predicted() {
        let (l, f) = logs();
        let mut r = resp();
        r.timings =
            Some(json!({"prompt_n": 12, "prompt_ms": 30.5, "predicted_n": 0, "predicted_ms": 0.0}));
        assert!(cells(&r, &l, &f, "gpu")[2]);
        r.timings =
            Some(json!({"prompt_n": 12, "prompt_ms": 30.5, "predicted_n": 4, "predicted_ms": 0.0}));
        assert!(!cells(&r, &l, &f, "gpu")[2]);
    }

    #[test]
    fn sse_takes_timings_and_usage_from_the_terminal_chunk() {
        let raw = format!(
            "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
            json!({"id": ID, "choices": [], "timings": null}),
            json!({"id": ID, "choices": [], "timings": timings(), "usage": {"prompt_tokens": 12}})
        );
        let (id, t, u) = parse_body(&raw);
        assert_eq!(id.as_deref(), Some(ID));
        assert_eq!(t, Some(timings()));
        assert_eq!(u, Some(json!({"prompt_tokens": 12})));
    }

    #[test]
    fn a_json_body_with_null_timings_parses_as_absent() {
        let (_, t, _) = parse_body(&json!({"id": ID, "timings": null}).to_string());
        assert!(t.is_none());
    }

    #[test]
    fn records_skip_non_request_lines() {
        let lines = [
            "INFO starting".to_string(),
            format!("{LOG_PREFIX}not json"),
            format!("{LOG_PREFIX}{}", rec()),
        ];
        let r = records(lines.into_iter(), true);
        assert_eq!(r.len(), 1);
        assert!(r.contains_key(ID));
    }
}
