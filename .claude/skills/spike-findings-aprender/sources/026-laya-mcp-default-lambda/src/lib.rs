//! Spike 026: Laya (spike-025 Rust port) behind ONE stateless pmcp `decide` tool taking a real
//! `/v1/systemone`-shaped request: `{ state, questions: { id: { type, instructions, criteria?, labels? } } }`.
//! Every response carries the cold-start forensics of the process that answered it (spike-021 pattern).
#![allow(clippy::disallowed_methods)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use once_cell::sync::OnceCell;
use pmcp::types::capabilities::ServerCapabilities;
use pmcp::Server;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

static PROCESS_START: OnceCell<Instant> = OnceCell::new();
static ENGINE: OnceCell<Engine> = OnceCell::new();
static LOAD_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
static TIMELINE: Mutex<Vec<Step>> = Mutex::new(Vec::new());
static FIRST_CALL: AtomicBool = AtomicBool::new(true);
const FILES: [&str; 4] = ["rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json", "model.safetensors"];

pub fn mark_process_start() {
    let _ = PROCESS_START.set(Instant::now());
}
fn since_start_ms() -> f64 {
    PROCESS_START.get_or_init(Instant::now).elapsed().as_secs_f64() * 1e3
}

#[derive(Clone, Serialize)]
pub struct Step {
    step: String,
    t_ms: f64,
    rss_mb: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    detail: Option<String>,
}
fn proc_status_mb(field: &str) -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with(field)).and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok()))
        .map_or(0.0, |kb| kb / 1024.0)
}
fn record(step: &str, detail: Option<String>) {
    let s = Step { step: step.into(), t_ms: since_start_ms(), rss_mb: proc_status_mb("VmRSS"), detail };
    tracing::info!(step = %s.step, t_ms = s.t_ms, rss_mb = s.rss_mb, detail = ?s.detail, "timeline");
    if let Ok(mut t) = TIMELINE.lock() {
        t.push(s);
    }
}
fn host_info() -> Value {
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let part = cpuinfo.lines().find(|l| l.starts_with("CPU part")).and_then(|l| l.split(':').nth(1)).map(|s| s.trim().to_string());
    let guess = match part.as_deref() {
        Some("0xd0c") => "Neoverse-N1 (Graviton2)",
        Some("0xd40") => "Neoverse-V1 (Graviton3)",
        Some("0xd4f") => "Neoverse-V2 (Graviton4)",
        Some("0xd84") => "Neoverse-V3",
        _ => "unknown",
    };
    json!({"platform": std::env::var("PLATFORM").unwrap_or_else(|_| "local".into()), "cpu_part": part, "cpu_guess": guess,
           "available_parallelism": std::thread::available_parallelism().map_or(0, std::num::NonZero::get),
           "rayon_threads": rayon::current_num_threads(), "aws_lambda_memory_mb": std::env::var("AWS_LAMBDA_FUNCTION_MEMORY_SIZE").ok()})
}

pub struct Engine {
    model: laya::Laya,
    builder: laya::Builder,
    agent_cfg: Value,
}

/// `LAYA_DIR=/path` (local snapshot) or `LAYA_S3=s3://bucket/prefix` (downloaded to `LAYA_LOCAL_DIR`, default /tmp/laya).
async fn resolve_dir() -> Result<std::path::PathBuf, String> {
    if let Ok(uri) = std::env::var("LAYA_S3") {
        let local = std::path::PathBuf::from(std::env::var("LAYA_LOCAL_DIR").unwrap_or_else(|_| "/tmp/laya".into()));
        let rest = uri.strip_prefix("s3://").ok_or("LAYA_S3 must be s3://")?;
        let (bucket, prefix) = rest.split_once('/').ok_or("s3://bucket/prefix")?;
        let cfg = aws_config::load_from_env().await;
        let s3 = aws_sdk_s3::Client::new(&cfg);
        let t0 = Instant::now();
        let mut total = 0u64;
        for f in FILES {
            let dest = local.join(f);
            std::fs::create_dir_all(dest.parent().ok_or("parent")?).map_err(|e| e.to_string())?;
            total += s3_download(&s3, bucket, &format!("{prefix}/{f}"), &dest).await?;
        }
        let secs = t0.elapsed().as_secs_f64();
        record("s3 download", Some(format!("{:.0} MB in {secs:.2} s = {:.0} MB/s", total as f64 / 1e6, total as f64 / 1e6 / secs)));
        return Ok(local);
    }
    std::env::var("LAYA_DIR").map(Into::into).map_err(|_| "set LAYA_DIR or LAYA_S3".into())
}

/// Parallel ranged GETs (16 x 64 MB) with per-part retry, straight into a pre-sized file (spike 021).
async fn s3_download(s3: &aws_sdk_s3::Client, bucket: &str, key: &str, dest: &std::path::Path) -> Result<u64, String> {
    use std::os::unix::fs::FileExt;
    let head = s3.head_object().bucket(bucket).key(key).send().await.map_err(|e| format!("head {key}: {}", aws_sdk_s3::error::DisplayErrorContext(&e)))?;
    let len = head.content_length().unwrap_or(0) as u64;
    let file = std::sync::Arc::new(std::fs::File::create(dest).map_err(|e| format!("create {dest:?}: {e}"))?);
    file.set_len(len).map_err(|e| e.to_string())?;
    let part = 64 * 1024 * 1024u64;
    let sem = std::sync::Arc::new(tokio::sync::Semaphore::new(16));
    let mut tasks = Vec::new();
    for s in (0..len).step_by(part as usize) {
        let e = (s + part).min(len) - 1;
        let (s3, file, sem, bucket, key) = (s3.clone(), file.clone(), sem.clone(), bucket.to_string(), key.to_string());
        tasks.push(tokio::spawn(async move {
            let _p = sem.acquire_owned().await.map_err(|e| e.to_string())?;
            let mut attempt = 0;
            let bytes = loop {
                attempt += 1;
                let r = match s3.get_object().bucket(&bucket).key(&key).range(format!("bytes={s}-{e}")).send().await {
                    Ok(o) => o.body.collect().await.map(|b| b.into_bytes()).map_err(|e| format!("body: {e}")),
                    Err(e) => Err(format!("get: {}", aws_sdk_s3::error::DisplayErrorContext(&e))),
                };
                match r {
                    Ok(b) => break b,
                    Err(m) if attempt < 5 => tracing::warn!(part = s, attempt, %m, "retrying part"),
                    Err(m) => return Err(format!("{key} part {s}: {m}")),
                }
            };
            tokio::task::spawn_blocking(move || file.write_all_at(&bytes, s)).await.map_err(|e| e.to_string())?.map_err(|e| e.to_string())
        }));
    }
    for t in tasks {
        t.await.map_err(|e| e.to_string())??;
    }
    Ok(len)
}

pub async fn load_engine() -> Result<&'static Engine, String> {
    if let Some(e) = ENGINE.get() {
        return Ok(e);
    }
    let _g = LOAD_LOCK.lock().await; // concurrent first calls load once
    if let Some(e) = ENGINE.get() {
        return Ok(e);
    }
    record("load start", None);
    let dir = resolve_dir().await?;
    let engine = tokio::task::spawn_blocking(move || -> Result<Engine, String> {
        let rd = |f: &str| std::fs::read_to_string(dir.join(f)).map_err(|e| format!("{f}: {e}"));
        let enc_cfg: Value = serde_json::from_str(&rd("encoder/config.json")?).map_err(|e| e.to_string())?;
        let agent_cfg: Value = serde_json::from_str(&rd("rl_agent_config.json")?).map_err(|e| e.to_string())?;
        let bytes = std::fs::read(dir.join("model.safetensors")).map_err(|e| e.to_string())?;
        record("read weights", Some(format!("{:.0} MB F16", bytes.len() as f64 / 1e6)));
        let model = laya::Laya::from_tensors(laya::load_tensors(&bytes), &enc_cfg, agent_cfg["head_layers"].as_u64().unwrap_or(2) as usize);
        drop(bytes);
        record("build model (F16 -> f32)", None);
        let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer/tokenizer.json")).map_err(|e| e.to_string())?;
        let id = |t: &str| tok.token_to_id(t).ok_or(format!("token {t}"));
        let builder = laya::Builder {
            cls: id("[CLS]")?, sep: id("[SEP]")?, mask: id("[MASK]")?, mask_token: "[MASK]".into(),
            max_len: agent_cfg["max_len"].as_u64().unwrap_or(512) as usize,
            head_max_len: agent_cfg["head_max_len"].as_u64().unwrap_or(192) as usize, tok,
        };
        Ok(Engine { model, builder, agent_cfg })
    })
    .await
    .map_err(|e| e.to_string())??;
    let _ = ENGINE.set(engine);
    record("engine ready", None);
    ENGINE.get().ok_or_else(|| "engine".into())
}

fn render_criterion(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

/// Port of `laya.common.render_options` -> (answer keys in label order, option texts). Errors, not silence.
pub fn render_options(t: &str, criteria: Option<&Value>, labels: Option<&Value>) -> Result<(Vec<String>, Vec<String>), String> {
    match t {
        "choice" => {
            let pairs: Vec<(String, Option<String>)> = match criteria {
                Some(Value::Object(m)) => m.iter().map(|(k, v)| (k.clone(), if v.is_null() || v == "" { None } else { Some(render_criterion(v)) })).collect(),
                Some(Value::Array(a)) => a.iter().map(|k| (render_criterion(k), None)).collect(),
                _ => return Err("choice needs criteria (object or list)".into()),
            };
            if pairs.len() < 2 {
                return Err("choice needs at least 2 options".into());
            }
            let opts = pairs.iter().map(|(k, v)| v.as_ref().map_or_else(|| k.clone(), |v| format!("{k}: {v}"))).collect();
            Ok((pairs.into_iter().map(|p| p.0).collect(), opts))
        }
        "score" => {
            let a = criteria.and_then(Value::as_array).ok_or("score needs criteria (list of levels)")?;
            Ok(((0..a.len()).map(|i| i.to_string()).collect(), a.iter().enumerate().map(|(i, c)| format!("level {i}: {}", render_criterion(c))).collect()))
        }
        "noul" => {
            let c = criteria.and_then(Value::as_object);
            let get = |k: &str| c.and_then(|m| m.iter().find(|(kk, _)| kk.to_lowercase() == k).map(|(_, v)| v)).filter(|v| !v.is_null() && *v != "");
            let (fl, tl) = match labels {
                Some(l) => (l["false"].as_str().ok_or("labels.false")?.trim().to_string(), l["true"].as_str().ok_or("labels.true")?.trim().to_string()),
                None => ("false".into(), "true".into()),
            };
            Ok((vec!["false".into(), "true".into()], vec![
                format!("{fl}: {}", get("false").map_or_else(|| "no, the statement does not hold".into(), render_criterion)),
                format!("{tl}: {}", get("true").map_or_else(|| "yes, the statement holds".into(), render_criterion)),
            ]))
        }
        other => Err(format!("unknown question type {other:?} (choice | score | noul)")),
    }
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DecideArgs {
    /// The text (or JSON-serialised document) the questions are about.
    pub state: String,
    /// `{ id: { "type": "choice"|"score"|"noul", "instructions": str, "criteria": object|list, "labels"?: {false,true} } }`
    pub questions: Value,
    /// Repeat the whole request this many extra times and report p50 (steady-state latency). Default 0.
    #[serde(default)]
    pub warm_rounds: usize,
}

fn run_decide(e: &Engine, a: &DecideArgs) -> Result<Value, String> {
    let qs = a.questions.as_object().ok_or("questions must be an object")?;
    if qs.is_empty() {
        return Err("no questions".into());
    }
    let first = FIRST_CALL.swap(false, Ordering::SeqCst);
    let mut prepared = Vec::new();
    for (qid, q) in qs {
        let t = q["type"].as_str().ok_or(format!("{qid}: type"))?;
        let ins = q["instructions"].as_str().ok_or(format!("{qid}: instructions must be a string"))?;
        let (keys, opts) = render_options(t, q.get("criteria"), q.get("labels")).map_err(|m| format!("{qid}: {m}"))?;
        let qtype = match t { "choice" => 0, "score" => 1, _ => 2 };
        let (ids, markers) = e.builder.build(&a.state, t, ins, &opts);
        if markers.len() != opts.len() {
            return Err(format!("{qid}: options exceed head_max_len"));
        }
        prepared.push((qid.clone(), t.to_string(), keys, qtype, ids, markers));
    }
    let once = || -> (Vec<Value>, f64) {
        let s = Instant::now();
        let answers = prepared.iter().map(|(qid, t, keys, qtype, ids, markers)| {
            let qs = Instant::now();
            let z = e.model.forward(ids, markers, *qtype, |_, _| {});
            let p = laya::softmax_t(&z, laya::temperature(&e.agent_cfg, *qtype, markers.len()));
            let mut a = json!({"id": qid, "type": t, "tokens": ids.len(), "ms": qs.elapsed().as_secs_f64() * 1e3, "probabilities": p, "keys": keys});
            match t.as_str() {
                "choice" => a["choice"] = json!(keys[(0..p.len()).fold(0, |m, i| if p[i] > p[m] { i } else { m })]),
                "score" => a["score"] = json!(p.iter().enumerate().map(|(i, v)| i as f32 * v).sum::<f32>()),
                _ => a["noul"] = json!(p[1]),
            }
            a
        }).collect();
        (answers, s.elapsed().as_secs_f64() * 1e3)
    };
    let (answers, request_ms) = once();
    let mut warm: Vec<f64> = (0..a.warm_rounds.min(20)).map(|_| once().1).collect();
    warm.sort_by(f64::total_cmp);
    Ok(json!({"answers": answers, "request_ms": request_ms, "warm_p50_ms": warm.get(warm.len() / 2), "first_call_in_process": first,
              "process_uptime_ms": since_start_ms(), "peak_rss_mb": proc_status_mb("VmHWM"), "host": host_info(),
              "load_timeline": TIMELINE.lock().map(|t| t.clone()).unwrap_or_default()}))
}

pub fn build_server(name: &str) -> pmcp::Result<Server> {
    Server::builder()
        .name(name)
        .version(env!("CARGO_PKG_VERSION"))
        .capabilities(ServerCapabilities::tools_only())
        .tool_typed_with_description::<DecideArgs, _, _>(
            "decide",
            "Answer typed questions (choice / score / noul) about a text with calibrated probabilities (Laya, one forward per question).",
            move |args, _extra| async move {
                let engine = load_engine().await.map_err(pmcp::Error::internal)?;
                tokio::task::spawn_blocking(move || run_decide(engine, &args))
                    .await
                    .map_err(|e| pmcp::Error::internal(format!("join: {e}")))?
                    .map_err(pmcp::Error::validation)
            },
        )
        .build()
}

pub fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .with_ansi(false)
        .try_init();
}
