//! Spike 021-023: Kev-0.8B behind ONE stateless pmcp `decide` tool, instrumented for cold-start forensics.
//!
//! The same library runs on three AWS hosts (default Lambda, Lambda Managed Instances, Fargate), so the
//! measured difference between them is the host, not the code. Every response carries the load timeline
//! of the process that answered it (ms since process start + RSS at each step), the hardware it ran on,
//! and the parity of its probabilities against Python fp32.
#![allow(clippy::disallowed_methods)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use once_cell::sync::OnceCell;
use pmcp::types::capabilities::ServerCapabilities;
use pmcp::Server;
use realizar::gguf::forward_qwen35::Qwen35Model;
use realizar::gguf::MappedGGUFModel;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

static PROCESS_START: OnceCell<Instant> = OnceCell::new();
static ENGINE: OnceCell<Engine> = OnceCell::new();
static TIMELINE: Mutex<Vec<Step>> = Mutex::new(Vec::new());
static FIRST_CALL: AtomicBool = AtomicBool::new(true);

const ROWS_JSON: &str = include_str!("../assets/rows.json");
const HEAD_BYTES: &[u8] = include_bytes!("../assets/kev-0.8b-head.safetensors");

/// Call first thing in `main`: every timeline entry is measured from here.
pub fn mark_process_start() {
    let _ = PROCESS_START.set(Instant::now());
}

fn since_start_ms() -> f64 {
    PROCESS_START.get_or_init(Instant::now).elapsed().as_secs_f64() * 1e3
}

#[derive(Clone, Serialize)]
pub struct Step {
    pub step: String,
    pub t_ms: f64,
    pub rss_mb: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

fn record(step: &str, detail: Option<String>) {
    let s = Step { step: step.to_string(), t_ms: since_start_ms(), rss_mb: proc_status_mb("VmRSS"), detail };
    tracing::info!(step = %s.step, t_ms = s.t_ms, rss_mb = s.rss_mb, detail = ?s.detail, "timeline");
    if let Ok(mut t) = TIMELINE.lock() {
        t.push(s);
    }
}

/// `/proc/self/status` field in MB (Linux); 0 elsewhere.
fn proc_status_mb(field: &str) -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines().find(|l| l.starts_with(field)).and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok())
        })
        .map_or(0.0, |kb| kb / 1024.0)
}

#[derive(Serialize, Clone)]
pub struct HostInfo {
    pub platform: String,
    pub arch: &'static str,
    pub cpu_part: Option<String>,
    pub cpu_guess: String,
    pub nproc_online: usize,
    pub available_parallelism: usize,
    pub rayon_threads: usize,
    pub cgroup_cpu_max: Option<String>,
    pub mem_total_mb: f64,
    pub aws_lambda_memory_mb: Option<String>,
}

fn host_info() -> HostInfo {
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let cpu_part = cpuinfo.lines().find(|l| l.starts_with("CPU part")).and_then(|l| l.split(':').nth(1)).map(|s| s.trim().to_string());
    // Arm MIDR part numbers: N1 = Graviton2, V1 = Graviton3, V2 = Graviton4.
    let cpu_guess = match cpu_part.as_deref() {
        Some("0xd0c") => "Neoverse-N1 (Graviton2)".to_string(),
        Some("0xd40") => "Neoverse-V1 (Graviton3)".to_string(),
        Some("0xd4f") => "Neoverse-V2 (Graviton4)".to_string(),
        Some(p) => format!("arm part {p}"),
        None => cpuinfo.lines().find(|l| l.starts_with("model name")).and_then(|l| l.split(':').nth(1)).map_or("unknown".into(), |s| s.trim().to_string()),
    };
    let nproc_online = cpuinfo.lines().filter(|l| l.starts_with("processor")).count();
    let mem_total_mb = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| s.lines().next().and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok()))
        .map_or(0.0, |kb| kb / 1024.0);
    HostInfo {
        platform: std::env::var("PLATFORM").unwrap_or_else(|_| "local".into()),
        arch: std::env::consts::ARCH,
        cpu_part,
        cpu_guess,
        nproc_online,
        available_parallelism: std::thread::available_parallelism().map_or(0, std::num::NonZero::get),
        rayon_threads: rayon::current_num_threads(),
        cgroup_cpu_max: std::fs::read_to_string("/sys/fs/cgroup/cpu.max").ok().map(|s| s.trim().to_string()),
        mem_total_mb,
        aws_lambda_memory_mb: std::env::var("AWS_LAMBDA_FUNCTION_MEMORY_SIZE").ok(),
    }
}

#[derive(Deserialize)]
struct Rows {
    temperature: f32,
    rows: Vec<Row>,
}
#[derive(Deserialize)]
struct Row {
    ids: Vec<u32>,
    decide: usize,
    opts: Vec<usize>,
    probs: Vec<f32>,
}

struct Head {
    qw: Vec<f32>,
    qb: Vec<f32>,
    kw: Vec<f32>,
    kb: Vec<f32>,
    dp: usize,
    d: usize,
}
impl Head {
    fn load(bytes: &[u8]) -> Result<Head, String> {
        let st = safetensors::SafeTensors::deserialize(bytes).map_err(|e| format!("head: {e}"))?;
        let get = |n: &str| -> Result<(Vec<f32>, Vec<usize>), String> {
            let t = st.tensor(n).map_err(|e| format!("head {n}: {e}"))?;
            Ok((t.data().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(), t.shape().to_vec()))
        };
        let (qw, s) = get("q.weight")?;
        let (qb, _) = get("q.bias")?;
        let (kw, _) = get("k.weight")?;
        let (kb, _) = get("k.bias")?;
        Ok(Head { qw, qb, kw, kb, dp: s[0], d: s[1] })
    }
    fn proj(&self, w: &[f32], b: &[f32], x: &[f32]) -> Vec<f32> {
        (0..self.dp).map(|o| b[o] + w[o * self.d..(o + 1) * self.d].iter().zip(x).map(|(a, c)| a * c).sum::<f32>()).collect()
    }
    fn probs(&self, hd: &[f32], ho: &[&[f32]], t: f32) -> Vec<f32> {
        let q = self.proj(&self.qw, &self.qb, hd);
        let s = 1.0 / (self.dp as f32).sqrt();
        let z: Vec<f32> =
            ho.iter().map(|h| self.proj(&self.kw, &self.kb, h).iter().zip(&q).map(|(a, b)| a * b).sum::<f32>() * s / t).collect();
        let m = z.iter().fold(f32::MIN, |a, &b| a.max(b));
        let e: Vec<f32> = z.iter().map(|v| (v - m).exp()).collect();
        let sum: f32 = e.iter().sum();
        e.iter().map(|v| v / sum).collect()
    }
}

pub struct Engine {
    qwen: Qwen35Model<'static>,
    head: Head,
    d: usize,
    rows: Rows,
}

/// Where the GGUF comes from: `KEV_GGUF_S3=s3://bucket/key` (download to `KEV_LOCAL_DIR`, default /tmp)
/// or `KEV_GGUF=/path` (baked into the image). Records every step on the timeline.
async fn resolve_gguf_path() -> Result<String, String> {
    if let Ok(uri) = std::env::var("KEV_GGUF_S3") {
        let dir = std::env::var("KEV_LOCAL_DIR").unwrap_or_else(|_| "/tmp".into());
        let dest = format!("{dir}/kev.gguf");
        s3_download(&uri, &dest).await?;
        return Ok(dest);
    }
    std::env::var("KEV_GGUF").map_err(|_| "set KEV_GGUF or KEV_GGUF_S3".to_string())
}

/// Parallel ranged GETs straight into a pre-sized file (`S3_PART_MB` x `S3_CONCURRENCY` in flight).
async fn s3_download(uri: &str, dest: &str) -> Result<(), String> {
    use std::os::unix::fs::FileExt;
    let rest = uri.strip_prefix("s3://").ok_or("KEV_GGUF_S3 must be s3://")?;
    let (bucket, key) = rest.split_once('/').ok_or("s3://bucket/key")?;
    let part_mb: u64 = std::env::var("S3_PART_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
    let conc: usize = std::env::var("S3_CONCURRENCY").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
    let cfg = aws_config::load_from_env().await;
    let s3 = aws_sdk_s3::Client::new(&cfg);
    let t0 = Instant::now();
    let head = s3.head_object().bucket(bucket).key(key).send().await.map_err(|e| format!("head_object: {}", aws_sdk_s3::error::DisplayErrorContext(&e)))?;
    let len = head.content_length().unwrap_or(0) as u64;
    let file = std::sync::Arc::new(std::fs::File::create(dest).map_err(|e| format!("create {dest}: {e}"))?);
    file.set_len(len).map_err(|e| e.to_string())?;
    let part = part_mb * 1024 * 1024;
    let ranges: Vec<(u64, u64)> = (0..len).step_by(part as usize).map(|s| (s, (s + part).min(len) - 1)).collect();
    let sem = std::sync::Arc::new(tokio::sync::Semaphore::new(conc));
    let mut tasks = Vec::new();
    for (s, e) in ranges {
        let (s3, file, sem) = (s3.clone(), file.clone(), sem.clone());
        let (bucket, key) = (bucket.to_string(), key.to_string());
        tasks.push(tokio::spawn(async move {
            let _permit = sem.acquire_owned().await.map_err(|e| e.to_string())?;
            // A ranged GET can fail mid-body ("streaming error", seen at 32 x 32 MB on Lambda): retry the part.
            let mut attempt = 0;
            let bytes = loop {
                attempt += 1;
                let res = match s3.get_object().bucket(&bucket).key(&key).range(format!("bytes={s}-{e}")).send().await {
                    Ok(obj) => obj.body.collect().await.map(|b| b.into_bytes()).map_err(|e| format!("body: {e}")),
                    Err(e) => Err(format!("get_object: {}", aws_sdk_s3::error::DisplayErrorContext(&e))),
                };
                match res {
                    Ok(b) => break b,
                    Err(msg) if attempt < 5 => tracing::warn!(part_start = s, attempt, %msg, "retrying part"),
                    Err(msg) => return Err(format!("part {s}: {msg} after {attempt} attempts")),
                }
            };
            tokio::task::spawn_blocking(move || file.write_all_at(&bytes, s)).await.map_err(|e| e.to_string())?.map_err(|e| e.to_string())
        }));
    }
    for t in tasks {
        t.await.map_err(|e| e.to_string())??;
    }
    let secs = t0.elapsed().as_secs_f64();
    record("s3 download", Some(format!("{:.0} MB in {secs:.2} s = {:.0} MB/s ({part_mb} MB parts x {conc})", len as f64 / 1e6, len as f64 / 1e6 / secs)));
    Ok(())
}

/// Load once per process. Blocking work runs on a blocking thread.
pub async fn load_engine() -> Result<&'static Engine, String> {
    if let Some(e) = ENGINE.get() {
        return Ok(e);
    }
    record("load start", None);
    let path = resolve_gguf_path().await?;
    let engine = tokio::task::spawn_blocking(move || -> Result<Engine, String> {
        let mapped = MappedGGUFModel::from_path(&path).map_err(|e| format!("mmap: {e}"))?;
        record("mmap GGUF", Some(path.clone()));
        let base = Qwen35Model::create_base_model(&mapped.model, mapped.data()).map_err(|e| format!("base: {e}"))?;
        record("base (embeddings, norms, lm_head)", None);
        let base: &'static _ = Box::leak(Box::new(base));
        let qwen = Qwen35Model::from_model_and_layers(base, &mapped.model, mapped.data()).map_err(|e| format!("layers: {e}"))?;
        record("owned layers", None);
        drop(mapped);
        record("drop mmap", None);
        let head = Head::load(HEAD_BYTES)?;
        let rows: Rows = serde_json::from_str(ROWS_JSON).map_err(|e| e.to_string())?;
        Ok(Engine { qwen, head, d: base.config().hidden_dim, rows })
    })
    .await
    .map_err(|e| e.to_string())??;
    let _ = ENGINE.set(engine);
    record("engine ready", None);
    ENGINE.get().ok_or_else(|| "engine".to_string())
}

impl Engine {
    fn decide(&self, row: &Row) -> Vec<f32> {
        let mut st = self.qwen.new_state(row.ids.len() + 1);
        let h = self.qwen.prefill_hidden(&row.ids, &mut st).expect("prefill");
        let d = self.d;
        let ho: Vec<&[f32]> = row.opts.iter().map(|&o| &h[o * d..(o + 1) * d]).collect();
        self.head.probs(&h[row.decide * d..(row.decide + 1) * d], &ho, self.rows.temperature)
    }
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DecideArgs {
    /// Index of a fixture decision row (0..12). The fixture carries token ids; the tokenizer is not in this spike.
    pub row: usize,
    /// Repeat the decision this many extra times and report p50/min (steady-state latency). Default 0.
    #[serde(default)]
    pub warm_rounds: usize,
}

#[derive(Serialize)]
struct DecideOut {
    probs: Vec<f32>,
    python_fp32_probs: Vec<f32>,
    max_abs_dp: f32,
    tokens: usize,
    first_call_in_process: bool,
    decision_ms: f64,
    warm_p50_ms: Option<f64>,
    warm_min_ms: Option<f64>,
    process_uptime_ms: f64,
    peak_rss_mb: f64,
    host: HostInfo,
    load_timeline: Vec<Step>,
}

fn run_decide(e: &Engine, args: &DecideArgs) -> Result<DecideOut, String> {
    let row = e.rows.rows.get(args.row).ok_or_else(|| format!("row must be < {}", e.rows.rows.len()))?;
    let first = FIRST_CALL.swap(false, Ordering::SeqCst);
    let s = Instant::now();
    let p = e.decide(row);
    let decision_ms = s.elapsed().as_secs_f64() * 1e3;
    let mut warm = Vec::new();
    for _ in 0..args.warm_rounds.min(20) {
        let s = Instant::now();
        let _ = e.decide(row);
        warm.push(s.elapsed().as_secs_f64() * 1e3);
    }
    warm.sort_by(f64::total_cmp);
    let max_abs_dp = p.iter().zip(&row.probs).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    Ok(DecideOut {
        max_abs_dp,
        python_fp32_probs: row.probs.clone(),
        probs: p,
        tokens: row.ids.len(),
        first_call_in_process: first,
        decision_ms,
        warm_p50_ms: warm.get(warm.len() / 2).copied(),
        warm_min_ms: warm.first().copied(),
        process_uptime_ms: since_start_ms(),
        peak_rss_mb: proc_status_mb("VmHWM"),
        host: host_info(),
        load_timeline: TIMELINE.lock().map(|t| t.clone()).unwrap_or_default(),
    })
}

/// The thin server: one stateless `decide` tool over one immutable model.
pub fn build_server(name: &str) -> pmcp::Result<Server> {
    Server::builder()
        .name(name)
        .version(env!("CARGO_PKG_VERSION"))
        .capabilities(ServerCapabilities::tools_only())
        .tool_typed_with_description::<DecideArgs, _, _>(
            "decide",
            "Run Kev-0.8B on a fixture decision row and return option probabilities with cold-start forensics.",
            move |args, _extra| async move {
                let engine = load_engine().await.map_err(pmcp::Error::internal)?;
                let out = tokio::task::spawn_blocking(move || run_decide(engine, &args))
                    .await
                    .map_err(|e| pmcp::Error::internal(format!("join: {e}")))?
                    .map_err(pmcp::Error::validation)?;
                serde_json::to_value(&out).map_err(|e| pmcp::Error::internal(e.to_string()))
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
