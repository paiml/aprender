//! MEAS-001 R1 (#4522): measured resource usage for one run of a command.
//!
//! Every receipt we had was timing-only. This is the `resources{}` block that
//! `apr bench`, `apr qa` and `apr profile` put in their JSON, and that serve-bench
//! reads instead of its estimates. Each field is **measured, never estimated**:
//! it is either a number with the source it was read from (`sources`), or `null`
//! with the reason it could not be read (`null_reasons`). A field is never a
//! default or a constant.
//!
//! | field | source |
//! |---|---|
//! | `peak_rss_bytes` | `/proc/self/status` `VmHWM` (the kernel's high-water mark, = getrusage `ru_maxrss`) |
//! | `cpu_user_s`, `cpu_sys_s` | `/proc/self/stat` utime/stime delta, all threads, USER_HZ = 100 |
//! | `minor_faults`, `major_faults` | `/proc/self/stat` minflt/majflt delta, all threads |
//! | `voluntary_ctx_switches`, `involuntary_ctx_switches` | Σ `/proc/self/task/*/status` delta (live threads) |
//! | `threads_peak` | `/proc/self/status` `Threads`, sampled, the sampler's own thread excluded |
//! | `io_read_bytes`, `io_write_bytes` | `/proc/self/io` `read_bytes`/`write_bytes` delta (storage layer) |
//! | `io_rchar_bytes`, `io_wchar_bytes` | `/proc/self/io` `rchar`/`wchar` delta (syscall layer, incl. page cache) |
//! | `vram_peak_bytes` | `nvidia-smi --query-compute-apps`, this pid, sampled |
//! | `energy_j` | RAPL `/sys/class/powercap/intel-rapl:N/energy_uj` delta, CPU packages, **system-wide** |
//! | `gpu_energy_j` | `nvidia-smi --query-gpu=total_energy_consumption` delta, all GPUs, **system-wide** |
//!
//! Pure std, no `unsafe`: the parsers take `&str` and are unit-tested on
//! fixtures; the readers are thin. Off Linux every `/proc` field is `null` with
//! the reason the read failed.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// `/proc/<pid>/stat` times are in USER_HZ, which the Linux ABI fixes at 100.
const USER_HZ: f64 = 100.0;
/// How often the sampler reads the thread count.
const THREAD_SAMPLE: Duration = Duration::from_millis(100);
/// How often the sampler asks the driver for this process's VRAM.
const VRAM_SAMPLE: Duration = Duration::from_millis(1000);

/// The measured `resources{}` block. `None` fields are explained in `null_reasons`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// Wall-clock seconds between `start` and `finish`.
    pub wall_s: f64,
    /// Peak resident set, bytes (process lifetime high-water mark).
    pub peak_rss_bytes: Option<u64>,
    /// User CPU seconds, all threads, over the window.
    pub cpu_user_s: Option<f64>,
    /// System CPU seconds, all threads, over the window.
    pub cpu_sys_s: Option<f64>,
    /// Minor page faults over the window.
    pub minor_faults: Option<u64>,
    /// Major page faults over the window.
    pub major_faults: Option<u64>,
    /// Voluntary context switches over the window.
    pub voluntary_ctx_switches: Option<u64>,
    /// Involuntary context switches over the window.
    pub involuntary_ctx_switches: Option<u64>,
    /// Most threads seen at once (the sampler's thread excluded).
    pub threads_peak: Option<u64>,
    /// Bytes read from storage over the window.
    pub io_read_bytes: Option<u64>,
    /// Bytes written to storage over the window.
    pub io_write_bytes: Option<u64>,
    /// Bytes read through read-like syscalls over the window.
    pub io_rchar_bytes: Option<u64>,
    /// Bytes written through write-like syscalls over the window.
    pub io_wchar_bytes: Option<u64>,
    /// Highest VRAM this process held in any sample, bytes, summed over GPUs.
    pub vram_peak_bytes: Option<u64>,
    /// CPU package energy over the window, joules, system-wide.
    pub energy_j: Option<f64>,
    /// GPU energy over the window, joules, system-wide.
    pub gpu_energy_j: Option<f64>,
    /// Where each non-null field was read from.
    pub sources: BTreeMap<String, String>,
    /// Why each null field is null.
    pub null_reasons: BTreeMap<String, String>,
}

/// Counters read once at `start` and once at `finish`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StatCounters {
    /// minflt
    pub minor_faults: u64,
    /// majflt
    pub major_faults: u64,
    /// utime, USER_HZ ticks
    pub utime_ticks: u64,
    /// stime, USER_HZ ticks
    pub stime_ticks: u64,
}

/// `/proc/self/io` counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IoCounters {
    /// rchar
    pub rchar: u64,
    /// wchar
    pub wchar: u64,
    /// read_bytes
    pub read_bytes: u64,
    /// write_bytes
    pub write_bytes: u64,
}

/// Parse `/proc/<pid>/stat`. The comm field (2) may hold spaces and parens, so
/// fields are counted from the LAST `)`.
#[must_use]
pub fn parse_proc_stat(text: &str) -> Option<StatCounters> {
    let rest = &text[text.rfind(')')? + 1..];
    // After ")": field 3 (state) is index 0, so field N is index N - 3.
    let f: Vec<&str> = rest.split_whitespace().collect();
    let at = |n: usize| f.get(n - 3)?.parse::<u64>().ok();
    Some(StatCounters {
        minor_faults: at(10)?,
        major_faults: at(12)?,
        utime_ticks: at(14)?,
        stime_ticks: at(15)?,
    })
}

/// One `Key:   value [kB]` line of a `/proc/<pid>/status` file, as a number
/// (`kB` values are converted to bytes).
#[must_use]
pub fn status_field(text: &str, key: &str) -> Option<u64> {
    text.lines().find_map(|line| {
        let v = line.strip_prefix(key)?.strip_prefix(':')?.trim();
        let mut parts = v.split_whitespace();
        let n = parts.next()?.parse::<u64>().ok()?;
        match parts.next() {
            Some("kB") => n.checked_mul(1024),
            None => Some(n),
            Some(_) => None,
        }
    })
}

/// Parse `/proc/<pid>/io`.
#[must_use]
pub fn parse_proc_io(text: &str) -> Option<IoCounters> {
    let get = |k: &str| {
        text.lines().find_map(|l| {
            l.strip_prefix(k)?
                .strip_prefix(':')?
                .trim()
                .parse::<u64>()
                .ok()
        })
    };
    Some(IoCounters {
        rchar: get("rchar")?,
        wchar: get("wchar")?,
        read_bytes: get("read_bytes")?,
        write_bytes: get("write_bytes")?,
    })
}

/// This pid's VRAM in `nvidia-smi --query-compute-apps=pid,used_memory
/// --format=csv,noheader,nounits` output, bytes, summed over GPUs. `None` when
/// the pid holds no compute context.
#[must_use]
pub fn parse_compute_apps(text: &str, pid: u32) -> Option<u64> {
    let mut total: Option<u64> = None;
    for line in text.lines() {
        let mut cols = line.split(',').map(str::trim);
        let (Some(p), Some(mib)) = (cols.next(), cols.next()) else {
            continue;
        };
        if p.parse::<u32>().ok() != Some(pid) {
            continue;
        }
        if let Ok(mib) = mib.parse::<u64>() {
            total = Some(total.unwrap_or(0) + mib * 1024 * 1024);
        }
    }
    total
}

/// Sum of `total_energy_consumption` (mJ) over GPUs, in joules. `None` when
/// any GPU does not report it.
#[must_use]
pub fn parse_gpu_energy_mj(text: &str) -> Option<f64> {
    let mut sum = 0.0;
    let mut any = false;
    for line in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
        sum += line.parse::<f64>().ok()?;
        any = true;
    }
    any.then_some(sum / 1000.0)
}

/// RAPL counter delta in µJ, allowing for one wrap at `max_range_uj`.
#[must_use]
pub fn rapl_delta_uj(start: u64, end: u64, max_range_uj: u64) -> u64 {
    if end >= start {
        end - start
    } else {
        max_range_uj.saturating_sub(start) + end
    }
}

fn read(path: &str) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))
}

fn read_stat() -> Result<StatCounters, String> {
    parse_proc_stat(&read("/proc/self/stat")?).ok_or_else(|| "/proc/self/stat: unparsable".into())
}

fn read_io() -> Result<IoCounters, String> {
    parse_proc_io(&read("/proc/self/io")?).ok_or_else(|| "/proc/self/io: unparsable".into())
}

/// Σ (voluntary, involuntary) context switches over the live threads.
fn read_ctx_switches() -> Result<(u64, u64), String> {
    let dir = std::fs::read_dir("/proc/self/task").map_err(|e| format!("/proc/self/task: {e}"))?;
    let (mut v, mut nv) = (0u64, 0u64);
    for entry in dir.flatten() {
        // A thread can exit between readdir and read; that thread is skipped.
        if let Ok(s) = std::fs::read_to_string(entry.path().join("status")) {
            v += status_field(&s, "voluntary_ctxt_switches").unwrap_or(0);
            nv += status_field(&s, "nonvoluntary_ctxt_switches").unwrap_or(0);
        }
    }
    Ok((v, nv))
}

fn read_threads() -> Option<u64> {
    status_field(&read("/proc/self/status").ok()?, "Threads")
}

/// Package-level RAPL domains (`intel-rapl:N`, not the `intel-rapl:N:M`
/// sub-domains, which the package already counts): (energy_uj path, max range).
fn rapl_domains() -> Result<Vec<(String, u64)>, String> {
    let base = "/sys/class/powercap";
    let dir = std::fs::read_dir(base).map_err(|e| format!("{base}: {e}"))?;
    let mut out = Vec::new();
    for entry in dir.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        let is_pkg = name
            .strip_prefix("intel-rapl:")
            .is_some_and(|rest| !rest.contains(':'));
        if !is_pkg {
            continue;
        }
        let p = entry.path();
        let max = std::fs::read_to_string(p.join("max_energy_range_uj"))
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(u64::MAX);
        out.push((p.join("energy_uj").to_string_lossy().to_string(), max));
    }
    if out.is_empty() {
        return Err(format!("{base}: no intel-rapl package domain"));
    }
    out.sort();
    Ok(out)
}

fn read_rapl(domains: &[(String, u64)]) -> Result<Vec<u64>, String> {
    domains
        .iter()
        .map(|(p, _)| {
            read(p)?
                .trim()
                .parse::<u64>()
                .map_err(|e| format!("{p}: {e}"))
        })
        .collect()
}

fn nvidia_smi(args: &[&str]) -> Result<String, String> {
    let out = std::process::Command::new("nvidia-smi")
        .args(args)
        .output()
        .map_err(|e| format!("nvidia-smi: {e}"))?;
    if !out.status.success() {
        return Err(format!("nvidia-smi {}: {}", args.join(" "), out.status));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn gpu_energy() -> Result<f64, String> {
    let t = nvidia_smi(&[
        "--query-gpu=total_energy_consumption",
        "--format=csv,noheader,nounits",
    ])?;
    parse_gpu_energy_mj(&t).ok_or_else(|| {
        format!(
            "nvidia-smi total_energy_consumption unsupported: {}",
            t.trim()
        )
    })
}

/// What the sampler thread saw.
#[derive(Debug, Default)]
struct Sampled {
    threads_peak: Option<u64>,
    vram_peak: Option<u64>,
    vram_samples: u64,
    vram_err: Option<String>,
}

type RaplStart = Result<(Vec<(String, u64)>, Vec<u64>), String>;

/// Measures one window: [`ResourceSampler::start`] … [`ResourceSampler::finish`].
#[derive(Debug)]
pub struct ResourceSampler {
    t0: Instant,
    stat0: Result<StatCounters, String>,
    io0: Result<IoCounters, String>,
    ctx0: Result<(u64, u64), String>,
    rapl: RaplStart,
    gpu_e0: Result<f64, String>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<Sampled>>,
}

impl ResourceSampler {
    /// Read the start counters and start the sampling thread.
    #[must_use]
    pub fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);
        let handle = std::thread::Builder::new()
            .name("apr-resources".into())
            .spawn(move || sample_loop(&flag))
            .ok();
        Self {
            t0: Instant::now(),
            stat0: read_stat(),
            io0: read_io(),
            ctx0: read_ctx_switches(),
            rapl: rapl_domains().and_then(|d| {
                let v = read_rapl(&d)?;
                Ok((d, v))
            }),
            gpu_e0: gpu_energy(),
            stop,
            handle,
        }
    }

    /// Stop sampling and read the end counters.
    #[must_use]
    pub fn finish(mut self) -> ResourceUsage {
        let wall_s = self.t0.elapsed().as_secs_f64();
        self.stop.store(true, Ordering::Relaxed);
        let sampled = match self.handle.take() {
            Some(h) => {
                h.thread().unpark();
                h.join().unwrap_or_default()
            },
            None => Sampled::default(),
        };
        let mut r = ResourceUsage {
            wall_s,
            ..ResourceUsage::default()
        };
        let mut rec = Recorder(&mut r);
        self.fill_proc(&mut rec, &sampled);
        self.fill_gpu(&mut rec, &sampled);
        self.fill_energy(&mut rec);
        r
    }

    fn fill_proc(&self, rec: &mut Recorder<'_>, sampled: &Sampled) {
        match read("/proc/self/status").map(|s| status_field(&s, "VmHWM")) {
            Ok(Some(b)) => {
                rec.0.peak_rss_bytes = Some(b);
                rec.src(
                    "peak_rss_bytes",
                    "/proc/self/status VmHWM (process high-water mark)",
                );
            },
            Ok(None) => rec.null("peak_rss_bytes", "/proc/self/status has no VmHWM"),
            Err(e) => rec.null("peak_rss_bytes", &e),
        }
        const STAT: [&str; 4] = ["cpu_user_s", "cpu_sys_s", "minor_faults", "major_faults"];
        match (&self.stat0, &read_stat()) {
            (Ok(a), Ok(b)) => {
                rec.0.cpu_user_s =
                    Some(b.utime_ticks.saturating_sub(a.utime_ticks) as f64 / USER_HZ);
                rec.0.cpu_sys_s =
                    Some(b.stime_ticks.saturating_sub(a.stime_ticks) as f64 / USER_HZ);
                rec.0.minor_faults = Some(b.minor_faults.saturating_sub(a.minor_faults));
                rec.0.major_faults = Some(b.major_faults.saturating_sub(a.major_faults));
                for k in STAT {
                    rec.src(k, "/proc/self/stat delta, all threads");
                }
            },
            (Err(e), _) | (_, Err(e)) => STAT.iter().for_each(|k| rec.null(k, e)),
        }
        match (&self.ctx0, &read_ctx_switches()) {
            (Ok(a), Ok(b)) => {
                rec.0.voluntary_ctx_switches = Some(b.0.saturating_sub(a.0));
                rec.0.involuntary_ctx_switches = Some(b.1.saturating_sub(a.1));
                let src = "sum of /proc/self/task/*/status delta (threads alive at both reads)";
                rec.src("voluntary_ctx_switches", src);
                rec.src("involuntary_ctx_switches", src);
            },
            (Err(e), _) | (_, Err(e)) => {
                rec.null("voluntary_ctx_switches", e);
                rec.null("involuntary_ctx_switches", e);
            },
        }
        match sampled.threads_peak {
            Some(t) => {
                rec.0.threads_peak = Some(t);
                rec.src(
                    "threads_peak",
                    "/proc/self/status Threads, sampled every 100 ms, sampler thread excluded",
                );
            },
            None => rec.null("threads_peak", "/proc/self/status Threads unreadable"),
        }
        const IO: [&str; 4] = [
            "io_read_bytes",
            "io_write_bytes",
            "io_rchar_bytes",
            "io_wchar_bytes",
        ];
        match (&self.io0, &read_io()) {
            (Ok(a), Ok(b)) => {
                rec.0.io_read_bytes = Some(b.read_bytes.saturating_sub(a.read_bytes));
                rec.0.io_write_bytes = Some(b.write_bytes.saturating_sub(a.write_bytes));
                rec.0.io_rchar_bytes = Some(b.rchar.saturating_sub(a.rchar));
                rec.0.io_wchar_bytes = Some(b.wchar.saturating_sub(a.wchar));
                rec.src(
                    "io_read_bytes",
                    "/proc/self/io read_bytes delta (storage layer)",
                );
                rec.src(
                    "io_write_bytes",
                    "/proc/self/io write_bytes delta (storage layer)",
                );
                rec.src(
                    "io_rchar_bytes",
                    "/proc/self/io rchar delta (syscall layer)",
                );
                rec.src(
                    "io_wchar_bytes",
                    "/proc/self/io wchar delta (syscall layer)",
                );
            },
            (Err(e), _) | (_, Err(e)) => IO.iter().for_each(|k| rec.null(k, e)),
        }
    }

    fn fill_gpu(&self, rec: &mut Recorder<'_>, sampled: &Sampled) {
        let pid = std::process::id();
        let n = sampled.vram_samples;
        match (&sampled.vram_err, sampled.vram_peak) {
            (Some(e), _) => rec.null("vram_peak_bytes", e),
            (None, Some(b)) => {
                rec.0.vram_peak_bytes = Some(b);
                rec.src(
                    "vram_peak_bytes",
                    &format!("nvidia-smi --query-compute-apps used_memory, pid {pid}, max of {n} samples at 1 s"),
                );
            },
            (None, None) if n > 0 => {
                // The driver answered and this pid held no compute context.
                rec.0.vram_peak_bytes = Some(0);
                rec.src(
                    "vram_peak_bytes",
                    &format!("nvidia-smi --query-compute-apps: pid {pid} held no compute context in {n} samples"),
                );
            },
            (None, None) => rec.null("vram_peak_bytes", "no VRAM sample was taken"),
        }
        match (&self.gpu_e0, &gpu_energy()) {
            (Ok(a), Ok(b)) => {
                rec.0.gpu_energy_j = Some((b - a).max(0.0));
                rec.src(
                    "gpu_energy_j",
                    "nvidia-smi total_energy_consumption delta, all GPUs, system-wide (not per process)",
                );
            },
            (Err(e), _) | (_, Err(e)) => rec.null("gpu_energy_j", e),
        }
    }

    fn fill_energy(&self, rec: &mut Recorder<'_>) {
        let end = self
            .rapl
            .as_ref()
            .map_err(Clone::clone)
            .and_then(|(d, v0)| Ok((d, v0, read_rapl(d)?)));
        match end {
            Ok((domains, v0, v1)) => {
                let uj: u64 = domains
                    .iter()
                    .zip(v0.iter().zip(&v1))
                    .map(|((_, max), (a, b))| rapl_delta_uj(*a, *b, *max))
                    .sum();
                rec.0.energy_j = Some(uj as f64 / 1e6);
                rec.src(
                    "energy_j",
                    &format!(
                        "RAPL energy_uj delta over {} CPU package domain(s), system-wide (not per process)",
                        domains.len()
                    ),
                );
            },
            Err(e) => rec.null("energy_j", &e),
        }
    }
}

impl Drop for ResourceSampler {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            h.thread().unpark();
            let _ = h.join();
        }
    }
}

struct Recorder<'a>(&'a mut ResourceUsage);

impl Recorder<'_> {
    fn src(&mut self, k: &str, v: &str) {
        self.0.sources.insert(k.to_string(), v.to_string());
    }
    fn null(&mut self, k: &str, why: &str) {
        self.0.null_reasons.insert(k.to_string(), why.to_string());
    }
}

fn sample_vram(s: &mut Sampled, pid: u32) {
    match nvidia_smi(&[
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]) {
        Ok(t) => {
            s.vram_samples += 1;
            if let Some(b) = parse_compute_apps(&t, pid) {
                s.vram_peak = Some(s.vram_peak.map_or(b, |p| p.max(b)));
            }
        },
        Err(e) => s.vram_err = Some(e),
    }
}

fn sample_loop(stop: &AtomicBool) -> Sampled {
    let pid = std::process::id();
    let mut s = Sampled::default();
    let mut next_vram = Instant::now();
    loop {
        if let Some(t) = read_threads() {
            // This sampler thread is not the workload's.
            let t = t.saturating_sub(1);
            s.threads_peak = Some(s.threads_peak.map_or(t, |p| p.max(t)));
        }
        if s.vram_err.is_none() && Instant::now() >= next_vram {
            sample_vram(&mut s, pid);
            next_vram = Instant::now() + VRAM_SAMPLE;
        }
        if stop.load(Ordering::Relaxed) {
            return s;
        }
        std::thread::park_timeout(THREAD_SAMPLE);
    }
}

/// The process's peak RSS so far (`/proc/self/status` `VmHWM`), or `None`
/// where that file is unreadable (non-Linux). A reading, never an estimate.
pub fn peak_rss_bytes() -> Option<u64> {
    read("/proc/self/status")
        .ok()
        .and_then(|s| status_field(&s, "VmHWM"))
}

/// Measure `f` and return its result with the resources it used.
pub fn measure<T>(f: impl FnOnce() -> T) -> (T, ResourceUsage) {
    let s = ResourceSampler::start();
    let out = f();
    (out, s.finish())
}

#[cfg(test)]
#[path = "resources_tests.rs"]
mod resources_tests;
