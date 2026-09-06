//! BackendRegistry — probe → enumerate → print (PP-066 R-0a, #2904, PMAT-989).
//!
//! Hardware selection used to be bound to the BUILD (`cfg!(feature = "cuda")`
//! read at decision time), so one binary had to serve every installer and a
//! missing `libcuda.so.1` was a silent CPU run. This module discovers the
//! machine at startup and represents every backend kind of the fixed list
//! `{cpu, cuda, wgpu, metal, hip}` as an explicit entry — `Ready` or
//! `Unavailable(reason)` — so the absence of a backend is a line the user
//! reads (REG-11), never a silence. The resolution half (refusing a request
//! that is not in the Ready set) is R-0b (#3002); this half exists and prints.
//!
//! Design decisions from the 2026-09-06 quorum (docs/audits/pp-066-r0-design-quorum.md):
//! object-safe [`BackendFactory`] registered at startup (REG-13); entries carry
//! `api` + `device_uid` so one physical GPU reachable through two APIs is two
//! entries and ONE device for selection; `vendor_id` is optional (Apple silicon
//! has no PCI id); unified memory carries its working-set limit and the REG-7
//! reserve is checked against free memory; nothing is persisted (REG-12).

use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
mod wgpu_probe;

/// The fixed kind list every `apr devices` output carries (REG-11).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendKind {
    /// Always present, always Ready.
    Cpu,
    /// NVIDIA through the driver API (`libcuda.so.1` / `nvcuda.dll`, dlopen).
    Cuda,
    /// wgpu adapters (Vulkan / Metal / DX12 / GL transports).
    Wgpu,
    /// Native Metal — no native backend in 0.66; a Metal adapter appears under `wgpu`.
    Metal,
    /// AMD HIP — no backend in 0.66; an AMD device is listed as `NoBackend`, not ignored (REG-4).
    Hip,
}

impl BackendKind {
    /// The fixed, ordered list (REG-11).
    pub const ALL: [BackendKind; 5] = [Self::Cpu, Self::Cuda, Self::Wgpu, Self::Metal, Self::Hip];

    /// The printed name.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Wgpu => "wgpu",
            Self::Metal => "metal",
            Self::Hip => "hip",
        }
    }
}

/// The API an entry was discovered through (lane 2: wgpu is a transport over a
/// physical backend, so the API is a field, not a peer kind).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Api {
    /// Host CPU.
    Cpu,
    /// `libcuda.so.1` driver API.
    CudaDriver,
    /// wgpu adapter enumeration; `transport` names Vulkan/Metal/DX12/GL.
    Wgpu,
    /// Native Metal.
    Metal,
    /// HIP.
    Hip,
}

/// Where the entry came from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "detail", rename_all = "kebab-case")]
pub enum Source {
    /// Linked into the binary.
    CompiledIn,
    /// Loaded at run time from this library path.
    Dlopen(String),
    /// The feature was not compiled into this binary.
    NotCompiled,
    /// Read from a fixture file (tests, dogfood); never silent.
    Fixture(String),
}

impl Source {
    /// One printable token.
    #[must_use]
    pub fn text(&self) -> String {
        match self {
            Self::CompiledIn => "compiled-in".to_string(),
            Self::Dlopen(p) => format!("dlopen({p})"),
            Self::NotCompiled => "not-compiled".to_string(),
            Self::Fixture(p) => format!("fixture({p})"),
        }
    }
}

/// Why an entry is not Ready (REG-11: every non-Ready entry names its reason).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum Reason {
    /// The feature is not in this binary.
    NotCompiled,
    /// The driver library could not be loaded.
    DriverNotFound { path: String },
    /// The driver loaded but reports no device.
    NoDevice,
    /// A device exists but apr has no backend for it (e.g. AMD without HIP).
    NoBackend { vendor: String },
    /// The probe returned an error; the process stayed up (REG-1).
    ProbeFailed { error: String },
    /// REG-7: the reserve does not fit in free memory.
    ReserveExceedsFree { reserve_bytes: u64, free_bytes: u64 },
}

impl Reason {
    /// One printable token, `Name(detail)`.
    #[must_use]
    pub fn text(&self) -> String {
        match self {
            Self::NotCompiled => "NotCompiled".to_string(),
            Self::DriverNotFound { path } => format!("DriverNotFound({path})"),
            Self::NoDevice => "NoDevice".to_string(),
            Self::NoBackend { vendor } => format!("NoBackend({vendor})"),
            Self::ProbeFailed { error } => format!("ProbeFailed({error})"),
            Self::ReserveExceedsFree { reserve_bytes, free_bytes } => {
                format!("ReserveExceedsFree{{reserve={reserve_bytes}, free={free_bytes}}}")
            }
        }
    }
}

/// Ready, or not — with the reason.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "kebab-case")]
pub enum Status {
    /// Usable now.
    Ready,
    /// Not usable; the reason is printed.
    Unavailable(Reason),
}

/// Discrete VRAM, or a pool shared with the host (REG-6).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum MemKind {
    /// Dedicated device memory.
    Discrete,
    /// Shared pool (gx10 GB10, Apple silicon); the OS working-set limit, when
    /// known, is what allocations are capped at — not physical RAM.
    Unified { working_set_limit: Option<u64> },
}

/// One line of `apr devices`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BackendEntry {
    /// Which of the five kinds.
    pub kind: BackendKind,
    /// The API this entry was discovered through.
    pub api: Api,
    /// Device ordinal within its API, if any.
    pub device_index: Option<u32>,
    /// Stable identity of the PHYSICAL device; two entries with the same uid are one device.
    pub device_uid: Option<String>,
    /// Human name.
    pub device_name: String,
    /// Vendor name ("NVIDIA", "AMD", "Apple", "Intel", "host").
    pub vendor: String,
    /// PCI vendor id where one exists.
    pub vendor_id: Option<u32>,
    /// "cpu", "discrete-gpu", "integrated-gpu", "virtual-gpu", "software".
    pub device_type: String,
    /// Bytes.
    pub mem_total: Option<u64>,
    /// Bytes, measured at discovery (never cached across runs).
    pub mem_free: Option<u64>,
    /// Discrete or unified.
    pub mem_kind: MemKind,
    /// "sm_89", "avx512", ...
    pub compute_class: Option<String>,
    /// ggml-style capability names.
    pub caps: Vec<String>,
    /// Where it came from.
    pub source: Source,
    /// Ready or the reason.
    pub status: Status,
    /// For wgpu: the transport ("vulkan", "metal", "dx12", "gl").
    pub transport: Option<String>,
}

impl BackendEntry {
    /// The placeholder line for a kind nothing discovered (REG-11: absence is a line).
    #[must_use]
    pub fn unavailable(kind: BackendKind, api: Api, source: Source, reason: Reason) -> Self {
        Self {
            kind,
            api,
            device_index: None,
            device_uid: None,
            device_name: String::new(),
            vendor: String::new(),
            vendor_id: None,
            device_type: String::new(),
            mem_total: None,
            mem_free: None,
            mem_kind: MemKind::Discrete,
            compute_class: None,
            caps: Vec::new(),
            source,
            status: Status::Unavailable(reason),
            transport: None,
        }
    }

    fn is_ready(&self) -> bool {
        self.status == Status::Ready
    }

    fn identity(&self) -> String {
        self.device_uid
            .clone()
            .unwrap_or_else(|| format!("{}:{:?}", self.kind.as_str(), self.device_index))
    }
}

/// A backend that can discover its devices (REG-13: registered by trait, not
/// by feature flag; object-safe on purpose — `ComputeBackend` is not).
pub trait BackendFactory: Send + Sync {
    /// The kind this factory reports.
    fn kind(&self) -> BackendKind;
    /// Every device this backend can see, Ready or not. Must never abort the
    /// process (REG-1): a driver fault is an `Unavailable(ProbeFailed)` entry.
    fn discover(&self) -> Vec<BackendEntry>;
}

/// A factory that returns canned entries (tests, the CI case table, REG-13's mock).
pub struct MockBackendFactory {
    kind: BackendKind,
    entries: Vec<BackendEntry>,
}

impl MockBackendFactory {
    /// Canned entries for `kind`.
    #[must_use]
    pub fn new(kind: BackendKind, entries: Vec<BackendEntry>) -> Self {
        Self { kind, entries }
    }
}

impl BackendFactory for MockBackendFactory {
    fn kind(&self) -> BackendKind {
        self.kind
    }
    fn discover(&self) -> Vec<BackendEntry> {
        self.entries.clone()
    }
}

/// REG-8: what `apr` would run on with no flag, and why.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Selection {
    /// The selected kind (cpu when no GPU entry is Ready).
    pub kind: BackendKind,
    /// The device ordinal within that kind, if any.
    pub device_index: Option<u32>,
    /// The device identity, if any.
    pub device_uid: Option<String>,
    /// Always printed: why this one.
    pub reason: String,
}

/// The default reserve until PP-LLAMA-001 master row 6 measures `vram_peak` (REG-7).
pub const DEFAULT_RESERVE_BYTES: u64 = 3_584 * 1024 * 1024;
/// The basis tag printed beside the default reserve.
pub const DEFAULT_RESERVE_BASIS: &str = "[U] default until master row 6 measures vram_peak";
/// The JSON schema id `apr devices --json` carries.
pub const SCHEMA: &str = "apr-devices-v1";

/// The machine as discovered at startup. Never cached, never persisted (REG-12).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BackendRegistry {
    /// Schema id.
    pub schema: String,
    /// Unix seconds at discovery.
    pub discovered_at_unix: u64,
    /// "machine" or "fixture(path)".
    pub source: String,
    /// REG-7 reserve applied to every Ready GPU entry.
    pub reserve_bytes: u64,
    /// Where the reserve number comes from.
    pub reserve_basis: String,
    /// Every entry, in kind order; at least one per kind.
    pub entries: Vec<BackendEntry>,
    /// REG-8 default selection (recomputed on every discovery).
    pub selected: Selection,
}

impl BackendRegistry {
    /// Discover the machine with the built-in factories.
    #[must_use]
    pub fn discover() -> Self {
        Self::discover_with(&default_factories(), None)
    }

    /// Discover with explicit factories (tests, mocks) and an optional reserve
    /// override (`APR_RESERVE_BYTES`; the CLI prints the override — REG-8).
    #[must_use]
    pub fn discover_with(
        factories: &[Box<dyn BackendFactory>],
        reserve_bytes: Option<u64>,
    ) -> Self {
        let (reserve, basis) = match reserve_bytes {
            Some(r) => (r, "APR_RESERVE_BYTES override".to_string()),
            None => (DEFAULT_RESERVE_BYTES, DEFAULT_RESERVE_BASIS.to_string()),
        };
        let mut entries = vec![cpu_entry()];
        for kind in [BackendKind::Cuda, BackendKind::Wgpu, BackendKind::Metal, BackendKind::Hip] {
            let mut found: Vec<BackendEntry> =
                factories.iter().filter(|f| f.kind() == kind).flat_map(|f| f.discover()).collect();
            if found.is_empty() {
                found.push(missing_entry(kind));
            }
            entries.extend(found);
        }
        apply_reserve(&mut entries, reserve);
        let selected = select(&entries, reserve);
        Self {
            schema: SCHEMA.to_string(),
            discovered_at_unix: now_unix(),
            source: "machine".to_string(),
            reserve_bytes: reserve,
            reserve_basis: basis,
            entries,
            selected,
        }
    }

    /// Build from a fixture document (the JSON `to_json` writes). The source
    /// names the file so a fixture-built registry is never mistaken for the machine.
    ///
    /// # Errors
    /// The JSON does not parse as a registry document.
    pub fn from_fixture_json(json: &str, path: &str) -> Result<Self, String> {
        let mut reg: Self =
            serde_json::from_str(json).map_err(|e| format!("fixture {path}: {e}"))?;
        reg.source = format!("fixture({path})");
        reg.selected = select(&reg.entries, reg.reserve_bytes);
        Ok(reg)
    }

    /// The Ready entries.
    pub fn ready(&self) -> impl Iterator<Item = &BackendEntry> {
        self.entries.iter().filter(|e| e.is_ready())
    }

    /// REG-8: first Ready non-cpu entry (deduplicated by `device_uid`), else cpu.
    #[must_use]
    pub fn select_default(&self) -> Selection {
        select(&self.entries, self.reserve_bytes)
    }

    /// Physical devices among the Ready non-cpu entries (two entries with one
    /// uid count once — lane 2's rule).
    #[must_use]
    pub fn distinct_devices(&self) -> usize {
        let mut seen: Vec<String> = Vec::new();
        for e in self.entries.iter().filter(|e| e.is_ready() && e.kind != BackendKind::Cpu) {
            let id = e.identity();
            if !seen.contains(&id) {
                seen.push(id);
            }
        }
        seen.len()
    }

    /// The JSON document `apr devices --json` prints.
    ///
    /// # Errors
    /// Serialisation failed (cannot happen for these types; surfaced anyway).
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| e.to_string())
    }

    /// The printed block (spec §5 R-0, normative shape).
    #[must_use]
    pub fn render_block(&self, version: &str) -> String {
        let mut out = format!(
            "apr {version}  discovery unix={}  source={}\n",
            self.discovered_at_unix, self.source
        );
        for e in &self.entries {
            out.push_str(&render_entry(e));
            out.push('\n');
        }
        let s = &self.selected;
        let dev = s.device_index.map(|i| format!(" device[{i}]")).unwrap_or_default();
        out.push_str(&format!(
            "selected: {}{dev}  reserve={}MiB basis={}  ({})\n",
            s.kind.as_str(),
            self.reserve_bytes / (1024 * 1024),
            self.reserve_basis,
            s.reason
        ));
        out
    }
}

fn render_entry(e: &BackendEntry) -> String {
    let kind = format!("{:<6}", e.kind.as_str());
    match &e.status {
        Status::Unavailable(r) => {
            format!("backend: {kind} unavailable  reason={} source={}", r.text(), e.source.text())
        }
        Status::Ready => {
            let mut line = format!("backend: {kind} ready       ");
            if let Some(i) = e.device_index {
                line.push_str(&format!(" device[{i}]=\"{}\"", e.device_name));
            } else {
                line.push_str(&format!(" {}", e.device_name));
            }
            if let Some(cc) = &e.compute_class {
                line.push_str(&format!(" class={cc}"));
            }
            if let Some(t) = e.mem_total {
                line.push_str(&format!(" mem={}MiB", t / (1024 * 1024)));
            }
            if let Some(f) = e.mem_free {
                line.push_str(&format!(" free={}MiB", f / (1024 * 1024)));
            }
            line.push_str(match &e.mem_kind {
                MemKind::Discrete => " kind=discrete",
                MemKind::Unified { .. } => " kind=unified",
            });
            if let Some(t) = &e.transport {
                line.push_str(&format!(" transport={t}"));
            }
            if !e.caps.is_empty() {
                line.push_str(&format!(" caps={{{}}}", e.caps.join(",")));
            }
            line.push_str(&format!(" source={}", e.source.text()));
            line
        }
    }
}

fn apply_reserve(entries: &mut [BackendEntry], reserve: u64) {
    for e in entries.iter_mut().filter(|e| e.kind != BackendKind::Cpu && e.is_ready()) {
        if let Some(free) = e.mem_free {
            if free < reserve {
                e.status = Status::Unavailable(Reason::ReserveExceedsFree {
                    reserve_bytes: reserve,
                    free_bytes: free,
                });
            }
        }
    }
}

fn select(entries: &[BackendEntry], reserve: u64) -> Selection {
    if let Some(e) = entries.iter().find(|e| e.kind != BackendKind::Cpu && e.is_ready()) {
        return Selection {
            kind: e.kind,
            device_index: e.device_index,
            device_uid: e.device_uid.clone(),
            reason: format!(
                "first Ready non-cpu entry; {} physical device(s) Ready",
                count_distinct(entries)
            ),
        };
    }
    let why = entries
        .iter()
        .filter(|e| e.kind != BackendKind::Cpu)
        .filter_map(|e| match &e.status {
            Status::Unavailable(r) => Some(format!("{}={}", e.kind.as_str(), r.text())),
            Status::Ready => None,
        })
        .collect::<Vec<_>>()
        .join(", ");
    let reserve_note = if why.contains("ReserveExceedsFree") {
        format!("; reserve={reserve} B exceeds free memory")
    } else {
        String::new()
    };
    Selection {
        kind: BackendKind::Cpu,
        device_index: None,
        device_uid: None,
        reason: format!("no ready gpu: {why}{reserve_note}"),
    }
}

fn count_distinct(entries: &[BackendEntry]) -> usize {
    let mut seen: Vec<String> = Vec::new();
    for e in entries.iter().filter(|e| e.is_ready() && e.kind != BackendKind::Cpu) {
        let id = e.identity();
        if !seen.contains(&id) {
            seen.push(id);
        }
    }
    seen.len()
}

fn missing_entry(kind: BackendKind) -> BackendEntry {
    match kind {
        BackendKind::Cuda => BackendEntry::unavailable(
            kind,
            Api::CudaDriver,
            Source::NotCompiled,
            Reason::NotCompiled,
        ),
        BackendKind::Wgpu => {
            BackendEntry::unavailable(kind, Api::Wgpu, Source::NotCompiled, Reason::NotCompiled)
        }
        BackendKind::Metal => BackendEntry::unavailable(
            kind,
            Api::Metal,
            Source::NotCompiled,
            Reason::NoBackend {
                vendor: "no native Metal backend in 0.66 (a Metal adapter appears under wgpu)"
                    .to_string(),
            },
        ),
        BackendKind::Hip => BackendEntry::unavailable(
            kind,
            Api::Hip,
            Source::NotCompiled,
            Reason::NoBackend { vendor: "no HIP backend in 0.66".to_string() },
        ),
        BackendKind::Cpu => cpu_entry(),
    }
}

fn now_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn cpu_entry() -> BackendEntry {
    let threads =
        std::thread::available_parallelism().map(std::num::NonZeroUsize::get).unwrap_or(1);
    BackendEntry {
        kind: BackendKind::Cpu,
        api: Api::Cpu,
        device_index: None,
        device_uid: Some("host-cpu".to_string()),
        device_name: format!("{} host cpu, {threads} threads", std::env::consts::ARCH),
        vendor: "host".to_string(),
        vendor_id: None,
        device_type: "cpu".to_string(),
        mem_total: host_mem_total(),
        mem_free: None,
        mem_kind: MemKind::Unified { working_set_limit: None },
        compute_class: Some(cpu_isa()),
        caps: Vec::new(),
        source: Source::CompiledIn,
        status: Status::Ready,
        transport: None,
    }
}

fn cpu_isa() -> String {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            return "avx512".to_string();
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return "avx2".to_string();
        }
        return "sse2".to_string();
    }
    #[cfg(target_arch = "aarch64")]
    {
        return "neon".to_string();
    }
    #[allow(unreachable_code)]
    std::env::consts::ARCH.to_string()
}

fn host_mem_total() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = text.lines().find(|l| l.starts_with("MemTotal:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

/// The factories this binary was built with. A kind whose feature is off is
/// still a line (`NotCompiled`), produced by the registry itself.
#[must_use]
pub fn default_factories() -> Vec<Box<dyn BackendFactory>> {
    let v: Vec<Box<dyn BackendFactory>> = vec![
        #[cfg(feature = "cuda")]
        Box::new(cuda::CudaFactory),
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        Box::new(wgpu_probe::WgpuFactory),
    ];
    v
}

/// Stable identity for a physical device seen through any API: vendor prefix +
/// normalised name, so the cuda-driver and wgpu twins of one card agree.
#[must_use]
pub fn device_uid(vendor: &str, name: &str) -> String {
    let norm: String = name
        .trim()
        .to_ascii_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    format!("{}:{}", vendor.to_ascii_lowercase(), norm.trim_matches('-'))
}
