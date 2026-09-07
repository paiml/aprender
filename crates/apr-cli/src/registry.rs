//! R-0b (#3002, PMAT-1073, PP-066 claim 1): backend RESOLUTION reads the
//! registry, never `cfg!`.
//!
//! R-0a (`trueno::registry`) discovers every backend kind as an entry — Ready or
//! `Unavailable(reason)` with a `source` (compiled-in / dlopen / not-compiled).
//! This module is the ONE place apr-cli turns a request (`--gpu`, `--no-gpu`,
//! `--backend <kind>`, `--gpu-layers`) into a selection or a refusal:
//!
//! * a request the build cannot honour (the kind is `not-compiled`) refuses with
//!   [`CliError::FeatureDisabled`] (exit 9) and the install line;
//! * a request the build compiled but this host does not have Ready refuses with
//!   [`CliError::BackendUnavailable`] (exit 14) and the registry's reason;
//! * a forced request NEVER downgrades to cpu; only the default (no flag) may
//!   fall to cpu, and then the selection says why (REG-8).
//!
//! Every caller that used to read `cfg!(feature = "cuda" | "wgpu")` for a
//! backend decision goes through here; `scripts/check_backend_registry.sh
//! --static` keeps the count at zero. Without the `inference` feature there is
//! no registry crate: the stub below is a cpu-only registry in which nothing but
//! cpu is compiled — the same answers, from the same rules.
use crate::error::{CliError, Result};

/// What resolution decided, printed as the `selected:` line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolved {
    /// `cpu`, `cuda`, `wgpu`, `metal`, `hip`.
    pub kind: &'static str,
    /// Device index within the kind, when a physical device was selected.
    pub device_index: Option<u32>,
    /// Stable device identity (REG-9), when known.
    pub device_uid: Option<String>,
    /// Human name of the selected device (`host cpu` for cpu).
    pub device_name: String,
    /// Why this selection (REG-8): the registry's reason or the request.
    pub reason: String,
    /// When the registry was discovered (unix seconds; 0 for the stub).
    pub discovered_at_unix: u64,
}

/// The request as the user typed it.
#[derive(Debug, Clone, Copy, Default)]
pub struct Request<'a> {
    /// `--gpu`
    pub gpu: bool,
    /// `--no-gpu` (wins over `--gpu`, as every command already treats it)
    pub no_gpu: bool,
    /// `--backend <kind>` (`cpu`, `cuda`, `wgpu`; `gpu` = any accelerator)
    pub backend: Option<&'a str>,
    /// `--gpu-layers` asked for an accelerator (`all` or `n > 0`)
    pub layers_want_accelerator: bool,
}

/// The one thing a request wants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wanted {
    /// `--no-gpu` / `--backend cpu` / `--gpu-layers 0`
    Cpu,
    /// `--backend <kind>`
    Kind(&'static str),
    /// `--gpu`, `--backend gpu`, `--gpu-layers all|n`
    AnyAccelerator,
    /// no flag: the registry's default (first Ready accelerator, else cpu)
    Default,
}

impl Request<'_> {
    /// Classify the request. `--no-gpu` wins over everything (it always has).
    #[must_use]
    pub fn wanted(&self) -> Wanted {
        if self.no_gpu {
            return Wanted::Cpu;
        }
        match self.backend {
            Some("cpu") => Wanted::Cpu,
            Some("cuda") => Wanted::Kind("cuda"),
            Some("wgpu") => Wanted::Kind("wgpu"),
            Some("metal") => Wanted::Kind("metal"),
            Some("hip") => Wanted::Kind("hip"),
            Some("gpu") => Wanted::AnyAccelerator,
            _ if self.gpu || self.layers_want_accelerator => Wanted::AnyAccelerator,
            _ => Wanted::Default,
        }
    }
}

/// The install line every not-compiled refusal ends with (aprender#2696).
pub(crate) const INSTALL_HINT: &str = "Install a build that has one:\n\
\n\
\x20    cargo install aprender --features cuda    # NVIDIA\n\
\x20    cargo install aprender --features wgpu    # portable GPU backend\n\
\n\
Or pass --no-gpu to run on CPU deliberately.";

/// `FeatureDisabled`: the kind is not compiled into this binary.
pub(crate) fn not_compiled(asked: &str, kind: &str) -> CliError {
    CliError::FeatureDisabled(format!(
        "{asked} was requested, but the {kind} backend is not compiled into this build \n\
         (registry: {kind}=NotCompiled), so it would have run on CPU without telling you. \n\
         On a 7B Q4_K_M model that is roughly a tenth of the decode rate and several \n\
         seconds of extra latency to the first token (aprender#2696).\n\
         \n\
         {INSTALL_HINT}"
    ))
}

/// `BackendUnavailable`: compiled, but this host has nothing Ready for it.
pub(crate) fn unavailable(asked: &str, reasons: &str) -> CliError {
    CliError::BackendUnavailable(format!(
        "{asked} was requested, but no such backend is Ready on this host \n\
         (registry: {reasons}). A forced backend is never downgraded to cpu: \n\
         run `apr devices` to see every backend with its reason, or pass --no-gpu \n\
         to run on CPU deliberately."
    ))
}

/// The `selected:` line every run/chat/serve prints at load (REG-8 + REG-15):
/// the parity text, when a gate ran, joins the registry's reason.
#[must_use]
pub fn selected_line(r: &Resolved, parity: Option<&str>) -> String {
    let device = match r.device_index {
        Some(i) => format!(" device[{i}]={}", r.device_name),
        None if r.kind == "cpu" => String::new(),
        None => format!(" {}", r.device_name),
    };
    match parity {
        Some(p) => format!("selected: {}{device} ({p}; registry: {})", r.kind, r.reason),
        None => format!("selected: {}{device} (registry: {})", r.kind, r.reason),
    }
}

#[cfg(feature = "inference")]
mod real {
    use super::{not_compiled, unavailable, CliError, Request, Resolved, Result, Wanted};
    use std::sync::OnceLock;
    use trueno::registry::{default_factories, BackendKind, BackendRegistry, Source, Status};

    /// Load the registry the way `apr devices` does: `APR_REGISTRY_FIXTURE` and
    /// `APR_RESERVE_BYTES` are honoured (and printed by `apr devices`, REG-8).
    ///
    /// # Errors
    /// Only a malformed override is an error (exit 4).
    pub fn load() -> Result<BackendRegistry> {
        let reserve = crate::commands::devices::reserve_override()?;
        let fixture = std::env::var("APR_REGISTRY_FIXTURE")
            .ok()
            .filter(|p| !p.is_empty());
        Ok(match &fixture {
            Some(path) => {
                let text = std::fs::read_to_string(path).map_err(|e| {
                    CliError::InvalidInput(format!("APR_REGISTRY_FIXTURE {path}: {e}"))
                })?;
                let reg = BackendRegistry::from_fixture_json(&text, path)
                    .map_err(CliError::InvalidInput)?;
                match reserve {
                    Some(r) => reg.with_reserve(r, "APR_RESERVE_BYTES override"),
                    None => reg,
                }
            }
            None => BackendRegistry::discover_with(&default_factories(), reserve),
        })
    }

    /// The process-wide registry: discovered once, on first use.
    pub fn current() -> Result<&'static BackendRegistry> {
        static REG: OnceLock<std::result::Result<BackendRegistry, String>> = OnceLock::new();
        match REG.get_or_init(|| load().map_err(|e| e.to_string())) {
            Ok(r) => Ok(r),
            Err(e) => Err(CliError::InvalidInput(e.clone())),
        }
    }

    fn kind_of(s: &str) -> Option<BackendKind> {
        match s {
            "cpu" => Some(BackendKind::Cpu),
            "cuda" => Some(BackendKind::Cuda),
            "wgpu" => Some(BackendKind::Wgpu),
            "metal" => Some(BackendKind::Metal),
            "hip" => Some(BackendKind::Hip),
            _ => None,
        }
    }

    fn is_compiled(e: &trueno::registry::BackendEntry) -> bool {
        !matches!(e.source, Source::NotCompiled)
            && !matches!(
                e.status,
                Status::Unavailable(trueno::registry::Reason::NotCompiled)
            )
    }

    fn resolved_from(
        reg: &BackendRegistry,
        e: &trueno::registry::BackendEntry,
        reason: String,
    ) -> Resolved {
        Resolved {
            kind: e.kind.as_str(),
            device_index: e.device_index,
            device_uid: e.device_uid.clone(),
            device_name: e.device_name.clone(),
            reason,
            discovered_at_unix: reg.discovered_at_unix,
        }
    }

    fn cpu_resolved(reg: &BackendRegistry, reason: &str) -> Resolved {
        let cpu = reg.entries.iter().find(|e| e.kind == BackendKind::Cpu);
        Resolved {
            kind: "cpu",
            device_index: None,
            device_uid: None,
            device_name: cpu.map_or_else(|| "host cpu".to_string(), |e| e.device_name.clone()),
            reason: reason.to_string(),
            discovered_at_unix: reg.discovered_at_unix,
        }
    }

    /// Whether any accelerator kind is compiled into this build (REG-2).
    #[must_use]
    pub fn build_has_accelerator_in(reg: &BackendRegistry) -> bool {
        reg.entries
            .iter()
            .any(|e| e.kind != BackendKind::Cpu && is_compiled(e))
    }

    /// Whether `kind` is compiled into this build.
    #[must_use]
    pub fn compiled_in(reg: &BackendRegistry, kind: &str) -> bool {
        kind_of(kind).is_some_and(|k| reg.entries.iter().any(|e| e.kind == k && is_compiled(e)))
    }

    /// Resolve `req` against `reg`. Forced requests never downgrade.
    ///
    /// # Errors
    /// `FeatureDisabled` (not compiled) or `BackendUnavailable` (not Ready here).
    pub fn resolve_in(req: &Request<'_>, asked: &str, reg: &BackendRegistry) -> Result<Resolved> {
        match req.wanted() {
            Wanted::Cpu => Ok(cpu_resolved(reg, &format!("{asked}: cpu requested"))),
            Wanted::Default => Ok(resolve_default(reg)),
            Wanted::Kind(kind) => resolve_kind(reg, asked, kind),
            Wanted::AnyAccelerator => resolve_any(reg, asked),
        }
    }

    fn ready_entry<'a>(
        reg: &'a BackendRegistry,
        pred: impl Fn(&&'a trueno::registry::BackendEntry) -> bool,
    ) -> Option<&'a trueno::registry::BackendEntry> {
        reg.entries
            .iter()
            .find(|e| matches!(e.status, trueno::registry::Status::Ready) && pred(e))
    }

    fn resolve_default(reg: &BackendRegistry) -> Resolved {
        let sel = reg.select_default();
        match ready_entry(reg, |e| e.kind == sel.kind && e.device_index == sel.device_index) {
            Some(e) if e.kind != BackendKind::Cpu => resolved_from(reg, e, sel.reason),
            _ => cpu_resolved(reg, &sel.reason),
        }
    }

    fn resolve_kind(reg: &BackendRegistry, asked: &str, kind: &str) -> Result<Resolved> {
        let Some(k) = kind_of(kind) else {
            return Err(CliError::InvalidInput(format!("{asked}: unknown backend `{kind}`")));
        };
        if k == BackendKind::Cpu {
            return Ok(cpu_resolved(reg, &format!("{asked}: cpu requested")));
        }
        if let Some(e) = ready_entry(reg, |e| e.kind == k) {
            return Ok(resolved_from(reg, e, format!("{asked}: Ready")));
        }
        let mine: Vec<&trueno::registry::BackendEntry> =
            reg.entries.iter().filter(|e| e.kind == k).collect();
        if mine.iter().all(|e| !is_compiled(e)) {
            return Err(not_compiled(asked, kind));
        }
        Err(unavailable(asked, &reasons_of(&mine)))
    }

    fn resolve_any(reg: &BackendRegistry, asked: &str) -> Result<Resolved> {
        if let Some(e) = ready_entry(reg, |e| e.kind != BackendKind::Cpu) {
            return Ok(resolved_from(reg, e, format!("{asked}: first Ready accelerator")));
        }
        if !build_has_accelerator_in(reg) {
            return Err(not_compiled(asked, "cuda/wgpu"));
        }
        let non_cpu: Vec<&trueno::registry::BackendEntry> = reg
            .entries
            .iter()
            .filter(|e| e.kind != BackendKind::Cpu && is_compiled(e))
            .collect();
        Err(unavailable(asked, &reasons_of(&non_cpu)))
    }

    fn reasons_of(entries: &[&trueno::registry::BackendEntry]) -> String {
        let v: Vec<String> = entries
            .iter()
            .map(|e| match &e.status {
                Status::Unavailable(r) => format!("{}={}", e.kind.as_str(), r.text()),
                Status::Ready => format!("{}=Ready", e.kind.as_str()),
            })
            .collect();
        if v.is_empty() {
            "no entry".to_string()
        } else {
            v.join(", ")
        }
    }

    pub fn resolve(req: &Request<'_>, asked: &str) -> Result<Resolved> {
        resolve_in(req, asked, current()?)
    }
    pub fn build_has_accelerator() -> bool {
        current().map(build_has_accelerator_in).unwrap_or(false)
    }
    pub fn compiled(kind: &str) -> bool {
        current().map(|r| compiled_in(r, kind)).unwrap_or(false)
    }
    pub fn compute_class() -> &'static str {
        current()
            .map(|r| r.select_default().kind.as_str())
            .unwrap_or("cpu")
    }
}

#[cfg(not(feature = "inference"))]
mod real {
    //! No inference stack: nothing but cpu is compiled. Same rules, same refusals.
    use super::{not_compiled, Request, Resolved, Result, Wanted};

    fn cpu(reason: &str) -> Resolved {
        Resolved {
            kind: "cpu",
            device_index: None,
            device_uid: None,
            device_name: "host cpu".to_string(),
            reason: reason.to_string(),
            discovered_at_unix: 0,
        }
    }
    pub fn resolve(req: &Request<'_>, asked: &str) -> Result<Resolved> {
        match req.wanted() {
            Wanted::Cpu => Ok(cpu(&format!("{asked}: cpu requested"))),
            Wanted::Default => Ok(cpu("no inference feature: cpu is the only backend")),
            Wanted::Kind(kind) => Err(not_compiled(asked, kind)),
            Wanted::AnyAccelerator => Err(not_compiled(asked, "cuda/wgpu")),
        }
    }
    pub fn build_has_accelerator() -> bool {
        false
    }
    pub fn compiled(kind: &str) -> bool {
        kind == "cpu"
    }
    pub fn compute_class() -> &'static str {
        "cpu"
    }
}

pub use real::{build_has_accelerator, compiled, compute_class, resolve};

/// Resolve `req` and print the `selected:` line (REG-8, R-0b: "selected:
/// always") to stderr — once per process, so a command whose preflight
/// resolves twice (an explicit `--backend` and then `--gpu`) prints one line.
/// The parity text joins it at the CUDA load site, which prints its own
/// `parity:` line from the gate record (REG-15).
///
/// # Errors
/// The same refusals as [`resolve`].
pub fn announce(req: &Request<'_>, asked: &str) -> Result<Resolved> {
    let r = resolve(req, asked)?;
    if !ANNOUNCED.swap(true, std::sync::atomic::Ordering::SeqCst) {
        eprintln!("{}", selected_line(&r, None));
    }
    Ok(r)
}

static ANNOUNCED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The `parity:` line a CUDA load site prints from its admission record
/// (REG-15, L0-1a): the companion of the `selected:` line, printed once the
/// gate has actually run on this model.
#[must_use]
pub fn parity_line(
    status: &str,
    cosine: Option<f32>,
    positions: usize,
    threshold: f32,
    basis: &str,
) -> String {
    match cosine {
        Some(c) => format!(
            "parity: {status} cosine={c:.6} positions={positions} threshold={threshold} basis={basis}"
        ),
        None => format!("parity: {status} positions={positions} threshold={threshold} basis={basis}"),
    }
}
#[cfg(feature = "inference")]
pub use real::{build_has_accelerator_in, compiled_in, current, load, resolve_in};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_gpu_wins_and_the_backend_flag_names_its_kind() {
        assert_eq!(
            Request {
                gpu: true,
                no_gpu: true,
                ..Default::default()
            }
            .wanted(),
            Wanted::Cpu
        );
        assert_eq!(
            Request {
                backend: Some("cpu"),
                gpu: true,
                ..Default::default()
            }
            .wanted(),
            Wanted::Cpu
        );
        assert_eq!(
            Request {
                backend: Some("cuda"),
                ..Default::default()
            }
            .wanted(),
            Wanted::Kind("cuda")
        );
        assert_eq!(
            Request {
                gpu: true,
                ..Default::default()
            }
            .wanted(),
            Wanted::AnyAccelerator
        );
        assert_eq!(
            Request {
                layers_want_accelerator: true,
                ..Default::default()
            }
            .wanted(),
            Wanted::AnyAccelerator
        );
        assert_eq!(Request::default().wanted(), Wanted::Default);
    }

    #[test]
    fn the_selected_line_names_the_kind_the_device_and_the_reason() {
        let r = Resolved {
            kind: "cuda",
            device_index: Some(0),
            device_uid: None,
            device_name: "NVIDIA X".into(),
            reason: "first Ready".into(),
            discovered_at_unix: 1,
        };
        assert_eq!(
            selected_line(&r, Some("parity: PASS cosine=0.9998 positions=78")),
            "selected: cuda device[0]=NVIDIA X (parity: PASS cosine=0.9998 positions=78; registry: first Ready)"
        );
        let c = Resolved {
            kind: "cpu",
            device_index: None,
            device_uid: None,
            device_name: "host cpu".into(),
            reason: "--no-gpu: cpu requested".into(),
            discovered_at_unix: 0,
        };
        assert_eq!(
            selected_line(&c, None),
            "selected: cpu (registry: --no-gpu: cpu requested)"
        );
    }

    #[test]
    fn a_forced_accelerator_on_a_build_without_one_is_feature_disabled_never_cpu() {
        // On every build the stub answers; on inference builds the machine registry answers.
        let r = resolve(
            &Request {
                backend: Some("metal"),
                ..Default::default()
            },
            "--backend metal",
        );
        match r {
            Ok(res) => assert_eq!(
                res.kind, "metal",
                "a Ready metal is the only way to get one"
            ),
            Err(e) => assert!(
                matches!(
                    e,
                    CliError::FeatureDisabled(_) | CliError::BackendUnavailable(_)
                ),
                "{e}"
            ),
        }
    }
}
