//! APR model serving command (PMAT-200: split from monolithic serve.rs)
//!
//! Serves ML models via HTTP API with support for APR, GGUF, and SafeTensors formats.
//! Implements OpenAI-compatible endpoints for generation, prediction, and transcription.

// Submodules (PMAT-200: split from 4351-line serve.rs)
pub mod auth;
#[cfg(feature = "inference")]
pub mod handlers;
#[cfg(feature = "inference")]
pub mod ollama;
pub mod routes;
#[cfg(feature = "inference")]
pub mod safetensors;
pub mod types;

// Re-exports for backward compatibility
pub use types::*;

// Test modules
#[cfg(test)]
mod tests;
// PP-LLAMA-001 PP-14/PP-15/§9 #8: the offload report the served process
// publishes. `inference`-gated because the report type comes from realizar.
#[cfg(all(test, feature = "inference"))]
#[path = "tests_offload_report_pp14.rs"]
mod tests_offload_report_pp14;

use std::path::Path;

use colored::Colorize;

use crate::error::{CliError, Result};
pub(crate) use types::GpuLayerRequest;

/// PERF-021 countermeasure 2: report what this BUILD can dispatch to.
///
/// #2696's user had no way to ask this. `apr serve run --gpu` accepted the flag
/// and ran on CPU; nothing anywhere said the binary had no CUDA in it. This
/// answers the question without a model, a port, or a GPU.
pub(crate) fn list_devices() -> Result<()> {
    println!("accelerators this BUILD can dispatch to:");
    let mut any = false;
    // R-0b (#3002): what is compiled comes from the registry, never `cfg!`
    // (`apr devices` prints every backend with its Ready/Unavailable reason).
    for kind in ["cuda", "wgpu", "metal", "hip"] {
        if crate::registry::compiled(kind) {
            println!("  {kind:<7} compiled in");
            any = true;
        }
    }
    println!("  cpu     always available");
    if !any {
        println!();
        println!("This build has NO accelerator compiled in. `--gpu-layers` above 0");
        println!("will be refused rather than silently served from the CPU.");
        println!();
        println!("    cargo install aprender --features cuda    # NVIDIA");
        println!("    cargo install aprender --features wgpu    # portable GPU backend");
    }
    Ok(())
}

/// PP-LLAMA-001 §9 #8 / PP-2: the `cfg!` feature set of THIS binary.
///
/// `realizar` cannot see it — `cuda-batch` is an `apr-cli` feature and no
/// mechanism carried it into the served process's HTTP surface, so
/// `feature_set` in a receipt was whatever the operator typed at
/// `--server-feature`. The list travels with the offload report and is
/// reported as `build_features_cli`.
#[cfg(feature = "inference")]
#[must_use]
pub(crate) fn cli_build_features() -> Vec<String> {
    let mut features: Vec<&'static str> = Vec::new();
    if cfg!(feature = "inference") {
        features.push("inference");
    }
    // R-0b: the compiled backends come from the registry, never `cfg!`.
    if crate::registry::compiled("cuda") {
        features.push("cuda");
    }
    // §9 #8: `cuda-batch = ["cuda"]` is a compatibility alias now, but a receipt
    // still has to state whether the binary was built with it — the §2.1
    // INVALID-BUILD rule reads exactly this entry.
    if cfg!(feature = "cuda-batch") {
        features.push("cuda-batch");
    }
    if crate::registry::compiled("wgpu") {
        features.push("wgpu");
    }
    if cfg!(feature = "training") {
        features.push("training");
    }
    features.into_iter().map(str::to_string).collect()
}

/// PP-14/PP-15: what this loader resolved for `--gpu-layers`, as a value the
/// served process can report.
///
/// The three numbers were computed here and PRINTED ONLY, beside a `backend=`
/// label that was a `cfg!` — a build-time string, not what loaded. Attaching
/// this to the `AppState` is what turns the printed line into a fact a receipt
/// can carry, and what lets `backend_loaded` be derived from residency instead.
///
/// `explicit_args` lists the arguments whose value DIFFERS from the default,
/// i.e. the ones the operator must have set. An operator who typed the default
/// (`--context-length 4096`) is indistinguishable from one who typed nothing,
/// and is reported as not explicit — the conservative direction, since PP-14's
/// invariant is violated by claiming auto-fit chose something the operator set.
///
/// `autofit_applied` is EMPTY on this loader, and that is a fact rather than an
/// omission: `resolve_layers` has no free-VRAM query to fit against (`fits ==
/// total_layers`), so `auto` resolves to all, a partial request is refused, and
/// auto-fit never changes anything. When a fitting loader lands, this is where
/// it records what it changed.
#[cfg(feature = "inference")]
#[must_use]
pub(crate) fn offload_report(
    config: &ServerConfig,
    resolved_layers: u32,
    total_layers: u32,
) -> realizar::api::OffloadReport {
    let mut explicit_args: Vec<String> = Vec::new();
    if config.gpu_layers.is_some() {
        explicit_args.push("gpu_layers".to_string());
    }
    if config.no_gpu {
        explicit_args.push("no_gpu".to_string());
    }
    if config.context_length != types::DEFAULT_CONTEXT_LENGTH {
        explicit_args.push("context_length".to_string());
    }
    if config.batch {
        explicit_args.push("batch".to_string());
    }
    if config.no_fp8_cache {
        explicit_args.push("no_fp8_cache".to_string());
    }
    if config.backend.is_some() {
        explicit_args.push("backend".to_string());
    }
    realizar::api::OffloadReport {
        gpu_layers_requested: config
            .gpu_layers
            .map_or_else(|| "none".to_string(), |r| r.to_string()),
        gpu_layers_resolved: resolved_layers,
        gpu_layers_total: total_layers,
        offload_policy: "all_or_nothing",
        autofit_applied: Vec::new(),
        explicit_args,
        build_features: cli_build_features(),
        build_commit: Some(env!("APR_GIT_SHA").to_string()),
    }
}

/// PERF-021 countermeasure 3: a request has a RESOLUTION, and it is reported.
///
/// `requested` is what the user asked for, `resolved` what the server will do,
/// `total` how many layers exist. A boolean could express none of this, which
/// is finding N4 and the reason the defect was invisible.
///
/// I-17, EXPLICIT WINS: `Exact(n)` and `All` are user instructions. When they
/// cannot be honoured this returns an error rather than quietly resolving lower
/// — automation overriding an explicit instruction is the v2.2 root cause of
/// defect #1. Only `Auto` may be reduced, because `auto` is the value that
/// asked to be.
pub(crate) fn resolve_gpu_layers(
    request: GpuLayerRequest,
    total_layers: u32,
    fits: u32,
) -> Result<u32> {
    match request {
        GpuLayerRequest::None => Ok(0),
        GpuLayerRequest::Auto => Ok(fits.min(total_layers)),
        GpuLayerRequest::All => {
            if fits >= total_layers {
                Ok(total_layers)
            } else {
                Err(CliError::InvalidInput(format!(
                    "--gpu-layers all asked for {total_layers} layers and only {fits} fit. \
                     Auto-fit will not silently lower an explicit request; pass \
                     --gpu-layers auto to offload what fits, or --gpu-layers {fits}."
                )))
            }
        }
        GpuLayerRequest::Exact(n) => {
            if n <= fits {
                Ok(n.min(total_layers))
            } else {
                Err(CliError::InvalidInput(format!(
                    "--gpu-layers {n} asked for more than the {fits} that fit. Auto-fit \
                     will not silently lower an explicit request; pass --gpu-layers auto \
                     to offload what fits."
                )))
            }
        }
    }
}

/// `--gpu` on a build with no accelerated backend FAILS, naming a remedy that
/// works.
///
/// Until #2696 this flag was accepted and silently ignored. `use_gpu` is only
/// read inside `#[cfg(feature = "cuda")]` blocks (handlers.rs:776 and :1084),
/// so on a build without the feature those blocks vanish, `config.gpu` is never
/// consulted, and the server starts on CPU having warned nobody.
///
/// `cargo install aprender` produces exactly that build — root `Cargo.toml` has
/// `default = ["cli"]` and `cuda` is opt-in. Measured on 2026-08-24 with an idle
/// RTX 4090 in the machine, it decoded at a fraction of llama.cpp's rate with
/// seconds to first token: no diagnostic, and a plausible-looking number at the
/// end of it. The figures and their basis are in #2696, which is where a reader
/// should get them.
///
/// The remedy in the message is checked to be real. #2527 is the counter-case:
/// `aprender-test-cli` printed "rebuild with --features llm" for a feature its
/// Cargo.toml never declared, so the instruction could not be followed. Both
/// `cuda` and `wgpu` are declared at the facade, so both spellings below work.
///
/// `--backend wgpu` is covered by the same check: naming a backend the build
/// cannot reach is the same defect wearing a different flag.
#[allow(clippy::unnecessary_wraps)] // wraps only when no accelerator is compiled in
/// The accelerator request a `ServerConfig` makes, with the flag text to quote
/// back — `None` when nothing asked for an accelerator. Precedence is this
/// command's, unchanged: `--no-gpu` beats `--gpu`; `--backend cpu` asks for
/// nothing; `--gpu-layers 0` asks for nothing.
fn accelerator_request(config: &ServerConfig) -> Option<(crate::registry::Request<'_>, String)> {
    let wants_backend = config.backend.as_deref();
    // PERF-021: `--gpu-layers` is the request; `--gpu` is its deprecated
    // boolean spelling and means `all`. `--gpu-layers 0` is an explicit CPU
    // request and asks for no accelerator, which is why it is not simply
    // "is Some".
    let wants_layers = config
        .gpu_layers
        .is_some_and(GpuLayerRequest::wants_accelerator);
    let gpu = config.gpu && !config.no_gpu;
    let wants_gpu = wants_layers || gpu || matches!(wants_backend, Some("wgpu" | "cuda" | "gpu"));
    if !wants_gpu {
        return None;
    }
    // Quote back the flag the USER typed. `--gpu` sets gpu_layers to All on the
    // way in, so checking gpu_layers first would tell a user who typed `--gpu`
    // about a flag they did not use.
    let asked = if config.gpu {
        "--gpu".to_string()
    } else if config.gpu_layers.is_some() {
        "--gpu-layers".to_string()
    } else if let Some(b) = wants_backend.filter(|b| *b != "cpu") {
        format!("--backend {b}")
    } else {
        "--gpu".to_string()
    };
    let req = crate::registry::Request {
        gpu,
        no_gpu: false,
        backend: wants_backend.filter(|b| *b != "cpu"),
        layers_want_accelerator: wants_layers,
    };
    Some((req, asked))
}

/// R-0b (#3002): the request resolves against the backend registry. A forced
/// backend never downgrades: not compiled ⇒ `FeatureDisabled` (9), compiled but
/// not Ready on this host ⇒ `BackendUnavailable` (14).
fn ensure_accelerator_available(
    config: &ServerConfig,
) -> Result<Option<crate::registry::Resolved>> {
    match accelerator_request(config) {
        // R-0b "selected: always": the registry's default is announced with its
        // reason even when nothing asked for an accelerator.
        None => {
            // `--no-gpu`, `--backend cpu` or `--gpu-layers 0` mean cpu; else the
            // registry's default. Nothing here can refuse.
            let cpu = config.no_gpu || config.gpu_layers.is_some();
            let asked = if config.no_gpu {
                "--no-gpu"
            } else if cpu {
                "--gpu-layers 0"
            } else {
                "default"
            };
            let req = crate::registry::Request {
                gpu: false,
                no_gpu: cpu,
                backend: config.backend.as_deref(),
                layers_want_accelerator: false,
            };
            Ok(crate::registry::announce(&req, asked).ok())
        }
        Some((req, asked)) => crate::registry::announce(&req, &asked).map(Some),
    }
}

/// The same gate over an explicit registry — tests hand it fixtures instead of
/// the live host, so "a build with no accelerator" is a fixture, not a `cfg`.
#[cfg(feature = "inference")]
pub(crate) fn ensure_accelerator_available_in(
    config: &ServerConfig,
    reg: &trueno::registry::BackendRegistry,
) -> Result<Option<crate::registry::Resolved>> {
    match accelerator_request(config) {
        None => Ok(None),
        Some((req, asked)) => crate::registry::resolve_in(&req, &asked, reg).map(Some),
    }
}

/// Serve command entry point (blocking)
#[provable_contracts_macros::contract("apr-cli-operations-v1", equation = "long_running_graceful")]
pub(crate) fn run(model_path: &Path, config: &ServerConfig) -> Result<()> {
    // Record which file we are serving so the metadata endpoints can MEASURE
    // it instead of reporting constants. Everything downstream takes
    // `&ServerConfig`, so stamping it once here reaches every serve path.
    let config = &ServerConfig {
        model_path: Some(model_path.to_path_buf()),
        ..config.clone()
    };
    contract_pre_graceful_shutdown!();
    contract_pre_resource_cleanup!();
    contract_pre_concurrent_isolation!();
    contract_pre_request_routing!();
    contract_pre_cors_negotiation!();
    contract_pre_concurrent_model_access!();
    contract_pre_server_lifecycle!();

    // `--gpu` must not be accepted by a build that has no GPU to dispatch to.
    let resolved = ensure_accelerator_available(config)?;
    // R-0b / REG-12: the startup resolution reaches GET /v1/effective-config.
    #[cfg(feature = "inference")]
    if let Some(r) = resolved {
        let _ = realizar::api::effective_config::set_backend_resolution(
            realizar::api::effective_config::BackendResolution {
                kind: r.kind.to_string(),
                device_index: r.device_index,
                device_uid: r.device_uid,
                device_name: r.device_name,
                reason: r.reason,
                discovered_at_unix: r.discovered_at_unix,
                basis: "apr-cli backend registry at startup (R-0b, #3002)".to_string(),
                matches_loaded: None,
            },
        );
    }
    #[cfg(not(feature = "inference"))]
    let _ = resolved;

    // PMAT-297: Configure rayon thread pool to physical core count.
    // Default (all threads incl. HT) causes 44% regression from contention.
    #[cfg(feature = "inference")]
    if let Err(e) = realizar::inference::configure_optimal_thread_pool() {
        eprintln!("[PMAT-297] Thread pool config: {e} (may already be initialized)");
    }

    // GH-286: Set env vars for realizr's KV cache and FP8 control
    std::env::set_var("REALIZR_CONTEXT_LENGTH", config.context_length.to_string());
    if config.no_fp8_cache {
        std::env::set_var("REALIZR_NO_FP8_CACHE", "1");
    }

    println!("{}", "=== APR Serve ===".cyan().bold());
    println!();
    println!("Model: {}", model_path.display());
    println!("Binding: {}", config.bind_addr());
    if config.context_length != 4096 {
        println!(
            "Context length: {} (--context-length)",
            config.context_length
        );
    }
    if config.no_fp8_cache {
        println!("FP8 cache: DISABLED (--no-fp8-cache, saves ~1.5 GB)");
    }
    println!();

    // Validate model
    if !model_path.exists() {
        return Err(CliError::FileNotFound(model_path.to_path_buf()));
    }

    let state = ServerState::new(model_path.to_path_buf(), config.clone())?;

    println!(
        "{}",
        format!(
            "Model loading: {}",
            if state.uses_mmap { "mmap" } else { "full" }
        )
        .dimmed()
    );

    // aprender#2376(8): no endpoint list here. This point in the program is BEFORE
    // the magic bytes are read, so the format is unknown, the router does not exist
    // and nothing that could be printed would be a measurement. The list printed
    // here claimed "POST /v1/predict - Model prediction (APR)" on every path — it
    // answers 503 even when the served file IS a .apr — and "POST /generate -
    // Text generation (GGUF)", which 404s on the APR server. The real list is
    // printed by the server that mounted it, after bind, from its own route table.

    // GH-153: "Server ready" message now printed AFTER TcpListener::bind succeeds
    // in start_*_server functions, not here (was misleading since bind happens later)
    println!();
    println!("{}", "Press Ctrl+C to stop".dimmed());

    // Try to start real server with realizar
    #[cfg(feature = "inference")]
    let result = { handlers::start_realizar_server(model_path, config) };

    // Fallback: stub mode
    #[cfg(not(feature = "inference"))]
    let result = {
        println!();
        println!("{}", "[Server requires --features inference]".yellow());
        Ok(())
    };

    contract_post_graceful_shutdown!(&());
    contract_post_resource_cleanup!(&());
    contract_post_concurrent_isolation!(&());
    contract_post_request_routing!(&());
    contract_post_cors_negotiation!(&());
    contract_post_concurrent_model_access!(&());
    contract_post_server_lifecycle!(&());
    result
}

#[cfg(test)]
mod accelerator_guard_tests {
    use super::*;

    fn cfg_with(gpu: bool, no_gpu: bool, backend: Option<&str>) -> ServerConfig {
        ServerConfig {
            gpu,
            no_gpu,
            backend: backend.map(str::to_string),
            ..ServerConfig::default()
        }
    }

    /// An R-0a registry fixture (`crates/apr-cli/tests/fixtures/registry/`).
    #[cfg(feature = "inference")]
    pub(super) fn fixture(name: &str) -> trueno::registry::BackendRegistry {
        let path = format!(
            "{}/tests/fixtures/registry/{name}.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        trueno::registry::BackendRegistry::from_fixture_json(&text, &path).expect("fixture parses")
    }

    /// The defect itself (#2696): on a build with no accelerator compiled in,
    /// `--gpu` must not be waved through — before #2696 this returned Ok and the
    /// server ran on CPU. R-0b: "compiled" is the registry's `not-compiled`
    /// lines, so the case is a fixture, never a `cfg`.
    #[test]
    #[cfg(feature = "inference")]
    fn gpu_without_a_backend_is_an_error_not_a_silent_cpu_run() {
        let reg = fixture("no-accelerator-compiled");
        let err = ensure_accelerator_available_in(&cfg_with(true, false, None), &reg)
            .expect_err("--gpu on a build with no accelerator compiled must fail");
        let msg = err.to_string();
        assert!(
            matches!(err, crate::error::CliError::FeatureDisabled(_)),
            "not compiled ⇒ FeatureDisabled (exit 9): {msg}"
        );
        assert!(
            msg.contains("cargo install aprender --features cuda"),
            "the error must name a remedy that works: {msg}"
        );
        assert!(
            msg.contains("--no-gpu"),
            "and the deliberate-CPU escape hatch: {msg}"
        );
    }

    /// cpu-only: cuda's driver is missing and wgpu sees no device — compiled,
    /// not Ready. That is `BackendUnavailable` (exit 14), never a cpu run, and
    /// the message quotes the flag the user typed.
    #[test]
    #[cfg(feature = "inference")]
    fn an_unreachable_backend_is_also_an_error() {
        let reg = fixture("cpu-only");
        for backend in ["wgpu", "cuda", "gpu"] {
            let err = ensure_accelerator_available_in(&cfg_with(false, false, Some(backend)), &reg)
                .unwrap_err();
            assert!(
                err.to_string().contains(&format!("--backend {backend}")),
                "the message must quote the flag the user typed, not a generic one: {err}"
            );
            assert!(
                matches!(err, crate::error::CliError::BackendUnavailable(_)),
                "compiled but not Ready ⇒ BackendUnavailable, never cpu: {err}"
            );
        }
    }

    /// A build WITH the feature must not be blocked by its own guard.
    #[test]
    #[cfg(any(feature = "cuda", feature = "wgpu"))]
    fn a_gpu_build_is_not_blocked() {
        ensure_accelerator_available(&cfg_with(true, false, None))
            .expect("a build with an accelerator must pass its own guard");
    }

    /// THE CALL SITE IS THE GATE, NOT THE FUNCTION.
    ///
    /// Registry mutation for this row is "remove `ensure_accelerator_available`",
    /// and every test above would survive it — they call the function directly,
    /// so deleting the CALL in `run()` leaves them green while `--gpu` goes back
    /// to being silently ignored. A test that cannot see the defect return is
    /// not the gate; it is a unit test of a function nobody invokes.
    ///
    /// This reads the source of `run()` and asserts the call is present. Crude,
    /// and it is the only thing here that fails when the wiring is removed —
    /// `serve::run` needs a model file and a bound port, so it cannot be driven
    /// from a unit test.
    /// PERF-021 / I-2: NO DECISION SITE MAY READ THE BOOLEAN.
    ///
    /// The sibling gate above proves `--gpu` still REACHES the guard. This one
    /// proves the quantity reaches the DECISION, which is the half that was
    /// missing and the half #2696 actually turns on.
    ///
    /// Four sites chose GPU over CPU by reading `config.gpu`: handlers.rs:776,
    /// handlers.rs:1084, handler_gpu_completion.rs:407 and :412. Meanwhile
    /// `--gpu-layers` set only `config.gpu_layers`. On a `--features cuda`
    /// build `--gpu-layers all` therefore passed the guard and served on CPU —
    /// #2696 in the new spelling, shipped by the change that retired the old
    /// one.
    ///
    /// Every unit test in this file survives that defect, because they all call
    /// helpers directly. So does the branch's own
    /// `gpu_layers_is_refused_on_a_build_with_no_accelerator`, which carries
    /// `#[cfg(not(any(feature = "cuda", feature = "wgpu")))]` and therefore
    /// COMPILES ONLY WHERE THE BUG CANNOT HAPPEN. This test has no cfg: it is a
    /// source scan, so it runs on every build including the CUDA one.
    #[test]
    fn no_decision_site_reads_the_bare_accelerator_boolean() {
        // (file, source) pairs for every module that chooses GPU vs CPU.
        let sites: [(&str, &str); 2] = [
            ("handlers.rs", include_str!("handlers.rs")),
            (
                "handler_gpu_completion.rs",
                include_str!("handler_gpu_completion.rs"),
            ),
        ];
        let mut offenders = Vec::new();
        for (name, src) in sites {
            for (i, line) in src.lines().enumerate() {
                let t = line.trim_start();
                if t.starts_with("//") || t.starts_with("///") {
                    continue;
                }
                // The decision shapes that reintroduce the defect.
                if line.contains("config.gpu &&")
                    || line.contains("config.gpu ||")
                    || line.contains("= config.gpu;")
                    || line.contains("if config.gpu {")
                {
                    offenders.push(format!("{name}:{}: {}", i + 1, t));
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "a GPU/CPU decision reads the boolean `config.gpu` instead of \
             `config.wants_accelerator()`. `--gpu-layers all` then parses, \
             validates, stores — and serves on CPU (#2696, new spelling). \
             Offenders:\n  {}",
            offenders.join("\n  ")
        );
    }

    /// The positive half of I-2: the quantity alone must select the accelerator,
    /// with the deprecated boolean unset. Build-independent by construction, so
    /// unlike the cfg'd test below it runs on the CUDA build too.
    #[test]
    fn the_quantity_alone_selects_the_accelerator() {
        let mut cfg = ServerConfig::default();
        cfg.gpu = false; // the user did NOT type the deprecated boolean
        cfg.gpu_layers = Some(GpuLayerRequest::All);
        assert!(
            cfg.wants_accelerator(),
            "`--gpu-layers all` with no `--gpu` must select the accelerator; \
             reading the boolean here is #2696"
        );

        cfg.gpu_layers = Some(GpuLayerRequest::None);
        assert!(
            !cfg.wants_accelerator(),
            "`--gpu-layers 0` is an explicit CPU request and must NOT select it"
        );

        cfg.gpu_layers = Some(GpuLayerRequest::All);
        cfg.no_gpu = true;
        assert!(
            !cfg.wants_accelerator(),
            "`--no-gpu` must still win over a quantity"
        );
    }

    /// PERF-021: the resolver must be CALLED, not merely defined.
    ///
    /// Every one of the 7 calls to `resolve_gpu_layers` lived inside
    /// `#[cfg(test)]` while its own doc comment said "a request has a
    /// RESOLUTION, and it is reported". It reported nothing. Its sibling
    /// `ensure_accelerator_available` landed WIRED because it had a source-grep
    /// gate; this one landed dead because it had none. That difference is the
    /// entire explanation, so the gate comes with the call.
    /// #2762 source gate. The behaviour test for `resolve_serve_max_seq_len`
    /// lives beside the function; this one proves the function is REACHED from
    /// the serve path, and that the path reads the variable `--context-length`
    /// actually writes.
    ///
    /// It has no cfg on purpose. The defect is that a CUDA-only code path read
    /// a name nothing sets, so a test compiled only under `--features cuda`
    /// would be the same blind spot one level up.
    #[test]
    fn the_gguf_cuda_serve_path_reads_the_context_length_flag() {
        // SHIPPING CODE ONLY, and it took two tries to get that right.
        //
        // v1 scanned the whole file, so the doc comment on
        // `resolve_serve_max_seq_len` -- which names the variable -- satisfied it.
        // v2 stripped comments, and the ASSERTION MESSAGE inside that file's own
        // `#[cfg(test)]` module still named it: the mutation that passes `None`
        // for the context argument was applied and this gate STAYED GREEN.
        // A source gate its own text satisfies is theater, in both spellings.
        //
        // So: cut at the first `#[cfg(test)]`, drop comments, and look for the
        // ARGUMENT rather than the name -- assembled at runtime so this line
        // cannot be its own evidence.
        let whole = include_str!("handler_gpu_completion.rs");
        let shipping = whole.split("#[cfg(test)]").next().unwrap_or(whole);
        let src: String = shipping
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        let reads_the_flag = format!(
            "std::env::var({:?}).ok().as_deref()",
            "REALIZR_CONTEXT".to_string() + "_LENGTH"
        );
        assert!(
            src.contains(&reads_the_flag),
            "the GGUF+CUDA serve path does not pass REALIZR_CONTEXT_LENGTH to the \
             context resolver, and that variable is the only thing \
             `--context-length` writes (serve/mod.rs). Before #2762 it sized the \
             KV cache from REALIZR_MAX_SEQ_LEN -- a name READ in one place and \
             WRITTEN in none -- so every server got 2048 and the batched KV \
             stride was a constant the operator could not move"
        );
        assert!(
            src.contains("resolve_serve_max_seq_len("),
            "the context-length resolver is not called from the serve path"
        );
    }

    /// `REALIZR_MAX_SEQ_LEN` is a GH-129 escape hatch. If a later change makes
    /// something in the tree SET it, the precedence in
    /// `resolve_serve_max_seq_len` stops being an operator override and becomes
    /// a second, hidden default that outranks `--context-length` again.
    #[test]
    fn the_seq_len_escape_hatch_is_still_read_only() {
        let src = include_str!("mod.rs");
        assert!(
            !src.contains("set_var(\"REALIZR_MAX_SEQ_LEN\""),
            "REALIZR_MAX_SEQ_LEN is now written by the CLI; it takes precedence \
             over --context-length, so writing it re-creates #2762"
        );
    }

    #[test]
    fn the_resolver_is_actually_called_from_a_decision_path() {
        let src = include_str!("handler_gpu_completion.rs");
        assert!(
            src.contains("config.resolve_layers(total_layers)?"),
            "no decision path calls the resolver — `--gpu-layers` is parsed, \
             validated, stored and never resolved, so nothing reports how many \
             layers were placed (N4, and the reason #2696 was invisible)"
        );
        assert!(
            src.contains("gpu-layers: requested="),
            "the resolution is computed but not REPORTED. I-2 requires \
             resolved-vs-requested to be observable; a resolution nobody can \
             see is the boolean defect with more arithmetic"
        );
    }

    /// A PARTIAL request must be refused, not rounded.
    ///
    /// `OwnedQuantizedModelCuda` takes no layer count and uploads every layer,
    /// so accepting `--gpu-layers 12` on a 29-layer model would place 29 and
    /// print 12 — a fabricated number in a log, which is worse than a refusal
    /// and is precisely what this epic exists to remove.
    #[test]
    fn a_partial_offload_is_refused_because_the_loader_cannot_do_it() {
        let mut cfg = ServerConfig::default();
        cfg.gpu_layers = Some(GpuLayerRequest::Exact(12));
        let e = cfg.resolve_layers(29).expect_err("partial must be refused");
        let m = e.to_string();
        assert!(m.contains("PARTIAL"), "must name the limitation: {m}");
        assert!(m.contains("PERF-023"), "must cite the tracking item: {m}");

        // The two honourable requests still work, or the refusal is a wall.
        cfg.gpu_layers = Some(GpuLayerRequest::All);
        assert_eq!(cfg.resolve_layers(29).expect("all"), 29);
        cfg.gpu_layers = Some(GpuLayerRequest::None);
        assert_eq!(cfg.resolve_layers(29).expect("none"), 0);
        // `Exact(total)` is `all` by another spelling and must be accepted.
        cfg.gpu_layers = Some(GpuLayerRequest::Exact(29));
        assert_eq!(cfg.resolve_layers(29).expect("exact==total"), 29);
    }

    #[test]
    fn the_guard_is_actually_wired_into_run() {
        let src = include_str!("mod.rs");
        let run_start = src
            .find("pub(crate) fn run(model_path: &Path, config: &ServerConfig)")
            .expect("serve::run must exist");
        let run_body = &src[run_start..];
        let end = run_body.find("\n}\n").unwrap_or(run_body.len());
        assert!(
            run_body[..end].contains("ensure_accelerator_available(config)?"),
            "serve::run no longer calls ensure_accelerator_available — `--gpu` on a \
             build with no GPU backend is silently ignored again (#2696). The unit \
             tests below pass without it, which is exactly why this test exists."
        );
    }

    /// Every {gpu, no_gpu, backend} the server accepts.
    #[cfg(feature = "inference")]
    fn every_request() -> Vec<(bool, bool, Option<&'static str>)> {
        let mut v = Vec::new();
        for gpu in [false, true] {
            for no_gpu in [false, true] {
                for backend in [None, Some("cpu"), Some("cuda"), Some("wgpu"), Some("gpu")] {
                    v.push((gpu, no_gpu, backend));
                }
            }
        }
        v
    }

    /// Whether `reg` has a Ready accelerator (of kind `k`, when named).
    #[cfg(feature = "inference")]
    fn ready(reg: &trueno::registry::BackendRegistry, k: Option<&str>) -> bool {
        use trueno::registry::{BackendKind, Status};
        reg.entries.iter().any(|e| {
            e.kind != BackendKind::Cpu
                && matches!(e.status, Status::Ready)
                && k.is_none_or(|k| e.kind.as_str() == k)
        })
    }

    /// Err exactly when a request is made that this HOST cannot honour (R-0b):
    /// over four fixtures — nothing compiled, compiled-but-absent, one Ready
    /// cuda, two vendors — and every request the server accepts.
    #[test]
    #[cfg(feature = "inference")]
    fn the_refusal_is_total_over_every_input() {
        for name in [
            "no-accelerator-compiled",
            "cpu-only",
            "one-cuda",
            "two-vendors",
        ] {
            let reg = fixture(name);
            for (gpu, no_gpu, backend) in every_request() {
                let requested =
                    (gpu && !no_gpu) || matches!(backend, Some("cuda" | "wgpu" | "gpu"));
                let honourable = match backend {
                    Some(k @ ("cuda" | "wgpu")) => ready(&reg, Some(k)),
                    _ => ready(&reg, None),
                };
                assert_eq!(
                    ensure_accelerator_available_in(&cfg_with(gpu, no_gpu, backend), &reg).is_err(),
                    requested && !honourable,
                    "{name}: gpu={gpu} no_gpu={no_gpu} backend={backend:?}: \
                     Err exactly when a request is made that this host cannot honour"
                );
            }
        }
    }

    /// Silence stays silent: nothing about the default or explicit-CPU paths
    /// changes, on any build.
    #[test]
    fn cpu_paths_are_untouched() {
        ensure_accelerator_available(&cfg_with(false, false, None)).expect("default");
        ensure_accelerator_available(&cfg_with(false, true, None)).expect("--no-gpu");
        ensure_accelerator_available(&cfg_with(true, true, None))
            .expect("--no-gpu overrides --gpu, as it always did");
        ensure_accelerator_available(&cfg_with(false, false, Some("cpu"))).expect("--backend cpu");
    }
}

#[cfg(test)]
mod gpu_layers_contract_tests {
    //! PERF-021. The v2.2 root cause of defect #1 is not "we defaulted to CPU";
    //! it is that AUTOMATION OVERRODE AN EXPLICIT USER INSTRUCTION AND THE
    //! OVERRIDE WAS UNOBSERVABLE. These test both halves.

    #[cfg(feature = "inference")]
    use super::accelerator_guard_tests::fixture;
    use super::*;

    #[test]
    fn a_request_parses_as_a_quantity_not_a_flag() {
        assert_eq!(GpuLayerRequest::parse("0"), Ok(GpuLayerRequest::None));
        assert_eq!(GpuLayerRequest::parse("auto"), Ok(GpuLayerRequest::Auto));
        assert_eq!(GpuLayerRequest::parse("all"), Ok(GpuLayerRequest::All));
        assert_eq!(GpuLayerRequest::parse("28"), Ok(GpuLayerRequest::Exact(28)));
        assert_eq!(GpuLayerRequest::parse("AUTO"), Ok(GpuLayerRequest::Auto));
    }

    /// A mistyped accelerator request must not become CPU by default. That is
    /// the silent-degradation shape in miniature.
    #[test]
    fn a_mistyped_request_is_rejected_not_defaulted() {
        let err = GpuLayerRequest::parse("gpu").expect_err("must reject");
        assert!(
            err.contains("auto"),
            "the error lists the legal values: {err}"
        );
        assert!(GpuLayerRequest::parse("").is_err());
        assert!(GpuLayerRequest::parse("-1").is_err());
    }

    /// I-17. `auto` is the ONLY value auto-fit may modify, because it is the
    /// one that asked to be fitted.
    #[test]
    fn only_auto_may_be_autofitted() {
        assert!(GpuLayerRequest::Auto.may_autofit());
        assert!(!GpuLayerRequest::All.may_autofit());
        assert!(!GpuLayerRequest::Exact(12).may_autofit());
        assert!(!GpuLayerRequest::None.may_autofit());
    }

    /// EXPLICIT WINS. An instruction that cannot be honoured is an ERROR, not a
    /// quiet reduction — quiet reduction is exactly how #2696 stayed invisible.
    #[test]
    fn an_explicit_request_that_does_not_fit_is_an_error() {
        let e = resolve_gpu_layers(GpuLayerRequest::All, 29, 12).expect_err("all must not shrink");
        assert!(e.to_string().contains("12"), "names what did fit: {e}");
        assert!(e.to_string().contains("auto"), "names the remedy: {e}");

        let e =
            resolve_gpu_layers(GpuLayerRequest::Exact(28), 29, 12).expect_err("N must not shrink");
        assert!(e.to_string().contains("auto"), "names the remedy: {e}");
    }

    /// And auto DOES fit, silently, because that is what it means.
    #[test]
    fn auto_resolves_to_what_fits() {
        assert_eq!(
            resolve_gpu_layers(GpuLayerRequest::Auto, 29, 12).expect("auto"),
            12
        );
        assert_eq!(
            resolve_gpu_layers(GpuLayerRequest::Auto, 29, 99).expect("auto"),
            29
        );
        assert_eq!(
            resolve_gpu_layers(GpuLayerRequest::None, 29, 12).expect("none"),
            0
        );
        assert_eq!(
            resolve_gpu_layers(GpuLayerRequest::All, 29, 29).expect("all fits"),
            29
        );
        assert_eq!(
            resolve_gpu_layers(GpuLayerRequest::Exact(8), 29, 12).expect("8 fits"),
            8
        );
    }

    /// THE FLAG MUST REACH THE CONFIG, which the tests below cannot see.
    ///
    /// I shipped this ticket once with `--gpu-layers all` parsed by clap,
    /// destructured in the dispatch arm, and never written into `ServerConfig`.
    /// Every test in this module passed and the real binary started a CPU
    /// server anyway — the identical failure PERF-003 already taught, one layer
    /// out. `gpu_layers` is the field; if the dispatch stops populating it this
    /// asserts on the source, because a unit test over a struct literal cannot.
    #[test]
    fn the_cli_flag_is_actually_wired_into_the_config() {
        let src = include_str!("../../dispatch_run.rs");
        assert!(
            src.contains("gpu_layers: match gpu_layers.as_deref()"),
            "dispatch_serve no longer populates ServerConfig.gpu_layers — \
             --gpu-layers parses and is then dropped, so the request is silently \
             ignored exactly as #2696's --gpu was"
        );
        assert!(
            src.contains("None if gpu && !no_gpu => Some(serve::GpuLayerRequest::All)"),
            "the deprecated --gpu no longer maps to --gpu-layers all, so the old \
             spelling stops reaching the new gate"
        );
    }

    #[test]
    #[cfg(feature = "inference")]
    fn gpu_layers_is_refused_on_a_build_with_no_accelerator() {
        let reg = fixture("no-accelerator-compiled");
        let mut cfg = ServerConfig::default();
        cfg.gpu_layers = Some(GpuLayerRequest::All);
        let err = ensure_accelerator_available_in(&cfg, &reg).expect_err("must refuse");
        assert!(
            err.to_string().contains("--gpu-layers"),
            "quotes what was asked: {err}"
        );
        cfg.gpu_layers = Some(GpuLayerRequest::None);
        ensure_accelerator_available_in(&cfg, &reg)
            .expect("--gpu-layers 0 asks for no accelerator");
    }
}
