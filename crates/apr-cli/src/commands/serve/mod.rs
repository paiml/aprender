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
    if cfg!(feature = "cuda") {
        println!("  cuda    compiled in");
        any = true;
    }
    if cfg!(feature = "wgpu") {
        println!("  wgpu    compiled in");
        any = true;
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
/// `default = ["cli"]` and `cuda` is opt-in. Measured on 2026-08-24 with an
/// idle RTX 4090 in the machine: 15.7 tok/s decode against llama.cpp's 158.9,
/// and 7.5 SECONDS to first token. A tenth of the speed, no diagnostic, and a
/// plausible-looking number at the end of it.
///
/// The remedy in the message is checked to be real. #2527 is the counter-case:
/// `aprender-test-cli` printed "rebuild with --features llm" for a feature its
/// Cargo.toml never declared, so the instruction could not be followed. Both
/// `cuda` and `wgpu` are declared at the facade, so both spellings below work.
///
/// `--backend wgpu` is covered by the same check: naming a backend the build
/// cannot reach is the same defect wearing a different flag.
#[allow(clippy::unnecessary_wraps)] // wraps only when no accelerator is compiled in
fn ensure_accelerator_available(config: &ServerConfig) -> Result<()> {
    let wants_backend = config.backend.as_deref();
    // PERF-021: `--gpu-layers` is the request; `--gpu` is its deprecated
    // boolean spelling and means `all`. `--gpu-layers 0` is an explicit CPU
    // request and asks for no accelerator, which is why it is not simply
    // "is Some".
    let wants_layers = config
        .gpu_layers
        .is_some_and(GpuLayerRequest::wants_accelerator);
    let wants_gpu = wants_layers
        || (config.gpu && !config.no_gpu)
        || matches!(wants_backend, Some("wgpu" | "cuda" | "gpu"));
    if !wants_gpu {
        return Ok(());
    }
    if cfg!(any(feature = "cuda", feature = "wgpu")) {
        return Ok(());
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
    Err(CliError::FeatureDisabled(format!(
        "{asked} was requested, but this build has no GPU backend compiled in, \n\
         so the server would have run on CPU without telling you. On a 7B Q4_K_M \n\
         model that is roughly a tenth of the decode rate and several seconds of \n\
         extra latency to the first token (aprender#2696).\n\
         \n\
         Install a build that has one:\n\
         \n\
        \x20    cargo install aprender --features cuda    # NVIDIA\n\
        \x20    cargo install aprender --features wgpu    # portable GPU backend\n\
         \n\
         Or pass --no-gpu to run on CPU deliberately."
    )))
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
    ensure_accelerator_available(config)?;

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
    // PERF-006 (aprender#2706) — the andon lamp, on the startup banner.
    //
    // ONE renderer, shared with `GET /health` and with the provenance block of
    // `apr bench --json`, so a reader comparing this terminal against a JSON
    // body is comparing one source against itself. Deleting the call below, or
    // rendering either fact locally, is caught by `andon_surface_wiring_tests`.
    //
    // The bound reads 1 here because nothing has recorded a scheduler yet; a
    // CUDA build prints the line again from `handler_gpu_completion.rs` once
    // the batch scheduler is wired, and the difference between the two lines
    // IS the andon signal. On every build that never wires one — which is
    // every `cargo install aprender` — 1 is the final answer and the reason
    // this line exists (defect #2: apr does not batch; wall time linear in
    // concurrency, `contracts/batch-admission-v1.yaml`).
    //
    // NOTE: prose in this region is deliberately not written as a callable
    // path. The first cut of the guard matched its own explanatory comment
    // and stayed GREEN on its own named mutation.
    #[cfg(feature = "inference")]
    println!("{}", realizar::andon::andon_line());
    #[cfg(not(feature = "inference"))]
    println!("Compute: no inference engine linked (built without --features inference)");
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

/// PERF-006 (aprender#2706) — THE CALL SITE IS THE GATE, NOT THE FUNCTION.
///
/// `realizar::andon` has its own unit tests, and every one of them would
/// survive a surface going back to computing its own compute class. That is
/// the failure this module refuses: three surfaces, three derivations, three
/// answers — which is precisely the state APR-PERF-GATE-001 v2.2 §4 recorded
/// as **pending** on the Andon row before this ticket.
///
/// So this reads the source of the three renderers and asserts each one
/// delegates. Crude, and it is the only thing here that fails when a surface
/// re-derives: `apr serve` needs a model and a bound port, `/health` needs a
/// live `AppState`, and a build with a different feature set cannot be
/// observed from this one, so none of the three can be driven from a unit test
/// in this crate.
#[cfg(test)]
mod andon_surface_wiring_tests {
    use std::path::PathBuf;

    /// The entry points that route through the ONE `compute_class()`.
    ///
    /// `andon_line()` counts because `aprender-serve`'s
    /// `the_banner_line_renders_the_same_class_and_bound_as_the_accessors`
    /// proves it renders the accessors' own answers — so a surface printing
    /// the line cannot be showing a different class from a surface reading the
    /// accessor.
    const CLASS_ENTRY_POINTS: [&str; 2] = ["andon::compute_class()", "andon::andon_line()"];
    const BOUND_ENTRY_POINTS: [&str; 2] = ["andon::max_in_flight()", "andon::andon_line()"];

    /// Does this function body get its compute class from the shared andon?
    ///
    /// A pure predicate over text, so it can be pointed at a body that is
    /// known-good and a body that is known-bad. Only that pair tells a gate
    /// that discriminates from one that flags everything or nothing.
    fn renders_shared_compute_class(body: &str) -> bool {
        CLASS_ENTRY_POINTS.iter().any(|e| body.contains(e))
    }

    /// Same question for `max_in_flight`.
    fn renders_shared_max_in_flight(body: &str) -> bool {
        BOUND_ENTRY_POINTS.iter().any(|e| body.contains(e))
    }

    /// Lexer state for [`strip_comments`].
    ///
    /// A flat state machine rather than nested `if`s: the transition table is
    /// one `match` with no nesting, which keeps it readable and keeps the
    /// cognitive-complexity gate satisfied.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Scan {
        /// Ordinary code.
        Code,
        /// Saw `/` in code; the next char decides comment vs division.
        Slash,
        /// Inside a `"..."` literal.
        Str,
        /// Saw `\` inside a string literal.
        StrEsc,
        /// Inside a `//` comment.
        Line,
        /// Inside a `/* */` comment.
        Block,
        /// Saw `*` inside a block comment.
        BlockStar,
    }

    /// Drop `//` and `/* */` comments, keeping string literals intact.
    ///
    /// NOT a nicety. The first cut of this module scanned raw source, and the
    /// banner's own explanatory comment named the call it was explaining — so
    /// deleting the real call left the comment behind and the guard passed its
    /// own named mutation (rc=0, all six green). Prose must not be able to
    /// satisfy a wiring check. Rewording the one comment would have fixed that
    /// instance; removing comments from the scanned text fixes the class.
    fn strip_comments(src: &str) -> String {
        let mut out = String::with_capacity(src.len());
        let mut state = Scan::Code;
        let mut prev = '\0';
        for ch in src.chars() {
            state = match (state, ch) {
                // A char literal holding a double quote (`'"'`) must not open a
                // string — that would disable stripping for the rest of the
                // region, and this guard has to fail CLOSED.
                (Scan::Code, '"') if prev == '\'' => {
                    out.push(ch);
                    Scan::Code
                }
                (Scan::Code, '"') => {
                    out.push(ch);
                    Scan::Str
                }
                (Scan::Code, '/') => Scan::Slash,
                (Scan::Code, _) => {
                    out.push(ch);
                    Scan::Code
                }
                (Scan::Slash, '/') => Scan::Line,
                (Scan::Slash, '*') => Scan::Block,
                (Scan::Slash, '"') => {
                    out.push('/');
                    out.push(ch);
                    Scan::Str
                }
                (Scan::Slash, _) => {
                    out.push('/');
                    out.push(ch);
                    Scan::Code
                }
                (Scan::Str, '\\') => {
                    out.push(ch);
                    Scan::StrEsc
                }
                (Scan::Str, '"') => {
                    out.push(ch);
                    Scan::Code
                }
                (Scan::Str, _) => {
                    out.push(ch);
                    Scan::Str
                }
                (Scan::StrEsc, _) => {
                    out.push(ch);
                    Scan::Str
                }
                (Scan::Line, '\n') => {
                    out.push(ch);
                    Scan::Code
                }
                (Scan::Line, _) => Scan::Line,
                (Scan::Block, '*') => Scan::BlockStar,
                (Scan::Block, '\n') => {
                    out.push(ch);
                    Scan::Block
                }
                (Scan::Block, _) => Scan::Block,
                (Scan::BlockStar, '/') => Scan::Code,
                (Scan::BlockStar, '*') => Scan::BlockStar,
                (Scan::BlockStar, '\n') => {
                    out.push(ch);
                    Scan::Block
                }
                (Scan::BlockStar, _) => Scan::Block,
            };
            prev = ch;
        }
        out
    }

    /// The CODE between the braces of the function opened by `signature`,
    /// comments removed.
    ///
    /// Brace-matched rather than cut at the first `\n}\n`, so a nested block
    /// or a closure cannot truncate the region and hide a missing call
    /// outside it.
    fn body_after(src: &str, signature: &str) -> String {
        let at = src
            .find(signature)
            .unwrap_or_else(|| panic!("the scanned surface no longer exists: {signature}"));
        let open = at + signature.len() - 1;
        assert!(
            src[open..].starts_with('{'),
            "the signature must end at its opening brace"
        );
        let mut depth = 0usize;
        for (offset, ch) in src[open..].char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return strip_comments(&src[open + 1..open + offset]);
                    }
                }
                _ => {}
            }
        }
        panic!("unbalanced braces after {signature}");
    }

    fn read(rel: &str) -> String {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel);
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read the scanned surface {}: {e}", path.display()))
    }

    /// Assembled from pieces so the search key never appears verbatim in this
    /// file. A guard whose own text matches its own pattern finds itself and
    /// survives the deletion it exists to catch — that self-reference has
    /// reddened a sibling guard three times in this epic.
    fn banner_signature() -> String {
        let name = "run";
        format!("pub(crate) fn {name}(model_path: &Path, config: &ServerConfig) -> Result<()> {{")
    }

    fn health_signature() -> String {
        let name = "build_health_response";
        format!("fn {name}(state: &AppState) -> HealthResponse {{")
    }

    fn receipt_signature() -> String {
        let name = "provenance_json";
        format!("fn {name}() -> serde_json::Value {{")
    }

    /// The two accessors the receipt renders, pinned to their `inference`
    /// arm. `apr bench` cannot call into realizar on a build that does not
    /// link it, so the accessor is where the delegation lives and the
    /// `not(inference)` arm is the stated absence of an engine, not a second
    /// derivation of a class.
    fn receipt_accessor_signature(name: &str, ret: &str) -> String {
        format!("#[cfg(feature = \"inference\")]\nfn {name}() -> {ret} {{")
    }

    /// SURFACE 1 — the serve startup banner.
    #[test]
    fn the_banner_delegates_to_the_one_andon() {
        let body = body_after(&read("src/commands/serve/mod.rs"), &banner_signature());
        assert!(
            renders_shared_compute_class(&body) && renders_shared_max_in_flight(&body),
            "serve::run no longer prints the shared andon. The banner is one of the \
             three surfaces APR-PERF-GATE-001 v2.2 §4 requires to read a single \
             compute_class(); a locally derived class here is how the receipt, \
             /health and the terminal drifted apart in the first place."
        );
    }

    /// SURFACE 2 — `GET /health`, in the OTHER crate.
    ///
    /// Read from disk rather than `include_str!`: a cross-crate `include_str!`
    /// would compile here and then break `cargo package -p apr-cli`, because
    /// the sibling crate's file is not in this crate's package.
    #[test]
    fn the_health_route_delegates_to_the_one_andon() {
        let body = body_after(
            &read("../aprender-serve/src/api/router.rs"),
            &health_signature(),
        );
        assert!(
            renders_shared_compute_class(&body),
            "build_health_response no longer reads the shared compute class. \
             /health then answers from its own derivation and can disagree with \
             the receipt that a release is judged on."
        );
        assert!(
            renders_shared_max_in_flight(&body),
            "build_health_response no longer reports max_in_flight. A server that \
             runs one request at a time must say so on the endpoint operators poll."
        );
    }

    /// SURFACE 3 — the `apr bench --json` receipt.
    ///
    /// Two hops, and BOTH are asserted. `provenance_json` renders the local
    /// accessors and the local accessors delegate; checking only the second
    /// hop would let the receipt inline a derivation of its own while a still
    /// correct accessor sat unused three lines above it.
    #[test]
    fn the_receipt_delegates_to_the_one_andon() {
        let src = read("src/commands/bench.rs");

        // Hop 1 — the provenance block renders the accessors.
        let prov = body_after(&src, &receipt_signature());
        assert!(
            prov.contains("compute_class()"),
            "provenance_json no longer records a compute class. This is the field \
             a cross-class ratio is caught by — a CPU-only apr side measured \
             against a CUDA comparator validates cleanly without it."
        );
        assert!(
            prov.contains("max_in_flight()"),
            "provenance_json no longer records max_in_flight — the receipt then \
             says nothing about how many requests the measured process ran at once."
        );

        // Hop 2 — the accessors are delegations, not derivations.
        let class_fn = body_after(
            &src,
            &receipt_accessor_signature("compute_class", "&'static str"),
        );
        assert!(
            renders_shared_compute_class(&class_fn),
            "apr bench derives its own compute class again. That is the third of \
             three answers APR-PERF-GATE-001 v2.2 §4 exists to collapse into one."
        );
        let bound_fn = body_after(&src, &receipt_accessor_signature("max_in_flight", "usize"));
        assert!(
            renders_shared_max_in_flight(&bound_fn),
            "apr bench counts its own in-flight requests again."
        );
    }

    /// DISCRIMINATION, half one: the predicate must not accept a body that
    /// merely NAMES the fact.
    ///
    /// Without this, `body.contains("compute_class")` would pass on a doc
    /// comment and the three assertions above would be decoration.
    #[test]
    fn naming_the_field_is_not_delegating_to_it() {
        let mentions_only = r#"
            // compute_class is the dispatch path taken, not the hardware present.
            let compute_class = if cfg!(feature = "cuda") { "cuda" } else { "cpu" };
            let max_in_flight = 1usize;
            serde_json::json!({ "compute_class": compute_class, "max_in_flight": max_in_flight })
        "#;
        assert!(
            !renders_shared_compute_class(mentions_only),
            "a body that spells the field name and derives the value itself must be \
             reported as NOT delegating — this is the mutation the three tests above \
             are supposed to catch"
        );
        assert!(!renders_shared_max_in_flight(mentions_only));
    }

    /// DISCRIMINATION, half two: the predicate must accept a real call, in
    /// every spelling the three crates use.
    ///
    /// A gate that flags everything is as broken as one that flags nothing,
    /// and only this half tells them apart.
    #[test]
    fn every_spelling_of_the_real_call_is_accepted() {
        for good in [
            "crate::andon::compute_class().to_string()",
            "realizar::andon::compute_class()",
            "println!(\"{}\", realizar::andon::andon_line());",
        ] {
            assert!(
                renders_shared_compute_class(good),
                "the gate must not fire on a genuine delegation: {good}"
            );
        }
        for good in [
            "crate::andon::max_in_flight()",
            "realizar::andon::max_in_flight()",
            "println!(\"{}\", realizar::andon::andon_line());",
        ] {
            assert!(
                renders_shared_max_in_flight(good),
                "the gate must not fire on a genuine delegation: {good}"
            );
        }
    }

    /// DISCRIMINATION, half three: prose cannot satisfy the gate, and code is
    /// not lost to the stripper.
    ///
    /// The mutation that exposed this was `the_banner_delegates_to_the_one_andon`
    /// passing with the real call deleted, because a comment two lines above
    /// spelled it out. Both polarities are asserted: a comment-only mention is
    /// REJECTED, and a real call sitting next to a comment is ACCEPTED.
    #[test]
    fn a_comment_that_names_the_call_does_not_count_as_making_it() {
        let prose_only =
            "fn f() {\n    // realizar::andon::andon_line() renders the class.\n    let x = 1;\n}";
        let body = body_after(prose_only, "fn f() {");
        assert!(
            !renders_shared_compute_class(&body),
            "a comment naming the call must not satisfy the wiring check: {body}"
        );

        let real_call = "fn f() {\n    // see andon_line, below\n    println!(\"{}\", realizar::andon::andon_line());\n}";
        let body = body_after(real_call, "fn f() {");
        assert!(
            renders_shared_compute_class(&body),
            "stripping comments must not eat the code next to them: {body}"
        );

        let block_comment =
            "fn f() {\n    /* realizar::andon::compute_class() */\n    let x = 1;\n}";
        assert!(!renders_shared_compute_class(&body_after(
            block_comment,
            "fn f() {"
        )));
    }

    /// The stripper must not treat a `//` inside a string literal as the start
    /// of a comment — that would silently delete real code to end of line.
    #[test]
    fn a_double_slash_inside_a_string_is_not_a_comment() {
        let src = "fn f() {\n    let u = \"http://x\"; realizar::andon::compute_class();\n}";
        let body = body_after(src, "fn f() {");
        assert!(
            renders_shared_compute_class(&body),
            "a URL in a string truncated the line: {body}"
        );
    }

    /// The brace matcher must return the function it was asked for, not the
    /// whole file. If it over-reads, every assertion above passes on any file
    /// that mentions the andon anywhere.
    #[test]
    fn the_scanner_reads_one_function_not_the_file() {
        let src = "fn a() {\n    let x = MARKER;\n}\nfn b() {\n    let y = 2;\n}\n";
        let body = body_after(src, "fn a() {");
        assert!(body.contains("MARKER"));
        assert!(
            !body.contains("let y = 2"),
            "the scanner leaked into the next function: {body}"
        );
    }
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

    /// The defect itself: on a build with no accelerator, `--gpu` must not be
    /// waved through. Before #2696 this returned Ok and the server ran on CPU.
    #[test]
    #[cfg(not(any(feature = "cuda", feature = "wgpu")))]
    fn gpu_without_a_backend_is_an_error_not_a_silent_cpu_run() {
        let err = ensure_accelerator_available(&cfg_with(true, false, None))
            .expect_err("--gpu on a CPU-only build must fail");
        let msg = err.to_string();
        // The remedy must be present AND runnable. #2527 shipped an error
        // naming a rebuild that could not be performed.
        assert!(
            msg.contains("cargo install aprender --features cuda"),
            "the error must name a remedy that works: {msg}"
        );
        assert!(
            msg.contains("--no-gpu"),
            "and the deliberate-CPU escape hatch: {msg}"
        );
    }

    /// Naming a backend the build cannot reach is the same defect in a
    /// different flag, so it takes the same path.
    #[test]
    #[cfg(not(any(feature = "cuda", feature = "wgpu")))]
    fn an_unreachable_backend_is_also_an_error() {
        for backend in ["wgpu", "cuda", "gpu"] {
            let err =
                ensure_accelerator_available(&cfg_with(false, false, Some(backend))).unwrap_err();
            assert!(
                err.to_string().contains(&format!("--backend {backend}")),
                "the message must quote the flag the user typed, not a generic one"
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

    /// EXHAUSTIVE OVER THE WHOLE INPUT SPACE.
    ///
    /// The predicate reads four booleans and allocates nothing, so "for all
    /// inputs" is sixteen cases, not a bounded proof. contracts/
    /// accelerator-request-v1.yaml declares a Kani harness for this and marks
    /// it `declared-not-written`; this is the cheaper alternative it names,
    /// written rather than deferred. `result` is a total function of
    /// (requested, can_dispatch) with exactly one Err cell.
    #[test]
    fn the_refusal_is_total_over_every_input() {
        let linked = cfg!(any(feature = "cuda", feature = "wgpu"));
        for gpu in [false, true] {
            for no_gpu in [false, true] {
                for backend in [None, Some("cpu"), Some("cuda"), Some("wgpu")] {
                    let cfg = cfg_with(gpu, no_gpu, backend);
                    let requested =
                        (gpu && !no_gpu) || matches!(backend, Some("cuda" | "wgpu" | "gpu"));
                    let expect_err = requested && !linked;
                    assert_eq!(
                        ensure_accelerator_available(&cfg).is_err(),
                        expect_err,
                        "gpu={gpu} no_gpu={no_gpu} backend={backend:?} linked={linked}: \
                         Err exactly when a request is made that this build cannot honour"
                    );
                }
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

    /// The quantity reaches the same refusal the boolean does — one gate, both
    /// spellings, so retiring `--gpu` cannot reopen the hole it closed.
    #[test]
    #[cfg(not(any(feature = "cuda", feature = "wgpu")))]
    fn gpu_layers_is_refused_on_a_build_with_no_accelerator() {
        let mut cfg = ServerConfig::default();
        cfg.gpu_layers = Some(GpuLayerRequest::All);
        let err = ensure_accelerator_available(&cfg).expect_err("must refuse");
        assert!(
            err.to_string().contains("--gpu-layers"),
            "quotes what was asked: {err}"
        );

        // ...and an explicit CPU request is not an accelerator request.
        cfg.gpu_layers = Some(GpuLayerRequest::None);
        ensure_accelerator_available(&cfg).expect("--gpu-layers 0 asks for no accelerator");
    }
}
