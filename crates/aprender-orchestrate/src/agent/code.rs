//! Public entry point for `apr code` / `batuta code`.
//!
//! This module provides the library-level API that both the `batuta` binary
//! and `apr-cli` use to launch the coding assistant. All logic lives here;
//! CLI wrappers are thin dispatchers.
//!
//! PMAT-162: Phase 6 — makes `cmd_code` accessible from the library crate
//! so `apr-cli` can call `batuta::agent::code::cmd_code()` directly.

use std::io::IsTerminal;
use std::path::PathBuf;
use std::sync::Arc;

use crate::agent::capability::Capability;
use crate::agent::driver::LlmDriver;
use crate::agent::manifest::{AgentManifest, ModelConfig, ResourceQuota};
use crate::agent::tool::file::{FileEditTool, FileReadTool, FileWriteTool};
use crate::agent::tool::search::{GlobTool, GrepTool};
use crate::agent::tool::shell::ShellTool;
use crate::agent::tool::ToolRegistry;
use crate::serve::backends::PrivacyTier;

/// Permission to execute exactly ONE agent turn.
///
/// Cannot be constructed except by [`TurnBudget::try_permit`], and
/// [`run_single_prompt`] cannot be called without one. That is the whole point:
/// `--max-turns` used to be a `u32` parameter that the non-interactive branch
/// simply never read, so `apr code --max-turns 0 -p "…"` launched `apr serve`,
/// ran a full turn and answered at exit 0 (dogfood-0.63.0, issue #2444). The
/// cap was inert at *every* value in `-p` mode, not merely mishandled at zero.
/// Forgetting to consult the budget is now a compile error, not a silent
/// over-run.
#[must_use]
#[derive(Debug)]
pub struct TurnPermit(());

/// The `--max-turns` cap, as a resource that must be spent to run a turn.
pub struct TurnBudget {
    max_turns: u32,
    used: u32,
}

impl TurnBudget {
    /// A budget of `max_turns` turns. `0` permits nothing — the same meaning
    /// the REPL has always given it (`repl.rs`: `turn_count >= max_turns`
    /// breaks before reading any input).
    pub fn new(max_turns: u32) -> Self {
        Self { max_turns, used: 0 }
    }

    /// Spend one turn, or refuse when the cap is exhausted.
    pub fn try_permit(&mut self) -> Option<TurnPermit> {
        if self.used >= self.max_turns {
            return None;
        }
        self.used += 1;
        Some(TurnPermit(()))
    }

    /// The cap this budget was created with (for error messages).
    pub fn max_turns(&self) -> u32 {
        self.max_turns
    }
}

/// Spend the one turn a non-interactive (`-p`) run costs, or refuse.
///
/// Returns `Ok(None)` for an interactive session, which spends its budget per
/// turn inside the REPL loop instead. Refusal is an error, not a quiet exit-0:
/// `apr code --max-turns 0 -p "…" > out.txt && use out.txt` must not treat an
/// empty answer as a successful one.
pub fn permit_single_prompt(
    budget: &mut TurnBudget,
    non_interactive: bool,
) -> anyhow::Result<Option<TurnPermit>> {
    if !non_interactive {
        return Ok(None);
    }
    match budget.try_permit() {
        Some(permit) => Ok(Some(permit)),
        None => anyhow::bail!(
            "--max-turns {}: refusing to run — a single-prompt run costs one turn and the budget allows none",
            budget.max_turns()
        ),
    }
}

/// The shape of an `apr code` invocation, as far as start-up policy cares.
///
/// Issue #2607: on the 0.63.0 dogfood sweep host, `apr code` with **no
/// arguments at all** and stdin closed did not print help. It scanned the
/// filesystem, auto-discovered the largest local GGUF (a 30 B MoE), and
/// spawned an `apr serve` child for it — a consequential action chosen by
/// looking at the disk, for a session that could never run: the REPL's very
/// first `read_line` on a closed stdin returns EOF, so the parent exited
/// immediately and left the child behind.
///
/// Only a *bare* invocation with **nothing on stdin** is refused. Every named
/// argument is an explicit operator choice and keeps working, including on a
/// pipe: `-p`/a prompt (non-interactive), `--model`, `--manifest`, `--resume`.
/// So does a pipe that actually carries bytes — see [`Self::stdin_has_input`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CodeInvocation {
    /// A positional prompt was supplied.
    pub has_prompt: bool,
    /// `--print` / `-p` was supplied.
    pub print: bool,
    /// `--model` was supplied.
    pub has_model: bool,
    /// `--manifest` was supplied.
    pub has_manifest: bool,
    /// `--resume` was supplied (with or without an id).
    pub has_resume: bool,
    /// stdin is an interactive terminal (so a REPL is actually possible).
    pub stdin_is_terminal: bool,
    /// stdin is not a terminal but **has bytes waiting** — a pipe or a
    /// redirected file carrying REPL input.
    ///
    /// `echo "hi" | apr code` has always driven the REPL:
    /// [`run_repl`](crate::agent::repl::run_repl) reads
    /// stdin line by line and treats EOF as `/exit`, so one piped line is one
    /// turn. Refusing on `!stdin_is_terminal` alone would have silently
    /// narrowed that to an exit-2 usage error. Piped bytes ARE an
    /// instruction; `/dev/null` and a closed fd are not, and those are the
    /// shape #2607 was reported against.
    ///
    /// [`from_args`](Self::from_args) only populates this when every other
    /// field already says "would refuse" — the peek blocks until the writer
    /// sends a byte or closes, exactly as the REPL's first `read_line` would,
    /// and there is no reason to pay that on a shape that runs regardless.
    pub stdin_has_input: bool,
}

impl CodeInvocation {
    /// Read the invocation shape of the current process' stdin plus the
    /// parsed flags. Split from [`Self::wants_help`] so the policy is
    /// testable without a controlled terminal.
    ///
    /// The stdin peek is deliberately last and deliberately conditional: it
    /// runs **only** when every flag already says "would refuse", because
    /// [`reader_has_input`] blocks until the writer produces a byte or closes
    /// the pipe. On every other shape the field is not consulted by
    /// [`Self::wants_help`], so leaving it `false` cannot change the verdict.
    #[must_use]
    pub fn from_args(
        prompt: &[String],
        print: bool,
        model: Option<&PathBuf>,
        manifest_path: Option<&PathBuf>,
        resume: Option<&Option<String>>,
    ) -> Self {
        let mut inv = Self {
            has_prompt: !prompt.is_empty(),
            print,
            has_model: model.is_some(),
            has_manifest: manifest_path.is_some(),
            has_resume: resume.is_some(),
            stdin_is_terminal: std::io::stdin().is_terminal(),
            stdin_has_input: false,
        };
        if inv.wants_help() {
            inv.stdin_has_input = stdin_may_carry_input();
        }
        inv
    }

    /// `true` when this invocation must print help and do nothing else.
    ///
    /// The invariant: **no argument was given AND no input can ever arrive**,
    /// so there is no work this run could legitimately do. Taking any action
    /// here — least of all launching an inference server for a model picked
    /// by scanning the disk — is a guess, not an instruction.
    ///
    /// "No input can ever arrive" is three distinct things, and only the
    /// first two were checked when this guard was first written:
    /// stdin is not a terminal (no REPL), **and** stdin has no bytes waiting
    /// (not a pipe carrying a prompt). Dropping the third clause turned
    /// `echo "hi" | apr code` — a working invocation — into an exit-2 usage
    /// error, which is a narrowing #2607 never asked for.
    #[must_use]
    pub fn wants_help(&self) -> bool {
        !self.has_prompt
            && !self.print
            && !self.has_model
            && !self.has_manifest
            && !self.has_resume
            && !self.stdin_is_terminal
            && !self.stdin_has_input
    }
}

/// `true` when `reader` has at least one byte available, without consuming it.
///
/// Blocks until the writer produces a byte or closes: `/dev/null`, a closed
/// fd, and a writer that closed without writing all report `false`; a pipe or
/// a redirected file carrying data reports `true`. `fill_buf` is a peek, so
/// the bytes stay queued for whoever reads next — and `Stdin`'s buffer is
/// process-global, so the later `read_line`/`read_to_string` drains the very
/// bytes this made visible.
///
/// Free function, generic over [`std::io::BufRead`], so the predicate is
/// falsifiable against real readers without a controlled terminal or a real
/// pipe. An I/O error is reported as "no input": a stdin that cannot be read
/// is exactly the shape that must refuse.
#[must_use]
pub fn reader_has_input(reader: &mut impl std::io::BufRead) -> bool {
    matches!(reader.fill_buf(), Ok(bytes) if !bytes.is_empty())
}

/// `true` when a stdin of this file type could ever deliver a byte.
///
/// A FIFO, a redirected regular file and a socket can; a character device
/// cannot, and that is the whole point — `apr code < /dev/null`, the shape
/// #2607 was reported against, lands here as a character device. A terminal
/// is also a character device, but [`CodeInvocation::from_args`] has already
/// settled that case with `IsTerminal` before this is consulted.
///
/// Split out as a pure predicate over a [`std::fs::FileType`] so it can be
/// falsified against real files (`/dev/null` vs a `tempfile`) rather than
/// against a mock.
#[cfg(unix)]
#[must_use]
pub fn kind_can_carry_input(file_type: &std::fs::FileType) -> bool {
    use std::os::unix::fs::FileTypeExt;
    file_type.is_fifo() || file_type.is_file() || file_type.is_socket()
}

/// Whether this process' stdin can still deliver input.
///
/// Two steps, in this order, and the order is the load-bearing part:
///
/// 1. Stat stdin. A character device (`/dev/null`) can never deliver a byte,
///    so it is refused **without reading** — which is also what keeps step 2
///    unreachable from any test harness, since `cargo nextest` hands each
///    test `/dev/null` and a plain `cargo test` in a terminal is short-
///    circuited by `IsTerminal` one level up.
/// 2. Otherwise peek. [`reader_has_input`] blocks until the writer sends a
///    byte or closes, which is exactly what the REPL's first `read_line`
///    always did — so `sleep 5 | apr code` waits, and `true | apr code`
///    (a pipe closed without a byte) is refused instead of launching a
///    server for a session that has nothing to say.
///
/// A stat that fails (no `/dev/stdin` on this host) falls through to the peek
/// rather than refusing: an unreadable `/dev/stdin` says nothing about fd 0,
/// and a closed fd 0 makes the peek return `false` on its own.
#[cfg(unix)]
fn stdin_may_carry_input() -> bool {
    match std::fs::metadata("/dev/stdin") {
        Ok(meta) if !kind_can_carry_input(&meta.file_type()) => false,
        _ => reader_has_input(&mut std::io::stdin().lock()),
    }
}

/// Non-Unix hosts have no `/dev/stdin` to stat; peek directly.
#[cfg(not(unix))]
fn stdin_may_carry_input() -> bool {
    reader_has_input(&mut std::io::stdin().lock())
}

/// Message shown when [`CodeInvocation::wants_help`] refuses a run.
pub const NO_ARG_NON_INTERACTIVE: &str =
    "apr code: no arguments, stdin is not a terminal, and nothing was piped in — \
     nothing to do.\n\
     An interactive session needs a terminal; a non-interactive run needs a prompt.\n\
     Try:  apr code -p \"explain src/lib.rs\"   |   echo \"hi\" | apr code   |   \
     apr code --model <path>";

/// #2607: refuse a bare `apr code` on a non-interactive stdin, before the
/// process does anything at all.
///
/// The `apr` CLI checks the same predicate one level up so it can render
/// clap's real help for the subcommand; this is [`cmd_code`] — a public
/// library API — failing closed, so no embedder can reach the
/// scan-the-disk-and-launch-a-server path by accident either.
fn refuse_bare_non_interactive(
    prompt: &[String],
    print: bool,
    model: Option<&PathBuf>,
    manifest_path: Option<&PathBuf>,
    resume: Option<&Option<String>>,
) -> anyhow::Result<()> {
    if CodeInvocation::from_args(prompt, print, model, manifest_path, resume).wants_help() {
        anyhow::bail!(NO_ARG_NON_INTERACTIVE);
    }
    Ok(())
}

/// Release every owner of the inference driver so its `Drop` actually runs.
///
/// Issue #2607, second defect. The `-p` branch below ends in
/// `std::process::exit`, which runs **no** destructors, so the `apr serve`
/// child has to be reaped explicitly. It called `drop(driver)` and a comment
/// claimed that killed the subprocess — but `driver` is an `Arc`, and
/// [`register_task_tool`](crate::agent::task_tool::register_task_tool) stores
/// a clone of it inside the tool registry. Dropping the local handle only
/// decremented the strong count from 2 to 1, so `AprServeDriver::drop` — the
/// one thing that SIGTERMs the child — never ran, and the server was orphaned
/// holding the whole model in RSS.
///
/// Taking both by value makes the ordering a compile-time obligation: the
/// registry cannot still be alive when the last driver handle is dropped.
fn release_driver(tools: ToolRegistry, driver: Arc<dyn LlmDriver>) {
    // The registry owns the TaskTool, which owns the other Arc clone.
    drop(tools);
    debug_assert_eq!(
        Arc::strong_count(&driver),
        1,
        "#2607: something still holds the driver; its Drop will not kill `apr serve`"
    );
    drop(driver);
}

/// Entry point for `batuta code` / `apr code`.
///
/// This is the public library API — callable from both the batuta binary
/// and apr-cli (PMAT-162). Handles model discovery, driver selection,
/// tool registration, and REPL launch.
#[allow(clippy::too_many_arguments)]
pub fn cmd_code(
    model: Option<PathBuf>,
    project: PathBuf,
    resume: Option<Option<String>>,
    prompt: Vec<String>,
    print: bool,
    max_turns: u32,
    manifest_path: Option<PathBuf>,
    emit_trace: Option<PathBuf>,
    // PMAT-CODE-OUTPUT-FORMAT-001 / PMAT-CODE-INPUT-FORMAT-001:
    // accepted as &str ("text" | "json") to keep this crate's public API
    // independent of apr-cli's ValueEnum types. Unknown values fall back
    // to "text" — the legacy behavior — under Poka-Yoke.
    output_format: &str,
    input_format: &str,
) -> anyhow::Result<()> {
    // #2607: settled BEFORE the working directory changes, before any
    // settings file is read, and — the point of the issue — before any model
    // is discovered or any `apr serve` child is spawned.
    refuse_bare_non_interactive(
        &prompt,
        print,
        model.as_ref(),
        manifest_path.as_ref(),
        resume.as_ref(),
    )?;

    // --project: change working directory for project instructions.
    // A path that is not a directory used to be skipped silently, so
    // `--project /typo` ran the agent against the CURRENT directory while the
    // operator believed it was scoped to another tree. Fail closed instead.
    if project.as_os_str() != "." {
        if !project.is_dir() {
            anyhow::bail!("--project: not a directory: {}", project.display());
        }
        std::env::set_current_dir(&project)?;
    }

    // --max-turns: settled BEFORE any model is launched, so a run that is not
    // allowed to do anything costs no `apr serve` subprocess and no weights
    // load. A non-interactive (`-p`) run is exactly one turn, so it needs one
    // permit; the REPL spends the same budget per turn inside its loop.
    let mut turn_budget = TurnBudget::new(max_turns);
    let single_prompt_permit = permit_single_prompt(&mut turn_budget, print || !prompt.is_empty())?;

    // --resume <id>: resolve the session BEFORE any model is launched.
    // An unknown id used to be discarded without a word: `-p` mode returned
    // from the non-interactive branch below without ever reading `resume`,
    // and the REPL only printed a warning — so a typo'd id left the user
    // believing a conversation was being continued that the model had no
    // history of. Fail closed, and name the id.
    let resumed_store = match resume {
        Some(Some(ref id)) => Some(
            crate::agent::session::SessionStore::resume(id)
                .map_err(|e| anyhow::anyhow!("--resume: no such session {id:?} ({e})"))?,
        ),
        _ => None,
    };

    // Load manifest or build default. When `--manifest` is set it short-
    // circuits the settings ladder (the manifest is treated as a complete
    // agent specification); otherwise we fold in
    // `~/.config/apr/settings.json` (user-global) and
    // `<project_root>/.apr/settings.json` (project-local) as Claude-Code
    // parity defaults (PMAT-CODE-CONFIG-LADDER-001). CLI flags always win.
    let mut manifest = match manifest_path {
        Some(ref path) => {
            let content = std::fs::read_to_string(path)
                .map_err(|e| anyhow::anyhow!("cannot read manifest {}: {e}", path.display()))?;
            let m = AgentManifest::from_toml(&content)
                .map_err(|e| anyhow::anyhow!("invalid manifest: {e}"))?;
            eprintln!("✓ Loaded manifest: {}", path.display());
            m
        }
        None => {
            let mut m = build_default_manifest();
            // PMAT-CODE-CONFIG-LADDER-001: settings.json layered defaults.
            // Errors are surfaced (Poka-Yoke) — a malformed settings file
            // is reported rather than silently ignored.
            let project_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let settings = crate::agent::settings::AprSettings::load_layered(&project_root)?;
            apply_settings_to_manifest(&mut m, &settings)?;
            m
        }
    };

    // --model flag overrides manifest model_path (and therefore overrides
    // any settings.json `model` field — CLI always wins, per the parity
    // ladder contract).
    if let Some(ref model_path) = model {
        manifest.model.model_path = Some(model_path.clone());
    }

    // PMAT-150: discover model with Jidoka validation (broken APR → GGUF fallback)
    discover_and_set_model(&mut manifest);

    // PMAT-198: Scale system prompt based on model size.
    // Small models (<2B) degrade with the full tool table + project context.
    if let Some(ref path) = manifest.model.model_path {
        let params_b = estimate_model_params_from_name(path);
        if params_b < 2.0 {
            manifest.model.system_prompt = scale_prompt_for_model(params_b);
        }
    }

    // Contract: no_model_error — never silently use MockDriver
    if manifest.model.resolve_model_path().is_none() && manifest_path.is_none() {
        print_no_model_error();
        std::process::exit(exit_code::NO_MODEL);
    }

    // PMAT-160: Try AprServeDriver first (apr serve has full CUDA/GPU).
    // Falls back to embedded RealizarDriver if `apr` binary not found.
    // PMAT-CODE-SPAWN-PARITY-001: driver stored as Arc so TaskTool can
    // share it with the AgentPool for sub-agent execution.
    let driver: Arc<dyn LlmDriver> = if let Some(model_path) = manifest.model.resolve_model_path() {
        match crate::agent::driver::apr_serve::AprServeDriver::launch(
            model_path,
            manifest.model.context_window,
        ) {
            Ok(d) => Arc::new(d),
            Err(e) => {
                eprintln!("⚠ apr serve unavailable ({e}), using embedded inference");
                Arc::from(build_fallback_driver(&manifest)?)
            }
        }
    } else {
        Arc::from(build_fallback_driver(&manifest)?)
    };

    // PMAT-CODE-MCP-JSON-LOADER-001: merge `<project>/.mcp.json` (Claude-Code-
    // shape) servers into manifest.mcp_servers BEFORE tool registration. The
    // manifest's TOML-declared servers always win on name collision (operator-
    // declared > project-default), matching the settings-ladder semantics.
    // Missing .mcp.json is a non-error; malformed JSON is a hard error.
    #[cfg(feature = "agents-mcp")]
    {
        let project_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        match crate::agent::mcp_json::load_and_merge(&mut manifest, &project_root) {
            Ok(0) => {}
            Ok(n) => {
                eprintln!("✓ Loaded {n} MCP server(s) from .mcp.json");
            }
            Err(e) => {
                anyhow::bail!("invalid .mcp.json: {e}");
            }
        }
    }

    // Build tool registry with coding tools
    let mut tools = build_code_tools(&manifest);

    // PMAT-CODE-MCP-CLIENT-001: register MCP client tools from manifest.mcp_servers.
    // Synchronous wrapper over async discover_mcp_tools — a no-op when mcp_servers is
    // empty (the default for `apr code` without a manifest).
    register_mcp_client_tools(&mut tools, &manifest);

    // PMAT-CODE-SPAWN-PARITY-001: register Task tool (Claude-Code Agent parity).
    // `task` lets the agent delegate to typed subagents (general-purpose,
    // explore, plan) with bounded recursion depth (Jidoka).
    crate::agent::task_tool::register_task_tool(
        &mut tools,
        &manifest,
        Arc::clone(&driver),
        /* max_depth */ 3,
    );

    // PMAT-CODE-HOOKS-001: build hook registry from manifest and fire SessionStart.
    // Returned Warn messages are surfaced to the user; a Block here aborts session
    // startup (matching Claude Code's exit-code-2 semantics).
    let hooks_reg = crate::agent::hooks::HookRegistry::from_configs(manifest.hooks.clone());
    let hook_cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    match hooks_reg.run(crate::agent::hooks::HookEvent::SessionStart, "", &hook_cwd) {
        crate::agent::hooks::HookDecision::Allow => {}
        crate::agent::hooks::HookDecision::Warn(msg) => {
            if !msg.is_empty() {
                eprintln!("⚠ SessionStart hook: {msg}");
            }
        }
        crate::agent::hooks::HookDecision::Block(reason) => {
            anyhow::bail!("SessionStart hook blocked session: {reason}");
        }
    }

    // Build memory
    let memory = crate::agent::memory::InMemorySubstrate::new();

    // Non-interactive mode: single prompt.
    // The branch condition IS the turn permit: the permit exists exactly when
    // `--max-turns` allowed this run, and `run_single_prompt` consumes it, so
    // no path can reach a turn without having spent budget for it (#2444).
    // PMAT-161: Return exit code instead of process::exit() so driver Drop
    // runs and kills the apr serve subprocess (no zombie processes).
    if let Some(permit) = single_prompt_permit {
        let prompt_text = if prompt.is_empty() {
            let mut buf = String::new();
            std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)?;
            // PMAT-CODE-INPUT-FORMAT-001: when --input-format=json, parse
            // a `{"role":"user","content":"..."}` envelope and use `content`
            // as the prompt. Empty/missing content is a hard error so the
            // operator notices the malformed envelope.
            if input_format.eq_ignore_ascii_case("json") {
                parse_json_input_envelope(&buf)?
            } else {
                buf
            }
        } else {
            prompt.join(" ")
        };
        // `--resume <id> -p ...` used to run with an EMPTY history — the
        // resumed session was accepted and then ignored, so the reply looked
        // plausible while the model had no idea what came before. Restore the
        // stored messages and append this turn back to the same session.
        let code = run_single_prompt(
            &manifest,
            driver.as_ref(),
            &tools,
            &memory,
            &prompt_text,
            emit_trace.as_deref(),
            output_format,
            resumed_store,
            permit,
        );
        // #2607: `drop(driver)` alone did NOT kill the `apr serve` child —
        // the tool registry holds a second `Arc` clone, so the strong count
        // never reached zero and `AprServeDriver::drop` never ran. `exit`
        // below runs no destructors, so this is the last chance to reap it.
        release_driver(tools, driver);
        std::process::exit(code);
    }

    // --resume: load previous session
    // PMAT-165: auto-resume prompt when recent session exists (spec §6.3)
    let resume_session_id = match resume {
        // Already proven to exist above (`resumed_store`); a bad id never
        // reaches here.
        Some(Some(id)) => Some(id), // --resume=<session-id>
        Some(None) => {
            // --resume (no ID): find most recent for cwd
            crate::agent::session::SessionStore::find_recent_for_cwd().map(|m| m.id)
        }
        None => {
            // No --resume flag: check for recent session and prompt
            crate::agent::session::offer_auto_resume()
        }
    };

    // Interactive REPL (local inference is free — budget unlimited)
    crate::agent::repl::run_repl(
        &manifest,
        driver.as_ref(),
        &tools,
        &memory,
        max_turns,
        f64::MAX,
        resume_session_id.as_deref(),
    )
}

/// PMAT-CODE-CONFIG-LADDER-001: fold loaded `~/.config/apr/settings.json` /
/// `<project>/.apr/settings.json` defaults into the default manifest **before**
/// CLI flags apply. Each `Some(_)` field on settings overrides the manifest
/// default; `None` fields leave the manifest alone. The CLI surface is wired
/// AFTER this so `--model` / `--max-turns` always win over settings.
///
/// PMAT-CODE-CONFIG-LADDER-FIELDS-001 (2026-05-07): also honors
/// `permissionMode` (validated via [`PermissionMode::parse`]; unknown
/// strings produce a hard error so a typo doesn't run the agent under the
/// wrong policy) and `allowedHosts` (mapped to [`AgentManifest::allowed_hosts`];
/// Sovereign privacy tier still wins as a Poka-Yoke).
fn apply_settings_to_manifest(
    manifest: &mut AgentManifest,
    settings: &crate::agent::settings::AprSettings,
) -> anyhow::Result<()> {
    if let Some(ref model) = settings.model {
        // Heuristic: a slash or starts with `hf://` / `./` / `/` → repo or
        // path. We keep this loose because the same field accepts both
        // `qwen3:1.7b-q4k` (apr pull alias) and `/abs/path.gguf`.
        if std::path::Path::new(model).is_absolute()
            || model.starts_with("./")
            || model.starts_with("../")
            || (!model.contains(':') && !model.starts_with("hf://"))
        {
            manifest.model.model_path = Some(std::path::PathBuf::from(model));
        } else {
            manifest.model.model_repo = Some(model.clone());
        }
    }
    if let Some(extra) = settings.extra_system_prompt.as_deref() {
        if !extra.trim().is_empty() {
            // Append, don't replace — base prompt must keep tool-calling
            // grammar guidance intact.
            manifest.model.system_prompt.push_str("\n\n");
            manifest.model.system_prompt.push_str(extra);
        }
    }
    if let Some(mt) = settings.max_turns {
        manifest.resources.max_iterations = mt;
    }
    if let Some(ref pm) = settings.permission_mode {
        // Parse once at apply time so the operator sees a clear error with
        // the bad value rather than a generic serde error. Currently only
        // the parse + validate is enforced — the runtime per-tool verdict
        // gate is tracked by PMAT-CODE-PERMISSIONS-RUNTIME-001.
        if crate::agent::permission::PermissionMode::parse(pm).is_none() {
            anyhow::bail!(
                "settings.json permissionMode: unknown mode {pm:?} \
                 (expected default | plan | acceptEdits | bypassPermissions)"
            );
        }
    }
    if let Some(ref hosts) = settings.allowed_hosts {
        // Only apply if the operator hasn't already declared an explicit
        // list via TOML manifest. Keeps manifest > settings precedence.
        if manifest.allowed_hosts.is_empty() {
            manifest.allowed_hosts = hosts.clone();
        }
    }
    Ok(())
}

/// Build fallback driver (embedded RealizarDriver) when AprServeDriver unavailable.
fn build_fallback_driver(manifest: &AgentManifest) -> anyhow::Result<Box<dyn LlmDriver>> {
    #[cfg(feature = "inference")]
    {
        if let Some(model_path) = manifest.model.resolve_model_path() {
            let driver = crate::agent::driver::realizar::RealizarDriver::new(
                model_path,
                manifest.model.context_window,
            )?;
            return Ok(Box::new(driver));
        }
    }
    let _ = manifest;
    // No model or no inference feature — return MockDriver
    Ok(Box::new(crate::agent::driver::mock::MockDriver::single_response(
        "Hello! I'm running in dry-run mode. \
         Set model_path in your agent manifest or install the `apr` binary.",
    )))
}

/// Auto-discover model if none explicitly set (APR preferred over GGUF).
fn discover_and_set_model(manifest: &mut AgentManifest) {
    if manifest.model.model_path.is_some() || manifest.model.model_repo.is_some() {
        return;
    }
    let Some(discovered) = ModelConfig::discover_model() else {
        return;
    };
    eprintln!(
        "Model: {} (auto-discovered)",
        discovered.file_name().unwrap_or_default().to_string_lossy()
    );
    let ext = discovered.extension().and_then(|e| e.to_str()).unwrap_or("");
    if ext == "gguf" && check_invalid_apr_in_search_dirs() {
        eprintln!(
            "⚠ APR model found but invalid (missing tokenizer). Using GGUF fallback: {}",
            discovered.display()
        );
        eprintln!("  Re-convert with: apr convert <source>.gguf -o <output>.apr\n");
    }
    manifest.model.model_path = Some(discovered);
}

/// Print actionable error when no local model is available.
fn print_no_model_error() {
    eprintln!("✗ No local model found. apr code requires a local model.\n");
    if check_invalid_apr_in_search_dirs() {
        eprintln!("  ⚠ APR model(s) found but invalid (missing embedded tokenizer).");
        eprintln!("  Re-convert: apr convert <source>.gguf -o <output>.apr\n");
    }
    eprintln!("  Download a model (APR format preferred):");
    eprintln!("    apr pull qwen3:1.7b-q4k            (default — best tool use at 1.2GB)");
    eprintln!("    apr pull qwen3:8b-q4k              (recommended for complex tasks)");
    eprintln!();
    eprintln!("  Or place a .apr/.gguf file in ~/.apr/models/ (auto-discovered)");
    eprintln!();
    eprintln!("  Then run: apr code or apr code --model <path>");
}

/// Check if any APR files in standard model search dirs are invalid.
fn check_invalid_apr_in_search_dirs() -> bool {
    for dir in &ModelConfig::model_search_dirs() {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "apr")
                    && !crate::agent::driver::validate::is_valid_model_file(&path)
                {
                    return true;
                }
            }
        }
    }
    false
}

/// Load project-level instructions from APR.md or CLAUDE.md.
fn load_project_instructions(max_bytes: usize) -> Option<String> {
    let cwd = std::env::current_dir().ok()?;

    for filename in &["APR.md", "CLAUDE.md"] {
        let path = cwd.join(filename);
        if path.is_file() {
            if let Ok(content) = std::fs::read_to_string(&path) {
                if max_bytes == 0 {
                    return None;
                }
                let truncated = if content.len() > max_bytes {
                    let end = content
                        .char_indices()
                        .take_while(|(i, _)| *i < max_bytes)
                        .last()
                        .map(|(i, c)| i + c.len_utf8())
                        .unwrap_or(max_bytes.min(content.len()));
                    format!("{}...\n(truncated from {} bytes)", &content[..end], content.len())
                } else {
                    content
                };
                return Some(truncated);
            }
        }
    }
    None
}

/// Compute instruction budget based on model context window.
fn instruction_budget(context_window: usize) -> usize {
    if context_window < 4096 {
        return 0;
    }
    let budget = context_window / 4;
    budget.min(4096)
}

/// PMAT-CODE-ORG-POLICY-RUNTIME-001: assemble the system prompt from
/// its component blocks in the canonical order (matches PolicyTier
/// precedence + project-instruction conventions).
///
/// Pure function — no I/O, no global state. Each input is `Option`-
/// wrapped so the caller can pass `None` for a missing block; the
/// helper is responsible for choosing whether to emit the section
/// heading at all.
///
/// Ordering rationale:
///
/// 1. `base` — the always-present `CODE_SYSTEM_PROMPT` (tool table,
///    grammar, sovereign-by-default reminders).
/// 2. `## Enforced organization policy` — `PolicyTier::Enforced`,
///    highest precedence; surfaced FIRST after `base` so downstream
///    sections cannot override it.
/// 3. `## Project Context` — git branch, file stats, language.
/// 4. `## Project Instructions` — CLAUDE.md / APR.md (with @import
///    expansion + user-level fallback).
/// 5. `## Auto-memory` — per-project memory directory contents.
fn assemble_system_prompt(
    base: &str,
    project_context: &str,
    project_instructions: Option<&str>,
    auto_memory: Option<&str>,
    org_policy: Option<&crate::agent::org_policy::OrgPolicy>,
) -> String {
    let mut out = String::from(base);
    if let Some(pol) = org_policy {
        out.push_str(&format!(
            "\n\n## Enforced organization policy ({source})\n\n{content}",
            source = pol.source.display(),
            content = pol.content
        ));
    }
    out.push_str(&format!("\n\n## Project Context\n\n{project_context}"));
    if let Some(instructions) = project_instructions {
        out.push_str(&format!("\n## Project Instructions\n\n{instructions}"));
    }
    if let Some(mem) = auto_memory {
        out.push_str(&format!("\n## Auto-memory\n\n{mem}"));
    }
    out
}

/// Gather project context — git info, file stats, language.
fn gather_project_context() -> String {
    let mut ctx = String::new();
    let cwd = std::env::current_dir().unwrap_or_default();
    ctx.push_str(&format!("Working directory: {}\n", cwd.display()));

    if let Ok(output) =
        std::process::Command::new("git").args(["rev-parse", "--abbrev-ref", "HEAD"]).output()
    {
        if output.status.success() {
            let branch = String::from_utf8_lossy(&output.stdout).trim().to_string();
            ctx.push_str(&format!("Git branch: {branch}\n"));
        }
    }
    if let Ok(output) =
        std::process::Command::new("git").args(["diff", "--stat", "--no-color"]).output()
    {
        if output.status.success() {
            let diff = String::from_utf8_lossy(&output.stdout);
            let dirty_count = diff.lines().count().saturating_sub(1);
            if dirty_count > 0 {
                ctx.push_str(&format!("Dirty files: {dirty_count}\n"));
            }
        }
    }

    let mut rs_count = 0u32;
    let mut py_count = 0u32;
    let mut total = 0u32;
    if let Ok(entries) = std::fs::read_dir("src") {
        for e in entries.flatten() {
            total += 1;
            if let Some(ext) = e.path().extension() {
                match ext.to_str() {
                    Some("rs") => rs_count += 1,
                    Some("py") => py_count += 1,
                    _ => {}
                }
            }
        }
    }
    let lang = if rs_count > py_count {
        "Rust"
    } else if py_count > 0 {
        "Python"
    } else {
        "unknown"
    };
    ctx.push_str(&format!("Language: {lang} ({total} files in src/)\n"));

    if PathBuf::from("Cargo.toml").exists() {
        ctx.push_str("Build system: Cargo (Rust)\n");
    } else if PathBuf::from("pyproject.toml").exists() {
        ctx.push_str("Build system: pyproject.toml (Python)\n");
    }

    ctx
}

/// Build a default `AgentManifest` for coding tasks.
fn build_default_manifest() -> AgentManifest {
    let ctx_window = 4096_usize;
    let budget = instruction_budget(ctx_window);
    // PMAT-CODE-MEMORY-PARITY-001: Use layered loader (user-global → project)
    // with `@import` resolution. Falls through to legacy single-file load
    // when nothing matches at either layer.
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut import_warnings = Vec::new();
    let project_instructions =
        crate::agent::instructions::load_layered_instructions(&cwd, budget, &mut import_warnings)
            .or_else(|| load_project_instructions(budget));
    for w in &import_warnings {
        eprintln!("⚠ instructions: {w}");
    }
    let project_context = gather_project_context();

    // PMAT-CODE-MEMORY-AUTO-001: load `*.md` files from
    // `~/.config/apr/projects/<slug>/memory/` into the system prompt
    // under a `## Auto-memory` section. Slug matches Claude Code's
    // hyphenated-path convention so `~/.claude/projects/` symlinks
    // continue to work cross-tool.
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut auto_warns: Vec<String> = Vec::new();
    let auto_memory = crate::agent::auto_memory::load_auto_memory(&cwd, &mut auto_warns);
    for w in &auto_warns {
        eprintln!("⚠ {w}");
    }

    // PMAT-CODE-ORG-POLICY-RUNTIME-001: load enforced org policy from
    // `/etc/apr-code/CLAUDE.md` (native first) or `/etc/claude-code/CLAUDE.md`
    // (cross-compat). The loader silently skips missing files + I/O errors so
    // a sandboxed runtime can't ransom REPL boot. PolicyTier::Enforced is the
    // highest tier — surfaced FIRST in the system prompt so a downstream
    // project / user / auto-memory section cannot override it. Uses the same
    // 25%-of-context budget as project_instructions; `max_bytes == 0`
    // disables the loader entirely (small models).
    let org_policy = crate::agent::org_policy::load_org_policy(
        &crate::agent::org_policy::canonical_system_roots(),
        "CLAUDE.md",
        budget,
    );

    let system_prompt = assemble_system_prompt(
        CODE_SYSTEM_PROMPT,
        &project_context,
        project_instructions.as_deref(),
        auto_memory.as_deref(),
        org_policy.as_ref(),
    );

    AgentManifest {
        name: "apr-code".to_string(),
        description: "Interactive AI coding assistant".to_string(),
        privacy: PrivacyTier::Sovereign,
        model: ModelConfig {
            system_prompt,
            max_tokens: 4096,
            temperature: 0.0,
            // PMAT-197: Qwen3 supports 32K context. Default 4096 caused
            // truncate_messages to drop user query (9 tool schemas ~4000 tokens
            // consumed the entire window). Set to 32K for Qwen3-class models.
            context_window: Some(32768),
            ..ModelConfig::default()
        },
        resources: ResourceQuota {
            max_iterations: 50,
            max_tool_calls: 200,
            max_cost_usd: 0.0,
            max_tokens_budget: None,
        },
        capabilities: vec![
            Capability::FileRead { allowed_paths: vec!["*".into()] },
            Capability::FileWrite { allowed_paths: vec!["*".into()] },
            Capability::Shell { allowed_commands: vec!["*".into()] },
            Capability::Memory,
            Capability::Rag,
        ],
        ..AgentManifest::default()
    }
}

/// PMAT-CODE-MCP-CLIENT-001 — register external MCP servers declared in
/// `manifest.mcp_servers[]` as tools in the `apr code` registry. Mirrors
/// Claude Code's `.mcp.json` → agent-tool-provider wiring. Synchronous
/// wrapper because `cmd_code` is sync; opens a scoped current-thread
/// runtime for the discovery handshake. No-op when the feature is off
/// or the manifest has no servers.
#[allow(unused_variables)]
fn register_mcp_client_tools(tools: &mut ToolRegistry, manifest: &AgentManifest) {
    #[cfg(feature = "agents-mcp")]
    {
        if manifest.mcp_servers.is_empty() {
            return;
        }
        let rt = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
            Ok(rt) => rt,
            Err(e) => {
                eprintln!("⚠ failed to create MCP discovery runtime: {e}");
                return;
            }
        };
        let discovered = rt.block_on(crate::agent::tool::mcp_client::discover_mcp_tools(manifest));
        let count = discovered.len();
        for tool in discovered {
            tools.register(Box::new(tool));
        }
        if count > 0 {
            eprintln!(
                "✓ Registered {count} MCP tool(s) from {} server(s)",
                manifest.mcp_servers.len()
            );
        }
    }
}

/// Register all coding tools.
fn build_code_tools(manifest: &AgentManifest) -> ToolRegistry {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    let mut tools = ToolRegistry::new();
    tools.register(Box::new(FileReadTool::new(vec!["*".into()])));
    tools.register(Box::new(FileWriteTool::new(vec!["*".into()])));
    tools.register(Box::new(FileEditTool::new(vec!["*".into()])));
    tools.register(Box::new(GlobTool::new(vec!["*".into()])));
    tools.register(Box::new(GrepTool::new(vec!["*".into()])));
    tools.register(Box::new(ShellTool::new(vec!["*".into()], cwd)));

    let memory_sub = Arc::new(crate::agent::memory::InMemorySubstrate::new());
    tools.register(Box::new(crate::agent::tool::memory::MemoryTool::new(
        memory_sub,
        manifest.name.clone(),
    )));

    // PMAT-163: dedicated pmat_query tool
    tools.register(Box::new(crate::agent::tool::pmat_query::PmatQueryTool::new()));

    #[cfg(feature = "rag")]
    {
        let oracle = Arc::new(crate::oracle::rag::RagOracle::new());
        tools.register(Box::new(crate::agent::tool::rag::RagTool::new(oracle, 5)));
    }

    // PMAT-CODE-WEB-TOOLS-001: register NetworkTool behind the privacy-tier
    // gate. Sovereign tier always blocks (Poka-Yoke); Standard/Private
    // tiers register iff `allowed_hosts` is non-empty (explicit opt-in).
    register_web_tools(&mut tools, manifest);

    tools
}

/// Register NetworkTool (+ BrowserTool when the `agents-browser` feature is
/// on) when the manifest declares a non-Sovereign privacy tier and a
/// non-empty `allowed_hosts` list.
fn register_web_tools(tools: &mut ToolRegistry, manifest: &AgentManifest) {
    use crate::serve::backends::PrivacyTier;

    if matches!(manifest.privacy, PrivacyTier::Sovereign) {
        return;
    }
    if manifest.allowed_hosts.is_empty() {
        return;
    }

    tools.register(Box::new(crate::agent::tool::network::NetworkTool::new(
        manifest.allowed_hosts.clone(),
    )));

    #[cfg(feature = "agents-browser")]
    {
        tools.register(Box::new(crate::agent::tool::browser::BrowserTool::new(manifest.privacy)));
    }
}

pub use super::code_prompts::exit_code;

/// Run a single prompt (non-interactive). PMAT-172: cap iterations at 10.
///
/// `resumed` is `Some` when `--resume <id>` named an existing session: its
/// stored messages become this turn's history, and the messages produced here
/// are appended back to it. Without this the flag was accepted and dropped.
#[allow(clippy::too_many_arguments)]
fn run_single_prompt(
    manifest: &AgentManifest,
    driver: &dyn LlmDriver,
    tools: &ToolRegistry,
    memory: &dyn crate::agent::memory::MemorySubstrate,
    prompt: &str,
    emit_trace: Option<&std::path::Path>,
    // PMAT-CODE-OUTPUT-FORMAT-001: "text" (default) or "json".
    output_format: &str,
    resumed: Option<crate::agent::session::SessionStore>,
    // The turn this call spends. Taken by value so it is consumed here and
    // cannot be reused; its existence proves `--max-turns` was consulted.
    permit: TurnPermit,
) -> i32 {
    let TurnPermit(()) = permit;
    let mut single_manifest = manifest.clone();
    single_manifest.resources.max_iterations = single_manifest.resources.max_iterations.min(10);
    // PMAT-197: Use compact system prompt for -p mode.
    // The full CODE_SYSTEM_PROMPT (9-tool table + project context + CLAUDE.md)
    // overwhelms Qwen3 1.7B causing </think> loops. For -p mode, use a minimal
    // prompt that lets the model answer directly. Tools still available if needed.
    single_manifest.model.system_prompt = COMPACT_SYSTEM_PROMPT.to_string();
    // Note: context_window is set at driver launch time (build_default_manifest),
    // not here. See PMAT-197 fix in build_default_manifest.

    let rt = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("Error: failed to create tokio runtime: {e}");
            return exit_code::AGENT_ERROR;
        }
    };

    let started = std::time::Instant::now();

    // History is empty unless `--resume <id>` named a session, in which case
    // the stored messages are replayed so this turn actually continues the
    // conversation (run_agent_loop is exactly this call with an empty vec).
    let mut resumed = resumed;
    let mut history = match resumed {
        Some(ref store) => store.load_messages().unwrap_or_default(),
        None => Vec::new(),
    };
    let prior_len = history.len();
    if let Some(ref store) = resumed {
        eprintln!("✓ Resumed session {} ({prior_len} messages)", store.id());
    }

    // PMAT-197: Use non-nudge loop for -p mode. The nudge ("Use a tool!") forces
    // small models to make tool calls even for simple questions like "What is 2+2?"
    // which causes stuck loops. Let the model decide whether to use tools.
    let result = rt.block_on(crate::agent::runtime::run_agent_turn(
        &single_manifest,
        &mut history,
        prompt,
        driver,
        tools,
        memory,
        None,
    ));

    // Persist this turn back into the resumed session so a follow-up
    // `--resume` sees it.
    if let Some(ref mut store) = resumed {
        if history.len() > prior_len {
            let _ = store.append_messages(&history[prior_len..]);
        }
        let _ = store.record_turn();
    }

    match result {
        Ok(r) => {
            let elapsed = started.elapsed();
            if r.text.is_empty() {
                // PMAT-190: Empty response — model may be emitting only thinking tokens
                // that get stripped by strip_thinking_blocks(). Common with Qwen3 when
                // the serve backend doesn't use Qwen3NoThinkTemplate.
                eprintln!(
                    "⚠ Empty response ({} iterations, {} tool calls). \
                     Model may be in thinking mode — rebuild apr from source for Qwen3NoThinkTemplate fix.",
                    r.iterations, r.tool_calls
                );
                if output_format.eq_ignore_ascii_case("json") {
                    println!("{}", build_json_result_envelope(&r, elapsed, /*is_error*/ true));
                }
            } else if output_format.eq_ignore_ascii_case("json") {
                // PMAT-CODE-OUTPUT-FORMAT-001: structured envelope mirroring
                // Claude Code's `claude -p --output-format json` shape.
                println!("{}", build_json_result_envelope(&r, elapsed, /*is_error*/ false));
            } else {
                println!("{}", r.text);
            }

            // PMAT-CODE-EMIT-TRACE-001 (M28): write a ccpa-trace.jsonl
            // describing this run. Used by `ccpa measure` to score
            // apr code against canonical Claude Code reference fixtures.
            if let Some(trace_path) = emit_trace {
                let model = single_manifest
                    .model
                    .resolve_model_path()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "apr-code-unknown".to_owned());
                if let Err(e) = emit_ccpa_trace(trace_path, prompt, &r, started.elapsed(), &model) {
                    eprintln!("⚠ failed to write ccpa-trace to {}: {e}", trace_path.display());
                }
            }

            exit_code::SUCCESS
        }
        Err(e) => {
            eprintln!("Error: {e}");
            map_error_to_exit_code(&e)
        }
    }
}

/// Emit a `ccpa-trace.jsonl` (M28) describing a single apr-code run.
///
/// Schema mirrors `claude-code-parity-apr-v1.yaml § trace_schema`. For
/// the M28 minimum-viable scope we emit four records:
///
///   1. `session_start`  with a synthetic `session_id` derived from
///      `started`'s wall-clock ts so re-runs differ; `cwd_sha256`
///      placeholder is normalized at compare time by the differ.
///   2. `user_prompt`    turn 0, verbatim text.
///   3. `assistant_turn` turn 1, single `Block::Text` carrying
///      `result.text`. Tool dispatch + hook + skill records are
///      M29+ enrichment follow-ups.
///   4. `session_end`    real elapsed_ms + token counts from
///      `result.usage`.
fn emit_ccpa_trace(
    path: &std::path::Path,
    prompt: &str,
    result: &super::result::AgentLoopResult,
    elapsed: std::time::Duration,
    model: &str,
) -> std::io::Result<()> {
    use std::time::{SystemTime, UNIX_EPOCH};

    let ts_micros =
        SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_micros()).unwrap_or(0);
    // session_id: UUIDv7-shaped hex string of the start ts. Normalized
    // by the differ at compare time so this only needs to be stable
    // across teacher and student of the SAME fixture (re-running the
    // same fixture produces a different session_id, which is fine).
    let session_id = format!(
        "{:08x}-{:04x}-7000-{:04x}-{:012x}",
        (ts_micros >> 64) as u32 & 0xFFFF_FFFF,
        ((ts_micros >> 48) & 0xFFFF) as u16,
        ((ts_micros >> 32) & 0xFFFF) as u16,
        (ts_micros & 0xFFFF_FFFF_FFFF) as u64
    );
    // ts in ISO 8601 — not strictly RFC 3339, but the differ
    // normalizes ts at compare time.
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let ts = format!("@{secs}");
    let cwd_sha256 = "0".repeat(64);

    let session_start = serde_json::json!({
        "v": 1,
        "kind": "session_start",
        "session_id": session_id,
        "ts": ts,
        "actor": "apr-code",
        "model": model,
        "cwd_sha256": cwd_sha256,
    });
    let user_prompt = serde_json::json!({
        "v": 1,
        "kind": "user_prompt",
        "turn": 0,
        "text": prompt,
    });
    let assistant_turn = serde_json::json!({
        "v": 1,
        "kind": "assistant_turn",
        "turn": 1,
        "blocks": [{"type": "text", "text": result.text}],
        "stop_reason": "end_turn",
    });
    let session_end = serde_json::json!({
        "v": 1,
        "kind": "session_end",
        "turn": 1,
        "stop_reason": "end_turn",
        "elapsed_ms": elapsed.as_millis() as u64,
        "tokens_in": result.usage.input_tokens,
        "tokens_out": result.usage.output_tokens,
    });

    let body = format!("{}\n{}\n{}\n{}\n", session_start, user_prompt, assistant_turn, session_end);
    std::fs::write(path, body)
}

/// PMAT-CODE-INPUT-FORMAT-001 (M-NON-INT-002): parse a `{"role":"user","content":"..."}`
/// JSON envelope from stdin and return the prompt text. Mirrors the shape Claude
/// Code accepts on `claude -p --input-format json`.
///
/// Errors are surfaced (not silently downgraded) so a malformed envelope fails
/// loudly instead of running the agent on garbage. `role` other than `"user"`
/// is also rejected — the non-interactive surface is single-user-turn only.
fn parse_json_input_envelope(buf: &str) -> anyhow::Result<String> {
    let trimmed = buf.trim();
    if trimmed.is_empty() {
        anyhow::bail!("--input-format=json: stdin is empty (expected JSON envelope)");
    }
    let v: serde_json::Value = serde_json::from_str(trimmed)
        .map_err(|e| anyhow::anyhow!("--input-format=json: invalid JSON on stdin: {e}"))?;
    let role = v.get("role").and_then(|r| r.as_str()).unwrap_or("user");
    if role != "user" {
        anyhow::bail!("--input-format=json: only role=\"user\" supported, got \"{role}\"");
    }
    let content = v
        .get("content")
        .and_then(|c| c.as_str())
        .ok_or_else(|| anyhow::anyhow!("--input-format=json: missing string field `content`"))?;
    Ok(content.to_owned())
}

/// PMAT-CODE-OUTPUT-FORMAT-001 (M-NON-INT-001): build a structured JSON
/// envelope mirroring Claude Code's `claude -p --output-format json` shape:
///
/// ```json
/// {
///   "type": "result",
///   "subtype": "success",
///   "is_error": false,
///   "duration_ms": 1234,
///   "result": "the assistant text",
///   "session_id": "<uuidv7-shaped>",
///   "num_turns": 1,
///   "total_cost_usd": 0
/// }
/// ```
fn build_json_result_envelope(
    result: &super::result::AgentLoopResult,
    elapsed: std::time::Duration,
    is_error: bool,
) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts_micros =
        SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_micros()).unwrap_or(0);
    // Same UUIDv7-shaped stable-per-run session id used by emit_ccpa_trace.
    let session_id = format!(
        "{:08x}-{:04x}-7000-{:04x}-{:012x}",
        (ts_micros >> 64) as u32 & 0xFFFF_FFFF,
        ((ts_micros >> 48) & 0xFFFF) as u16,
        ((ts_micros >> 32) & 0xFFFF) as u16,
        (ts_micros & 0xFFFF_FFFF_FFFF) as u64
    );
    let envelope = serde_json::json!({
        "type": "result",
        "subtype": if is_error { "error" } else { "success" },
        "is_error": is_error,
        "duration_ms": elapsed.as_millis() as u64,
        "result": result.text,
        "session_id": session_id,
        "num_turns": result.iterations,
        "tokens_in": result.usage.input_tokens,
        "tokens_out": result.usage.output_tokens,
        // Local sovereign inference: cost is always zero by construction.
        "total_cost_usd": 0,
    });
    envelope.to_string()
}

// Prompts and exit codes extracted to code_prompts.rs
use super::code_prompts::{
    estimate_model_params_from_name, map_error_to_exit_code, scale_prompt_for_model,
    CODE_SYSTEM_PROMPT, COMPACT_SYSTEM_PROMPT,
};

#[cfg(test)]
#[path = "code_tests.rs"]
mod tests;
