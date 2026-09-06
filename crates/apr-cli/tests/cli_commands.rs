// Integration tests: unwrap()/panic!() are idiomatic; strict workspace lints relaxed here.
#![allow(
    clippy::disallowed_methods,
    clippy::needless_range_loop,
    clippy::format_collect,
    clippy::format_push_string,
    clippy::manual_assert,
    clippy::uninlined_format_args,
    clippy::unnecessary_debug_formatting,
    clippy::unwrap_or_default,
    clippy::expect_fun_call,
    clippy::manual_repeat_n,
    clippy::unnecessary_map_or
)]

//! APR CLI Commands Integration Tests
//!
//! Enforces: contracts/apr-cli-commands-v1.yaml
//! FALSIFY-CLI-001 through FALSIFY-CLI-005
//!
//! Every `apr` subcommand must:
//! 1. Respond to `--help` with exit code 0
//! 2. Be registered in the command contract
//! 3. Never panic

use std::process::Command;

/// All commands registered in contracts/apr-cli-commands-v1.yaml.
/// This is the Rust-side mirror of the YAML contract.
///
/// As of aprender#1638 / M150, all commands are unconditionally
/// compiled — the `code` feature flag was removed so `apr code` ships
/// in the default build. The contract YAML lists `code` unconditionally
/// too (see contracts/apr-cli-commands-v1.yaml § code).
fn registered_commands() -> Vec<&'static str> {
    vec![
        "run",
        "devices",
        "serve",
        "chat",
        "inspect",
        "debug",
        "validate",
        "validate-manifest",
        "lint",
        "beat-run",
        "manifest",
        "explain",
        "tensors",
        // aprender#2377 finding 3: the producers `*-lint` help documents.
        "dataset",
        "kernel",
        "trace",
        "diff",
        "hex",
        "tree",
        "flow",
        "export",
        "import",
        "convert",
        "stamp",
        "compile",
        "merge",
        "quantize",
        "rosetta",
        "pull",
        "list",
        "rm",
        "registry",
        "publish",
        "finetune",
        "prune",
        "distill",
        "train",
        "pretrain",
        "tokenize",
        "tune",
        "bench",
        "eval",
        "check",
        "qa",
        "qualify",
        "canary",
        "compare-hf",
        "parity",
        "gpu",
        "profile",
        "ptx",
        "ptx-map",
        "cbtop",
        "data",
        "pipeline",
        "tui",
        "monitor",
        "runs",
        "experiment",
        "showcase",
        // Renamed from "probar" (#2525). `apr probar` survives as a HIDDEN
        // clap alias, and clap omits hidden aliases from --help -- so the
        // name that must appear in this list is the visible one.
        "test",
        "modelfile",
        "diagnose",
        "ollama-chat-lint",
        "ollama-tools-lint",
        "dry-sampling-lint",
        "awq-lint",
        "fp8-lint",
        "nf4-lint",
        "gptq-lint",
        "oom-lint",
        "tool-use-lint",
        "gbnf-lint",
        "typical-p-lint",
        "registry-quota-lint",
        "imatrix-lint",
        "embeddings-lint",
        "unified-search-lint",
        "rm-gc-lint",
        "shared-cache-lint",
        "ppl",
        "quant-preservation-lint",
        "prometheus-lint",
        "otlp-lint",
        "kv-timeline-lint",
        "gpu-memtrace-lint",
        "explain-token-lint",
        "check-finite-lint",
        "attn-viz-lint",
        "embed-viz-lint",
        "nccl-diag-lint",
        "react-trace-lint",
        "hang-trace-lint",
        "ddp-metrics-lint",
        "audio-inspect-lint",
        "attn-parity-lint",
        "rerank",
        "embed",
        "shard",
        "unshard",
        "oracle",
        "grad-norm",
        "encrypt",
        "decrypt",
        "mcp",
        "code",
        // APR-MONO: sibling CLIs that had no route through apr at all
        "rag",
        "zram",
        "sim",
        "cgp",
        "pv",
    ]
}

fn apr_binary() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_apr"));
    cmd.env("NO_COLOR", "1");
    cmd
}

/// The `Commands:` block of a `--help` text: every line after the header up to
/// the next `Options:`/`Arguments:` header.
fn help_block(stdout: &str) -> impl Iterator<Item = &str> {
    stdout
        .lines()
        .skip_while(|l| !l.starts_with("Commands:"))
        .skip(1)
        .take_while(|l| !l.starts_with("Options:") && !l.starts_with("Arguments:"))
}

/// The command name of a row, or None for anything that is not a row.
///
/// Command rows have exactly 2-space indent (`  cmd  description...`).
/// Wrapped description continuation lines have a much wider indent
/// (column-aligned to the description start, typically 20+ spaces), so the
/// first word of a wrap continuation is never taken for a name (CRUX-B-19).
fn row_name(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("  ")?;
    if rest.starts_with(' ') {
        return None;
    }
    rest.split_whitespace().next()
}

/// Valid command names: lowercase, may contain hyphens, no parens/uppercase.
fn looks_like_command(name: &str) -> bool {
    name.chars().next().is_some_and(|c| c.is_ascii_lowercase())
        && !name.contains('(')
        && !name.contains(')')
}

fn get_help_commands() -> Vec<String> {
    let output = apr_binary()
        .arg("--help")
        .output()
        .expect("failed to run apr --help");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut commands = Vec::new();
    for line in help_block(&stdout) {
        if line.is_empty() && commands.len() > 5 {
            break;
        }
        if let Some(name) = row_name(line).filter(|n| looks_like_command(n)) {
            commands.push(name.to_string());
        }
    }
    commands
}

/// FALSIFY-CLI-003 + FALSIFY-CLI-004: Every command responds to --help with exit 0.
#[test]
fn test_all_commands_respond_to_help() {
    let mut failures = Vec::new();

    for cmd in registered_commands() {
        let output = apr_binary()
            .args([cmd, "--help"])
            .output()
            .unwrap_or_else(|e| panic!("failed to run apr {} --help: {}", cmd, e));

        if !output.status.success() {
            failures.push(format!(
                "apr {} --help exited with {:?}",
                cmd,
                output.status.code()
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "FALSIFY-CLI-003: Commands that failed --help:\n{}",
        failures.join("\n")
    );
}

/// FALSIFY-CLI-001: Every command in the contract exists in `apr --help`.
#[test]
fn test_all_contract_commands_exist() {
    let help_commands = get_help_commands();
    let mut missing = Vec::new();

    for cmd in registered_commands() {
        if !help_commands.iter().any(|h| h == cmd) {
            missing.push(cmd);
        }
    }

    assert!(
        missing.is_empty(),
        "FALSIFY-CLI-001: Commands in contract but missing from `apr --help`: {:?}",
        missing
    );
}

/// FALSIFY-CLI-002: Every command in `apr --help` is in the contract.
#[test]
fn test_no_unregistered_commands() {
    let help_commands = get_help_commands();
    let cmds = registered_commands();
    let registered: std::collections::HashSet<&str> = cmds.iter().copied().collect();
    let mut unregistered = Vec::new();

    for cmd in &help_commands {
        if !registered.contains(cmd.as_str()) {
            // Skip "help" which is auto-generated by clap
            if cmd != "help" {
                unregistered.push(cmd.clone());
            }
        }
    }

    assert!(
        unregistered.is_empty(),
        "FALSIFY-CLI-002: Commands in `apr --help` but not in contract: {:?}\n\
         Add them to contracts/apr-cli-commands-v1.yaml AND this test's registered_commands().",
        unregistered
    );
}

/// FALSIFY-CLI-005: Command count matches between contract and --help.
#[test]
fn test_command_count_matches() {
    let help_commands = get_help_commands();
    // Subtract "help" which clap adds automatically
    let help_count = help_commands
        .iter()
        .filter(|c| c.as_str() != "help")
        .count();
    let contract_count = registered_commands().len();

    assert_eq!(
        help_count, contract_count,
        "FALSIFY-CLI-005: Command count mismatch.\n\
         `apr --help` has {} commands, contract has {}.\n\
         Help commands: {:?}",
        help_count, contract_count, help_commands
    );
}

/// FALSIFY-CLI-006: `apr --version` outputs version string.
#[test]
fn test_version_flag() {
    let output = apr_binary()
        .arg("--version")
        .output()
        .expect("failed to run apr --version");

    assert!(output.status.success(), "apr --version should exit 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("apr"),
        "apr --version should contain 'apr': got {:?}",
        stdout
    );
}

/// FALSIFY-CLI-007: `apr` with no args exits with code 2 (usage error).
#[test]
fn test_no_args_exits_usage_error() {
    let output = apr_binary()
        .output()
        .expect("failed to run apr with no args");

    let code = output.status.code().unwrap_or(-1);
    assert_eq!(
        code, 2,
        "apr with no args should exit 2 (usage error), got {}",
        code
    );
}

/// FALSIFY-APR-PRETRAIN-INIT-007: 3-surface drift prevention.
///
/// Asserts the `--init` flag appears in `apr pretrain --help` output. This is
/// the integration-test surface that completes the 3-surface drift triangle:
/// (1) clap field in `Pretrain { init: Option<PathBuf> }`,
/// (2) unit tests `pretrain_init_*` in `crates/apr-cli/src/commands/pretrain.rs`,
/// (3) integration test (this one) — confirming the flag is reachable from the
/// installed binary's help surface.
///
/// If clap definition drifts (renamed, removed, hidden), this test fails.
#[test]
fn pretrain_init_flag_registered() {
    let output = apr_binary()
        .args(["pretrain", "--help"])
        .output()
        .expect("failed to run apr pretrain --help");

    assert!(
        output.status.success(),
        "apr pretrain --help should exit 0, got {:?}",
        output.status.code()
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("--init"),
        "FALSIFY-APR-PRETRAIN-INIT-007: `--init` flag missing from `apr pretrain --help`.\n\
         Either clap definition drifted, or pretrain subcommand wasn't built with the flag.\n\
         Full --help output:\n{}",
        stdout
    );
}

/// FALSIFY-TOK-IMPORT-HF-001: `apr tokenize import-hf` subcommand is
/// registered in the dispatch surface.
///
/// Asserts `apr tokenize import-hf --help` exits 0 and prints help text
/// mentioning the `--input`, `--output`, and `--include-added-tokens` flags
/// that `apr-cli-tokenize-import-hf-v1` v1.0.0 §extraction_signature pins.
///
/// 3-surface drift prevention triangle for `import-hf`:
///   (1) clap variant `TokenizeCommands::ImportHf { ... }` in
///       `crates/apr-cli/src/tokenize_commands.rs`
///   (2) unit tests `commands::tokenize::tests::import_hf_*` in
///       `crates/apr-cli/src/commands/tokenize.rs`
///   (3) integration test (this one) — confirming the subcommand is
///       reachable from the installed binary's help surface.
///
/// If `tokenize_commands.rs` drops the `ImportHf` variant, or
/// `dispatch_analysis.rs` fails to wire it through, or the binary is built
/// without the subcommand registered, this test fails.
#[test]
fn tokenize_import_hf_subcommand_registered() {
    let output = apr_binary()
        .args(["tokenize", "import-hf", "--help"])
        .output()
        .expect("failed to run apr tokenize import-hf --help");

    assert!(
        output.status.success(),
        "FALSIFY-TOK-IMPORT-HF-001: `apr tokenize import-hf --help` should exit 0, got {:?}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    for flag in ["--input", "--output", "--include-added-tokens"] {
        assert!(
            stdout.contains(flag),
            "FALSIFY-TOK-IMPORT-HF-001: `{flag}` flag missing from `apr tokenize import-hf --help`.\n\
             Either clap definition drifted, dispatch isn't wired, or the binary was built without the subcommand.\n\
             Full --help output:\n{stdout}"
        );
    }
}

/// FALSIFY-BACKEND-CUDA-HONESTY-001: `--backend cuda` must not silently serve wgpu/CPU.
///
/// The CUDA generate path lives behind `#[cfg(feature = "cuda")]`
/// (aprender-serve/src/infer/gguf_gpu_generate.rs). On a build without that feature the
/// block vanishes, control falls through to the GH-559 wgpu fallback, and the run prints
/// `Backend override: cuda` followed by `Backend: wgpu (Vulkan)` — accepting the flag and
/// then ignoring it. wgpu fails its own cpu-parity gate and degrades again, so the user
/// gets ~20 tok/s where CUDA gives ~400.
///
/// Measured 2026-07-27 on an RTX 4090 with nvcc 12.8 installed, so this is not a
/// missing-hardware case: it is a binary that cannot honour the flag reporting success.
/// Any throughput measured through it is meaningless — the Pillar-4 decode beat run
/// against such a build reports `ratio_median=0.070x` and a BEAT-REGRESSION panic, a
/// fabricated 14x regression with nothing actually wrong in apr's decode path.
///
/// This test runs on the default (non-cuda) test build, so it exercises exactly that
/// case. RED-on-bug: without the guard in dispatch.rs the process proceeds and does not
/// report the refusal.
#[test]
fn backend_cuda_on_non_cuda_build_refuses_instead_of_falling_back() {
    if cfg!(feature = "cuda") {
        // A CUDA-capable build is allowed to proceed; the guard is build-capability only.
        return;
    }

    let output = apr_binary()
        .args([
            "run",
            "/nonexistent-model-path-for-backend-guard.gguf",
            "--prompt",
            "hi",
            "--max-tokens",
            "1",
            "--backend",
            "cuda",
        ])
        .output()
        .expect("failed to run apr");

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    assert!(
        !output.status.success(),
        "FALSIFY-BACKEND-CUDA-HONESTY-001: `--backend cuda` exited 0 on a build with no \
         CUDA compiled in. It must refuse rather than silently serve wgpu/CPU (~20x slower).\n\
         Output:\n{combined}"
    );
    assert!(
        combined.contains("built WITHOUT the `cuda` feature"),
        "FALSIFY-BACKEND-CUDA-HONESTY-001: expected an explicit refusal naming the missing \
         `cuda` feature, so the user is not left to infer it from a throughput number.\n\
         Output:\n{combined}"
    );
    assert!(
        !combined.contains("Backend: wgpu"),
        "FALSIFY-BACKEND-CUDA-HONESTY-001: `--backend cuda` fell through to the wgpu \
         backend — this is the exact silent-downgrade this gate exists to prevent.\n\
         Output:\n{combined}"
    );
}

/// True when the report printed `<gate> : Ok` — i.e. a positive assertion that
/// the named gate examined the observation and was satisfied by it.
fn report_says_gate_is_ok(stdout: &str, gate: &str) -> bool {
    stdout.lines().any(|line| {
        let line = line.trim();
        match line.split_once(':') {
            Some((label, verdict)) => label.trim() == gate && verdict.trim().starts_with("Ok"),
            None => false,
        }
    })
}

/// FALSIFY-CLI-THRESHOLD-NAN-001: a NaN (or negative) tolerance must never turn
/// a failing CRUX lint gate into a reported pass.
///
/// Every one of these gates compares an observation against a threshold —
/// `mad > tol_abs`, `efficiency < floor`, `used_pct < threshold`. IEEE-754
/// makes every comparison against NaN false, so in apr 0.63.0 `--tol-abs nan`,
/// `--scaling-floor nan`, `--preempt-threshold nan`, `--tolerance nan` and
/// `--epsilon nan` each made the failing branch unreachable: the command exited
/// 0 AND the report printed a positive `Ok` next to the violating number, e.g.
/// `scaling_efficiency : Ok { efficiency: 0.0025 }` for a 0.25%-efficient DDP
/// run. A CI log scraper reading that gets the wrong answer.
///
/// Each case runs twice against the SAME observation file: once with the
/// shipped defaults (must fail — proving the body is genuinely bad and the gate
/// works) and once with the disarming value (must also fail).
#[test]
fn nan_threshold_never_reports_a_failing_lint_gate_as_ok() {
    let dir = tempfile::tempdir().expect("tempdir");
    let p = |name: &str, body: &str| -> String {
        let path = dir.path().join(name);
        std::fs::write(&path, body).expect("write fixture");
        path.display().to_string()
    };

    let kv = p(
        "kv.json",
        r#"{"block_size_tokens":16,"total_blocks":100,"peak_used_pct":0.1,"preemption_count":3,
            "timeline":[{"step":0,"t_ms":1.0,"used_blocks":10,"free_blocks":90,
                         "used_pct":0.10,"active_seqs":1,"preempted_seqs":3}]}"#,
    );
    let parity = p("parity.json", r#"{"max_abs_diff":999.0,"cosine_sim":-0.5}"#);
    let attn = p("attn.json", "[[[[3.0,2.0],[0.5,0.5]]]]");
    let explain = p(
        "explain.jsonl",
        "{\"step\":0,\"sampled_id\":7,\"candidates\":[\
         {\"token_id\":7,\"pre_prob\":0.9,\"post_prob\":0.5,\"rank\":0},\
         {\"token_id\":3,\"pre_prob\":0.1,\"post_prob\":0.1,\"rank\":1}]}\n",
    );
    let ddp1 = p("ddp1.json", r#"{"tokens_per_sec":1000.0,"final_loss":2.0}"#);
    let ddpn = p(
        "ddpn.json",
        r#"{"tokens_per_sec":10.0,"final_loss":9.9,
            "ddp_metrics":{"allreduce_bandwidth_gbps":[12.5]}}"#,
    );

    // (args that fail at the defaults, the disarming values, the gate label
    //  that must never be reported as Ok)
    let cases: Vec<(Vec<String>, Vec<String>, &str)> = vec![
        (
            vec!["kv-timeline-lint".into(), "--timeline-file".into(), kv],
            vec!["--preempt-threshold".into(), "nan".into()],
            "preemption_trigger",
        ),
        (
            vec!["attn-parity-lint".into(), "--parity-file".into(), parity],
            vec![
                "--tol-abs".into(),
                "nan".into(),
                "--tol-cos".into(),
                "nan".into(),
            ],
            "parity_numerics",
        ),
        (
            vec!["attn-viz-lint".into(), "--attn-file".into(), attn],
            vec![
                "--tolerance".into(),
                "nan".into(),
                "--epsilon".into(),
                "nan".into(),
            ],
            "row_softmax",
        ),
        (
            vec!["explain-token-lint".into(), "--jsonl-file".into(), explain],
            vec!["--tolerance".into(), "nan".into()],
            "probs_normalize",
        ),
        (
            vec![
                "ddp-metrics-lint".into(),
                "--metrics-1gpu-file".into(),
                ddp1,
                "--metrics-ngpu-file".into(),
                ddpn,
                "--world-size".into(),
                "4".into(),
            ],
            vec![
                "--scaling-floor".into(),
                "nan".into(),
                "--loss-tolerance".into(),
                "nan".into(),
            ],
            "scaling_efficiency",
        ),
    ];

    for (base, disarm, gate) in cases {
        let control = apr_binary().args(&base).output().expect("run apr");
        assert!(
            !control.status.success(),
            "FALSIFY-CLI-THRESHOLD-NAN-001 control: `apr {}` must FAIL at the shipped \
             defaults, otherwise the disarm case below proves nothing.\nstdout:\n{}",
            base.join(" "),
            String::from_utf8_lossy(&control.stdout)
        );

        let mut args = base.clone();
        args.extend(disarm.iter().cloned());
        let out = apr_binary().args(&args).output().expect("run apr");
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            !out.status.success(),
            "FALSIFY-CLI-THRESHOLD-NAN-001: `apr {}` exited 0 — a NaN threshold disarmed \
             the {gate} gate on a body that fails at the defaults.\nstdout:\n{stdout}",
            args.join(" ")
        );
        assert!(
            !report_says_gate_is_ok(&stdout, gate),
            "FALSIFY-CLI-THRESHOLD-NAN-001: the report asserted `{gate} : Ok` for an \
             observation it never actually checked.\nstdout:\n{stdout}"
        );
    }
}

// ============================================================================
// FALSIFY-CLI-006: the DEPTH-2 surface must be locked, not just the top level
// ============================================================================
//
// SURF-13 (#2505): `registered_commands()` holds only top-level names -- none
// of them contains a space -- so of 238 invocable paths only 111 were gated.
// The other 127 could be renamed or deleted and every surface gate stayed
// green. 81 of those 127 are added by this branch's six consolidated sibling
// CLIs, which is why the lock lands with them rather than after them.
//
// Same shape as FALSIFY-CLI-002/005 one level down: the contract's
// `subcommands:` list and the built binary must agree in BOTH directions.

/// Parse the `Commands:` block out of one `--help` invocation.
fn help_subcommands(path: &[&str]) -> Vec<String> {
    let mut cmd = apr_binary();
    cmd.args(path).arg("--help");
    let out = cmd.output().expect("apr --help");
    let stdout = String::from_utf8_lossy(&out.stdout);
    help_block(&stdout)
        .filter_map(row_name)
        .filter(|name| *name != "help")
        .map(str::to_string)
        .collect()
}

/// `(parent, [children])` for every parent the CONTRACT declares as having a
/// `subcommands:` list.
fn contract_subcommands() -> Vec<(String, Vec<String>)> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../contracts/apr-cli-commands-v1.yaml"
    );
    let text = std::fs::read_to_string(path).expect("read the command contract");

    let mut out = Vec::new();
    let mut current: Option<String> = None;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("  - name: ") {
            current = Some(rest.trim().trim_matches('"').to_string());
        } else if let Some(rest) = line.strip_prefix("    subcommands: [") {
            let kids: Vec<String> = rest
                .trim_end_matches(']')
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            if let Some(parent) = current.clone() {
                out.push((parent, kids));
            }
        }
    }
    out
}

#[test]
fn every_declared_subcommand_exists_in_the_binary() {
    let declared = contract_subcommands();

    // Vacuity: a parse that found nothing would make the loop below pass
    // trivially -- which is exactly how a depth-2 gate reports green while
    // gating nothing.
    assert!(
        declared.len() >= 20,
        "only {} parents parsed from the contract; the parser is broken, not the tree",
        declared.len()
    );
    let total: usize = declared.iter().map(|(_, k)| k.len()).sum();
    // Vacuity floor, NOT an exact count -- it exists to catch a broken parser,
    // not to freeze the surface. Lowered 120 -> 105 when `apr qa-playbook` and
    // its 15 subcommands were removed (#2539: it routed into aprender-qa-cli,
    // which is `publish = false`, making apr-cli impossible to publish). The
    // real count went 128 -> 113; keep the floor comfortably below it so
    // ordinary command changes do not trip a check about parser health.
    assert!(
        total >= 105,
        "only {total} depth-2 paths parsed; the contract parser is broken"
    );

    for (parent, kids) in &declared {
        let actual = help_subcommands(&[parent.as_str()]);
        let missing: Vec<&String> = kids.iter().filter(|k| !actual.contains(k)).collect();
        assert!(
            missing.is_empty(),
            "FALSIFY-CLI-006: contract declares `apr {parent} {missing:?}` but the \
             binary does not offer them.\nbinary has: {actual:?}"
        );
    }
}

#[test]
fn every_subcommand_in_the_binary_is_declared() {
    let declared: std::collections::HashMap<String, Vec<String>> =
        contract_subcommands().into_iter().collect();

    let mut undeclared: Vec<String> = Vec::new();
    let mut seen = 0usize;
    for parent in registered_commands() {
        let actual = help_subcommands(&[parent]);
        if actual.is_empty() {
            continue;
        }
        seen += actual.len();
        let empty = Vec::new();
        let kids = declared.get(parent).unwrap_or(&empty);
        for a in &actual {
            if !kids.contains(a) {
                undeclared.push(format!("{parent} {a}"));
            }
        }
    }

    // Vacuity companion: if no parent reported children, "nothing undeclared"
    // would be true and meaningless.
    assert!(
        seen >= 105,
        "only {seen} depth-2 paths seen in the binary; the help parser is broken"
    );
    assert!(
        undeclared.is_empty(),
        "FALSIFY-CLI-006: the binary offers depth-2 commands the contract does not \
         declare: {undeclared:?}\nAdd them to contracts/apr-cli-commands-v1.yaml \
         under their parent's `subcommands:`."
    );
}

// ---------------------------------------------------------------------------
// Issue #2607 — `apr code` with no arguments and stdin closed
//
// Measured on the published 0.63.0 (dogfood sweep): `apr code </dev/null`
// printed no help. It scanned the filesystem, picked the largest local GGUF
// (a 30 B MoE), spawned an `apr serve` child for it, and exited — leaving the
// server running. Two defects: a no-argument invocation took a consequential
// action chosen by looking at the disk, and the child it spawned outlived the
// parent.
//
// The load-bearing assertion below is the ABSENCE of the child. The help text
// is the easy half; a run that prints help and still launches a server has
// not fixed anything.
// ---------------------------------------------------------------------------

/// A stand-in for the `apr serve` backend. `apr code` launches its inference
/// server through `$APR_BIN` (aprender#2384), so pointing that at this script
/// makes the spawn observable without a real model or a real server: the
/// script records its own pid and then blocks, exactly as a live server would.
#[cfg(unix)]
fn write_fake_serve_script(dir: &std::path::Path, pidfile: &std::path::Path) -> std::path::PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let script = dir.join("fake-apr-serve.sh");
    std::fs::write(
        &script,
        // fds are detached so the fake server cannot hold the test harness'
        // pipes open and stretch a failing run out to the full sleep.
        format!(
            "#!/bin/sh\necho $$ > '{}'\nexec sleep 30 >/dev/null 2>&1 </dev/null\n",
            pidfile.display()
        ),
    )
    .expect("write fake serve script");
    let mut perms = std::fs::metadata(&script)
        .expect("stat script")
        .permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&script, perms).expect("chmod script");
    script
}

/// A stand-in for `apr serve` that is **reachable** and **SIGTERM-deaf**.
///
/// Two properties, and both are load-bearing.
///
/// *Reachable*: `AprServeDriver::wait_for_ready` returns `Ok` only once
/// something answers a TCP connect on the `--port` it was given, and until it
/// does, `apr code` never reaches the `-p` branch where the child is
/// released. So the script records **both** its pid and the port it was told
/// to use; the test binds that port itself.
///
/// *SIGTERM-deaf*: `PR_SET_PDEATHSIG` (aprender#1712) asks the kernel to send
/// the child **one** `SIGTERM` when the parent dies, and does not escalate. A
/// child that ignores `SIGTERM` is therefore reaped only by
/// `AprServeDriver::drop`, which sends `SIGTERM`, waits 2s, and then
/// `SIGKILL`s — and `drop` runs only if every owner of the driver `Arc` was
/// released before `std::process::exit`. Measured: with a SIGTERM-*obeying*
/// fixture, reverting `release_driver` back to the pre-fix `drop(driver)`
/// leaves this test GREEN, because `PR_SET_PDEATHSIG` reaps the child on
/// Linux no matter what the process did on the way out. The whole assertion
/// would have been theater. Ignoring `SIGTERM` removes that mask, so the test
/// observes what the fix actually changed — and it is not an artificial
/// shape: the 2s-then-`SIGKILL` escalation exists in `Drop` precisely because
/// a real server can be slow or deaf to `SIGTERM`.
#[cfg(unix)]
fn write_reachable_serve_script(
    dir: &std::path::Path,
    handshake: &std::path::Path,
) -> std::path::PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let script = dir.join("reachable-apr-serve.sh");
    std::fs::write(
        &script,
        format!(
            "#!/bin/sh\n\
             trap '' TERM\n\
             port=''\n\
             prev=''\n\
             for a in \"$@\"; do\n\
             \tif [ \"$prev\" = \"--port\" ]; then port=\"$a\"; fi\n\
             \tprev=\"$a\"\n\
             done\n\
             printf '%s %s\\n' \"$$\" \"$port\" > '{}'\n\
             while : ; do sleep 1; done\n",
            handshake.display()
        ),
    )
    .expect("write reachable serve script");
    let mut perms = std::fs::metadata(&script)
        .expect("stat script")
        .permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&script, perms).expect("chmod script");
    script
}

/// Answer every connection with a minimal OpenAI chat-completion, so a
/// `-p` turn completes and `apr code` reaches its exit path.
#[cfg(unix)]
fn answer_completions_forever(listener: std::net::TcpListener) {
    use std::io::{Read, Write};
    const BODY: &str = concat!(
        r#"{"choices":[{"message":{"role":"assistant","content":"ok"},"#,
        r#""finish_reason":"stop"}],"#,
        r#""usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#
    );
    for stream in listener.incoming() {
        let Ok(mut sock) = stream else { break };
        std::thread::spawn(move || {
            let _ = sock.set_read_timeout(Some(std::time::Duration::from_millis(500)));
            let mut buf = [0u8; 8192];
            let _ = sock.read(&mut buf);
            let resp = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\
                 content-length: {}\r\nconnection: close\r\n\r\n{BODY}",
                BODY.len()
            );
            let _ = sock.write_all(resp.as_bytes());
            let _ = sock.flush();
        });
    }
}

#[cfg(unix)]
fn pid_is_alive(pid: i32) -> bool {
    // `kill -0` probes for existence without signalling.
    std::process::Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[test]
#[cfg(unix)]
fn falsify_2607_bare_apr_code_with_closed_stdin_spawns_no_serve_child() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let home = tmp.path();

    // A model for auto-discovery to find. `ModelConfig::discover_model()`
    // scans `$HOME/.apr/models` and `./models`; before the fix, finding this
    // was enough to make `apr code` launch a server for it.
    let models = home.join("models");
    std::fs::create_dir_all(&models).expect("create ./models");
    std::fs::write(models.join("fake-30b-moe.gguf"), b"GGUF").expect("write fake gguf");
    std::fs::create_dir_all(home.join(".apr").join("models")).expect("create ~/.apr/models");
    std::fs::write(
        home.join(".apr").join("models").join("fake-30b-moe.gguf"),
        b"GGUF",
    )
    .expect("write fake gguf in HOME");

    let pidfile = home.join("serve.pid");
    let fake_serve = write_fake_serve_script(home, &pidfile);

    let output = apr_binary()
        .arg("code")
        .current_dir(home)
        .env("HOME", home)
        .env("XDG_CONFIG_HOME", home.join(".config"))
        .env("APR_BIN", &fake_serve)
        // Bound the pre-fix path: without this the health poll waits ~30s for
        // a server that will never answer.
        .env("APR_SERVE_READY_TIMEOUT_S", "1")
        .stdin(std::process::Stdio::null())
        .output()
        .expect("run apr code");

    // If the child was spawned at all, report whether it is still alive —
    // an orphan outliving the parent is the worst form of this defect, and a
    // reaped one still means a bare invocation launched an inference server.
    let spawned = std::fs::read_to_string(&pidfile).ok();
    if let Some(ref raw) = spawned {
        if let Ok(pid) = raw.trim().parse::<i32>() {
            let alive = pid_is_alive(pid);
            if alive {
                // Do not leave it running for the rest of the suite.
                let _ = std::process::Command::new("kill")
                    .args(["-9", &pid.to_string()])
                    .status();
            }
            panic!(
                "#2607: bare `apr code` with stdin closed spawned an inference server \
                 (pid {pid}, still alive after the parent exited: {alive}). \
                 A no-argument invocation must print help, not pick a model off the disk."
            );
        }
    }
    assert!(
        spawned.is_none(),
        "#2607: bare `apr code` with stdin closed spawned a serve child ({spawned:?})"
    );

    // The easy half, asserted second: it must actually say how to use it, and
    // it must not exit 0 — a script must not read "I did nothing" as success.
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Usage:"),
        "#2607: bare `apr code` must print help. stderr was:\n{stderr}"
    );
    assert!(
        stderr.contains("stdin is not a terminal"),
        "#2607: help must say WHY nothing ran. stderr was:\n{stderr}"
    );
    assert_eq!(
        output.status.code(),
        Some(2),
        "#2607: a usage refusal exits 2 (clap's usage-error code), not 0. stderr:\n{stderr}"
    );
}

// ---------------------------------------------------------------------------
// Issue #2607, second call site.
//
// The falsifier above covers the BARE invocation: no child may be spawned at
// all. It says nothing about the invocation that legitimately spawns one.
// `apr code -p "..."` discovers a model, launches `apr serve`, answers the
// prompt, and then ends in `std::process::exit` — which runs no destructors.
// That is the call site where the orphan was actually observed, and it was
// only covered at unit level (`release_driver`'s Arc ordering, with a fake
// driver). This covers it live, against the real binary and a real child.
// ---------------------------------------------------------------------------

/// `apr code -p "..."` must leave no `apr serve` child behind.
///
/// The fixture is deliberately *reachable*, so the run goes all the way
/// through `release_driver` + `process::exit` rather than bailing out in
/// `AprServeDriver::launch`'s error path, and deliberately *SIGTERM-deaf*, so
/// `PR_SET_PDEATHSIG` cannot reap the child on the parent's behalf and mask
/// the very defect this asserts (see `write_reachable_serve_script`). Both
/// halves are checked — that the child really was spawned (otherwise the
/// absence check is vacuous) and that it did not outlive the parent.
#[test]
#[cfg(unix)]
fn falsify_2607_non_interactive_p_run_leaves_no_serve_child() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let home = tmp.path();

    // A model for auto-discovery. `-p` is an explicit instruction, so unlike
    // the bare case this run is SUPPOSED to launch a server for it.
    let models = home.join("models");
    std::fs::create_dir_all(&models).expect("create ./models");
    std::fs::write(models.join("fake-30b-moe.gguf"), b"GGUF").expect("write fake gguf");

    let handshake = home.join("serve.handshake");
    let fake_serve = write_reachable_serve_script(home, &handshake);

    let mut child = apr_binary()
        .args(["code", "-p", "hi"])
        .current_dir(home)
        .env("HOME", home)
        .env("XDG_CONFIG_HOME", home.join(".config"))
        .env("APR_BIN", &fake_serve)
        .env("APR_SERVE_READY_TIMEOUT_S", "20")
        // Bound the HTTP turn: the default is 1800s.
        .env("APR_AGENT_HTTP_TIMEOUT_S", "20")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn apr code -p");

    // Wait for the fake server to announce its pid and the port it was told
    // to listen on, then bind that port so the readiness check can pass.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let (serve_pid, port) = loop {
        if let Ok(raw) = std::fs::read_to_string(&handshake) {
            let mut parts = raw.split_whitespace();
            if let (Some(pid), Some(port)) = (parts.next(), parts.next()) {
                if let (Ok(pid), Ok(port)) = (pid.parse::<i32>(), port.parse::<u16>()) {
                    break (pid, port);
                }
            }
        }
        assert!(
            std::time::Instant::now() < deadline,
            "#2607: `apr code -p` never spawned its inference backend — the fixture is not \
             exercising the path this test exists for"
        );
        std::thread::sleep(std::time::Duration::from_millis(20));
    };

    let listener = std::net::TcpListener::bind(("127.0.0.1", port)).unwrap_or_else(|e| {
        let _ = child.kill();
        let _ = std::process::Command::new("kill")
            .args(["-9", &serve_pid.to_string()])
            .status();
        panic!(
            "#2607: could not bind 127.0.0.1:{port} to stand in for `apr serve` ({e}); \
             another process is holding the port `apr code` derived from its own pid"
        )
    });
    std::thread::spawn(move || answer_completions_forever(listener));

    let output = child.wait_with_output().expect("wait for apr code");

    // Reap unconditionally before asserting, so a failure does not leak a
    // process into the rest of the suite.
    let alive = pid_is_alive(serve_pid);
    if alive {
        let _ = std::process::Command::new("kill")
            .args(["-9", &serve_pid.to_string()])
            .status();
    }

    assert!(
        !alive,
        "#2607: `apr code -p` exited leaving its `apr serve` child (pid {serve_pid}) running. \
         The `-p` branch ends in std::process::exit, which runs no destructors, so every owner \
         of the driver Arc must be released explicitly first. stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// #2607 follow-up: `echo "hi" | apr code` must NOT be refused.
///
/// `run_repl` reads stdin line by line and treats EOF as `/exit`, so a pipe
/// carrying a line has always been one REPL turn. The #2607 guard keys on
/// "no argument AND no interactive session", and reading that as
/// `!stdin.is_terminal()` alone would silently convert this working
/// invocation into an exit-2 usage error. The refusal is for stdin that can
/// never deliver a byte (`/dev/null`, a closed fd), not for a pipe with a
/// prompt in it.
///
/// Asserted against an empty HOME and cwd, so the run has no model to find
/// and stops at `NO_MODEL` — which is proof it got past the guard, all the
/// way to model resolution, without spawning anything.
#[test]
#[cfg(unix)]
fn falsify_2607_piped_prompt_is_not_refused_as_a_bare_invocation() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let home = tmp.path();

    let mut child = apr_binary()
        .arg("code")
        .current_dir(home)
        .env("HOME", home)
        .env("XDG_CONFIG_HOME", home.join(".config"))
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn apr code with a pipe");
    {
        use std::io::Write;
        let mut stdin = child.stdin.take().expect("piped stdin");
        stdin.write_all(b"hi\n").expect("write piped prompt");
        // Leave it open long enough that the peek cannot be answered by EOF.
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    let output = child.wait_with_output().expect("wait for apr code");
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        !stderr.contains("nothing to do"),
        "#2607 follow-up: `echo \"hi\" | apr code` carries an instruction and must not be \
         refused as a bare invocation. stderr:\n{stderr}"
    );
    assert_ne!(
        output.status.code(),
        Some(2),
        "#2607 follow-up: a piped prompt must not exit with the usage-error code. \
         stderr:\n{stderr}"
    );
    assert_eq!(
        output.status.code(),
        Some(5),
        "#2607 follow-up: with a prompt on stdin and no model anywhere, the run must reach \
         model resolution and stop at NO_MODEL — proof the guard let it through. stderr:\n{stderr}"
    );
}
