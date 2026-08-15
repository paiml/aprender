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
        "probar",
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
        "qa-playbook",
    ]
}

fn apr_binary() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_apr"));
    cmd.env("NO_COLOR", "1");
    cmd
}

fn get_help_commands() -> Vec<String> {
    let output = apr_binary()
        .arg("--help")
        .output()
        .expect("failed to run apr --help");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut commands = Vec::new();
    let mut in_commands = false;

    for line in stdout.lines() {
        if line.starts_with("Commands:") {
            in_commands = true;
            continue;
        }
        if in_commands {
            if line.starts_with("Options:") || line.is_empty() && commands.len() > 5 {
                break;
            }
            // Command rows have exactly 2-space indent (`  cmd  description...`).
            // Wrapped description continuation lines have a much wider indent
            // (column-aligned to the description start, typically 20+ spaces).
            // Filter on exact 2-space indent to avoid picking up the first
            // word of a wrap continuation as a "command name" (CRUX-B-19).
            let leading_spaces = line.chars().take_while(|c| *c == ' ').count();
            if leading_spaces != 2 {
                continue;
            }
            let trimmed = line.trim();
            if let Some(cmd_name) = trimmed.split_whitespace().next() {
                // Valid command names: lowercase, may contain hyphens, no parens/uppercase
                if !cmd_name.is_empty()
                    && cmd_name
                        .chars()
                        .next()
                        .map_or(false, |c| c.is_ascii_lowercase())
                    && !cmd_name.contains('(')
                    && !cmd_name.contains(')')
                {
                    commands.push(cmd_name.to_string());
                }
            }
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
