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
/// Feature-gated commands are conditionally included.
fn registered_commands() -> Vec<&'static str> {
    #[cfg_attr(not(feature = "code"), allow(unused_mut))]
    let mut cmds = vec![
        "run",
        "serve",
        "chat",
        "inspect",
        "debug",
        "validate",
        "validate-manifest",
        "lint",
        "explain",
        "tensors",
        "trace",
        "diff",
        "hex",
        "tree",
        "flow",
        "export",
        "import",
        "convert",
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
        "diagnose",
        "ollama-chat-lint",
        "dry-sampling-lint",
        "awq-lint",
        "oom-lint",
        "tool-use-lint",
        "gbnf-lint",
        "typical-p-lint",
        "speculative-lint",
        "oracle",
        "encrypt",
        "decrypt",
        "mcp",
    ];
    // Feature-gated commands — only expected when feature is enabled
    #[cfg(feature = "code")]
    cmds.push("code");
    cmds
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
