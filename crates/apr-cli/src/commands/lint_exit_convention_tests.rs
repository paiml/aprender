//! Falsifier for the `apr *-lint` family exit-code convention (dogfood #2404).
//!
//! Before this, identical failure conditions produced five different exit codes
//! across sibling commands documented for the same CI harness: a missing input
//! exited 1 on the ten `Result<(), String>` linters and 3 on the rest; bad JSON
//! exited 1 vs 4; a gate rejection exited 1 vs 5; and `hang-trace-lint` handed
//! back 7 (an OS error) when `--trace-dir` named a regular file.
//!
//! These tests assert the BEHAVIOUR a CI wrapper depends on:
//!   * one condition -> one exit code, family-wide;
//!   * the three conditions a wrapper must tell apart (missing input,
//!     unusable input, gate rejection) map to three DIFFERENT codes;
//!   * no lint calls a captured JSON observation an "APR format" problem.

use std::path::{Path, PathBuf};

use crate::error::CliError;

type Invoke = fn(&Path) -> std::result::Result<(), CliError>;

struct Member {
    name: &'static str,
    /// Invoke the lint with `path` as its primary input.
    invoke: Invoke,
    /// True when the primary input is parsed as a single JSON document, so a
    /// non-JSON body is an unusable-input failure rather than a gate outcome.
    json_input: bool,
}

fn s(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

/// Every `*-lint` command that takes a captured-observation path.
fn family() -> Vec<Member> {
    vec![
        // --- the ten that used to return Result<(), String> and exit 1 ---
        Member {
            name: "awq-lint",
            json_input: true,
            invoke: |p| {
                super::awq_lint::run(super::awq_lint::AwqLintArgs {
                    observation_file: s(p),
                    json: false,
                })
                .map_err(CliError::from)
            },
        },
        Member {
            name: "gptq-lint",
            json_input: true,
            invoke: |p| {
                super::gptq_lint::run(super::gptq_lint::GptqLintArgs {
                    observation_file: s(p),
                    json: false,
                })
                .map_err(CliError::from)
            },
        },
        Member {
            name: "fp8-lint",
            json_input: true,
            invoke: |p| {
                super::fp8_lint::run(super::fp8_lint::Fp8LintArgs {
                    observation_file: s(p),
                    json: false,
                })
                .map_err(CliError::from)
            },
        },
        Member {
            name: "imatrix-lint",
            json_input: true,
            invoke: |p| {
                super::imatrix_lint::run(super::imatrix_lint::ImatrixLintArgs {
                    observation_file: s(p),
                    json: false,
                })
                .map_err(CliError::from)
            },
        },
        Member {
            name: "embeddings-lint",
            json_input: true,
            invoke: |p| {
                super::embeddings_lint::run(super::embeddings_lint::EmbeddingsLintArgs {
                    observation_file: s(p),
                    json: false,
                })
                .map_err(CliError::from)
            },
        },
        Member {
            name: "nf4-lint",
            json_input: true,
            invoke: |p| {
                super::nf4_lint::run(super::nf4_lint::Nf4LintArgs {
                    observation_file: s(p),
                    json: false,
                })
                .map_err(CliError::from)
            },
        },
        Member {
            name: "registry-quota-lint",
            json_input: true,
            invoke: |p| {
                super::registry_quota_lint::run(super::registry_quota_lint::RegistryQuotaLintArgs {
                    observation_file: s(p),
                    json: false,
                })
                .map_err(CliError::from)
            },
        },
        Member {
            name: "rm-gc-lint",
            json_input: true,
            invoke: |p| {
                super::rm_gc_lint::run(super::rm_gc_lint::RmGcLintArgs {
                    observation_file: s(p),
                    json: false,
                })
                .map_err(CliError::from)
            },
        },
        Member {
            name: "shared-cache-lint",
            json_input: true,
            invoke: |p| {
                super::shared_cache_lint::run(super::shared_cache_lint::SharedCacheLintArgs {
                    observation_file: s(p),
                    json: false,
                })
                .map_err(CliError::from)
            },
        },
        Member {
            name: "unified-search-lint",
            json_input: true,
            invoke: |p| {
                super::unified_search_lint::run(super::unified_search_lint::UnifiedSearchLintArgs {
                    observation_file: s(p),
                    json: false,
                })
                .map_err(CliError::from)
            },
        },
        // --- the CliError half ---
        Member {
            name: "kv-timeline-lint",
            json_input: true,
            invoke: |p| super::kv_timeline_lint::run(p, 0.9, false),
        },
        Member {
            name: "tool-use-lint",
            json_input: true,
            invoke: |p| super::tool_use_lint::run(p, false),
        },
        Member {
            name: "gbnf-lint",
            json_input: true,
            invoke: |p| super::gbnf_lint::run(p, false),
        },
        Member {
            name: "dry-sampling-lint",
            json_input: true,
            invoke: |p| super::dry_sampling_lint::run(p, false),
        },
        Member {
            name: "gpu-memtrace-lint",
            json_input: true,
            invoke: |p| super::gpu_memtrace_lint::run(p, false),
        },
        Member {
            name: "typical-p-lint",
            json_input: true,
            invoke: |p| super::typical_p_lint::run(p, false),
        },
        Member {
            name: "react-trace-lint",
            json_input: true,
            invoke: |p| super::react_trace_lint::run(p, None, false, false),
        },
        Member {
            name: "nccl-diag-lint",
            json_input: true,
            invoke: |p| super::nccl_diag_lint::run(p, None, false, false),
        },
        Member {
            name: "otlp-lint",
            json_input: true,
            invoke: |p| super::otlp_lint::run(p, false, false, None, false),
        },
        Member {
            name: "oom-lint",
            json_input: true,
            invoke: |p| super::oom_lint::run(p, None, false),
        },
        Member {
            name: "audio-inspect-lint",
            json_input: true,
            invoke: |p| super::audio_inspect_lint::run(p, None, None, false),
        },
        Member {
            name: "ollama-chat-lint",
            json_input: true,
            invoke: |p| super::ollama_chat::run(p, false, false),
        },
        Member {
            name: "ollama-tools-lint",
            json_input: true,
            invoke: |p| super::ollama_tools_lint::run(p, None, false, false),
        },
        Member {
            name: "ddp-metrics-lint",
            json_input: true,
            invoke: |p| super::ddp_metrics_lint::run(p, p, 2, 0.7, 0.1, false),
        },
        Member {
            name: "attn-viz-lint",
            json_input: true,
            invoke: |p| super::attn_viz_lint::run(Some(p), None, 1, 0.05, 1e-6, false),
        },
        Member {
            name: "prometheus-lint",
            json_input: false,
            invoke: |p| super::prometheus_lint::run(p, None, false, false),
        },
        Member {
            name: "embed-viz-lint",
            json_input: false,
            invoke: |p| super::embed_viz_lint::run(p, None, None, false),
        },
        Member {
            name: "explain-token-lint",
            json_input: false,
            invoke: |p| super::explain_token_lint::run(p, 1e-3, false, false),
        },
        Member {
            name: "check-finite-lint",
            json_input: false,
            invoke: |p| super::check_finite_lint::run(Some(p), None, 1, false),
        },
    ]
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "apr-lint-exit-{tag}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

#[test]
fn a_missing_input_exits_3_for_every_lint_in_the_family() {
    let missing = PathBuf::from("/nonexistent/apr-lint-family/observation.json");
    let mut wrong = Vec::new();
    for m in family() {
        let err = (m.invoke)(&missing).expect_err("a missing input must be an error");
        if err.exit_code_value() != 3 {
            wrong.push(format!(
                "{}: exit {} ({err})",
                m.name,
                err.exit_code_value()
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "missing input must exit 3 family-wide; offenders: {wrong:#?}"
    );
}

#[test]
fn an_unparseable_observation_exits_4_for_every_json_lint() {
    let dir = scratch("badjson");
    let f = dir.join("bad.json");
    std::fs::write(&f, "not json at all\n").expect("write");
    let mut wrong = Vec::new();
    for m in family().into_iter().filter(|m| m.json_input) {
        let err = (m.invoke)(&f).expect_err("a non-JSON observation must be an error");
        if err.exit_code_value() != 4 {
            wrong.push(format!(
                "{}: exit {} ({err})",
                m.name,
                err.exit_code_value()
            ));
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        wrong.is_empty(),
        "an unparseable observation must exit 4 family-wide; offenders: {wrong:#?}"
    );
}

#[test]
fn no_lint_blames_the_apr_format_for_a_captured_json_observation() {
    let dir = scratch("notapr");
    let f = dir.join("bad.json");
    std::fs::write(&f, "not json at all\n").expect("write");
    let mut liars = Vec::new();
    for m in family().into_iter().filter(|m| m.json_input) {
        let rendered = (m.invoke)(&f)
            .expect_err("a non-JSON observation must be an error")
            .to_string();
        if rendered.contains("Invalid APR format") {
            liars.push(format!("{}: {rendered}", m.name));
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        liars.is_empty(),
        "these lints read a captured JSON observation, never an APR model: {liars:#?}"
    );
}

/// The property a CI wrapper actually needs: for one command, the three
/// conditions it must branch on land on three different exit codes.
#[test]
fn missing_unusable_and_gate_failure_are_three_distinct_exit_codes() {
    let dir = scratch("distinct");
    let bad = dir.join("bad.json");
    std::fs::write(&bad, "not json at all\n").expect("write");

    // A well-formed AWQ observation whose quality gate must fail.
    let failing = dir.join("awq-fail.json");
    std::fs::write(
        &failing,
        r#"{"quality":{"p_fp16":0.90,"p_awq":0.10,"threshold":0.80}}"#,
    )
    .expect("write");

    let run = |p: &Path| {
        super::awq_lint::run(super::awq_lint::AwqLintArgs {
            observation_file: s(p),
            json: false,
        })
        .map_err(CliError::from)
    };

    let missing = run(Path::new("/nonexistent/apr-lint-family/x.json"))
        .expect_err("missing")
        .exit_code_value();
    let unusable = run(&bad).expect_err("unusable").exit_code_value();
    let gate = run(&failing).expect_err("gate").exit_code_value();
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!((missing, unusable, gate), (3, 4, 5));
    assert_ne!(missing, unusable);
    assert_ne!(unusable, gate);
    assert_ne!(missing, gate);
}

/// The same three conditions on the other half of the family must agree with
/// the numbers above — that is the whole point of the convention.
#[test]
fn the_cli_error_half_uses_the_same_three_codes() {
    let dir = scratch("otherhalf");
    let bad = dir.join("bad.json");
    std::fs::write(&bad, "not json at all\n").expect("write");
    let failing = dir.join("kv-fail.json");
    std::fs::write(&failing, r#"{"not_a_timeline":true}"#).expect("write");

    let missing =
        super::kv_timeline_lint::run(Path::new("/nonexistent/apr-lint-family/x.json"), 0.9, false)
            .expect_err("missing")
            .exit_code_value();
    let unusable = super::kv_timeline_lint::run(&bad, 0.9, false)
        .expect_err("unusable")
        .exit_code_value();
    let gate = super::kv_timeline_lint::run(&failing, 0.9, false)
        .expect_err("gate")
        .exit_code_value();
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!((missing, unusable, gate), (3, 4, 5));
}

#[test]
fn a_gate_rejection_exits_5_not_1_on_the_formerly_string_returning_lints() {
    let dir = scratch("gate5");
    let f = dir.join("quota.json");
    // registry-quota-lint used to collapse a real contract violation onto
    // exit 1, the same code it used for "your capture step wrote no file".
    std::fs::write(
        &f,
        r#"{"quota":{"limit_bytes":100,"used_bytes":10,"incoming_bytes":1000}}"#,
    )
    .expect("write");
    let err = super::registry_quota_lint::run(super::registry_quota_lint::RegistryQuotaLintArgs {
        observation_file: s(&f),
        json: false,
    })
    .map_err(CliError::from)
    .expect_err("over-quota write must be rejected");
    let code = err.exit_code_value();
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(code, 5, "gate rejection must exit 5, got {code}: {err}");
}

#[test]
fn hang_trace_lint_rejects_a_regular_file_as_an_input_error_not_an_os_error() {
    let dir = scratch("hangdir");
    let f = dir.join("not-a-dir");
    std::fs::write(&f, "x").expect("write");
    let err = super::hang_trace_lint::run(
        &f,
        super::hang_trace_lint::HangMode::Success,
        1,
        None,
        None,
        false,
    )
    .expect_err("a regular file is not a trace dir");
    let code = err.exit_code_value();
    let rendered = err.to_string();
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(code, 4, "got exit {code}: {rendered}");
    assert!(
        !rendered.contains("os error"),
        "the user gets a raw OS error instead of a diagnostic: {rendered}"
    );
    assert!(rendered.contains("not a directory"), "got: {rendered}");
}
