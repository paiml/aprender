//! Unit tests for `pretrain` (extracted from `pretrain.rs` to keep file-size invariant).
//!
//! Included via `#[cfg(test)] #[path = "pretrain_tests.rs"] mod tests;` in the parent.

use super::*;
use entrenar::models::llama_370m::Llama370MConfig;
use tempfile::TempDir;

/// Stage a `vocab.json` with exactly `n` distinct integer-string tokens at
/// `<dir>/vocab.json`. Used by pre-flight gate tests + by other tests that
/// need to get PAST the GATE-ARCH-370M-011 pre-flight to exercise a later
/// failure mode (e.g. empty dataset shards).
fn stage_vocab_json(dir: &std::path::Path, n: usize) {
    std::fs::create_dir_all(dir).expect("mkdir tokenizer dir");
    let mut obj = serde_json::Map::with_capacity(n);
    for i in 0..n {
        obj.insert(format!("t{i}"), serde_json::Value::from(i as u64));
    }
    let json = serde_json::to_string(&obj).expect("serialize");
    std::fs::write(dir.join("vocab.json"), json).expect("write vocab.json");
}

#[test]
fn preflight_accepts_matching_vocab() {
    // GATE-ARCH-370M-011 acceptance case: tokenizer vocab.json with
    // exactly Llama370MConfig::VOCAB_SIZE entries must pass pre-flight.
    let tmp = TempDir::new().expect("tempdir");
    stage_vocab_json(tmp.path(), Llama370MConfig::VOCAB_SIZE);
    preflight_tokenizer_vocab_matches_model(tmp.path())
        .expect("matching vocab must pass GATE-ARCH-370M-011");
}

#[test]
fn preflight_rejects_tokenizer_vocab_mismatch() {
    // FALSIFY-ARCH-370M-011: a tokenizer whose vocab size drifts from
    // the model's pinned VOCAB_SIZE MUST abort dispatch with an error
    // message that names both values and the gate id, so the operator
    // can see the mismatch without stepping through code. Task #131
    // bumped VOCAB_SIZE to 50_257 (Option A) — the counter-example
    // below now exercises a tokenizer one token short of contract.
    let tmp = TempDir::new().expect("tempdir");
    let mismatch = Llama370MConfig::VOCAB_SIZE - 1;
    stage_vocab_json(tmp.path(), mismatch);
    let err = preflight_tokenizer_vocab_matches_model(tmp.path())
        .expect_err("tokenizer/model vocab mismatch must be rejected");
    match err {
        CliError::ValidationFailed(msg) => {
            assert!(
                msg.contains("GATE-ARCH-370M-011"),
                "msg must cite gate: {msg}"
            );
            assert!(
                msg.contains(&mismatch.to_string()),
                "msg must name tokenizer vocab: {msg}"
            );
            assert!(
                msg.contains(&Llama370MConfig::VOCAB_SIZE.to_string()),
                "msg must name model vocab: {msg}"
            );
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn preflight_rejects_missing_vocab_json() {
    // Missing vocab.json is a pre-flight failure (not a later shard
    // error) — the operator should know the tokenizer layout is
    // wrong, not that the dataset is empty.
    let tmp = TempDir::new().expect("tempdir");
    let err = preflight_tokenizer_vocab_matches_model(tmp.path())
        .expect_err("missing vocab.json must be rejected");
    match err {
        CliError::ValidationFailed(msg) => {
            assert!(
                msg.contains("GATE-ARCH-370M-011"),
                "msg must cite gate: {msg}"
            );
            assert!(
                msg.contains("cannot read"),
                "msg must name I/O failure: {msg}"
            );
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

// ── FALSIFY-CORPUS-004 / GATE-CORPUS-PREFLIGHT falsifiers ──────────
// Contract: pretraining-corpus-v1 v2.0.0 §FALSIFY-CORPUS-004.
//
// These tests pin the pre-flight dispatch-budget gate: dispatch
// MUST refuse when planned_tokens > corpus total_tokens unless
// the operator passes `--allow-shard-cycle`. Together with
// GATE-TRAIN-EXHAUST they close the task #141 silent `(1.0, 1.0)`
// placeholder loophole.

/// Stage a directory holding `num_shards` `.bin` files, each
/// containing `tokens_per_shard` u32 tokens (4 bytes each). Returns
/// the total token count the pre-flight will observe.
fn stage_shard_dir(dir: &std::path::Path, num_shards: usize, tokens_per_shard: u32) -> u64 {
    std::fs::create_dir_all(dir).expect("mkdir shard dir");
    for s in 0..num_shards {
        let path = dir.join(format!("shard-{s}.bin"));
        let mut bytes = Vec::with_capacity(tokens_per_shard as usize * 4);
        for t in 0..tokens_per_shard {
            bytes.extend_from_slice(&t.to_le_bytes());
        }
        std::fs::write(&path, bytes).expect("write shard");
    }
    (num_shards as u64) * (tokens_per_shard as u64)
}

#[test]
fn preflight_budget_accepts_within_budget() {
    // planned = 2 × 2 × 4 = 16; corpus = 4 × 100 = 400. 16 ≤ 400.
    // Pre-flight must succeed and return both numbers unchanged.
    let tmp = TempDir::new().expect("tempdir");
    let total = stage_shard_dir(tmp.path(), 4, 100);
    let (planned, seen_total) = preflight_dispatch_budget(tmp.path(), 2, 2, 4, false)
        .expect("within budget must pass pre-flight");
    assert_eq!(planned, 16, "planned = num_steps × batch × seq");
    assert_eq!(seen_total, total, "total_tokens must match corpus size");
}

#[test]
fn preflight_budget_rejects_over_budget_without_opt_in() {
    // planned = 50 × 4 × 8 = 1600; corpus = 2 × 100 = 200.
    // 1600 > 200 AND --allow-shard-cycle absent → hard refusal.
    let tmp = TempDir::new().expect("tempdir");
    stage_shard_dir(tmp.path(), 2, 100);
    let err = preflight_dispatch_budget(tmp.path(), 50, 4, 8, false)
        .expect_err("over-budget without --allow-shard-cycle must refuse");
    match err {
        CliError::ValidationFailed(msg) => {
            assert!(
                msg.contains("GATE-CORPUS-PREFLIGHT"),
                "msg must cite gate id: {msg}"
            );
            assert!(msg.contains("1600"), "msg must name planned_tokens: {msg}");
            assert!(
                msg.contains("200"),
                "msg must name corpus total_tokens: {msg}"
            );
            assert!(
                msg.contains("--allow-shard-cycle"),
                "msg must name the opt-in flag: {msg}"
            );
            assert!(
                msg.contains("FALSIFY-CORPUS-004"),
                "msg must cite the contract: {msg}"
            );
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn preflight_budget_accepts_over_budget_with_opt_in() {
    // planned 1600 > total 200, but --allow-shard-cycle is on so the
    // pre-flight returns Ok — the cycling iterator handles the rest.
    let tmp = TempDir::new().expect("tempdir");
    stage_shard_dir(tmp.path(), 2, 100);
    let (planned, total) = preflight_dispatch_budget(tmp.path(), 50, 4, 8, true)
        .expect("over-budget WITH --allow-shard-cycle must succeed pre-flight");
    assert_eq!(planned, 1600);
    assert_eq!(total, 200);
}

#[test]
fn preflight_budget_rejects_missing_dataset() {
    // Missing dataset dir → pre-flight refuses with a message that
    // names the path + gate so the operator knows which input is
    // bad before any trainer allocation.
    let missing = std::path::PathBuf::from("/nonexistent/_gate_corpus_preflight_missing");
    let err = preflight_dispatch_budget(&missing, 10, 2, 4, false)
        .expect_err("missing dataset dir must be rejected by pre-flight");
    match err {
        CliError::ValidationFailed(msg) => {
            assert!(
                msg.contains("GATE-CORPUS-PREFLIGHT"),
                "msg must cite gate: {msg}"
            );
            assert!(
                msg.contains("cannot count corpus tokens"),
                "msg must name I/O failure: {msg}"
            );
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn preflight_budget_zero_corpus_refuses_without_opt_in() {
    // An empty shard dir (0 .bin files) is not an I/O error — it
    // returns Ok(0). With planned > 0 AND --allow-shard-cycle absent
    // the pre-flight MUST still refuse; the factor rendering must
    // not panic on /0 and must say `inf×`.
    let tmp = TempDir::new().expect("tempdir");
    // no shard files; count_tokens returns Ok(0)
    let err = preflight_dispatch_budget(tmp.path(), 1, 1, 1, false)
        .expect_err("empty corpus must be refused without opt-in");
    match err {
        CliError::ValidationFailed(msg) => {
            assert!(
                msg.contains("total_tokens=0"),
                "msg must surface zero: {msg}"
            );
            assert!(
                msg.contains("inf"),
                "msg must render infinite factor safely: {msg}"
            );
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn synthetic_pretrain_end_to_end_happy_path() {
    let tmp = TempDir::new().expect("tempdir");
    let dataset = tmp.path().join("data.jsonl");
    let tokenizer = tmp.path().join("tok");
    let run_dir = tmp.path().join("run");

    let result = run(
        &dataset,
        &tokenizer,
        &run_dir,
        PretrainMode::Finetune,
        Some(5.0e-5),
        25,
        Some(5),
        2,
        4,
        5,
        42,
        Some(2.2),
        50257,
        true,
        "cpu",
        false,
        true,
    );
    assert!(
        result.is_ok(),
        "synthetic pretrain end-to-end must succeed: got {result:?}"
    );
}

#[test]
fn real_mode_empty_dataset_dir_errors() {
    // When --synthetic is off, the real-corpus branch must surface a
    // clear error if the dataset directory has no .bin shards. This
    // supersedes the old "non-synthetic is not implemented" guard.
    // Stage a valid vocab.json first so GATE-ARCH-370M-011 pre-flight
    // passes — otherwise the shard-iterator error below is never reached.
    let tmp = TempDir::new().expect("tempdir");
    let tok_dir = tmp.path().join("tok");
    stage_vocab_json(&tok_dir, Llama370MConfig::VOCAB_SIZE);
    let err = run(
        tmp.path(),
        &tok_dir,
        tmp.path(),
        PretrainMode::Finetune,
        Some(5.0e-5),
        10,
        Some(2),
        2,
        4,
        5,
        42,
        Some(2.2),
        50257,
        false,
        "cpu",
        true, // allow_shard_cycle — bypass GATE-CORPUS-PREFLIGHT
        // so this test exercises the later shard-iterator failure.
        true,
    )
    .expect_err("empty dataset dir must fail to initialise the shard iterator");
    match err {
        CliError::ValidationFailed(msg) => {
            assert!(
                msg.contains("shard iterator init failed"),
                "unexpected message: {msg}"
            );
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn invalid_target_val_loss_rejected() {
    let tmp = TempDir::new().expect("tempdir");
    let err = run(
        tmp.path(),
        tmp.path(),
        tmp.path(),
        PretrainMode::Finetune,
        Some(5.0e-5),
        10,
        Some(2),
        2,
        4,
        5,
        42,
        Some(-1.0),
        50257,
        true,
        "cpu",
        false,
        true,
    )
    .expect_err("negative target_val_loss must be rejected");
    assert!(matches!(err, CliError::ValidationFailed(_)));
}

// ── GATE-TRAIN-009 / INV-TRAIN-009 falsifiers ──────────────────────
// Contract: training-loop-pretrain-v1 v1.3.0 §hyperparameter_defaults
//
// These tests bind the CLI's `mode_defaults` resolver to the
// hyperparameter_defaults YAML table. If the table is ever edited
// without also updating this resolver (or vice versa), the tests
// fail. That is exactly the drift INV-TRAIN-009 forbids.

#[test]
fn mode_finetune_is_default_and_matches_contract() {
    // No overrides → resolved HP matches the `finetune` YAML row
    // (lr_max=5e-5, warmup_steps=100, target_val_loss=2.2) AND the
    // regime is Finetune so INV-TRAIN-005 epoch-zero cap = 10.0.
    let hp = mode_defaults(PretrainMode::Finetune, 50257, None, None, None);
    assert_eq!(hp.regime, TrainingRegime::Finetune);
    assert!(
        (hp.lr_max - 5.0e-5).abs() < 1.0e-12,
        "lr_max={} must equal finetune default 5e-5",
        hp.lr_max
    );
    assert_eq!(hp.warmup_steps, 100);
    assert!(
        (hp.target_val_loss - 2.2).abs() < 1.0e-6,
        "target_val_loss={} must equal finetune default 2.2",
        hp.target_val_loss
    );
}

#[test]
fn mode_from_scratch_applies_all_four_defaults() {
    // `--mode from-scratch` with no HP overrides MUST yield the full
    // cold-start 4-tuple atomically — regime=FromScratch, lr=3e-4,
    // warmup=1000, target=3.0. INV-TRAIN-009 falsifier (a).
    let hp = mode_defaults(PretrainMode::FromScratch, 50257, None, None, None);
    assert_eq!(hp.regime, TrainingRegime::FromScratch { vocab_size: 50257 });
    assert!(
        (hp.lr_max - 3.0e-4).abs() < 1.0e-12,
        "lr_max={} must equal from_scratch default 3e-4",
        hp.lr_max
    );
    assert_eq!(hp.warmup_steps, 1000);
    assert!(
        (hp.target_val_loss - 3.0).abs() < 1.0e-6,
        "target_val_loss={} must equal from_scratch default 3.0",
        hp.target_val_loss
    );
}

#[test]
fn mode_from_scratch_honors_explicit_lr_override() {
    // `--mode from-scratch --lr 1e-4` → regime still flips to
    // FromScratch AND warmup/target keep the from_scratch defaults,
    // but lr_max is the operator-supplied 1e-4. INV-TRAIN-009
    // falsifier (b): overrides win, regime still moves.
    let hp = mode_defaults(PretrainMode::FromScratch, 50257, Some(1.0e-4), None, None);
    assert_eq!(hp.regime, TrainingRegime::FromScratch { vocab_size: 50257 });
    assert!(
        (hp.lr_max - 1.0e-4).abs() < 1.0e-12,
        "lr_max={} must equal explicit override 1e-4",
        hp.lr_max
    );
    // Remaining two fields retained their mode defaults.
    assert_eq!(hp.warmup_steps, 1000);
    assert!((hp.target_val_loss - 3.0).abs() < 1.0e-6);
}

// ── GATE-TRAIN-010 / INV-TRAIN-010 falsifiers ──────────────────────
// Contract: training-loop-pretrain-v1 v1.4.0 §INV-TRAIN-010
//
// Task #105's original wiring shipped `synthetic: bool` with
// `default_value = "true"`. The `--synthetic` flag had no
// companion to turn it off, so every invocation of `apr pretrain`
// silently routed to drive_synthetic. Tasks #119 / #124 / #125
// all captured scripted-loss output and mis-labeled it real
// compute. These two tests parse actual argv through clap and
// assert the routing discriminator byte-for-byte.

fn parse_pretrain_synthetic(extra: &[&str]) -> bool {
    // The `Commands` enum is large enough in debug builds to overflow
    // the default 2 MiB test-thread stack during clap's recursive
    // destructuring. Run the parse on a worker thread with a 16 MiB
    // stack so this falsifier passes in both debug and release.
    let extra: Vec<String> = extra.iter().map(|s| (*s).to_string()).collect();
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(move || {
            use clap::Parser;
            let mut argv: Vec<String> = vec![
                "apr".to_string(),
                "pretrain".to_string(),
                "--dataset".to_string(),
                "/tmp/_gate_train_010/ds".to_string(),
                "--tokenizer".to_string(),
                "/tmp/_gate_train_010/tok".to_string(),
                "--run-dir".to_string(),
                "/tmp/_gate_train_010/run".to_string(),
            ];
            argv.extend(extra);
            let cli = crate::Cli::try_parse_from(&argv).expect("clap parse must succeed");
            match *cli.command {
                crate::Commands::Extended(crate::ExtendedCommands::Training(
                    crate::TrainingCommands::Pretrain { synthetic, .. },
                )) => synthetic,
                other => panic!("expected ExtendedCommands::Training(Pretrain), got {other:?}"),
            }
        })
        .expect("spawn parse thread")
        .join()
        .expect("parse thread must not panic")
}

#[test]
fn cli_pretrain_defaults_to_real_compute() {
    // Absent `--synthetic` MUST parse to synthetic=false so the
    // dispatcher routes through drive_real.
    assert!(
        !parse_pretrain_synthetic(&[]),
        "INV-TRAIN-010: `apr pretrain` (no --synthetic) must parse to synthetic=false"
    );
}

#[test]
fn cli_pretrain_synthetic_flag_routes_to_synthetic() {
    // `--synthetic` present MUST parse to synthetic=true.
    assert!(
        parse_pretrain_synthetic(&["--synthetic"]),
        "INV-TRAIN-010: `apr pretrain --synthetic` must parse to synthetic=true"
    );
}

// ── FALSIFY-GPUTRAIN-001 / 002 CLI surface (contract phase 1) ────
// Contract: gpu-training-backend-v1 §device_dispatch
//
// These tests parse actual `apr pretrain --device …` argv through
// clap and assert the string is surfaced byte-for-byte to the
// dispatcher. `resolve_device()` itself is exercised by
// `aprender-train::train::device::tests` — these tests verify that
// the CLI flag exists and that its default is `auto` (the only
// spec allowed to fall back).

fn parse_pretrain_device(extra: &[&str]) -> String {
    let extra: Vec<String> = extra.iter().map(|s| (*s).to_string()).collect();
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(move || {
            use clap::Parser;
            let mut argv: Vec<String> = vec![
                "apr".to_string(),
                "pretrain".to_string(),
                "--dataset".to_string(),
                "/tmp/_gputrain_device/ds".to_string(),
                "--tokenizer".to_string(),
                "/tmp/_gputrain_device/tok".to_string(),
                "--run-dir".to_string(),
                "/tmp/_gputrain_device/run".to_string(),
            ];
            argv.extend(extra);
            let cli = crate::Cli::try_parse_from(&argv).expect("clap parse must succeed");
            match *cli.command {
                crate::Commands::Extended(crate::ExtendedCommands::Training(
                    crate::TrainingCommands::Pretrain { device, .. },
                )) => device,
                other => panic!("expected ExtendedCommands::Training(Pretrain), got {other:?}"),
            }
        })
        .expect("spawn parse thread")
        .join()
        .expect("parse thread must not panic")
}

#[test]
fn cli_pretrain_device_defaults_to_auto() {
    // Absent `--device`, the flag MUST parse to `"auto"` — the only
    // spec allowed to silently fall back to CPU when CUDA is not
    // available. Any other default would violate the contract's
    // "explicit request → hard-fail" invariant.
    assert_eq!(
        parse_pretrain_device(&[]),
        "auto",
        "gpu-training-backend-v1 INV-GPUTRAIN-002: default --device must be `auto`",
    );
}

#[test]
fn cli_pretrain_device_accepts_cpu() {
    // `--device cpu` MUST round-trip through clap unchanged.
    assert_eq!(parse_pretrain_device(&["--device", "cpu"]), "cpu");
}

#[test]
fn cli_pretrain_device_accepts_cuda_index() {
    // `--device cuda:7` MUST round-trip unchanged; grammar
    // enforcement happens in `resolve_device`, not at clap.
    assert_eq!(parse_pretrain_device(&["--device", "cuda:7"]), "cuda:7");
}

// ── FALSIFY-CORPUS-004 CLI surface ─────────────────────────────────
// Contract: pretraining-corpus-v1 v2.0.0 §FALSIFY-CORPUS-004.
//
// The `--allow-shard-cycle` flag MUST default to `false` so the
// pre-flight gate is the active default. Operators who want the
// cycle path must opt in explicitly — the default cannot be the
// lenient one.

fn parse_pretrain_allow_shard_cycle(extra: &[&str]) -> bool {
    let extra: Vec<String> = extra.iter().map(|s| (*s).to_string()).collect();
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(move || {
            use clap::Parser;
            let mut argv: Vec<String> = vec![
                "apr".to_string(),
                "pretrain".to_string(),
                "--dataset".to_string(),
                "/tmp/_corpus_preflight/ds".to_string(),
                "--tokenizer".to_string(),
                "/tmp/_corpus_preflight/tok".to_string(),
                "--run-dir".to_string(),
                "/tmp/_corpus_preflight/run".to_string(),
            ];
            argv.extend(extra);
            let cli = crate::Cli::try_parse_from(&argv).expect("clap parse must succeed");
            match *cli.command {
                crate::Commands::Extended(crate::ExtendedCommands::Training(
                    crate::TrainingCommands::Pretrain {
                        allow_shard_cycle, ..
                    },
                )) => allow_shard_cycle,
                other => panic!("expected ExtendedCommands::Training(Pretrain), got {other:?}"),
            }
        })
        .expect("spawn parse thread")
        .join()
        .expect("parse thread must not panic")
}

#[test]
fn cli_pretrain_allow_shard_cycle_defaults_to_false() {
    // Absent flag MUST parse to `false` so the pre-flight gate is
    // the active default. GATE-CORPUS-PREFLIGHT refuses over-budget
    // dispatches unless the operator consciously opts in.
    assert!(
        !parse_pretrain_allow_shard_cycle(&[]),
        "FALSIFY-CORPUS-004: default --allow-shard-cycle must be false"
    );
}

#[test]
fn cli_pretrain_allow_shard_cycle_flag_sets_true() {
    // `--allow-shard-cycle` present MUST parse to `true`.
    assert!(
        parse_pretrain_allow_shard_cycle(&["--allow-shard-cycle"]),
        "FALSIFY-CORPUS-004: --allow-shard-cycle must flip the flag to true"
    );
}
