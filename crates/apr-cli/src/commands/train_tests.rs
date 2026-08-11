// PMAT-540 Phase 5: Tests for train command handler functions
// Contract: co-evolution with apr-cli-commands-v1.yaml

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // --format (dogfood 0.63.0, issue #2374 finding 7)
    //
    // `apr train plan --format` was declared `_format: &str` — underscore
    // prefixed, never read — so text, json, yaml and an invalid value all
    // produced byte-identical text output with exit 0.
    // ========================================================================

    /// A minimal, valid pretrain spec parsed from YAML (TrainSpec has no Default).
    fn scratch_spec(tag: &str) -> entrenar::config::TrainSpec {
        let dir = std::env::temp_dir().join(format!("apr-2374-fmt-{}-{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        let cfg = dir.join("c.yaml");
        // validate_config requires both paths to exist on disk.
        let model = dir.join("m.safetensors");
        let data = dir.join("d.json");
        std::fs::write(&model, b"stub").expect("scratch model");
        std::fs::write(&data, b"{}").expect("scratch data");
        std::fs::write(
            &cfg,
            format!(
                "model:\n  path: {}\n  mode: tabular\ndata:\n  train: {}\n  batch_size: 8\noptimizer:\n  name: adam\n  lr: 0.001\ntraining:\n  epochs: 1\n",
                model.display(),
                data.display()
            ),
        )
        .expect("scratch config");
        let spec = entrenar::config::load_config(&cfg).expect("minimal config must load");
        let _ = std::fs::remove_dir_all(&dir);
        spec
    }

    #[test]
    fn plan_format_parses_the_three_documented_values() {
        assert_eq!(PlanFormat::parse("text").expect("text is documented"), PlanFormat::Text);
        assert_eq!(PlanFormat::parse("json").expect("json is documented"), PlanFormat::Json);
        assert_eq!(PlanFormat::parse("yaml").expect("yaml is documented"), PlanFormat::Yaml);
    }

    #[test]
    fn plan_format_distinguishes_the_documented_values() {
        // The defect was that all three collapsed to the same rendering.
        let text = PlanFormat::parse("text").expect("text is documented");
        let json = PlanFormat::parse("json").expect("json is documented");
        let yaml = PlanFormat::parse("yaml").expect("yaml is documented");
        assert_ne!(text, json);
        assert_ne!(json, yaml);
        assert_ne!(text, yaml);
    }

    #[test]
    fn plan_format_rejects_an_invalid_value() {
        let err = PlanFormat::parse("bogus").expect_err("an invalid --format must not be accepted");
        let msg = err.to_string();
        assert!(msg.contains("bogus"), "error must echo the bad value: {msg}");
        assert!(msg.contains("yaml"), "error must list the supported formats: {msg}");
    }

    #[test]
    fn plan_format_is_case_insensitive_and_accepts_yml() {
        assert_eq!(PlanFormat::parse("YAML").expect("case-insensitive"), PlanFormat::Yaml);
        assert_eq!(PlanFormat::parse("yml").expect("yml is a yaml spelling"), PlanFormat::Yaml);
    }

    #[test]
    fn plan_yaml_rendering_is_valid_yaml_and_not_the_text_table() {
        // Falsifies "yaml prints the human table": the manifest must round-trip
        // through a YAML parser and carry the plan's fields.
        let spec = scratch_spec("yaml");
        let manifest = pretrain_plan_manifest(std::path::Path::new("c.yaml"), &spec);
        let rendered = serde_yaml::to_string(&manifest).expect("manifest must serialize to YAML");
        let parsed: serde_yaml::Value =
            serde_yaml::from_str(&rendered).expect("--format yaml must emit parseable YAML");
        assert_eq!(parsed["task"], serde_yaml::Value::String("pretrain".into()));
        assert_eq!(parsed["verdict"], serde_yaml::Value::String("ready".into()));
        assert!(
            !rendered.contains("Pre-training Plan"),
            "yaml must not be the human table: {rendered}"
        );
    }

    #[test]
    fn plan_json_and_yaml_carry_the_same_manifest() {
        let spec = scratch_spec("same");
        let manifest = pretrain_plan_manifest(std::path::Path::new("c.yaml"), &spec);
        let from_yaml: serde_json::Value = serde_yaml::from_str(
            &serde_yaml::to_string(&manifest).expect("yaml render"),
        )
        .expect("yaml re-parse");
        assert_eq!(from_yaml["config"], manifest["config"]);
        assert_eq!(from_yaml["model"]["path"], manifest["model"]["path"]);
    }

    // ========================================================================
    // --strategy (dogfood 0.63.0, issue #2374 finding 13)
    //
    // `apr train sweep --strategy bogus` fell through the `"random" | _` arm to
    // a RANDOM search, printed "Strategy: bogus" back as though it were real,
    // and wrote sweep files byte-identical to `--strategy random`. A user who
    // believed they had a grid search had a random one.
    // ========================================================================

    #[test]
    fn sweep_strategy_parses_both_real_strategies() {
        assert_eq!(parse_sweep_strategy("grid").expect("grid is real"), SweepStrategy::Grid);
        assert_eq!(parse_sweep_strategy("random").expect("random is real"), SweepStrategy::Random);
    }

    #[test]
    fn sweep_strategy_rejects_a_typo_instead_of_silently_randomising() {
        for typo in ["bogus", "gird", "Grid search", "", "tpe"] {
            assert!(
                parse_sweep_strategy(typo).is_err(),
                "--strategy {typo:?} must be rejected, not silently randomised"
            );
        }
    }

    #[test]
    fn sweep_strategy_typo_is_an_error_naming_the_alternatives() {
        let err = parse_sweep_strategy("gird")
            .expect_err("a typo must not silently become a random search");
        let msg = err.to_string();
        assert!(msg.contains("gird"), "error must echo the typo: {msg}");
        assert!(msg.contains("grid") && msg.contains("random"), "error must list both: {msg}");
    }

    #[test]
    fn sweep_strategy_round_trips_through_display() {
        // The banner prints Display, so it must never echo a bad value back.
        assert_eq!(SweepStrategy::Grid.to_string(), "grid");
        assert_eq!(SweepStrategy::Random.to_string(), "random");
    }

    #[test]
    fn sweep_rejects_a_bad_strategy_before_writing_anything() {
        // Validation must precede create_dir_all: a typo must not leave files.
        let dir = std::env::temp_dir().join(format!("apr-2374-sweep-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let cfg = dir.join("base.yaml");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&cfg, "training:\n  epochs: 1\n").unwrap();
        let out = dir.join("sweeps");

        let err = run_sweep(&cfg, "bogus", 2, &out, 7, false)
            .expect_err("a bogus strategy must be rejected");
        assert!(matches!(err, CliError::ValidationFailed(_)));
        assert!(!out.exists(), "no sweep directory may be created for a rejected strategy");

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ========================================================================
    // classify_exit_code
    // ========================================================================

    #[test]
    fn classify_exit_code_success() {
        assert_eq!(classify_exit_code(0), "success");
    }

    #[test]
    fn classify_exit_code_error() {
        assert_eq!(classify_exit_code(1), "error");
    }

    #[test]
    fn classify_exit_code_usage() {
        assert_eq!(classify_exit_code(2), "usage");
    }

    #[test]
    fn classify_exit_code_oom() {
        assert_eq!(classify_exit_code(137), "oom");
    }

    #[test]
    fn classify_exit_code_sigsegv() {
        assert_eq!(classify_exit_code(139), "sigsegv");
    }

    #[test]
    fn classify_exit_code_sigabrt() {
        assert_eq!(classify_exit_code(134), "sigabrt");
    }

    #[test]
    fn classify_exit_code_sigbus() {
        assert_eq!(classify_exit_code(135), "sigbus");
    }

    #[test]
    fn classify_exit_code_generic_signal() {
        assert_eq!(classify_exit_code(143), "signal"); // SIGTERM
        assert_eq!(classify_exit_code(129), "signal"); // SIGHUP
    }

    #[test]
    fn classify_exit_code_unknown() {
        assert_eq!(classify_exit_code(42), "unknown");
        assert_eq!(classify_exit_code(-1), "unknown");
    }

    // ========================================================================
    // format_archive_size
    // ========================================================================

    #[test]
    fn format_archive_size_bytes() {
        assert_eq!(format_archive_size(0), "0 B");
        assert_eq!(format_archive_size(512), "512 B");
        assert_eq!(format_archive_size(1023), "1023 B");
    }

    #[test]
    fn format_archive_size_kb() {
        assert_eq!(format_archive_size(1024), "1.0 KB");
        assert_eq!(format_archive_size(1536), "1.5 KB");
    }

    #[test]
    fn format_archive_size_mb() {
        assert_eq!(format_archive_size(1_048_576), "1.0 MB");
        assert_eq!(format_archive_size(10_485_760), "10.0 MB");
    }

    #[test]
    fn format_archive_size_gb() {
        assert_eq!(format_archive_size(1_073_741_824), "1.0 GB");
        assert_eq!(format_archive_size(4_294_967_296), "4.0 GB");
    }

    // ========================================================================
    // watch_max_restarts_exceeded
    // ========================================================================

    #[test]
    fn watch_max_restarts_exceeded_returns_error() {
        let err = watch_max_restarts_exceeded(5, true);
        match err {
            CliError::ValidationFailed(msg) => {
                assert!(msg.contains("5"), "Should mention restart count");
                assert!(msg.contains("exceeded"), "Should mention exceeded");
            }
            _ => panic!("Expected ValidationFailed"),
        }
    }

    // ========================================================================
    // classify_not_available
    // ========================================================================

    /// This test used to assert the message mentioned "entrenar", which locked
    /// in a claim that was false in the binary printing it: entrenar is the
    /// in-tree `crates/aprender-train` built at the workspace version, so
    /// "requires entrenar >= 0.8 (not yet published)" named a blocker that
    /// could not exist. The message must instead route the user to the command
    /// that does implement classification.
    #[test]
    fn classify_not_available_names_the_command_that_works() {
        let err = classify_not_available();
        match err {
            CliError::ValidationFailed(msg) => {
                assert!(msg.contains("classify"), "Should mention classify");
                assert!(
                    msg.contains("apr finetune"),
                    "must point at the command that implements classification: {msg}"
                );
                assert!(
                    !msg.contains("not yet published"),
                    "claims an unpublished dependency that this binary already links: {msg}"
                );
                assert!(
                    !msg.contains("entrenar >= 0.8"),
                    "names a version blocker that does not exist: {msg}"
                );
            }
            _ => panic!("Expected ValidationFailed"),
        }
    }

    /// The DEFAULT task of `apr train plan` / `apr train apply` must be one
    /// that can succeed. It was `classify`, so the documented bare invocation
    /// `apr train plan --data <file>` always exited 5.
    /// Parse `apr train <sub>` argv through clap and return the `--task` value.
    /// Runs on a wide-stack thread: the full `Cli` enum overflows the default
    /// 2 MiB test-thread stack in a debug build (same pattern as
    /// `parse_pretrain_device`).
    fn parse_train_task(sub: &str) -> String {
        let sub = sub.to_string();
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(move || {
                use clap::Parser;
                let argv = vec![
                    "apr".to_string(),
                    "train".to_string(),
                    sub,
                    "--config".to_string(),
                    "/nonexistent/c.yaml".to_string(),
                ];
                let cli = crate::Cli::try_parse_from(&argv).expect("clap parse must succeed");
                match *cli.command {
                    crate::Commands::Extended(crate::ExtendedCommands::Train { command }) => {
                        match command {
                            crate::TrainCommands::Plan { task, .. }
                            | crate::TrainCommands::Apply { task, .. } => task,
                            _ => panic!("expected train plan/apply"),
                        }
                    }
                    _ => panic!("expected ExtendedCommands::Train"),
                }
            })
            .expect("spawn parse thread")
            .join()
            .expect("parse thread must not panic")
    }

    #[test]
    fn train_plan_and_apply_default_to_a_task_that_can_run() {
        for sub in ["plan", "apply"] {
            let task = parse_train_task(sub);
            assert_eq!(
                task, "pretrain",
                "`apr train {sub}` defaults to `{task}`, a task this command cannot run"
            );
        }
    }

    // ========================================================================
    // PMAT-125 B1: lcg_f64 — deterministic LCG PRNG
    // ========================================================================

    #[test]
    fn lcg_f64_is_deterministic_for_same_seed() {
        let mut s1 = 42u64;
        let mut s2 = 42u64;
        for _ in 0..10 {
            assert_eq!(lcg_f64(&mut s1), lcg_f64(&mut s2));
        }
    }

    #[test]
    fn lcg_f64_in_unit_range() {
        let mut state = 1u64;
        for _ in 0..1000 {
            let v = lcg_f64(&mut state);
            assert!((0.0..1.0).contains(&v), "value {v} out of [0,1)");
        }
    }

    #[test]
    fn lcg_f64_different_seeds_diverge() {
        let mut a = 7u64;
        let mut b = 999u64;
        // First draw from distinct seeds should differ (overwhelmingly likely).
        assert_ne!(lcg_f64(&mut a), lcg_f64(&mut b));
    }

    // ========================================================================
    // PMAT-125 B1: set_yaml_f64 / set_yaml_u64 — nested YAML mutation
    // ========================================================================

    #[test]
    fn set_yaml_f64_sets_existing_nested_key() {
        let mut root: serde_yaml::Value =
            serde_yaml::from_str("optimizer:\n  lr: 0.1\n").unwrap();
        set_yaml_f64(&mut root, &["optimizer", "lr"], 3e-4);
        let got = root["optimizer"]["lr"].as_f64().unwrap();
        assert!((got - 3e-4).abs() < 1e-12);
    }

    #[test]
    fn set_yaml_f64_creates_missing_intermediate_maps() {
        let mut root = serde_yaml::Value::Mapping(serde_yaml::Mapping::new());
        set_yaml_f64(&mut root, &["optimizer", "weight_decay"], 0.05);
        assert!((root["optimizer"]["weight_decay"].as_f64().unwrap() - 0.05).abs() < 1e-12);
    }

    #[test]
    fn set_yaml_u64_sets_and_creates() {
        let mut root = serde_yaml::Value::Mapping(serde_yaml::Mapping::new());
        set_yaml_u64(&mut root, &["data", "batch_size"], 16);
        assert_eq!(root["data"]["batch_size"].as_u64(), Some(16));
        // Overwrite existing value.
        set_yaml_u64(&mut root, &["data", "batch_size"], 4);
        assert_eq!(root["data"]["batch_size"].as_u64(), Some(4));
    }

    // ========================================================================
    // PMAT-125 B1: build_distributed_yaml — DDP config section
    // ========================================================================

    #[test]
    fn build_distributed_yaml_coordinator_role_for_rank0() {
        let v = build_distributed_yaml(Some(4), Some(0), Some("10.0.0.1:9000"));
        assert_eq!(v["world_size"].as_u64(), Some(4));
        assert_eq!(v["rank"].as_u64(), Some(0));
        assert_eq!(v["coordinator_addr"].as_str(), Some("10.0.0.1:9000"));
        assert_eq!(v["role"].as_str(), Some("coordinator"));
    }

    #[test]
    fn build_distributed_yaml_worker_role_for_nonzero_rank() {
        let v = build_distributed_yaml(Some(8), Some(3), None);
        assert_eq!(v["rank"].as_u64(), Some(3));
        assert_eq!(v["role"].as_str(), Some("worker"));
        // Default coordinator address when None.
        assert_eq!(v["coordinator_addr"].as_str(), Some("0.0.0.0:9000"));
    }

    #[test]
    fn build_distributed_yaml_defaults_when_all_none() {
        let v = build_distributed_yaml(None, None, None);
        assert_eq!(v["world_size"].as_u64(), Some(2));
        assert_eq!(v["rank"].as_u64(), Some(0));
        assert_eq!(v["role"].as_str(), Some("coordinator"));
    }

    // ========================================================================
    // PMAT-125 B1: patch_yaml_config — overlay CLI flags onto a YAML config
    // ========================================================================

    #[test]
    fn patch_yaml_config_writes_distributed_and_seed() {
        let dir = std::env::temp_dir().join(format!("apr-pmat125-patch-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = dir.join("base.yaml");
        std::fs::write(&cfg, "training:\n  epochs: 1\n").unwrap();

        let out = patch_yaml_config(&cfg, None, true, Some(2), Some(0), Some("a:1"), true, Some(123))
            .expect("patch should succeed");
        let patched = std::fs::read_to_string(&out).unwrap();
        let yaml: serde_yaml::Value = serde_yaml::from_str(&patched).unwrap();
        assert_eq!(yaml["training"]["distributed"]["world_size"].as_u64(), Some(2));
        assert_eq!(yaml["training"]["deterministic"].as_bool(), Some(true));
        assert_eq!(yaml["training"]["seed"].as_u64(), Some(123));

        let _ = std::fs::remove_file(&out);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ─── -o/--output (dogfood 0.63.0, issue #2374 finding 5) ────────────────
    //
    // `apr train apply -o DIR` was accepted, documented with a default of
    // /tmp/training-output, and silently discarded: only training.output_dir in
    // the YAML was ever honoured, and a full training run was then thrown away
    // at the save step with a bare "No such file or directory (os error 2)".

    #[test]
    fn patch_yaml_config_output_flag_overrides_the_yaml_output_dir() {
        let dir = std::env::temp_dir().join(format!("apr-2374-out-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = dir.join("base.yaml");
        std::fs::write(&cfg, "training:\n  epochs: 1\n  output_dir: /tmp/FROM_YAML\n").unwrap();

        let flag = std::path::Path::new("/tmp/FROM_FLAG");
        let out = patch_yaml_config(&cfg, Some(flag), false, None, None, None, false, None)
            .expect("patch should succeed");
        let yaml: serde_yaml::Value =
            serde_yaml::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
        assert_eq!(
            yaml["training"]["output_dir"].as_str(),
            Some("/tmp/FROM_FLAG"),
            "-o must win over training.output_dir; it used to be discarded"
        );

        let _ = std::fs::remove_file(&out);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn patch_yaml_config_without_output_flag_leaves_the_yaml_alone() {
        let dir = std::env::temp_dir().join(format!("apr-2374-noout-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = dir.join("base.yaml");
        std::fs::write(&cfg, "training:\n  epochs: 1\n  output_dir: /tmp/FROM_YAML\n").unwrap();

        let out = patch_yaml_config(&cfg, None, false, None, None, None, true, None)
            .expect("patch should succeed");
        let yaml: serde_yaml::Value =
            serde_yaml::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
        assert_eq!(yaml["training"]["output_dir"].as_str(), Some("/tmp/FROM_YAML"));

        let _ = std::fs::remove_file(&out);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn patch_yaml_config_missing_training_section_errors() {
        let dir = std::env::temp_dir().join(format!("apr-pmat125-patch2-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = dir.join("notraining.yaml");
        std::fs::write(&cfg, "model:\n  path: foo\n").unwrap();

        let err = patch_yaml_config(&cfg, None, false, None, None, None, true, None).unwrap_err();
        match err {
            CliError::ValidationFailed(m) => assert!(m.contains("training")),
            _ => panic!("expected ValidationFailed for missing training section"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn patch_yaml_config_invalid_yaml_errors() {
        let dir = std::env::temp_dir().join(format!("apr-pmat125-patch3-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = dir.join("bad.yaml");
        std::fs::write(&cfg, "training: [unterminated\n").unwrap();
        let err = patch_yaml_config(&cfg, None, false, None, None, None, false, Some(1)).unwrap_err();
        assert!(matches!(err, CliError::ValidationFailed(_)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ========================================================================
    // PMAT-125 B1: generate_grid_configs — grid search over hyperparameters
    // ========================================================================

    #[test]
    fn generate_grid_configs_respects_max_cap() {
        let base: serde_yaml::Value = serde_yaml::from_str("model:\n  path: m\n").unwrap();
        let configs = generate_grid_configs(&base, 7);
        assert_eq!(configs.len(), 7, "should stop at max_configs");
    }

    #[test]
    fn generate_grid_configs_sets_distinct_hyperparams() {
        let base: serde_yaml::Value = serde_yaml::from_str("model:\n  path: m\n").unwrap();
        let configs = generate_grid_configs(&base, 3);
        // The grid varies weight_decay fastest, so the first three differ in wd.
        let wds: Vec<f64> = configs
            .iter()
            .map(|c| c["optimizer"]["weight_decay"].as_f64().unwrap())
            .collect();
        assert_eq!(wds, vec![0.0, 0.01, 0.1]);
        // All share the first lr (1e-5) and first batch size (2).
        for c in &configs {
            assert!((c["optimizer"]["lr"].as_f64().unwrap() - 1e-5).abs() < 1e-12);
            assert_eq!(c["data"]["batch_size"].as_u64(), Some(2));
        }
    }

    #[test]
    fn generate_grid_configs_full_grid_is_45() {
        let base: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
        // 5 lr × 3 bs × 3 wd = 45 combinations.
        let configs = generate_grid_configs(&base, 1000);
        assert_eq!(configs.len(), 45);
    }

    // ========================================================================
    // PMAT-125 B1: generate_random_configs — random search via LCG
    // ========================================================================

    #[test]
    fn generate_random_configs_count_and_ranges() {
        let base: serde_yaml::Value = serde_yaml::from_str("model:\n  path: m\n").unwrap();
        let configs = generate_random_configs(&base, 20, 1234);
        assert_eq!(configs.len(), 20);
        for c in &configs {
            let lr = c["optimizer"]["lr"].as_f64().unwrap();
            assert!((1e-5..=1e-2).contains(&lr), "lr {lr} out of log-uniform range");
            let bs = c["data"]["batch_size"].as_u64().unwrap();
            assert!([1, 2, 4, 8, 16].contains(&bs), "bs {bs} not a valid choice");
            let warmup = c["training"]["warmup_steps"].as_u64().unwrap();
            assert!((50..=2000).contains(&warmup), "warmup {warmup} out of range");
        }
    }

    #[test]
    fn generate_random_configs_is_seed_deterministic() {
        let base: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
        let a = generate_random_configs(&base, 5, 77);
        let b = generate_random_configs(&base, 5, 77);
        for (ca, cb) in a.iter().zip(b.iter()) {
            assert_eq!(
                ca["optimizer"]["lr"].as_f64(),
                cb["optimizer"]["lr"].as_f64()
            );
        }
    }

    // ========================================================================
    // PMAT-125 B1: parse_best_ppl — extract val_ppl from training logs
    // ========================================================================

    #[test]
    fn parse_best_ppl_picks_minimum() {
        let log = "[eval] step=1 val_loss=2.0 val_ppl=7.39\n\
                   [eval] step=2 val_loss=1.5 val_ppl=4.48\n\
                   [eval] step=3 val_loss=1.8 val_ppl=6.05\n";
        let best = parse_best_ppl(log);
        assert!((best - 4.48).abs() < 1e-9, "expected 4.48, got {best}");
    }

    #[test]
    fn parse_best_ppl_no_matches_returns_infinity() {
        let best = parse_best_ppl("nothing useful here\nstep=1 loss=2.0\n");
        assert!(best.is_infinite());
    }

    #[test]
    fn parse_best_ppl_handles_trailing_text() {
        let best = parse_best_ppl("val_ppl=3.14 (done)\n");
        assert!((best - 3.14).abs() < 1e-9);
    }

    // ========================================================================
    // PMAT-125 B1: discover_sweep_configs — find sweep-*.yaml files
    // ========================================================================

    #[test]
    fn discover_sweep_configs_finds_and_sorts() {
        let dir = std::env::temp_dir().join(format!("apr-pmat125-sweep-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("sweep-002.yaml"), "{}").unwrap();
        std::fs::write(dir.join("sweep-000.yaml"), "{}").unwrap();
        std::fs::write(dir.join("sweep-001.yaml"), "{}").unwrap();
        std::fs::write(dir.join("ignore.txt"), "x").unwrap();
        std::fs::write(dir.join("other.yaml"), "{}").unwrap();

        let configs = discover_sweep_configs(&dir).unwrap();
        assert_eq!(configs.len(), 3, "only sweep-*.yaml files counted");
        let names: Vec<String> = configs
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert_eq!(
            names,
            vec!["sweep-000.yaml", "sweep-001.yaml", "sweep-002.yaml"]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn discover_sweep_configs_empty_dir_errors() {
        let dir = std::env::temp_dir().join(format!("apr-pmat125-empty-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let err = discover_sweep_configs(&dir).unwrap_err();
        match err {
            CliError::ValidationFailed(m) => assert!(m.contains("No sweep")),
            _ => panic!("expected ValidationFailed"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ========================================================================
    // PMAT-125 B1: parse_sweep_config_params — pull HP fields from sweep YAML
    // ========================================================================

    #[test]
    fn parse_sweep_config_params_reads_fields() {
        let dir = std::env::temp_dir().join(format!("apr-pmat125-parse-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("sweep-000.yaml");
        std::fs::write(
            &p,
            "optimizer:\n  lr: 0.0003\n  weight_decay: 0.01\ntraining:\n  warmup_steps: 500\n",
        )
        .unwrap();
        let entries = parse_sweep_config_params(&[p.clone()]).unwrap();
        assert_eq!(entries.len(), 1);
        let e = &entries[0];
        assert!((e.lr - 0.0003).abs() < 1e-12);
        assert!((e.weight_decay - 0.01).abs() < 1e-12);
        assert_eq!(e.warmup_steps, 500);
        assert!(e.best_ppl.is_infinite());
        assert!(e.eliminated_round.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_sweep_config_params_missing_fields_default_to_zero() {
        let dir = std::env::temp_dir().join(format!("apr-pmat125-parse2-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("sweep-001.yaml");
        std::fs::write(&p, "model:\n  path: m\n").unwrap();
        let entries = parse_sweep_config_params(&[p]).unwrap();
        assert_eq!(entries[0].lr, 0.0);
        assert_eq!(entries[0].weight_decay, 0.0);
        assert_eq!(entries[0].warmup_steps, 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ========================================================================
    // PMAT-125 B1: build_halving_json — winner + all-results serialization
    // ========================================================================

    #[test]
    fn build_halving_json_includes_winner_and_mutransfer_lr() {
        let dir = std::env::temp_dir().join(format!("apr-pmat125-halve-{}", std::process::id()));
        let winner = HalvingEntry {
            path: dir.join("sweep-000.yaml"),
            best_ppl: 4.2,
            lr: 1e-3,
            weight_decay: 0.01,
            warmup_steps: 100,
            eliminated_round: None,
            last_failure: None,
        };
        let other = HalvingEntry {
            path: dir.join("sweep-001.yaml"),
            best_ppl: f64::INFINITY,
            lr: 2e-3,
            weight_decay: 0.0,
            warmup_steps: 50,
            eliminated_round: Some(0),
            last_failure: None,
        };
        let results = vec![winner, other];
        // μTransfer scales the LR by source/target width.
        let target_lr = results[0].lr * (256.0 / 1024.0);
        let json = build_halving_json(
            &results,
            "sweep-000.yaml",
            &results[0],
            target_lr,
            256,
            1024,
            2,
            100,
        );
        assert_eq!(json["winner"]["config"].as_str(), Some("sweep-000.yaml"));
        assert!((json["winner"]["proxy_lr"].as_f64().unwrap() - 1e-3).abs() < 1e-12);
        assert!((json["winner"]["target_lr"].as_f64().unwrap() - target_lr).abs() < 1e-12);
        assert_eq!(json["winner"]["source_width"].as_u64(), Some(256));
        assert_eq!(json["winner"]["target_width"].as_u64(), Some(1024));
        // Non-finite best_ppl serializes as null.
        assert!(json["all_results"][1]["best_ppl"].is_null());
        assert_eq!(json["all_results"][1]["eliminated_round"].as_u64(), Some(0));
        assert_eq!(json["settings"]["rounds"].as_u64(), Some(2));
    }

    // ========================================================================
    // PMAT-125 B1: copy_checkpoint_files — copy + BLAKE3 hash manifest
    // ========================================================================

    #[test]
    fn copy_checkpoint_files_hashes_and_totals() {
        let base = std::env::temp_dir().join(format!("apr-pmat125-ckpt-{}", std::process::id()));
        let src = base.join("src");
        let dst = base.join("dst");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&dst).unwrap();
        std::fs::write(src.join("a.bin"), b"hello").unwrap();
        std::fs::write(src.join("b.bin"), b"world!!").unwrap();
        std::fs::create_dir_all(src.join("subdir")).unwrap(); // ignored (not a file)

        let (entries, total) = copy_checkpoint_files(&src, &dst, true).unwrap();
        assert_eq!(entries.len(), 2, "two files copied, subdir skipped");
        assert_eq!(total, 5 + 7);
        // Files actually copied to dst with identical bytes.
        assert_eq!(std::fs::read(dst.join("a.bin")).unwrap(), b"hello");
        // Manifest entries carry a blake3 hash.
        for e in &entries {
            assert!(e["blake3"].as_str().unwrap().len() == 64);
        }
        let _ = std::fs::remove_dir_all(&base);
    }

    // ========================================================================
    // PMAT-125 B1: high-level dispatch — run_plan / run_apply task routing
    // ========================================================================

    #[test]
    fn run_plan_unknown_task_errors() {
        let out_dir = std::path::Path::new(".");
        let err = run_plan(
            None, "small", None, 2, "bogus-task", None, out_dir, "auto", 1, false, 1, None, None,
            None, None, None, "text", false,
        )
        .unwrap_err();
        match err {
            CliError::ValidationFailed(m) => assert!(m.contains("Unknown task type")),
            _ => panic!("expected ValidationFailed"),
        }
    }

    #[test]
    fn run_plan_classify_task_not_available() {
        let out_dir = std::path::Path::new(".");
        let err = run_plan(
            None, "small", None, 2, "classify", None, out_dir, "auto", 1, false, 1, None, None,
            None, None, None, "text", false,
        )
        .unwrap_err();
        match err {
            CliError::ValidationFailed(m) => assert!(m.contains("classify")),
            _ => panic!("expected ValidationFailed"),
        }
    }

    #[test]
    fn run_plan_pretrain_without_config_errors() {
        let out_dir = std::path::Path::new(".");
        let err = run_plan(
            None, "small", None, 2, "pretrain", None, out_dir, "auto", 1, false, 1, None, None,
            None, None, None, "text", false,
        )
        .unwrap_err();
        match err {
            CliError::ValidationFailed(m) => assert!(m.contains("--config")),
            _ => panic!("expected ValidationFailed about missing config"),
        }
    }

    #[test]
    fn run_apply_unknown_task_errors() {
        let out_dir = std::path::Path::new(".");
        let err = run_apply(
            None, None, "nope", None, "small", None, 2, Some(out_dir), "auto", 1, false, 1, None,
            None, None, false, false, None, None, None, false, None,
        )
        .unwrap_err();
        assert!(matches!(err, CliError::ValidationFailed(_)));
    }

    #[test]
    fn run_apply_pretrain_missing_config_errors() {
        let out_dir = std::path::Path::new(".");
        let err = run_apply(
            None, None, "pretrain", None, "small", None, 2, Some(out_dir), "auto", 1, false, 1,
            None, None, None, false, false, None, None, None, false, None,
        )
        .unwrap_err();
        match err {
            CliError::ValidationFailed(m) => assert!(m.contains("--config")),
            _ => panic!("expected missing config error"),
        }
    }

    #[test]
    fn run_apply_pretrain_nonexistent_config_is_file_not_found() {
        let out_dir = std::path::Path::new(".");
        let missing = std::path::Path::new("/nonexistent/apr-pmat125/config.yaml");
        let err = run_apply(
            None,
            Some(missing),
            "pretrain",
            None,
            "small",
            None,
            2,
            Some(out_dir),
            "auto",
            1,
            false,
            1,
            None,
            None,
            None,
            false,
            false,
            None,
            None,
            None,
            false,
            None,
        )
        .unwrap_err();
        assert!(matches!(err, CliError::FileNotFound(_)));
    }

    // ========================================================================
    // PMAT-125 B1: run_sweep / run_halving / run_archive — path validation
    // ========================================================================

    #[test]
    fn run_sweep_missing_config_is_file_not_found() {
        let err = run_sweep(
            std::path::Path::new("/nonexistent/apr-pmat125/base.yaml"),
            "grid",
            4,
            std::path::Path::new("/tmp/apr-pmat125-out"),
            0,
            true,
        )
        .unwrap_err();
        assert!(matches!(err, CliError::FileNotFound(_)));
    }

    #[test]
    fn run_halving_missing_dir_is_file_not_found() {
        let err = run_halving(
            std::path::Path::new("/nonexistent/apr-pmat125/sweeps"),
            2,
            10,
            256,
            1024,
            std::path::Path::new("/tmp/apr-pmat125-halving.json"),
            true,
        )
        .unwrap_err();
        assert!(matches!(err, CliError::FileNotFound(_)));
    }

    #[test]
    fn run_archive_non_directory_errors() {
        let err = run_archive(
            std::path::Path::new("/nonexistent/apr-pmat125/ckpt"),
            std::path::Path::new("/tmp/apr-pmat125-arch-out"),
            Some("1.0.0"),
            None,
            true,
        )
        .unwrap_err();
        match err {
            CliError::ValidationFailed(m) => assert!(m.contains("Not a directory")),
            _ => panic!("expected ValidationFailed for non-directory source"),
        }
    }

    #[test]
    fn run_sweep_generates_grid_files() {
        let base = std::env::temp_dir().join(format!("apr-pmat125-rsweep-{}", std::process::id()));
        std::fs::create_dir_all(&base).unwrap();
        let cfg = base.join("base.yaml");
        std::fs::write(
            &cfg,
            "optimizer:\n  lr: 0.1\ndata:\n  batch_size: 2\ntraining:\n  warmup_steps: 10\n",
        )
        .unwrap();
        let out = base.join("out");
        run_sweep(&cfg, "grid", 3, &out, 0, true).expect("sweep should succeed");
        // Three sweep-NNN.yaml files were generated.
        let generated = discover_sweep_configs(&out).unwrap();
        assert_eq!(generated.len(), 3);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn run_archive_copies_files_and_writes_manifest() {
        let base = std::env::temp_dir().join(format!("apr-pmat125-rarch-{}", std::process::id()));
        let ckpt = base.join("ckpt");
        let out = base.join("out");
        std::fs::create_dir_all(&ckpt).unwrap();
        std::fs::write(ckpt.join("model.bin"), b"weights").unwrap();
        run_archive(&ckpt, &out, Some("2.1.0"), Some("note"), true).expect("archive ok");
        let manifest = std::fs::read_to_string(out.join("MANIFEST.json")).unwrap();
        let v: serde_json::Value = serde_json::from_str(&manifest).unwrap();
        assert_eq!(v["version"].as_str(), Some("2.1.0"));
        assert_eq!(v["notes"].as_str(), Some("note"));
        assert_eq!(v["total_bytes"].as_u64(), Some(7));
        assert!(out.join("model.bin").exists());
        let _ = std::fs::remove_dir_all(&base);
    }

    // ========================================================================
    // train halving — a WINNER requires evidence (#2374 finding 10)
    // ========================================================================

    fn halving_entry(name: &str, best_ppl: f64, failure: Option<&str>) -> HalvingEntry {
        HalvingEntry {
            path: std::path::PathBuf::from(name),
            best_ppl,
            lr: 5.0e-4,
            weight_decay: 0.01,
            warmup_steps: 100,
            eliminated_round: None,
            last_failure: failure.map(str::to_string),
        }
    }

    /// A trial process that exited non-zero produced no evidence. It must be
    /// reported as FAILED, not silently scored — `Command::output()` returning
    /// `Ok` only means the process spawned.
    #[test]
    fn classify_trial_nonzero_exit_is_a_failure_not_a_score() {
        let combined = "Loading JSON: d.json\nerror: Validation failed: Training failed: \
                        I/O error: No such file or directory (os error 2)\n";
        match classify_trial(false, Some(5), combined) {
            TrialOutcome::Failed(reason) => {
                assert!(reason.contains("exit 5"), "reason lost the exit code: {reason}");
                assert!(
                    reason.contains("No such file or directory"),
                    "reason lost the trial's own error: {reason}"
                );
            }
            TrialOutcome::Scored(p) => panic!("failed trial scored {p}"),
            TrialOutcome::NoEval => panic!("failed trial reported as a clean no-eval"),
        }
    }

    /// A trial that exits 0 but prints no val_ppl is "no eval" — distinct from
    /// a failure, and still not a score.
    #[test]
    fn classify_trial_clean_exit_without_eval_is_no_eval() {
        assert!(matches!(
            classify_trial(true, Some(0), "Training complete\n"),
            TrialOutcome::NoEval
        ));
    }

    /// A trial that exits 0 and reports val_ppl keeps the BEST (lowest) value.
    #[test]
    fn classify_trial_scores_best_val_ppl_on_clean_exit() {
        let out = "[eval] step=1 val_loss=2.0 val_ppl=7.39\n[eval] step=2 val_loss=1.5 val_ppl=4.48\n";
        match classify_trial(true, Some(0), out) {
            TrialOutcome::Scored(p) => assert!((p - 4.48).abs() < 1e-9, "got {p}"),
            other => panic!("clean scored trial misclassified: {:?}", match other {
                TrialOutcome::Failed(r) => r,
                _ => "no-eval".to_string(),
            }),
        }
    }

    /// The defect: three trials each exited 5, every best_ppl stayed infinite,
    /// and halving still printed `═══ WINNER ═══ sweep-000.yaml` with a
    /// μTransfer LR and exited 0. With no finite score there is no winner.
    #[test]
    fn select_halving_winner_refuses_to_crown_all_failed_trials() {
        let results = vec![
            halving_entry("sweep-000.yaml", f64::INFINITY, Some("exit 5: I/O error")),
            halving_entry("sweep-001.yaml", f64::INFINITY, Some("exit 5: I/O error")),
            halving_entry("sweep-002.yaml", f64::INFINITY, Some("exit 5: I/O error")),
        ];
        let err = select_halving_winner(&results, &[0, 1, 2])
            .expect_err("all-failed trials must not produce a winner");
        let msg = err.to_string();
        assert!(msg.contains("no halving winner"), "unhelpful error: {msg}");
        assert!(msg.contains("sweep-000.yaml"), "error names no failing config: {msg}");
        assert!(msg.contains("exit 5"), "error drops the trial exit status: {msg}");
    }

    /// All trials exiting 0 with no eval line is also not a result.
    #[test]
    fn select_halving_winner_refuses_when_no_trial_printed_a_val_ppl() {
        let results = vec![
            halving_entry("sweep-000.yaml", f64::INFINITY, None),
            halving_entry("sweep-001.yaml", f64::INFINITY, None),
        ];
        let err = select_halving_winner(&results, &[0, 1]).expect_err("no scores ⇒ no winner");
        assert!(err.to_string().contains("val_ppl"), "{err}");
    }

    /// The healthy path still works: the surviving entry with a finite score
    /// wins, and an infinite-scored survivor ahead of it is skipped.
    #[test]
    fn select_halving_winner_picks_the_scored_survivor() {
        let results = vec![
            halving_entry("sweep-000.yaml", f64::INFINITY, Some("exit 5: boom")),
            halving_entry("sweep-001.yaml", 12.5, None),
        ];
        assert_eq!(
            select_halving_winner(&results, &[0, 1]).expect("a scored survivor exists"),
            1
        );
    }
}
