
    // ════════════════════════════════════════════════════════════════════
    // Coverage tests for dispatch_model_commands, dispatch_analysis_commands,
    // dispatch_profiling_commands, dispatch_train_command (PMAT coverage gap)
    // ════════════════════════════════════════════════════════════════════

    // ── dispatch_model_commands ──────────────────────────────────────────

    #[test]
    fn test_dispatch_model_commands_returns_none_for_inspect() {
        // Inspect is an inspection command, not a model management command
        let cli = make_cli(Commands::Inspect {
            file: PathBuf::from("model.apr"),
            vocab: false,
            filters: false,
            weights: false,
            json: false,
        });
        let result = dispatch_model_commands(&cli);
        assert!(result.is_none(), "Inspect should not be handled by dispatch_model_commands");
    }

    #[test]
    fn test_dispatch_model_commands_returns_none_for_list_command() {
        // List IS a model command — should return Some
        let cli = make_cli(Commands::List);
        let result = dispatch_model_commands(&cli);
        assert!(result.is_some(), "List should be handled by dispatch_model_commands");
    }

    #[test]
    fn test_dispatch_model_commands_pull_nonexistent() {
        let cli = make_cli(Commands::Pull {
            model_ref: "nonexistent-model-that-does-not-exist-xyz123".to_string(),
            force: false,
            dry_run: false,
            revision: None,
            offline: false,
        });
        let result = dispatch_model_commands(&cli);
        assert!(result.is_some(), "Pull should be handled by dispatch_model_commands");
        // Pull may succeed (network) or fail (no network) — we just test routing
    }

    #[test]
    fn test_dispatch_model_commands_rm_nonexistent() {
        let cli = make_cli(Commands::Rm {
            model_ref: "nonexistent-model-xyz789".to_string(),
        });
        let result = dispatch_model_commands(&cli);
        assert!(result.is_some(), "Rm should be handled by dispatch_model_commands");
    }

    #[test]
    fn test_dispatch_model_commands_tui_none_file() {
        let cli = make_cli(Commands::Tui { file: None });
        let result = dispatch_model_commands(&cli);
        assert!(result.is_some(), "Tui should be handled by dispatch_model_commands");
    }

    #[test]
    fn test_dispatch_model_commands_tui_with_file() {
        let cli = make_cli(Commands::Tui {
            file: Some(PathBuf::from("/tmp/nonexistent_tui_model.apr")),
        });
        let result = dispatch_model_commands(&cli);
        assert!(result.is_some(), "Tui with file should be handled by dispatch_model_commands");
    }

    #[test]
    fn test_dispatch_model_commands_merge_nonexistent() {
        let cli = make_cli(Commands::Merge {
            files: vec![
                PathBuf::from("/tmp/nonexistent_merge_a.apr"),
                PathBuf::from("/tmp/nonexistent_merge_b.apr"),
            ],
            strategy: "average".to_string(),
            output: Some(PathBuf::from("/tmp/merged_out.apr")),
            weights: None,
            base_model: None,
            drop_rate: 0.9,
            density: 0.2,
            seed: 42,
            plan: false,
        });
        let result = dispatch_model_commands(&cli);
        assert!(result.is_some(), "Merge should be handled by dispatch_model_commands");
        // merge with nonexistent files should error
        assert!(result.expect("should be some").is_err());
    }

    #[test]
    fn test_dispatch_model_commands_merge_plan_mode() {
        let cli = make_cli(Commands::Merge {
            files: vec![
                PathBuf::from("/tmp/nonexistent_merge_plan_a.apr"),
                PathBuf::from("/tmp/nonexistent_merge_plan_b.apr"),
            ],
            strategy: "slerp".to_string(),
            output: None, // plan mode doesn't need output
            weights: Some(vec![0.5, 0.5]),
            base_model: None,
            drop_rate: 0.9,
            density: 0.2,
            seed: 42,
            plan: true,
        });
        let result = dispatch_model_commands(&cli);
        assert!(result.is_some(), "Merge plan should be handled by dispatch_model_commands");
    }

    #[test]
    fn test_dispatch_model_commands_returns_none_for_validate() {
        let cli = make_cli(Commands::Validate {
            file: PathBuf::from("model.apr"),
            quality: false,
            strict: false,
            min_score: None,
        });
        let result = dispatch_model_commands(&cli);
        assert!(result.is_none(), "Validate should not be handled by dispatch_model_commands");
    }

    #[test]
    fn test_dispatch_model_commands_returns_none_for_debug() {
        let cli = make_cli(Commands::Debug {
            file: PathBuf::from("model.apr"),
            drama: false,
            hex: false,
            strings: false,
            limit: 256,
        });
        let result = dispatch_model_commands(&cli);
        assert!(result.is_none(), "Debug should not be handled by dispatch_model_commands");
    }

    #[test]
    fn test_dispatch_model_commands_prune_nonexistent() {
        let cli = make_cli(Commands::ModelOps(ModelOpsCommands::Prune {
            file: PathBuf::from("/tmp/nonexistent_prune_model.apr"),
            method: "magnitude".to_string(),
            target_ratio: 0.5,
            sparsity: 0.0,
            output: Some(PathBuf::from("/tmp/pruned_out.apr")),
            remove_layers: None,
            analyze: false,
            plan: false,
            calibration: None,
        }));
        let result = dispatch_model_commands(&cli);
        assert!(result.is_some(), "Prune should be handled by dispatch_model_commands");
        assert!(result.expect("should be some").is_err(), "Prune with nonexistent file should error");
    }

include!("lib_dispatch_coverage_analysis.rs");
include!("lib_dispatch_coverage_profiling.rs");
include!("lib_dispatch_coverage_train.rs");

