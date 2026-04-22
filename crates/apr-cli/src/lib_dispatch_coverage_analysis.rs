    // ── dispatch_analysis_commands ──────────────────────────────────────

    #[test]
    fn test_dispatch_analysis_returns_none_for_non_extended() {
        let cli = make_cli(Commands::List);
        let result = dispatch_analysis_commands(&cli);
        assert!(result.is_none(), "Non-extended command should not match analysis dispatcher");
    }

    #[test]
    fn test_dispatch_analysis_returns_none_for_chat() {
        // Chat is Extended but not an analysis command
        let cli = make_cli(Commands::Extended(ExtendedCommands::Chat {
            file: PathBuf::from("model.apr"),
            temperature: 0.7,
            top_p: 0.9,
            max_tokens: 512,
            system: None,
            inspect: false,
            no_gpu: true,
            gpu: false,
            trace: false,
            trace_steps: None,
            trace_verbose: false,
            trace_output: None,
            trace_level: "basic".to_string(),
            profile: false,
            backend: None,
        }));
        let result = dispatch_analysis_commands(&cli);
        assert!(result.is_none(), "Chat command should not be handled by analysis dispatcher");
    }

    #[test]
    fn test_dispatch_analysis_tree_nonexistent_file() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Forensics(ForensicsCommands::Tree {
            file: PathBuf::from("/tmp/nonexistent_tree_model.apr"),
            filter: None,
            format: "ascii".to_string(),
            sizes: false,
            depth: None,
        })));
        let result = dispatch_analysis_commands(&cli);
        assert!(result.is_some(), "Tree should be handled by analysis dispatcher");
        assert!(result.expect("should be some").is_err(), "Tree with nonexistent file should error");
    }

    #[test]
    fn test_dispatch_analysis_hex_nonexistent_file() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Forensics(ForensicsCommands::Hex {
            file: PathBuf::from("/tmp/nonexistent_hex_model.apr"),
            tensor: None,
            limit: 64,
            stats: false,
            list: false,
            json: false,
            header: false,
            blocks: false,
            distribution: false,
            contract: false,
            entropy: false,
            raw: false,
            offset: "0".to_string(),
            width: 16,
            slice: None,
        })));
        let result = dispatch_analysis_commands(&cli);
        assert!(result.is_some(), "Hex should be handled by analysis dispatcher");
        assert!(result.expect("should be some").is_err(), "Hex with nonexistent file should error");
    }

    #[test]
    fn test_dispatch_analysis_flow_nonexistent_file() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Forensics(ForensicsCommands::Flow {
            file: PathBuf::from("/tmp/nonexistent_flow_model.apr"),
            layer: None,
            component: "full".to_string(),
            verbose: false,
            json: false,
        })));
        let result = dispatch_analysis_commands(&cli);
        assert!(result.is_some(), "Flow should be handled by analysis dispatcher");
        assert!(result.expect("should be some").is_err(), "Flow with nonexistent file should error");
    }

    #[test]
    fn test_dispatch_analysis_qualify_nonexistent_file() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Forensics(ForensicsCommands::Qualify {
            file: PathBuf::from("/tmp/nonexistent_qualify_model.apr"),
            tier: "basic".to_string(),
            timeout: 30,
            json: false,
            verbose: false,
            skip: None,
        })));
        let result = dispatch_analysis_commands(&cli);
        assert!(result.is_some(), "Qualify should be handled by analysis dispatcher");
        assert!(result.expect("should be some").is_err(), "Qualify with nonexistent file should error");
    }

    #[test]
    fn test_dispatch_analysis_diagnose_nonexistent() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Training(
            TrainingCommands::Diagnose {
                checkpoint_dir: PathBuf::from("/tmp/nonexistent_diagnose_dir"),
                data: None,
                model_size: None,
                num_classes: 5,
            },
        )));
        let result = dispatch_analysis_commands(&cli);
        assert!(result.is_some(), "Diagnose should be handled by analysis dispatcher");
    }

    #[test]
    fn test_dispatch_analysis_compare_hf_offline_rejected() {
        let mut cli = make_cli(Commands::Extended(ExtendedCommands::Forensics(ForensicsCommands::CompareHf {
            file: PathBuf::from("/tmp/nonexistent_compare_hf.apr"),
            hf: "org/repo".to_string(),
            tensor: None,
            threshold: 0.01,
            json: false,
        })));
        cli.offline = true;
        let result = dispatch_analysis_commands(&cli);
        assert!(result.is_some(), "CompareHf should be handled by analysis dispatcher");
        let err = result.expect("should be some");
        assert!(err.is_err(), "CompareHf in offline mode should be rejected");
    }

    // ── dispatch_profiling_commands ─────────────────────────────────────
