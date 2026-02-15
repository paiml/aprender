
    /// Test Serve command defaults
    #[test]
    fn test_parse_serve_defaults() {
        let args = vec!["apr", "serve", "model.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Serve {
                port,
                host,
                no_cors,
                no_metrics,
                no_gpu,
                gpu,
                batch,
                trace,
                trace_level,
                profile,
                ..
            } => {
                assert_eq!(port, 8080);
                assert_eq!(host, "127.0.0.1");
                assert!(!no_cors);
                assert!(!no_metrics);
                assert!(!no_gpu);
                assert!(!gpu);
                assert!(!batch);
                assert!(!trace);
                assert_eq!(trace_level, "basic");
                assert!(!profile);
            }
            _ => panic!("Expected Serve command"),
        }
    }

    /// Test Bench command defaults
    #[test]
    fn test_parse_bench_defaults() {
        let args = vec!["apr", "bench", "model.gguf"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Bench {
                warmup,
                iterations,
                max_tokens,
                prompt,
                fast,
                brick,
                ..
            } => {
                assert_eq!(warmup, 3);
                assert_eq!(iterations, 5);
                assert_eq!(max_tokens, 32);
                assert!(prompt.is_none());
                assert!(!fast);
                assert!(brick.is_none());
            }
            _ => panic!("Expected Bench command"),
        }
    }

    /// Test Cbtop command defaults
    #[test]
    fn test_parse_cbtop_defaults() {
        let args = vec!["apr", "cbtop"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Cbtop {
                model,
                attach,
                model_path,
                headless,
                json,
                output,
                ci,
                throughput,
                brick_score,
                warmup,
                iterations,
                speculative,
                speculation_k,
                draft_model,
                concurrent,
                simulated,
            } => {
                assert!(model.is_none());
                assert!(attach.is_none());
                assert!(model_path.is_none());
                assert!(!headless);
                assert!(!json);
                assert!(output.is_none());
                assert!(!ci);
                assert!(throughput.is_none());
                assert!(brick_score.is_none());
                assert_eq!(warmup, 10);
                assert_eq!(iterations, 100);
                assert!(!speculative);
                assert_eq!(speculation_k, 4);
                assert!(draft_model.is_none());
                assert_eq!(concurrent, 1);
                assert!(!simulated);
            }
            _ => panic!("Expected Cbtop command"),
        }
    }

    /// Test Profile command defaults
    #[test]
    fn test_parse_profile_defaults() {
        let args = vec!["apr", "profile", "model.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Profile {
                granular,
                format,
                focus,
                detect_naive,
                threshold,
                compare_hf,
                energy,
                perf_grade,
                callgraph,
                fail_on_naive,
                output,
                ci,
                assert_throughput,
                assert_p99,
                assert_p50,
                warmup,
                measure,
                ..
            } => {
                assert!(!granular);
                assert_eq!(format, "human");
                assert!(focus.is_none());
                assert!(!detect_naive);
                assert!((threshold - 10.0).abs() < f64::EPSILON);
                assert!(compare_hf.is_none());
                assert!(!energy);
                assert!(!perf_grade);
                assert!(!callgraph);
                assert!(!fail_on_naive);
                assert!(output.is_none());
                assert!(!ci);
                assert!(assert_throughput.is_none());
                assert!(assert_p99.is_none());
                assert!(assert_p50.is_none());
                assert_eq!(warmup, 3);
                assert_eq!(measure, 10);
            }
            _ => panic!("Expected Profile command"),
        }
    }

    /// Test Qa command defaults
    #[test]
    fn test_parse_qa_defaults() {
        let args = vec!["apr", "qa", "model.gguf"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Qa {
                assert_tps,
                assert_speedup,
                assert_gpu_speedup,
                skip_golden,
                skip_throughput,
                skip_ollama,
                skip_gpu_speedup,
                skip_contract,
                skip_format_parity,
                safetensors_path,
                iterations,
                warmup,
                max_tokens,
                json,
                verbose,
                ..
            } => {
                assert!(assert_tps.is_none());
                assert!(assert_speedup.is_none());
                assert!(assert_gpu_speedup.is_none());
                assert!(!skip_golden);
                assert!(!skip_throughput);
                assert!(!skip_ollama);
                assert!(!skip_gpu_speedup);
                assert!(!skip_contract);
                assert!(!skip_format_parity);
                assert!(safetensors_path.is_none());
                assert_eq!(iterations, 10);
                assert_eq!(warmup, 3);
                assert_eq!(max_tokens, 32);
                assert!(!json);
                assert!(!verbose);
            }
            _ => panic!("Expected Qa command"),
        }
    }

    /// Test Chat command defaults
    #[test]
    fn test_parse_chat_defaults() {
        let args = vec!["apr", "chat", "model.gguf"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Chat {
                temperature,
                top_p,
                max_tokens,
                system,
                inspect,
                no_gpu,
                gpu,
                trace,
                trace_steps,
                trace_verbose,
                trace_output,
                trace_level,
                profile,
                ..
            } => {
                assert!((temperature - 0.7).abs() < f32::EPSILON);
                assert!((top_p - 0.9).abs() < f32::EPSILON);
                assert_eq!(max_tokens, 512);
                assert!(system.is_none());
                assert!(!inspect);
                assert!(!no_gpu);
                assert!(!gpu);
                assert!(!trace);
                assert!(trace_steps.is_none());
                assert!(!trace_verbose);
                assert!(trace_output.is_none());
                assert_eq!(trace_level, "basic");
                assert!(!profile);
            }
            _ => panic!("Expected Chat command"),
        }
    }
