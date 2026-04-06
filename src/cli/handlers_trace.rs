
/// Parse trace configuration from CLI argument
pub fn parse_trace_config(
    trace: Option<Option<String>>,
) -> Option<crate::inference_trace::TraceConfig> {
    match trace {
        Some(Some(steps)) => {
            // --trace=step1,step2
            let mut config = crate::inference_trace::TraceConfig::enabled();
            config.steps = crate::inference_trace::TraceConfig::parse_steps(&steps);
            config.verbose = true;
            Some(config)
        },
        Some(None) => {
            // --trace (no value, trace all)
            let mut config = crate::inference_trace::TraceConfig::enabled();
            config.verbose = true;
            Some(config)
        },
        None => None,
    }
}

/// Validate that a model path exists and is readable
pub fn validate_model_path(model_path: &str) -> Result<()> {
    let path = std::path::Path::new(model_path);
    if !path.exists() {
        return Err(RealizarError::ModelNotFound(model_path.to_string()));
    }
    if !path.is_file() {
        return Err(RealizarError::UnsupportedOperation {
            operation: "validate_model".to_string(),
            reason: format!("{model_path} is not a file"),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parse_info_command() {
        let cli = Cli::try_parse_from(["realizar", "info"]).expect("cli");
        assert!(matches!(cli.command, Commands::Info));
    }

    #[test]
    fn test_cli_parse_run_command() {
        let cli = Cli::try_parse_from(["realizar", "run", "model.gguf", "hello"]).expect("cli");
        match cli.command {
            Commands::Run { model, prompt, .. } => {
                assert_eq!(model, "model.gguf");
                assert_eq!(prompt, Some("hello".to_string()));
            },
            _ => panic!("Expected Run command"),
        }
    }

    #[test]
    fn test_cli_parse_serve_demo() {
        let cli = Cli::try_parse_from(["realizar", "serve", "--demo"]).expect("cli");
        match cli.command {
            Commands::Serve { demo, .. } => {
                assert!(demo);
            },
            _ => panic!("Expected Serve command"),
        }
    }

    #[test]
    fn test_cli_parse_serve_with_model() {
        let cli =
            Cli::try_parse_from(["realizar", "serve", "--model", "test.gguf", "--gpu"]).expect("expected value");
        match cli.command {
            Commands::Serve { model, gpu, .. } => {
                assert_eq!(model, Some("test.gguf".to_string()));
                assert!(gpu);
            },
            _ => panic!("Expected Serve command"),
        }
    }

    #[test]
    fn test_cli_parse_list_json() {
        let cli = Cli::try_parse_from(["realizar", "list", "--format", "json"]).expect("cli");
        match cli.command {
            Commands::List { format, .. } => {
                assert_eq!(format, "json");
            },
            _ => panic!("Expected List command"),
        }
    }

    #[test]
    fn test_cli_parse_bench_list() {
        let cli = Cli::try_parse_from(["realizar", "bench", "--list"]).expect("cli");
        match cli.command {
            Commands::Bench { list, .. } => {
                assert!(list);
            },
            _ => panic!("Expected Bench command"),
        }
    }

    #[test]
    fn test_cli_parse_viz() {
        let cli = Cli::try_parse_from(["realizar", "viz", "--color", "--samples", "50"]).expect("cli");
        match cli.command {
            Commands::Viz { color, samples } => {
                assert!(color);
                assert_eq!(samples, 50);
            },
            _ => panic!("Expected Viz command"),
        }
    }

    #[test]
    fn test_serve_config_default() {
        let config = ServeConfig {
            host: "127.0.0.1".to_string(),
            port: 8080,
            model: None,
            demo: true,
            batch: false,
            gpu: false,
            openai_api: true,
            trace: false,
            context_length: 4096,
            no_fp8_cache: false,
        };
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 8080);
        assert!(config.demo);
        assert!(!config.trace);
    }

    #[test]
    fn test_run_config_creation() {
        let config = RunConfig {
            model: "test.gguf".to_string(),
            prompt: Some("hello".to_string()),
            max_tokens: 256,
            temperature: 0.7,
            format: "text".to_string(),
            system: None,
            raw: false,
            gpu: false,
            verbose: false,
            trace: None,
        };
        assert_eq!(config.model, "test.gguf");
        assert_eq!(config.max_tokens, 256);
    }

    #[test]
    fn test_parse_trace_config_none() {
        let result = parse_trace_config(None);
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_trace_config_enabled() {
        let result = parse_trace_config(Some(None));
        assert!(result.is_some());
        let config = result.expect("config");
        assert!(config.verbose);
    }

    #[test]
    fn test_parse_trace_config_with_steps() {
        let result = parse_trace_config(Some(Some("attention,ffn".to_string())));
        assert!(result.is_some());
        let config = result.expect("config");
        assert!(config.verbose);
    }

    #[test]
    fn test_validate_model_path_not_found() {
        let result = validate_model_path("/nonexistent/path/model.gguf");
        assert!(result.is_err());
    }

    #[test]
    fn test_handle_list_no_models() {
        // This test just verifies the function doesn't panic
        // It will print output about no models found
        let result = handle_list(None, "table");
        assert!(result.is_ok());
    }

    #[test]
    fn test_handle_list_json_format() {
        let result = handle_list(None, "json");
        assert!(result.is_ok());
    }
}
