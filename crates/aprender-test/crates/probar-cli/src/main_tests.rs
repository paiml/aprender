    use super::*;
    use probador::{
        ConfigArgs, CoverageArgs, InitArgs, PaletteArg, RecordArgs, RecordFormat, ReportArgs,
        ReportFormat,
    };
    use std::path::PathBuf;

    mod build_config_tests {
        use super::*;

        #[test]
        fn test_build_config_default() {
            let cli = Cli::parse_from(["probar", "test"]);
            let config = build_config(&cli);
            assert_eq!(config.verbosity, Verbosity::Normal);
        }

        #[test]
        fn test_build_config_verbose() {
            let cli = Cli::parse_from(["probar", "-v", "test"]);
            let config = build_config(&cli);
            assert_eq!(config.verbosity, Verbosity::Verbose);
        }

        #[test]
        fn test_build_config_debug() {
            let cli = Cli::parse_from(["probar", "-vv", "test"]);
            let config = build_config(&cli);
            assert_eq!(config.verbosity, Verbosity::Debug);
        }

        #[test]
        fn test_build_config_very_verbose() {
            let cli = Cli::parse_from(["probar", "-vvv", "test"]);
            let config = build_config(&cli);
            assert_eq!(config.verbosity, Verbosity::Debug);
        }

        #[test]
        fn test_build_config_quiet() {
            let cli = Cli::parse_from(["probar", "-q", "test"]);
            let config = build_config(&cli);
            assert_eq!(config.verbosity, Verbosity::Quiet);
        }

        #[test]
        fn test_build_config_color_never() {
            let cli = Cli::parse_from(["probar", "--color", "never", "test"]);
            let config = build_config(&cli);
            assert_eq!(config.color, ColorChoice::Never);
        }

        #[test]
        fn test_build_config_color_always() {
            let cli = Cli::parse_from(["probar", "--color", "always", "test"]);
            let config = build_config(&cli);
            assert_eq!(config.color, ColorChoice::Always);
        }
    }

    mod run_record_tests {
        use super::*;

        #[test]
        fn test_run_record() {
            let config = CliConfig::default();
            let args = RecordArgs {
                test: "my_test".to_string(),
                format: RecordFormat::Gif,
                output: None,
                fps: 10,
                quality: 80,
            };
            run_record(&config, &args);
            // Just verify it doesn't panic
        }

        #[test]
        fn test_run_record_png() {
            let config = CliConfig::default();
            let args = RecordArgs {
                test: "another_test".to_string(),
                format: RecordFormat::Png,
                output: Some(PathBuf::from("output.png")),
                fps: 30,
                quality: 100,
            };
            run_record(&config, &args);
        }
    }

    mod run_report_tests {
        use super::*;

        #[test]
        fn test_run_report_html() {
            let config = CliConfig::default();
            let args = ReportArgs {
                format: ReportFormat::Html,
                output: PathBuf::from("/tmp/probar_test_report"),
                open: false,
            };
            run_report(&config, &args);
        }

        #[test]
        fn test_run_report_json() {
            let config = CliConfig::default();
            let args = ReportArgs {
                format: ReportFormat::Json,
                output: PathBuf::from("/tmp/probar_test_report.json"),
                open: false,
            };
            run_report(&config, &args);
        }

        #[test]
        fn test_run_report_with_open() {
            let config = CliConfig::default();
            let args = ReportArgs {
                format: ReportFormat::Html,
                output: PathBuf::from("/tmp/probar_test_report_open"),
                open: true,
            };
            run_report(&config, &args);
        }
    }

    mod run_init_tests {
        use super::*;
        use std::fs;

        #[test]
        fn test_run_init_basic() {
            let temp_dir = std::env::temp_dir().join("probar_init_test");
            let _ = fs::remove_dir_all(&temp_dir);

            let config = CliConfig::default();
            let args = InitArgs {
                path: temp_dir.clone(),
                force: false,
            };
            run_init(&config, &args);

            // Cleanup
            let _ = fs::remove_dir_all(&temp_dir);
        }

        #[test]
        fn test_run_init_force() {
            let temp_dir = std::env::temp_dir().join("probar_init_force_test");
            let _ = fs::remove_dir_all(&temp_dir);

            let config = CliConfig::default();
            let args = InitArgs {
                path: temp_dir.clone(),
                force: true,
            };
            run_init(&config, &args);

            // Run again with force
            run_init(&config, &args);

            // Cleanup
            let _ = fs::remove_dir_all(&temp_dir);
        }
    }

    mod run_config_tests {
        use super::*;

        #[test]
        fn test_run_config_show() {
            let config = CliConfig::default();
            let args = ConfigArgs {
                show: true,
                set: None,
                reset: false,
            };
            run_config(&config, &args);
        }

        #[test]
        fn test_run_config_set_valid() {
            let config = CliConfig::default();
            let args = ConfigArgs {
                show: false,
                set: Some("key=value".to_string()),
                reset: false,
            };
            run_config(&config, &args);
        }

        #[test]
        fn test_run_config_set_invalid() {
            let config = CliConfig::default();
            let args = ConfigArgs {
                show: false,
                set: Some("invalid_format".to_string()),
                reset: false,
            };
            run_config(&config, &args);
        }

        #[test]
        fn test_run_config_reset() {
            let config = CliConfig::default();
            let args = ConfigArgs {
                show: false,
                set: None,
                reset: true,
            };
            run_config(&config, &args);
        }

        #[test]
        fn test_run_config_all_flags() {
            let config = CliConfig::default();
            let args = ConfigArgs {
                show: true,
                set: Some("test=value".to_string()),
                reset: true,
            };
            run_config(&config, &args);
        }
    }

    mod run_tests_tests {
        use super::*;
        use probador::TestArgs;

        #[test]
        #[ignore = "Spawns cargo test --list subprocess - causes nested builds in CI"]
        fn test_run_tests_no_tests() {
            let config = CliConfig::default();
            let args = TestArgs {
                filter: None,
                parallel: 0,
                coverage: false,
                mutants: false,
                fail_fast: false,
                watch: false,
                timeout: 30000,
                output: PathBuf::from("target/probar"),
                skip_compile: true, // Skip compile in tests to avoid recursive cargo calls
            };
            // run_tests returns Ok when no tests are found
            let result = run_tests(config, &args);
            assert!(result.is_ok());
        }

        #[test]
        #[ignore = "Spawns cargo test --list subprocess - causes nested builds in CI"]
        fn test_run_tests_with_filter() {
            let config = CliConfig::default();
            let args = TestArgs {
                filter: Some("game::*".to_string()),
                parallel: 4,
                coverage: true,
                mutants: false,
                fail_fast: true,
                watch: false,
                timeout: 5000,
                output: PathBuf::from("target/test_output"),
                skip_compile: true, // Skip compile in tests to avoid recursive cargo calls
            };
            let result = run_tests(config, &args);
            assert!(result.is_ok());
        }
    }

    mod run_coverage_tests {
        use super::*;

        #[test]
        fn test_run_coverage_no_output() {
            let config = CliConfig::default();
            let args = CoverageArgs {
                png: None,
                json: None,
                palette: PaletteArg::Viridis,
                legend: false,
                gaps: false,
                title: None,
                width: 400,
                height: 300,
                input: None,
            };
            let result = run_coverage(&config, &args);
            assert!(result.is_ok());
        }

        #[test]
        fn test_run_coverage_with_png() {
            let temp_dir = std::env::temp_dir();
            let png_path = temp_dir.join("test_coverage.png");

            let config = CliConfig::default();
            let args = CoverageArgs {
                png: Some(png_path.clone()),
                json: None,
                palette: PaletteArg::Magma,
                legend: true,
                gaps: true,
                title: Some("Test Coverage".to_string()),
                width: 800,
                height: 600,
                input: None,
            };

            let result = run_coverage(&config, &args);
            assert!(result.is_ok());

            // Verify PNG was created
            assert!(png_path.exists());

            // Cleanup
            let _ = std::fs::remove_file(&png_path);
        }

        #[test]
        fn test_run_coverage_with_json() {
            let temp_dir = std::env::temp_dir();
            let json_path = temp_dir.join("test_coverage.json");

            let config = CliConfig::default();
            let args = CoverageArgs {
                png: None,
                json: Some(json_path.clone()),
                palette: PaletteArg::Heat,
                legend: false,
                gaps: false,
                title: None,
                width: 640,
                height: 480,
                input: None,
            };

            let result = run_coverage(&config, &args);
            assert!(result.is_ok());

            // Verify JSON was created
            assert!(json_path.exists());

            // Verify JSON content
            let content = std::fs::read_to_string(&json_path).unwrap();
            assert!(content.contains("overall_coverage"));

            // Cleanup
            let _ = std::fs::remove_file(&json_path);
        }

        // Tests for coverage, report generation, and gap cell detection
        // have been moved to handlers module for better testability
    }

    mod compliance_check_tests {
        use super::*;
        use std::fs;
        use tempfile::TempDir;

        #[test]
        fn test_check_c001_with_wasm_and_tests() {
            let temp = TempDir::new().unwrap();
            fs::write(temp.path().join("app.wasm"), b"wasm").unwrap();
            fs::write(temp.path().join("test.rs"), b"#[test]").unwrap();

            let result = check_c001_code_execution(temp.path());
            assert!(result.passed);
        }

        #[test]
        fn test_check_c001_no_wasm() {
            let temp = TempDir::new().unwrap();
            fs::write(temp.path().join("test.rs"), b"#[test]").unwrap();

            let result = check_c001_code_execution(temp.path());
            assert!(!result.passed);
        }

        #[test]
        fn test_check_c001_no_tests() {
            let temp = TempDir::new().unwrap();
            fs::write(temp.path().join("app.wasm"), b"wasm").unwrap();

            let result = check_c001_code_execution(temp.path());
            assert!(!result.passed);
        }

        #[test]
        fn test_check_c002_console_errors() {
            let result = check_c002_console_errors();
            assert!(result.passed);
        }

        #[test]
        fn test_check_c003_with_custom_elements() {
            let temp = TempDir::new().unwrap();
            let html = r#"<html><script>customElements.define('my-el', MyEl)</script></html>"#;
            fs::write(temp.path().join("index.html"), html).unwrap();

            let result = check_c003_custom_elements(temp.path());
            assert!(result.passed);
        }

        #[test]
        fn test_check_c003_with_wasm_element() {
            let temp = TempDir::new().unwrap();
            let html = r#"<html><wasm-app></wasm-app></html>"#;
            fs::write(temp.path().join("index.html"), html).unwrap();

            let result = check_c003_custom_elements(temp.path());
            assert!(result.passed);
        }

        #[test]
        fn test_check_c003_no_custom_elements() {
            let temp = TempDir::new().unwrap();
            let html = r#"<html><div>Hello</div></html>"#;
            fs::write(temp.path().join("index.html"), html).unwrap();

            let result = check_c003_custom_elements(temp.path());
            assert!(result.passed); // Still passes, just with different detail
        }

        #[test]
        fn test_check_c004_threading_modes() {
            let result = check_c004_threading_modes();
            assert!(result.passed);
        }

        #[test]
        fn test_check_c005_low_memory() {
            let result = check_c005_low_memory();
            assert!(result.passed);
        }

        #[test]
        fn test_check_c006_with_htaccess() {
            let temp = TempDir::new().unwrap();
            fs::write(temp.path().join(".htaccess"), "Header set").unwrap();

            let result = check_c006_headers(temp.path());
            assert!(result.passed);
        }

        #[test]
        fn test_check_c006_with_vercel() {
            let temp = TempDir::new().unwrap();
            fs::write(temp.path().join("vercel.json"), "{}").unwrap();

            let result = check_c006_headers(temp.path());
            assert!(result.passed);
        }

        #[test]
        fn test_check_c006_with_netlify() {
            let temp = TempDir::new().unwrap();
            fs::write(temp.path().join("netlify.toml"), "").unwrap();

            let result = check_c006_headers(temp.path());
            assert!(result.passed);
        }

        #[test]
        fn test_check_c006_with_headers_file() {
            let temp = TempDir::new().unwrap();
            fs::write(temp.path().join("_headers"), "/*\n  COOP").unwrap();

            let result = check_c006_headers(temp.path());
            assert!(result.passed);
        }

        #[test]
        fn test_check_c006_with_probar_config() {
            let temp = TempDir::new().unwrap();
            fs::write(
                temp.path().join("probar.toml"),
                "cross_origin_isolated = true",
            )
            .unwrap();

            let result = check_c006_headers(temp.path());
            assert!(result.passed);
        }

        #[test]
        fn test_check_c006_with_makefile() {
            let temp = TempDir::new().unwrap();
            fs::write(
                temp.path().join("Makefile"),
                "serve:\n\tprobador serve --cross-origin-isolated",
            )
            .unwrap();

            let result = check_c006_headers(temp.path());
            assert!(result.passed);
        }

        #[test]
        fn test_check_c006_no_config() {
            let temp = TempDir::new().unwrap();

            let result = check_c006_headers(temp.path());
            assert!(!result.passed);
        }

        #[test]
        fn test_check_c007_replay_hash() {
            let result = check_c007_replay_hash();
            assert!(result.passed);
        }

        #[test]
        fn test_check_c008_cache() {
            let result = check_c008_cache();
            assert!(result.passed);
        }

        #[test]
        fn test_check_c009_wasm_size_under_limit() {
            let temp = TempDir::new().unwrap();
            fs::write(temp.path().join("small.wasm"), vec![0u8; 1000]).unwrap();

            let result = check_c009_wasm_size(temp.path(), 10000);
            assert!(result.passed);
        }

        #[test]
        fn test_check_c009_wasm_size_over_limit() {
            let temp = TempDir::new().unwrap();
            fs::write(temp.path().join("large.wasm"), vec![0u8; 10000]).unwrap();

            let result = check_c009_wasm_size(temp.path(), 1000);
            assert!(!result.passed);
        }

        #[test]
        fn test_check_c009_no_wasm() {
            let temp = TempDir::new().unwrap();

            let result = check_c009_wasm_size(temp.path(), 10000);
            assert!(result.passed);
        }

        #[test]
        fn test_check_c010_with_panic_abort() {
            let temp = TempDir::new().unwrap();
            let cargo = r#"[profile.release]
panic = "abort""#;
            fs::write(temp.path().join("Cargo.toml"), cargo).unwrap();

            let result = check_c010_panic_paths(temp.path());
            assert!(result.passed);
            assert!(result
                .details
                .iter()
                .any(|d| d.contains("panic = \"abort\"")));
        }

        #[test]
        fn test_check_c010_without_panic_abort() {
            let temp = TempDir::new().unwrap();
            fs::write(temp.path().join("Cargo.toml"), "[package]\nname = \"test\"").unwrap();

            let result = check_c010_panic_paths(temp.path());
            assert!(result.passed); // Still passes with different detail
        }
    }

    // NOTE: file_finder_tests moved to handlers/comply.rs

    mod cross_origin_config_tests {
        use super::*;
        use std::fs;
        use tempfile::TempDir;

        #[test]
        fn test_check_probar_cross_origin_config_true() {
            let temp = TempDir::new().unwrap();
            fs::write(
                temp.path().join("probar.toml"),
                "cross_origin_isolated = true",
            )
            .unwrap();

            assert!(check_probar_cross_origin_config(temp.path()));
        }

        #[test]
        fn test_check_probar_cross_origin_config_no_space() {
            let temp = TempDir::new().unwrap();
            fs::write(
                temp.path().join("probar.toml"),
                "cross_origin_isolated=true",
            )
            .unwrap();

            assert!(check_probar_cross_origin_config(temp.path()));
        }

        #[test]
        fn test_check_probar_cross_origin_config_dot_prefixed() {
            let temp = TempDir::new().unwrap();
            fs::write(
                temp.path().join(".probar.toml"),
                "cross_origin_isolated = true",
            )
            .unwrap();

            assert!(check_probar_cross_origin_config(temp.path()));
        }

        #[test]
        fn test_check_probar_cross_origin_config_probador() {
            let temp = TempDir::new().unwrap();
            fs::write(
                temp.path().join("probador.toml"),
                "cross_origin_isolated = true",
            )
            .unwrap();

            assert!(check_probar_cross_origin_config(temp.path()));
        }

        #[test]
        fn test_check_probar_cross_origin_config_false() {
            let temp = TempDir::new().unwrap();
            fs::write(
                temp.path().join("probar.toml"),
                "cross_origin_isolated = false",
            )
            .unwrap();

            assert!(!check_probar_cross_origin_config(temp.path()));
        }

        #[test]
        fn test_check_probar_cross_origin_config_missing() {
            let temp = TempDir::new().unwrap();

            assert!(!check_probar_cross_origin_config(temp.path()));
        }

        #[test]
        fn test_check_makefile_cross_origin_probador() {
            let temp = TempDir::new().unwrap();
            fs::write(
                temp.path().join("Makefile"),
                "serve:\n\tprobador serve --cross-origin-isolated",
            )
            .unwrap();

            assert!(check_makefile_cross_origin(temp.path()));
        }

        #[test]
        fn test_check_makefile_cross_origin_probar() {
            let temp = TempDir::new().unwrap();
            fs::write(
                temp.path().join("Makefile"),
                "serve:\n\tprobar serve --cross-origin-isolated",
            )
            .unwrap();

            assert!(check_makefile_cross_origin(temp.path()));
        }

        #[test]
        fn test_check_makefile_cross_origin_lowercase() {
            let temp = TempDir::new().unwrap();
            fs::write(
                temp.path().join("makefile"),
                "serve:\n\tprobador serve --cross-origin-isolated",
            )
            .unwrap();

            assert!(check_makefile_cross_origin(temp.path()));
        }

        #[test]
        fn test_check_makefile_cross_origin_gnu() {
            let temp = TempDir::new().unwrap();
            fs::write(
                temp.path().join("GNUmakefile"),
                "serve:\n\tprobador serve --cross-origin-isolated",
            )
            .unwrap();

            assert!(check_makefile_cross_origin(temp.path()));
        }

        #[test]
        fn test_check_makefile_cross_origin_package_json() {
            let temp = TempDir::new().unwrap();
            let pkg = r#"{"scripts": {"serve": "probador serve --cross-origin-isolated"}}"#;
            fs::write(temp.path().join("package.json"), pkg).unwrap();

            assert!(check_makefile_cross_origin(temp.path()));
        }

        #[test]
        fn test_check_makefile_cross_origin_without_flag() {
            let temp = TempDir::new().unwrap();
            fs::write(temp.path().join("Makefile"), "serve:\n\tprobador serve").unwrap();

            assert!(!check_makefile_cross_origin(temp.path()));
        }

        #[test]
        fn test_check_makefile_cross_origin_missing() {
            let temp = TempDir::new().unwrap();

            assert!(!check_makefile_cross_origin(temp.path()));
        }
    }

    // NOTE: compliance_result_tests moved to handlers/comply.rs
