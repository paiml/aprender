    use super::*;

    mod cli_tests {
        use super::*;

        #[test]
        fn test_parse_test_command() {
            let cli = Cli::parse_from(["probar", "test"]);
            assert!(matches!(cli.command, Commands::Test(_)));
        }

        #[test]
        fn test_parse_test_with_filter() {
            let cli = Cli::parse_from(["probar", "test", "--filter", "game::*"]);
            if let Commands::Test(args) = cli.command {
                assert_eq!(args.filter, Some("game::*".to_string()));
            } else {
                panic!("expected Test command");
            }
        }

        #[test]
        fn test_parse_test_with_parallel() {
            let cli = Cli::parse_from(["probar", "test", "-j", "4"]);
            if let Commands::Test(args) = cli.command {
                assert_eq!(args.parallel, 4);
            } else {
                panic!("expected Test command");
            }
        }

        #[test]
        fn test_parse_test_with_coverage() {
            let cli = Cli::parse_from(["probar", "test", "--coverage"]);
            if let Commands::Test(args) = cli.command {
                assert!(args.coverage);
            } else {
                panic!("expected Test command");
            }
        }

        #[test]
        fn test_parse_test_with_fail_fast() {
            let cli = Cli::parse_from(["probar", "test", "--fail-fast"]);
            if let Commands::Test(args) = cli.command {
                assert!(args.fail_fast);
            } else {
                panic!("expected Test command");
            }
        }

        #[test]
        fn test_parse_record_command() {
            let cli = Cli::parse_from(["probar", "record", "test_login"]);
            if let Commands::Record(args) = cli.command {
                assert_eq!(args.test, "test_login");
            } else {
                panic!("expected Record command");
            }
        }

        #[test]
        fn test_parse_record_with_format() {
            let cli = Cli::parse_from(["probar", "record", "test_login", "--format", "png"]);
            if let Commands::Record(args) = cli.command {
                assert!(matches!(args.format, RecordFormat::Png));
            } else {
                panic!("expected Record command");
            }
        }

        #[test]
        fn test_parse_report_command() {
            let cli = Cli::parse_from(["probar", "report"]);
            assert!(matches!(cli.command, Commands::Report(_)));
        }

        #[test]
        fn test_parse_report_with_format() {
            let cli = Cli::parse_from(["probar", "report", "--format", "lcov"]);
            if let Commands::Report(args) = cli.command {
                assert!(matches!(args.format, ReportFormat::Lcov));
            } else {
                panic!("expected Report command");
            }
        }

        #[test]
        fn test_parse_init_command() {
            let cli = Cli::parse_from(["probar", "init"]);
            assert!(matches!(cli.command, Commands::Init(_)));
        }

        #[test]
        fn test_parse_config_command() {
            let cli = Cli::parse_from(["probar", "config", "--show"]);
            if let Commands::Config(args) = cli.command {
                assert!(args.show);
            } else {
                panic!("expected Config command");
            }
        }

        #[test]
        fn test_global_verbose_flag() {
            let cli = Cli::parse_from(["probar", "-vvv", "test"]);
            assert_eq!(cli.verbose, 3);
        }

        #[test]
        fn test_global_quiet_flag() {
            let cli = Cli::parse_from(["probar", "-q", "test"]);
            assert!(cli.quiet);
        }

        #[test]
        fn test_global_color_flag() {
            let cli = Cli::parse_from(["probar", "--color", "never", "test"]);
            assert!(matches!(cli.color, ColorArg::Never));
        }
    }

    mod format_tests {
        use super::*;

        #[test]
        fn test_record_format_default() {
            let format = RecordFormat::default();
            assert!(matches!(format, RecordFormat::Gif));
        }

        #[test]
        fn test_report_format_default() {
            let format = ReportFormat::default();
            assert!(matches!(format, ReportFormat::Html));
        }

        #[test]
        fn test_color_arg_conversion() {
            use crate::config::ColorChoice;

            let auto: ColorChoice = ColorArg::Auto.into();
            assert!(matches!(auto, ColorChoice::Auto));

            let always: ColorChoice = ColorArg::Always.into();
            assert!(matches!(always, ColorChoice::Always));

            let never: ColorChoice = ColorArg::Never.into();
            assert!(matches!(never, ColorChoice::Never));
        }
    }

    mod record_format_tests {
        use super::*;

        #[test]
        fn test_default() {
            let format = RecordFormat::default();
            assert!(matches!(format, RecordFormat::Gif));
        }

        #[test]
        fn test_all_variants() {
            let _ = RecordFormat::Gif;
            let _ = RecordFormat::Png;
            let _ = RecordFormat::Svg;
            let _ = RecordFormat::Mp4;
        }

        #[test]
        fn test_debug() {
            let debug = format!("{:?}", RecordFormat::Gif);
            assert!(debug.contains("Gif"));
        }

        #[test]
        fn test_clone() {
            let format = RecordFormat::Mp4;
            let cloned = format;
            assert!(matches!(cloned, RecordFormat::Mp4));
        }
    }

    mod report_format_tests {
        use super::*;

        #[test]
        fn test_default() {
            let format = ReportFormat::default();
            assert!(matches!(format, ReportFormat::Html));
        }

        #[test]
        fn test_all_variants() {
            let _ = ReportFormat::Html;
            let _ = ReportFormat::Junit;
            let _ = ReportFormat::Lcov;
            let _ = ReportFormat::Cobertura;
            let _ = ReportFormat::Json;
        }

        #[test]
        fn test_debug() {
            let debug = format!("{:?}", ReportFormat::Junit);
            assert!(debug.contains("Junit"));
        }
    }

    mod test_args_tests {
        use super::*;

        #[test]
        fn test_defaults() {
            // Verify TestArgs can be created with defaults via clap
            let args = TestArgs {
                filter: None,
                parallel: 0,
                coverage: false,
                mutants: false,
                fail_fast: false,
                watch: false,
                timeout: 30000,
                output: PathBuf::from("target/probar"),
                skip_compile: false,
            };
            assert!(!args.coverage);
            assert_eq!(args.timeout, 30000);
        }

        #[test]
        fn test_debug() {
            let args = TestArgs {
                filter: Some("test_*".to_string()),
                parallel: 4,
                coverage: true,
                mutants: false,
                fail_fast: true,
                watch: false,
                timeout: 5000,
                output: PathBuf::from("target"),
                skip_compile: false,
            };
            let debug = format!("{args:?}");
            assert!(debug.contains("TestArgs"));
        }

        #[test]
        fn test_skip_compile_flag() {
            let args = TestArgs {
                filter: None,
                parallel: 0,
                coverage: false,
                mutants: false,
                fail_fast: false,
                watch: false,
                timeout: 30000,
                output: PathBuf::from("target/probar"),
                skip_compile: true,
            };
            assert!(args.skip_compile);
        }
    }

    mod record_args_tests {
        use super::*;

        #[test]
        fn test_creation() {
            let args = RecordArgs {
                test: "my_test".to_string(),
                format: RecordFormat::Gif,
                output: None,
                fps: 10,
                quality: 80,
            };
            assert_eq!(args.test, "my_test");
            assert_eq!(args.fps, 10);
        }

        #[test]
        fn test_debug() {
            let args = RecordArgs {
                test: "test".to_string(),
                format: RecordFormat::Png,
                output: Some(PathBuf::from("out.png")),
                fps: 30,
                quality: 100,
            };
            let debug = format!("{args:?}");
            assert!(debug.contains("RecordArgs"));
        }
    }

    mod report_args_tests {
        use super::*;

        #[test]
        fn test_creation() {
            let args = ReportArgs {
                format: ReportFormat::Lcov,
                output: PathBuf::from("coverage"),
                open: true,
            };
            assert!(args.open);
        }

        #[test]
        fn test_debug() {
            let args = ReportArgs {
                format: ReportFormat::Html,
                output: PathBuf::from("reports"),
                open: false,
            };
            let debug = format!("{args:?}");
            assert!(debug.contains("ReportArgs"));
        }
    }

    mod init_args_tests {
        use super::*;

        #[test]
        fn test_creation() {
            let args = InitArgs {
                path: PathBuf::from("."),
                force: false,
            };
            assert!(!args.force);
        }
    }

    mod config_args_tests {
        use super::*;

        #[test]
        fn test_creation() {
            let args = ConfigArgs {
                show: false,
                set: None,
                reset: false,
            };
            assert!(!args.show);
        }
    }

    mod cli_additional_tests {
        use super::*;

        #[test]
        fn test_cli_debug() {
            let cli = Cli {
                verbose: 0,
                quiet: false,
                color: ColorArg::Auto,
                command: Commands::Config(ConfigArgs {
                    show: true,
                    set: None,
                    reset: false,
                }),
            };
            let debug = format!("{cli:?}");
            assert!(debug.contains("Cli"));
        }
    }

    mod coverage_tests {
        use super::*;

        #[test]
        fn test_parse_coverage_command() {
            let cli = Cli::parse_from(["probar", "coverage"]);
            assert!(matches!(cli.command, Commands::Coverage(_)));
        }

        #[test]
        fn test_parse_coverage_with_png() {
            let cli = Cli::parse_from(["probar", "coverage", "--png", "output.png"]);
            if let Commands::Coverage(args) = cli.command {
                assert_eq!(args.png, Some(PathBuf::from("output.png")));
            } else {
                panic!("expected Coverage command");
            }
        }

        #[test]
        fn test_parse_coverage_with_palette() {
            let cli = Cli::parse_from(["probar", "coverage", "--palette", "magma"]);
            if let Commands::Coverage(args) = cli.command {
                assert!(matches!(args.palette, PaletteArg::Magma));
            } else {
                panic!("expected Coverage command");
            }
        }

        #[test]
        fn test_parse_coverage_with_legend() {
            let cli = Cli::parse_from(["probar", "coverage", "--legend"]);
            if let Commands::Coverage(args) = cli.command {
                assert!(args.legend);
            } else {
                panic!("expected Coverage command");
            }
        }

        #[test]
        fn test_parse_coverage_with_gaps() {
            let cli = Cli::parse_from(["probar", "coverage", "--gaps"]);
            if let Commands::Coverage(args) = cli.command {
                assert!(args.gaps);
            } else {
                panic!("expected Coverage command");
            }
        }

        #[test]
        fn test_parse_coverage_with_title() {
            let cli = Cli::parse_from(["probar", "coverage", "--title", "My Coverage"]);
            if let Commands::Coverage(args) = cli.command {
                assert_eq!(args.title, Some("My Coverage".to_string()));
            } else {
                panic!("expected Coverage command");
            }
        }

        #[test]
        fn test_parse_coverage_with_dimensions() {
            let cli = Cli::parse_from(["probar", "coverage", "--width", "1024", "--height", "768"]);
            if let Commands::Coverage(args) = cli.command {
                assert_eq!(args.width, 1024);
                assert_eq!(args.height, 768);
            } else {
                panic!("expected Coverage command");
            }
        }

        #[test]
        fn test_parse_coverage_full_options() {
            let cli = Cli::parse_from([
                "probar",
                "coverage",
                "--png",
                "heatmap.png",
                "--palette",
                "heat",
                "--legend",
                "--gaps",
                "--title",
                "Test Coverage",
                "--width",
                "1920",
                "--height",
                "1080",
            ]);
            if let Commands::Coverage(args) = cli.command {
                assert_eq!(args.png, Some(PathBuf::from("heatmap.png")));
                assert!(matches!(args.palette, PaletteArg::Heat));
                assert!(args.legend);
                assert!(args.gaps);
                assert_eq!(args.title, Some("Test Coverage".to_string()));
                assert_eq!(args.width, 1920);
                assert_eq!(args.height, 1080);
            } else {
                panic!("expected Coverage command");
            }
        }

        #[test]
        fn test_palette_default() {
            let palette = PaletteArg::default();
            assert!(matches!(palette, PaletteArg::Viridis));
        }

        #[test]
        fn test_coverage_args_defaults() {
            let args = CoverageArgs {
                png: None,
                json: None,
                palette: PaletteArg::default(),
                legend: false,
                gaps: false,
                title: None,
                width: 800,
                height: 600,
                input: None,
            };
            assert_eq!(args.width, 800);
            assert_eq!(args.height, 600);
            assert!(matches!(args.palette, PaletteArg::Viridis));
        }

        #[test]
        fn test_coverage_args_debug() {
            let args = CoverageArgs {
                png: Some(PathBuf::from("test.png")),
                json: None,
                palette: PaletteArg::Magma,
                legend: true,
                gaps: true,
                title: Some("Test".to_string()),
                width: 640,
                height: 480,
                input: None,
            };
            let debug = format!("{args:?}");
            assert!(debug.contains("CoverageArgs"));
        }
    }

    mod playbook_tests {
        use super::*;

        #[test]
        fn test_parse_playbook_command() {
            let cli = Cli::parse_from(["probar", "playbook", "test.yaml"]);
            assert!(matches!(cli.command, Commands::Playbook(_)));
        }

        #[test]
        fn test_parse_playbook_multiple_files() {
            let cli = Cli::parse_from(["probar", "playbook", "a.yaml", "b.yaml", "c.yaml"]);
            if let Commands::Playbook(args) = cli.command {
                assert_eq!(args.files.len(), 3);
            } else {
                panic!("expected Playbook command");
            }
        }

        #[test]
        fn test_parse_playbook_validate() {
            let cli = Cli::parse_from(["probar", "playbook", "test.yaml", "--validate"]);
            if let Commands::Playbook(args) = cli.command {
                assert!(args.validate);
            } else {
                panic!("expected Playbook command");
            }
        }

        #[test]
        fn test_parse_playbook_export_dot() {
            let cli = Cli::parse_from(["probar", "playbook", "test.yaml", "--export", "dot"]);
            if let Commands::Playbook(args) = cli.command {
                assert!(matches!(args.export, Some(DiagramFormat::Dot)));
            } else {
                panic!("expected Playbook command");
            }
        }

        #[test]
        fn test_parse_playbook_export_svg() {
            let cli = Cli::parse_from([
                "probar",
                "playbook",
                "test.yaml",
                "--export",
                "svg",
                "--export-output",
                "diagram.svg",
            ]);
            if let Commands::Playbook(args) = cli.command {
                assert!(matches!(args.export, Some(DiagramFormat::Svg)));
                assert_eq!(args.export_output, Some(PathBuf::from("diagram.svg")));
            } else {
                panic!("expected Playbook command");
            }
        }

        #[test]
        fn test_parse_playbook_mutate() {
            let cli = Cli::parse_from(["probar", "playbook", "test.yaml", "--mutate"]);
            if let Commands::Playbook(args) = cli.command {
                assert!(args.mutate);
            } else {
                panic!("expected Playbook command");
            }
        }

        #[test]
        fn test_parse_playbook_mutation_classes() {
            let cli = Cli::parse_from([
                "probar",
                "playbook",
                "test.yaml",
                "--mutate",
                "--mutation-classes",
                "M1,M2,M3",
            ]);
            if let Commands::Playbook(args) = cli.command {
                assert!(args.mutate);
                let classes = args.mutation_classes.expect("mutation classes");
                assert_eq!(classes.len(), 3);
                assert!(classes.contains(&"M1".to_string()));
                assert!(classes.contains(&"M2".to_string()));
                assert!(classes.contains(&"M3".to_string()));
            } else {
                panic!("expected Playbook command");
            }
        }

        #[test]
        fn test_parse_playbook_fail_fast() {
            let cli = Cli::parse_from(["probar", "playbook", "test.yaml", "--fail-fast"]);
            if let Commands::Playbook(args) = cli.command {
                assert!(args.fail_fast);
            } else {
                panic!("expected Playbook command");
            }
        }

        #[test]
        fn test_parse_playbook_continue_on_error() {
            let cli = Cli::parse_from(["probar", "playbook", "test.yaml", "--continue-on-error"]);
            if let Commands::Playbook(args) = cli.command {
                assert!(args.continue_on_error);
            } else {
                panic!("expected Playbook command");
            }
        }

        #[test]
        fn test_parse_playbook_format_json() {
            let cli = Cli::parse_from(["probar", "playbook", "test.yaml", "--format", "json"]);
            if let Commands::Playbook(args) = cli.command {
                assert!(matches!(args.format, PlaybookOutputFormat::Json));
            } else {
                panic!("expected Playbook command");
            }
        }

        #[test]
        fn test_parse_playbook_output_dir() {
            let cli =
                Cli::parse_from(["probar", "playbook", "test.yaml", "--output", "results/pb"]);
            if let Commands::Playbook(args) = cli.command {
                assert_eq!(args.output, PathBuf::from("results/pb"));
            } else {
                panic!("expected Playbook command");
            }
        }

        #[test]
        fn test_playbook_args_defaults() {
            let args = PlaybookArgs {
                files: vec![PathBuf::from("test.yaml")],
                validate: false,
                export: None,
                export_output: None,
                mutate: false,
                mutation_classes: None,
                fail_fast: false,
                continue_on_error: false,
                format: PlaybookOutputFormat::default(),
                output: PathBuf::from("target/probar/playbooks"),
            };
            assert!(!args.validate);
            assert!(!args.mutate);
            assert!(matches!(args.format, PlaybookOutputFormat::Text));
        }

        #[test]
        fn test_playbook_args_debug() {
            let args = PlaybookArgs {
                files: vec![PathBuf::from("login.yaml")],
                validate: true,
                export: Some(DiagramFormat::Svg),
                export_output: Some(PathBuf::from("out.svg")),
                mutate: true,
                mutation_classes: Some(vec!["M1".to_string(), "M2".to_string()]),
                fail_fast: true,
                continue_on_error: false,
                format: PlaybookOutputFormat::Json,
                output: PathBuf::from("output"),
            };
            let debug = format!("{args:?}");
            assert!(debug.contains("PlaybookArgs"));
        }

        #[test]
        fn test_diagram_format_debug() {
            let dot_debug = format!("{:?}", DiagramFormat::Dot);
            assert!(dot_debug.contains("Dot"));

            let svg_debug = format!("{:?}", DiagramFormat::Svg);
            assert!(svg_debug.contains("Svg"));
        }

        #[test]
        fn test_playbook_output_format_default() {
            let format = PlaybookOutputFormat::default();
            assert!(matches!(format, PlaybookOutputFormat::Text));
        }

        #[test]
        fn test_playbook_output_format_all_variants() {
            let _ = PlaybookOutputFormat::Text;
            let _ = PlaybookOutputFormat::Json;
            let _ = PlaybookOutputFormat::Junit;
        }
    }

    mod av_sync_tests {
        use super::*;

        #[test]
        fn test_parse_av_sync_check() {
            let cli = Cli::parse_from(["probar", "av-sync", "check", "video.mp4"]);
            assert!(matches!(cli.command, Commands::AvSync(_)));
        }

        #[test]
        fn test_parse_av_sync_check_with_edl() {
            let cli = Cli::parse_from([
                "probar",
                "av-sync",
                "check",
                "video.mp4",
                "--edl",
                "video.edl.json",
            ]);
            if let Commands::AvSync(args) = cli.command {
                if let AvSyncSubcommand::Check(check_args) = args.subcommand {
                    assert_eq!(check_args.edl, Some(PathBuf::from("video.edl.json")));
                } else {
                    panic!("expected Check subcommand");
                }
            } else {
                panic!("expected AvSync command");
            }
        }

        #[test]
        fn test_parse_av_sync_check_with_tolerance() {
            let cli = Cli::parse_from([
                "probar",
                "av-sync",
                "check",
                "video.mp4",
                "--tolerance-ms",
                "30",
            ]);
            if let Commands::AvSync(args) = cli.command {
                if let AvSyncSubcommand::Check(check_args) = args.subcommand {
                    assert!((check_args.tolerance_ms - 30.0).abs() < f64::EPSILON);
                } else {
                    panic!("expected Check subcommand");
                }
            } else {
                panic!("expected AvSync command");
            }
        }

        #[test]
        fn test_parse_av_sync_report() {
            let cli = Cli::parse_from(["probar", "av-sync", "report", "/output/dir"]);
            if let Commands::AvSync(args) = cli.command {
                if let AvSyncSubcommand::Report(report_args) = args.subcommand {
                    assert_eq!(report_args.dir, PathBuf::from("/output/dir"));
                } else {
                    panic!("expected Report subcommand");
                }
            } else {
                panic!("expected AvSync command");
            }
        }

        #[test]
        fn test_parse_av_sync_check_detailed() {
            let cli = Cli::parse_from(["probar", "av-sync", "check", "video.mp4", "--detailed"]);
            if let Commands::AvSync(args) = cli.command {
                if let AvSyncSubcommand::Check(check_args) = args.subcommand {
                    assert!(check_args.detailed);
                } else {
                    panic!("expected Check subcommand");
                }
            } else {
                panic!("expected AvSync command");
            }
        }

        #[test]
        fn test_parse_av_sync_report_with_output() {
            let cli = Cli::parse_from([
                "probar",
                "av-sync",
                "report",
                "/output",
                "-o",
                "report.json",
            ]);
            if let Commands::AvSync(args) = cli.command {
                if let AvSyncSubcommand::Report(report_args) = args.subcommand {
                    assert_eq!(report_args.output, Some(PathBuf::from("report.json")));
                } else {
                    panic!("expected Report subcommand");
                }
            } else {
                panic!("expected AvSync command");
            }
        }

        #[test]
        fn test_av_sync_output_format_default() {
            let format = AvSyncOutputFormat::default();
            assert!(matches!(format, AvSyncOutputFormat::Text));
        }

        #[test]
        fn test_parse_av_sync_check_json_format() {
            let cli = Cli::parse_from([
                "probar",
                "av-sync",
                "check",
                "video.mp4",
                "--format",
                "json",
            ]);
            if let Commands::AvSync(args) = cli.command {
                if let AvSyncSubcommand::Check(check_args) = args.subcommand {
                    assert!(matches!(check_args.format, AvSyncOutputFormat::Json));
                } else {
                    panic!("expected Check subcommand");
                }
            } else {
                panic!("expected AvSync command");
            }
        }
    }

    mod audio_tests {
        use super::*;

        #[test]
        fn test_parse_audio_check() {
            let cli = Cli::parse_from(["probar", "audio", "check", "video.mp4"]);
            assert!(matches!(cli.command, Commands::Audio(_)));
        }

        #[test]
        fn test_parse_audio_check_with_options() {
            let cli = Cli::parse_from([
                "probar",
                "audio",
                "check",
                "video.mp4",
                "--min-rms-dbfs",
                "-30",
                "--sample-rate",
                "44100",
            ]);
            if let Commands::Audio(args) = cli.command {
                if let AudioSubcommand::Check(check_args) = args.subcommand {
                    assert!((check_args.min_rms_dbfs - (-30.0)).abs() < f64::EPSILON);
                    assert_eq!(check_args.sample_rate, 44100);
                } else {
                    panic!("expected Check subcommand");
                }
            } else {
                panic!("expected Audio command");
            }
        }

        #[test]
        fn test_output_format_default() {
            let format = OutputFormat::default();
            assert!(matches!(format, OutputFormat::Text));
        }
    }

    mod video_tests {
        use super::*;

        #[test]
        fn test_parse_video_check() {
            let cli = Cli::parse_from(["probar", "video", "check", "video.mp4"]);
            assert!(matches!(cli.command, Commands::Video(_)));
        }

        #[test]
        fn test_parse_video_check_with_expectations() {
            let cli = Cli::parse_from([
                "probar",
                "video",
                "check",
                "video.mp4",
                "--width",
                "1920",
                "--height",
                "1080",
                "--fps",
                "24",
                "--codec",
                "h264",
                "--require-audio",
            ]);
            if let Commands::Video(args) = cli.command {
                if let VideoSubcommand::Check(check_args) = args.subcommand {
                    assert_eq!(check_args.width, Some(1920));
                    assert_eq!(check_args.height, Some(1080));
                    assert!((check_args.fps.unwrap() - 24.0).abs() < f64::EPSILON);
                    assert_eq!(check_args.codec.as_deref(), Some("h264"));
                    assert!(check_args.require_audio);
                } else {
                    panic!("expected Check subcommand");
                }
            } else {
                panic!("expected Video command");
            }
        }
    }

    mod animation_tests {
        use super::*;

        #[test]
        fn test_parse_animation_check() {
            let cli = Cli::parse_from([
                "probar",
                "animation",
                "check",
                "timeline.json",
                "observed.json",
            ]);
            assert!(matches!(cli.command, Commands::Animation(_)));
        }

        #[test]
        fn test_parse_animation_check_with_tolerance() {
            let cli = Cli::parse_from([
                "probar",
                "animation",
                "check",
                "timeline.json",
                "observed.json",
                "--tolerance-ms",
                "30",
            ]);
            if let Commands::Animation(args) = cli.command {
                if let AnimationSubcommand::Check(check_args) = args.subcommand {
                    assert!((check_args.tolerance_ms - 30.0).abs() < f64::EPSILON);
                } else {
                    panic!("expected Check subcommand");
                }
            } else {
                panic!("expected Animation command");
            }
        }
    }

    mod stress_args_tests {
        use super::*;

        fn make_stress_args(
            atomics: bool,
            worker_msg: bool,
            render: bool,
            trace: bool,
            full: bool,
            mode: &str,
        ) -> StressArgs {
            StressArgs {
                mode: mode.to_string(),
                duration: 30,
                concurrency: 4,
                output: "text".to_string(),
                atomics,
                worker_msg,
                render,
                trace,
                full,
            }
        }

        #[test]
        fn test_get_mode_atomics() {
            let args = make_stress_args(true, false, false, false, false, "default");
            assert_eq!(args.get_mode(), "atomics");
        }

        #[test]
        fn test_get_mode_worker_msg() {
            let args = make_stress_args(false, true, false, false, false, "default");
            assert_eq!(args.get_mode(), "worker-msg");
        }

        #[test]
        fn test_get_mode_render() {
            let args = make_stress_args(false, false, true, false, false, "default");
            assert_eq!(args.get_mode(), "render");
        }

        #[test]
        fn test_get_mode_trace() {
            let args = make_stress_args(false, false, false, true, false, "default");
            assert_eq!(args.get_mode(), "trace");
        }

        #[test]
        fn test_get_mode_full() {
            let args = make_stress_args(false, false, false, false, true, "default");
            assert_eq!(args.get_mode(), "full");
        }

        #[test]
        fn test_get_mode_default() {
            let args = make_stress_args(false, false, false, false, false, "custom-mode");
            assert_eq!(args.get_mode(), "custom-mode");
        }

        #[test]
        fn test_stress_args_debug() {
            let args = make_stress_args(true, false, false, false, false, "atomics");
            let debug = format!("{args:?}");
            assert!(debug.contains("StressArgs"));
        }

        #[test]
        fn test_parse_stress_command() {
            let cli = Cli::parse_from(["probar", "stress"]);
            assert!(matches!(cli.command, Commands::Stress(_)));
        }

        #[test]
        fn test_parse_stress_with_duration() {
            let cli = Cli::parse_from(["probar", "stress", "--duration", "60"]);
            if let Commands::Stress(args) = cli.command {
                assert_eq!(args.duration, 60);
            } else {
                panic!("expected Stress command");
            }
        }

        #[test]
        fn test_parse_stress_with_concurrency() {
            let cli = Cli::parse_from(["probar", "stress", "--concurrency", "8"]);
            if let Commands::Stress(args) = cli.command {
                assert_eq!(args.concurrency, 8);
            } else {
                panic!("expected Stress command");
            }
        }

        #[test]
        fn test_parse_stress_with_atomics_flag() {
            let cli = Cli::parse_from(["probar", "stress", "--atomics"]);
            if let Commands::Stress(args) = cli.command {
                assert!(args.atomics);
                assert_eq!(args.get_mode(), "atomics");
            } else {
                panic!("expected Stress command");
            }
        }
    }
