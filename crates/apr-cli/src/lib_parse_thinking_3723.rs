    // =========================================================================
    // #3723: `--thinking on|off` on `apr run` and `apr chat` -- ONE declaration (ThinkingArg)
    // =========================================================================

    /// `--thinking on` and `off` parse to Some(true) / Some(false), and ABSENT is None, which
    /// keeps the production rendering. An absent flag must never read as an explicit `off`.
    #[test]
    fn test_run_thinking_flag_values_3723() {
        for (argv, want) in [
            (vec!["apr", "run", "m.gguf", "--thinking", "on"], Some(true)),
            (vec!["apr", "run", "m.gguf", "--thinking", "off"], Some(false)),
            (vec!["apr", "run", "m.gguf"], None),
        ] {
            let cli = parse_cli(argv.clone()).expect("parse");
            match *cli.command {
                Commands::Run { ref thinking, .. } => assert_eq!(thinking.mode(), want, "{argv:?}"),
                _ => panic!("Expected Run command"),
            }
        }
    }

    /// The same flag, with the same values, on `apr chat`.
    #[test]
    fn test_chat_thinking_flag_values_3723() {
        for (argv, want) in [
            (vec!["apr", "chat", "m.gguf", "--thinking", "on"], Some(true)),
            (vec!["apr", "chat", "m.gguf", "--thinking", "off"], Some(false)),
            (vec!["apr", "chat", "m.gguf"], None),
        ] {
            let cli = parse_cli(argv.clone()).expect("parse");
            match *cli.command {
                Commands::Extended(ExtendedCommands::Chat { ref thinking, .. }) => {
                    assert_eq!(thinking.mode(), want, "{argv:?}");
                }
                _ => panic!("Expected Chat command"),
            }
        }
    }

    /// A value outside {on, off} is rejected by the parser, not mapped to a default.
    #[test]
    fn test_thinking_flag_rejects_other_values_3723() {
        for argv in [
            vec!["apr", "run", "m.gguf", "--thinking", "maybe"],
            vec!["apr", "chat", "m.gguf", "--thinking", "true"],
        ] {
            assert!(parse_cli(argv.clone()).is_err(), "{argv:?} must be rejected");
        }
    }
