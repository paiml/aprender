
    /// B1: every `--step` name in `SHOWCASE_STEPS` resolves to its own step.
    #[test]
    fn test_parse_showcase_step_known_names_round_trip() {
        assert!(!SHOWCASE_STEPS.is_empty());
        for (name, expected) in SHOWCASE_STEPS {
            let got = parse_showcase_step(name).expect("known step must parse");
            assert_eq!(got, *expected, "step '{name}' resolved to the wrong variant");
        }
    }

    /// B1: an unknown `--step` is rejected, names the input and lists every valid name.
    #[test]
    fn test_parse_showcase_step_unknown_names_input_and_lists_all() {
        let err = parse_showcase_step("bogus").expect_err("unknown step must fail");
        let CliError::ValidationFailed(msg) = err else {
            panic!("expected ValidationFailed, got {err:?}");
        };
        assert!(msg.contains("unknown step 'bogus'"), "msg was: {msg}");
        for (name, _) in SHOWCASE_STEPS {
            assert!(msg.contains(name), "msg must list '{name}': {msg}");
        }
        // Matching is exact: case and whitespace variants are unknown too.
        assert!(parse_showcase_step("Import").is_err());
        assert!(parse_showcase_step("").is_err());
    }
