// #4062: `apr ptx-debug` absorbs the `aprender-ptx-debug` binary. These pin
// that apr parses its argv, dispatches it to the same library code, and never
// turns a failing analysis verdict into exit 0.

const PTX_HEADER_ONLY: &str = ".version 8.0\n.target sm_70\n.address_size 64\n";

#[test]
fn apr_ptx_debug_parses_the_standalone_analyze_argv() {
    let cli = parse_cli(vec![
        "apr",
        "ptx-debug",
        "analyze",
        "k.ptx",
        "--min-score",
        "80",
        "--json",
    ])
    .expect("apr ptx-debug analyze must parse");
    match *cli.command {
        Commands::PtxDebug(trueno_ptx_debug::cli::Command::Analyze(a)) => {
            assert_eq!(a.file, "k.ptx");
            assert!((a.min_score - 80.0).abs() < f64::EPSILON);
            assert!(a.json);
        }
        other => panic!("expected ptx-debug analyze, got {other:?}"),
    }
    assert!(parse_cli(vec!["apr", "ptx-debug", "bogus"]).is_err());
}

#[test]
fn apr_ptx_debug_gen_fkr_dispatches_to_the_library() {
    let dir = tempfile::tempdir().expect("tempdir");
    let ptx = dir.path().join("k.ptx");
    let out = dir.path().join("fkr.rs");
    std::fs::write(&ptx, PTX_HEADER_ONLY).expect("write ptx");
    let cli = parse_cli_owned(vec![
        "apr".into(),
        "ptx-debug".into(),
        "gen-fkr".into(),
        ptx.display().to_string(),
        "-o".into(),
        out.display().to_string(),
    ])
    .expect("apr ptx-debug gen-fkr must parse");
    let result = dispatch_sibling_cli_commands(&cli).expect("ptx-debug is a sibling CLI route");
    assert!(result.is_ok(), "gen-fkr failed: {result:?}");
    let written = std::fs::read_to_string(&out).expect("gen-fkr must write -o");
    assert!(!written.trim().is_empty());
}

#[test]
fn apr_ptx_debug_a_failing_verdict_is_never_exit_zero() {
    assert!(ptx_debug_result(Ok(0)).is_ok());
    let mut messages = Vec::new();
    for code in [1, 2, 3, 42] {
        let err = ptx_debug_result(Ok(code)).expect_err("non-zero verdict must fail");
        assert_eq!(err.exit_code_value(), 5, "verdict {code}");
        messages.push(err.to_string());
    }
    messages.dedup();
    assert_eq!(messages.len(), 4, "each verdict names itself: {messages:?}");
    let io = ptx_debug_result(Err("Failed to read x.ptx".into())).expect_err("io error");
    assert!(io.to_string().contains("Failed to read x.ptx"));
}
