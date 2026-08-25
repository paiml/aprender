// FALSIFY-2606 — the MCP `apr.serve` argv must be accepted by the CLI's own parser.
//
// # Why this test is here and not in `aprender-mcp`
//
// The 0.63.0 dogfood sweep measured, against the published crates.io binary:
//
//     $ apr serve "$MODEL" --port 8080
//     rc=2
//     error: unrecognized subcommand '/home/noah/models/…q4_k_m.apr'
//     Usage: apr serve [OPTIONS] <COMMAND>
//
// …while the MCP tool that built that argv returned `{"pid":…,"url":…}` with
// `isError` absent. Two defects; #2388 fixed both, but the guard it left on the
// argv half was
//
//     assert_eq!(serve_argv(m, p), vec!["serve", "run", m, "--port", p]);
//
// which is *tautological with the implementation*. It re-states the argv rather
// than asking anything whether that argv parses. Rename `ServeCommands::Run` to
// `Start` tomorrow and that assertion stays green while `apr.serve` regresses to
// exactly the 0.63.0 behaviour — a child that dies at clap with rc=2.
//
// The only surface that can answer "does the CLI accept this?" is the CLI's own
// clap parser, the one that produced the rc=2 above. `apr-cli` depends on
// `aprender-mcp` (not the reverse), so the check has to live on this side of the
// edge. It costs no subprocess, no model, and no port.
//
// This is a SHAPE guard on argv. The behavioural half of #2606 — that the tool
// never reports a URL for a port it did not watch accept a connection — is
// guarded in `aprender-mcp/src/tools/serve.rs`, and is the half that survives
// any future argv change.

/// Parse a runtime-built argv through the real `Cli` parser.
///
/// Mirrors `parsing.rs::parse_cli` (16 MB stack — clap's parser for 100+
/// subcommands blows the default test-thread stack in debug builds) but takes
/// owned `String`s, because `serve_argv` builds its argv at runtime.
fn parse_cli_owned(args: Vec<String>) -> Result<Cli, clap::error::Error> {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(move || Cli::try_parse_from(args))
        .expect("spawn thread")
        .join()
        .expect("join thread")
}

/// Build the argv `apr.serve` actually spawns, prefixed with argv[0].
fn mcp_serve_argv(model: &str, port: u16) -> Vec<String> {
    let mut argv = vec!["apr".to_string()];
    argv.extend(aprender_mcp::tools::serve::serve_argv(model, port));
    argv
}

/// FALSIFY-2606-A: the argv the MCP `apr.serve` tool spawns must parse.
///
/// RED before #2388's argv fix (`serve <model> --port N` → clap error
/// "unrecognized subcommand"). RED again if anyone renames the subcommand
/// without updating `serve_argv`, which the hard-coded-vector assertion in
/// `aprender-mcp` cannot detect.
#[test]
fn falsify_2606_mcp_serve_argv_is_accepted_by_the_cli_parser() {
    let model = "/models/qwen2.5-coder-1.5b-instruct-q4_k_m.apr";
    let argv = mcp_serve_argv(model, 38621);

    let parsed = parse_cli_owned(argv.clone()).unwrap_or_else(|e| {
        panic!(
            "#2606: the MCP apr.serve tool spawns an argv the CLI REJECTS.\n\
             argv: {argv:?}\n\
             clap said: {e}\n\
             This is the rc=2 the 0.63.0 sweep measured; the tool reported \
             {{pid, url}} anyway."
        )
    });

    // Parsing is necessary but not sufficient: it must reach the server
    // launcher, with the model and port the caller asked for. A form that
    // parsed into `serve plan` would also "parse" and start nothing.
    match *parsed.command {
        Commands::Serve {
            command: ServeCommands::Run { file: Some(ref file), port, .. },
        } => {
            assert_eq!(
                file.to_string_lossy(),
                model,
                "model path must land in the FILE slot, not the subcommand slot"
            );
            assert_eq!(port, 38621, "--port must reach the server launcher");
        }
        ref other => panic!(
            "#2606: argv parsed, but not into `serve run` — it would never start \
             a server. argv: {argv:?}, parsed: {other:?}"
        ),
    }
}

/// FALSIFY-2606-B: the exact argv shape 0.63.0 shipped must still be rejected.
///
/// Pins the negative half. Without it, a future change that made `apr serve
/// <model>` parse (e.g. a default subcommand) would silently drain FALSIFY-2606-A
/// of meaning — the defect would become unobservable rather than fixed. If this
/// ever goes RED it is a real decision to make, not a test to delete: the tool's
/// liveness guard, not clap, would become the only thing standing between a user
/// and a fabricated URL.
#[test]
fn falsify_2606_the_0_63_0_argv_is_still_rejected_by_clap() {
    let model = "/models/qwen2.5-coder-1.5b-instruct-q4_k_m.apr";
    let broken = vec![
        "apr".to_string(),
        "serve".to_string(),
        model.to_string(),
        "--port".to_string(),
        "38621".to_string(),
    ];
    let err = parse_cli_owned(broken.clone())
        .err()
        .unwrap_or_else(|| panic!("0.63.0's argv {broken:?} unexpectedly parses now"));
    let rendered = err.to_string();
    assert!(
        rendered.contains("subcommand") || rendered.contains("Usage"),
        "expected the measured clap rejection, got: {rendered}"
    );
}

/// FALSIFY-2606-C: the port the caller passes must survive into the parse, for
/// the whole u16 range including the edges the sweep did not probe.
#[test]
fn falsify_2606_serve_argv_round_trips_port_across_the_range() {
    for probe in [1_u16, 80, 8080, 38641, 65535] {
        let argv = mcp_serve_argv("/models/m.gguf", probe);
        let parsed = parse_cli_owned(argv.clone())
            .unwrap_or_else(|e| panic!("argv {argv:?} rejected: {e}"));
        match *parsed.command {
            Commands::Serve {
                command: ServeCommands::Run { port, .. },
            } => assert_eq!(port, probe, "port {probe} must round-trip"),
            ref other => panic!("argv {argv:?} parsed as {other:?}, not `serve run`"),
        }
    }
}
