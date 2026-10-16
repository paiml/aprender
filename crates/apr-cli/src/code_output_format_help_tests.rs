//! #5211: `apr code --output-format json` help names the envelope that
//! `code_envelope::envelope` actually writes. It used to list only
//! `{type, subtype:"success", result, session_id, duration_ms}`, so a harness
//! reading `--help` could not learn that `backend` (the device that ran) or the
//! error object exist.

/// Every top-level key `envelope()` writes, the `backend` and `error` sub-keys,
/// and the two `subtype` values.
const ENVELOPE_FIELDS: [&str; 20] = [
    "type",
    "subtype",
    "\"success\"",
    "\"error\"",
    "is_error",
    "status",
    "duration_ms",
    "result",
    "session_id",
    "num_turns",
    "tokens_in",
    "tokens_out",
    "total_cost_usd",
    "backend",
    "requested",
    "ran",
    "fell_back",
    "kind",
    "message",
    "exit_code",
];

/// Clap's recursive build of the full `Commands` enum overflows the default
/// test-thread stack in debug builds, as in run_tests_top_k_default_3754.rs.
#[test]
fn code_output_format_help_names_every_envelope_field() {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(code_output_format_help_body)
        .expect("spawn")
        .join()
        .expect("--output-format help row panicked");
}

fn code_output_format_help_body() {
    use clap::CommandFactory;
    let cmd = crate::Cli::command();
    let code = cmd.find_subcommand("code").expect("code subcommand");
    let arg = code
        .get_arguments()
        .find(|a| a.get_id() == "output_format")
        .expect("--output-format");
    let arg_help = arg
        .get_long_help()
        .or_else(|| arg.get_help())
        .expect("--output-format has help")
        .to_string();
    let json_value = arg
        .get_possible_values()
        .into_iter()
        .find(|v| v.get_name() == "json")
        .expect("json value");
    let value_help = json_value
        .get_help()
        .expect("json value has help")
        .to_string();
    for (what, help) in [("--output-format", &arg_help), ("json value", &value_help)] {
        for field in ENVELOPE_FIELDS {
            assert!(
                help.contains(field),
                "{what} help omits envelope field {field}: {help}"
            );
        }
    }
}
