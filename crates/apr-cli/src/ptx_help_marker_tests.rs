//! `apr ptx --help` must carry the "unavailable in this build" marker (#4474, #2399 finding 1).
//!
//! Clap derives `long_about` from the doc comment unless it is set explicitly, so
//! overriding only `about` left `--help` (the long form) without the marker.

const MARKER: &str = "unavailable in this build";

/// Clap's recursive build of the full `Commands` enum overflows the default
/// test-thread stack in debug builds, as in code_output_format_help_tests.rs.
#[test]
fn ptx_help_says_when_the_analyzer_is_not_built_in() {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(ptx_help_body)
        .expect("spawn")
        .join()
        .expect("ptx help row panicked");
}

fn ptx_help_body() {
    use clap::CommandFactory;
    let cmd = crate::Cli::command();
    let ptx = cmd.find_subcommand("ptx").expect("ptx subcommand");
    let short = ptx.get_about().map(ToString::to_string).unwrap_or_default();
    let long = ptx
        .get_long_about()
        .or_else(|| ptx.get_about())
        .map(ToString::to_string)
        .unwrap_or_default();
    let built = cfg!(feature = "trueno-explain");
    for (what, text) in [("about", &short), ("long_about", &long)] {
        assert_eq!(
            text.contains(MARKER),
            !built,
            "apr ptx {what} marker presence must be {} when trueno-explain is {}: {text}",
            !built,
            if built { "on" } else { "off" }
        );
    }
}
