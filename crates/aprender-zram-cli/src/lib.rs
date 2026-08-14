//! zram device management (a `zramctl` replacement) with SIMD-accelerated
//! compression benchmarking.
//!
//! # Why this is a library
//!
//! This crate used to be binary-only: `main.rs` declared `mod commands;`,
//! defined the clap tree inline, and dispatched it. That made the command
//! surface unreachable from anywhere except a binary named `trueno-zram` —
//! a pre-consolidation name that collides with nothing in `apr` and tells a
//! user nothing about which project shipped it.
//!
//! The clap tree and its dispatch now live here, so `apr zram` calls
//! [`run`] — the *same* entry point the standalone binary called — rather
//! than a copy-pasted reimplementation that can drift from it.

#![deny(missing_docs)]
#![deny(clippy::panic)]
#![warn(clippy::all, clippy::pedantic)]

pub mod commands;
pub mod output;

use clap::Subcommand;

/// zram device management subcommands.
///
/// Mounted as `apr zram <SUBCOMMAND>`.
#[derive(Subcommand, Debug)]
pub enum ZramCommands {
    /// Create and configure a zram device
    Create(commands::CreateArgs),

    /// Remove a zram device
    Remove(commands::RemoveArgs),

    /// Show zram device status
    Status(commands::StatusArgs),

    /// Run compression benchmarks
    Benchmark(commands::BenchmarkArgs),
}

/// Execute one zram subcommand.
///
/// `format` is the output format selector; it is only consulted by
/// [`ZramCommands::Status`], which is the sole subcommand with tabular output.
/// The other three print progress lines and are format-independent — that was
/// true of the standalone binary too, where `--format` was a global flag
/// threaded only into `status`.
///
/// # Errors
///
/// Propagates whatever the underlying `trueno_zram_core` operation returned:
/// sysfs I/O failures, an unparseable `--size`, an unknown algorithm, or a
/// device that is in use.
pub fn run(
    command: &ZramCommands,
    format: output::OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        ZramCommands::Create(args) => commands::create(args),
        ZramCommands::Remove(args) => commands::remove(args),
        ZramCommands::Status(args) => commands::status(args, format),
        ZramCommands::Benchmark(args) => commands::benchmark(args),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// A harness parser standing in for whatever host mounts [`ZramCommands`]
    /// — the standalone binary yesterday, `apr zram` today.
    #[derive(Parser, Debug)]
    #[command(name = "zram-harness")]
    struct Harness {
        #[command(subcommand)]
        command: ZramCommands,
    }

    fn parse(argv: &[&str]) -> Harness {
        Harness::try_parse_from(argv).expect("harness args should parse")
    }

    #[test]
    fn create_keeps_every_flag_the_standalone_binary_accepted() {
        let h = parse(&[
            "zram-harness",
            "create",
            "--device",
            "3",
            "--size",
            "4G",
            "--algorithm",
            "zstd",
            "--streams",
            "8",
        ]);
        let ZramCommands::Create(args) = h.command else {
            panic!("expected create");
        };
        assert_eq!(
            (
                args.device,
                args.size.as_str(),
                args.algorithm.as_str(),
                args.streams
            ),
            (3, "4G", "zstd", 8)
        );
    }

    #[test]
    fn create_short_flags_match_the_standalone_binary() {
        let h = parse(&[
            "zram-harness",
            "create",
            "-d",
            "1",
            "-s",
            "512M",
            "-a",
            "lz4",
        ]);
        let ZramCommands::Create(args) = h.command else {
            panic!("expected create");
        };
        assert_eq!(
            (
                args.device,
                args.size.as_str(),
                args.algorithm.as_str(),
                args.streams
            ),
            (1, "512M", "lz4", 0)
        );
    }

    /// `--device` was declared `range(0..=16)`. If the range were dropped in
    /// the move, device 17 would be accepted here and then addressed as
    /// `/sys/block/zram17`, which does not exist.
    #[test]
    fn create_refuses_a_device_number_above_the_declared_range() {
        let err = Harness::try_parse_from(["zram-harness", "create", "-d", "17", "-s", "1G"])
            .expect_err("device 17 is outside 0..=16 and must be refused");
        assert_eq!(err.kind(), clap::error::ErrorKind::ValueValidation);
    }

    #[test]
    fn create_requires_size() {
        let err = Harness::try_parse_from(["zram-harness", "create", "-d", "0"])
            .expect_err("--size has no default and must be required");
        assert_eq!(err.kind(), clap::error::ErrorKind::MissingRequiredArgument);
    }

    #[test]
    fn remove_keeps_device_and_force() {
        let h = parse(&["zram-harness", "remove", "--device", "2", "--force"]);
        let ZramCommands::Remove(args) = h.command else {
            panic!("expected remove");
        };
        assert_eq!((args.device, args.force), (2, true));
    }

    #[test]
    fn status_device_is_optional_and_defaults_to_all_devices() {
        let h = parse(&["zram-harness", "status"]);
        let ZramCommands::Status(args) = h.command else {
            panic!("expected status");
        };
        assert_eq!(args.device, None);
    }

    #[test]
    fn status_accepts_a_single_device() {
        let h = parse(&["zram-harness", "status", "-d", "4"]);
        let ZramCommands::Status(args) = h.command else {
            panic!("expected status");
        };
        assert_eq!(args.device, Some(4));
    }

    #[test]
    fn benchmark_keeps_pages_algorithm_and_pattern() {
        let h = parse(&[
            "zram-harness",
            "benchmark",
            "--pages",
            "128",
            "--algorithm",
            "zstd",
            "--pattern",
            "text",
        ]);
        let ZramCommands::Benchmark(args) = h.command else {
            panic!("expected benchmark");
        };
        assert_eq!(
            (args.pages, args.algorithm.as_str(), args.pattern.as_str()),
            (128, "zstd", "text")
        );
    }

    #[test]
    fn benchmark_defaults_match_the_standalone_binary() {
        let h = parse(&["zram-harness", "benchmark"]);
        let ZramCommands::Benchmark(args) = h.command else {
            panic!("expected benchmark");
        };
        assert_eq!(
            (args.pages, args.algorithm.as_str(), args.pattern.as_str()),
            (10000, "all", "mixed")
        );
    }

    /// REGRESSION GUARD. `--pattern` declares `short = 'p'` explicitly while
    /// `--pages` used to derive `-p` from its field name. Two arguments cannot
    /// own one short: clap's `debug_asserts` refused the whole `benchmark`
    /// command outright in debug builds, and in the release build `cargo
    /// install` produces, `-p` silently resolved to `--pages` — so
    /// `benchmark -p text` died in an integer parser and `--pattern`'s
    /// declared short was unreachable.
    ///
    /// Restoring `short` on `pages` turns this RED with
    /// `invalid digit found in string` for `--pages <PAGES>`.
    #[test]
    fn benchmark_short_p_means_pattern_not_pages() {
        let h = parse(&["zram-harness", "benchmark", "-p", "zero"]);
        let ZramCommands::Benchmark(args) = h.command else {
            panic!("expected benchmark");
        };
        assert_eq!(args.pattern.as_str(), "zero");
        // and --pages kept its default, i.e. -p did not land there
        assert_eq!(args.pages, 10000);
    }

    /// `--pages` is still reachable — the collision fix removed only its
    /// short form, not the argument.
    #[test]
    fn benchmark_pages_long_form_still_parses() {
        let h = parse(&["zram-harness", "benchmark", "--pages", "7"]);
        let ZramCommands::Benchmark(args) = h.command else {
            panic!("expected benchmark");
        };
        assert_eq!(args.pages, 7);
    }
}
