//! `pv-roadmap` — Rust port of `scripts/lib/roadmap_fragments.py` (epic #5902, slice S1 of 12, #6353).
//!
//! S1 is the scaffold only: the bin stub with `--help` and `--version`. Every subcommand lands in a later slice
//! (S2 yamlval, S3 merge, S5 fragments, S8/S9 diff, S10 dispatch); until then a subcommand is refused with exit
//! 2 and a message naming the slice plan, never a silent success.

use std::process::ExitCode;

const USAGE: &str = "\
pv-roadmap — roadmap fragment tooling (port of scripts/lib/roadmap_fragments.py, #5902)

usage: pv-roadmap [-h | --help] [-V | --version]

Subcommands (aggregate, changed, adopt, fragment-ids, fragments) are not ported yet: slices S2-S10 of #5902.";

/// What `args` (argv without the program name) asks for, as `(stdout, stderr, exit code)`.
#[must_use]
pub fn run(args: &[String]) -> (String, String, u8) {
    match args {
        [flag] if flag == "-h" || flag == "--help" => (format!("{USAGE}\n"), String::new(), 0),
        [flag] if flag == "-V" || flag == "--version" => (
            format!(
                "pv-roadmap {} ({}) (aprender roadmap fragment tooling)\n",
                env!("CARGO_PKG_VERSION"),
                env!("APR_GIT_SHA")
            ),
            String::new(),
            0,
        ),
        [] => (String::new(), format!("{USAGE}\n"), 2),
        [first, ..] => (
            String::new(),
            format!("pv-roadmap: `{first}` is not ported yet (slices S2-S10 of #5902)\n{USAGE}\n"),
            2,
        ),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (out, err, code) = run(&args);
    print!("{out}");
    eprint!("{err}");
    ExitCode::from(code)
}

#[cfg(test)]
mod tests {
    use super::run;

    fn a(v: &[&str]) -> Vec<String> {
        v.iter().map(ToString::to_string).collect()
    }

    #[test]
    fn help_and_version_exit_zero_on_stdout() {
        for f in ["-h", "--help"] {
            let (o, e, c) = run(&a(&[f]));
            assert!(o.contains("usage: pv-roadmap") && e.is_empty() && c == 0);
        }
        for f in ["-V", "--version"] {
            let (o, e, c) = run(&a(&[f]));
            assert!(o.starts_with("pv-roadmap ") && e.is_empty() && c == 0);
        }
    }

    #[test]
    fn no_args_and_unported_subcommands_are_refused_not_passed() {
        let (o, e, c) = run(&[]);
        assert!(o.is_empty() && e.contains("usage:") && c == 2);
        for sub in ["aggregate", "changed", "adopt", "fragment-ids", "fragments"] {
            let (o, e, c) = run(&a(&[sub, "--check"]));
            assert!(
                o.is_empty() && e.contains("not ported yet") && c == 2,
                "{sub}"
            );
        }
    }
}
