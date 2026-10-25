//! `pv-roadmap`: roadmap fragments / additive-diff / merge-driver / trim (#5902).
//! Slice 1 (scaffold): `--help` and `--version` only; the subcommands arrive in later slices.

const USAGE: &str =
    "pv-roadmap: roadmap fragments, additive-diff and merge-driver tooling (#5902)\n\
\n\
USAGE:\n    pv-roadmap <fragments|diff|merge|trim> [ARGS]\n    pv-roadmap --help | --version\n";

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    std::process::exit(run(&argv));
}

fn run(argv: &[String]) -> i32 {
    match argv.first().map(String::as_str) {
        Some("--help" | "-h") => {
            print!("{USAGE}");
            0
        }
        Some("--version" | "-V") => {
            println!("pv-roadmap {}", env!("CARGO_PKG_VERSION"));
            0
        }
        _ => {
            eprint!("{USAGE}");
            2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::run;

    fn a(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| (*x).to_string()).collect()
    }

    #[test]
    fn help_and_version_exit_zero() {
        assert_eq!(run(&a(&["--help"])), 0);
        assert_eq!(run(&a(&["--version"])), 0);
    }

    #[test]
    fn anything_else_is_usage_exit_two() {
        assert_eq!(run(&a(&[])), 2);
        assert_eq!(run(&a(&["fragments"])), 2);
    }
}
