//! `pv-readme` — render README.md's generated parts and SHACL-validate it (#6394, epic #2870 PVL-001).
//!
//! Invoked as `pv-readme` until the fold into `pv readme` is decided (the `pv --help` surface is untouched).
//! All logic lives in `aprender_contracts_cli::readme_cmd`; see there for the design and exit codes.

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // G0.1 (#4476): the same `<semver> (<sha>)` shape as `pv --version`, so the binary snapshot row carries the commit.
    if matches!(args.as_slice(), [flag] if flag == "-V" || flag == "--version") {
        println!(
            "pv-readme {} ({}) (aprender README renderer)",
            env!("CARGO_PKG_VERSION"),
            env!("APR_GIT_SHA")
        );
        return std::process::ExitCode::SUCCESS;
    }
    aprender_contracts_cli::readme_cmd::main_with(&args)
}
