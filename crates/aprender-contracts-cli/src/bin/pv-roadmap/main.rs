//! `pv-roadmap`: roadmap fragments / additive-diff / merge-driver / trim (#5902).

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    std::process::exit(aprender_contracts_cli::roadmap::run::main_with(&argv));
}
