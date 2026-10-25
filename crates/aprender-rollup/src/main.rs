//! `aprender-rollup fold <base.yaml> <entries-dir>` — the aggregate on stdout; exit 0 / 2 not measured.
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (base, entries) = match aprender_rollup::fold::parse_args(&args) {
        Ok(parsed) => parsed,
        Err(e) => {
            eprintln!("fold: {e}");
            return ExitCode::from(2);
        }
    };
    match aprender_rollup::fold::fold(&base, &entries) {
        Ok(text) => {
            print!("{text}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("fold: {e}; not measured");
            ExitCode::from(2)
        }
    }
}
