//! `aprender-rollup shape <fragment> [--roadmap <path>]` — exit 0 ok / 1 refused / 2 not measured.
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (file, roadmap) = match aprender_rollup::shape::parse_args(&args) {
        Ok(parsed) => parsed,
        Err(e) => {
            eprintln!("shape: {e}; usage: aprender-rollup shape <fragment> [--roadmap <path>]");
            return ExitCode::from(2);
        }
    };
    match aprender_rollup::shape::judge(&file, roadmap.as_deref()) {
        Ok(None) => ExitCode::SUCCESS,
        Ok(Some(why)) => {
            println!("{why}");
            ExitCode::from(1)
        }
        Err(e) => {
            eprintln!("shape: {e}; not measured");
            ExitCode::from(2)
        }
    }
}
