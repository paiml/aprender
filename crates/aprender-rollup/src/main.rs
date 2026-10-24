//! `aprender-rollup shape <fragment> [--roadmap <path>]` — exit 0 ok / 1 refused / 2 not measured.
use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some("shape"), Some(file)) = (args.first().map(String::as_str), args.get(1)) else {
        eprintln!("usage: aprender-rollup shape <fragment> [--roadmap <path>]");
        return ExitCode::from(2);
    };
    let roadmap = match args.get(2).map(String::as_str) {
        Some("--roadmap") => args.get(3).map(PathBuf::from),
        _ => None,
    };
    match aprender_rollup::shape::judge(&PathBuf::from(file), roadmap.as_deref()) {
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
