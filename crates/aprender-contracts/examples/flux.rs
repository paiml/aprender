// Example binary: `.unwrap()` is idiomatic in example setup.
#![allow(clippy::disallowed_methods)]

use provable_contracts::flux_gen::generate_flux_annotations;
use provable_contracts::schema::parse_contract;
fn main() {
    let p = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: flux <contract.yaml>");
        std::process::exit(1);
    });
    let c = parse_contract(std::path::Path::new(&p)).unwrap();
    let s = std::path::Path::new(&p)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("x");
    print!("{}", generate_flux_annotations(&c, s));
}
