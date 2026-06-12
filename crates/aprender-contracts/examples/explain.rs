// Test/example/bench binary: `.unwrap()` is idiomatic here; the lib`s
// cfg(test) allow does not reach this separate crate.
#![allow(clippy::disallowed_methods)]

use provable_contracts::explain::explain_contract;
use provable_contracts::schema::parse_contract;
fn main() {
    let p = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: explain <contract.yaml>");
        std::process::exit(1);
    });
    let c = parse_contract(std::path::Path::new(&p)).unwrap();
    let s = std::path::Path::new(&p)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("x");
    let t = explain_contract(&c, s, None);
    for l in t.lines().take(20) {
        println!("{l}");
    }
}
