// Example binary: `.unwrap()` is idiomatic in example setup.
#![allow(clippy::disallowed_methods)]

use provable_contracts::invariant_gen::generate_invariants;
use provable_contracts::schema::parse_contract;
fn main() {
    let p = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: invariants <contract.yaml>");
        std::process::exit(1);
    });
    let c = parse_contract(std::path::Path::new(&p)).unwrap();
    let o = generate_invariants(&c);
    if o.is_empty() {
        println!("No type_invariants in this contract.");
    } else {
        print!("{o}");
    }
}
