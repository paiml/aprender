use provable_contracts::coq_gen::generate_coq_spec;
use provable_contracts::schema::parse_contract;
fn main() {
    let p = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: coq <contract.yaml>");
        std::process::exit(1);
    });
    let c = parse_contract(std::path::Path::new(&p)).unwrap();
    let s = std::path::Path::new(&p)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("x");
    print!("{}", generate_coq_spec(&c, s));
}
