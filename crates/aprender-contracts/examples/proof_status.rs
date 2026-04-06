use provable_contracts::proof_status::proof_status_report;
use provable_contracts::schema::parse_contract;
fn main() {
    let d = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: proof_status <dir/>");
        std::process::exit(1);
    });
    let mut cs = Vec::new();
    for e in std::fs::read_dir(&d).unwrap().flatten() {
        let p = e.path();
        if p.extension().and_then(|e| e.to_str()) == Some("yaml") {
            let s = p
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("x")
                .to_string();
            if let Ok(c) = parse_contract(&p) {
                cs.push((s, c));
            }
        }
    }
    cs.sort_by(|a, b| a.0.cmp(&b.0));
    let refs: Vec<_> = cs.iter().map(|(s, c)| (s.clone(), c)).collect();
    let r = proof_status_report(&refs, None, true);
    println!(
        "Contracts: {} | Obligations: {} | Kani: {} | Lean: {}",
        r.totals.contracts, r.totals.obligations, r.totals.kani_harnesses, r.totals.lean_proved
    );
}
