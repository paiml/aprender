use provable_contracts::graph::dependency_graph;
use provable_contracts::schema::{Contract, parse_contract};
use provable_contracts::tla_gen::generate_tla_module;
fn main() {
    let d = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: tla <dir/>");
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
    let refs: Vec<(String, &Contract)> = cs.iter().map(|(s, c)| (s.clone(), c)).collect();
    let g = dependency_graph(&refs);
    let out = generate_tla_module("Contracts", &refs, &g);
    for (i, l) in out.lines().enumerate() {
        if i >= 30 {
            println!("...");
            break;
        }
        println!("{l}");
    }
}
