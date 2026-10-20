use std::collections::HashSet;
use std::path::Path;

use provable_contracts::binding::{parse_binding, BindingRegistry, ImplStatus};
use provable_contracts::obligation_matrix::{format_obligation_table, obligation_matrix};
use provable_contracts::proof_status::{format_text, proof_status_report};
use provable_contracts::schema::ContractKind;

use crate::contract_walk::{collect_corpus, require_contracts};

pub fn run(
    path: &Path,
    binding_path: Option<&Path>,
    verify_root: Option<&Path>,
    format: &str,
    table: bool,
    kind_filter: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    // PVL-001 EV-2 (PVL-2): a binding is RESOLVED, always. `--binding` used to count
    // entries without resolving them (a binding naming a function that exists
    // nowhere printed its report and exited 0), and `--verify-bindings` downgraded
    // silently. Now every `implemented` binding is looked up with the
    // `pv verify-bindings` resolver; a ghost is downgraded, listed under
    // `GHOST BINDINGS (n)`, and the command exits 1. `--verify-bindings` is a no-op
    // alias, kept so existing invocations still parse.
    let (binding, ghosts) = match binding_path {
        Some(bp) => {
            let (reg, ghosts) = resolve_bindings(bp, parse_binding(bp)?);
            (Some(reg), ghosts)
        }
        None => (None, Vec::new()),
    };

    let kind = kind_filter.map(parse_kind).transpose()?;

    // Collect contracts (single file or directory tree). PVL-1 (PMAT-1099): an
    // empty corpus is refused (exit 2), and so is one that `--kind` filters to zero.
    let mut contracts = collect_corpus(path)?;

    if let Some(k) = kind {
        contracts.retain(|(_, c)| c.kind() == k);
        require_contracts(path, &contracts, kind_filter)?;
    }

    contracts.sort_by(|a, b| a.0.cmp(&b.0));

    // #4531: with `--verify-bindings <root>`, a `kani_harnesses[].harness` name counts
    // toward L3 only if a `#[kani::proof]` fn of that name exists under the root.
    let phantoms =
        verify_root.map_or_else(Vec::new, |root| drop_phantom_kani(&mut contracts, root));

    let refs: Vec<(String, &provable_contracts::schema::Contract)> =
        contracts.iter().map(|(s, c)| (s.clone(), c)).collect();

    let include_classes = contracts.len() > 1;
    let report = proof_status_report(&refs, binding.as_ref(), include_classes);

    if format == "json" {
        let json = serde_json::to_string_pretty(&report)?;
        println!("{json}");
    } else {
        print!("{}", format_text(&report));
        // Append kind breakdown when showing >1 contract.
        if contracts.len() > 1 {
            print_kind_breakdown(&contracts);
        }
    }

    if table {
        let matrices = obligation_matrix(&refs);
        print!("{}", format_obligation_table(&matrices));
    }

    if !phantoms.is_empty() {
        let block = phantom_block(&phantoms);
        if format == "json" {
            eprint!("{block}");
        } else {
            print!("{block}");
        }
    }
    if ghosts.is_empty() {
        return Ok(());
    }
    // Text mode prints the block on stdout with the report; JSON mode keeps stdout a
    // single JSON document and prints the block on stderr.
    let block = ghost_block(&ghosts);
    if format == "json" {
        eprint!("{block}");
    } else {
        print!("{block}");
    }
    Err(format!(
        "{} ghost binding(s): claimed implemented, not found in source",
        ghosts.len()
    )
    .into())
}

/// A declared Kani harness with no `#[kani::proof]` fn of that name under the scanned root.
struct Phantom {
    contract: String,
    harness: String,
}

/// Names of every `#[kani::proof]` fn under `root` (skipping `target/` and dot dirs).
fn scan_kani_proofs(root: &Path) -> HashSet<String> {
    let mut found = HashSet::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if path.is_dir() {
                if name != "target" && !name.starts_with('.') {
                    stack.push(path);
                }
            } else if path.extension().is_some_and(|e| e == "rs") {
                if let Ok(src) = std::fs::read_to_string(&path) {
                    collect_kani_proofs(&src, &mut found);
                }
            }
        }
    }
    found
}

/// A `#[kani::proof...]` attribute names the next `fn` line.
fn collect_kani_proofs(src: &str, found: &mut HashSet<String>) {
    let mut pending = false;
    for line in src.lines() {
        let t = line.trim();
        if t.starts_with("#[kani::proof") {
            pending = true;
        } else if pending {
            if let Some(rest) = t.split("fn ").nth(1).filter(|_| t.contains("fn ")) {
                let name: String = rest
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                found.insert(name);
                pending = false;
            }
        }
    }
}

/// Remove every declared harness the source does not define, so the L3 credit is honest.
fn drop_phantom_kani(
    contracts: &mut [(String, provable_contracts::schema::Contract)],
    root: &Path,
) -> Vec<Phantom> {
    let found = scan_kani_proofs(root);
    let mut phantoms = Vec::new();
    for (name, contract) in contracts.iter_mut() {
        contract.kani_harnesses.retain(|h| {
            let phantom = h.harness.as_ref().is_some_and(|n| !found.contains(n));
            if phantom {
                phantoms.push(Phantom {
                    contract: name.clone(),
                    harness: h.harness.clone().unwrap_or_default(),
                });
            }
            !phantom
        });
    }
    phantoms
}

/// `PHANTOM KANI HARNESSES (n)`, then one line per harness that cannot run.
fn phantom_block(phantoms: &[Phantom]) -> String {
    let mut out = format!("\nPHANTOM KANI HARNESSES ({})\n", phantoms.len());
    for p in phantoms {
        out.push_str(&format!("  {}: {}\n", p.contract, p.harness));
    }
    out
}

/// One binding that claims `implemented` for a function the resolver cannot find.
struct Ghost {
    contract: String,
    equation: String,
    function: String,
}

/// Resolve every `implemented` binding against source with the `pv verify-bindings`
/// resolver (`scan_all_sources`: the binding's derived source root, its `crates/`,
/// and the local `src/`). A ghost is downgraded to `not_implemented` so the report's
/// levels are honest, and returned so the caller can name it and reject.
fn resolve_bindings(binding_path: &Path, reg: BindingRegistry) -> (BindingRegistry, Vec<Ghost>) {
    use crate::commands::verify_bindings::{scan_all_sources, short_name};
    let found = scan_all_sources(binding_path, &reg.target_crate);
    let mut ghosts = Vec::new();
    let bindings = reg
        .bindings
        .into_iter()
        .map(|mut b| {
            let unresolved = b.status == ImplStatus::Implemented
                && b.function
                    .as_deref()
                    .and_then(short_name)
                    .is_some_and(|s| !found.contains(&s));
            if unresolved {
                ghosts.push(Ghost {
                    contract: b.contract.clone(),
                    equation: b.equation.clone(),
                    function: b.function.clone().unwrap_or_default(),
                });
                b.status = ImplStatus::NotImplemented;
            }
            b
        })
        .collect();
    (BindingRegistry { bindings, ..reg }, ghosts)
}

/// `GHOST BINDINGS (n)`, then one line per ghost: the line PVL-001 EV-2's probe reads.
fn ghost_block(ghosts: &[Ghost]) -> String {
    let mut out = format!("\nGHOST BINDINGS ({})\n", ghosts.len());
    for g in ghosts {
        out.push_str(&format!(
            "  {} {}: {}\n",
            g.contract, g.equation, g.function
        ));
    }
    out
}

fn print_kind_breakdown(contracts: &[(String, provable_contracts::schema::Contract)]) {
    let mut counts = std::collections::BTreeMap::<ContractKind, usize>::new();
    for (_, c) in contracts {
        *counts.entry(c.kind()).or_insert(0) += 1;
    }
    // Only print if there's > 1 kind represented.
    if counts.len() < 2 {
        return;
    }
    println!();
    print!("By kind:");
    for (kind, count) in &counts {
        print!("  {kind}={count}");
    }
    println!();
}

fn parse_kind(s: &str) -> Result<ContractKind, Box<dyn std::error::Error>> {
    match s.to_lowercase().as_str() {
        "kernel" => Ok(ContractKind::Kernel),
        "registry" => Ok(ContractKind::Registry),
        "model-family" | "modelfamily" => Ok(ContractKind::ModelFamily),
        "pattern" => Ok(ContractKind::Pattern),
        "schema" => Ok(ContractKind::Schema),
        other => Err(format!(
            "invalid --kind value '{other}': expected one of \
             kernel, registry, model-family, pattern, schema"
        )
        .into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kani_proofs_are_found_by_attribute_not_by_name() {
        let mut found = HashSet::new();
        collect_kani_proofs(
            "#[kani::proof]\n#[kani::unwind(4)]\nfn verify_a() {}\nfn verify_plain() {}\n#[kani::proof]\npub fn verify_b() {}\n",
            &mut found,
        );
        assert!(found.contains("verify_a") && found.contains("verify_b"));
        assert!(!found.contains("verify_plain"));
    }

    #[test]
    fn declared_harness_without_proof_fn_is_dropped() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("k.rs"), "#[kani::proof]\nfn real() {}\n").expect("write");
        let yaml = "metadata: {version: \"1.0.0\", description: d}\nequations: {}\nkani_harnesses:\n  - {id: K1, obligation: O, harness: real}\n  - {id: K2, obligation: O, harness: ghost}\n";
        let c = provable_contracts::schema::parse_contract_str(yaml).expect("parse");
        let mut contracts = vec![("c.yaml".to_string(), c)];
        let phantoms = drop_phantom_kani(&mut contracts, dir.path());
        assert_eq!(phantoms.len(), 1);
        assert_eq!(phantoms[0].harness, "ghost");
        assert_eq!(contracts[0].1.kani_harnesses.len(), 1);
    }
}
