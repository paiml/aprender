use std::path::Path;

use provable_contracts::binding::parse_binding;
use provable_contracts::coverage::{CoverageReport, coverage_report, overall_percentage};
use provable_contracts::reverse_coverage::reverse_coverage;
use provable_contracts::schema::{Contract, parse_contract};

pub fn run(
    contract_dir: &Path,
    binding_path: Option<&Path>,
    _show_fuzz: bool,
    reverse_crate: Option<&Path>,
    enforcement_crate: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(crate_dir) = reverse_crate {
        return run_reverse_coverage(crate_dir, binding_path);
    }

    let binding = match binding_path {
        Some(bp) => Some(parse_binding(bp)?),
        None => None,
    };

    let contracts = load_yaml_contracts(contract_dir);
    let refs: Vec<(String, &Contract)> = contracts.iter().map(|(s, c)| (s.clone(), c)).collect();
    let report = coverage_report(&refs, binding.as_ref());
    let pct = overall_percentage(&report);

    print_coverage_report(&report, pct, binding_path.is_some());

    if let Some(crate_dir) = enforcement_crate {
        let bp = binding_path.ok_or("--enforcement requires --binding <path>")?;
        let binding = provable_contracts::binding::parse_binding(bp)?;
        print_enforcement_report(crate_dir, &binding);
    }

    Ok(())
}

/// Run `--reverse` mode: list public functions not covered by `binding.yaml`.
fn run_reverse_coverage(
    crate_dir: &Path,
    binding_path: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let bp = binding_path.ok_or("--reverse requires --binding <path>")?;
    let report = reverse_coverage(crate_dir, bp);
    println!("Reverse Coverage Report");
    println!("=======================");
    println!("  Public functions: {}", report.total_pub_fns);
    println!("  Bound (in binding.yaml): {}", report.bound_fns);
    println!("  Annotated (#[contract]): {}", report.annotated_fns);
    println!("  Auto-exempt (trivial): {}", report.exempt_fns);
    println!("  Unbound: {}", report.unbound.len());
    println!(
        "  Coverage: {:.1}% (bound + exempt / total)",
        report.coverage_pct
    );

    let gated_count = report
        .unbound
        .iter()
        .filter(|f| f.feature_gate.is_some())
        .count();
    if gated_count > 0 {
        println!("  Feature-gated:   {gated_count} (require --features to test)");
    }

    print_unbound_functions(&report.unbound);
    Ok(())
}

fn print_unbound_functions(unbound: &[provable_contracts::reverse_coverage::PubFn]) {
    if unbound.is_empty() {
        return;
    }
    println!("\nUnbound functions:");
    for f in unbound.iter().take(20) {
        let gate_note = match &f.feature_gate {
            Some(feat) => format!(" [requires --features {feat}]"),
            None => String::new(),
        };
        println!("  {} ({}:{}){gate_note}", f.path, f.file, f.line);
    }
    if unbound.len() > 20 {
        println!("  ... and {} more", unbound.len() - 20);
    }
}

/// Load, parse, and sort all contract `.yaml` files under `contract_dir`.
fn load_yaml_contracts(contract_dir: &Path) -> Vec<(String, Contract)> {
    let mut yaml_paths = Vec::new();
    collect_yaml_files(contract_dir, &mut yaml_paths);

    let mut contracts = Vec::new();
    for path in &yaml_paths {
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        match parse_contract(path) {
            Ok(c) => contracts.push((stem, c)),
            Err(e) => eprintln!("warning: skipping {}: {e}", path.display()),
        }
    }
    contracts.sort_by(|a, b| a.0.cmp(&b.0));
    contracts
}

fn print_coverage_report(report: &CoverageReport, pct: f64, has_binding: bool) {
    println!("Obligation Coverage Report");
    println!("==========================");
    println!();

    for cc in &report.contracts {
        println!(
            "  {:<35} eq={} ob={} ft={} kani={} impl={}/{}",
            cc.stem,
            cc.equations,
            cc.obligations,
            cc.falsification_covered,
            cc.kani_covered,
            cc.binding_implemented,
            cc.equations,
        );
    }

    println!();
    println!("Totals:");
    println!("  Contracts:            {}", report.totals.contracts);
    println!("  Equations:            {}", report.totals.equations);
    println!("  Obligations:          {}", report.totals.obligations);
    println!(
        "  Falsification tests:  {}",
        report.totals.falsification_tests
    );
    println!("  Kani harnesses:       {}", report.totals.kani_harnesses);
    if has_binding {
        println!(
            "  Binding implemented:  {}",
            report.totals.binding_implemented
        );
        println!("  Binding partial:      {}", report.totals.binding_partial);
        println!("  Binding missing:      {}", report.totals.binding_missing);
    }
    println!();
    println!("Overall obligation coverage: {pct:.1}%");
}

/// Scan crate source for contract call sites and classify enforcement quality.
fn print_enforcement_report(
    crate_dir: &Path,
    binding: &provable_contracts::binding::BindingRegistry,
) {
    let src_dir = crate_dir.join("src");
    if !src_dir.exists() {
        eprintln!("warning: {}/src not found", crate_dir.display());
        return;
    }

    // Find all contract_pre_* call sites in .rs files
    let mut call_sites: Vec<CallSite> = Vec::new();
    scan_call_sites(&src_dir, &mut call_sites);

    // Read generated_contracts.rs to classify macro quality
    let gen_path = src_dir.join("generated_contracts.rs");
    let gen_content = std::fs::read_to_string(&gen_path).unwrap_or_default();

    // Classify each call site
    for site in &mut call_sites {
        site.level = classify_macro(&site.macro_name, &gen_content);
    }

    let total_bindings = binding.bindings.len();
    let total_sites = call_sites.len();
    let e0_count = call_sites
        .iter()
        .filter(|s| s.level == EnforcementLevel::E0)
        .count();
    let e1_count = call_sites
        .iter()
        .filter(|s| s.level == EnforcementLevel::E1)
        .count();
    let e2_count = call_sites
        .iter()
        .filter(|s| s.level == EnforcementLevel::E2)
        .count();

    #[allow(clippy::cast_precision_loss)]
    let penetration = if total_bindings > 0 {
        total_sites as f64 / total_bindings as f64
    } else {
        0.0
    };

    #[allow(clippy::cast_precision_loss)]
    let quality_score = if total_sites > 0 {
        (e0_count as f64 * 0.1 + e1_count as f64 * 0.5 + e2_count as f64 * 1.0) / total_sites as f64
    } else {
        0.0
    };

    let enforcement_score = penetration * quality_score;

    println!();
    println!("Enforcement Quality Report");
    println!("==========================");
    println!();
    println!("  Bindings declared:    {total_bindings}");
    println!("  Call sites found:     {total_sites}");
    println!(
        "  Penetration:          {:.1}% ({total_sites}/{total_bindings})",
        penetration * 100.0
    );
    println!();
    println!("  E0 (generic !is_empty):  {e0_count}");
    println!("  E1 (domain pre-checks):  {e1_count}");
    println!("  E2 (pre + post checks):  {e2_count}");
    println!("  Quality score:           {quality_score:.2} (E0=0.1, E1=0.5, E2=1.0)");
    println!();
    println!("  Enforcement score:       {enforcement_score:.4} (penetration × quality)");
    println!();

    if !call_sites.is_empty() {
        println!("  Call sites:");
        for site in &call_sites {
            let level_str = match site.level {
                EnforcementLevel::E0 => "E0",
                EnforcementLevel::E1 => "E1",
                EnforcementLevel::E2 => "E2",
            };
            println!(
                "    [{level_str}] {}:{} — {}",
                site.file, site.line, site.macro_name
            );
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EnforcementLevel {
    E0,
    E1,
    E2,
}

struct CallSite {
    file: String,
    line: usize,
    macro_name: String,
    level: EnforcementLevel,
}

/// Recursively scan `.rs` files for `contract_pre_*` and `contract_post_*` invocations.
fn scan_call_sites(dir: &Path, sites: &mut Vec<CallSite>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            scan_call_sites(&path, sites);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs")
            && path.file_name().and_then(|n| n.to_str()) != Some("generated_contracts.rs")
        {
            let Ok(content) = std::fs::read_to_string(&path) else {
                continue;
            };
            for (i, line) in content.lines().enumerate() {
                if let Some(pos) = line.find("contract_pre_") {
                    let rest = &line[pos..];
                    let end = rest.find('!').unwrap_or(rest.len());
                    let macro_name = rest[..end].to_string();
                    sites.push(CallSite {
                        file: path.display().to_string(),
                        line: i + 1,
                        macro_name,
                        level: EnforcementLevel::E0,
                    });
                }
            }
        }
    }
}

/// Classify a macro's enforcement level by inspecting its body in `generated_contracts.rs`.
fn classify_macro(macro_name: &str, gen_content: &str) -> EnforcementLevel {
    // Find the macro definition
    let pattern = format!("macro_rules! {macro_name} {{");
    let Some(start) = gen_content.find(&pattern) else {
        return EnforcementLevel::E0;
    };
    // Extract the macro body (next ~20 lines)
    let body: String = gen_content[start..]
        .lines()
        .take(20)
        .collect::<Vec<_>>()
        .join("\n");

    let has_domain_pre = body.contains("is_finite")
        || body.contains("len() >")
        || body.contains("len() %")
        || body.contains("len() ==")
        || body.contains("is_empty()");

    // Check if there's a corresponding post macro
    let post_name = macro_name.replace("contract_pre_", "contract_post_");
    let has_post = gen_content.contains(&format!("macro_rules! {post_name} {{"));

    if has_domain_pre && has_post {
        EnforcementLevel::E2
    } else if has_domain_pre {
        EnforcementLevel::E1
    } else {
        EnforcementLevel::E0
    }
}

/// Recursively collect `.yaml` contract files, skipping non-contract directories.
fn collect_yaml_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let dirname = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if dirname == "kaizen" || dirname == "legacy" || dirname == "pipelines" {
                continue;
            }
            collect_yaml_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("yaml")
            && path.file_name().and_then(|n| n.to_str()) != Some("binding.yaml")
        {
            out.push(path);
        }
    }
}
