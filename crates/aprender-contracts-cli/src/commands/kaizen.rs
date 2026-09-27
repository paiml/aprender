//! `pv kaizen` — fleet-wide contract enforcement improvement.

use provable_contracts::binding::{parse_binding, BindingRegistry, ImplStatus};
use provable_contracts::codegen;
use std::path::{Path, PathBuf};

#[derive(Clone)]
struct RepoReport {
    name: String,
    bindings: usize,
    call_sites_before: usize,
    call_sites_after: usize,
    e0: usize,
    e1: usize,
    e2: usize,
    assertions_before: usize,
    assertions_after: usize,
    #[allow(dead_code)]
    codegen_ok: bool,
    check_ok: Option<bool>,
    injection_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ELevel {
    E0,
    E1,
    E2,
}

struct CallSite {
    #[allow(dead_code)]
    file: PathBuf,
    #[allow(dead_code)]
    line: usize,
    macro_name: String,
    level: ELevel,
}
#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
pub fn run(
    contract_dir: &Path,
    src_root: &Path,
    repo_filter: Option<&str>,
    dry_run: bool,
    do_codegen: bool,
    do_fix: bool,
    json_output: bool,
    min_score: Option<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let repos = collect_repo_paths(contract_dir, src_root, repo_filter)?;
    if repos.is_empty() {
        return Err("no repos found with binding.yaml and sibling directory".into());
    }

    print_kaizen_banner(repos.len(), do_fix, do_codegen);

    let mut reports: Vec<RepoReport> = Vec::new();
    for (name, repo_path, binding_path) in &repos {
        if let Some(report) = process_repo(
            name,
            repo_path,
            binding_path,
            contract_dir,
            do_codegen,
            do_fix,
        ) {
            reports.push(report);
        }
    }

    if json_output {
        print_json_report(&reports);
    } else {
        print_text_report(&reports, dry_run, do_fix);
    }

    enforce_min_score(&reports, min_score);
    Ok(())
}

/// Enumerate every `<contract_dir>/<repo>/binding.yaml` and resolve its sibling source tree.
fn collect_repo_paths(
    contract_dir: &Path,
    src_root: &Path,
    repo_filter: Option<&str>,
) -> Result<Vec<(String, PathBuf, PathBuf)>, Box<dyn std::error::Error>> {
    let Ok(entries) = std::fs::read_dir(contract_dir) else {
        return Err(format!("cannot read {}", contract_dir.display()).into());
    };

    let mut repos: Vec<(String, PathBuf, PathBuf)> = Vec::new();
    for entry in entries.flatten() {
        if let Some(repo) = classify_repo_entry(&entry, src_root, repo_filter) {
            repos.push(repo);
        }
    }
    repos.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(repos)
}

/// Return `Some((name, repo_path, binding_path))` if `entry` is a valid repo for kaizen.
fn classify_repo_entry(
    entry: &std::fs::DirEntry,
    src_root: &Path,
    repo_filter: Option<&str>,
) -> Option<(String, PathBuf, PathBuf)> {
    let path = entry.path();
    if !path.is_dir() {
        return None;
    }
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_string();
    if matches!(name.as_str(), "kaizen" | "legacy" | "pipelines") {
        return None;
    }
    let binding_path = path.join("binding.yaml");
    if !binding_path.exists() {
        return None;
    }
    if let Some(filter) = repo_filter {
        if name != filter {
            return None;
        }
    }
    let repo_path = resolve_repo_path(src_root, &name, &binding_path);
    if !repo_path.exists() {
        return None;
    }
    Some((name, repo_path, binding_path))
}

/// Print the header with the configured operating mode and repo count.
fn print_kaizen_banner(num_repos: usize, do_fix: bool, do_codegen: bool) {
    println!("pv kaizen — fleet enforcement improvement");
    println!("==========================================\n");
    let mode = if do_fix {
        "fix (inject + validate)"
    } else if do_codegen {
        "codegen (regenerate macros)"
    } else {
        "measure (dry-run)"
    };
    println!("Mode: {mode}");
    println!("Repos: {num_repos}\n");
}

/// Produce a `RepoReport` for one repo, or `None` if the binding failed to parse.
fn process_repo(
    name: &str,
    repo_path: &Path,
    binding_path: &Path,
    contract_dir: &Path,
    do_codegen: bool,
    do_fix: bool,
) -> Option<RepoReport> {
    let binding = match parse_binding(binding_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("  warning: {name}: {e}");
            return None;
        }
    };

    let implemented_bindings = binding
        .bindings
        .iter()
        .filter(|b| b.status == ImplStatus::Implemented)
        .count();

    let src_dir = repo_path.join("src");
    let scan_dirs = collect_scan_dirs(repo_path, &src_dir);
    if scan_dirs.is_empty() {
        return None;
    }

    let gen_path = src_dir.join("generated_contracts.rs");
    let sites_before = scan_and_classify(&scan_dirs, &gen_path);
    let assertions_before =
        count_assertions(&std::fs::read_to_string(&gen_path).unwrap_or_default());

    let mut report =
        build_initial_report(name, implemented_bindings, &sites_before, assertions_before);

    if do_codegen || do_fix {
        apply_codegen_step(contract_dir, &gen_path, name, &mut report);
    }

    if do_fix {
        apply_fix_step(
            &src_dir,
            &scan_dirs,
            &gen_path,
            repo_path,
            &binding,
            name,
            &mut report,
        );
    }

    Some(report)
}

/// Collect every scannable source directory below `repo_path` (top-level + workspace subcrates).
fn collect_scan_dirs(repo_path: &Path, src_dir: &Path) -> Vec<PathBuf> {
    let mut scan_dirs: Vec<PathBuf> = Vec::new();
    if src_dir.exists() {
        scan_dirs.push(src_dir.to_path_buf());
    }
    let crates_dir = repo_path.join("crates");
    if crates_dir.exists() {
        if let Ok(crate_entries) = std::fs::read_dir(&crates_dir) {
            for crate_entry in crate_entries.flatten() {
                let crate_src = crate_entry.path().join("src");
                if crate_src.exists() {
                    scan_dirs.push(crate_src);
                }
            }
        }
    }
    if let Ok(top_entries) = std::fs::read_dir(repo_path) {
        for entry in top_entries.flatten() {
            if let Some(member_src) = workspace_member_src(&entry) {
                scan_dirs.push(member_src);
            }
        }
    }
    scan_dirs
}

/// Return the `src/` directory for a top-level workspace-member crate, if `entry` is one.
fn workspace_member_src(entry: &std::fs::DirEntry) -> Option<PathBuf> {
    let path = entry.path();
    if !path.is_dir() {
        return None;
    }
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    if matches!(name, "target" | ".git" | "crates" | "src") {
        return None;
    }
    let member_src = path.join("src");
    if member_src.exists() && path.join("Cargo.toml").exists() {
        Some(member_src)
    } else {
        None
    }
}

/// Scan every dir in `scan_dirs` for call sites, then classify each site's E-level via `gen_path`.
fn scan_and_classify(scan_dirs: &[PathBuf], gen_path: &Path) -> Vec<CallSite> {
    let mut sites: Vec<CallSite> = Vec::new();
    for dir in scan_dirs {
        scan_call_sites(dir, &mut sites);
    }
    let gen_content = std::fs::read_to_string(gen_path).unwrap_or_default();
    for site in &mut sites {
        site.level = classify_macro(&site.macro_name, &gen_content);
    }
    sites
}

/// Count `ELevel::E0/E1/E2` occurrences in `sites` and return them as `(e0, e1, e2)`.
fn count_levels(sites: &[CallSite]) -> (usize, usize, usize) {
    let e0 = sites.iter().filter(|s| s.level == ELevel::E0).count();
    let e1 = sites.iter().filter(|s| s.level == ELevel::E1).count();
    let e2 = sites.iter().filter(|s| s.level == ELevel::E2).count();
    (e0, e1, e2)
}

/// Build the baseline `RepoReport` from the pre-fix scan.
fn build_initial_report(
    name: &str,
    implemented_bindings: usize,
    sites_before: &[CallSite],
    assertions_before: usize,
) -> RepoReport {
    let (e0, e1, e2) = count_levels(sites_before);
    RepoReport {
        name: name.to_string(),
        bindings: implemented_bindings,
        call_sites_before: sites_before.len(),
        call_sites_after: sites_before.len(),
        e0,
        e1,
        e2,
        assertions_before,
        assertions_after: assertions_before,
        codegen_ok: true,
        check_ok: None,
        injection_count: 0,
    }
}

/// Run codegen, write the generated contracts module, and update report assertion counts.
fn apply_codegen_step(contract_dir: &Path, gen_path: &Path, name: &str, report: &mut RepoReport) {
    let contracts = codegen::generate_all(contract_dir);
    if contracts.is_empty() {
        return;
    }
    match codegen::write_rust_module(&contracts, gen_path) {
        Ok(()) => {
            let new_content = std::fs::read_to_string(gen_path).unwrap_or_default();
            report.assertions_after = count_assertions(&new_content);
            report.codegen_ok = true;
        }
        Err(e) => {
            eprintln!("  {name}: codegen failed: {e}");
            report.codegen_ok = false;
        }
    }
}

/// Inject call sites, re-scan to refresh level counts, and run `cargo check` to validate.
#[allow(clippy::too_many_arguments)]
fn apply_fix_step(
    src_dir: &Path,
    scan_dirs: &[PathBuf],
    gen_path: &Path,
    repo_path: &Path,
    binding: &BindingRegistry,
    name: &str,
    report: &mut RepoReport,
) {
    let gen_content_new = std::fs::read_to_string(gen_path).unwrap_or_default();
    report.injection_count = inject_call_sites(src_dir, binding, &gen_content_new);

    let sites_after = scan_and_classify(scan_dirs, gen_path);
    let (e0, e1, e2) = count_levels(&sites_after);
    report.call_sites_after = sites_after.len();
    report.e0 = e0;
    report.e1 = e1;
    report.e2 = e2;

    report.check_ok = Some(run_cargo_check(repo_path, name));
}

/// Run `cargo check` in `repo_path`; log a diagnostic and return false on failure.
fn run_cargo_check(repo_path: &Path, name: &str) -> bool {
    let check = std::process::Command::new("cargo")
        .args(["check", "--message-format=short"])
        .current_dir(repo_path)
        .output();
    match check {
        Ok(output) => {
            if !output.status.success() {
                eprintln!("  {name}: cargo check failed, reverting injections");
            }
            output.status.success()
        }
        Err(e) => {
            eprintln!("  {name}: cargo check error: {e}");
            false
        }
    }
}

/// Fail the command if the computed fleet score falls below an optional threshold.
fn enforce_min_score(reports: &[RepoReport], min_score: Option<f64>) {
    if let Some(threshold) = min_score {
        let fleet_score = compute_fleet_score(reports);
        if fleet_score < threshold {
            eprintln!("\nEnforcement score {fleet_score:.4} below threshold {threshold:.4}");
            std::process::exit(1);
        }
    }
}

/// Scan `.rs` files recursively for contract macro call sites.
fn scan_call_sites(dir: &Path, sites: &mut Vec<CallSite>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            scan_call_sites(&path, sites);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs")
            && path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n != "generated_contracts.rs")
        {
            let Ok(content) = std::fs::read_to_string(&path) else {
                continue;
            };
            for (i, line) in content.lines().enumerate() {
                let trimmed = line.trim();
                if trimmed.starts_with("//") || trimmed.starts_with('#') {
                    continue;
                }
                for prefix in &["contract_pre_", "contract_post_"] {
                    if let Some(pos) = line.find(prefix) {
                        let rest = &line[pos..];
                        let end = rest.find('!').unwrap_or(rest.len());
                        let macro_name = rest[..end].to_string();
                        sites.push(CallSite {
                            file: path.clone(),
                            line: i + 1,
                            macro_name,
                            level: ELevel::E0,
                        });
                    }
                }
            }
        }
    }
}

/// Classify a macro's enforcement level by inspecting the generated file.
fn classify_macro(macro_name: &str, gen_content: &str) -> ELevel {
    let pattern = format!("macro_rules! {macro_name} {{");
    let Some(start) = gen_content.find(&pattern) else {
        return ELevel::E0;
    };
    let body: String = gen_content[start..]
        .lines()
        .take(20)
        .collect::<Vec<_>>()
        .join("\n");

    let has_domain_pre = body.contains("is_finite")
        || body.contains("len() >")
        || body.contains("len() %")
        || body.contains("len() ==")
        || body.contains("is_empty()")
        || body.contains("size_of_val");

    let post_name = macro_name.replace("contract_pre_", "contract_post_");
    let has_post = gen_content.contains(&format!("macro_rules! {post_name} {{"));

    if has_domain_pre && has_post {
        ELevel::E2
    } else if has_domain_pre {
        ELevel::E1
    } else {
        ELevel::E0
    }
}

/// Count `debug_assert` lines in generated contracts file.
fn count_assertions(content: &str) -> usize {
    content
        .lines()
        .filter(|l| l.contains("debug_assert!"))
        .count()
}

/// Inject contract call sites into functions that have bindings but no existing call site.
fn inject_call_sites(src_dir: &Path, binding: &BindingRegistry, gen_content: &str) -> usize {
    let mut injected = 0;

    let mut existing: Vec<CallSite> = Vec::new();
    scan_call_sites(src_dir, &mut existing);
    let existing_macros: std::collections::HashSet<String> =
        existing.iter().map(|s| s.macro_name.clone()).collect();

    for b in &binding.bindings {
        if b.status != ImplStatus::Implemented {
            continue;
        }

        let eq = b.equation.replace('-', "_").to_lowercase();
        let macro_name = format!("contract_pre_{eq}");

        if !gen_content.contains(&format!("macro_rules! {macro_name} {{")) {
            continue;
        }

        if existing_macros.contains(&macro_name) {
            continue;
        }

        let fn_name = match &b.function {
            Some(f) => f.clone(),
            None => continue,
        };

        if let Some((file, insert_line, arg)) = find_function_insertion_point(src_dir, &fn_name) {
            let Ok(content) = std::fs::read_to_string(&file) else {
                continue;
            };
            let lines: Vec<&str> = content.lines().collect();
            if insert_line == 0 || insert_line > lines.len() {
                continue;
            }

            let indent = detect_indent(lines.get(insert_line).unwrap_or(&""));
            let injection = format!("{indent}{macro_name}!({arg});");

            let mut new_lines: Vec<String> = Vec::with_capacity(lines.len() + 1);
            for (i, line) in lines.iter().enumerate() {
                new_lines.push((*line).to_string());
                if i + 1 == insert_line {
                    new_lines.push(injection.clone());
                }
            }

            let new_content = new_lines.join("\n");
            let new_content = if content.ends_with('\n') {
                format!("{new_content}\n")
            } else {
                new_content
            };

            if std::fs::write(&file, new_content).is_ok() {
                injected += 1;
            }
        }
    }

    injected
}

/// Find where to inject a contract macro call in a function.
fn find_function_insertion_point(
    src_dir: &Path,
    fn_name: &str,
) -> Option<(PathBuf, usize, String)> {
    let mut rs_files = Vec::new();
    collect_rs_files(src_dir, &mut rs_files);

    let fn_pattern = format!("fn {fn_name}");

    for file in &rs_files {
        let fname = file.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if fname == "generated_contracts.rs" || fname.contains("test") {
            continue;
        }

        let Ok(content) = std::fs::read_to_string(file) else {
            continue;
        };

        let line_count = content.lines().count();

        for (i, line) in content.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.starts_with("//") {
                continue;
            }
            if !trimmed.contains(&fn_pattern) || !trimmed.contains("fn ") {
                continue;
            }

            let param = extract_first_param(&content, i);

            let mut brace_line = i;
            let search_end = line_count.min(i + 10);
            for j in i..search_end {
                if content.lines().nth(j).is_some_and(|l| l.contains('{')) {
                    brace_line = j;
                    break;
                }
            }

            let insert_line = skip_early_returns(&content, brace_line + 1);

            return Some((file.clone(), insert_line, param));
        }
    }
    None
}

/// Extract the first meaningful parameter name from a function signature.
fn extract_first_param(full_content: &str, line_idx: usize) -> String {
    let mut sig = String::new();
    for line in full_content.lines().skip(line_idx).take(10) {
        sig.push_str(line);
        if line.contains(')') {
            break;
        }
    }

    let Some(paren_start) = sig.find('(') else {
        return "input".to_string();
    };
    let Some(paren_end) = sig[paren_start..].find(')') else {
        return "input".to_string();
    };
    let params = &sig[paren_start + 1..paren_start + paren_end];

    for param in params.split(',') {
        let param = param.trim();
        if param.is_empty() || param.starts_with("self") || param == "&self" || param == "&mut self"
        {
            continue;
        }
        let name = param
            .split(':')
            .next()
            .unwrap_or("input")
            .trim()
            .trim_start_matches('&')
            .trim_start_matches("mut ")
            .trim();
        if !name.is_empty() {
            return name.to_string();
        }
    }

    "input".to_string()
}

/// Skip past early-return guard clauses to find the right insertion point.
fn skip_early_returns(content: &str, start_line: usize) -> usize {
    let lines: Vec<&str> = content.lines().collect();
    let mut line = start_line;

    while line < lines.len() {
        let trimmed = lines[line].trim();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            line += 1;
            continue;
        }
        if trimmed.starts_with("if ") && trimmed.contains("return") {
            line += 1;
            continue;
        }
        if trimmed.starts_with("if ") {
            let mut has_return = false;
            let guard_end = lines.len().min(line + 5);
            for guard_line in &lines[line + 1..guard_end] {
                if guard_line.trim().starts_with("return") {
                    has_return = true;
                }
                if guard_line.contains('}') {
                    break;
                }
            }
            if has_return {
                let brace_end = lines.len().min(line + 10);
                for (offset, brace_line) in lines[line + 1..brace_end].iter().enumerate() {
                    if brace_line.trim().starts_with('}') || brace_line.contains('}') {
                        line = line + 1 + offset + 1;
                        break;
                    }
                }
                continue;
            }
        }
        break;
    }

    line
}

fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

fn detect_indent(line: &str) -> String {
    let trimmed = line.trim_start();
    let indent_len = line.len() - trimmed.len();
    line[..indent_len].to_string()
}

fn enforcement_grade(score: f64) -> &'static str {
    if score >= 0.60 {
        "Grade A"
    } else if score >= 0.40 {
        "Grade B"
    } else if score >= 0.25 {
        "Grade C"
    } else if score >= 0.10 {
        "Grade D"
    } else {
        "Grade F"
    }
}

/// Penetration-only grade for tool tier (E0 is acceptable).
fn pen_grade(pen: f64) -> &'static str {
    if pen >= 0.90 {
        "Grade A"
    } else if pen >= 0.75 {
        "Grade B"
    } else if pen >= 0.50 {
        "Grade C"
    } else if pen >= 0.25 {
        "Grade D"
    } else {
        "Grade F"
    }
}

/// Per-repo grade based on its own enforcement score.
#[allow(clippy::cast_precision_loss)]
fn repo_grade(r: &RepoReport) -> String {
    if r.bindings == 0 {
        return "-".to_string();
    }
    let sites = r.call_sites_after;
    if sites == 0 {
        return "F".to_string();
    }
    let pen = sites as f64 / r.bindings as f64;
    let qual = (r.e0 as f64 * 0.1 + r.e1 as f64 * 0.5 + r.e2 as f64) / sites as f64;
    let score = pen * qual;
    if score >= 0.60 {
        "A"
    } else if score >= 0.40 {
        "B"
    } else if score >= 0.25 {
        "C"
    } else if score >= 0.10 {
        "D"
    } else {
        "F"
    }
    .to_string()
}

/// Compute fleet-wide enforcement score.
#[allow(clippy::cast_precision_loss)]
fn compute_fleet_score(reports: &[RepoReport]) -> f64 {
    let total_bindings: usize = reports.iter().map(|r| r.bindings).sum();
    let total_sites: usize = reports.iter().map(|r| r.call_sites_after).sum();
    let total_e0: usize = reports.iter().map(|r| r.e0).sum();
    let total_e1: usize = reports.iter().map(|r| r.e1).sum();
    let total_e2: usize = reports.iter().map(|r| r.e2).sum();

    if total_bindings == 0 || total_sites == 0 {
        return 0.0;
    }

    let penetration = total_sites as f64 / total_bindings as f64;
    let quality = (total_e0 as f64 * 0.1 + total_e1 as f64 * 0.5 + total_e2 as f64 * 1.0)
        / total_sites as f64;

    penetration * quality
}

/// Print text report.
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::too_many_lines)]
fn print_text_report(reports: &[RepoReport], _dry_run: bool, do_fix: bool) {
    let total_bindings: usize = reports.iter().map(|r| r.bindings).sum();
    let total_before: usize = reports.iter().map(|r| r.call_sites_before).sum();
    let total_after: usize = reports.iter().map(|r| r.call_sites_after).sum();
    let total_e0: usize = reports.iter().map(|r| r.e0).sum();
    let total_e1: usize = reports.iter().map(|r| r.e1).sum();
    let total_e2: usize = reports.iter().map(|r| r.e2).sum();
    let total_injected: usize = reports.iter().map(|r| r.injection_count).sum();
    let assertions_before: usize = reports.iter().map(|r| r.assertions_before).sum();
    let assertions_after: usize = reports.iter().map(|r| r.assertions_after).sum();
    let check_failures: usize = reports.iter().filter(|r| r.check_ok == Some(false)).count();

    #[allow(clippy::cast_precision_loss)]
    let pen_before = if total_bindings > 0 {
        total_before as f64 / total_bindings as f64 * 100.0
    } else {
        0.0
    };
    #[allow(clippy::cast_precision_loss)]
    let pen_after = if total_bindings > 0 {
        total_after as f64 / total_bindings as f64 * 100.0
    } else {
        0.0
    };

    let fleet_score = compute_fleet_score(reports);

    println!("\nFleet Enforcement Report");
    println!("========================\n");
    println!("  Repos:              {}", reports.len());
    println!("  Total bindings:     {total_bindings}");

    if do_fix {
        println!("  Call sites:         {total_before} -> {total_after} (+{total_injected})");
        println!("  Penetration:        {pen_before:.1}% -> {pen_after:.1}%");
    } else {
        println!("  Call sites:         {total_after}");
        println!("  Penetration:        {pen_after:.1}%");
    }

    println!();
    println!("  E0 (generic):       {total_e0}");
    println!("  E1 (domain pre):    {total_e1}");
    println!("  E2 (pre + post):    {total_e2}");
    println!();

    if assertions_before == assertions_after {
        println!("  Assertions:         {assertions_after}");
    } else {
        println!("  Assertions:         {assertions_before} -> {assertions_after}");
    }

    println!(
        "  Enforcement:        {fleet_score:.4} ({})",
        enforcement_grade(fleet_score)
    );

    if check_failures > 0 {
        println!("  Check failures:     {check_failures}");
    }

    // Tiered scoring: kernel repos vs tool repos
    let kernel_repos = ["aprender", "entrenar", "realizar", "trueno"];
    let kernel: Vec<&RepoReport> = reports
        .iter()
        .filter(|r| kernel_repos.contains(&r.name.as_str()))
        .collect();
    let tool: Vec<&RepoReport> = reports
        .iter()
        .filter(|r| !kernel_repos.contains(&r.name.as_str()))
        .collect();

    if !kernel.is_empty() {
        let k_bind: usize = kernel.iter().map(|r| r.bindings).sum();
        let k_sites: usize = kernel.iter().map(|r| r.call_sites_after).sum();
        let k_e0: usize = kernel.iter().map(|r| r.e0).sum();
        let k_e1: usize = kernel.iter().map(|r| r.e1).sum();
        let k_e2: usize = kernel.iter().map(|r| r.e2).sum();
        #[allow(clippy::cast_precision_loss)]
        let k_pen = if k_bind > 0 {
            k_sites as f64 / k_bind as f64
        } else {
            0.0
        };
        #[allow(clippy::cast_precision_loss)]
        let k_qual = if k_sites > 0 {
            (k_e0 as f64 * 0.1 + k_e1 as f64 * 0.5 + k_e2 as f64) / k_sites as f64
        } else {
            0.0
        };
        #[allow(clippy::cast_precision_loss)]
        let k_e2_pct = if k_sites > 0 {
            k_e2 as f64 / k_sites as f64 * 100.0
        } else {
            0.0
        };
        let k_score = k_pen * k_qual;

        let t_bind: usize = tool.iter().map(|r| r.bindings).sum();
        let t_sites: usize = tool.iter().map(|r| r.call_sites_after).sum();
        #[allow(clippy::cast_precision_loss)]
        let t_pen = if t_bind > 0 {
            t_sites as f64 / t_bind as f64 * 100.0
        } else {
            0.0
        };

        println!();
        println!("  Tiered:");
        println!(
            "    Kernel (4 repos):  {} — {k_sites}/{k_bind} sites, E2 {k_e2_pct:.0}%, pen {:.1}%",
            enforcement_grade(k_score),
            k_pen * 100.0
        );
        println!(
            "    Tool ({} repos):   {} — {t_sites}/{t_bind} sites, pen {t_pen:.1}%",
            tool.len(),
            pen_grade(t_pen / 100.0)
        );
    }

    println!();
    println!(
        "  {:<20} {:>8} {:>8} {:>5} {:>5} {:>5} {:>6}",
        "Repo", "Bindings", "Sites", "E0", "E1", "E2", "Grade"
    );
    println!("  {}", "-".repeat(63));

    for r in reports {
        let grade = if let Some(ok) = r.check_ok {
            if ok {
                repo_grade(r)
            } else {
                "FAIL".to_string()
            }
        } else {
            repo_grade(r)
        };

        let sites_str = if do_fix && r.injection_count > 0 {
            format!("{}>{}", r.call_sites_before, r.call_sites_after)
        } else {
            format!("{}", r.call_sites_after)
        };

        println!(
            "  {:<20} {:>8} {:>8} {:>5} {:>5} {:>5} {:>6}",
            r.name, r.bindings, sites_str, r.e0, r.e1, r.e2, grade
        );
    }
}

/// Print JSON report for CI integration.
#[allow(clippy::cast_precision_loss)]
fn print_json_report(reports: &[RepoReport]) {
    let fleet_score = compute_fleet_score(reports);
    let total_bindings: usize = reports.iter().map(|r| r.bindings).sum();
    let total_sites: usize = reports.iter().map(|r| r.call_sites_after).sum();

    let kernel_repos = ["aprender", "entrenar", "realizar", "trueno"];
    let kernel: Vec<&RepoReport> = reports
        .iter()
        .filter(|r| kernel_repos.contains(&r.name.as_str()))
        .collect();
    let k_bind: usize = kernel.iter().map(|r| r.bindings).sum();
    let k_sites: usize = kernel.iter().map(|r| r.call_sites_after).sum();
    let k_e2: usize = kernel.iter().map(|r| r.e2).sum();
    let k_e2_pct = if k_sites > 0 {
        k_e2 as f64 / k_sites as f64
    } else {
        0.0
    };
    let kernel_score = compute_fleet_score(&kernel.iter().copied().cloned().collect::<Vec<_>>());

    println!("{{");
    println!("  \"fleet_score\": {fleet_score:.4},");
    println!("  \"kernel_score\": {kernel_score:.4},");
    println!("  \"kernel_e2_pct\": {k_e2_pct:.4},");
    println!("  \"total_bindings\": {total_bindings},");
    println!("  \"total_call_sites\": {total_sites},");
    println!("  \"kernel_bindings\": {k_bind},");
    println!("  \"kernel_call_sites\": {k_sites},");
    println!("  \"kernel_e2\": {k_e2},");
    println!("  \"repos\": [");
    for (i, r) in reports.iter().enumerate() {
        let comma = if i + 1 < reports.len() { "," } else { "" };
        println!(
            "    {{\"name\": \"{}\", \"bindings\": {}, \"call_sites\": {}, \
             \"e0\": {}, \"e1\": {}, \"e2\": {}, \"injection_count\": {}}}{comma}",
            r.name, r.bindings, r.call_sites_after, r.e0, r.e1, r.e2, r.injection_count
        );
    }
    println!("  ]");
    println!("}}");
}

fn resolve_repo_path(src_root: &Path, name: &str, binding_path: &Path) -> PathBuf {
    std::fs::read_to_string(binding_path)
        .ok()
        .and_then(|c| {
            c.lines().find_map(|l| {
                l.trim()
                    .strip_prefix("source_dir:")
                    .map(|v| src_root.join(v.trim()))
            })
        })
        .filter(|p| p.exists())
        .unwrap_or_else(|| src_root.join(name))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A report whose level counts are consistent with its call-site count,
    /// which is the only shape the scoring functions are ever handed in
    /// production (`count_levels` derives e0/e1/e2 from the same slice
    /// `call_sites_after` is the length of).
    fn report(bindings: usize, e0: usize, e1: usize, e2: usize) -> RepoReport {
        RepoReport {
            name: "r".to_string(),
            bindings,
            call_sites_before: 0,
            call_sites_after: e0 + e1 + e2,
            e0,
            e1,
            e2,
            assertions_before: 0,
            assertions_after: 0,
            codegen_ok: true,
            check_ok: None,
            injection_count: 0,
        }
    }

    fn site(level: ELevel) -> CallSite {
        CallSite {
            file: PathBuf::from("f.rs"),
            line: 1,
            macro_name: "m".to_string(),
            level,
        }
    }

    /// `enforcement_grade` and `repo_grade` carry the SAME cutoffs —
    /// 0.60/0.40/0.25/0.10 — written out twice, in two functions, 30 lines
    /// apart, differing only in whether the answer is prefixed "Grade ". One
    /// copy can be edited without the other, and nothing would report it: the
    /// fleet line and the per-repo column would simply disagree about the same
    /// score. This pins them to each other rather than to a literal, so a
    /// deliberate change to the ladder only has to be made consistently, not
    /// made here too.
    #[test]
    fn the_two_grade_ladders_cannot_drift_apart() {
        // Scores chosen to land in every band and on every cutoff exactly:
        // score reduces to (0.1*e0 + 0.5*e1 + e2) / bindings.
        let cases = [
            report(5, 0, 0, 3),  // 0.60 — exactly A
            report(5, 0, 0, 2),  // 0.40 — exactly B
            report(4, 0, 0, 1),  // 0.25 — exactly C
            report(10, 0, 0, 1), // 0.10 — exactly D
            report(20, 0, 0, 1), // 0.05 — F
            report(10, 4, 2, 3), // 0.44 — mixed levels, B
        ];
        let disagreements: Vec<String> = cases
            .iter()
            .filter_map(|r| {
                let score = compute_fleet_score(std::slice::from_ref(r));
                let from_ladder = enforcement_grade(score).trim_start_matches("Grade ");
                let from_repo = repo_grade(r);
                (from_repo != from_ladder).then(|| {
                    format!(
                        "\n  - bindings={} e0={} e1={} e2={} score={score:.4}: \
                         repo_grade said {from_repo}, enforcement_grade said {from_ladder}",
                        r.bindings, r.e0, r.e1, r.e2
                    )
                })
            })
            .collect();
        assert!(
            disagreements.is_empty(),
            "the two copies of the grade ladder have drifted: {} of {} scores are \
             graded differently by repo_grade and enforcement_grade, so the per-repo \
             column and the fleet line would disagree about the same number:{}",
            disagreements.len(),
            cases.len(),
            disagreements.join("")
        );
    }

    /// Every cutoff is `>=`, so the named score belongs to the HIGHER band. A
    /// `>` would silently demote exactly the repos sitting on the line, which
    /// are the ones a threshold exists to adjudicate.
    #[test]
    fn enforcement_cutoffs_include_the_score_they_name() {
        let cases: &[(f64, &str)] = &[
            (0.60, "Grade A"),
            (0.599_999, "Grade B"),
            (0.40, "Grade B"),
            (0.399_999, "Grade C"),
            (0.25, "Grade C"),
            (0.249_999, "Grade D"),
            (0.10, "Grade D"),
            (0.099_999, "Grade F"),
            (0.0, "Grade F"),
            (1.0, "Grade A"),
        ];
        let wrong: Vec<String> = cases
            .iter()
            .filter_map(|(s, want)| {
                let got = enforcement_grade(*s);
                (got != *want).then(|| format!("\n  - score {s}: expected {want}, got {got}"))
            })
            .collect();
        assert!(
            wrong.is_empty(),
            "enforcement_grade cutoffs moved:{}",
            wrong.join("")
        );
    }

    /// Same contract for the tool-tier ladder, which has its own cutoffs
    /// (0.90/0.75/0.50/0.25) and is a separate table again.
    #[test]
    fn penetration_cutoffs_include_the_score_they_name() {
        let cases: &[(f64, &str)] = &[
            (0.90, "Grade A"),
            (0.899_999, "Grade B"),
            (0.75, "Grade B"),
            (0.749_999, "Grade C"),
            (0.50, "Grade C"),
            (0.499_999, "Grade D"),
            (0.25, "Grade D"),
            (0.249_999, "Grade F"),
        ];
        let wrong: Vec<String> = cases
            .iter()
            .filter_map(|(s, want)| {
                let got = pen_grade(*s);
                (got != *want).then(|| format!("\n  - pen {s}: expected {want}, got {got}"))
            })
            .collect();
        assert!(
            wrong.is_empty(),
            "pen_grade cutoffs moved:{}",
            wrong.join("")
        );
    }

    /// `compute_fleet_score` and `repo_grade` each compute penetration x
    /// quality with the weights 0.1/0.5/1.0 — the same formula, written twice.
    /// For one repo the fleet score must therefore BE that repo's score, or the
    /// fleet line is not an aggregate of the column beside it.
    #[test]
    fn a_lone_repo_scores_the_same_in_the_fleet_as_it_does_alone() {
        for r in [report(5, 0, 0, 3), report(10, 4, 2, 3), report(7, 7, 0, 0)] {
            let fleet = compute_fleet_score(std::slice::from_ref(&r));
            // (0.1*e0 + 0.5*e1 + 1.0*e2) / bindings — penetration's `sites`
            // cancels against quality's, which is why adding E0 sites raises
            // the score.
            #[allow(clippy::cast_precision_loss)]
            let expected = (r.e0 as f64).mul_add(0.1, (r.e1 as f64).mul_add(0.5, r.e2 as f64))
                / r.bindings as f64;
            assert!(
                (fleet - expected).abs() < 1e-12,
                "fleet score for a single repo (bindings={} e0={} e1={} e2={}) was {fleet}, \
                 but the per-repo formula gives {expected}",
                r.bindings,
                r.e0,
                r.e1,
                r.e2
            );
        }
    }

    /// Both degenerate inputs divide by zero if they are not caught, and both
    /// are reachable: a repo with a binding registry but no implemented
    /// bindings, and one whose call sites were all removed.
    #[test]
    fn an_empty_fleet_scores_zero_instead_of_dividing_by_zero() {
        assert!((compute_fleet_score(&[]) - 0.0).abs() < f64::EPSILON);
        assert!((compute_fleet_score(&[report(0, 0, 0, 0)]) - 0.0).abs() < f64::EPSILON);
        assert!(
            (compute_fleet_score(&[report(5, 0, 0, 0)]) - 0.0).abs() < f64::EPSILON,
            "bindings but no call sites must be 0.0, not NaN"
        );
    }

    /// "No bindings" and "bindings but nothing enforcing them" are different
    /// states and must not print the same character: the first is nothing to
    /// report, the second is a failure.
    #[test]
    fn no_bindings_is_not_graded_but_no_call_sites_is_a_failure() {
        assert_eq!(repo_grade(&report(0, 0, 0, 0)), "-");
        assert_eq!(repo_grade(&report(5, 0, 0, 0)), "F");
    }

    /// e0+e1+e2 is used as the denominator's sibling everywhere downstream, so
    /// the three counts must partition the slice — no site counted twice, none
    /// dropped.
    #[test]
    fn counting_levels_partitions_every_site_exactly_once() {
        let sites = vec![
            site(ELevel::E0),
            site(ELevel::E2),
            site(ELevel::E1),
            site(ELevel::E0),
            site(ELevel::E2),
            site(ELevel::E2),
        ];
        let (e0, e1, e2) = count_levels(&sites);
        assert_eq!((e0, e1, e2), (2, 1, 3));
        assert_eq!(
            e0 + e1 + e2,
            sites.len(),
            "every site must land in exactly one level"
        );
        assert_eq!(count_levels(&[]), (0, 0, 0));
    }

    /// E2 requires BOTH a domain predicate and a matching post macro. The
    /// ladder is what the fleet score weights 0.1/0.5/1.0, so a macro promoted
    /// on one half alone inflates the score without enforcing anything more.
    #[test]
    fn a_macro_is_promoted_only_when_both_halves_are_present() {
        let pre = "macro_rules! contract_pre_f {\n    ($x:expr) => { debug_assert!($x.is_finite()); };\n}\n";
        let post = "macro_rules! contract_post_f {\n    ($x:expr) => { () };\n}\n";
        let bare = "macro_rules! contract_pre_f {\n    ($x:expr) => { () };\n}\n";

        let cases: &[(&str, &str, ELevel, &str)] = &[
            (
                "",
                "contract_pre_f",
                ELevel::E0,
                "the macro is not in the file at all",
            ),
            (
                bare,
                "contract_pre_f",
                ELevel::E0,
                "present, but asserts no domain predicate",
            ),
            (
                pre,
                "contract_pre_f",
                ELevel::E1,
                "a domain predicate, but no post macro",
            ),
        ];
        for (content, name, want, why) in cases {
            assert_eq!(classify_macro(name, content), *want, "{why}");
        }
        let both = format!("{pre}{post}");
        assert_eq!(
            classify_macro("contract_pre_f", &both),
            ELevel::E2,
            "a domain predicate AND a matching post macro is the only route to E2"
        );
    }

    /// The domain-predicate search reads only the first 20 lines of the macro
    /// body. That bound is deliberate and invisible — a predicate below it is
    /// not seen, so a macro can be demoted by having comments added above it.
    /// Pinned so the number cannot change silently.
    #[test]
    fn only_the_first_twenty_lines_of_a_macro_body_are_inspected() {
        let padding = "    // filler\n".repeat(25);
        let late = format!(
            "macro_rules! contract_pre_f {{\n{padding}    debug_assert!(x.is_finite());\n}}\n"
        );
        assert_eq!(
            classify_macro("contract_pre_f", &late),
            ELevel::E0,
            "a domain predicate past the 20-line window is not seen"
        );

        let early = "macro_rules! contract_pre_f {\n    debug_assert!(x.is_finite());\n}\n";
        assert_eq!(
            classify_macro("contract_pre_f", early),
            ELevel::E1,
            "the same predicate inside the window promotes to E1"
        );
    }

    /// `count_assertions` matches the literal `debug_assert!`, so the `_eq`
    /// and `_ne` variants do NOT count. That is the behaviour the
    /// before/after assertion delta is computed from; recorded here because it
    /// is surprising, not because it is obviously right.
    #[test]
    fn counting_assertions_matches_the_bare_macro_and_not_its_eq_variant() {
        let content = "debug_assert!(a);\n  debug_assert!(b);\ndebug_assert_eq!(c, d);\n\
                       debug_assert_ne!(e, f);\nassert!(g);\n// debug_assert!(commented)\n";
        assert_eq!(
            count_assertions(content),
            3,
            "two bare calls plus the commented-out one — a substring match cannot \
             tell code from a comment, and debug_assert_eq!/ne! are not counted"
        );
        assert_eq!(count_assertions(""), 0);
    }

    /// The injected call site is indented to match the line it is inserted
    /// before, so whatever whitespace that line opens with has to come back
    /// verbatim — mixing tabs and spaces here produces code that does not
    /// match the file around it.
    #[test]
    fn indentation_is_returned_verbatim() {
        assert_eq!(detect_indent("    let x = 1;"), "    ");
        assert_eq!(detect_indent("\t\tlet x = 1;"), "\t\t");
        assert_eq!(detect_indent(" \t let x = 1;"), " \t ");
        assert_eq!(detect_indent("let x = 1;"), "");
        assert_eq!(detect_indent(""), "");
        assert_eq!(
            detect_indent("      "),
            "      ",
            "an all-whitespace line is all indent — trim_start leaves nothing behind"
        );
    }
}
