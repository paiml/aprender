//! `pv score` — Quantitative contract and codebase scoring.

use std::path::Path;

use std::collections::HashSet;

use provable_contracts::binding::BindingRegistry;
use provable_contracts::query::ContractIndex;
use provable_contracts::schema::parse_contract;
use provable_contracts::scoring;
use provable_contracts::scoring::drift;
use provable_contracts::scoring::pvscore_10dim;
use provable_contracts::scoring::{CodebaseScore, ContractScore, Grade, ScoringWeights};
use serde_json::Value;

use crate::json_obj::obj;

#[allow(clippy::too_many_arguments)]
pub fn run(
    path: &Path,
    binding: Option<&Path>,
    format: &str,
    min_score: Option<f64>,
    summary: bool,
    top_gaps: usize,
    weights_json: Option<&str>,
    pvscore: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let binding_registry = binding
        .map(|p| {
            let content = std::fs::read_to_string(p)?;
            let reg: BindingRegistry = serde_yaml::from_str(&content)?;
            Ok::<_, Box<dyn std::error::Error>>(reg)
        })
        .transpose()?;

    let weights = match weights_json {
        Some(json) => serde_json::from_str::<ScoringWeights>(json)?,
        None => ScoringWeights::default(),
    };

    if path.is_dir() {
        run_directory(
            path,
            binding_registry.as_ref(),
            binding,
            format,
            min_score,
            summary,
            top_gaps,
            &weights,
            pvscore,
        )
    } else {
        run_single(path, binding_registry.as_ref(), format, min_score, &weights)
    }
}

fn run_single(
    path: &Path,
    binding: Option<&BindingRegistry>,
    format: &str,
    min_score: Option<f64>,
    weights: &ScoringWeights,
) -> Result<(), Box<dyn std::error::Error>> {
    let contract = parse_contract(path)?;
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    let score = scoring::score_contract_weighted(&contract, binding, stem, weights);

    match format {
        "json" => println!("{}", serde_json::to_string_pretty(&score)?),
        "markdown" => print!("{}", score_to_markdown(&score)),
        _ => {
            print!("{score}");
            print_probes(&score);
        }
    }

    if let Some(threshold) = min_score {
        if score.composite < threshold {
            return Err(format!(
                "Score {:.2} below threshold {threshold:.2}",
                score.composite,
            )
            .into());
        }
    }

    Ok(())
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::cast_precision_loss
)]
fn run_directory(
    dir: &Path,
    binding: Option<&BindingRegistry>,
    binding_path: Option<&Path>,
    format: &str,
    min_score: Option<f64>,
    summary: bool,
    top_gaps: usize,
    weights: &ScoringWeights,
    pvscore: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut yaml_paths: Vec<std::path::PathBuf> = Vec::new();
    collect_yaml_files(dir, &mut yaml_paths);
    yaml_paths.sort();

    let mut scores = Vec::new();
    for path in &yaml_paths {
        let Ok(contract) = parse_contract(path) else {
            continue;
        };
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        scores.push(scoring::score_contract_weighted(
            &contract, binding, stem, weights,
        ));
    }

    let mean: f64 = if scores.is_empty() {
        0.0
    } else {
        scores.iter().map(|s| s.composite).sum::<f64>() / scores.len() as f64
    };

    if summary {
        print_summary_only(&scores, mean, format)?;
    } else {
        print_directory_scores(&scores, mean, format)?;
    }

    // Show top gaps by lowest score
    if top_gaps > 0 && !scores.is_empty() {
        print_top_gaps(&scores, top_gaps, format);
    }

    // Codebase scoring if binding is provided
    if let Some(binding) = binding {
        let mut parsed = Vec::new();
        for path in &yaml_paths {
            let Ok(contract) = parse_contract(path) else {
                continue;
            };
            let stem = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();
            parsed.push((stem, contract));
        }
        let refs: Vec<_> = parsed.iter().map(|(s, c)| (s.clone(), c)).collect();

        // Build pagerank from contract index for impact-weighted gap analysis
        let pagerank = ContractIndex::from_directory(dir).ok().map(|idx| {
            idx.entries
                .iter()
                .filter_map(|e| idx.cached_pagerank(&e.stem).map(|s| (e.stem.clone(), s)))
                .collect::<std::collections::HashMap<String, f64>>()
        });

        // CD5: Detect stale contracts via git timestamps
        let drift_score = binding_path.map(|bp| {
            let bound_stems: HashSet<&str> = binding
                .bindings
                .iter()
                .map(|b| b.contract.as_str())
                .collect();
            let stale = drift::detect_stale_contracts(dir, bp, &bound_stems);
            drift::compute_drift(stale.len(), bound_stems.len())
        });

        let codebase = scoring::score_codebase_full(&refs, binding, pagerank.as_ref(), drift_score);

        match format {
            "json" => {
                let output = obj([("codebase", serde_json::to_value(&codebase)?)]);
                println!("{}", serde_json::to_string_pretty(&output)?);
            }
            "markdown" => {
                println!("\n## Codebase Score\n");
                println!("| Dimension | Value |");
                println!("|-----------|-------|");
                println!("| Coverage | {:.0}% |", codebase.contract_coverage * 100.0);
                println!(
                    "| Binding | {:.0}% |",
                    codebase.binding_completeness * 100.0
                );
                println!("| Mean Score | {:.2} |", codebase.mean_contract_score);
                println!("| Proof Depth | {:.2} |", codebase.proof_depth_dist);
                println!("| Drift | {:.2} |", codebase.drift);
                println!(
                    "\n**Composite:** {:.2} (Grade {})",
                    codebase.composite, codebase.grade
                );
            }
            _ => println!("\n{codebase}"),
        }

        if pvscore {
            print_pvscore(&codebase);
        }
    }

    if let Some(threshold) = min_score {
        if mean < threshold {
            return Err(format!("Mean score {mean:.2} below threshold {threshold:.2}").into());
        }
    }

    Ok(())
}

#[allow(clippy::cast_precision_loss)]
fn print_directory_scores(
    scores: &[ContractScore],
    mean: f64,
    format: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    match format {
        "json" => {
            let output = obj([
                ("contracts", Value::from(scores.len())),
                ("mean_score", Value::from(mean)),
                (
                    "mean_grade",
                    Value::from(scoring::Grade::from_score(mean).to_string()),
                ),
                ("scores", serde_json::to_value(scores)?),
            ]);
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        "markdown" => {
            println!("## Contract Scores\n");
            println!("| Contract | Score | Grade | Spec | Falsify | Kani | Lean | Bind |");
            println!("|----------|-------|-------|------|---------|------|------|------|");
            for s in scores {
                println!(
                    "| {} | {:.2} | {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} |",
                    s.stem,
                    s.composite,
                    s.grade,
                    s.spec_depth,
                    s.falsification_coverage,
                    s.kani_coverage,
                    s.lean_coverage,
                    s.binding_coverage
                );
            }
            println!(
                "\n**{} contracts** — Mean: {mean:.2} (Grade {})",
                scores.len(),
                scoring::Grade::from_score(mean)
            );
        }
        _ => {
            for s in scores {
                print!("{s}");
            }
            println!(
                "\n{} contracts — Mean: {mean:.2} (Grade {})",
                scores.len(),
                scoring::Grade::from_score(mean)
            );
        }
    }
    Ok(())
}

#[allow(clippy::cast_precision_loss)]
fn print_summary_only(
    scores: &[ContractScore],
    mean: f64,
    format: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let grade = scoring::Grade::from_score(mean);
    match format {
        "json" => {
            let output = obj([
                ("contracts", Value::from(scores.len())),
                ("mean_score", Value::from(mean)),
                ("mean_grade", Value::from(grade.to_string())),
            ]);
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        "markdown" => {
            println!(
                "**{} contracts** — Mean: {mean:.2} (Grade {grade})",
                scores.len()
            );
        }
        _ => {
            println!(
                "{} contracts — Mean: {mean:.2} (Grade {grade})",
                scores.len()
            );
        }
    }
    Ok(())
}

fn print_top_gaps(scores: &[ContractScore], n: usize, format: &str) {
    let mut sorted: Vec<_> = scores.iter().collect();
    sorted.sort_by(|a, b| {
        a.composite
            .partial_cmp(&b.composite)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top: Vec<_> = sorted.into_iter().take(n).collect();

    match format {
        "json" => {} // Already in JSON output
        "markdown" => {
            println!("\n### Top {} Gaps\n", top.len());
            for s in &top {
                println!("- **{}** — {:.2} ({})", s.stem, s.composite, s.grade);
            }
        }
        _ => {
            println!("\nTop {} gaps:", top.len());
            for s in &top {
                println!("  {} — {:.2} ({})", s.stem, s.composite, s.grade);
            }
        }
    }
}

fn print_pvscore(codebase: &CodebaseScore) {
    let pv = pvscore_10dim(codebase);
    let grade = Grade::from_score(pv / 100.0);
    println!("\nPVScore: {pv:.1} (Grade: {grade})");
    println!(
        "  D1  Spec Depth:        {:.1}",
        codebase.contract_coverage * 100.0
    );
    println!(
        "  D2  Falsification:     {:.1}",
        codebase.binding_completeness * 100.0
    );
    println!(
        "  D3  Mean Score:        {:.1}",
        codebase.mean_contract_score * 100.0
    );
    println!(
        "  D4  Proof Depth:       {:.1}",
        codebase.proof_depth_dist * 100.0
    );
    println!("  D5  Drift:             {:.1}", codebase.drift * 100.0);
    println!(
        "  D6  Reverse Coverage:  {:.1}{}",
        codebase.reverse_coverage * 100.0,
        default_suffix(codebase.reverse_coverage)
    );
    println!(
        "  D7  Mutation Testing:  {:.1}{}",
        codebase.mutation_testing * 100.0,
        default_suffix(codebase.mutation_testing)
    );
    println!(
        "  D8  CI Pipeline:       {:.1}{}",
        codebase.ci_pipeline_depth * 100.0,
        default_suffix(codebase.ci_pipeline_depth)
    );
    println!(
        "  D9  Proof Freshness:   {:.1}{}",
        codebase.proof_freshness * 100.0,
        default_suffix(codebase.proof_freshness)
    );
    println!(
        "  D10 Defect Patterns:   {:.1}{}",
        codebase.defect_patterns * 100.0,
        default_suffix(codebase.defect_patterns)
    );
}

fn default_suffix(value: f64) -> &'static str {
    if value == 0.0 {
        " (default)"
    } else {
        ""
    }
}

fn score_to_markdown(score: &ContractScore) -> String {
    format!(
        "### {}\n\n- **Score:** {:.2} (Grade {})\n- Spec: {:.2} | Falsify: {:.2} | Kani: {:.2} | Lean: {:.2} | Bind: {:.2}\n",
        score.stem,
        score.composite,
        score.grade,
        score.spec_depth,
        score.falsification_coverage,
        score.kani_coverage,
        score.lean_coverage,
        score.binding_coverage
    )
}

/// Print probe-level score decomposition for a single contract.
///
/// Groups probes by dimension and prints each with a pass/fail indicator,
/// showing what contributed to each dimension's score.
fn print_probes(score: &ContractScore) {
    if score.probes.is_empty() {
        return;
    }

    // Dimension display order and labels
    let dimensions = [
        ("spec_depth", "D1 Spec Depth", score.spec_depth),
        (
            "falsification",
            "D2 Falsification",
            score.falsification_coverage,
        ),
        ("kani", "D3 Kani", score.kani_coverage),
        ("lean", "D4 Lean", score.lean_coverage),
        ("binding", "D5 Binding", score.binding_coverage),
    ];

    println!("  Probes:");
    for (dim_key, dim_label, dim_score) in &dimensions {
        let dim_probes: Vec<_> = score
            .probes
            .iter()
            .filter(|p| p.dimension == *dim_key)
            .collect();
        if dim_probes.is_empty() {
            continue;
        }
        println!("  {dim_label:20} {dim_score:.2}");
        for p in &dim_probes {
            let icon = if p.outcome { "+" } else { "-" };
            println!("    {icon} {}: {}", p.probe, p.detail);
        }
    }
}

/// Recursively collect all `.yaml` contract files from a directory.
///
/// Skips:
/// - `binding.yaml` (binding registries, not contracts)
/// - `kaizen/` directories (work items, not contracts)
/// - `legacy/` directories (deprecated contracts)
/// - `pipelines/` directories (pipeline contracts, different schema)
fn collect_yaml_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let dirname = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            // Skip non-contract directories
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
