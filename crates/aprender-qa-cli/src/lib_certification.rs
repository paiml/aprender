// Certification functions for the CLI library.
// This file is included by lib.rs via include!().

/// Build an ExecutionConfig for dimensional smoke tier.
///
/// Minimal config: no conversion, no golden rule, no contracts, no profiling.
/// SafeTensors-only, CPU-only, single worker, 30s timeout, fail-fast.
fn build_dimensional_smoke_config(model_cache_path: Option<String>) -> ExecutionConfig {
    ExecutionConfig {
        failure_policy: FailurePolicy::FailFast,
        dry_run: false,
        max_workers: 1,
        model_path: model_cache_path,
        default_timeout_ms: 30_000,
        no_gpu: true,
        run_conversion_tests: false,
        run_profile_ci: false,
        run_golden_rule_test: false,
        golden_reference_path: None,
        lock_file_path: None,
        playbook_file_path: None,
        check_integrity: false,
        warn_implicit_skips: false,
        run_hf_parity: false,
        hf_parity_corpus_path: None,
        hf_parity_model_family: None,
        output_dir: Some("output".to_string()),
        run_contract_tests: false,
        run_ollama_parity: false,
        metadata_only: true,
    }
}

/// Generate playbook path from model ID and tier
pub fn playbook_path_for_model(model_id: &str, tier: CertTier) -> String {
    let short = model_id.split('/').next_back().unwrap_or(model_id);
    let base = short
        .to_lowercase()
        .replace("-instruct", "")
        .replace("-it", "");
    format!(
        "playbooks/models/{}{}.playbook.yaml",
        base,
        tier.playbook_suffix()
    )
}

/// Bootstrap a playbook from a family contract.
///
/// Loads the family contract from the contracts directory, extracts architecture
/// constraints and size variant, then generates an architecture-aware playbook
/// with kernel-targeted prompts.
pub fn bootstrap_playbook_from_contract(
    family: &str,
    size: &str,
    hf_repo: &str,
    tier: &str,
    contracts_path: &std::path::Path,
) -> Result<String, String> {
    use apr_qa_gen::bootstrapper;
    use apr_qa_runner::family_contract::FamilyRegistry;

    let mut registry = FamilyRegistry::with_path(contracts_path);
    let contract = registry
        .load_family(family)
        .map_err(|e| format!("Failed to load family contract '{family}': {e}"))?;

    let variant = contract.get_size_variant(size).ok_or_else(|| {
        let available: Vec<&str> = contract.size_variants.keys().map(String::as_str).collect();
        format!(
            "Size variant '{size}' not found for family '{family}'. Available: {}",
            available.join(", ")
        )
    })?;

    let constraints = contract
        .constraints
        .clone()
        .ok_or_else(|| format!("Family '{family}' has no constraints defined in contract"))?;

    let arch_constraints = constraints.to_arch_constraints();
    let arch_variant = variant.to_arch_size_variant();
    let size_category = contract.get_size_category(size).unwrap_or("small");

    let config = apr_qa_gen::BootstrapConfig {
        family: family.to_string(),
        size_variant: size.to_string(),
        hf_repo: hf_repo.to_string(),
        tier: tier.to_string(),
        kernel_profile: None,
    };

    let playbook =
        bootstrapper::bootstrap_playbook(&config, &arch_constraints, &arch_variant, size_category);

    bootstrapper::to_yaml(&playbook)
}

/// Certify a single model with the given configuration
///
/// Returns a `ModelCertificationResult` with the outcome.
pub fn certify_model(model_id: &str, config: &CertificationConfig) -> ModelCertificationResult {
    let playbook_path = playbook_path_for_model(model_id, config.tier);
    let playbook_file = std::path::Path::new(&playbook_path);

    if !playbook_file.exists() {
        return ModelCertificationResult {
            model_id: model_id.to_string(),
            success: false,
            mqs_score: 0,
            grade: "-".to_string(),
            pass_rate: 0.0,
            gateway_failed: None,
            error: Some(format!("Playbook not found: {playbook_path}")),
        };
    }

    let playbook = match load_playbook(playbook_file) {
        Ok(p) => p,
        Err(e) => {
            return ModelCertificationResult {
                model_id: model_id.to_string(),
                success: false,
                mqs_score: 0,
                grade: "-".to_string(),
                pass_rate: 0.0,
                gateway_failed: None,
                error: Some(e),
            };
        }
    };

    // Build model cache path
    let short = model_id.split('/').next_back().unwrap_or(model_id);
    let model_cache_path = config.model_cache.as_ref().map(|cache| {
        cache
            .join(short.to_lowercase().replace('.', "-"))
            .to_string_lossy()
            .to_string()
    });

    let exec_config = build_certification_config(config.tier, model_cache_path);

    match execute_playbook(&playbook, exec_config) {
        Ok(result) => {
            let evidence_vec: Vec<_> = result.evidence.all().to_vec();
            let collector = collect_evidence(evidence_vec);

            let pass_rate = result.pass_rate();
            let gateway_failed = result.gateway_failed;
            match calculate_mqs_score(model_id, &collector) {
                Ok(mqs) => ModelCertificationResult {
                    model_id: model_id.to_string(),
                    success: true,
                    mqs_score: mqs.raw_score,
                    grade: mqs.grade,
                    pass_rate,
                    gateway_failed,
                    error: None,
                },
                Err(e) => ModelCertificationResult {
                    model_id: model_id.to_string(),
                    success: false,
                    mqs_score: 0,
                    grade: "-".to_string(),
                    pass_rate,
                    gateway_failed: None, // MQS calculation failed, gateway status unknown
                    error: Some(e),
                },
            }
        }
        Err(e) => ModelCertificationResult {
            model_id: model_id.to_string(),
            success: false,
            mqs_score: 0,
            grade: "-".to_string(),
            pass_rate: 0.0,
            gateway_failed: None,
            error: Some(e),
        },
    }
}

/// Generate a playbook lock file from all YAML playbooks in a directory (§3.1)
///
/// Scans the directory recursively for `.playbook.yaml` files, computes SHA-256
/// hashes, and writes the lock file.
///
/// # Errors
///
/// Returns an error if the directory cannot be read or the lock file cannot be written.
pub fn generate_lock_file(dir: &Path, output: &Path) -> Result<usize, String> {
    use apr_qa_runner::{PlaybookLockFile, generate_lock_entry, save_lock_file};
    use std::collections::HashMap;

    // Walk directory for .playbook.yaml files
    /// Recursively walk a directory collecting playbook lock entries
    fn walk_dir(dir: &Path, entries: &mut HashMap<String, apr_qa_runner::PlaybookLockEntry>) {
        let Ok(read_dir) = std::fs::read_dir(dir) else {
            eprintln!(
                "[WARN] Cannot read directory: {}",
                dir.display()
            );
            return;
        };
        for entry in read_dir.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk_dir(&path, entries);
            } else if path
                .file_name()
                .is_some_and(|n| n.to_string_lossy().ends_with(".playbook.yaml"))
            {
                match generate_lock_entry(&path) {
                    Ok((name, lock_entry)) => {
                        if entries.contains_key(&name) {
                            eprintln!(
                                "[WARN] Duplicate playbook name '{}' at {}",
                                name,
                                path.display()
                            );
                        }
                        entries.insert(name, lock_entry);
                    }
                    Err(e) => {
                        eprintln!(
                            "[WARN] Failed to hash {}: {e}",
                            path.display()
                        );
                    }
                }
            }
        }
    }

    let mut entries = HashMap::new();
    walk_dir(dir, &mut entries);
    let count = entries.len();

    let lock_file = PlaybookLockFile { entries };
    save_lock_file(&lock_file, output).map_err(|e| format!("Failed to save lock file: {e}"))?;

    Ok(count)
}

/// Execute auto-ticket generation from evidence using the defect-fixture map (§3.6)
///
/// Classifies failures, deduplicates by root cause, and returns structured tickets.
pub fn execute_auto_tickets(evidence: &[Evidence], _repo: &str) -> Vec<UpstreamTicket> {
    let defect_map = match apr_qa_report::defect_map::load_defect_fixture_map() {
        Ok(map) => map,
        Err(e) => {
            eprintln!("[WARN] Could not load defect fixture map: {e}");
            return Vec::new();
        }
    };

    generate_structured_tickets(evidence, &defect_map)
}
