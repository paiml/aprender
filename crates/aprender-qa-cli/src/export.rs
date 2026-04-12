
/// Auto-populate model cache directory with symlinks from pacha and HF caches.
///
/// Creates `gguf/`, `apr/`, `safetensors/` subdirectories and symlinks model files
/// from the pacha cache (`~/.cache/pacha/models/`) and HuggingFace cache
/// (`~/.cache/huggingface/hub/`). The `apr/` subdirectory is populated during
/// 6-column profiling (GGUF → APR conversion).
fn auto_populate_model_cache(model_id: &str, model_dir: &std::path::Path, apr_binary: &str) {
    let gguf_dir = model_dir.join("gguf");
    let apr_dir = model_dir.join("apr");
    let st_dir = model_dir.join("safetensors");

    if gguf_dir.exists() && has_file_with_ext(&gguf_dir, "gguf") {
        println!("  Cache already populated: {}", model_dir.display());
        return;
    }

    println!("  Auto-populating model cache...");

    for dir in [&gguf_dir, &apr_dir, &st_dir] {
        if let Err(e) = std::fs::create_dir_all(dir) {
            eprintln!("  Error creating {}: {e}", dir.display());
            return;
        }
    }

    run_apr_pull(apr_binary, model_id);

    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let home = std::path::Path::new(&home);

    link_gguf_from_pacha(model_id, home, &gguf_dir);
    link_safetensors_from_hf(model_id, home, &st_dir);
}

/// Execute `apr pull` to download the model into the local cache
fn run_apr_pull(apr_binary: &str, model_id: &str) {
    println!("  Running: {apr_binary} pull {model_id}");
    let pull_status = std::process::Command::new(apr_binary)
        .args(["pull", model_id])
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .status();

    match pull_status {
        Ok(s) if s.success() => println!("  Pull succeeded"),
        Ok(s) => eprintln!("  Pull exited with: {s}"),
        Err(e) => eprintln!("  Pull failed: {e}"),
    }
}

/// Create a symlink to the GGUF file from the pacha cache into the model cache
fn link_gguf_from_pacha(model_id: &str, home: &std::path::Path, gguf_dir: &std::path::Path) {
    let manifest_path = home.join(".cache/pacha/models/manifest.json");
    let Some(gguf_path) = find_gguf_in_pacha(&manifest_path, model_id) else {
        eprintln!("  No GGUF found in pacha cache for {model_id}");
        return;
    };
    let link = gguf_dir.join("model.gguf");
    if link.exists() {
        return;
    }
    match std::os::unix::fs::symlink(&gguf_path, &link) {
        Ok(()) => println!("  Linked GGUF: {gguf_path}"),
        Err(e) => eprintln!("  Error symlinking GGUF: {e}"),
    }
}

/// Create symlinks to SafeTensors files and config.json from the HuggingFace cache
fn link_safetensors_from_hf(model_id: &str, home: &std::path::Path, st_dir: &std::path::Path) {
    let (org, repo) = split_model_id(model_id);
    let hf_model_dir = home
        .join(".cache/huggingface/hub")
        .join(format!("models--{org}--{repo}"))
        .join("snapshots");

    let Some(st_path) = find_safetensors_in_hf(&hf_model_dir) else {
        eprintln!("  No SafeTensors found in HF cache for {model_id}");
        return;
    };

    let link = st_dir.join("model.safetensors");
    if !link.exists() {
        match std::os::unix::fs::symlink(&st_path, &link) {
            Ok(()) => println!("  Linked SafeTensors: {}", st_path.display()),
            Err(e) => eprintln!("  Error symlinking SafeTensors: {e}"),
        }
    }

    // Copy config.json from the same snapshot directory
    let Some(snapshot_dir) = st_path.parent() else {
        return;
    };
    let config_src = snapshot_dir.join("config.json");
    let config_dst = st_dir.join("config.json");
    if config_src.exists() && !config_dst.exists() {
        match std::fs::copy(&config_src, &config_dst) {
            Ok(_) => println!("  Copied config.json"),
            Err(e) => eprintln!("  Error copying config.json: {e}"),
        }
    }
}

/// Check if a directory contains a file with the given extension.
fn has_file_with_ext(dir: &std::path::Path, ext: &str) -> bool {
    dir.read_dir()
        .map(|entries| {
            entries
                .flatten()
                .any(|e| e.path().extension().is_some_and(|x| x == ext))
        })
        .unwrap_or(false)
}

/// Find a GGUF file in the pacha cache manifest matching the model ID.
///
/// Pacha manifest entries use the naming convention:
/// `hf_Org_Repo-GGUF_repo-name-q4_k_m.gguf`
fn find_gguf_in_pacha(manifest_path: &std::path::Path, model_id: &str) -> Option<String> {
    let content = std::fs::read_to_string(manifest_path).ok()?;
    let entries: Vec<serde_json::Value> = serde_json::from_str(&content).ok()?;

    // Build search key from model_id: "Qwen/Qwen2.5-Coder-1.5B-Instruct" → "Qwen_Qwen2.5-Coder-1.5B-Instruct"
    let (org, repo) = split_model_id(model_id);
    let gguf_key = format!("hf_{org}_{repo}-GGUF_");

    // Find first GGUF entry matching this model
    for entry in &entries {
        let name = entry.get("name")?.as_str()?;
        if name.starts_with(&gguf_key)
            && std::path::Path::new(name)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        {
            return entry.get("path")?.as_str().map(String::from);
        }
    }

    None
}

/// Find a `model.safetensors` file in the HuggingFace cache snapshots directory.
fn find_safetensors_in_hf(snapshots_dir: &std::path::Path) -> Option<std::path::PathBuf> {
    let entries = std::fs::read_dir(snapshots_dir).ok()?;
    for entry in entries.flatten() {
        let snapshot = entry.path();
        if snapshot.is_dir() {
            let st_file = snapshot.join("model.safetensors");
            if st_file.exists() {
                return Some(st_file);
            }
        }
    }
    None
}

/// Split a HuggingFace model ID into (org, repo).
///
/// e.g. `"Qwen/Qwen2.5-Coder-1.5B-Instruct"` → `("Qwen", "Qwen2.5-Coder-1.5B-Instruct")`
fn split_model_id(model_id: &str) -> (&str, &str) {
    model_id.split_once('/').unwrap_or(("unknown", model_id))
}

/// Export certification data to models.csv (PMAT-264)
///
/// Scans evidence directory, calculates MQS for each evidence file,
/// and writes/updates models.csv for oracle consumption.
#[allow(clippy::too_many_lines)]
fn export_csv(evidence_dir: &Path, output: &Path, append: bool) {
    use apr_qa_report::write_models_csv;

    println!("Exporting certification data to CSV...");
    println!("  Evidence directory: {}", evidence_dir.display());
    println!("  Output: {}", output.display());
    println!("  Mode: {}", if append { "append" } else { "overwrite" });

    let mut rows = load_existing_csv_rows(output, append);
    let (processed, updated) = process_evidence_files(evidence_dir, &mut rows);

    if processed == 0 {
        eprintln!(
            "Error: No EvidenceExport files found in {}.\n  If using `certifications/` directory, run `apr-qa export-evidence` first.",
            evidence_dir.display()
        );
        std::process::exit(1);
    }

    ensure_parent_dir(output);
    if let Err(e) = write_models_csv(&rows, output) {
        eprintln!("Error: Failed to write CSV: {e}");
        std::process::exit(1);
    }

    println!("\nExported {} row(s) to {}", rows.len(), output.display());
    println!("  Processed: {processed}");
    println!("  Updated: {updated}");
    println!("  New: {}", processed - updated);
}

/// Load existing CSV rows when appending, or return an empty vec
fn load_existing_csv_rows(output: &Path, append: bool) -> Vec<apr_qa_report::CertificationRow> {
    use apr_qa_report::read_models_csv;

    if !append || !output.exists() {
        return Vec::new();
    }
    match read_models_csv(output) {
        Ok(existing) => {
            println!("  Loaded {} existing row(s)", existing.len());
            existing
        }
        Err(e) => {
            eprintln!("Warning: Could not read existing CSV: {e}");
            Vec::new()
        }
    }
}

/// Scan the evidence directory and upsert rows from each JSON evidence file.
///
/// Searches both top-level JSON files AND `{dir}/{model}/evidence.json` files
/// (the structure written by `apr-qa certify` and `apr-qa run`). Top-level files
/// must be in `EvidenceExport` format (written by `export-evidence`). For nested
/// plain-array evidence files, a summary hint is printed once at the end.
fn process_evidence_files(
    evidence_dir: &Path,
    rows: &mut Vec<apr_qa_report::CertificationRow>,
) -> (usize, usize) {
    let entries = match std::fs::read_dir(evidence_dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Error: Cannot read evidence directory: {e}");
            std::process::exit(1);
        }
    };

    let mut processed = 0;
    let mut updated = 0;
    let mut plain_array_count = 0;

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            // Nested: {evidence_dir}/{model}/evidence.json
            let nested = path.join("evidence.json");
            if nested.exists() {
                process_single_file(&nested, rows, &mut processed, &mut updated, &mut plain_array_count);
            }
        } else if path.extension().is_some_and(|ext| ext == "json") {
            process_single_file(&path, rows, &mut processed, &mut updated, &mut plain_array_count);
        }
    }

    if plain_array_count > 0 {
        eprintln!(
            "  Hint: {plain_array_count} plain evidence array(s) skipped.\n  Run `apr-qa export-evidence` for each model first, then re-run export-csv."
        );
    }

    (processed, updated)
}

/// Attempt to parse a JSON file as EvidenceExport and upsert into rows.
fn process_single_file(
    path: &Path,
    rows: &mut Vec<apr_qa_report::CertificationRow>,
    processed: &mut usize,
    updated: &mut usize,
    plain_array_count: &mut usize,
) {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  Warning: Could not read {}: {e}", path.display());
            return;
        }
    };
    let Ok(export) = serde_json::from_str::<apr_qa_report::EvidenceExport>(&content) else {
        // Plain Evidence arrays (from `certify`/`run`) can't be processed directly.
        // The user must run `export-evidence` first to add model metadata.
        if content.trim_start().starts_with('[') {
            *plain_array_count += 1;
        } else {
            eprintln!("  Warning: Malformed JSON (skipped): {}", path.display());
        }
        return;
    };
    *processed += 1;
    let was_updated = update_row_from_export(rows, &export);
    if was_updated {
        *updated += 1;
    }
}

/// Update or insert a certification row from an evidence export record
#[allow(clippy::option_if_let_else, clippy::single_match_else)]
fn update_row_from_export(
    rows: &mut Vec<apr_qa_report::CertificationRow>,
    export: &apr_qa_report::EvidenceExport,
) -> bool {
    use apr_qa_report::CertificationRow;
    use chrono::Utc;

    let model_id = &export.model.hf_repo;
    // Can't use map_or_else here due to borrow checker - we need mutable access to rows
    let (row_idx, was_updated) = match rows.iter().position(|r| r.model_id == *model_id) {
        Some(idx) => (idx, true),
        None => {
            rows.push(CertificationRow::new(model_id, &export.model.family));
            (rows.len() - 1, false)
        }
    };

    let row = &mut rows[row_idx];
    row.parameters.clone_from(&export.model.size);
    row.mqs_score = export.mqs.score;
    row.grade.clone_from(&export.mqs.grade);
    row.certified_tier.clone_from(&export.playbook.tier);
    row.last_certified = Utc::now();
    row.status = derive_status_from_mqs(&export.mqs);
    update_gateway_flags(row, &export.gates);

    println!(
        "  Processed: {} → MQS {}, {}",
        model_id, row.mqs_score, row.status
    );
    was_updated
}

/// Derive the model certification status from MQS score and gateway results.
///
/// Called from `process_evidence_files()` — the presence of an evidence file
/// means the model WAS tested. Score 0 here means tested-and-failed, not
/// untested. Use `Blocked`, not `Pending`.
#[allow(clippy::missing_const_for_fn)] // Can't be const due to internal use statement
fn derive_status_from_mqs(mqs: &apr_qa_report::MqsExport) -> apr_qa_report::ModelStatus {
    use apr_qa_report::ModelStatus;

    if mqs.score >= 800 && mqs.gateway_passed {
        ModelStatus::Certified
    } else {
        ModelStatus::Blocked
    }
}

/// Set per-gateway pass/fail flags (G1-G4) on the certification row
fn update_gateway_flags(
    row: &mut apr_qa_report::CertificationRow,
    gates: &std::collections::HashMap<String, apr_qa_report::GateResult>,
) {
    if let Some(g1) = gates.get("G1-MODEL-LOADS") {
        row.g1 = g1.passed;
    }
    if let Some(g2) = gates.get("G2-BASIC-INFERENCE") {
        row.g2 = g2.passed;
    }
    if let Some(g3) = gates.get("G3-NO-CRASHES") {
        row.g3 = g3.passed;
    }
    if let Some(g4) = gates.get("G4-OUTPUT-QUALITY") {
        row.g4 = g4.passed;
    }
}

/// Create the parent directory for the given path if it does not exist
fn ensure_parent_dir(path: &Path) {
    if let Some(parent) = path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            eprintln!("Error: Cannot create output directory: {e}");
            std::process::exit(1);
        }
    }
}

include!("contract.rs");
