//! Playbook Integrity Lock Demo (§3.1)
//!
//! Demonstrates the playbook integrity lock system for preventing
//! unauthorized modification of test specifications.
//!
//! Run with:
//! ```bash
//! cargo run --example integrity_lock_demo -p apr-qa-cli
//! ```

#![allow(clippy::expect_used)]
#![allow(clippy::unwrap_used)]

use aprender_qa_runner::{
    compute_playbook_hash, generate_lock_entry, load_lock_file, save_lock_file,
    verify_playbook_integrity, PlaybookLockEntry, PlaybookLockFile,
};
use tempfile::TempDir;

fn main() {
    println!("=== Playbook Integrity Lock Demo (§3.1) ===\n");

    // Create a temporary directory for demo
    let tmp = TempDir::new().expect("create temp dir");

    // Create a sample playbook
    let playbook_content = r#"
name: demo-model-mvp
version: "1.0.0"
model:
  hf_repo: "Demo/Model-1B-Instruct"
  formats: [safetensors]
test_matrix:
  modalities: [run, chat]
  backends: [cpu]
  scenario_count: 3
falsification_gates:
  - id: G1-LOAD
    description: "Model loads successfully"
    condition: "exit_code == 0"
    severity: P0
"#;

    let playbook_path = tmp.path().join("demo-model-mvp.playbook.yaml");
    std::fs::write(&playbook_path, playbook_content).expect("write playbook");

    // Demonstrate hash computation
    println!("=== Hash Computation ===\n");
    let hash = compute_playbook_hash(&playbook_path).expect("compute hash");
    println!("Playbook: {}", playbook_path.display());
    println!("SHA-256:  {hash}");
    println!("(64 hex characters = 256 bits)\n");

    // Demonstrate lock entry generation
    println!("=== Lock Entry Generation ===\n");
    let (name, entry) = generate_lock_entry(&playbook_path).expect("generate entry");
    println!("Name: {name}");
    println!("Hash: {}", entry.sha256);
    println!("Locked fields:");
    for field in &entry.locked_fields {
        println!("  - {field}");
    }
    println!();

    // Demonstrate lock file structure
    println!("=== Lock File Structure ===\n");
    let mut lock_file = PlaybookLockFile::default();
    lock_file.entries.insert(name.clone(), entry.clone());

    // Add another entry for demonstration
    lock_file.entries.insert(
        "other-model-quick".to_string(),
        PlaybookLockEntry {
            sha256: "abcd1234...".to_string(),
            locked_fields: vec![
                "model.hf_repo".to_string(),
                "model.formats".to_string(),
                "test_matrix".to_string(),
            ],
        },
    );

    let lock_yaml = serde_yaml::to_string(&lock_file).expect("serialize");
    println!("playbook.lock.yaml:");
    println!("---");
    for line in lock_yaml.lines().take(20) {
        println!("{line}");
    }
    println!("...\n");

    // Demonstrate save/load roundtrip
    println!("=== Save/Load Roundtrip ===\n");
    let lock_path = tmp.path().join("playbook.lock.yaml");
    save_lock_file(&lock_file, &lock_path).expect("save lock file");
    println!("Saved to: {}", lock_path.display());

    let loaded = load_lock_file(&lock_path).expect("load lock file");
    println!("Loaded {} entries", loaded.entries.len());
    println!();

    // Demonstrate integrity verification - PASS case
    println!("=== Integrity Verification ===\n");

    println!("Case 1: Unmodified playbook");
    match verify_playbook_integrity(&playbook_path, &lock_file, &name) {
        Ok(()) => println!("  Result: PASSED ✓"),
        Err(e) => println!("  Result: FAILED - {e}"),
    }
    println!();

    // Demonstrate integrity verification - FAIL case
    println!("Case 2: Modified playbook");

    // Modify the playbook
    let modified_content = playbook_content.replace("scenario_count: 3", "scenario_count: 1");
    std::fs::write(&playbook_path, modified_content).expect("write modified");

    match verify_playbook_integrity(&playbook_path, &lock_file, &name) {
        Ok(()) => println!("  Result: PASSED (unexpected!)"),
        Err(e) => {
            println!("  Result: BLOCKED ✗");
            println!("  Error: {e}");
        }
    }
    println!();

    // Demonstrate hash difference
    println!("=== Hash Comparison ===\n");
    let new_hash = compute_playbook_hash(&playbook_path).expect("compute new hash");
    println!("Original hash: {}", entry.sha256);
    println!("Modified hash: {new_hash}");
    println!("Hashes match:  {}", entry.sha256 == new_hash);
    println!();

    // CLI usage summary
    println!("=== CLI Usage ===\n");
    println!("# Generate lock file for all playbooks:");
    println!("apr-qa lock-playbooks playbooks/models -o playbooks/playbook.lock.yaml\n");

    println!("# Run with integrity check (default):");
    println!("apr-qa run playbook.yaml");
    println!("# Output: Integrity check: PASSED\n");

    println!("# If playbook was modified:");
    println!("# [INTEGRITY] Playbook hash does not match lock file.");
    println!("# [INTEGRITY] Either:");
    println!("#   1. Run `apr-qa lock-playbooks` to regenerate the lock file");
    println!("#   2. Use --no-integrity-check to bypass (NOT RECOMMENDED)\n");

    println!("# Bypass integrity check (not recommended):");
    println!("apr-qa run playbook.yaml --no-integrity-check\n");

    println!("=== Demo Complete ===");
}
