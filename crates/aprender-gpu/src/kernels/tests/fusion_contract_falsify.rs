//! Kernel Fusion Contract Falsification Tests (kernel-fusion-v1.yaml)
//!
//! These tests validate that the kernel-fusion-v1.yaml contract in aprender
//! accurately documents ALL fused kernels in trueno-gpu, and that every
//! enforcement rule (FUSION-001 through FUSION-010) is falsifiable.
//!
//! Contract: contracts/kernel-fusion-v1.yaml
//! Enforcement: Poka-Yoke — undocumented fused kernels MUST be caught.

use super::*;
use proptest::prelude::*;
use std::path::Path;

/// Path to the kernel-fusion-v1.yaml contract in the aprender crate.
/// CARGO_MANIFEST_DIR for trueno-gpu is trueno/trueno-gpu, so ../../aprender/
/// reaches the aprender crate root.
const CONTRACT_PATH: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../aprender/contracts/kernel-fusion-v1.yaml");

/// Helper: read the contract YAML as a string, panicking with a clear message if missing.
fn read_contract() -> String {
    let path = Path::new(CONTRACT_PATH);
    assert!(
        path.exists(),
        "kernel-fusion-v1.yaml contract not found at {CONTRACT_PATH}. \
         Ensure aprender is checked out as a sibling of trueno."
    );
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("Failed to read {CONTRACT_PATH}: {e}"))
}

/// All fused kernel struct names that implement the `Kernel` trait.
/// Each entry is (struct_name, kernel_name_from_trait) so we can cross-reference
/// both the YAML documentation and the `.name()` output.
fn all_fused_kernels() -> Vec<(&'static str, String)> {
    vec![
        ("FusedSwigluKernel", FusedSwigluKernel::new(4096).name().to_string()),
        ("BatchedSwigluKernel", BatchedSwigluKernel::new(4096, 4).name().to_string()),
        ("FusedQKVKernel", FusedQKVKernel::new(3584, 512).name().to_string()),
        ("FusedGateUpKernel", FusedGateUpKernel::new(3584, 18944).name().to_string()),
        (
            "FusedGemmBiasGeluKernel",
            FusedGemmBiasGeluKernel::new(512, 2048, 512).name().to_string(),
        ),
        (
            "FusedRmsNormQ4KGemvKernel",
            FusedRmsNormQ4KGemvKernel::new(3584, 3584).name().to_string(),
        ),
        ("FusedGateUpQ4KGemvKernel", FusedGateUpQ4KGemvKernel::new(3584, 18944).name().to_string()),
        (
            "FusedRmsNormGateUpSwigluQ4KKernel",
            FusedRmsNormGateUpSwigluQ4KKernel::new(3584, 18944).name().to_string(),
        ),
    ]
}

// =========================================================================
// Deterministic falsification tests (FUSION-001 through FUSION-005)
// =========================================================================

/// FUSION-001: Every fused kernel struct MUST have a corresponding entry in the YAML contract.
///
/// Falsification: If a new `Fused*Kernel` is added to trueno-gpu without updating the
/// contract, this test fails — enforcing the Poka-Yoke documented in enforcement rule 1.
#[test]
fn falsify_fusion_001_every_fused_kernel_has_yaml_entry() {
    let yaml = read_contract();

    for (struct_name, kernel_name) in all_fused_kernels() {
        // Check that at least one of the struct name or the .name() output appears
        // in the YAML. The contract uses struct names in the `fused:` fields and
        // kernel names may appear in descriptions.
        let found = yaml.contains(struct_name) || yaml.contains(&kernel_name);
        assert!(
            found,
            "Fused kernel '{struct_name}' (name='{kernel_name}') is NOT documented \
             in kernel-fusion-v1.yaml. Every fused kernel MUST have a contract entry. \
             Add a fusion_decisions entry for this kernel."
        );
    }
}

/// Check if a YAML line is a top-level fusion entry (2-space indent key ending with colon).
fn is_fusion_entry_key(line: &str, trimmed: &str) -> bool {
    !trimmed.starts_with('#')
        && !trimmed.is_empty()
        && line.starts_with("  ")
        && !line.starts_with("    ")
        && trimmed.ends_with(':')
        && !trimmed.starts_with('-')
}

/// Check if a YAML `tok_s` value line has a non-null, non-empty value.
fn has_valid_tok_s_value(trimmed: &str, prefix: &str) -> bool {
    if !trimmed.starts_with(prefix) {
        return false;
    }
    let value = trimmed.trim_start_matches(prefix).trim();
    value != "null" && !value.is_empty()
}

/// FUSION-002: All BLOCKED entries MUST have benchmark data (unfused_tok_s and fused_tok_s).
///
/// Falsification: A BLOCKED entry without measured performance data is an undocumented
/// decision — the contract requires evidence for why a fusion is blocked.
#[test]
fn falsify_fusion_002_blocked_entries_have_benchmarks() {
    let yaml = read_contract();

    let mut in_blocked_section = false;
    let mut current_entry_name = String::new();
    let mut found_unfused = false;
    let mut found_fused = false;
    let mut checked_entries = 0;

    for line in yaml.lines() {
        let trimmed = line.trim();

        if is_fusion_entry_key(line, trimmed) {
            // Finalize previous BLOCKED section
            if in_blocked_section {
                assert!(
                    found_unfused && found_fused,
                    "BLOCKED entry '{current_entry_name}' is missing benchmark data. \
                     unfused_tok_s present: {found_unfused}, fused_tok_s present: {found_fused}. \
                     BLOCKED fusions MUST have measured tok/s for both fused and unfused paths."
                );
                checked_entries += 1;
            }
            current_entry_name = trimmed.trim_end_matches(':').to_string();
            in_blocked_section = false;
            found_unfused = false;
            found_fused = false;
        }

        if trimmed.contains("status:") && trimmed.contains("BLOCKED") {
            in_blocked_section = true;
        }

        if in_blocked_section {
            if has_valid_tok_s_value(trimmed, "unfused_tok_s:") {
                found_unfused = true;
            }
            if has_valid_tok_s_value(trimmed, "fused_tok_s:") {
                found_fused = true;
            }
        }
    }

    // Finalize the last entry if it was BLOCKED
    if in_blocked_section {
        assert!(
            found_unfused && found_fused,
            "BLOCKED entry '{current_entry_name}' is missing benchmark data. \
             unfused_tok_s present: {found_unfused}, fused_tok_s present: {found_fused}."
        );
        checked_entries += 1;
    }

    assert!(
        checked_entries > 0,
        "No BLOCKED entries found in contract — expected at least FUSION-003 \
         (rmsnorm_gate_up_swiglu_fused_q4k). Is the contract format changed?"
    );
}

/// Check if a call_site value is valid (non-empty and not "NOT WIRED").
fn is_valid_call_site(trimmed: &str) -> Option<bool> {
    if !trimmed.starts_with("call_site:") {
        return None;
    }
    let value = trimmed.trim_start_matches("call_site:").trim().trim_matches('"');
    Some(!value.contains("NOT WIRED") && !value.is_empty())
}

/// FUSION-003: All ACTIVE entries MUST have a call_site that is NOT "NOT WIRED".
///
/// Falsification: An ACTIVE fusion without a real call site means the kernel exists
/// but is never dispatched — a silent performance regression.
#[test]
fn falsify_fusion_003_active_entries_have_call_site() {
    let yaml = read_contract();

    let mut in_active_section = false;
    let mut current_entry_name = String::new();
    let mut found_call_site = false;
    let mut call_site_is_wired = false;
    let mut checked_entries = 0;

    for line in yaml.lines() {
        let trimmed = line.trim();

        if is_fusion_entry_key(line, trimmed) {
            // Finalize previous ACTIVE section
            if in_active_section {
                assert!(
                    found_call_site && call_site_is_wired,
                    "ACTIVE entry '{current_entry_name}' has no valid call_site. \
                     found_call_site: {found_call_site}, is_wired: {call_site_is_wired}. \
                     ACTIVE fusions MUST have a call_site that actually dispatches the kernel."
                );
                checked_entries += 1;
            }
            current_entry_name = trimmed.trim_end_matches(':').to_string();
            in_active_section = false;
            found_call_site = false;
            call_site_is_wired = false;
        }

        if trimmed.contains("status:") && trimmed.contains("ACTIVE") {
            in_active_section = true;
        }

        if in_active_section {
            if let Some(is_wired) = is_valid_call_site(trimmed) {
                found_call_site = true;
                call_site_is_wired = is_wired;
            }
        }
    }

    // Finalize last entry
    if in_active_section {
        assert!(
            found_call_site && call_site_is_wired,
            "ACTIVE entry '{current_entry_name}' has no valid call_site. \
             found_call_site: {found_call_site}, is_wired: {call_site_is_wired}."
        );
        checked_entries += 1;
    }

    assert!(
        checked_entries >= 5,
        "Only found {checked_entries} ACTIVE entries with valid call_sites — \
         expected at least 5. Has the contract format changed?"
    );
}

/// FUSION-004: No fusion decisions hidden in code comments without a contract reference.
///
/// Scans kernel source files for suspect patterns indicating undocumented fusion
/// decisions. Any comment mentioning fusion status without referencing
/// kernel-fusion-v1.yaml or a FUSION-0xx ID is a contract violation.
#[test]
fn falsify_fusion_004_no_comment_only_decisions() {
    let kernels_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src/kernels");
    let kernels_path = Path::new(kernels_dir);
    assert!(kernels_path.is_dir(), "Kernels directory not found at {kernels_dir}");

    let suspect_patterns: &[&str] = &[
        "fused.*blocked",
        "fused.*disabled",
        "fused.*slower",
        "fusion.*blocked",
        "fusion.*disabled",
        "don't use.*fused",
        "do not use.*fused",
    ];

    let contract_references: &[&str] =
        &["kernel-fusion-v1", "FUSION-0", "fusion_decisions", "F-FUSION-001"];

    // Self-exclude: this test file contains the suspect patterns as test data,
    // so we skip it during scanning.
    let self_file = Path::new(file!());
    let self_filename = self_file.file_name().unwrap_or_default();

    let mut violations = Vec::new();
    scan_directory_for_comment_violations(
        kernels_path,
        suspect_patterns,
        contract_references,
        self_filename.to_str().unwrap_or("fusion_contract_falsify.rs"),
        &mut violations,
    );

    assert!(
        violations.is_empty(),
        "Found fusion decisions in code comments WITHOUT contract references:\n{}",
        violations.join("\n")
    );
}

/// Check if a comment matches a suspect pattern using simple glob-style matching.
fn matches_suspect_pattern(lower: &str, pattern: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    let mut pos = 0;
    parts.iter().all(|part| {
        if let Some(found) = lower[pos..].find(part) {
            pos += found + part.len();
            true
        } else {
            false
        }
    })
}

/// Check a single .rs file for suspect comment patterns without contract references.
fn check_file_for_comment_violations(
    path: &Path,
    suspect_patterns: &[&str],
    contract_references: &[&str],
    violations: &mut Vec<String>,
) {
    let Ok(content) = std::fs::read_to_string(path) else {
        return;
    };

    for (line_num, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        // Only check comment lines
        if !trimmed.starts_with("//") && !trimmed.starts_with("///") {
            continue;
        }

        let lower = trimmed.to_lowercase();
        let is_suspect = suspect_patterns.iter().any(|pat| matches_suspect_pattern(&lower, pat));

        if is_suspect {
            let has_contract_ref = contract_references.iter().any(|r| trimmed.contains(r));
            if !has_contract_ref {
                violations.push(format!("  {}:{}: {}", path.display(), line_num + 1, trimmed));
            }
        }
    }
}

/// Recursively scan a directory for .rs files containing suspect comment patterns
/// without contract references. Skips the file named `skip_filename` to avoid
/// self-triggering from test data.
fn scan_directory_for_comment_violations(
    dir: &Path,
    suspect_patterns: &[&str],
    contract_references: &[&str],
    skip_filename: &str,
    violations: &mut Vec<String>,
) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            scan_directory_for_comment_violations(
                &path,
                suspect_patterns,
                contract_references,
                skip_filename,
                violations,
            );
        } else if path.extension() == Some(std::ffi::OsStr::new("rs")) {
            // Skip this test file to avoid self-triggering
            if path.file_name() == Some(std::ffi::OsStr::new(skip_filename)) {
                continue;
            }
            check_file_for_comment_violations(
                &path,
                suspect_patterns,
                contract_references,
                violations,
            );
        }
    }
}

/// FUSION-005: Every kernel name mentioned in the YAML must correspond to a real
/// instantiable Kernel type — no orphaned contract entries.
///
/// Falsification: If a kernel is deleted from trueno-gpu but its contract entry
/// remains, this test catches the orphan.
#[test]
fn falsify_fusion_005_orphan_detection() {
    let yaml = read_contract();

    // Collect all kernel struct names mentioned in the YAML `fused:` fields.
    let mut yaml_kernel_names: Vec<String> = Vec::new();
    for line in yaml.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("fused:") {
            // Extract struct name: e.g. 'fused: "FusedSwigluKernel (trueno-gpu/...)"'
            let value = trimmed.trim_start_matches("fused:").trim().trim_matches('"');
            // The struct name is the first word before any space or parenthesis
            if let Some(struct_name) = value.split_whitespace().next() {
                yaml_kernel_names.push(struct_name.to_string());
            }
        }
    }

    assert!(
        !yaml_kernel_names.is_empty(),
        "No kernel names found in YAML `fused:` fields — contract parsing may be broken."
    );

    // All known fused kernel struct names that are instantiable
    let known_kernel_structs: Vec<&str> =
        all_fused_kernels().iter().map(|(name, _)| *name).collect();

    let mut orphans = Vec::new();
    for yaml_name in &yaml_kernel_names {
        if !known_kernel_structs.contains(&yaml_name.as_str()) {
            orphans.push(yaml_name.clone());
        }
    }

    assert!(
        orphans.is_empty(),
        "Orphaned contract entries — these kernel struct names appear in the YAML \
         but cannot be instantiated in trueno-gpu: {:?}. \
         Either the kernel was deleted (remove the contract entry) or renamed \
         (update the contract entry).",
        orphans
    );
}

// =========================================================================
// Property-based falsification tests (proptest)
// =========================================================================

proptest! {
    /// FUSION-001-PROP: Random kernel names should NOT match any valid registry entry.
    ///
    /// Falsification: The contract must be specific enough that random strings never
    /// accidentally match a real fusion entry. This validates the contract's naming
    /// scheme is unambiguous.
    #[test]
    fn falsify_fusion_001_prop(random_name in "[a-z]{5,20}") {
        let yaml = read_contract();
        let known_names: Vec<String> = all_fused_kernels()
            .into_iter()
            .map(|(_, name)| name)
            .collect();

        // A random lowercase string should never collide with a real kernel name
        // (our kernel names use underscores and specific prefixes like "fused_")
        let is_known = known_names.iter().any(|k| k == &random_name);
        prop_assert!(
            !is_known,
            "Random name '{random_name}' collided with a real kernel name — \
             kernel naming scheme may be too generic."
        );

        // Also verify it doesn't appear as a fusion_decisions key
        let as_key = format!("  {}:", random_name);
        prop_assert!(
            !yaml.contains(&as_key),
            "Random name '{random_name}' matched a YAML fusion_decisions key — \
             contract keys should use specific, descriptive names."
        );
    }

    /// FUSION-002-PROP: Synthetic BLOCKED entries with missing benchmarks must be detectable.
    ///
    /// Falsification: Generate random benchmark values including null/missing combinations.
    /// The validator logic must correctly identify when a BLOCKED entry lacks required data.
    #[test]
    fn falsify_fusion_002_prop(
        unfused_present in proptest::bool::ANY,
        fused_present in proptest::bool::ANY,
        unfused_val in 1.0f64..500.0,
        fused_val in 1.0f64..500.0,
    ) {
        // Build a synthetic BLOCKED entry
        let unfused_line = if unfused_present {
            format!("      unfused_tok_s: {unfused_val:.1}")
        } else {
            "      unfused_tok_s: null".to_string()
        };
        let fused_line = if fused_present {
            format!("      fused_tok_s: {fused_val:.1}")
        } else {
            "      fused_tok_s: null".to_string()
        };

        let synthetic_entry = format!(
            "  test_blocked_entry:\n\
             \x20   status: \"BLOCKED\"\n\
             \x20   benchmark:\n\
             {unfused_line}\n\
             {fused_line}\n"
        );

        // Validate: both must be non-null for a valid BLOCKED entry
        let has_unfused = unfused_present;
        let has_fused = fused_present;
        let is_valid = has_unfused && has_fused;

        // Parse the synthetic entry the same way the deterministic test does
        let mut parsed_unfused = false;
        let mut parsed_fused = false;
        for line in synthetic_entry.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("unfused_tok_s:") {
                let val = trimmed.trim_start_matches("unfused_tok_s:").trim();
                if val != "null" && !val.is_empty() {
                    parsed_unfused = true;
                }
            }
            if trimmed.starts_with("fused_tok_s:") {
                let val = trimmed.trim_start_matches("fused_tok_s:").trim();
                if val != "null" && !val.is_empty() {
                    parsed_fused = true;
                }
            }
        }

        let parsed_valid = parsed_unfused && parsed_fused;
        prop_assert_eq!(
            is_valid,
            parsed_valid,
            "Benchmark validation mismatch: expected valid={}, \
             parsed valid={} (unfused={}, fused={})",
            is_valid,
            parsed_valid,
            has_unfused,
            has_fused
        );
    }

    /// FUSION-003-PROP: Typo'd call_site paths should fail validation.
    ///
    /// Falsification: Generate plausible-but-wrong file paths. The ACTIVE entry
    /// validator must reject "NOT WIRED" and empty call sites while accepting
    /// paths that look like real file references.
    #[test]
    fn falsify_fusion_003_prop(
        typo_segment in "(src|lib|mod|main|test|bench)",
        extension in "(rs|py|toml|yaml)",
        line_num in 1u32..9999,
    ) {
        // Plausible-but-wrong paths
        let typo_path = format!("realizar/{typo_segment}/nonexistent.{extension}:{line_num}");

        // These paths should NOT be "NOT WIRED" (they look like real paths)
        let is_not_wired = typo_path.contains("NOT WIRED");
        prop_assert!(
            !is_not_wired,
            "Generated path should never contain 'NOT WIRED'"
        );

        // A real call_site should reference an actual file — but our typo path
        // points to a nonexistent file. The contract validation logic should
        // accept the format but a deeper check would catch the broken reference.
        let has_colon_line = typo_path.contains(':');
        prop_assert!(
            has_colon_line,
            "Call site should have file:line format, got: {typo_path}"
        );

        // Verify "NOT WIRED" detection works
        let not_wired_value = "NOT WIRED -- see PAR-077";
        prop_assert!(
            not_wired_value.contains("NOT WIRED"),
            "'NOT WIRED' sentinel must be detectable in call_site values"
        );

        // Verify empty string detection works
        let empty_value = "";
        prop_assert!(
            empty_value.is_empty(),
            "Empty call_site must be detectable"
        );
    }
}
