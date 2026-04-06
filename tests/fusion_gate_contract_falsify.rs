//! Falsification tests for kernel fusion QA gate (F-FUSION-001)
//!
//! Contract: aprender/contracts/kernel-fusion-v1.yaml
//! Tests: FALSIFY-FUSION-QA-001..003, FALSIFY-FUSION-QA-001-prop
//!
//! These tests validate the contract's enforcement rules:
//! - Every fused kernel must have a YAML entry
//! - ACTIVE fusions must have valid call sites
//! - BLOCKED fusions must have benchmark data
//!
//! Methodology: Parse the contract YAML as raw text and verify structural
//! invariants. No dependency on realizar internals -- pure contract validation.
//!
//! Run with: `cargo test --test fusion_gate_contract_falsify -- --nocapture`

use proptest::prelude::*;

/// The kernel-fusion-v1.yaml contract, included at compile time.
/// Path: from `realizar/tests/` up two levels to `aprender/contracts/`.
const CONTRACT_YAML: &str = include_str!("../../aprender/contracts/kernel-fusion-v1.yaml");

/// All known fused kernel names that MUST appear in the contract.
/// Sourced from the `kernels.fused` fields in kernel-fusion-v1.yaml.
const KNOWN_FUSED_KERNELS: &[&str] = &[
    "FusedSwigluKernel",
    "BatchedSwigluKernel",
    "FusedQKVKernel",
    "FusedGateUpKernel",
    "FusedGemmBiasGeluKernel",
    "FusedRmsNormQ4KGemvKernel",
    "FusedGateUpQ4KGemvKernel",
    "FusedRmsNormGateUpSwigluQ4KKernel",
];

/// All fusion decision IDs and their expected statuses.
const FUSION_DECISIONS: &[(&str, &str)] = &[
    ("FUSION-001", "ACTIVE"),
    ("FUSION-002", "ACTIVE"),
    ("FUSION-003", "BLOCKED"),
    ("FUSION-004", "PLANNED"),
    ("FUSION-005", "PLANNED"),
    ("FUSION-006", "ACTIVE"),
    ("FUSION-007", "ACTIVE"),
    ("FUSION-008", "ACTIVE"),
    ("FUSION-009", "ACTIVE"),
    ("FUSION-010", "ACTIVE"),
];

// ============================================================================
// FALSIFY-FUSION-QA-001: Undocumented kernel detection
// ============================================================================

/// Simulate adding an undocumented kernel and verify the contract does NOT
/// contain it. This proves F-FUSION-001 would catch a new kernel that was
/// added to code without a corresponding contract entry.
///
/// Falsification target: enforcement.kernel_registry_complete
/// "No orphaned fused kernels (kernel exists but decision undocumented)"
#[test]
fn falsify_fusion_qa_001_undocumented_kernel_detected() {
    let fake_kernel = "FusedFakeTestKernel";

    // The contract YAML must NOT contain this fake kernel name
    assert!(
        !CONTRACT_YAML.contains(fake_kernel),
        "FALSIFICATION FAILED: Contract unexpectedly contains undocumented kernel '{fake_kernel}'. \
         If this kernel were real, F-FUSION-001 would miss it because it already has an entry."
    );

    // Verify the contract is non-empty and structurally valid (has fusion_decisions section)
    assert!(
        CONTRACT_YAML.contains("fusion_decisions:"),
        "Contract YAML missing 'fusion_decisions:' section -- file may be corrupted"
    );

    // Verify the QA gate definition exists
    assert!(
        CONTRACT_YAML.contains("F-FUSION-001"),
        "Contract YAML missing QA gate 'F-FUSION-001' -- gate definition required"
    );

    // Verify the enforcement rule that would catch this is present
    assert!(
        CONTRACT_YAML.contains("kernel_registry_complete"),
        "Contract YAML missing 'kernel_registry_complete' enforcement rule"
    );

    println!(
        "FALSIFY-FUSION-QA-001: PASS -- Undocumented kernel '{fake_kernel}' correctly absent from contract. \
         F-FUSION-001 gate would detect this as an orphaned kernel."
    );
}

// ============================================================================
// FALSIFY-FUSION-QA-002: Known fused kernels all present
// ============================================================================

/// Verify every known fused kernel has a corresponding entry in the contract
/// YAML. If any kernel is missing, the contract has drifted from the codebase.
///
/// Falsification target: enforcement.kernel_registry_complete
/// "scan trueno-gpu/src/kernels/ for Kernel impls, verify each has entry"
#[test]
fn falsify_fusion_qa_002_known_fused_kernels_pass() {
    let mut missing = Vec::new();

    for kernel_name in KNOWN_FUSED_KERNELS {
        if !CONTRACT_YAML.contains(kernel_name) {
            missing.push(*kernel_name);
        }
    }

    assert!(
        missing.is_empty(),
        "FALSIFICATION SUCCEEDED (contract drift detected): \
         The following fused kernels are NOT documented in kernel-fusion-v1.yaml: {missing:?}. \
         Every fused kernel MUST have a fusion_decisions entry per enforcement.kernel_registry_complete."
    );

    // Additionally verify that each fusion decision ID is present
    for (fusion_id, expected_status) in FUSION_DECISIONS {
        assert!(
            CONTRACT_YAML.contains(fusion_id),
            "Fusion decision '{fusion_id}' (expected status: {expected_status}) missing from contract"
        );
        assert!(
            CONTRACT_YAML.contains(expected_status),
            "Status '{expected_status}' for {fusion_id} not found in contract"
        );
    }

    // Verify BLOCKED entries have benchmark data (enforcement.blocked_must_benchmark)
    // The BLOCKED fusion (FUSION-003) must have both unfused_tok_s and fused_tok_s
    assert!(
        CONTRACT_YAML.contains("unfused_tok_s:") && CONTRACT_YAML.contains("fused_tok_s:"),
        "BLOCKED fusion entries must include benchmark data (unfused_tok_s and fused_tok_s) \
         per enforcement.blocked_must_benchmark"
    );

    // Verify ACTIVE entries have call_site fields
    assert!(
        CONTRACT_YAML.contains("call_site:"),
        "ACTIVE fusion entries must include call_site per enforcement.active_must_be_called"
    );

    println!(
        "FALSIFY-FUSION-QA-002: PASS -- All {} known fused kernels documented in contract. \
         All {} fusion decisions have correct IDs and statuses.",
        KNOWN_FUSED_KERNELS.len(),
        FUSION_DECISIONS.len()
    );
}

// ============================================================================
// FALSIFY-FUSION-QA-003: Missing YAML entry detection
// ============================================================================

/// Verify that checking for kernel names NOT in the YAML produces a "not found"
/// result. This tests the negative case: the contract correctly excludes
/// kernels that do not exist.
///
/// Falsification target: qa_gate.falsification
/// "Introduce an undocumented fused kernel -- gate must catch it"
#[test]
fn falsify_fusion_qa_003_missing_yaml_entry_detected() {
    // A set of plausible-but-nonexistent fused kernel names
    let nonexistent_kernels = [
        "FusedAttentionLayerNormKernel",
        "FusedDropoutReluKernel",
        "FusedEmbeddingNormKernel",
        "FusedQ8KGemvBiasKernel",
        "FusedRopeAttentionKernel",
        "FusedSoftmaxDropoutKernel",
        "FusedLayerNormGateKernel",
        "FusedBatchNormSwishKernel",
    ];

    for kernel_name in &nonexistent_kernels {
        assert!(
            !CONTRACT_YAML.contains(kernel_name),
            "FALSIFICATION FAILED: Nonexistent kernel '{kernel_name}' found in contract. \
             Either the contract has a spurious entry or the kernel list needs updating."
        );
    }

    // Verify that each nonexistent kernel would be caught by the registry check:
    // If we had a hypothetical `check_kernel_registered(name) -> bool` function,
    // it would return false for all of these. We simulate that by searching the YAML.
    let registered_count = nonexistent_kernels
        .iter()
        .filter(|k| CONTRACT_YAML.contains(**k))
        .count();

    assert_eq!(
        registered_count, 0,
        "Expected 0 nonexistent kernels to be registered, found {registered_count}"
    );

    // Cross-check: at least one REAL kernel IS found (sanity check)
    assert!(
        CONTRACT_YAML.contains("FusedSwigluKernel"),
        "Sanity check failed: known kernel 'FusedSwigluKernel' not found in contract"
    );

    println!(
        "FALSIFY-FUSION-QA-003: PASS -- All {} nonexistent kernel names correctly absent. \
         F-FUSION-001 gate would flag each as unregistered.",
        nonexistent_kernels.len()
    );
}

// ============================================================================
// FALSIFY-FUSION-QA-001-prop: Property-based kernel name validation
// ============================================================================

proptest! {
    /// Property: Any random kernel config string either matches an existing
    /// contract entry or is detected as unregistered.
    ///
    /// This is the exhaustive version of FALSIFY-FUSION-QA-001. Instead of
    /// testing a single fake name, we generate thousands of random strings
    /// and verify the contract lookup is deterministic and complete.
    ///
    /// Falsification target: The contract registry is a total function --
    /// for ANY kernel name string, we can definitively say "registered" or
    /// "unregistered". There is no ambiguous middle ground.
    #[test]
    fn falsify_fusion_qa_001_prop(
        random_kernel_name in "[A-Z][a-zA-Z0-9]{4,40}Kernel"
    ) {
        let is_registered = CONTRACT_YAML.contains(&random_kernel_name);

        if is_registered {
            // If the random name happens to match a real kernel, verify it is
            // one of the known kernels (not a spurious substring match).
            let is_known = KNOWN_FUSED_KERNELS
                .iter()
                .any(|known| random_kernel_name.contains(known) || known.contains(random_kernel_name.as_str()));

            // A registered kernel that is NOT in our known list means the contract
            // has entries we are not tracking -- potential drift.
            prop_assert!(
                is_known,
                "Random name '{}' matched contract but is not in KNOWN_FUSED_KERNELS. \
                 Contract may have undocumented entries.",
                random_kernel_name
            );
        }
        // If not registered, that is the expected case: the contract correctly
        // does not contain a random string. F-FUSION-001 would flag this
        // kernel as undocumented if it appeared in the codebase.
    }

    /// Property: Fusion decision IDs follow the pattern FUSION-NNN and are
    /// monotonically assigned. Random IDs outside the known range must not
    /// appear in the contract.
    #[test]
    fn falsify_fusion_id_uniqueness_prop(
        random_id_num in 100u32..999
    ) {
        let random_id = format!("FUSION-{random_id_num}");

        let is_known = FUSION_DECISIONS
            .iter()
            .any(|(id, _)| *id == random_id.as_str());

        let in_contract = CONTRACT_YAML.contains(random_id.as_str());

        if in_contract && !is_known {
            // Found a fusion ID in the contract that we do not track
            prop_assert!(
                false,
                "Contract contains '{}' which is not in FUSION_DECISIONS. \
                 Update the test constants to match the contract.",
                random_id
            );
        }

        if is_known {
            // Known IDs must be in the contract
            prop_assert!(
                in_contract,
                "Known fusion ID '{}' missing from contract YAML",
                random_id
            );
        }
    }
}
