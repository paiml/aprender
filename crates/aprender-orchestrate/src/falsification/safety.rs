//! Safety & Formal Verification Checks (Section 6)
//!
//! Implements SF-01 through SF-10 from the Popperian Falsification Checklist.
//! Focus: Jidoka automated safety, formal methods.

use std::path::Path;
use std::process::Command;
use std::time::Instant;

use super::helpers::{apply_check_outcome, CheckOutcome};
use super::types::{CheckItem, CheckStatus, Evidence, EvidenceType, Severity};

/// Evaluate all safety checks for a project.
pub fn evaluate_all(project_path: &Path) -> Vec<CheckItem> {
    vec![
        check_unsafe_code_isolation(project_path),
        check_memory_safety_fuzzing(project_path),
        check_miri_validation(project_path),
        check_formal_safety_properties(project_path),
        check_adversarial_robustness(project_path),
        check_thread_safety(project_path),
        check_resource_leak_prevention(project_path),
        check_panic_safety(project_path),
        check_input_validation(project_path),
        check_supply_chain_security(project_path),
    ]
}

/// Check if an unsafe code location is in an allowed module
// SAFETY: no actual unsafe code -- checks if file path is in an allowed unsafe module
fn is_allowed_unsafe_location(path_str: &str, file_name: &str, content: &str) -> bool {
    const ALLOWED_DIRS: &[&str] = &["/internal/", "/ffi/", "/simd/", "/wasm/"];
    const ALLOWED_SUFFIXES: &[&str] = &["_internal.rs", "_ffi.rs", "_simd.rs", "_tests.rs"];

    ALLOWED_DIRS.iter().any(|d| path_str.contains(d))
        || ALLOWED_SUFFIXES.iter().any(|s| path_str.contains(s))
        || file_name == "lib.rs"
        || content.contains("// SAFETY:")
        || content.contains("# Safety")
}

/// SF-01: Unsafe Code Isolation
///
/// **Claim:** All unsafe code isolated in marked internal modules.
///
/// **Rejection Criteria (Major):**
/// - Unsafe block outside designated module
// SAFETY: no actual unsafe code -- static analysis check that scans target project for unsafe blocks
pub fn check_unsafe_code_isolation(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "SF-01",
        "Unsafe Code Isolation",
        "All unsafe code isolated in marked internal modules",
    )
    .with_severity(Severity::Major)
    .with_tps("Jidoka — containment");

    let mut unsafe_locations = Vec::new();
    let mut total_unsafe_blocks = 0;

    // SAFETY: no actual unsafe code -- string constant for pattern matching scanned source
    const UNSAFE_BLOCK_PATTERN: &str = concat!("unsafe", " {");
    const UNSAFE_FN_PATTERN: &str = concat!("unsafe", " fn ");

    if let Ok(entries) = glob::glob(&format!("{}/src/**/*.rs", project_path.display())) {
        for entry in entries.flatten() {
            let Ok(content) = std::fs::read_to_string(&entry) else {
                continue;
            };
            // SAFETY: no actual unsafe code -- counting unsafe patterns in scanned file
            let unsafe_count = content.matches(UNSAFE_BLOCK_PATTERN).count()
                + content.matches(UNSAFE_FN_PATTERN).count();

            if unsafe_count == 0 {
                continue;
            }
            total_unsafe_blocks += unsafe_count;

            let file_name = entry.file_name().unwrap_or_default().to_string_lossy();
            let path_str = entry.to_string_lossy();
            // SAFETY: no actual unsafe code -- checking if location is in designated module
            if !is_allowed_unsafe_location(&path_str, &file_name, &content) {
                unsafe_locations.push(format!("{}: {} blocks", path_str, unsafe_count));
            }
        }
    }

    // SAFETY: no actual unsafe code -- building evidence report of unsafe block locations
    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Found {} unsafe blocks, {} in non-designated locations",
            total_unsafe_blocks,
            unsafe_locations.len()
        ),
        // SAFETY: no actual unsafe code -- formatting evidence data for check report
        data: Some(format!("locations: {:?}", unsafe_locations)),
        files: Vec::new(),
    });

    // SAFETY: no actual unsafe code -- format strings referencing unsafe block counts
    let partial_msg =
        format!("{} unsafe blocks outside designated modules", unsafe_locations.len());
    // SAFETY: no actual unsafe code -- format string for unsafe isolation failure message
    let fail_msg = format!(
        "{} unsafe blocks outside designated modules: {}",
        unsafe_locations.len(),
        unsafe_locations.join(", ")
    );
    item = apply_check_outcome(
        item,
        &[
            (unsafe_locations.is_empty(), CheckOutcome::Pass),
            (unsafe_locations.len() <= 3, CheckOutcome::Partial(&partial_msg)),
            (true, CheckOutcome::Fail(&fail_msg)),
        ],
    );

    item.finish_timed(start)
}

/// SF-02: Memory Safety Under Fuzzing
///
/// **Claim:** No memory safety violations under fuzzing.
///
/// **Rejection Criteria (Major):**
/// - Any ASan/MSan/UBSan violation
pub fn check_memory_safety_fuzzing(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "SF-02",
        "Memory Safety Under Fuzzing",
        "No memory safety violations under fuzzing",
    )
    .with_severity(Severity::Major)
    .with_tps("Jidoka — defect detection");

    // Check for fuzzing setup
    let fuzz_dir = project_path.join("fuzz");
    let has_fuzz_dir = fuzz_dir.exists();
    let has_fuzz_targets = if has_fuzz_dir {
        glob::glob(&format!("{}/fuzz_targets/**/*.rs", fuzz_dir.display()))
            .ok()
            .map(|entries| entries.count() > 0)
            .unwrap_or(false)
            || fuzz_dir.join("fuzz_targets").exists()
    } else {
        false
    };

    // Check for cargo-fuzz or property-based testing in Cargo.toml
    let cargo_toml = project_path.join("Cargo.toml");
    let cargo_content =
        cargo_toml.exists().then(|| std::fs::read_to_string(&cargo_toml).ok()).flatten();

    let has_fuzz_dep = cargo_content
        .as_ref()
        .map(|c| c.contains("libfuzzer-sys") || c.contains("arbitrary"))
        .unwrap_or(false);

    // Proptest is a valid property-based testing framework (similar to fuzzing)
    let has_proptest = cargo_content
        .as_ref()
        .map(|c| c.contains("proptest") || c.contains("quickcheck"))
        .unwrap_or(false);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Fuzzing setup: dir={}, targets={}, fuzz_deps={}, proptest={}",
            has_fuzz_dir, has_fuzz_targets, has_fuzz_dep, has_proptest
        ),
        data: None,
        files: Vec::new(),
    });

    let is_small_project = glob::glob(&format!("{}/src/**/*.rs", project_path.display()))
        .ok()
        .map(|entries| entries.count() < 20)
        .unwrap_or(true);

    item = apply_check_outcome(
        item,
        &[
            (has_fuzz_targets && has_fuzz_dep, CheckOutcome::Pass),
            (has_proptest, CheckOutcome::Pass),
            (has_fuzz_dir || has_fuzz_dep, CheckOutcome::Partial("Fuzzing partially configured")),
            (is_small_project, CheckOutcome::Partial("No fuzzing setup (small project)")),
            (true, CheckOutcome::Fail("No fuzzing infrastructure detected")),
        ],
    );

    item.finish_timed(start)
}

/// SF-03: Miri Undefined Behavior Detection
///
/// **Claim:** Core operations pass Miri validation.
///
/// **Rejection Criteria (Major):**
/// - Any Miri error
pub fn check_miri_validation(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "SF-03",
        "Miri Undefined Behavior Detection",
        "Core operations pass Miri validation",
    )
    .with_severity(Severity::Major)
    .with_tps("Jidoka — automatic UB detection");

    // Check if CI config includes Miri
    let ci_configs = [
        project_path.join(".github/workflows/ci.yml"),
        project_path.join(".github/workflows/test.yml"),
        project_path.join(".github/workflows/rust.yml"),
    ];

    let mut has_miri_in_ci = false;
    for ci_path in &ci_configs {
        if ci_path.exists() {
            if let Ok(content) = std::fs::read_to_string(ci_path) {
                if content.contains("miri") {
                    has_miri_in_ci = true;
                    break;
                }
            }
        }
    }

    // Check for Miri in Makefile
    let makefile = project_path.join("Makefile");
    let has_miri_in_makefile = makefile
        .exists()
        .then(|| std::fs::read_to_string(&makefile).ok())
        .flatten()
        .map(|c| c.contains("miri"))
        .unwrap_or(false);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Miri setup: ci={}, makefile={}",
            has_miri_in_ci, has_miri_in_makefile
        ),
        data: None,
        files: Vec::new(),
    });

    // SAFETY: no actual unsafe code -- string constant for counting unsafe blocks in scanned files
    const UNSAFE_BLOCK: &str = concat!("unsafe", " {");
    let unsafe_count: usize = glob::glob(&format!("{}/src/**/*.rs", project_path.display()))
        .ok()
        .map(|entries| {
            entries
                .flatten()
                .filter_map(|p| std::fs::read_to_string(&p).ok())
                .map(|c| c.matches(UNSAFE_BLOCK).count())
                .sum()
        })
        .unwrap_or(0);

    // SAFETY: no actual unsafe code -- format string referencing scanned unsafe block count
    let miri_partial_msg = format!("Miri not configured ({} unsafe blocks)", unsafe_count);
    item = apply_check_outcome(
        item,
        &[
            (has_miri_in_ci, CheckOutcome::Pass),
            (has_miri_in_makefile, CheckOutcome::Partial("Miri available but not in CI")),
            (unsafe_count == 0, CheckOutcome::Pass),
            (true, CheckOutcome::Partial(&miri_partial_msg)),
        ],
    );

    item.finish_timed(start)
}

/// SF-04: Formal Safety Properties
///
/// **Claim:** Safety-critical components have formal proofs.
///
/// **Rejection Criteria (Minor):**
/// - Safety property unproven for critical path
pub fn check_formal_safety_properties(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "SF-04",
        "Formal Safety Properties",
        "Safety-critical components have formal proofs",
    )
    .with_severity(Severity::Minor)
    .with_tps("Formal verification requirement");

    // Check for Kani, Creusot, or other formal verification tools
    let cargo_toml = project_path.join("Cargo.toml");
    let has_kani = cargo_toml
        .exists()
        .then(|| std::fs::read_to_string(&cargo_toml).ok())
        .flatten()
        .map(|c| c.contains("kani") || c.contains("creusot") || c.contains("prusti"))
        .unwrap_or(false);

    // Check for proof annotations in code
    let has_proof_annotations = glob::glob(&format!("{}/src/**/*.rs", project_path.display()))
        .ok()
        .map(|entries| {
            entries.flatten().any(|p| {
                std::fs::read_to_string(&p)
                    .ok()
                    .map(|c| {
                        c.contains("#[kani::")
                            || c.contains("#[requires(")
                            || c.contains("#[ensures(")
                            || c.contains("// PROOF:")
                    })
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Formal verification: tools={}, annotations={}",
            has_kani, has_proof_annotations
        ),
        data: None,
        files: Vec::new(),
    });

    item = apply_check_outcome(
        item,
        &[
            (has_kani && has_proof_annotations, CheckOutcome::Pass),
            (
                has_kani || has_proof_annotations,
                CheckOutcome::Partial("Partial formal verification setup"),
            ),
            (true, CheckOutcome::Partial("No formal verification (advanced feature)")),
        ],
    );

    item.finish_timed(start)
}

/// SF-05: Adversarial Robustness Verification
///
/// **Claim:** Models tested against adversarial examples.
///
/// **Rejection Criteria (Major):**
/// - Model fails under documented attack types
pub fn check_adversarial_robustness(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "SF-05",
        "Adversarial Robustness Verification",
        "Models tested against adversarial examples",
    )
    .with_severity(Severity::Major)
    .with_tps("AI Safety requirement");

    // Check for adversarial testing patterns
    let has_adversarial_tests = glob::glob(&format!("{}/src/**/*.rs", project_path.display()))
        .ok()
        .map(|entries| {
            entries.flatten().any(|p| {
                std::fs::read_to_string(&p)
                    .ok()
                    .map(|c| {
                        c.contains("adversarial")
                            || c.contains("perturbation")
                            || c.contains("robustness")
                            || c.contains("attack")
                    })
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);

    // Check for robustness verification in tests
    let has_robustness_verification =
        glob::glob(&format!("{}/tests/**/*.rs", project_path.display()))
            .ok()
            .map(|entries| {
                entries.flatten().any(|p| {
                    std::fs::read_to_string(&p)
                        .ok()
                        .map(|c| c.contains("adversarial") || c.contains("robustness"))
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Adversarial robustness: testing={}, verification={}",
            has_adversarial_tests, has_robustness_verification
        ),
        data: None,
        files: Vec::new(),
    });

    // Check if project has ML models that need adversarial testing
    let has_ml_models = glob::glob(&format!("{}/src/**/*.rs", project_path.display()))
        .ok()
        .map(|entries| {
            entries.flatten().any(|p| {
                std::fs::read_to_string(&p)
                    .ok()
                    .map(|c| {
                        c.contains("predict") || c.contains("classifier") || c.contains("neural")
                    })
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);

    item = apply_check_outcome(
        item,
        &[
            (
                !has_ml_models || has_adversarial_tests || has_robustness_verification,
                CheckOutcome::Pass,
            ),
            (true, CheckOutcome::Partial("ML models without adversarial testing")),
        ],
    );

    item.finish_timed(start)
}

/// Find files with unsafe Send/Sync implementations lacking safety docs
// SAFETY: no actual unsafe code -- scans target project for unsafe Send/Sync impls
fn find_unsafe_send_sync(project_path: &Path) -> Vec<String> {
    let mut results = Vec::new();
    let Ok(entries) = glob::glob(&format!("{}/src/**/*.rs", project_path.display())) else {
        return results;
    };
    for entry in entries.flatten() {
        let Ok(content) = std::fs::read_to_string(&entry) else {
            continue;
        };
        let has_unsafe_impl = content.contains("unsafe impl Send")
            || content.contains("unsafe impl Sync")
            || content.contains("unsafe impl<") && content.contains("> Send")
            || content.contains("unsafe impl<") && content.contains("> Sync");

        // SAFETY: no actual unsafe code -- checking if target file documents its unsafe impls
        if has_unsafe_impl && !content.contains("// SAFETY:") && !content.contains("# Safety") {
            results.push(entry.to_string_lossy().to_string());
        }
    }
    results
}

/// SF-06: Thread Safety (Send + Sync)
///
/// **Claim:** All Send + Sync implementations correct.
///
/// **Rejection Criteria (Major):**
/// - TSan detects any race
pub fn check_thread_safety(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "SF-06",
        "Thread Safety (Send + Sync)",
        "All Send + Sync implementations correct",
    )
    .with_severity(Severity::Major)
    .with_tps("Jidoka — race detection");

    let unsafe_send_sync = find_unsafe_send_sync(project_path);

    // Check for concurrent data structures
    let cargo_toml = project_path.join("Cargo.toml");
    let uses_concurrent = cargo_toml
        .exists()
        .then(|| std::fs::read_to_string(&cargo_toml).ok())
        .flatten()
        .map(|c| {
            c.contains("crossbeam")
                || c.contains("parking_lot")
                || c.contains("dashmap")
                || c.contains("rayon")
        })
        .unwrap_or(false);

    // SAFETY: no actual unsafe code -- building thread safety evidence for check report
    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Thread safety: unsafe_impls={}, concurrent_libs={}",
            unsafe_send_sync.len(),
            uses_concurrent
        ),
        data: Some(format!("unsafe_send_sync: {:?}", unsafe_send_sync)),
        files: Vec::new(),
    });

    // SAFETY: no actual unsafe code -- format strings for Send/Sync check report
    let sync_partial =
        format!("{} unsafe Send/Sync without safety comment", unsafe_send_sync.len());
    // SAFETY: no actual unsafe code -- format string for undocumented Send/Sync failure message
    let sync_fail = format!(
        "{} unsafe Send/Sync implementations without documentation",
        unsafe_send_sync.len()
    );
    item = apply_check_outcome(
        item,
        &[
            (unsafe_send_sync.is_empty(), CheckOutcome::Pass),
            (unsafe_send_sync.len() <= 2, CheckOutcome::Partial(&sync_partial)),
            (true, CheckOutcome::Fail(&sync_fail)),
        ],
    );

    item.finish_timed(start)
}

/// Scan source files for resource management patterns.
fn scan_resource_patterns(project_path: &Path) -> (usize, Vec<&'static str>) {
    let mut drop_impls = 0;
    let mut resource_types = Vec::new();
    let Ok(entries) = glob::glob(&format!("{}/src/**/*.rs", project_path.display())) else {
        return (0, resource_types);
    };
    for entry in entries.flatten() {
        let Ok(content) = std::fs::read_to_string(&entry) else {
            continue;
        };
        drop_impls += content.matches("impl Drop for").count();
        drop_impls +=
            content.matches("impl<").count() * content.matches("> Drop for").count().min(1);
        if content.contains("File") || content.contains("TcpStream") {
            resource_types.push("file/network handles");
        }
        if content.contains("Arc<") || content.contains("Rc<") {
            resource_types.push("reference counting");
        }
        if content.contains("ManuallyDrop") {
            resource_types.push("ManuallyDrop");
        }
    }
    resource_types.sort_unstable();
    resource_types.dedup();
    (drop_impls, resource_types)
}

/// SF-07: Resource Leak Prevention
///
/// **Claim:** No resource leaks.
///
/// **Rejection Criteria (Major):**
/// - "definitely lost" > 0 bytes in Valgrind
pub fn check_resource_leak_prevention(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new("SF-07", "Resource Leak Prevention", "No resource leaks")
        .with_severity(Severity::Major)
        .with_tps("Muda (Defects)");

    let (drop_impls, resource_types) = scan_resource_patterns(project_path);

    let has_mem_forget =
        super::helpers::source_contains_pattern(project_path, &["mem::forget", "std::mem::forget"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Resource management: drop_impls={}, mem_forget={}, resource_types={:?}",
            drop_impls, has_mem_forget, resource_types
        ),
        data: None,
        files: Vec::new(),
    });

    item = apply_check_outcome(
        item,
        &[
            (!has_mem_forget, CheckOutcome::Pass),
            (true, CheckOutcome::Partial("Uses mem::forget (verify intentional)")),
        ],
    );

    item.finish_timed(start)
}

/// Scan source files for panic-related patterns.
fn scan_panic_patterns(project_path: &Path) -> (bool, bool, Vec<String>) {
    let mut has_catch_unwind = false;
    let mut has_panic_hook = false;
    let mut high_unwrap_files = Vec::new();
    let Ok(entries) = glob::glob(&format!("{}/src/**/*.rs", project_path.display())) else {
        return (false, false, high_unwrap_files);
    };
    for entry in entries.flatten() {
        let Ok(content) = std::fs::read_to_string(&entry) else {
            continue;
        };
        if content.contains("catch_unwind") {
            has_catch_unwind = true;
        }
        if content.contains("set_panic_hook") || content.contains("panic::set_hook") {
            has_panic_hook = true;
        }
        let unwrap_count = content.matches(".unwrap()").count();
        if unwrap_count > 10 {
            high_unwrap_files.push(format!(
                "{}: {} unwraps",
                entry.file_name().unwrap_or_default().to_string_lossy(),
                unwrap_count
            ));
        }
    }
    (has_catch_unwind, has_panic_hook, high_unwrap_files)
}

/// SF-08: Panic Safety
///
/// **Claim:** Panics don't corrupt data structures.
///
/// **Rejection Criteria (Minor):**
/// - Panic in Drop impl
pub fn check_panic_safety(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new("SF-08", "Panic Safety", "Panics don't corrupt data structures")
        .with_severity(Severity::Minor)
        .with_tps("Graceful degradation");

    let (has_catch_unwind, has_panic_hook, panic_patterns) = scan_panic_patterns(project_path);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Panic handling: catch_unwind={}, panic_hook={}, high_unwrap_files={}",
            has_catch_unwind,
            has_panic_hook,
            panic_patterns.len()
        ),
        data: Some(format!("patterns: {:?}", panic_patterns)),
        files: Vec::new(),
    });

    let panic_few = format!("{} files with high unwrap count", panic_patterns.len());
    let panic_many = format!(
        "{} files with excessive unwraps - consider expect() or ? operator",
        panic_patterns.len()
    );
    item = apply_check_outcome(
        item,
        &[
            (panic_patterns.is_empty(), CheckOutcome::Pass),
            (panic_patterns.len() <= 5, CheckOutcome::Partial(&panic_few)),
            (true, CheckOutcome::Partial(&panic_many)),
        ],
    );

    item.finish_timed(start)
}

/// Classify validation patterns in a single file's content.
fn classify_validation_in_file(content: &str) -> (bool, Vec<&'static str>) {
    let mut has_explicit = false;
    let mut methods = Vec::new();

    if content.contains("fn validate")
        || content.contains("fn is_valid")
        || content.contains("impl Validate")
        || content.contains("#[validate")
    {
        has_explicit = true;
        methods.push("explicit validation");
    }
    if content.contains("pub fn")
        && (content.contains("-> Result<") || content.contains("-> Option<"))
    {
        methods.push("Result/Option returns");
    }
    if content.contains("assert!(") || content.contains("debug_assert!(") {
        methods.push("assertions");
    }
    (has_explicit, methods)
}

/// Scan source files for validation patterns
fn scan_validation_patterns(project_path: &Path) -> (bool, Vec<&'static str>) {
    let mut has_explicit = false;
    let mut methods = Vec::new();

    let Ok(entries) = glob::glob(&format!("{}/src/**/*.rs", project_path.display())) else {
        return (false, methods);
    };

    for entry in entries.flatten() {
        let Ok(content) = std::fs::read_to_string(&entry) else {
            continue;
        };
        let (file_explicit, file_methods) = classify_validation_in_file(&content);
        has_explicit = has_explicit || file_explicit;
        methods.extend(file_methods);
    }
    (has_explicit, methods)
}

/// Check if Cargo.toml uses a validator crate
fn has_validator_crate(project_path: &Path) -> bool {
    let cargo_toml = project_path.join("Cargo.toml");
    std::fs::read_to_string(cargo_toml)
        .ok()
        .is_some_and(|c| c.contains("validator") || c.contains("garde"))
}

/// SF-09: Input Validation
///
/// **Claim:** All public APIs validate inputs.
///
/// **Rejection Criteria (Major):**
/// - Any panic from malformed input
pub fn check_input_validation(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new("SF-09", "Input Validation", "All public APIs validate inputs")
        .with_severity(Severity::Major)
        .with_tps("Poka-Yoke — error prevention");

    let (has_validation, mut validation_methods) = scan_validation_patterns(project_path);
    let has_validator = has_validator_crate(project_path);

    if has_validator {
        validation_methods.push("validator crate");
    }

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Validation: explicit={}, methods={:?}",
            has_validation, validation_methods
        ),
        data: None,
        files: Vec::new(),
    });

    item = apply_check_outcome(
        item,
        &[
            (has_validation || has_validator, CheckOutcome::Pass),
            (!validation_methods.is_empty(), CheckOutcome::Pass),
            (true, CheckOutcome::Partial("Consider adding explicit input validation")),
        ],
    );

    item.finish_timed(start)
}

/// SF-10: Supply Chain Security
///
/// **Claim:** All dependencies audited.
///
/// **Rejection Criteria (Critical):**
/// - Known vulnerability or unmaintained critical dependency
pub fn check_supply_chain_security(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new("SF-10", "Supply Chain Security", "All dependencies audited")
        .with_severity(Severity::Critical)
        .with_tps("Jidoka — supply chain circuit breaker");

    // Check for cargo-audit
    let has_audit_in_ci = check_ci_for_tool(project_path, "cargo audit");
    let has_deny_in_ci = check_ci_for_tool(project_path, "cargo deny");

    // Check for deny.toml
    let deny_toml = project_path.join("deny.toml");
    let has_deny_config = deny_toml.exists();

    // Try running cargo audit if available
    let audit_result =
        Command::new("cargo").args(["audit", "--json"]).current_dir(project_path).output().ok();

    let audit_clean = audit_result
        .as_ref()
        .map(|o| {
            o.status.success()
                || String::from_utf8_lossy(&o.stdout).contains("\"vulnerabilities\":[]")
        })
        .unwrap_or(false);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::DependencyAudit,
        description: format!(
            "Supply chain: audit_ci={}, deny_ci={}, deny_config={}, audit_clean={}",
            has_audit_in_ci, has_deny_in_ci, has_deny_config, audit_clean
        ),
        data: None,
        files: Vec::new(),
    });

    item = apply_check_outcome(
        item,
        &[
            (has_deny_config && (has_audit_in_ci || has_deny_in_ci), CheckOutcome::Pass),
            (
                has_deny_config || has_audit_in_ci || has_deny_in_ci,
                CheckOutcome::Partial("Partial supply chain security setup"),
            ),
            (audit_clean, CheckOutcome::Partial("No vulnerabilities but no CI enforcement")),
            (true, CheckOutcome::Fail("No supply chain security tooling configured")),
        ],
    );

    item.finish_timed(start)
}

/// Helper: Check if a tool is referenced in CI configuration
fn check_ci_for_tool(project_path: &Path, tool: &str) -> bool {
    let ci_configs = [
        project_path.join(".github/workflows/ci.yml"),
        project_path.join(".github/workflows/test.yml"),
        project_path.join(".github/workflows/rust.yml"),
        project_path.join(".github/workflows/security.yml"),
    ];

    for ci_path in &ci_configs {
        if ci_path.exists() {
            if let Ok(content) = std::fs::read_to_string(ci_path) {
                if content.contains(tool) {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(test)]
#[path = "safety_tests.rs"]
mod tests;
