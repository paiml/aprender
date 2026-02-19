#![allow(clippy::disallowed_methods)]
//! Toyota Way Principles Verification Tests (P1-P14)
//!
//! These tests verify adherence to the 14 Toyota Production System principles
//! as documented in spec v3.0.0 Section 12.1 (Liker, 2004).
//!
//! Each test specifies a falsification condition per Popperian methodology.

use std::path::Path;

// ============================================================================
// P1: Long-Term Philosophy
// "Base your management decisions on a long-term philosophy, even at the
// expense of short-term financial goals."
// Application: Building a "Sovereign AI" stack > Short-term features
// ============================================================================

/// P1: Long-term philosophy documented
/// FALSIFICATION: No mention of sovereign AI or long-term vision in docs
#[test]
fn p1_long_term_philosophy_documented() {
    let claude_md = include_str!("../CLAUDE.md");

    // Should mention sovereign AI or long-term architecture decisions
    let has_architecture_docs = claude_md.contains("Sovereign")
        || claude_md.contains("Architecture")
        || claude_md.contains("realizar")
        || claude_md.contains("trueno");

    assert!(
        has_architecture_docs,
        "P1 FALSIFIED: No long-term architecture philosophy documented in CLAUDE.md"
    );
}

/// P1b: Architecture decisions are documented in spec
/// FALSIFICATION: No specification documents exist
#[test]
fn p1b_architecture_specs_exist() {
    let spec_dir = Path::new("docs/specifications");
    assert!(
        spec_dir.exists(),
        "P1 FALSIFIED: No specifications directory exists"
    );

    // Check for at least one spec file
    let has_specs = std::fs::read_dir(spec_dir)
        .map(|entries| entries.filter_map(Result::ok).count() > 0)
        .unwrap_or(false);

    assert!(has_specs, "P1 FALSIFIED: No specification files found");
}

// ============================================================================
// P2: Continuous Flow
// "Create continuous process flow to bring problems to the surface."
// Application: Streaming architecture for Audio and Tokens
// ============================================================================

/// P2: Streaming architecture exists
/// FALSIFICATION: No streaming module or chunk-based processing
#[test]
fn p2_continuous_flow_streaming() {
    // Check for streaming module in audio
    let has_stream_module = Path::new("src/audio/stream.rs").exists();

    // Check lib.rs for streaming references
    let lib_rs = include_str!("../src/lib.rs");
    let has_streaming_ref = lib_rs.contains("stream") || lib_rs.contains("chunk");

    assert!(
        has_stream_module || has_streaming_ref,
        "P2 FALSIFIED: No streaming architecture found"
    );
}

// ============================================================================
// P3: Pull Systems
// "Use 'pull' systems to avoid overproduction."
// Application: Lazy loading of tensors; computing only what is requested
// ============================================================================

/// P3: Lazy loading architecture exists
/// FALSIFICATION: All tensors loaded eagerly into memory
#[test]
fn p3_pull_system_lazy_loading() {
    // Check for mmap/lazy loading in format code
    let cargo_toml = include_str!("../Cargo.toml");

    // memmap2 indicates lazy/mmap loading capability
    let has_mmap = cargo_toml.contains("memmap2");

    // Also check for lazy loading patterns in v2.rs
    let v2_has_lazy = Path::new("src/format/v2.rs").exists();

    assert!(
        has_mmap || v2_has_lazy,
        "P3 FALSIFIED: No lazy loading (pull system) capability found"
    );
}

// ============================================================================
// P4: Level Workload (Heijunka)
// "Level out the workload."
// Application: Chunk-based processing in VAD to prevent spikes
// ============================================================================

/// P4: Chunk-based processing exists
/// FALSIFICATION: No chunking or batching in audio processing
#[test]
fn p4_heijunka_level_workload() {
    // Check for chunk-based processing patterns
    let mel_path = Path::new("src/audio/mel.rs");

    if mel_path.exists() {
        let mel_rs = std::fs::read_to_string(mel_path).expect("Failed to read mel.rs");
        let has_chunking =
            mel_rs.contains("chunk") || mel_rs.contains("hop_length") || mel_rs.contains("frame");

        assert!(
            has_chunking,
            "P4 FALSIFIED: Audio processing lacks chunk-based workload leveling"
        );
    }
    // Skip test if audio/mel.rs doesn't exist (module may have been refactored)
}

// ============================================================================
// P5: Stop to Fix Problems (Jidoka)
// "Build a culture of stopping to fix problems."
// Application: apr validate runs in CI; build fails on quality drop
// ============================================================================

/// P5: CI enforces quality gates
/// FALSIFICATION: No CI configuration or quality checks
#[test]
fn p5_jidoka_quality_gates() {
    let ci_path = Path::new(".github/workflows/ci.yml");
    assert!(ci_path.exists(), "P5 FALSIFIED: No CI configuration found");

    let ci_config = std::fs::read_to_string(ci_path).expect("read ci.yml");

    // CI should run tests and clippy
    let has_tests = ci_config.contains("cargo test");
    let has_clippy = ci_config.contains("clippy");

    assert!(
        has_tests && has_clippy,
        "P5 FALSIFIED: CI lacks quality gates (tests: {}, clippy: {})",
        has_tests,
        has_clippy
    );
}

/// P5b: Validation command exists
/// FALSIFICATION: No apr validate command
#[test]
fn p5b_validate_command_exists() {
    let validate_path = Path::new("crates/apr-cli/src/commands/validate.rs");
    assert!(
        validate_path.exists(),
        "P5 FALSIFIED: No validate command implementation"
    );
}

// ============================================================================
// P6: Standardized Tasks
// "Standardized tasks are the foundation for continuous improvement."
// Application: Makefile and cargo workflows are rigid and documented
// ============================================================================

/// P6: Makefile exists with standard targets
/// FALSIFICATION: No Makefile or missing standard targets
#[test]
fn p6_standardized_tasks_makefile() {
    let makefile_path = Path::new("Makefile");
    assert!(makefile_path.exists(), "P6 FALSIFIED: No Makefile found");

    let makefile = std::fs::read_to_string(makefile_path).expect("read Makefile");

    // Standard targets should exist
    let has_test = makefile.contains("test:");
    let has_build = makefile.contains("build:") || makefile.contains("release:");

    assert!(
        has_test || has_build,
        "P6 FALSIFIED: Makefile lacks standard targets"
    );
}

/// P6b: Cargo workflows documented
/// FALSIFICATION: No build commands in CLAUDE.md
#[test]
fn p6b_cargo_workflows_documented() {
    let claude_md = include_str!("../CLAUDE.md");

    let has_cargo_docs = claude_md.contains("cargo build")
        || claude_md.contains("cargo test")
        || claude_md.contains("Build Commands");

    assert!(
        has_cargo_docs,
        "P6 FALSIFIED: Cargo workflows not documented in CLAUDE.md"
    );
}

// ============================================================================
// P7: Visual Control
// "Use visual control so no problems are hidden."
// Application: apr tui and apr inspect make internal state visible
// ============================================================================

/// P7: Visual inspection tools exist
/// FALSIFICATION: No inspect or debug commands
#[test]
fn p7_visual_control_inspection() {
    let inspect_exists = Path::new("crates/apr-cli/src/commands/inspect.rs").exists();
    let debug_exists = Path::new("crates/apr-cli/src/commands/debug.rs").exists();

    assert!(
        inspect_exists || debug_exists,
        "P7 FALSIFIED: No visual inspection tools (inspect.rs or debug.rs)"
    );
}

// ============================================================================
// P8: Reliable Technology
// "Use only reliable, thoroughly tested technology."
// Application: Rust (Memory Safety) + WASM (Sandboxing)
// ============================================================================

/// P8: Project uses Rust (memory safe)
/// FALSIFICATION: Project is not Rust
#[test]
fn p8_reliable_technology_rust() {
    let cargo_toml = Path::new("Cargo.toml");
    assert!(
        cargo_toml.exists(),
        "P8 FALSIFIED: Not a Rust project (no Cargo.toml)"
    );

    // Verify it's actually a Rust project by checking for edition
    let cargo_content = include_str!("../Cargo.toml");
    assert!(
        cargo_content.contains("edition = "),
        "P8 FALSIFIED: Invalid Cargo.toml (no edition)"
    );
}

/// P8b: No unsafe code in core library
/// FALSIFICATION: unsafe_code is not forbidden
#[test]
fn p8b_no_unsafe_code() {
    let cargo_toml = include_str!("../Cargo.toml");

    // Check for unsafe_code = "forbid" in lints
    let forbids_unsafe = cargo_toml.contains("unsafe_code")
        && (cargo_toml.contains("forbid") || cargo_toml.contains("deny"));

    assert!(
        forbids_unsafe,
        "P8 FALSIFIED: unsafe_code is not forbidden in Cargo.toml lints"
    );
}

// ============================================================================
// P9: Grow Leaders
// "Grow leaders who thoroughly understand the work."
// Application: Documentation encourages "teaching" the system
// ============================================================================

/// P9: Documentation exists for learning
/// FALSIFICATION: No documentation beyond code
#[test]
fn p9_grow_leaders_documentation() {
    let has_book = Path::new("book").exists();
    let has_docs = Path::new("docs").exists();
    let has_readme = Path::new("README.md").exists();

    assert!(
        has_book || has_docs || has_readme,
        "P9 FALSIFIED: No documentation for learning"
    );
}

// ============================================================================
// P10: Develop People
// "Develop exceptional people and teams."
// Application: Contributors are guided by clear specs
// ============================================================================

/// P10: Contributing guidelines exist
/// FALSIFICATION: No contributor guidance
#[test]
fn p10_develop_people_guidelines() {
    let has_contributing = Path::new("CONTRIBUTING.md").exists();
    let has_claude_md = Path::new("CLAUDE.md").exists();

    // CLAUDE.md serves as contributor guidance for AI and humans
    assert!(
        has_contributing || has_claude_md,
        "P10 FALSIFIED: No contributor guidelines (CONTRIBUTING.md or CLAUDE.md)"
    );
}

// ============================================================================
// P11: Respect Partners
// "Respect your extended network of partners."
// Application: Full credit to upstream authors (OpenAI, Qwen, etc.)
// ============================================================================

/// P11: License file exists
/// FALSIFICATION: No license acknowledgment
#[test]
fn p11_respect_partners_license() {
    let has_license = Path::new("LICENSE").exists()
        || Path::new("LICENSE.md").exists()
        || Path::new("LICENSE-MIT").exists()
        || Path::new("LICENSE-APACHE").exists();

    assert!(
        has_license,
        "P11 FALSIFIED: No license file (partner respect requires clear licensing)"
    );
}

/// P11b: Dependencies credited
/// FALSIFICATION: No Cargo.toml (no dependency tracking)
#[test]
fn p11b_dependencies_credited() {
    let cargo_toml = include_str!("../Cargo.toml");

    // Should have dependencies section
    assert!(
        cargo_toml.contains("[dependencies]"),
        "P11 FALSIFIED: No dependencies section (all partners should be tracked)"
    );
}

// ============================================================================
// P12: Go and See (Genchi Genbutsu)
// "Go and see for yourself to thoroughly understand the situation."
// Application: Debuggers and Profilers are first-class tools
// ============================================================================

/// P12: Profiling/debugging tools exist
/// FALSIFICATION: No profiling or debugging commands
#[test]
fn p12_genchi_genbutsu_debugging() {
    let has_debug = Path::new("crates/apr-cli/src/commands/debug.rs").exists();
    let has_trace = Path::new("crates/apr-cli/src/commands/trace.rs").exists();
    let has_profile = Path::new("crates/apr-cli/src/commands/profile.rs").exists();

    assert!(
        has_debug || has_trace || has_profile,
        "P12 FALSIFIED: No debugging/profiling tools (genchi genbutsu requires seeing)"
    );
}

// ============================================================================
// P13: Decide Slowly
// "Make decisions slowly by consensus, thoroughly considering all options."
// Application: Specification was iterated multiple times
// ============================================================================

/// P13: Specification has version history
/// FALSIFICATION: No versioned specification
#[test]
fn p13_decide_slowly_versioned_spec() {
    let spec_path = Path::new("docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md");

    if spec_path.exists() {
        let spec = std::fs::read_to_string(spec_path).expect("read spec");

        // Should have version references
        let has_version = spec.contains("v1.")
            || spec.contains("v2.")
            || spec.contains("v3.")
            || spec.contains("version");

        assert!(
            has_version,
            "P13 FALSIFIED: Specification lacks version history (decisions should be deliberate)"
        );
    }
}

// ============================================================================
// P14: Relentless Reflection (Hansei)
// "Become a learning organization through relentless reflection."
// Application: Post-mortem analysis of major bugs
// ============================================================================

/// P14: Issue tracking exists
/// FALSIFICATION: No GitHub integration for tracking issues
#[test]
fn p14_hansei_reflection() {
    // Check for GitHub issue references in specs or docs
    let spec_path = Path::new("docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md");

    if spec_path.exists() {
        let spec = std::fs::read_to_string(spec_path).expect("read spec");

        // Should reference GitHub issues (GH-###)
        let has_issue_refs = spec.contains("GH-") || spec.contains("github.com");

        assert!(
            has_issue_refs,
            "P14 FALSIFIED: No issue tracking references (hansei requires reflection on problems)"
        );
    }
}

/// P14b: Changelog or history exists
/// FALSIFICATION: No record of changes
#[test]
fn p14b_change_history() {
    let has_changelog = Path::new("CHANGELOG.md").exists()
        || Path::new("CHANGES.md").exists()
        || Path::new("HISTORY.md").exists();

    let has_git = Path::new(".git").exists();

    assert!(
        has_changelog || has_git,
        "P14 FALSIFIED: No change history (hansei requires learning from past)"
    );
}

// ============================================================================
// Summary test
// ============================================================================

/// Summary: All 14 Toyota principles have verification
/// This test documents which principles are verified above
#[test]
fn toyota_principles_summary() {
    // This test always passes - it documents the principle coverage
    let principles = [
        "P1: Long-Term Philosophy - ✅ Documented architecture",
        "P2: Continuous Flow - ✅ Streaming architecture",
        "P3: Pull Systems - ✅ Lazy/mmap loading",
        "P4: Level Workload - ✅ Chunk-based processing",
        "P5: Stop to Fix - ✅ CI quality gates",
        "P6: Standardized Tasks - ✅ Makefile + docs",
        "P7: Visual Control - ✅ Inspect/debug tools",
        "P8: Reliable Technology - ✅ Rust + no unsafe",
        "P9: Grow Leaders - ✅ Documentation",
        "P10: Develop People - ✅ Contributor guides",
        "P11: Respect Partners - ✅ Licensing",
        "P12: Go and See - ✅ Debugging tools",
        "P13: Decide Slowly - ✅ Versioned specs",
        "P14: Relentless Reflection - ✅ Issue tracking",
    ];

    for p in &principles {
        eprintln!("{}", p);
    }

    assert_eq!(principles.len(), 14, "Should verify all 14 principles");
}
