//! Command handlers - extracted from main.rs for testability
//!
//! Each handler module contains:
//! - The execution logic for a CLI command
//! - Pure helper functions
//! - Comprehensive tests

pub mod animation;
pub mod audio;
pub mod av_sync;
pub mod build;
pub mod comply;
pub mod config;
pub mod coverage;
pub mod init;
#[cfg(feature = "llm")]
pub mod llm;
pub mod report;
pub mod serve;
pub mod video;

// Re-export handlers for convenient access
pub use comply::{
    check_c001_code_execution, check_c002_console_errors, check_c003_custom_elements,
    check_c004_threading_modes, check_c005_low_memory, check_c006_headers, check_c007_replay_hash,
    check_c008_cache, check_c009_wasm_size, check_c010_panic_paths, check_makefile_cross_origin,
    check_probar_cross_origin_config, run_compliance_checks, ComplianceResult,
};
pub use config::execute_config;
pub use coverage::{
    calculate_coverage, create_sample_coverage_data, execute_coverage, generate_coverage_report,
    is_gap_cell, load_coverage_from_json,
};
pub use init::{execute_init, generate_probar_config, is_valid_init_path};
pub use report::{
    execute_report, generate_cobertura_report, generate_html_report, generate_json_report,
    generate_junit_report, generate_lcov_report, open_in_browser,
};
