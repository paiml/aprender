//! CI and build helpers ported from `scripts/**/*.py` (C301: no Python in the build; the
//! inventory and port order are in PY-INV). Each module is one port and prints
//! byte-for-byte what its original printed; `scripts/tests/ci_tools_py_parity_test.sh`
//! runs the same cases through both, with the `.py` original as the external validator.

pub mod coverage_report_scope;
pub mod dag_status;
pub mod git_patch_id;
pub mod llama_fit_verdict;
pub mod package_include_diff;
pub mod publishable_crates;
pub mod pystr;
pub mod tarball_build_errors;
pub mod tarball_shrink_report;
pub mod tarball_workspace;
