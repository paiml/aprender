
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
include!("qa_config.rs");
include!("qa_tests_json_and_verbose.rs");
include!("qa_tests_report_pass_fail.rs");
include!("qa_report.rs");
include!("qa_failed_gates_gate.rs");
include!("qa_config_print_gate.rs");
include!("fields.rs");
}
