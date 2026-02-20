
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
include!("qa_part_08_part_02.rs");
include!("qa_tests_json_and_verbose.rs");
include!("qa_tests_report_pass_fail.rs");
include!("qa_part_08_part_05.rs");
include!("qa_part_08_part_06.rs");
include!("qa_part_08_part_07.rs");
include!("fields.rs");
}
