
// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
include!("hex_part_06_part_02.rs");
include!("make.rs");
include!("tests_blocks_and_slices.rs");
}
