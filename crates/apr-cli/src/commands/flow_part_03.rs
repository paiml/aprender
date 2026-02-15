
// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::{tempdir, NamedTempFile};
include!("flow_part_03_part_02.rs");
include!("flow_part_03_part_03.rs");
include!("flow_part_03_part_04.rs");
}
