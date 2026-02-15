
// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use aprender::format::v2::AprV2Writer;
    use std::io::Write;
    use tempfile::{tempdir, NamedTempFile};
include!("inspect_part_03_part_02.rs");
include!("inspect_part_03_part_03.rs");
}
