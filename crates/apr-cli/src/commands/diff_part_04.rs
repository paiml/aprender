
// ============================================================================
// Tests (Minimal - Most logic is tested in library)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use aprender::format::diff::DiffEntry;
    use std::io::Write;
    use tempfile::{tempdir, NamedTempFile};
include!("diff_part_04_part_02.rs");
include!("diff_part_04_part_03.rs");
include!("diff_part_04_part_04.rs");
}
