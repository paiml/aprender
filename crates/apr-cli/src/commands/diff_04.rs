
// ============================================================================
// Tests (Minimal - Most logic is tested in library)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use aprender::format::diff::DiffEntry;
    use std::io::Write;
    use tempfile::{tempdir, NamedTempFile};
include!("diff_validate_paths.rs");
include!("diff_normalize_tensor_truncate.rs");
include!("diff_print_tensor.rs");
}
