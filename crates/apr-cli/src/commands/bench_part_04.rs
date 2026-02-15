
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::{tempdir, NamedTempFile};
include!("bench_part_04_part_02.rs");
include!("bench_part_04_part_03.rs");
include!("bench_part_04_part_04.rs");
include!("bench_part_04_part_05.rs");
}
