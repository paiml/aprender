
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::{tempdir, NamedTempFile};
include!("bench_config.rs");
include!("bench_bench.rs");
include!("bench_calculate_stats.rs");
include!("bench_brick_name.rs");
}
