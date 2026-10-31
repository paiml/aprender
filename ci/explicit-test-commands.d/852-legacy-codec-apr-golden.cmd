# .apr containers written by bincode 1.x still read through the new codec (FALSIFY-LEGACY-CODEC-005), aprender#6593
cargo test -p apr-format --test golden_fixtures
