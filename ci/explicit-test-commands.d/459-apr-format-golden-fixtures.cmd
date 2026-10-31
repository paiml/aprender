# Also proves FALSIFY-LEGACY-CODEC-005: .apr containers written by bincode 1.x still read through the new codec, aprender#6593
cargo test -p apr-format --test golden_fixtures
