# #4059: a KEPT binary's CLI tests, dark in CI until now. Measured on lambda before wiring; see docs/audits/impl-PMAT-4059-receipt.md.
cargo test -p apr-cli --test command_coverage
