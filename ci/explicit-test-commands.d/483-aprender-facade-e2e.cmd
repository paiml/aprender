# #4059: a KEPT binary's CLI tests, dark in CI until now. Measured on lambda before wiring; see docs/audits/impl-PMAT-4059-receipt.md.
cargo test -p aprender --test e2e_cli_t --test e2e_http_serve_t --test e2e_mcp_stdio_t
