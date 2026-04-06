Check the latest versions of all Sovereign AI Stack crates from crates.io.

Run: `cargo run --quiet -- stack versions`

This fetches current versions from crates.io with 15-minute caching. Compare against local Cargo.toml to identify outdated dependencies.

After checking, suggest any updates needed with `cargo update <crate>` commands.
