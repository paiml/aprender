# Distribution

Distribution is the final step in Phase 5, packaging the compiled binary for delivery to end users. Batuta supports multiple distribution channels depending on the target audience.

## Distribution Channels

| Channel | Audience | Format |
|---------|----------|--------|
| crates.io | Rust developers | Source crate |
| cargo-binstall | Rust developers | Pre-built binary |
| GitHub Releases | All developers | Tarball / zip |
| Homebrew | macOS / Linux users | Formula |
| Docker | Cloud deployment | Container image |
| npm/wasm-pack | Web developers | WASM package |

## crates.io Publishing

For libraries that other Rust projects will depend on:

```bash
# Verify package contents
cargo package --list

# Dry run (no upload)
cargo publish --dry-run

# Publish to crates.io
cargo publish
```

Key checks before publishing:

- `Cargo.toml` has `version`, `description`, `license`, `repository`
- No path dependencies (use crates.io versions)
- All tests pass with `--locked`
- MSRV (Minimum Supported Rust Version) is declared

## Binary Distribution

For end-user tools, distribute pre-built binaries:

```bash
# Build release binaries for multiple targets
batuta build --release --target x86_64-unknown-linux-musl
batuta build --release --target aarch64-unknown-linux-gnu
batuta build --release --target x86_64-apple-darwin

# Package with checksums
tar czf app-linux-x86_64.tar.gz -C target/x86_64-unknown-linux-musl/release app
sha256sum app-linux-x86_64.tar.gz > app-linux-x86_64.tar.gz.sha256
```

## cargo-binstall Support

Add metadata to `Cargo.toml` for automatic binary installation:

```toml
[package.metadata.binstall]
pkg-url = "{ repo }/releases/download/v{ version }/{ name }-{ target }.tar.gz"
bin-dir = "{ bin }{ binary-ext }"
pkg-fmt = "tgz"
```

Users can then install with:

```bash
cargo binstall my-app
```

## Docker Distribution

For cloud deployment, Batuta's `batuta deploy` command generates Dockerfiles using `scratch` base images (works because musl-linked binaries have no dynamic dependencies).

## Stack Publish Status

For Sovereign AI Stack crates, `batuta stack publish-status` checks which crates need publishing. Results are cached (warm: <100ms, cold: ~7s) with invalidation on Cargo.toml changes, git HEAD moves, or crates.io TTL expiry (15 minutes).

---

**Navigate:** [Table of Contents](../SUMMARY.md)
