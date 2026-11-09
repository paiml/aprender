# Release Process

This document describes the release process for Aprender.

## Prerequisites

1. All CI checks must pass
2. Quality gates must be satisfied:
   - Test coverage ≥ 97%
   - Mutation score ≥ 85%
   - TDG score ≥ 95 (A+)
   - Zero SATD comments
   - Complexity within limits
3. CHANGELOG.md updated with release notes
4. Version bumped in Cargo.toml

## Release Checklist

### 1. Pre-Release Validation

```bash
# Ensure all tests pass
make test

# Run quality gates
pmat quality-gate

# Check documentation builds
cargo doc --no-deps --all-features

# Verify examples work
cargo run --example boston_housing
cargo run --example iris_clustering
cargo run --example dataframe_basics

# Test crates.io package
cargo publish --dry-run
```

### 2. Create Release Tag

```bash
# Update version in Cargo.toml (e.g., 0.1.0 -> 0.1.1)
# Update CHANGELOG.md with release notes

git add Cargo.toml CHANGELOG.md
git commit -m "Release v0.1.1"
git tag -a v0.1.1 -m "Release v0.1.1"
git push origin main
git push origin v0.1.1
```

### 3. GitHub Actions Automation

The release workflow (`.github/workflows/release.yml`) will:

1. **Build artifacts** for Linux, macOS, and Windows
2. **Run tests** on all platforms
3. **Publish to crates.io** (when enabled)
4. **Create GitHub Release** with artifacts and release notes

### 4. Manual crates.io Publishing

If automatic publishing is not configured:

```bash
cargo publish --token $CARGO_REGISTRY_TOKEN
```

## Enabling Automatic crates.io Publishing

1. Get your crates.io API token from https://crates.io/me
2. Add token as GitHub Secret named `CARGO_REGISTRY_TOKEN`
3. Update `.github/workflows/release.yml`:
   ```yaml
   - name: Publish to crates.io
     run: cargo publish --token ${{ secrets.CARGO_REGISTRY_TOKEN }}
   ```

## Version Numbering

Follow Semantic Versioning (SemVer):

- **MAJOR** (0.x.0): Breaking API changes
- **MINOR** (0.x.0): New features, backward compatible
- **PATCH** (0.0.x): Bug fixes, backward compatible

Current: `0.1.0` (initial release)

## Release definition

Operator ruling, 2026-10-30, verbatim: "a release = the version is on crates.io and passed the clean-container smoke. A tag alone is not a release. A tag not on crates.io within 2 h of push is an andon that pages Noah live. The post-publish step closes the milestone/epic ONLY after crates.io shows the version. Fixture: tag pushed, publish fails -> milestone stays open, andon fires."

- **The andon**: `scripts/release/tag_publish_andon.sh` (user timer `scripts/release/systemd/aprender-tag-andon.timer`, every 15 min). For every `v<semver>` tag created on or after `--since` and older than 120 min, crate `aprender` must show that version on crates.io. If it does not, it writes a cop-inbox line every run, one escalation line and one live page per tag, and exits 1. An unreadable crates.io is NOT_MEASURED (exit 2, no page).
- **The close gate**: `crates_io_close_gate` in `scripts/release/lib_crates_io.sh`, called at the top of step 9 (`close`) of `scripts/release/autopilot.sh`. Unless crates.io shows the version, the epic and the milestone stay open.
- Proof: `bash scripts/release/falsify_tag_publish_andon.sh --mutants` and `scripts/check_release_autopilot_dogfood_close.sh`.

## Post-Release

1. Verify release on crates.io: https://crates.io/crates/aprender
2. Check documentation: https://docs.rs/aprender
3. Monitor GitHub Actions for completion
4. Announce release (if applicable)

## Rollback Procedure

If a release has critical issues:

1. **Yank the bad version** from crates.io:
   ```bash
   cargo yank --vers 0.1.1
   ```

2. **Fix the issue** and release a patch version

3. **Never delete tags** - use new versions instead

## Quality Metrics Maintained

- TDG Score: 95.6/100 (A+)
- Test Coverage: 97.72%
- Mutation Score: 85.3%
- Repository Score: 95.0/100 (A+)
- Zero SATD comments
- Cyclomatic complexity ≤ 10
- Cognitive complexity ≤ 15
