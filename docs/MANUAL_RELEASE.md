# Manual Release Guide for Aprender

This guide covers the manual release process for publishing to crates.io.

## Prerequisites

1. You have publish permissions for the `aprender` crate on crates.io
2. You have a crates.io API token
3. All CI checks are passing
4. Quality gates are satisfied (97%+ coverage, 85%+ mutation score)

## Release Steps

### 1. Prepare for Release

```bash
# Ensure clean working directory
git status

# Pull latest changes
git checkout main
git pull origin main

# Run quality checks
pmat quality-gate
make test
make coverage

# Verify package builds
cargo publish --dry-run
```

### 2. Update Version and Changelog

Edit `Cargo.toml` and update the version:
```toml
version = "0.1.0"  # Current version
```

Update `CHANGELOG.md`:
- Move changes from `[Unreleased]` to new version section
- Add release date
- Update comparison links at bottom

### 3. Commit Version Bump

```bash
git add Cargo.toml CHANGELOG.md
git commit -m "Release v0.1.0"
git push origin main
```

### 4. Create Git Tag

```bash
git tag -a v0.1.0 -m "Release v0.1.0

## Release Highlights
- Pure Rust machine learning library
- Linear Regression with OLS
- K-Means clustering with k-means++
- 97.72% test coverage
- 85.3% mutation score
- TDG Score: 95.6/100 (A+)
"

git push origin v0.1.0
```

### 5. Wait for CI Verification

Check GitHub Actions run for the tag:
- Go to https://github.com/paiml/aprender/actions
- Verify the "Release" workflow passes
- This validates the release build works

### 6. Publish to crates.io

```bash
# Final verification
cargo publish --dry-run

# Publish to crates.io
cargo publish

# When prompted, confirm the publish
```

### 7. Create GitHub Release

1. Go to https://github.com/paiml/aprender/releases/new
2. Select the tag you just created (v0.1.0)
3. Title: `v0.1.0`
4. Description: Copy from CHANGELOG.md
5. Click "Publish release"

### 8. Verify Release

1. **crates.io**: Visit https://crates.io/crates/aprender
   - Verify new version appears
   - Check that documentation builds

2. **docs.rs**: Visit https://docs.rs/aprender
   - Verify documentation is published
   - Check all modules render correctly

3. **Test installation**:
   ```bash
   # In a temporary directory
   cargo init --lib test-aprender
   cd test-aprender
   cargo add aprender
   cargo build
   ```

## Post-Release

### Update README badges (if needed)

If version badges need updating, create a new commit:
```bash
# Edit README.md
git add README.md
git commit -m "Update version badges to v0.1.0"
git push origin main
```

### Announce Release (Optional)

- Tweet/post about the release
- Update any external documentation
- Notify users/contributors

## Troubleshooting

### "crate name already exists"
- You can only publish each version once
- Bump to next version (e.g., 0.1.1)

### "permission denied"
- Ensure you're logged in: `cargo login`
- Check you have publish permissions: `cargo owner --list aprender`

### "failed to verify package"
- Run `cargo publish --dry-run` to see detailed errors
- Common issues: missing files, broken links in docs

### "documentation failed to build"
- Test locally: `cargo doc --no-deps --all-features`
- Fix any doc warnings or errors

## Rollback

If you need to yank a release:

```bash
# Yank the problematic version (doesn't delete it)
cargo yank --vers 0.1.0

# Fix issues and publish new version
# Edit Cargo.toml to 0.1.1
cargo publish

# Un-yank if you fix it in place (rare)
cargo yank --undo --vers 0.1.0
```

## Quality Checklist

Before releasing, ensure:

- ✅ TDG Score ≥ 95 (A+)
- ✅ Repository Score ≥ 95 (A+)
- ✅ Test Coverage ≥ 97%
- ✅ Mutation Score ≥ 85%
- ✅ Zero clippy warnings
- ✅ Zero SATD comments
- ✅ All examples run successfully
- ✅ Documentation builds without warnings
- ✅ CHANGELOG.md updated
- ✅ All CI checks passing

## Split-host T-2 preflight: per-host receipt merge (fallback, REL1, epic #6728)

`scripts/release/t2_preflight.sh` `decide()` judges ONE dogfood log and is unchanged. When the gate set
must run on two hosts (for example an x86_64 CUDA box and an arm64 box), never hand-write a merged
verdict. Use `scripts/release/merge_host_receipts.sh --commit SHA --out FILE --log HOST=PATH
--log HOST=PATH [--arch HOST=x86_64] [--x86-cuda-ruling PATH]`. Each log must carry a header line
`# receipt-commit: <sha>` (or t2_preflight's `judging <sha>` line). The merge writes `GO` only if all
header shas equal `--commit`; every `[FAIL]` row is the pre-bump class `decide()` tolerates or is
environment-only (`Feature not enabled: cuda`, `NOT HELD under`, `is not on this host`, `GitHub read
failed`) and green on another host; the declared gate count agrees; and the `model-parity` (C14) row is
green with `--features cuda` on a host named `--arch HOST=x86_64`, else exit 2 unless a quorum receipt
naming the commit is given with `--x86-cuda-ruling`. Exit 0 GO, 1 NO-GO, 2 refused; the verdict file
lists receipt path and host per row. The pre-bump model-ladder deferral is not merged: a red ladder row
fails the merge. Falsifier: `scripts/release/falsify_merge_host_receipts.sh` (CI fragment 862).
