# Mutation Testing Setup - Phase 3 of GH-55

## Status: CI Integration Complete

Mutation testing has been successfully integrated into the CI/CD pipeline via `.github/workflows/ci.yml`.

## CI Configuration

The mutation testing job runs on every PR and push to main:

```yaml
mutants:
  name: Mutation Testing
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@stable
    - uses: Swatinem/rust-cache@v2
    - name: Install cargo-mutants
      uses: taiki-e/install-action@v2
      with:
        tool: cargo-mutants
    - name: Run mutation tests (sample)
      run: cargo mutants --no-times --timeout 300 --in-place -- --all-features
      continue-on-error: true
    - name: Upload mutants results
      if: always()
      uses: actions/upload-artifact@v4
      with:
        name: mutants-results
        path: mutants.out/
        retention-days: 30
```

**Key features:**
- Runs with 300-second timeout per mutant
- Tests in-place for speed
- Continues on error (mutation testing is informational)
- Uploads results as artifacts for 30 days

## Local Execution

**Known Issue:** Local mutation testing has a package ambiguity issue when testing published crates:

```
error: There are multiple `aprender` packages in your project, and the specification `aprender@0.4.1` is ambiguous.
```

This occurs because cargo-mutants sees both:
1. `path+file:///home/noah/src/aprender#0.4.1` (local)
2. `registry+https://github.com/rust-lang/crates.io-index#aprender@0.4.1` (published)

**Workaround for local testing:**
- Temporarily bump version to unpublished (e.g., 0.4.2-dev)
- Or rely on CI for mutation testing (recommended)

## Viewing Results

### From CI:
```bash
# List recent CI runs
gh run list --workflow=ci.yml --limit 5

# Download mutation test artifacts
gh run download <run-id> -n mutants-results
```

### Local (when working):
```bash
# Run mutation tests
cargo mutants --no-times --timeout 300 --in-place -- --all-features

# View results
cat mutants.out/mutants.out

# Or view HTML report (if generated)
open mutants.out/html/index.html
```

## Mutation Score Baseline

**Total mutants:** ~13,705 (across entire codebase)

**Per-module estimates** (based on --list output):
- `src/loss/mod.rs`: 60 mutants
- `src/optim/mod.rs`: 92 mutants
- Other modules: TBD from CI results

**Target:** ≥80% mutation score (PMAT recommendation)

## Phase 3 Completion

- ✅ cargo-mutants installed (v25.3.1)
- ✅ Mutation testing integrated in CI
- ✅ Results uploaded as artifacts
- ✅ Configuration documented
- ⚠️ Local execution has known package ambiguity issue
- 📊 Baseline mutation score: Pending CI results analysis

## Next Steps

1. Analyze mutation test results from CI artifacts
2. Identify weak test coverage areas
3. Improve tests for uncaught mutants
4. Target ≥80% mutation score

## References

- cargo-mutants documentation: https://mutants.rs/
- CI workflow: `.github/workflows/ci.yml` (lines 86-106)
- PMAT recommendation: Testing Excellence ≥80% mutation score
