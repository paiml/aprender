## Description

Brief description of changes.

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring (no functional changes)

## Checklist

### Code Quality
- [ ] Code follows project style guidelines
- [ ] Self-review of code completed
- [ ] Comments added for complex logic
- [ ] No new warnings from `cargo clippy`

### Testing (Popperian Falsifiability)
- [ ] New tests added for new functionality
- [ ] All existing tests pass (`cargo test`)
- [ ] Interface tests written FIRST (test-defines-interface)
- [ ] Property-based tests added where applicable
- [ ] Random seeds are fixed for reproducibility

### Documentation
- [ ] API documentation updated
- [ ] CHANGELOG.md updated
- [ ] ADR created for architectural decisions (if applicable)

### Performance (Statistical Rigor)
- [ ] Benchmarks run with `cargo criterion`
- [ ] No performance regression (within 95% CI of baseline)
- [ ] Sample sizes documented for new benchmarks
- [ ] Effect sizes calculated for comparisons

### Reproducibility
- [ ] Build tested in clean environment
- [ ] Environment variables documented
- [ ] DVC tracked data changes (if applicable)

## Related Issues

Closes #

## Screenshots/Output (if applicable)

## Additional Notes

