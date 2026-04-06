# Contributing

Guidelines for contributing to Presentar.

## Getting Started

```bash
git clone https://github.com/your-org/presentar
cd presentar
make tier1  # Verify setup
```

## Development Workflow

1. **Fork and branch**
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Write tests first** (TDD)
   ```rust
   #[test]
   fn test_new_feature() {
       // Test first, implement second
   }
   ```

3. **Implement**
   ```bash
   cargo test  # Must pass
   ```

4. **Run quality gates**
   ```bash
   make tier2  # Must pass
   ```

5. **Submit PR**

## Code Standards

| Standard | Enforcement |
|----------|-------------|
| Formatting | `cargo fmt --check` |
| Linting | `cargo clippy -- -D warnings` |
| Tests | `cargo test` |
| Coverage | 85% minimum |

## Commit Messages

```
type(scope): description

- type: feat, fix, docs, refactor, test
- scope: core, widgets, layout, test
- description: imperative, lowercase
```

Example:
```
feat(widgets): add Toggle widget
```

## PR Requirements

- [ ] Tests pass
- [ ] Clippy clean
- [ ] Coverage maintained
- [ ] Documentation updated
- [ ] CHANGELOG entry

## Testing

```bash
make test-fast   # Quick check
make tier2       # Full validation
make tier3       # With mutation
```

## Documentation

```bash
cargo doc --open
```

## Verified Test

```rust
#[test]
fn test_contribution_guidelines() {
    // This test documents expectations
    let required_coverage = 85;
    let clippy_warnings_allowed = 0;

    assert!(required_coverage >= 85);
    assert_eq!(clippy_warnings_allowed, 0);
}
```
