# PMAT-002: Implement Locator Operations

**Status**: ✅ Complete (2025-12-12)
**Priority**: High
**Component**: probar/src/locator.rs
**Target Coverage**: 95%

## Summary

Implement Playwright-compatible locator composition and filtering operations.

## Requirements

### Operations to Implement

1. **filter()** - Filter locator by conditions
   - `has`: Child locator must match
   - `hasText`: Must contain text
   - `hasNot`: Child locator must NOT match
   - `hasNotText`: Must NOT contain text

2. **and()** - Intersection of two locators
   - Both conditions must match the same element

3. **or()** - Union of two locators
   - Either condition can match

4. **first()** - Get first matching element

5. **last()** - Get last matching element

6. **nth(index)** - Get element at index

## Implementation Details

```rust
impl Locator {
    pub fn filter(self, options: FilterOptions) -> Self { ... }
    pub fn and(self, other: Locator) -> Self { ... }
    pub fn or(self, other: Locator) -> Self { ... }
    pub fn first(self) -> Self { ... }
    pub fn last(self) -> Self { ... }
    pub fn nth(self, index: usize) -> Self { ... }
}

pub struct FilterOptions {
    pub has: Option<Locator>,
    pub has_text: Option<String>,
    pub has_not: Option<Locator>,
    pub has_not_text: Option<String>,
}
```

## Acceptance Criteria

- [ ] filter() supports has, hasText, hasNot, hasNotText
- [ ] and() returns intersection of locators
- [ ] or() returns union of locators
- [ ] first(), last(), nth() work correctly
- [ ] Operations are chainable
- [ ] Test coverage >= 95%

## Test Cases

1. `test_filter_has_child`
2. `test_filter_has_text`
3. `test_filter_has_not`
4. `test_and_both_conditions`
5. `test_or_either_condition`
6. `test_first_last_nth`
7. `test_chained_operations`

## References

- Playwright Locators: https://playwright.dev/docs/locators#filtering-locators
