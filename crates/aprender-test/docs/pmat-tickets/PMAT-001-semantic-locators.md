# PMAT-001: Implement Semantic Locators

**Status**: ✅ Complete (2025-12-12)
**Priority**: High
**Component**: probar/src/locator.rs
**Target Coverage**: 95%

## Summary

Implement Playwright-compatible semantic locators for accessible element selection.

## Requirements

### Locators to Implement

1. **Role Selector** (`role=`)
   - Match elements by ARIA role attribute
   - Support implicit roles (e.g., `<button>` has role="button")
   - Example: `page.locator("role=button")`

2. **Label Selector** (`label=`)
   - Match form elements by associated `<label>` text
   - Support `for` attribute association
   - Support nested labels
   - Example: `page.locator("label=Username")`

3. **Placeholder Selector** (`placeholder=`)
   - Match input/textarea by placeholder attribute
   - Example: `page.locator("placeholder=Enter email")`

4. **Alt Text Selector** (`alt=`)
   - Match images by alt attribute
   - Example: `page.locator("alt=Company Logo")`

## Implementation Details

```rust
pub enum SemanticSelector {
    Role(String),
    Label(String),
    Placeholder(String),
    AltText(String),
}

impl Locator {
    pub fn by_role(role: &str) -> Self { ... }
    pub fn by_label(text: &str) -> Self { ... }
    pub fn by_placeholder(text: &str) -> Self { ... }
    pub fn by_alt_text(text: &str) -> Self { ... }
}
```

## Acceptance Criteria

- [ ] Role selector matches ARIA roles and implicit roles
- [ ] Label selector finds elements by label association
- [ ] Placeholder selector matches input placeholders
- [ ] Alt text selector matches image alt attributes
- [ ] All selectors support exact and partial matching
- [ ] Test coverage >= 95%
- [ ] Documentation updated

## Test Cases

1. `test_role_selector_explicit_aria`
2. `test_role_selector_implicit_button`
3. `test_label_selector_for_attribute`
4. `test_label_selector_nested`
5. `test_placeholder_selector_input`
6. `test_placeholder_selector_textarea`
7. `test_alt_text_selector_image`

## References

- Playwright Locators: https://playwright.dev/docs/locators
- ARIA Roles: https://www.w3.org/TR/wai-aria-1.2/#role_definitions
