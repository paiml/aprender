# PMAT-003: Implement Mouse Actions

**Status**: ✅ Complete (2025-12-12)
**Priority**: High
**Component**: probar/src/actions.rs
**Target Coverage**: 95%

## Summary

Implement additional Playwright-compatible mouse actions.

## Requirements

### Actions to Implement

1. **dblclick()** - Double-click on element
   - Dispatch two click events in quick succession
   - Support click options (position, modifiers)

2. **right_click()** / **click({ button: 'right' })** - Right-click
   - Dispatch contextmenu event
   - Support position and modifiers

3. **hover()** - Move mouse over element
   - Dispatch mouseover, mouseenter events
   - Support position options

4. **focus()** - Focus element
   - Dispatch focus event

5. **blur()** - Remove focus from element
   - Dispatch blur event

## Implementation Details

```rust
impl Locator {
    pub async fn dblclick(&self) -> ProbarResult<()> { ... }
    pub async fn click_options(&self, options: ClickOptions) -> ProbarResult<()> { ... }
    pub async fn hover(&self) -> ProbarResult<()> { ... }
    pub async fn focus(&self) -> ProbarResult<()> { ... }
    pub async fn blur(&self) -> ProbarResult<()> { ... }
}

pub struct ClickOptions {
    pub button: MouseButton,
    pub click_count: u32,
    pub position: Option<Position>,
    pub modifiers: Vec<KeyModifier>,
}

pub enum MouseButton {
    Left,
    Right,
    Middle,
}
```

## Acceptance Criteria

- [ ] dblclick() dispatches proper events
- [ ] Right-click triggers contextmenu
- [ ] hover() moves mouse and triggers events
- [ ] focus()/blur() work correctly
- [ ] Options (position, modifiers) supported
- [ ] Test coverage >= 95%

## Test Cases

1. `test_dblclick_element`
2. `test_right_click_context_menu`
3. `test_hover_triggers_events`
4. `test_focus_blur_element`
5. `test_click_with_modifiers`
6. `test_click_at_position`

## References

- Playwright Actions: https://playwright.dev/docs/input
