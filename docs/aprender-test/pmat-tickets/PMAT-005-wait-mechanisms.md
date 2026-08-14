# PMAT-005: Implement Wait Mechanisms

**Status**: ✅ Complete (2025-12-12)
**Priority**: High
**Component**: probar/src/wait.rs
**Target Coverage**: 95%

## Summary

Implement Playwright-compatible wait mechanisms for synchronization.

## Requirements

### Wait Methods to Implement

1. **wait_for_navigation()** - Wait for page navigation
   - Support URL pattern matching
   - Support load state options

2. **wait_for_load_state()** - Wait for specific load state
   - `load`: window load event
   - `domcontentloaded`: DOMContentLoaded event
   - `networkidle`: no network requests for 500ms

3. **wait_for_url()** - Wait for URL to match pattern
   - Support string, regex, or predicate

4. **wait_for_function()** - Wait for JS function to return truthy
   - Execute custom JS condition
   - Support polling interval

5. **wait_for_response()** - Wait for network response
   - Match by URL pattern or predicate
   - Return response object

6. **wait_for_request()** - Wait for network request
   - Match by URL pattern or predicate
   - Return request object

7. **wait_for_event()** - Wait for page/browser event
   - Support various event types

## Implementation Details

```rust
impl Page {
    pub async fn wait_for_navigation(&self, options: NavigationOptions) -> ProbarResult<Response> { ... }
    pub async fn wait_for_load_state(&self, state: LoadState) -> ProbarResult<()> { ... }
    pub async fn wait_for_url(&self, pattern: UrlPattern) -> ProbarResult<()> { ... }
    pub async fn wait_for_function(&self, js: &str, options: WaitOptions) -> ProbarResult<JsValue> { ... }
    pub async fn wait_for_response(&self, pattern: UrlPattern) -> ProbarResult<Response> { ... }
    pub async fn wait_for_request(&self, pattern: UrlPattern) -> ProbarResult<Request> { ... }
    pub async fn wait_for_event(&self, event: EventType) -> ProbarResult<Event> { ... }
}

pub enum LoadState {
    Load,
    DomContentLoaded,
    NetworkIdle,
}

pub enum UrlPattern {
    String(String),
    Regex(Regex),
    Predicate(Box<dyn Fn(&str) -> bool>),
}
```

## Acceptance Criteria

- [ ] Navigation wait works with various load states
- [ ] URL pattern matching supports string/regex/predicate
- [ ] Custom JS function wait executes correctly
- [ ] Network request/response waits capture correct data
- [ ] Event wait handles page events
- [ ] Proper timeout handling
- [ ] Test coverage >= 95%

## Test Cases

1. `test_wait_for_navigation`
2. `test_wait_for_load_state_load`
3. `test_wait_for_load_state_networkidle`
4. `test_wait_for_url_string`
5. `test_wait_for_url_regex`
6. `test_wait_for_function_truthy`
7. `test_wait_for_response_url`
8. `test_wait_for_request_pattern`
9. `test_wait_for_event_popup`
10. `test_wait_timeout_exceeded`

## References

- Playwright Wait: https://playwright.dev/docs/navigations
