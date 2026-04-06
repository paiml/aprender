# PMAT-006: Implement Network Features

**Status**: ✅ Complete (2025-12-12)
**Priority**: Medium
**Component**: probar/src/network.rs
**Target Coverage**: 95%

## Summary

Implement additional Playwright-compatible network interception features.

## Requirements

### Features to Implement

1. **Request Abort** - Block/abort requests
   - Abort matching requests
   - Support URL patterns
   - Error reason support (failed, aborted, etc.)

2. **wait_for_response()** - Wait for specific response
   - Already covered in PMAT-005 but network-specific implementation

3. **HAR Recording** - Record HTTP Archive
   - Capture all network traffic
   - Export to HAR format

4. **HAR Playback** - Mock from HAR
   - Load HAR file
   - Replay recorded responses

5. **WebSocket Interception**
   - Inspect WebSocket frames
   - Mock WebSocket messages

## Implementation Details

```rust
impl Route {
    pub async fn abort(&self, reason: AbortReason) -> ProbarResult<()> { ... }
}

pub enum AbortReason {
    Failed,
    Aborted,
    TimedOut,
    AccessDenied,
    ConnectionClosed,
    ConnectionFailed,
    ConnectionRefused,
    ConnectionReset,
    InternetDisconnected,
    NameNotResolved,
    BlockedByClient,
}

impl Page {
    pub async fn route_abort(&self, pattern: UrlPattern, reason: AbortReason) -> ProbarResult<()> { ... }
}

// HAR Support
pub struct HarRecorder {
    entries: Vec<HarEntry>,
}

impl HarRecorder {
    pub fn new() -> Self { ... }
    pub fn start(&mut self, context: &BrowserContext) -> ProbarResult<()> { ... }
    pub fn stop(&mut self) -> ProbarResult<Har> { ... }
    pub fn save(&self, path: &Path) -> ProbarResult<()> { ... }
}

impl BrowserContext {
    pub async fn route_from_har(&self, har_path: &Path) -> ProbarResult<()> { ... }
}
```

## Acceptance Criteria

- [ ] Request abort blocks matching requests
- [ ] Abort reasons are correctly reported
- [ ] HAR recording captures all traffic
- [ ] HAR playback mocks responses correctly
- [ ] WebSocket frames can be inspected
- [ ] Test coverage >= 95%

## Test Cases

1. `test_abort_request_pattern`
2. `test_abort_with_reason`
3. `test_har_recording_basic`
4. `test_har_recording_complete`
5. `test_har_playback`
6. `test_websocket_inspection`

## References

- Playwright Network: https://playwright.dev/docs/network
- HAR Spec: http://www.softwareishard.com/blog/har-12-spec/
