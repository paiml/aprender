# Debugging

![Debug Coverage Analysis](../assets/coverage_pattern.png)

probador provides comprehensive debugging capabilities for WASM applications, including verbose tracing, step-by-step playback, and breakpoint support.

## Debug Mode Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      DEBUG MODE                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │                    Event Sources                           │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │   │
│  │  │ HTTP    │  │ File    │  │ State   │  │ WASM    │       │   │
│  │  │ Request │  │ Change  │  │ Machine │  │ Memory  │       │   │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘       │   │
│  │       └───────────┬┴───────────┴──────────────┘           │   │
│  │                   ▼                                        │   │
│  │           ┌───────────────┐                                │   │
│  │           │  Debug Tracer │                                │   │
│  │           └───────┬───────┘                                │   │
│  │                   │                                        │   │
│  │       ┌───────────┼───────────┐                           │   │
│  │       ▼           ▼           ▼                           │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐                      │   │
│  │  │ Console │ │ Log     │ │ Break-  │                      │   │
│  │  │ Output  │ │ File    │ │ points  │                      │   │
│  │  └─────────┘ └─────────┘ └─────────┘                      │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                   │
│  Verbosity Levels:                                               │
│    minimal → normal → verbose → trace                            │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Enable debug mode
probador serve --debug [PATH]

# Debug with step-by-step playback
probador test --debug --step playbook.yaml

# Debug with breakpoints
probador test --debug --break-on "state=recording" playbook.yaml
```

## Debug Output

When debug mode is enabled, you see detailed information about every operation:

```
DEBUG MODE ACTIVE
━━━━━━━━━━━━━━━━━

[14:23:45.123] SERVER │ Binding to 127.0.0.1:8080
[14:23:45.125] SERVER │ Registered routes:
                      │   GET /demos/realtime-transcription/ -> index.html
                      │   GET /demos/realtime-transcription/pkg/* -> static
                      │   GET /demos/realtime-transcription/models/* -> static
[14:23:45.130] SERVER │ CORS headers: enabled (Access-Control-Allow-Origin: *)
[14:23:45.131] SERVER │ COOP/COEP headers: enabled (SharedArrayBuffer support)

[14:23:46.001] REQUEST │ GET /demos/realtime-transcription/
                       │ Client: 127.0.0.1:52341
                       │ User-Agent: Chrome/120.0
[14:23:46.002] RESOLVE │ Path: /demos/realtime-transcription/
                       │ Resolved: /home/user/project/demos/index.html
                       │ Rule: Directory index (index.html)
[14:23:46.003] RESPONSE│ Status: 200 OK
                       │ Content-Type: text/html
                       │ Content-Length: 2345
                       │ Latency: 2ms
```

## Verbosity Levels

| Level | Flag | Shows |
|-------|------|-------|
| Minimal | `-q` | Errors only |
| Normal | (default) | Errors + warnings |
| Verbose | `-v` | All requests/responses |
| Trace | `-vvv` | Everything including internal state |

```bash
# Minimal (errors only)
probador serve -q --debug

# Verbose
probador serve -v --debug

# Trace (maximum detail)
probador serve -vvv --debug
```

## Error Debugging

Debug mode provides detailed error information with suggestions:

```
[14:23:46.100] ERROR   │ GET /demos/realtime-transcription/models/whisper-tiny.apr
                       │ Error: File not found
                       │ Searched paths:
                       │   1. /home/user/project/demos/models/whisper-tiny.apr
                       │   2. /home/user/project/models/whisper-tiny.apr (fallback)
                       │ Suggestion: Model file missing. Download with:
                       │   curl -o demos/models/whisper-tiny.apr \
                       │        https://models.example.com/tiny.apr
```

## Step-by-Step Playback

Debug state machine transitions one step at a time:

```bash
probador test --debug --step playbook.yaml
```

### Interactive Output

```
STEP-BY-STEP PLAYBACK: realtime-transcription.yaml
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

State: initializing
Invariants:
  ✓ !can_start_recording() [Start button disabled]
  ✓ !can_stop_recording()  [Stop button disabled]

Press [Enter] to trigger 'wasm_ready' event, or [q] to quit...

─────────────────────────────────────────────────────
Transition: init_to_loading
  Event: wasm_ready
  From: initializing -> To: loading_model
─────────────────────────────────────────────────────

State: loading_model
Invariants:
  ✓ has_element('.loading-spinner') [Loading indicator visible]

Press [Enter] to trigger 'model_loaded' event, or [q] to quit...
```

## Breakpoints

Pause execution at specific points:

```bash
# Break when entering a state
probador test --debug --break-on "state=recording" playbook.yaml

# Break when an event fires
probador test --debug --break-on "event=wasm_ready" playbook.yaml

# Break on matching HTTP requests
probador serve --debug --break-on "request=/api/*"

# Break on any error
probador test --debug --break-on "error" playbook.yaml
```

### Breakpoint Types

| Type | Syntax | Example |
|------|--------|---------|
| State | `state=<name>` | `--break-on "state=recording"` |
| Event | `event=<name>` | `--break-on "event=model_loaded"` |
| Request | `request=<pattern>` | `--break-on "request=/api/*"` |
| Error | `error` | `--break-on "error"` |

## Debug Log File

Write debug output to a file:

```bash
probador serve --debug --log debug.log
```

The log file contains structured output:

```
2024-12-14T14:23:45.123Z DEBUG [server] Binding to 127.0.0.1:8080
2024-12-14T14:23:46.001Z DEBUG [request] GET /demos/index.html
2024-12-14T14:23:46.002Z DEBUG [resolve] Path resolved: /home/user/demos/index.html
2024-12-14T14:23:46.003Z DEBUG [response] 200 OK, 2345 bytes, 2ms
```

## CORS Debugging

Debug mode highlights CORS-related issues:

```
[14:23:46.050] REQUEST │ GET /api/data (preflight OPTIONS)
[14:23:46.051] CORS    │ Origin: http://localhost:3000
                       │ Method: POST
                       │ Headers: Content-Type, X-Custom-Header
[14:23:46.052] CORS    │ ⚠ Missing header in allowed list: X-Custom-Header
                       │ Add with: --cors-headers "X-Custom-Header"
```

## SharedArrayBuffer Debugging

Debug mode shows when COOP/COEP headers are needed:

```
[14:23:46.100] WASM    │ Loading: realtime_wasm_bg.wasm
[14:23:46.150] WASM    │ ⚠ SharedArrayBuffer requested but COOP/COEP not enabled
                       │ WASM threading requires these headers.
                       │ Enable with: --coop-coep
                       │ Or add to config:
                       │   [serve]
                       │   coop_coep = true
```

## Memory Profiling

Track WASM linear memory usage:

```bash
probador serve --debug --memory-profile --threshold 100MB
```

Output:
```
MEMORY PROFILE: realtime_wasm
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Initial heap: 16MB
Peak heap: 147MB (at t=12.3s during model load)
Current heap: 89MB

Growth events:
  t=0.5s:   16MB -> 32MB  (+16MB) [model initialization]
  t=2.1s:   32MB -> 64MB  (+32MB) [audio buffer allocation]
  t=12.3s:  64MB -> 147MB (+83MB) [inference tensors]
  t=14.1s: 147MB -> 89MB  (-58MB) [tensor deallocation]

⚠ Threshold alert: Peak (147MB) exceeded threshold (100MB)
```

## CLI Reference

```bash
# Debug serve command
probador serve --debug [OPTIONS] [PATH]

Debug Options:
      --debug              Enable debug mode
      --log <FILE>         Write debug log to file
      --break-on <COND>    Set breakpoint condition
      --memory-profile     Track WASM memory usage
      --threshold <SIZE>   Memory alert threshold

# Debug test command
probador test --debug [OPTIONS] <PLAYBOOK>

Debug Options:
      --debug              Enable debug mode
      --step               Step-by-step playback
      --break-on <COND>    Set breakpoint condition
      --log <FILE>         Write debug log to file
```

## Programmatic API

```rust
use probador::{DebugConfig, DebugVerbosity, Breakpoint};

let debug_config = DebugConfig {
    enabled: true,
    verbosity: DebugVerbosity::Verbose,
    step_mode: false,
    breakpoints: vec![
        Breakpoint::State("recording".into()),
        Breakpoint::Error,
    ],
    log_file: Some("debug.log".into()),
};
```

## Best Practices

1. **Start with verbose mode** - Use `-v` to see what's happening
2. **Use step mode for state machines** - `--step` helps understand transitions
3. **Set breakpoints for specific issues** - Target the problem area
4. **Check CORS/COEP early** - Common source of WASM issues
5. **Monitor memory for long-running apps** - Catch leaks early
6. **Save debug logs for CI failures** - `--log debug.log` for later analysis

## Common Debug Scenarios

### WASM Won't Load

```bash
probador serve --debug -vvv
```

Look for:
- MIME type issues (`application/wasm` required)
- CORS errors
- Missing COOP/COEP headers

### State Machine Stuck

```bash
probador test --debug --step playbook.yaml
```

Check:
- Which state is current
- What events are expected
- Which invariants are failing

### Memory Issues

```bash
probador serve --debug --memory-profile --threshold 50MB
```

Monitor:
- Initial vs peak memory
- Growth patterns
- Deallocation behavior
