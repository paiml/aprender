# Development Server

![Dev Server Coverage](../assets/coverage_viridis.png)

The `probador serve` command provides a full-featured development server for WASM applications with hot reload, file visualization, and content linting.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROBADOR DEV SERVER                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐      │
│  │   HTTP        │   │   WebSocket   │   │   File        │      │
│  │   Server      │   │   Server      │   │   Watcher     │      │
│  │   (axum)      │   │   (tungstenite)│  │   (notify)    │      │
│  └───────┬───────┘   └───────┬───────┘   └───────┬───────┘      │
│          │                   │                   │               │
│          └───────────────────┼───────────────────┘               │
│                              ▼                                   │
│                    ┌─────────────────┐                           │
│                    │  Event Router   │                           │
│                    └────────┬────────┘                           │
│                             │                                    │
│         ┌───────────────────┼───────────────────┐               │
│         ▼                   ▼                   ▼               │
│  ┌────────────┐     ┌────────────┐     ┌────────────┐           │
│  │  Static    │     │  Hot       │     │  Content   │           │
│  │  Files     │     │  Reload    │     │  Linting   │           │
│  └────────────┘     └────────────┘     └────────────┘           │
│                                                                   │
│  Headers: CORS, COOP/COEP (SharedArrayBuffer support)            │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Serve current directory
probador serve

# Serve specific directory on custom port
probador serve ./www --port 3000

# Enable CORS for cross-origin requests
probador serve --cors

# Open browser automatically
probador serve --open

# Full development setup
probador serve ./dist --port 8080 --cors --open
```

## File Tree Visualization

See exactly what files are being served:

```bash
# ASCII tree output
probador serve tree [PATH]

# With depth limit
probador serve tree --depth 2

# Filter by pattern
probador serve tree --filter "*.wasm"
```

### Example Output

```
demos/realtime-transcription/
├── index.html (2.3 KB) [text/html]
├── styles.css (1.1 KB) [text/css]
├── pkg/
│   ├── realtime_wasm.js (45 KB) [text/javascript]
│   ├── realtime_wasm_bg.wasm (1.2 MB) [application/wasm]
│   └── realtime_wasm.d.ts (3.2 KB) [text/typescript]
├── models/
│   └── whisper-tiny.apr (39 MB) [application/octet-stream]
└── worker.js (5.6 KB) [text/javascript]

Total: 8 files, 41.3 MB
Served at: http://localhost:8080/demos/realtime-transcription/
```

## Hot Reload

Automatic browser refresh when files change:

```bash
# Enable hot reload (default)
probador serve --watch [PATH]

# Disable hot reload
probador serve --no-watch [PATH]

# Verbose change reporting
probador serve --watch --verbose [PATH]
```

### Hot Reload Display

```
HOT RELOAD ACTIVE - Watching demos/realtime-transcription/
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

14:23:45.123 │ MODIFIED │ index.html        │ +56 bytes │ 3 clients notified
14:23:47.891 │ MODIFIED │ styles.css        │ -12 bytes │ 3 clients notified
14:23:52.001 │ CREATED  │ new-component.js  │ 1.2 KB    │ 3 clients notified
14:24:01.555 │ DELETED  │ old-helper.js     │ -         │ 3 clients notified

Connected clients: 3 │ Files watched: 42 │ Reload count: 4
```

### WebSocket Protocol

Connected browsers receive JSON messages:

```json
{
  "type": "file_change",
  "event": "modified",
  "path": "demos/realtime-transcription/index.html",
  "timestamp": 1702567890123,
  "size_before": 2345,
  "size_after": 2401,
  "diff_summary": "+56 bytes"
}
```

## Content Linting

Validate HTML, CSS, JavaScript, and WASM files:

```bash
# Lint on startup
probador serve --lint [PATH]

# Lint specific files
probador lint [--html] [--css] [--js] [PATH]

# Continuous lint on file change
probador serve --lint --watch [PATH]
```

### Supported File Types

| File Type | Checks |
|-----------|--------|
| HTML | Valid structure, missing attributes, broken links |
| CSS | Parse errors, unknown properties, specificity issues |
| JavaScript | Syntax errors, undefined references, module resolution |
| WASM | Valid module structure, import/export validation |
| JSON | Parse validity, schema validation (optional) |

### Lint Output

```
LINT REPORT: demos/realtime-transcription/
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

index.html:
  ✓ Valid HTML5 structure
  ⚠ Line 23: <img> missing alt attribute
  ✗ Line 45: Broken link: ./missing.css

styles.css:
  ✓ Valid CSS3
  ⚠ Line 12: Unknown property 'webkit-transform' (use -webkit-transform)

worker.js:
  ✓ Valid ES6 module
  ⚠ Line 8: 'wasm_url' used before assignment in some paths

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Summary: 0 errors, 3 warnings, 4 files checked
```

## CORS and Security Headers

### Enable CORS

```bash
probador serve --cors
```

Adds headers:
```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, POST, OPTIONS
Access-Control-Allow-Headers: Content-Type
```

### SharedArrayBuffer Support

For WASM applications that require `SharedArrayBuffer`:

```bash
probador serve --coop-coep
```

Adds headers:
```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

## MIME Type Handling

probador automatically serves files with correct MIME types:

| Extension | MIME Type |
|-----------|-----------|
| `.wasm` | `application/wasm` |
| `.js` | `text/javascript` |
| `.mjs` | `text/javascript` |
| `.html` | `text/html` |
| `.css` | `text/css` |
| `.json` | `application/json` |
| `.png` | `image/png` |
| `.svg` | `image/svg+xml` |

## CLI Reference

```bash
probador serve [OPTIONS] [PATH]

Arguments:
  [PATH]  Directory to serve [default: .]

Options:
  -p, --port <PORT>      HTTP port [default: 8080]
      --ws-port <PORT>   WebSocket port for hot reload [default: 8081]
      --cors             Enable CORS headers
      --coop-coep        Enable COOP/COEP for SharedArrayBuffer
      --watch            Enable hot reload [default: true]
      --no-watch         Disable hot reload
      --lint             Lint files on startup
      --open             Open browser automatically
  -v, --verbose          Verbose output
  -h, --help             Print help
```

### Tree Subcommand

```bash
probador serve tree [OPTIONS] [PATH]

Arguments:
  [PATH]  Directory to visualize [default: .]

Options:
      --depth <N>        Maximum depth to display
      --filter <GLOB>    Filter files by pattern
      --sizes            Show file sizes [default: true]
      --mime             Show MIME types [default: true]
  -h, --help             Print help
```

## Programmatic API

```rust
use probador::{DevServer, DevServerConfig};

// Create server configuration
let config = DevServerConfig::builder()
    .port(8080)
    .ws_port(8081)
    .cors(true)
    .coop_coep(true)
    .watch(true)
    .build();

// Start server
let server = DevServer::new(config);
server.serve("./www").await?;
```

## Integration with Watch Mode

Combine with build watching for full development workflow:

```bash
# Watch for changes and rebuild + serve
probador watch --serve --port 8080

# Equivalent to running both:
# probador watch ./src
# probador serve ./dist --port 8080
```

## Best Practices

1. **Use `--cors` during development** - Prevents cross-origin issues
2. **Enable `--coop-coep` for WASM threading** - Required for `SharedArrayBuffer`
3. **Use `--lint` to catch errors early** - Validates content on startup
4. **Check file tree before debugging** - `probador serve tree` shows exactly what's served
5. **Monitor hot reload output** - See which files trigger reloads
