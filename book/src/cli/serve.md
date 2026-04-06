# Serve Command

The `serve` command starts the Realizar HTTP inference server, making your ML model available via REST API.

## Basic Usage

```bash
# Start server in demo mode (for testing)
realizar serve --demo

# Start server on custom port
realizar serve --demo --port 8080

# Start server on custom host and port
realizar serve --demo --host 0.0.0.0 --port 3000
```

## Command Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--host HOST` | `-H` | Host address to bind to | `127.0.0.1` |
| `--port PORT` | `-p` | Port number to bind to | `8080` |
| `--demo` | | Use demo model for testing | (required) |
| `--help` | `-h` | Print help information | |

## Demo Mode

Currently, Realizar supports demo mode which provides a minimal working server for testing the API:

```bash
$ realizar serve --demo
🚀 Realizar server starting...
📍 Host: 127.0.0.1
🔌 Port: 8080
🎯 Mode: Demo
✅ Server ready at http://127.0.0.1:8080
```

The demo mode includes:
- ✅ Health check endpoint (`/health`)
- ✅ Tokenization endpoint (`/tokenize`)
- ✅ Text generation endpoint (`/generate`)

## Example: Starting the Server

```bash
# Build the binary (if not already built)
cargo build --release

# Start the server in demo mode
./target/release/realizar serve --demo --port 8080
```

You should see output like:
```
🚀 Realizar server starting...
📍 Host: 127.0.0.1
🔌 Port: 8080
🎯 Mode: Demo
✅ Server ready at http://127.0.0.1:8080
```

## Testing the Server

Once the server is running, you can test it using `curl`:

### Health Check

```bash
curl http://127.0.0.1:8080/health
```

Response:
```json
{"status":"ok"}
```

### Tokenize Text

```bash
curl -X POST http://127.0.0.1:8080/tokenize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'
```

Response:
```json
{"tokens":[7,4,11,11,14,0,22,14,17,11,3]}
```

### Generate Text

```bash
curl -X POST http://127.0.0.1:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello",
    "max_tokens": 10,
    "strategy": "greedy"
  }'
```

Response:
```json
{
  "text": "Hello world this is",
  "num_generated": 4
}
```

## Network Configuration

### Localhost Only (Default)

```bash
realizar serve --demo --host 127.0.0.1 --port 8080
```

This binds to localhost only - the server is only accessible from the same machine.

### All Interfaces (External Access)

```bash
realizar serve --demo --host 0.0.0.0 --port 8080
```

⚠️ **Warning**: Binding to `0.0.0.0` makes the server accessible from any network interface. Only use this if you understand the security implications.

## Common Issues

### Port Already in Use

If you see an error like "Address already in use", another service is using that port:

```bash
# Use a different port
realizar serve --demo --port 8081
```

Or find and stop the process using the port:

```bash
# On Linux/macOS
lsof -i :8080
kill <PID>

# On Windows
netstat -ano | findstr :8080
taskkill /PID <PID> /F
```

### Permission Denied (Ports < 1024)

Ports below 1024 require root/administrator privileges:

```bash
# Don't use privileged ports
realizar serve --demo --port 8080  # ✅ Works

# Requires sudo/admin
realizar serve --demo --port 80    # ❌ Permission denied
```

## Integration with Development Workflow

### During Development

```bash
# Terminal 1: Run server
cargo run --release -- serve --demo

# Terminal 2: Test API
curl http://127.0.0.1:8080/health
```

### With Hot Reload

For development with automatic reloading, you can use `cargo watch`:

```bash
# Install cargo-watch
cargo install cargo-watch

# Run with auto-reload
cargo watch -x 'run --release -- serve --demo'
```

## Next Steps

- [API Endpoints](../api/endpoints.md) - Complete API documentation
- [Generate Endpoint](../api/generate.md) - Text generation parameters
- [Tokenize Endpoint](../api/tokenize.md) - Tokenization details
- [Testing HTTP Endpoints](../api/testing.md) - How to test the API

## See Also

- [Info Command](./info.md) - Display system information
- [Command Structure](./command-structure.md) - CLI architecture
- [CLI Testing](./testing.md) - Testing the CLI
