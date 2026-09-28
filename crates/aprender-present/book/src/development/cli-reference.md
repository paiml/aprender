# CLI Reference

The `presentar` CLI provides commands for development, building, and deployment.

## Installation

```bash
cargo install presentar-cli
# Or from source
cargo install --path crates/presentar-cli
```

## Commands

### `aprender-present serve`

Start a development server with hot reload.

```bash
aprender-present serve [OPTIONS]

Options:
  -p, --port <PORT>    Port to serve on [default: 8080]
  -d, --dir <DIR>      Directory to serve [default: www]
  -w, --watch          Watch for changes and rebuild
```

**Example:**
```bash
aprender-present serve --port 3000 --watch
```

When `--watch` is enabled:
- Rust changes trigger WASM rebuild
- YAML changes are validated and trigger hot reload
- HTML/CSS/JS changes trigger instant reload
- WebSocket server runs on port 35729

### `aprender-present bundle`

Build an optimized WASM bundle for production.

```bash
aprender-present bundle [OPTIONS]

Options:
  -o, --output <DIR>   Output directory [default: dist]
  --no-optimize        Skip wasm-opt optimization
```

**Example:**
```bash
aprender-present bundle --output ./release
```

Output includes:
- `pkg/presentar_bg.wasm` - Optimized WASM binary
- `pkg/presentar.js` - JavaScript bindings
- `index.html` - Copied from www/

### `aprender-present new`

Create a new Presentar project.

```bash
aprender-present new <NAME>
```

**Example:**
```bash
aprender-present new my-dashboard
cd my-dashboard
aprender-present serve
```

Creates:
- `app.yaml` - Starter manifest
- `www/index.html` - HTML template

### `aprender-present check`

Validate a YAML manifest.

```bash
aprender-present check [MANIFEST]

Arguments:
  [MANIFEST]  Path to manifest file [default: app.yaml]
```

**Example:**
```bash
aprender-present check app.yaml
# Output: Manifest valid!
#   Name: my-dashboard
#   Version: 1.0.0
#   Data sources: 2
#   Sections: 3
```

### `aprender-present score`

Compute quality score for a manifest.

```bash
aprender-present score [OPTIONS] [MANIFEST]

Arguments:
  [MANIFEST]  Path to manifest file [default: app.yaml]

Options:
  -f, --format <FMT>   Output format: text, json, badge [default: text]
  --badge <FILE>       Output file for SVG badge
```

**Example:**
```bash
aprender-present score --format json
# {"score": 85.0, "grade": "A", ...}

aprender-present score --badge quality.svg
```

Score breakdown:
| Category | Max Points |
|----------|------------|
| Structural | 25 |
| Performance | 20 |
| Accessibility | 20 |
| Data Quality | 15 |
| Documentation | 10 |
| Consistency | 10 |

### `aprender-present gate`

Run quality gates validation.

```bash
aprender-present gate [OPTIONS] [MANIFEST]

Arguments:
  [MANIFEST]  Path to manifest file [default: app.yaml]

Options:
  -g, --min-grade <GRADE>   Minimum passing grade [default: B]
  -s, --min-score <SCORE>   Minimum score (0-100)
  --strict                  Fail on any warning
```

**Example:**
```bash
# CI pipeline check
aprender-present gate --min-grade B --strict app.yaml
# Exit code 0 = pass, 1 = fail
```

### `aprender-present deploy`

Deploy to cloud hosting.

```bash
aprender-present deploy [OPTIONS]

Options:
  -s, --source <DIR>        Source directory [default: dist]
  -t, --target <TARGET>     Target: s3, cloudflare, vercel, netlify, local
  -b, --bucket <BUCKET>     S3 bucket name (required for S3)
  --distribution <ID>       CloudFront distribution ID
  --region <REGION>         AWS region [default: us-east-1]
  --dry-run                 Show what would be deployed
  --skip-build              Deploy existing files
```

**Examples:**
```bash
# Deploy to S3 with CloudFront
aprender-present deploy \
  --target s3 \
  --bucket my-app.example.com \
  --distribution EXXXXXXXXXXXXX \
  --region us-west-2

# Dry run to see what would happen
aprender-present deploy --target s3 --bucket my-bucket --dry-run

# Deploy to Cloudflare Pages
aprender-present deploy --target cloudflare

# Deploy to Vercel
aprender-present deploy --target vercel

# Deploy to Netlify
aprender-present deploy --target netlify
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `AWS_ACCESS_KEY_ID` | AWS credentials for S3 deploy |
| `AWS_SECRET_ACCESS_KEY` | AWS credentials for S3 deploy |
| `CLOUDFLARE_API_TOKEN` | Cloudflare API token |
| `VERCEL_TOKEN` | Vercel authentication token |
| `NETLIFY_AUTH_TOKEN` | Netlify authentication token |

## Makefile Integration

The Makefile wraps CLI commands:

```makefile
dev:
    aprender-present serve --watch

build:
    aprender-present bundle

deploy:
    aprender-present deploy --target s3 --bucket $(BUCKET)
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error or gate failure |
| 2 | Invalid arguments |
