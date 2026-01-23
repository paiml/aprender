# HuggingFace Hub Upload Specification

**Version:** 1.0.0
**Status:** In Progress
**Author:** paiml
**Date:** 2026-01-23
**Bug ID:** APR-PUB-001

## 1. Problem Statement

### 1.1 Root Cause Analysis (Toyota Way: Genchi Genbutsu)

The `push_to_hub()` function in `src/hf_hub/mod.rs` is a **STUB** that does not actually upload files to HuggingFace Hub. The function:

1. Saves files locally to cache directory
2. Prints a message saying "HTTP upload not yet implemented"
3. Returns success despite not uploading

**Evidence (lines 323-356):**
```rust
// Note: Actual upload requires multipart form data
// This is a simplified implementation that returns the expected URL
// A production implementation would use reqwest or similar

// For demonstration, we'll save locally and return info
```

### 1.2 Impact

- **Goldfish Bug**: Models train successfully but are not persisted to HuggingFace
- **Silent Failure**: Function returns `Ok()` despite not uploading
- **User Deception**: Message suggests success when upload never occurred
- **External Verification Failure**: Published models fail audit

### 1.3 Toyota Way Violations

| Principle | Violation |
|-----------|-----------|
| **Jidoka** | No built-in quality check for upload success |
| **Andon Cord** | Silent failure instead of stopping on error |
| **Genchi Genbutsu** | Stub code released without going to the gemba |
| **Respect for People** | User time wasted on failed publishes |

## 2. Technical Specification

### 2.1 HuggingFace Hub API Requirements

The HF Hub uses Git-LFS for large file storage. For files > 10MB:

1. **Create Repo** (if needed): `POST /api/repos/create`
2. **Request LFS Upload**: `POST /api/{repo_type}s/{repo_id}/preupload/{revision}`
3. **Upload via LFS**: `PUT` to LFS server with chunked transfer
4. **Commit Files**: `POST /api/{repo_type}s/{repo_id}/commit/{revision}`

### 2.2 Implementation Requirements

| Requirement | Description | Priority |
|-------------|-------------|----------|
| **REQ-001** | Implement actual HTTP upload to HF Hub API | P0 |
| **REQ-002** | Support large files (>5GB) via chunked upload | P0 |
| **REQ-003** | Progress tracking callback for uploads | P0 |
| **REQ-004** | Retry logic with exponential backoff | P1 |
| **REQ-005** | Resume interrupted uploads | P1 |
| **REQ-006** | Verify upload success via API | P0 |

### 2.3 API Endpoints

```
Base: https://huggingface.co

POST /api/repos/create
  - Create new repository
  - Headers: Authorization: Bearer {token}
  - Body: { "type": "model", "name": "{repo_id}", "private": bool }

POST /api/models/{repo_id}/preupload/main
  - Request LFS upload URL for large files
  - Headers: Authorization: Bearer {token}
  - Body: { "files": [{ "path": "model.safetensors", "size": N, "sample": "base64..." }] }

POST /api/models/{repo_id}/commit/main
  - Commit files to repository
  - Headers: Authorization: Bearer {token}
  - Body: multipart form with files and commit info
```

### 2.4 Dependencies

Add `ureq` as direct dependency (already transitive via `hf-hub`):

```toml
[dependencies]
ureq = { version = "2.12", optional = true, features = ["json"] }
presentar = { version = "0.3", optional = true }  # Terminal feedback
renacer = { version = "0.9", optional = true }    # Syscall/network tracing

[features]
hf-hub-integration = ["hf-hub", "dirs", "ureq", "presentar", "renacer"]
```

### 2.5 Observability Requirements (MANDATORY)

**Toyota Way: Genchi Genbutsu** - We MUST see what is actually happening, not guess.

| Requirement | Tool | Description |
|-------------|------|-------------|
| **OBS-001** | `presentar` | Real-time terminal progress bars for uploads |
| **OBS-002** | `renacer` | Network syscall tracing for HTTP requests |
| **OBS-003** | `trueno-viz` | Optional SVG timeline visualization |
| **OBS-004** | Structured logs | JSON-formatted events for all API calls |

#### 2.5.1 Tracing Events (Required)

Every HTTP operation MUST emit structured trace events:

```rust
use renacer::trace;

// Before request
trace::span!("hf_hub.preupload", repo_id = %repo_id, file = %filename, size = data.len());

// API response
trace::event!(Level::INFO, "hf_hub.preupload.response",
    status = response.status(),
    upload_url = ?upload_url,
    elapsed_ms = elapsed.as_millis()
);

// Error case
trace::event!(Level::ERROR, "hf_hub.upload.failed",
    error = %e,
    retry_attempt = attempt,
    backoff_ms = backoff_ms
);
```

#### 2.5.2 Terminal Progress (Required)

Use `presentar` for user-facing progress:

```rust
use presentar::{ProgressBar, MultiProgress, Style};

let multi = MultiProgress::new();
let pb = multi.add(ProgressBar::new(total_bytes));
pb.set_style(Style::default()
    .template("{spinner} [{bar:40}] {bytes}/{total_bytes} ({eta}) {msg}")
    .progress_chars("█▓░"));

// During upload
pb.set_position(bytes_sent);
pb.set_message(format!("Uploading {}", filename));

// On complete
pb.finish_with_message("✓ Upload complete");
```

#### 2.5.3 Debug Mode

When `--verbose` or `RUST_LOG=debug`:
- Print full HTTP request/response headers
- Print preupload API response JSON
- Print LFS upload URL
- Print commit API response

**NO GUESSING** - All failures must have clear diagnostic output.

## 3. Test Plan (Extreme TDD)

### 3.1 Unit Tests

| Test | Description |
|------|-------------|
| `test_upload_requires_token` | Upload fails with MissingToken if no auth |
| `test_upload_validates_repo_id` | Invalid repo ID format rejected |
| `test_upload_creates_repo_if_needed` | create_repo=true creates new repo |
| `test_upload_progress_callback` | Progress callback invoked with correct values |
| `test_upload_retry_on_network_error` | Retries with backoff on 5xx errors |
| `test_upload_verifies_success` | Verifies file exists after upload |

### 3.2 Integration Tests (Mocked)

| Test | Description |
|------|-------------|
| `test_full_upload_flow` | Mock server: create → preupload → upload → commit |
| `test_large_file_chunked` | Mock server: chunked upload for >10MB file |
| `test_upload_with_model_card` | README.md uploaded alongside model |

### 3.3 Property-Based Tests

```rust
proptest! {
    #[test]
    fn prop_progress_monotonic(chunks in 1usize..100) {
        // Progress should always increase
    }

    #[test]
    fn prop_retry_respects_max_attempts(attempts in 1usize..10) {
        // Should not exceed max retry attempts
    }
}
```

## 4. Implementation

### 4.1 New Types

```rust
/// Progress callback for upload tracking
pub type ProgressCallback = Box<dyn Fn(UploadProgress) + Send>;

/// Upload progress information
#[derive(Debug, Clone)]
pub struct UploadProgress {
    pub bytes_sent: u64,
    pub total_bytes: u64,
    pub current_file: String,
    pub files_completed: usize,
    pub total_files: usize,
}

/// Upload configuration with retry settings
#[derive(Debug, Clone)]
pub struct UploadConfig {
    pub max_retries: usize,
    pub initial_backoff_ms: u64,
    pub max_backoff_ms: u64,
    pub chunk_size: usize,
    pub progress_callback: Option<ProgressCallback>,
}
```

### 4.2 Updated push_to_hub Signature

```rust
pub fn push_to_hub(
    &self,
    repo_id: &str,
    model_data: &[u8],
    options: PushOptions,
) -> Result<UploadResult>
```

### 4.3 New UploadResult Type

```rust
#[derive(Debug)]
pub struct UploadResult {
    pub repo_url: String,
    pub commit_sha: String,
    pub files_uploaded: Vec<String>,
    pub bytes_transferred: u64,
}
```

## 5. Acceptance Criteria

### 5.1 Functional Requirements
- [x] `push_to_hub()` actually uploads files to HuggingFace Hub
- [ ] Large files (>5GB) require sharding (see Known Limitations)
- [x] Progress tracking works for all file sizes
- [x] Upload failures return `Err()`, not silent `Ok()`
- [ ] Tests achieve >95% coverage of new code
- [ ] Property-based tests pass for edge cases
- [ ] External verification audit passes

### 5.4 Known Limitations

**HuggingFace Hub 5GB File Size Limit**

Files larger than 5GB cannot be uploaded via the pure HTTP API. This is a HuggingFace Hub platform limitation:

> "You need to configure your repository to enable upload of files > 5GB."

HuggingFace requires using their `lfs-multipart-upload` git transfer agent for files > 5GB, which is part of `huggingface-cli` and not available as a pure HTTP API.

**Workarounds:**

1. **Shard the model** (Recommended): Split safetensors files into chunks < 5GB
   ```bash
   apr export --max-shard-size 4GB /path/to/model
   ```

2. **Use huggingface-cli**: For one-time uploads of large models
   ```bash
   git clone https://huggingface.co/{repo_id}
   huggingface-cli lfs-enable-largefiles ./
   cp model.safetensors ./
   git add . && git commit -m "Add model" && git push
   ```

**References:**
- [HuggingFace Storage Limits](https://huggingface.co/docs/hub/en/storage-limits)
- [Repository Recommendations](https://huggingface.co/docs/hub/repositories-recommendations)

### 5.2 Observability Requirements (MANDATORY)
- [ ] **OBS-001**: `presentar` progress bars show real-time upload status
- [ ] **OBS-002**: All HTTP requests logged with `renacer` tracing
- [ ] **OBS-003**: Preupload API response visible in verbose mode
- [ ] **OBS-004**: LFS upload URL printed when uploading
- [ ] **OBS-005**: Commit API response visible in verbose mode
- [ ] **OBS-006**: Error messages include full context (status, body, headers)
- [ ] **OBS-007**: `--verbose` flag enables detailed diagnostics
- [ ] **OBS-008**: No eprintln/println guesswork - all output structured

### 5.3 Failure Mode Requirements
When upload fails, output MUST include:
1. HTTP status code
2. Response body (truncated to 1KB)
3. Request URL
4. Retry attempt number
5. Time elapsed
6. Suggested fix or next step

## 6. Rollout Plan

1. Implement upload with tests (this spec)
2. Test with small model (<100MB)
3. Test with large model (5.8GB Qwen2)
4. Update apr-cli to use new upload
5. Re-publish corpus model
6. Run external verification audit

## 7. References

- [HuggingFace Hub API Documentation](https://huggingface.co/docs/hub/api)
- [Git LFS Specification](https://github.com/git-lfs/git-lfs/blob/main/docs/spec.md)
- [Toyota Way Principles](https://en.wikipedia.org/wiki/Toyota_Production_System)
