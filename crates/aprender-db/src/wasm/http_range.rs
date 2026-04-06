//! HTTP Range Request client for streaming Parquet files in browsers.
//!
//! Implements RFC 7233 byte-range requests to enable streaming large datasets
//! without exceeding browser memory limits (~2GB).
//!
//! # Architecture
//!
//! ```text
//! Browser
//!    ↓
//! fetch_range("bytes=0-1023")
//!    ↓
//! CDN/S3 Server (206 Partial Content)
//!    ↓
//! Vec<u8> chunk
//! ```
//!
//! # References
//! - RFC 7233: HTTP Range Requests
//! - DuckDB-WASM: Production implementation pattern

#![cfg(target_arch = "wasm32")]

use js_sys::Uint8Array;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Headers, Request, RequestInit, Response};

/// Byte range for HTTP range requests
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteRange {
    /// Starting byte (inclusive)
    pub start: u64,
    /// Ending byte (inclusive)
    pub end: u64,
}

impl ByteRange {
    /// Create a new byte range
    pub fn new(start: u64, end: u64) -> Self {
        assert!(start <= end, "start must be <= end");
        Self { start, end }
    }

    /// Get the size of this range in bytes
    pub fn size(&self) -> u64 {
        self.end - self.start + 1
    }

    /// Create range for last N bytes of file
    pub fn last_n_bytes(n: u64) -> Self {
        Self {
            start: u64::MAX - n + 1, // Will be converted to suffix-range
            end: u64::MAX,
        }
    }
}

impl std::fmt::Display for ByteRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "bytes={}-{}", self.start, self.end)
    }
}

/// HTTP range request client
pub struct RangeClient {
    base_url: String,
    retry_attempts: u32,
}

impl RangeClient {
    /// Create new range client for given URL
    pub fn new(url: impl Into<String>) -> Self {
        Self { base_url: url.into(), retry_attempts: 3 }
    }

    /// Fetch a byte range from the remote file
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Network request fails
    /// - Server doesn't support range requests (no 206 response)
    /// - Server returns error status
    pub async fn fetch_range(&self, range: ByteRange) -> Result<Vec<u8>, JsValue> {
        // Prepare request with Range header
        let mut opts = RequestInit::new();
        opts.method("GET");

        let headers = Headers::new()?;
        headers.set("Range", &range.to_string())?;

        let request = Request::new_with_str_and_init(&self.base_url, &opts)?;
        request.headers().set("Range", &range.to_string())?;

        // Execute fetch with retry logic
        for attempt in 0..self.retry_attempts {
            match self.try_fetch(&request).await {
                Ok(bytes) => return Ok(bytes),
                Err(e) if attempt < self.retry_attempts - 1 => {
                    // Exponential backoff: 100ms, 200ms, 400ms
                    let delay_ms = 100 * 2_u32.pow(attempt);
                    sleep_ms(delay_ms).await;
                    continue;
                }
                Err(e) => return Err(e),
            }
        }

        Err(JsValue::from_str("Max retries exceeded"))
    }

    /// Try to fetch once (no retry)
    async fn try_fetch(&self, request: &Request) -> Result<Vec<u8>, JsValue> {
        let window = web_sys::window().ok_or_else(|| JsValue::from_str("No window object"))?;

        // Execute fetch
        let response_promise = window.fetch_with_request(request);
        let response_value = JsFuture::from(response_promise).await?;
        let response: Response = response_value.dyn_into()?;

        // Check status
        let status = response.status();
        if status != 206 && status != 200 {
            return Err(JsValue::from_str(&format!("HTTP error: {}", status)));
        }

        // Read response body
        let array_buffer_promise = response.array_buffer()?;
        let array_buffer = JsFuture::from(array_buffer_promise).await?;
        let uint8_array = Uint8Array::new(&array_buffer);

        Ok(uint8_array.to_vec())
    }

    /// Get file size using HEAD request
    pub async fn get_file_size(&self) -> Result<u64, JsValue> {
        let window = web_sys::window().ok_or_else(|| JsValue::from_str("No window object"))?;

        let mut opts = RequestInit::new();
        opts.method("HEAD");

        let request = Request::new_with_str_and_init(&self.base_url, &opts)?;
        let response_promise = window.fetch_with_request(&request);
        let response_value = JsFuture::from(response_promise).await?;
        let response: Response = response_value.dyn_into()?;

        // Parse Content-Length header
        let headers = response.headers();
        let content_length_js = headers.get("content-length")?;

        match content_length_js {
            Some(length_str) => length_str
                .parse::<u64>()
                .map_err(|e| JsValue::from_str(&format!("Invalid content-length: {}", e))),
            None => Err(JsValue::from_str("No content-length header")),
        }
    }
}

/// Sleep for N milliseconds (WASM-compatible)
async fn sleep_ms(ms: u32) {
    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        let window = web_sys::window().unwrap();
        window.set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, ms as i32).unwrap();
    });

    let _ = JsFuture::from(promise).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_range_new() {
        let range = ByteRange::new(0, 1023);
        assert_eq!(range.start, 0);
        assert_eq!(range.end, 1023);
        assert_eq!(range.size(), 1024);
    }

    #[test]
    fn test_byte_range_display() {
        let range = ByteRange::new(100, 199);
        assert_eq!(range.to_string(), "bytes=100-199");
    }

    #[test]
    fn test_byte_range_size() {
        let range = ByteRange::new(0, 0);
        assert_eq!(range.size(), 1);

        let range = ByteRange::new(0, 1023);
        assert_eq!(range.size(), 1024);
    }

    #[test]
    #[should_panic(expected = "start must be <= end")]
    fn test_byte_range_invalid() {
        ByteRange::new(100, 50);
    }
}
