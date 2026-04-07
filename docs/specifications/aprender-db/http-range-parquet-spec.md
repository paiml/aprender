# HTTP Range Request Parquet Reader Specification

**Work Item**: phase4-http-range
**Status**: In Progress
**Priority**: High
**Phase**: 4 (WASM Browser Deployment)

## Problem Statement

Browsers have ~2GB JavaScript heap limits. Loading entire Parquet files into memory causes OOM crashes for datasets >1GB. We need streaming reads with HTTP range requests to stay within browser memory constraints.

### Real-World Constraint (Genchi Genbutsu)
- Chrome: ~2GB heap limit (64-bit), ~512MB (32-bit)
- Firefox: ~2GB heap limit
- Safari: ~2GB heap limit
- Mobile browsers: ~200-500MB

## Solution: HTTP Range Request Streaming

### Architecture

```
Browser
   ↓
HTTP Range Request (bytes=0-1023)
   ↓
CDN/S3 (Parquet file)
   ↓
Streaming Decoder
   ↓
Late Materialization
   ↓
Query Results (<2GB)
```

### Key Techniques (Academic Foundations)

**1. HTTP Range Requests (RFC 7233)**
- Request specific byte ranges: `Range: bytes=0-1023`
- Server responds with `206 Partial Content`
- Enables streaming without full download

**2. Late Materialization (Abadi et al. 2008)**
- Defer row reconstruction until final results
- Process columnar data in-place
- Only materialize rows that pass filters

**3. Parquet Footer First (Apache Parquet Spec)**
- Parquet footer at end of file contains metadata
- Single range request for footer
- Subsequent requests for needed row groups

## Implementation Plan (TDD)

### Phase 1: HTTP Range Client (RED → GREEN → REFACTOR)

**RED: Write failing test**
```rust
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test]
async fn test_fetch_byte_range() {
    let url = "https://example.com/data.parquet";
    let range = ByteRange::new(0, 1023);
    let result = fetch_range(url, range).await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 1024);
}
```

**GREEN: Minimal implementation**
```rust
pub async fn fetch_range(url: &str, range: ByteRange) -> Result<Vec<u8>> {
    let window = web_sys::window().unwrap();
    let headers = Headers::new()?;
    headers.set("Range", &format!("bytes={}-{}", range.start, range.end))?;

    let response = JsFuture::from(window.fetch_with_str(url)).await?;
    let bytes = JsFuture::from(response.array_buffer()?).await?;
    Ok(Uint8Array::new(&bytes).to_vec())
}
```

**REFACTOR: Error handling, retries**

### Phase 2: Streaming Parquet Decoder

**Parquet File Structure**
```
[Magic Number] [Row Group 1] [Row Group 2] ... [Footer] [Footer Length] [Magic Number]
4 bytes        variable        variable         variable  4 bytes         4 bytes
```

**Implementation Steps:**

1. **Read Footer (RED)**
```rust
#[test]
async fn test_read_parquet_footer() {
    let reader = StreamingParquetReader::new("http://cdn/data.parquet");
    let metadata = reader.read_metadata().await?;
    assert!(metadata.row_groups().len() > 0);
}
```

2. **Read Row Group on Demand (GREEN)**
```rust
impl StreamingParquetReader {
    pub async fn read_row_group(&mut self, index: usize) -> Result<RecordBatch> {
        let row_group_meta = &self.metadata.row_groups()[index];
        let start = row_group_meta.file_offset();
        let length = row_group_meta.compressed_size();

        let bytes = self.fetch_range(start, start + length).await?;
        self.decode_row_group(bytes)
    }
}
```

3. **Predicate Pushdown (REFACTOR)**
- Skip row groups that don't match filter
- Use Parquet statistics (min/max per column)

### Phase 3: Late Materialization

**Key Insight**: Don't reconstruct full rows until needed

```rust
// BAD: Early materialization (wastes memory)
let all_rows = parquet.read_all_rows().await?;  // OOM!
let filtered = all_rows.filter(|r| r.price > 100);
let result = filtered.aggregate();

// GOOD: Late materialization
let price_column = parquet.read_column("price").await?;
let indices = price_column.filter_indices(|&v| v > 100);
let result = parquet.aggregate_by_indices("revenue", &indices).await?;
```

**Implementation (RED)**
```rust
#[test]
async fn test_late_materialization() {
    let reader = StreamingParquetReader::new(url);

    // Read only filter column
    let price = reader.read_column("price").await?;
    let indices = price.filter_indices(|&v| v > 100);

    // Only materialize matching rows
    let result = reader.select_by_indices(&["name", "revenue"], &indices).await?;

    assert!(result.memory_size() < 100_MB);
}
```

### Phase 4: Memory Budgeting (Poka-Yoke)

**Prevent OOM with memory tracking**

```rust
pub struct MemoryBudget {
    limit: usize,  // e.g., 1.5GB (leave headroom)
    current: Arc<AtomicUsize>,
}

impl MemoryBudget {
    pub fn try_allocate(&self, size: usize) -> Result<()> {
        let new_total = self.current.fetch_add(size, Ordering::SeqCst) + size;
        if new_total > self.limit {
            self.current.fetch_sub(size, Ordering::SeqCst);
            Err(Error::OutOfMemory)
        } else {
            Ok(())
        }
    }
}
```

## Performance Targets

| Operation | Latency | Memory |
|-----------|---------|--------|
| Footer read | <500ms | <1MB |
| Row group read | <2s | <128MB |
| Full scan (1GB file) | <30s | <256MB |
| Filtered query (10% selectivity) | <5s | <100MB |

## Testing Strategy (EXTREME TDD)

### Unit Tests
- [ ] HTTP range request client
- [ ] Parquet footer parsing
- [ ] Row group decoder
- [ ] Column reader
- [ ] Filter pushdown

### Integration Tests
- [ ] End-to-end query on 500MB file
- [ ] Memory stays <2GB during scan
- [ ] Retry logic on network failure

### Property Tests
```rust
#[quickcheck]
fn prop_streaming_equals_full_load(data: Vec<Row>) {
    let full_result = load_full_file(&data).query("SELECT * WHERE x > 10");
    let stream_result = stream_file(&data).query("SELECT * WHERE x > 10");
    assert_eq!(full_result, stream_result);
}
```

### Browser E2E Tests (Playwright)
```javascript
test('Large dataset query under 2GB memory', async ({ page }) => {
    await page.goto('http://localhost:8080');

    // Load 1GB Parquet file
    await page.click('#load-large-dataset');

    // Monitor memory usage
    const memory = await page.evaluate(() => performance.memory.usedJSHeapSize);
    expect(memory).toBeLessThan(2_000_000_000); // 2GB

    // Execute query
    await page.fill('#sql', 'SELECT * FROM events WHERE value > 1000');
    await page.click('#execute');

    // Verify results
    const results = await page.textContent('#results');
    expect(results).toContain('Query completed');
});
```

## References

1. **HTTP Range Requests (RFC 7233)**
   - https://tools.ietf.org/html/rfc7233
   - Standard for partial content delivery

2. **Late Materialization (Abadi et al. 2008)**
   - DOI: 10.1145/1376616.1376712
   - CIDR 2008: "Materialization strategies in a column-oriented DBMS"

3. **DuckDB-WASM HTTP Parquet**
   - https://duckdb.org/docs/api/wasm/query
   - Production reference implementation

4. **Apache Parquet Format**
   - https://parquet.apache.org/docs/file-format/
   - File layout and metadata specification

5. **Browser Memory Limits**
   - Chrome: https://bugs.chromium.org/p/v8/issues/detail?id=847
   - Empirical testing on 2025-11-25

## Toyota Way Alignment

- **Poka-Yoke**: Memory budget prevents OOM crashes
- **Muda Elimination**: Late materialization avoids wasted row reconstruction
- **Genchi Genbutsu**: Tested on real browser memory limits
- **Jidoka**: Built-in quality with streaming tests

## Success Criteria

- [x] Specification complete
- [ ] HTTP range client passing tests
- [ ] Streaming Parquet reader passing tests
- [ ] Late materialization pattern implemented
- [ ] Memory stays <2GB on 5GB file query
- [ ] Browser E2E tests pass
- [ ] Performance meets targets
- [ ] Documentation complete
