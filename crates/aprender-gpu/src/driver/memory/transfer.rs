//! GPU memory transfer operations
//!
//! Host-to-device, device-to-host, and device-to-device copy methods
//! for `GpuBuffer<T>`. Both synchronous and asynchronous variants.

use std::ffi::c_void;

use crate::driver::context::{get_driver, CudaContext};
use crate::driver::stream::CudaStream;
use crate::driver::sys::CudaDriver;
use crate::GpuError;

// PMAT-420: All transfer methods call self.ensure_context() before any
// cuMemcpy* call.  Without this, cuMemcpyHtoD/DtoH silently produces
// zeros when the CUDA context is not current on the calling thread
// (see paiml/trueno#232).

use super::buffer::GpuBuffer;

// ============================================================================
// Blocking host-to-device primitive
// ============================================================================

/// Perform a host→device copy that is genuinely COMPLETE when it returns.
///
/// # The defect this closes (GPU-ORD-9)
///
/// CUDA's documented API synchronization behavior for a transfer out of
/// *pageable* host memory is that the driver stages the bytes into an internal
/// pinned buffer and returns "once the pageable buffer has been copied to the
/// staging memory — the DMA to final destination may **not** have completed".
/// That trailing DMA runs on the legacy default stream (handle 0).
///
/// When this was written, every `CudaStream` this crate handed out was
/// `CU_STREAM_NON_BLOCKING`, which is by definition *not* ordered against the
/// legacy default stream. So a kernel launch or `*_async` copy issued on a
/// `CudaStream` right after a "synchronous" upload could read the buffer's
/// pre-upload contents — silently, with every CUDA call returning success.
/// Measured on an RTX 4090 at 8 of 40 runs of
/// `test_gpu_buffer_async_device_to_host`, and 15 of 200 uploads with the
/// readback instrumented.
///
/// Draining the legacy default stream is the narrow wait that closes it.
///
/// # PERF-053 (aprender#2767) changed what that wait costs
///
/// [`crate::driver::stream::stream_create_flags`] now defaults to
/// `CU_STREAM_DEFAULT`, so this crate's streams ARE legacy-ordered. Two
/// consequences, and neither is a licence to delete the drain:
///
/// - the "costs microseconds when idle" claim no longer holds. The drain used
///   to skip non-blocking streams; it now waits for everything in flight, and
///   that wait grows with batch size. It is a candidate for removal *under the
///   default*, but removal is a correctness change and needs its own falsifier.
/// - it is still LOAD-BEARING under `APR_STREAM_NONBLOCKING=1`, which restores
///   the non-blocking flag this paragraph originally described.
///
/// # Why not a crate-owned stream
///
/// Issuing the copy on a private stream and synchronizing only that would
/// avoid the legacy stream entirely, which is tempting because a legacy-stream
/// synchronize is illegal while a graph capture is open. Such a stream cannot
/// be cached: a stream belongs to the context it was created in, and this crate
/// releases the device's primary context whenever the last `CudaContext` drops.
/// `test_buffer_roundtrip_fuzz` creates and drops one per proptest case, so a
/// process-global cached stream is left dangling and the next upload takes
/// SIGSEGV — measured, 3 nextest runs out of 3. Creating and destroying a
/// stream per upload instead would put two extra driver calls into the
/// weight-loading path.
///
/// The capture hazard is real but narrow, and it is closed from the other side:
/// nothing in this crate opens a `CaptureMode::Global` capture any more, and
/// under `ThreadLocal` capture CUDA does not police other threads' actions.
/// See `driver::cuda_tests::CAPTURE_VS_CTX_SYNC`.
///
/// # Poka-yoke
///
/// This is the ONLY place in the crate that calls `cuMemcpyHtoD`. Every
/// synchronous H2D path funnels through here, so a synchronous upload that
/// returns with its DMA still in flight is not expressible.
/// `raw_htod_symbol_has_exactly_one_call_site` (below) fails the crate's tests
/// if a second raw call site is ever introduced.
fn memcpy_htod_blocking(
    driver: &CudaDriver,
    dst_ptr: u64,
    src: *const c_void,
    size: usize,
) -> Result<(), GpuError> {
    // SAFETY: caller has validated src is readable for `size` bytes and
    // dst_ptr is a device pointer with at least `size` bytes allocated.
    let result = unsafe { (driver.cuMemcpyHtoD)(dst_ptr, src, size) };
    CudaDriver::check(result).map_err(|e| GpuError::Transfer(e.to_string()))?;

    if ord9_drain_disabled() {
        return Ok(());
    }
    // Drain the legacy default stream (handle 0) so the staged DMA has landed
    // before any non-blocking stream can observe the destination buffer.
    // SAFETY: the null handle is the legacy default stream; always valid.
    let result = unsafe { (driver.cuStreamSynchronize)(std::ptr::null_mut()) };
    CudaDriver::check(result)
        .map_err(|e| GpuError::Transfer(format!("H2D upload did not land on the device: {e}")))
}

/// PERF-053 measurement knob: `APR_ORD9_DRAIN_SKIP=1` removes the legacy-stream drain above.
///
/// **This reintroduces GPU-ORD-9 and must never be set in production.** It exists for exactly
/// one reason: since PERF-053 made this crate's streams legacy-ordered, that drain no longer
/// costs "microseconds when idle" -- it waits for every kernel in flight, on every H2D upload.
/// Whether that is where the measured cost of ordered streams lives is a question about this
/// binary, and answering it across two builds would not be an answer. Read once and cached, so
/// it cannot change under a running server.
///
/// ANSWERED, and the answer is no. agg_tok_s, median of SIX replicates, c=8, RTX 4090 sm_89,
/// 1.5B Q4_K_M, arms interleaved in one binary:
///
/// ```text
///   ordered (default)              1158.0    [1112.1 1208.2 1248.7 1192.1 1123.9 1008.7]
///   ordered + this knob set        1181.7    [1177.2 1262.8 1186.2 1112.8 1190.6 1055.4]
///   racy (APR_STREAM_NONBLOCKING)  1348.1    [1425.8 1164.0 1350.9 1345.3 1394.9 1267.7]
/// ```
///
/// If the drain were the cost, the middle row would sit on the third. It sits on the first --
/// 1.020x of ordered, still 0.877x of racy. So the drain stays, and this knob's only remaining
/// job is to keep that refutation re-runnable. (Ranges overlap: one racy replicate fell to
/// 1164.0. The refutation does not rest on separating ordered from racy, only on the drain arm
/// landing with ordered rather than with racy, which all six replicates do.)
///
/// SEPARATE FINDING, unresolved: `test_sync_upload_visible_to_nonblocking_stream` -- the ONLY
/// falsifier for GPU-ORD-9 -- passes 128/128 rounds even with this knob set AND
/// `APR_STREAM_NONBLOCKING=1`, which is exactly the pre-fix configuration it was written to
/// catch (measured 15 stale reads in 200 rounds at the time). It no longer discriminates on this
/// driver, so nothing in the tree can currently certify a change to this drain. That is why the
/// drain is NOT removed here despite costing measurable throughput at m=1.
fn ord9_drain_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var("APR_ORD9_DRAIN_SKIP").as_deref() == Ok("1"))
}

// ============================================================================
// Host <-> Device Transfers
// ============================================================================

impl<T: Copy> GpuBuffer<T> {
    /// Copy data from host to device (synchronous)
    ///
    /// # Arguments
    ///
    /// * `data` - Host data to copy (must have same length as buffer)
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::Transfer)` if copy fails.
    /// Returns `Err(GpuError::InvalidValue)` if lengths don't match.
    pub fn copy_from_host(&mut self, data: &[T]) -> Result<(), GpuError> {
        if data.len() != self.len {
            return Err(GpuError::Transfer(format!(
                "Length mismatch: host {} vs device {}",
                data.len(),
                self.len
            )));
        }

        if self.len == 0 {
            return Ok(());
        }

        self.ensure_context()?;

        let driver = get_driver()?;
        let size = self.size_bytes();

        // data is valid for size bytes, ptr is valid device pointer
        memcpy_htod_blocking(driver, self.ptr, data.as_ptr() as *const c_void, size)
    }

    /// Copy data from device to host (synchronous)
    ///
    /// Supports partial readback: if `data.len() < self.len`, copies only the first
    /// `data.len()` elements. This is safe because cuMemcpyDtoH respects the size parameter.
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::Transfer)` if `data.len() > self.len` or copy fails.
    pub fn copy_to_host(&self, data: &mut [T]) -> Result<(), GpuError> {
        if data.len() > self.len {
            return Err(GpuError::Transfer(format!(
                "Host buffer too large: host {} > device {}",
                data.len(),
                self.len
            )));
        }

        if data.is_empty() {
            return Ok(());
        }

        self.ensure_context()?;

        let driver = get_driver()?;
        let size = data.len() * std::mem::size_of::<T>();

        // SAFETY: data is valid for size bytes, ptr is valid device pointer
        let result =
            unsafe { (driver.cuMemcpyDtoH)(data.as_mut_ptr() as *mut c_void, self.ptr, size) };
        CudaDriver::check(result).map_err(|e| GpuError::Transfer(e.to_string()))
    }

    /// Copy data from host to device (asynchronous)
    ///
    /// # Arguments
    ///
    /// * `data` - Host data to copy (must have same length as buffer)
    /// * `stream` - Stream for async operation
    ///
    /// # Safety
    ///
    /// The host data must remain valid until the stream is synchronized.
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::Transfer)` if copy fails.
    pub unsafe fn copy_from_host_async(
        &mut self,
        data: &[T],
        stream: &CudaStream,
    ) -> Result<(), GpuError> {
        if data.len() != self.len {
            return Err(GpuError::Transfer(format!(
                "Length mismatch: host {} vs device {}",
                data.len(),
                self.len
            )));
        }

        if self.len == 0 {
            return Ok(());
        }

        self.ensure_context()?;

        let driver = get_driver()?;
        let size = self.size_bytes();

        // SAFETY: data is valid for size bytes, caller ensures data outlives stream ops
        let result = unsafe {
            (driver.cuMemcpyHtoDAsync)(self.ptr, data.as_ptr() as *const c_void, size, stream.raw())
        };
        CudaDriver::check(result).map_err(|e| GpuError::Transfer(e.to_string()))
    }

    /// Copy data from device to host (asynchronous)
    ///
    /// # Arguments
    ///
    /// * `data` - Host buffer to copy into
    /// * `stream` - Stream for async operation
    ///
    /// # Safety
    ///
    /// The host buffer must remain valid until the stream is synchronized.
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::Transfer)` if copy fails.
    pub unsafe fn copy_to_host_async(
        &self,
        data: &mut [T],
        stream: &CudaStream,
    ) -> Result<(), GpuError> {
        if data.len() != self.len {
            return Err(GpuError::Transfer(format!(
                "Length mismatch: host {} vs device {}",
                data.len(),
                self.len
            )));
        }

        if self.len == 0 {
            return Ok(());
        }

        self.ensure_context()?;

        let driver = get_driver()?;
        let size = self.size_bytes();

        // SAFETY: data is valid for size bytes, caller ensures data outlives stream ops
        let result = unsafe {
            (driver.cuMemcpyDtoHAsync)(
                data.as_mut_ptr() as *mut c_void,
                self.ptr,
                size,
                stream.raw(),
            )
        };
        CudaDriver::check(result).map_err(|e| GpuError::Transfer(e.to_string()))
    }

    /// Create buffer and initialize from host data
    ///
    /// Convenience method combining allocation and upload.
    ///
    /// # Arguments
    ///
    /// * `ctx` - CUDA context
    /// * `data` - Host data to upload
    ///
    /// # Errors
    ///
    /// Returns allocation or transfer errors.
    pub fn from_host(ctx: &CudaContext, data: &[T]) -> Result<Self, GpuError> {
        let mut buf = Self::new(ctx, data.len())?;
        buf.copy_from_host(data)?;
        Ok(buf)
    }

    /// Copy partial data from host to device at specific offset (PAR-018)
    ///
    /// # Arguments
    ///
    /// * `data` - Host data to copy
    /// * `offset` - Element offset in device buffer where copy begins
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::Transfer)` if offset + data.len() exceeds buffer size.
    pub fn copy_from_host_at(&mut self, data: &[T], offset: usize) -> Result<(), GpuError> {
        if offset + data.len() > self.len {
            return Err(GpuError::Transfer(format!(
                "Partial copy out of bounds: offset {} + len {} > buffer {}",
                offset,
                data.len(),
                self.len
            )));
        }

        if data.is_empty() {
            return Ok(());
        }

        self.ensure_context()?;

        let driver = get_driver()?;
        let size = std::mem::size_of_val(data);
        let dst_ptr = self.ptr + (offset * std::mem::size_of::<T>()) as u64;

        // bounds checked above, data and ptr are valid
        memcpy_htod_blocking(driver, dst_ptr, data.as_ptr() as *const c_void, size)
    }

    /// Copy partial data from device to host at specific offset (PAR-018)
    ///
    /// # Arguments
    ///
    /// * `data` - Host buffer to copy into
    /// * `offset` - Element offset in device buffer where copy begins
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::Transfer)` if offset + data.len() exceeds buffer size.
    pub fn copy_to_host_at(&self, data: &mut [T], offset: usize) -> Result<(), GpuError> {
        if offset + data.len() > self.len {
            return Err(GpuError::Transfer(format!(
                "Partial copy out of bounds: offset {} + len {} > buffer {}",
                offset,
                data.len(),
                self.len
            )));
        }

        if data.is_empty() {
            return Ok(());
        }

        self.ensure_context()?;

        let driver = get_driver()?;
        let size = std::mem::size_of_val(data);
        let src_ptr = self.ptr + (offset * std::mem::size_of::<T>()) as u64;

        // SAFETY: bounds checked above, data and ptr are valid
        let result =
            unsafe { (driver.cuMemcpyDtoH)(data.as_mut_ptr() as *mut c_void, src_ptr, size) };
        CudaDriver::check(result).map_err(|e| GpuError::Transfer(e.to_string()))
    }

    // =========================================================================
    // PAR-023: Device-to-Device Copy (Zero-Sync Pipeline)
    // =========================================================================

    /// Clone buffer to new GPU memory (device-to-device copy)
    ///
    /// Allocates new GPU memory and copies contents from self.
    ///
    /// # Arguments
    ///
    /// * `ctx` - CUDA context (must be current)
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::MemoryAllocation)` if allocation fails.
    /// Returns `Err(GpuError::Transfer)` if copy fails.
    pub fn clone(&self, ctx: &CudaContext) -> Result<Self, GpuError> {
        let mut new_buffer = GpuBuffer::new(ctx, self.len)?;
        new_buffer.copy_from_buffer(self)?;
        Ok(new_buffer)
    }

    /// Copy data from another GPU buffer (device-to-device, synchronous)
    ///
    /// Enables zero-sync GPU pipelines by keeping data on device.
    ///
    /// # Arguments
    ///
    /// * `src` - Source GPU buffer (must have same length)
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::Transfer)` if lengths don't match or copy fails.
    pub fn copy_from_buffer(&mut self, src: &GpuBuffer<T>) -> Result<(), GpuError> {
        if src.len != self.len {
            return Err(GpuError::Transfer(format!(
                "PAR-023: D2D length mismatch: src {} vs dst {}",
                src.len, self.len
            )));
        }

        if self.len == 0 {
            return Ok(());
        }

        self.ensure_context()?;

        let driver = get_driver()?;
        let size = self.size_bytes();

        // SAFETY: both buffers are valid, size is correct
        let result = unsafe { (driver.cuMemcpyDtoD)(self.ptr, src.ptr, size) };
        CudaDriver::check(result).map_err(|e| GpuError::Transfer(e.to_string()))
    }

    /// Copy partial data from another GPU buffer at specific offset (PAR-023)
    ///
    /// Enables GPU-resident KV cache updates without host round-trip.
    ///
    /// # Arguments
    ///
    /// * `src` - Source GPU buffer
    /// * `dst_offset` - Element offset in destination (this buffer)
    /// * `src_offset` - Element offset in source buffer
    /// * `count` - Number of elements to copy
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::Transfer)` if copy would exceed buffer bounds.
    pub fn copy_from_buffer_at(
        &mut self,
        src: &GpuBuffer<T>,
        dst_offset: usize,
        src_offset: usize,
        count: usize,
    ) -> Result<(), GpuError> {
        if dst_offset + count > self.len {
            return Err(GpuError::Transfer(format!(
                "PAR-023: D2D dst out of bounds: {} + {} > {}",
                dst_offset, count, self.len
            )));
        }
        if src_offset + count > src.len {
            return Err(GpuError::Transfer(format!(
                "PAR-023: D2D src out of bounds: {} + {} > {}",
                src_offset, count, src.len
            )));
        }

        if count == 0 {
            return Ok(());
        }

        self.ensure_context()?;

        let driver = get_driver()?;
        let size = count * std::mem::size_of::<T>();
        let dst_ptr = self.ptr + (dst_offset * std::mem::size_of::<T>()) as u64;
        let src_ptr = src.ptr + (src_offset * std::mem::size_of::<T>()) as u64;

        // SAFETY: bounds checked above, both ptrs are valid
        let result = unsafe { (driver.cuMemcpyDtoD)(dst_ptr, src_ptr, size) };
        CudaDriver::check(result).map_err(|e| GpuError::Transfer(e.to_string()))
    }

    /// Async copy from another GPU buffer (PAR-023)
    ///
    /// # Safety
    ///
    /// Both buffers must remain valid until stream is synchronized.
    pub unsafe fn copy_from_buffer_async(
        &mut self,
        src: &GpuBuffer<T>,
        stream: &CudaStream,
    ) -> Result<(), GpuError> {
        if src.len != self.len {
            return Err(GpuError::Transfer(format!(
                "PAR-023: Async D2D length mismatch: src {} vs dst {}",
                src.len, self.len
            )));
        }

        if self.len == 0 {
            return Ok(());
        }

        self.ensure_context()?;

        let driver = get_driver()?;
        let size = self.size_bytes();

        // SAFETY: both buffers valid, caller ensures lifetime
        let result = unsafe { (driver.cuMemcpyDtoDAsync)(self.ptr, src.ptr, size, stream.raw()) };
        CudaDriver::check(result).map_err(|e| GpuError::Transfer(e.to_string()))
    }

    /// PAR-023: Async D2D copy with offsets
    ///
    /// Copies a region from source buffer to destination buffer asynchronously.
    /// Does not synchronize - caller must ensure stream sync before accessing data.
    ///
    /// # Arguments
    ///
    /// * `src` - Source GPU buffer
    /// * `dst_offset` - Element offset in destination (this buffer)
    /// * `src_offset` - Element offset in source buffer
    /// * `count` - Number of elements to copy
    /// * `stream` - CUDA stream for async operation
    ///
    /// # Safety
    ///
    /// Both buffers must remain valid until stream is synchronized.
    pub unsafe fn copy_from_buffer_at_async(
        &mut self,
        src: &GpuBuffer<T>,
        dst_offset: usize,
        src_offset: usize,
        count: usize,
        stream: &CudaStream,
    ) -> Result<(), GpuError> {
        if dst_offset + count > self.len {
            return Err(GpuError::Transfer(format!(
                "PAR-023: Async D2D dst out of bounds: {} + {} > {}",
                dst_offset, count, self.len
            )));
        }
        if src_offset + count > src.len {
            return Err(GpuError::Transfer(format!(
                "PAR-023: Async D2D src out of bounds: {} + {} > {}",
                src_offset, count, src.len
            )));
        }

        if count == 0 {
            return Ok(());
        }

        self.ensure_context()?;

        let driver = get_driver()?;
        let size = count * std::mem::size_of::<T>();
        let dst_ptr = self.ptr + (dst_offset * std::mem::size_of::<T>()) as u64;
        let src_ptr = src.ptr + (src_offset * std::mem::size_of::<T>()) as u64;

        // SAFETY: bounds checked above, both ptrs are valid, caller ensures lifetime
        let result = unsafe { (driver.cuMemcpyDtoDAsync)(dst_ptr, src_ptr, size, stream.raw()) };
        CudaDriver::check(result).map_err(|e| GpuError::Transfer(e.to_string()))
    }

    /// PAR-023: Async D2D copy with raw stream handle
    ///
    /// Same as `copy_from_buffer_at_async` but takes raw CUstream handle.
    /// Useful when borrow checker prevents passing &CudaStream due to other borrows.
    ///
    /// # Safety
    ///
    /// - Both buffers must remain valid until stream is synchronized.
    /// - Stream handle must be valid.
    pub unsafe fn copy_from_buffer_at_async_raw(
        &mut self,
        src: &GpuBuffer<T>,
        dst_offset: usize,
        src_offset: usize,
        count: usize,
        stream_handle: crate::driver::sys::CUstream,
    ) -> Result<(), GpuError> {
        if dst_offset + count > self.len {
            return Err(GpuError::Transfer(format!(
                "PAR-023: Async D2D dst out of bounds: {} + {} > {}",
                dst_offset, count, self.len
            )));
        }
        if src_offset + count > src.len {
            return Err(GpuError::Transfer(format!(
                "PAR-023: Async D2D src out of bounds: {} + {} > {}",
                src_offset, count, src.len
            )));
        }

        if count == 0 {
            return Ok(());
        }

        self.ensure_context()?;

        let driver = get_driver()?;
        let size = count * std::mem::size_of::<T>();
        let dst_ptr = self.ptr + (dst_offset * std::mem::size_of::<T>()) as u64;
        let src_ptr = src.ptr + (src_offset * std::mem::size_of::<T>()) as u64;

        // SAFETY: bounds checked above, both ptrs valid, caller ensures lifetime + stream valid
        let result = unsafe { (driver.cuMemcpyDtoDAsync)(dst_ptr, src_ptr, size, stream_handle) };
        CudaDriver::check(result).map_err(|e| GpuError::Transfer(e.to_string()))
    }
}

#[cfg(test)]
mod htod_funnel_guard {
    /// Poka-yoke enforcement for GPU-ORD-9.
    ///
    /// The defect was a synchronous host-to-device upload returning while its
    /// DMA was still queued on the legacy default stream, where no
    /// `CU_STREAM_NON_BLOCKING` stream is ordered against it. The fix routes
    /// every synchronous H2D through `memcpy_htod_blocking`, which drains that
    /// stream before returning. That only stays true while
    /// `memcpy_htod_blocking` is the sole caller of the raw driver entry point
    /// — a second raw call site reintroduces the exact defect, silently, and
    /// surfaces as a flake weeks later.
    ///
    /// The guard scans the whole crate rather than this one file, because the
    /// decision to call the raw symbol can be made from any module.
    #[test]
    fn raw_htod_symbol_has_exactly_one_call_site() {
        // Assembled at run time so this guard's own source cannot match it.
        let needle = format!(".{})", "cuMemcpyHtoD");

        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut hits: Vec<String> = Vec::new();
        let mut files_scanned = 0usize;
        let mut stack = vec![src.clone()];
        while let Some(dir) = stack.pop() {
            let entries = std::fs::read_dir(&dir)
                .unwrap_or_else(|e| panic!("guard cannot read {}: {e}", dir.display()));
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.extension().is_some_and(|e| e == "rs") {
                    files_scanned += 1;
                    let text = std::fs::read_to_string(&path)
                        .unwrap_or_else(|e| panic!("guard cannot read {}: {e}", path.display()));
                    for (n, line) in text.lines().enumerate() {
                        if line.contains(&needle) {
                            hits.push(format!(
                                "{}:{}",
                                path.strip_prefix(&src).unwrap_or(&path).display(),
                                n + 1
                            ));
                        }
                    }
                }
            }
        }

        // A scan that finds nothing is indistinguishable from a scan pointed at
        // the wrong text, so zero hits must fail as loudly as many.
        assert!(
            files_scanned > 100,
            "guard scanned only {files_scanned} files under {} — it is not looking at the crate",
            src.display()
        );
        assert_eq!(
            hits.len(),
            1,
            "the raw synchronous H2D symbol must have exactly ONE call site, inside \
             memcpy_htod_blocking, which drains the legacy default stream. Found {}: {:?}. \
             Route the new upload through memcpy_htod_blocking, or GPU-ORD-9 returns as a \
             roughly 1-in-5 stale read under load.",
            hits.len(),
            hits
        );
        assert!(
            hits[0].starts_with("driver/memory/transfer.rs:"),
            "the single raw H2D call site moved out of transfer.rs: {:?}",
            hits
        );
    }
}
