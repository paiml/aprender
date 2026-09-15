//! CUDA factory: `libcuda.so.1` through the driver API, never cudart (REG-2).
use super::{Api, BackendEntry, BackendFactory, BackendKind, MemKind, Reason, Source, Status};
use trueno_gpu::driver::{device_count, CudaContext};

/// Discovers NVIDIA devices through the dlopen'd driver.
pub struct CudaFactory;

const LIB: &str = "libcuda.so.1";

impl BackendFactory for CudaFactory {
    fn kind(&self) -> BackendKind {
        BackendKind::Cuda
    }

    fn discover(&self) -> Vec<BackendEntry> {
        // Three distinct, reachable reasons (review quorum 2026-09-06, lane 1:
        // `cuda_available()` folds a load failure, a probe error and zero devices
        // into one bool, which made two arms dead). dlopen first, then count.
        if trueno_gpu::driver::sys::CudaDriver::load().is_none() {
            return vec![BackendEntry::unavailable(
                BackendKind::Cuda,
                Api::CudaDriver,
                Source::Dlopen(LIB.to_string()),
                Reason::DriverNotFound { path: LIB.to_string() },
            )];
        }
        let n = match device_count() {
            Ok(n) => n,
            Err(e) => {
                return vec![BackendEntry::unavailable(
                    BackendKind::Cuda,
                    Api::CudaDriver,
                    Source::Dlopen(LIB.to_string()),
                    Reason::ProbeFailed { error: format!("cuDeviceGetCount: {e:?}") },
                )]
            }
        };
        if n == 0 {
            return vec![BackendEntry::unavailable(
                BackendKind::Cuda,
                Api::CudaDriver,
                Source::Dlopen(LIB.to_string()),
                Reason::NoDevice,
            )];
        }
        (0..n).map(probe_device).collect()
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn probe_device(i: usize) -> BackendEntry {
    let ctx = match CudaContext::new(i as i32) {
        Ok(c) => c,
        Err(e) => {
            let mut entry = BackendEntry::unavailable(
                BackendKind::Cuda,
                Api::CudaDriver,
                Source::Dlopen(LIB.to_string()),
                Reason::ProbeFailed { error: format!("cuCtxCreate({i}): {e:?}") },
            );
            entry.device_index = Some(i as u32);
            return entry;
        }
    };
    let name = ctx.device_name().unwrap_or_else(|_| format!("cuda device {i}"));
    let (free, total) =
        ctx.memory_info().map(|(f, t)| (Some(f as u64), Some(t as u64))).unwrap_or((None, None));
    let cc = ctx.compute_capability().ok().map(|(maj, min)| format!("sm_{maj}{min}"));
    BackendEntry {
        kind: BackendKind::Cuda,
        api: Api::CudaDriver,
        device_index: Some(i as u32),
        device_uid: Some(super::device_uid("nvidia", &name)),
        device_name: name,
        vendor: "NVIDIA".to_string(),
        vendor_id: Some(0x10de),
        device_type: "discrete-gpu".to_string(),
        mem_total: total,
        mem_free: free,
        mem_kind: MemKind::Discrete,
        compute_class: cc,
        caps: vec!["async".to_string()],
        source: Source::Dlopen(LIB.to_string()),
        status: Status::Ready,
        transport: None,
    }
}
