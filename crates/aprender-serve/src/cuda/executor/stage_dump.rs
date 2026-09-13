//! L0-1b per-op stage dump on the GPU-resident path (#2971, PP-066).
//!
//! `apr parity --per-op` arms `per_op_tap::set_gpu_dump` on its thread (never an
//! env variable); [`CudaExecutor::dump_stage`] is a no-op when nothing is armed,
//! so the production path pays one thread-local read per stage. When armed, the
//! executor synchronises every stream, copies the device buffer to the host and
//! writes `<dir>/layer-N/<stage>.bin` in APRT — the same file the CPU tap writes
//! for the same (stage, layer), so the two trees compare file by file.
use super::CudaExecutor;
use super::GpuBuffer;
use crate::inference_trace::gpu_stage_dump::maybe_dump_host_buffer;
use crate::inference_trace::gpu_stage_dump::per_op_tap;
use crate::inference_trace::save_tensor_stage::SaveTensorStage;

impl CudaExecutor {
    /// Dump the first `n` floats of `buf` for `(stage, layer)` when armed. Non-fatal.
    pub(crate) fn dump_stage(
        &self,
        stage: SaveTensorStage,
        layer: u32,
        buf: &GpuBuffer<f32>,
        n: usize,
    ) {
        let Some(cfg) = per_op_tap::gpu_dump() else {
            return;
        };
        if let Err(e) = self.synchronize_all() {
            eprintln!(
                "[per-op-dump] {}: layer {layer}: sync failed: {e}",
                stage.canonical_name()
            );
            return;
        }
        let n = n.min(buf.len());
        let mut host = vec![0.0f32; n];
        if let Err(e) = buf.copy_to_host(&mut host) {
            eprintln!(
                "[per-op-dump] {}: layer {layer}: copy failed: {e}",
                stage.canonical_name()
            );
            return;
        }
        if let Err(e) = maybe_dump_host_buffer(Some(&cfg), stage, layer, &host) {
            eprintln!(
                "[per-op-dump] {}: layer {layer}: write failed: {e}",
                stage.canonical_name()
            );
        }
    }
}
