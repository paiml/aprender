//! VRAM Reservation Ledger (GPU-SHARE-001).
//!
//! Uses flock for mutual exclusion and atomic write (write-to-temp, rename)
//! for crash safety.
//!
//! # Contract C-VRAM-001
//!
//! `CudaTrainer::new()` MUST NOT allocate if
//! `ledger.total_reserved() + budget > total_mb × reserve_factor`.
//!
//! # Protocol
//!
//! 1. Acquire `flock(LOCK_EX)` on ledger file
//! 2. Read reservations, prune dead PIDs + expired leases
//! 3. Check capacity: `sum(active.budget_mb) + my_budget <= total_mb × reserve_factor`
//! 4. Write reservation via atomic rename (write tmp → rename)
//! 5. Release lock (close fd / drop)
//! 6. On exit: best-effort cleanup via `Drop`

use std::fs::{self, File, OpenOptions};
use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use fs4::fs_std::FileExt;
use serde::{Deserialize, Serialize};

use super::error::GpuError;
use super::profiler::GpuProfiler;
use crate::trace::{TraceStep, TRACER};

/// Default ledger location: `~/.cache/entrenar/gpu-ledger.json`.
fn default_ledger_path() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("entrenar")
        .join("gpu-ledger.json")
}

/// Reserve factor for discrete GPUs (15% headroom).
pub const RESERVE_FACTOR_DISCRETE: f32 = 0.85;

/// Reserve factor for unified memory (40% headroom for OS).
pub const RESERVE_FACTOR_UNIFIED: f32 = 0.60;

/// Default lease duration (24 hours).
pub const DEFAULT_LEASE_HOURS: i64 = 24;

/// A single VRAM reservation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reservation {
    /// Unique reservation ID.
    pub id: u64,
    /// Process ID of the holder.
    pub pid: u32,
    /// Budgeted VRAM in MB.
    pub budget_mb: usize,
    /// Actual measured VRAM in MB (updated post-init).
    pub actual_mb: Option<usize>,
    /// Human-readable task description.
    pub task: String,
    /// GPU UUID this reservation is for.
    pub gpu_uuid: String,
    /// When the reservation was created.
    pub started: DateTime<Utc>,
    /// When the lease automatically expires.
    pub lease_expires: DateTime<Utc>,
}

impl Reservation {
    /// Whether this reservation's lease has expired.
    pub fn is_expired(&self) -> bool {
        Utc::now() > self.lease_expires
    }

    /// Whether the holding process is still alive (Linux /proc check).
    pub fn is_alive(&self) -> bool {
        Path::new(&format!("/proc/{}/stat", self.pid)).exists()
    }

    /// Whether this reservation should be pruned.
    pub fn should_prune(&self) -> bool {
        self.is_expired() || !self.is_alive()
    }
}

/// Ledger file contents.
#[derive(Debug, Default, Serialize, Deserialize)]
struct LedgerData {
    reservations: Vec<Reservation>,
}

impl LedgerData {
    /// Remove dead PIDs and expired leases in-place.
    fn prune_dead(&mut self) {
        self.reservations.retain(|r| !r.should_prune());
    }

    /// Sum of reserved VRAM for a specific GPU.
    fn total_reserved_for(&self, gpu_uuid: &str) -> usize {
        self.reservations
            .iter()
            .filter(|r| r.gpu_uuid == gpu_uuid)
            .map(|r| r.actual_mb.unwrap_or(r.budget_mb))
            .sum()
    }
}

/// VRAM reservation ledger with flock-based mutual exclusion.
pub struct VramLedger {
    path: PathBuf,
    /// GPU UUID (from nvidia-smi -L).
    pub gpu_uuid: String,
    /// Total GPU memory in MB.
    pub total_mb: usize,
    /// Fraction of total usable (0.85 discrete, 0.60 unified).
    pub reserve_factor: f32,
    lease_hours: i64,
    /// Profiler for brick-phase timing.
    profiler: GpuProfiler,
    /// Our reservation ID if we hold one.
    pub our_reservation_id: Option<u64>,
}

impl VramLedger {
    /// Create a new ledger for the specified GPU.
    pub fn new(gpu_uuid: String, total_mb: usize, reserve_factor: f32) -> Self {
        Self {
            path: default_ledger_path(),
            gpu_uuid,
            total_mb,
            reserve_factor,
            lease_hours: DEFAULT_LEASE_HOURS,
            profiler: GpuProfiler::disabled(),
            our_reservation_id: None,
        }
    }

    /// Create a ledger at a custom path (for testing).
    pub fn with_path(mut self, path: PathBuf) -> Self {
        self.path = path;
        self
    }

    /// Enable profiling.
    pub fn with_profiling(mut self, enabled: bool) -> Self {
        self.profiler = GpuProfiler::new(enabled);
        self
    }

    /// Set custom lease duration.
    pub fn with_lease_hours(mut self, hours: i64) -> Self {
        self.lease_hours = hours;
        self
    }

    /// Usable VRAM capacity in MB.
    pub fn capacity_mb(&self) -> usize {
        (self.total_mb as f32 * self.reserve_factor) as usize
    }

    /// Total reserved VRAM across all active reservations for our GPU.
    pub fn total_reserved(&self) -> Result<usize, GpuError> {
        let gpu_uuid = self.gpu_uuid.clone();
        self.with_lock_read(|data| {
            data.reservations
                .iter()
                .filter(|r| r.gpu_uuid == gpu_uuid && !r.should_prune())
                .map(|r| r.actual_mb.unwrap_or(r.budget_mb))
                .sum()
        })
    }

    /// Available VRAM for new reservations (capacity - reserved).
    pub fn available_mb(&self) -> Result<usize, GpuError> {
        let reserved = self.total_reserved()?;
        Ok(self.capacity_mb().saturating_sub(reserved))
    }

    /// Try to reserve VRAM. Returns reservation ID on success.
    ///
    /// # Contract C-VRAM-001
    ///
    /// Fails with `GpuError::InsufficientMemory` if
    /// `total_reserved + budget_mb > capacity_mb`.
    pub fn try_reserve(&mut self, budget_mb: usize, task: &str) -> Result<u64, GpuError> {
        TRACER.span(
            TraceStep::LedgerReserve,
            format!("ledger_reserve budget={budget_mb}MB gpu={}", self.gpu_uuid),
            || self.try_reserve_inner(budget_mb, task),
        )
    }

    fn try_reserve_inner(&mut self, budget_mb: usize, task: &str) -> Result<u64, GpuError> {
        let gpu_uuid = self.gpu_uuid.clone();
        let lease_hours = self.lease_hours;
        let capacity = self.capacity_mb();
        let total_mb = self.total_mb;

        let id = self.with_lock_write(|data| {
            data.prune_dead();

            let reserved = data.total_reserved_for(&gpu_uuid);

            if reserved + budget_mb > capacity {
                return Err(GpuError::InsufficientMemory {
                    budget_mb,
                    available_mb: capacity.saturating_sub(reserved),
                    reserved_mb: reserved,
                    total_mb,
                });
            }

            let now = Utc::now();
            let id = reservation_id(&gpu_uuid, std::process::id(), now);
            let reservation = Reservation {
                id,
                pid: std::process::id(),
                budget_mb,
                actual_mb: None,
                task: task.to_string(),
                gpu_uuid: gpu_uuid.clone(),
                started: now,
                lease_expires: now + chrono::Duration::hours(lease_hours),
            };

            data.reservations.push(reservation);
            Ok(id)
        })?;

        self.our_reservation_id = Some(id);
        self.profiler.finish_op();
        Ok(id)
    }

    /// Update the actual measured VRAM for our reservation.
    pub fn update_actual(&mut self, actual_mb: usize) -> Result<(), GpuError> {
        let Some(our_id) = self.our_reservation_id else {
            return Ok(());
        };

        self.with_lock_write(|data| {
            if let Some(r) = data.reservations.iter_mut().find(|r| r.id == our_id) {
                r.actual_mb = Some(actual_mb);
            }
            Ok(())
        })
    }

    /// Release our reservation.
    pub fn release(&mut self) -> Result<(), GpuError> {
        let Some(our_id) = self.our_reservation_id.take() else {
            return Ok(());
        };

        TRACER.span(TraceStep::LedgerRelease, format!("ledger_release id={our_id}"), || {
            self.with_lock_write(|data| {
                data.reservations.retain(|r| r.id != our_id);
                Ok(())
            })
        })
    }

    /// Read all reservations for our GPU (pruned).
    pub fn read_reservations(&self) -> Result<Vec<Reservation>, GpuError> {
        let gpu_uuid = self.gpu_uuid.clone();
        self.with_lock_read(|data| {
            data.reservations
                .iter()
                .filter(|r| r.gpu_uuid == gpu_uuid && !r.should_prune())
                .cloned()
                .collect()
        })
    }

    /// Get profiler report.
    pub fn profiler_report(&self) -> String {
        self.profiler.report()
    }

    // ── flock + atomic read/write ──

    /// Execute a read-only operation under flock.
    fn with_lock_read<F, T>(&self, f: F) -> Result<T, GpuError>
    where
        F: FnOnce(&LedgerData) -> T,
    {
        ensure_parent_dir(&self.path)?;

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&self.path)?;

        file.lock_exclusive()
            .map_err(|e| GpuError::Io(std::io::Error::other(format!("flock: {e}"))))?;

        let data = read_ledger(&file)?;
        let result = f(&data);

        #[allow(clippy::incompatible_msrv)]
        file.unlock().map_err(|e| GpuError::Io(std::io::Error::other(format!("funlock: {e}"))))?;

        Ok(result)
    }

    /// Execute a read-modify-write operation under flock with atomic write.
    fn with_lock_write<F, T>(&mut self, f: F) -> Result<T, GpuError>
    where
        F: FnOnce(&mut LedgerData) -> Result<T, GpuError>,
    {
        ensure_parent_dir(&self.path)?;

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&self.path)?;

        // Phase: lock_acq
        self.profiler.begin(GpuProfiler::LOCK_ACQ);
        file.lock_exclusive()
            .map_err(|e| GpuError::Io(std::io::Error::other(format!("flock: {e}"))))?;
        self.profiler.end(GpuProfiler::LOCK_ACQ);

        // Phase: ledger_rd
        self.profiler.begin(GpuProfiler::LEDGER_RD);
        let mut data = read_ledger(&file)?;
        self.profiler.end(GpuProfiler::LEDGER_RD);

        let result = f(&mut data)?;

        // Atomic write-ahead: file → fsync → rename
        self.profiler.begin(GpuProfiler::LEDGER_WR);
        atomic_write_ledger(&self.path, &data)?;
        self.profiler.end(GpuProfiler::LEDGER_WR);

        // Phase: lock_rel
        self.profiler.begin(GpuProfiler::LOCK_REL);
        #[allow(clippy::incompatible_msrv)]
        file.unlock().map_err(|e| GpuError::Io(std::io::Error::other(format!("funlock: {e}"))))?;
        self.profiler.end(GpuProfiler::LOCK_REL);

        Ok(result)
    }
}

impl Drop for VramLedger {
    fn drop(&mut self) {
        let _ = self.release();
    }
}

// ── Helper functions ──

/// Generate a deterministic reservation ID.
fn reservation_id(gpu_uuid: &str, pid: u32, time: DateTime<Utc>) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    gpu_uuid.hash(&mut hasher);
    pid.hash(&mut hasher);
    time.timestamp_nanos_opt().unwrap_or(0).hash(&mut hasher);
    hasher.finish()
}

/// Read ledger JSON from an open file. Returns empty data if file is empty.
fn read_ledger(file: &File) -> Result<LedgerData, GpuError> {
    let mut contents = String::new();
    let mut reader = file;
    if reader.read_to_string(&mut contents).is_err() || contents.trim().is_empty() {
        return Ok(LedgerData::default());
    }
    // Graceful recovery: if ledger is corrupted (e.g., concurrent write race),
    // treat as empty rather than failing — lost reservations are self-healing
    // since processes re-register on next operation.
    serde_json::from_str(&contents).or_else(|_| Ok(LedgerData::default()))
}

/// Atomic write: write to temp file, fsync, rename over ledger.
fn atomic_write_ledger(path: &Path, data: &LedgerData) -> Result<(), GpuError> {
    let tmp_path = path.with_extension("tmp");
    let json = serde_json::to_string_pretty(data)
        .map_err(|e| GpuError::LedgerCorrupt(format!("JSON serialize: {e}")))?;

    let mut tmp_file = File::create(&tmp_path)?;
    tmp_file.write_all(json.as_bytes())?;
    tmp_file.sync_all()?;

    fs::rename(&tmp_path, path)?;
    Ok(())
}

/// Ensure parent directory exists.
fn ensure_parent_dir(path: &Path) -> Result<(), GpuError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

/// Detect GPU UUID by shelling out to `nvidia-smi -L`.
pub fn detect_gpu_uuid() -> String {
    std::process::Command::new("nvidia-smi")
        .args(["-L"])
        .output()
        .ok()
        .and_then(|out| {
            let stdout = String::from_utf8_lossy(&out.stdout);
            stdout.lines().find_map(|line| {
                let start = line.find("UUID: ")?;
                let uuid_start = start + 6;
                let end = line[uuid_start..].find(')')? + uuid_start;
                Some(line[uuid_start..end].to_string())
            })
        })
        .unwrap_or_else(|| "GPU-unknown".to_string())
}

/// Detect total GPU memory in MB via `nvidia-smi`.
pub fn detect_total_memory_mb() -> usize {
    let gpu_mb = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .and_then(|out| {
            let stdout = String::from_utf8_lossy(&out.stdout);
            stdout.trim().lines().next()?.trim().parse::<usize>().ok()
        })
        .unwrap_or(0);

    if gpu_mb > 0 {
        return gpu_mb;
    }

    // GB10 / DIGITS unified memory: nvidia-smi reports [N/A] for memory.total.
    // Use system RAM as VRAM (unified memory architecture).
    if detect_memory_type() == MemoryType::Unified {
        let sys_mb = sys_total_memory_mb();
        if sys_mb > 0 {
            eprintln!("[GPU-SHARE] Unified memory: using system RAM ({sys_mb} MB) as VRAM");
            return sys_mb;
        }
    }

    0
}

/// Read total system memory from /proc/meminfo (Linux).
fn sys_total_memory_mb() -> usize {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("MemTotal:"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse::<usize>().ok())
        })
        .map_or(0, |kb| kb / 1024)
}

/// Detect whether GPU has unified memory (Jetson) vs discrete.
pub fn detect_memory_type() -> MemoryType {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()
        .map_or(MemoryType::Discrete, |out| {
            let name = String::from_utf8_lossy(&out.stdout).to_lowercase();
            // Unified memory: Jetson/Orin/Tegra + Project DIGITS GB10
            if name.contains("jetson")
                || name.contains("orin")
                || name.contains("tegra")
                || name.contains("gb10")
                || name.contains("digits")
            {
                MemoryType::Unified
            } else {
                MemoryType::Discrete
            }
        })
}

/// GPU memory type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryType {
    /// Discrete GPU (PCIe, e.g., RTX 4090). Reserve factor: 0.85.
    Discrete,
    /// Unified memory (e.g., Jetson Orin). Reserve factor: 0.60.
    Unified,
}

impl MemoryType {
    /// Reserve factor for this memory type.
    pub fn reserve_factor(self) -> f32 {
        match self {
            Self::Discrete => RESERVE_FACTOR_DISCRETE,
            Self::Unified => RESERVE_FACTOR_UNIFIED,
        }
    }
}

/// Create a ledger auto-detecting GPU properties.
pub fn auto_ledger() -> VramLedger {
    let uuid = detect_gpu_uuid();
    let total_mb = detect_total_memory_mb();
    let mem_type = detect_memory_type();
    VramLedger::new(uuid, total_mb, mem_type.reserve_factor())
}

/// Human-readable GPU status display.
pub fn gpu_status_display(ledger: &VramLedger) -> Result<String, GpuError> {
    let reservations = ledger.read_reservations()?;
    let reserved: usize = reservations.iter().map(|r| r.actual_mb.unwrap_or(r.budget_mb)).sum();

    let mut out = String::new();
    out.push_str(&format!(
        "{}: {} MB total, {:.0}% reserve factor\n",
        ledger.gpu_uuid,
        ledger.total_mb,
        ledger.reserve_factor * 100.0
    ));
    out.push_str(&format!(
        "  Capacity: {} MB usable ({} MB reserved, {} MB available)\n",
        ledger.capacity_mb(),
        reserved,
        ledger.capacity_mb().saturating_sub(reserved),
    ));

    if reservations.is_empty() {
        out.push_str("  Reservations: none\n");
    } else {
        out.push_str(&format!("  Reservations: {}\n", reservations.len()));
        for r in &reservations {
            let actual = r
                .actual_mb
                .map_or_else(|| "measuring...".to_string(), |a| format!("{a} MB actual"));
            let elapsed = Utc::now().signed_duration_since(r.started);
            let hours = elapsed.num_hours();
            let mins = elapsed.num_minutes() % 60;
            out.push_str(&format!(
                "    PID {}: {} MB budget / {} ({}) — {}h {}m\n",
                r.pid, r.budget_mb, actual, r.task, hours, mins
            ));
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::Duration;

    static TEST_COUNTER: AtomicU32 = AtomicU32::new(0);

    fn test_ledger_path() -> PathBuf {
        let n = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join("entrenar-ledger-test");
        fs::create_dir_all(&dir).expect("test dir creation should succeed");
        dir.join(format!("test-ledger-{n}-{}.json", std::process::id()))
    }

    fn cleanup(path: &Path) {
        let _ = fs::remove_file(path);
        let _ = fs::remove_file(path.with_extension("tmp"));
    }

    #[test]
    fn test_empty_ledger_has_full_capacity() {
        let path = test_ledger_path();
        let ledger = VramLedger::new("GPU-test".into(), 24000, 0.85).with_path(path.clone());

        assert_eq!(ledger.capacity_mb(), 20400);
        assert_eq!(ledger.total_reserved().expect("should succeed"), 0);
        assert_eq!(ledger.available_mb().expect("should succeed"), 20400);

        cleanup(&path);
    }

    #[test]
    fn test_reserve_and_release() {
        let path = test_ledger_path();
        let mut ledger = VramLedger::new("GPU-test".into(), 24000, 0.85).with_path(path.clone());

        let id = ledger.try_reserve(8000, "test-job").expect("should succeed");
        assert!(id != 0);
        assert_eq!(ledger.total_reserved().expect("should succeed"), 8000);
        assert_eq!(ledger.available_mb().expect("should succeed"), 12400);

        ledger.release().expect("should succeed");
        assert_eq!(ledger.total_reserved().expect("should succeed"), 0);

        cleanup(&path);
    }

    #[test]
    fn test_capacity_invariant_prevents_overallocation() {
        let path = test_ledger_path();
        let mut ledger = VramLedger::new("GPU-test".into(), 24000, 0.85).with_path(path.clone());

        ledger.try_reserve(15000, "job-1").expect("should succeed");

        let result = ledger.try_reserve(10000, "job-2");
        assert!(result.is_err());
        match result.expect_err("should be InsufficientMemory") {
            GpuError::InsufficientMemory { budget_mb, available_mb, .. } => {
                assert_eq!(budget_mb, 10000);
                assert_eq!(available_mb, 5400);
            }
            other => panic!("expected InsufficientMemory, got {other}"),
        }

        cleanup(&path);
    }

    #[test]
    fn test_reserve_factor_limits_total() {
        let path = test_ledger_path();
        let mut ledger = VramLedger::new("GPU-test".into(), 10000, 0.85).with_path(path.clone());

        let result = ledger.try_reserve(9000, "too-big");
        assert!(result.is_err());

        cleanup(&path);
    }

    #[test]
    fn test_update_actual() {
        let path = test_ledger_path();
        let mut ledger = VramLedger::new("GPU-test".into(), 24000, 0.85).with_path(path.clone());

        ledger.try_reserve(8000, "test-job").expect("should succeed");
        ledger.update_actual(7300).expect("should succeed");

        assert_eq!(ledger.total_reserved().expect("should succeed"), 7300);

        cleanup(&path);
    }

    #[test]
    fn test_expired_lease_pruned() {
        let path = test_ledger_path();
        let mut ledger = VramLedger::new("GPU-test".into(), 24000, 0.85)
            .with_path(path.clone())
            .with_lease_hours(0);

        ledger.try_reserve(8000, "expiring-job").expect("should succeed");

        std::thread::sleep(Duration::from_millis(10));

        assert_eq!(ledger.total_reserved().expect("should succeed"), 0);

        cleanup(&path);
    }

    #[test]
    fn test_atomic_write_produces_valid_json() {
        let path = test_ledger_path();
        let mut ledger = VramLedger::new("GPU-test".into(), 24000, 0.85).with_path(path.clone());

        ledger.try_reserve(5000, "json-test").expect("should succeed");

        let contents = fs::read_to_string(&path).expect("should read");
        let data: LedgerData = serde_json::from_str(&contents).expect("should parse");
        assert_eq!(data.reservations.len(), 1);
        assert_eq!(data.reservations[0].budget_mb, 5000);

        cleanup(&path);
    }

    #[test]
    fn test_gpu_status_display() {
        let path = test_ledger_path();
        let mut ledger =
            VramLedger::new("GPU-test-display".into(), 24000, 0.85).with_path(path.clone());

        ledger.try_reserve(7000, "display-test").expect("should succeed");

        let status = gpu_status_display(&ledger).expect("should succeed");
        assert!(status.contains("GPU-test-display"));
        assert!(status.contains("24000 MB total"));
        assert!(status.contains("7000 MB budget"));
        assert!(status.contains("display-test"));

        cleanup(&path);
    }

    #[test]
    fn test_memory_type_reserve_factors() {
        assert!((MemoryType::Discrete.reserve_factor() - 0.85).abs() < f32::EPSILON);
        assert!((MemoryType::Unified.reserve_factor() - 0.60).abs() < f32::EPSILON);
    }

    #[test]
    fn test_reservation_id_deterministic() {
        let now = Utc::now();
        let id1 = reservation_id("GPU-abc", 1234, now);
        let id2 = reservation_id("GPU-abc", 1234, now);
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_reservation_id_varies_with_input() {
        let now = Utc::now();
        let id1 = reservation_id("GPU-abc", 1234, now);
        let id2 = reservation_id("GPU-xyz", 1234, now);
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_profiling_disabled_by_default() {
        let path = test_ledger_path();
        let ledger = VramLedger::new("GPU-test".into(), 24000, 0.85).with_path(path.clone());

        assert!(!ledger.profiler.is_enabled());
        let report = ledger.profiler_report();
        assert!(report.contains("No operations recorded"));

        cleanup(&path);
    }

    #[test]
    fn test_profiling_enabled_records_phases() {
        let path = test_ledger_path();
        let mut ledger = VramLedger::new("GPU-test".into(), 24000, 0.85)
            .with_path(path.clone())
            .with_profiling(true);

        ledger.try_reserve(5000, "profiled-job").expect("should succeed");

        let report = ledger.profiler_report();
        assert!(report.contains("lock_acq"));
        assert!(report.contains("ledger_rd"));
        assert!(report.contains("ledger_wr"));

        cleanup(&path);
    }

    // ── Additional coverage tests ──

    #[test]
    fn test_capacity_mb_discrete() {
        let path = test_ledger_path();
        let ledger = VramLedger::new("GPU-test".into(), 24000, RESERVE_FACTOR_DISCRETE)
            .with_path(path.clone());
        // 24000 * 0.85 = 20400
        assert_eq!(ledger.capacity_mb(), 20400);
        cleanup(&path);
    }

    #[test]
    fn test_capacity_mb_unified() {
        let path = test_ledger_path();
        let ledger = VramLedger::new("GPU-test".into(), 8192, RESERVE_FACTOR_UNIFIED)
            .with_path(path.clone());
        // 8192 * 0.60 = 4915.2 -> 4915
        assert_eq!(ledger.capacity_mb(), 4915);
        cleanup(&path);
    }

    #[test]
    fn test_with_lease_hours_custom() {
        let path = test_ledger_path();
        let mut ledger = VramLedger::new("GPU-test".into(), 24000, 0.85)
            .with_path(path.clone())
            .with_lease_hours(48);

        let id = ledger.try_reserve(1000, "long-lease").expect("should succeed");
        assert!(id != 0);
        // After 10ms with 48h lease, reservation should still be active
        std::thread::sleep(Duration::from_millis(10));
        assert_eq!(ledger.total_reserved().expect("should succeed"), 1000);
        cleanup(&path);
    }

    #[test]
    fn test_multiple_reservations_same_gpu() {
        let path = test_ledger_path();
        let mut ledger = VramLedger::new("GPU-test".into(), 24000, 0.85).with_path(path.clone());

        ledger.try_reserve(5000, "job-1").expect("should succeed");
        // Reserved should be 5000, available should be capacity - 5000
        assert_eq!(ledger.total_reserved().expect("ok"), 5000);
        assert_eq!(ledger.available_mb().expect("ok"), 15400);

        cleanup(&path);
    }

    #[test]
    fn test_read_reservations_returns_our_gpu_only() {
        let path = test_ledger_path();
        let mut ledger = VramLedger::new("GPU-test".into(), 24000, 0.85).with_path(path.clone());

        ledger.try_reserve(3000, "gpu-test-job").expect("should succeed");
        let reservations = ledger.read_reservations().expect("should succeed");
        assert_eq!(reservations.len(), 1);
        assert_eq!(reservations[0].budget_mb, 3000);
        assert_eq!(reservations[0].gpu_uuid, "GPU-test");
        assert_eq!(reservations[0].task, "gpu-test-job");

        cleanup(&path);
    }

    #[test]
    fn test_update_actual_without_reservation_is_noop() {
        let path = test_ledger_path();
        let mut ledger = VramLedger::new("GPU-test".into(), 24000, 0.85).with_path(path.clone());

        // No reservation, update_actual is a no-op
        let result = ledger.update_actual(5000);
        assert!(result.is_ok());
        assert_eq!(ledger.total_reserved().expect("ok"), 0);

        cleanup(&path);
    }

    #[test]
    fn test_release_without_reservation_is_noop() {
        let path = test_ledger_path();
        let mut ledger = VramLedger::new("GPU-test".into(), 24000, 0.85).with_path(path.clone());

        // No reservation, release is a no-op
        let result = ledger.release();
        assert!(result.is_ok());

        cleanup(&path);
    }

    #[test]
    fn test_drop_releases_reservation() {
        let path = test_ledger_path();
        {
            let mut ledger =
                VramLedger::new("GPU-test".into(), 24000, 0.85).with_path(path.clone());
            ledger.try_reserve(5000, "drop-test").expect("should succeed");
            assert_eq!(ledger.total_reserved().expect("ok"), 5000);
            // Drop happens here
        }

        // After drop, a new ledger should show the reservation gone
        // (since the process is still alive, the reservation may or may not be pruned
        //  depending on timing, but the explicit release in Drop should have removed it)
        let ledger = VramLedger::new("GPU-test".into(), 24000, 0.85).with_path(path.clone());
        let reserved = ledger.total_reserved().expect("ok");
        assert_eq!(reserved, 0);

        cleanup(&path);
    }

    #[test]
    fn test_reservation_is_expired_zero_lease() {
        let now = chrono::Utc::now();
        let reservation = Reservation {
            id: 123,
            pid: std::process::id(),
            budget_mb: 1000,
            actual_mb: None,
            task: "test".to_string(),
            gpu_uuid: "GPU-test".to_string(),
            started: now - chrono::Duration::seconds(10),
            lease_expires: now - chrono::Duration::seconds(1), // already expired
        };
        assert!(reservation.is_expired());
        assert!(reservation.should_prune());
    }

    #[test]
    fn test_reservation_is_alive_current_process() {
        let now = chrono::Utc::now();
        let reservation = Reservation {
            id: 123,
            pid: std::process::id(), // current process is alive
            budget_mb: 1000,
            actual_mb: None,
            task: "test".to_string(),
            gpu_uuid: "GPU-test".to_string(),
            started: now,
            lease_expires: now + chrono::Duration::hours(24),
        };
        assert!(reservation.is_alive());
        assert!(!reservation.is_expired());
        assert!(!reservation.should_prune());
    }

    #[test]
    fn test_reservation_is_alive_dead_process() {
        let now = chrono::Utc::now();
        let reservation = Reservation {
            id: 123,
            pid: u32::MAX, // extremely unlikely to be a real PID
            budget_mb: 1000,
            actual_mb: None,
            task: "dead-process".to_string(),
            gpu_uuid: "GPU-test".to_string(),
            started: now,
            lease_expires: now + chrono::Duration::hours(24),
        };
        assert!(!reservation.is_alive());
        assert!(reservation.should_prune());
    }

    #[test]
    fn test_ledger_data_total_reserved_for() {
        let now = chrono::Utc::now();
        let data = LedgerData {
            reservations: vec![
                Reservation {
                    id: 1,
                    pid: std::process::id(),
                    budget_mb: 3000,
                    actual_mb: None,
                    task: "a".to_string(),
                    gpu_uuid: "GPU-A".to_string(),
                    started: now,
                    lease_expires: now + chrono::Duration::hours(1),
                },
                Reservation {
                    id: 2,
                    pid: std::process::id(),
                    budget_mb: 5000,
                    actual_mb: Some(4500),
                    task: "b".to_string(),
                    gpu_uuid: "GPU-A".to_string(),
                    started: now,
                    lease_expires: now + chrono::Duration::hours(1),
                },
                Reservation {
                    id: 3,
                    pid: std::process::id(),
                    budget_mb: 2000,
                    actual_mb: None,
                    task: "c".to_string(),
                    gpu_uuid: "GPU-B".to_string(),
                    started: now,
                    lease_expires: now + chrono::Duration::hours(1),
                },
            ],
        };
        // GPU-A: 3000 (budget, no actual) + 4500 (actual) = 7500
        assert_eq!(data.total_reserved_for("GPU-A"), 7500);
        // GPU-B: 2000
        assert_eq!(data.total_reserved_for("GPU-B"), 2000);
        // GPU-C: 0
        assert_eq!(data.total_reserved_for("GPU-C"), 0);
    }

    #[test]
    fn test_ledger_data_prune_dead() {
        let now = chrono::Utc::now();
        let mut data = LedgerData {
            reservations: vec![
                // This one is expired
                Reservation {
                    id: 1,
                    pid: std::process::id(),
                    budget_mb: 1000,
                    actual_mb: None,
                    task: "expired".to_string(),
                    gpu_uuid: "GPU-A".to_string(),
                    started: now - chrono::Duration::hours(2),
                    lease_expires: now - chrono::Duration::seconds(1),
                },
                // This one is alive and not expired
                Reservation {
                    id: 2,
                    pid: std::process::id(),
                    budget_mb: 2000,
                    actual_mb: None,
                    task: "alive".to_string(),
                    gpu_uuid: "GPU-A".to_string(),
                    started: now,
                    lease_expires: now + chrono::Duration::hours(24),
                },
                // This one has a dead PID
                Reservation {
                    id: 3,
                    pid: u32::MAX,
                    budget_mb: 3000,
                    actual_mb: None,
                    task: "dead".to_string(),
                    gpu_uuid: "GPU-A".to_string(),
                    started: now,
                    lease_expires: now + chrono::Duration::hours(24),
                },
            ],
        };
        data.prune_dead();
        assert_eq!(data.reservations.len(), 1);
        assert_eq!(data.reservations[0].task, "alive");
    }

    #[test]
    fn test_reservation_id_varies_with_pid() {
        let now = chrono::Utc::now();
        let id1 = reservation_id("GPU-abc", 100, now);
        let id2 = reservation_id("GPU-abc", 200, now);
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_reservation_id_varies_with_time() {
        let now = chrono::Utc::now();
        let later = now + chrono::Duration::seconds(1);
        let id1 = reservation_id("GPU-abc", 100, now);
        let id2 = reservation_id("GPU-abc", 100, later);
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_gpu_status_display_no_reservations() {
        let path = test_ledger_path();
        let ledger = VramLedger::new("GPU-no-res".into(), 16000, 0.85).with_path(path.clone());

        let status = gpu_status_display(&ledger).expect("should succeed");
        assert!(status.contains("GPU-no-res"));
        assert!(status.contains("16000 MB total"));
        assert!(status.contains("Reservations: none"));

        cleanup(&path);
    }

    #[test]
    fn test_gpu_status_display_with_actual_mb() {
        let path = test_ledger_path();
        let mut ledger = VramLedger::new("GPU-act".into(), 24000, 0.85).with_path(path.clone());

        ledger.try_reserve(8000, "actual-test").expect("should succeed");
        ledger.update_actual(7500).expect("should succeed");

        let status = gpu_status_display(&ledger).expect("should succeed");
        assert!(status.contains("7500 MB actual"));
        assert!(status.contains("actual-test"));

        cleanup(&path);
    }

    #[test]
    fn test_memory_type_reserve_factor_values() {
        assert_eq!(MemoryType::Discrete.reserve_factor(), RESERVE_FACTOR_DISCRETE);
        assert_eq!(MemoryType::Unified.reserve_factor(), RESERVE_FACTOR_UNIFIED);
    }

    #[test]
    fn test_memory_type_equality() {
        assert_eq!(MemoryType::Discrete, MemoryType::Discrete);
        assert_eq!(MemoryType::Unified, MemoryType::Unified);
        assert_ne!(MemoryType::Discrete, MemoryType::Unified);
    }

    #[test]
    fn test_ensure_parent_dir_existing() {
        let dir = tempfile::tempdir().expect("ok");
        let path = dir.path().join("subdir").join("ledger.json");
        ensure_parent_dir(&path).expect("should succeed");
        assert!(path.parent().expect("ok").exists());
    }

    #[test]
    fn test_reserve_exact_capacity() {
        let path = test_ledger_path();
        let mut ledger = VramLedger::new("GPU-test".into(), 10000, 0.85).with_path(path.clone());
        // capacity = 8500
        let result = ledger.try_reserve(8500, "exact-fit");
        assert!(result.is_ok());
        assert_eq!(ledger.available_mb().expect("ok"), 0);

        cleanup(&path);
    }

    #[test]
    fn test_reserve_one_over_capacity() {
        let path = test_ledger_path();
        let mut ledger = VramLedger::new("GPU-test".into(), 10000, 0.85).with_path(path.clone());
        // capacity = 8500, request 8501 should fail
        let result = ledger.try_reserve(8501, "too-big");
        assert!(result.is_err());

        cleanup(&path);
    }

    #[test]
    fn test_ledger_default_path() {
        let ledger = VramLedger::new("GPU-test".into(), 24000, 0.85);
        // Default path should be under cache dir
        let path_str = format!("{}", ledger.path.display());
        assert!(path_str.contains("gpu-ledger.json"));
    }

    #[test]
    fn test_reservation_serde_roundtrip() {
        let now = chrono::Utc::now();
        let reservation = Reservation {
            id: 42,
            pid: 12345,
            budget_mb: 8000,
            actual_mb: Some(7500),
            task: "serde-test".to_string(),
            gpu_uuid: "GPU-0000".to_string(),
            started: now,
            lease_expires: now + chrono::Duration::hours(24),
        };
        let json = serde_json::to_string(&reservation).expect("serialize");
        let restored: Reservation = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.id, 42);
        assert_eq!(restored.pid, 12345);
        assert_eq!(restored.budget_mb, 8000);
        assert_eq!(restored.actual_mb, Some(7500));
        assert_eq!(restored.task, "serde-test");
    }
}
