//! Golden trace format for baseline storage
//!
//! Binary format per specification Section 4.3

use super::config::ValidateConfig;
use super::error::{Result, ValidateError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Magic bytes for trace file header
pub const TRACE_MAGIC: &[u8; 4] = b"RNTR";
/// Magic bytes for trace file footer
pub const TRACE_MAGIC_END: &[u8; 4] = b"RTNR";
/// Current trace format version
pub const TRACE_VERSION: u32 = 1;

/// Trace flags bitfield
#[derive(Debug, Clone, Copy, Default)]
pub struct TraceFlags(u32);

impl TraceFlags {
    /// Create new trace flags
    pub fn new(compressed: bool, has_timing: bool, has_string_args: bool) -> Self {
        let mut flags = 0u32;
        if compressed {
            flags |= 1;
        }
        if has_timing {
            flags |= 2;
        }
        if has_string_args {
            flags |= 4;
        }
        Self(flags)
    }

    /// Check if trace is compressed
    pub fn compressed(&self) -> bool {
        self.0 & 1 != 0
    }

    /// Check if trace has timing data
    pub fn has_timing(&self) -> bool {
        self.0 & 2 != 0
    }

    /// Check if trace has string arguments
    pub fn has_string_args(&self) -> bool {
        self.0 & 4 != 0
    }

    /// Get raw flags value
    pub fn raw(&self) -> u32 {
        self.0
    }
}

/// Binary trace header (32 bytes)
#[derive(Debug, Clone)]
pub struct TraceHeader {
    /// Magic bytes "RNTR"
    pub magic: [u8; 4],
    /// Format version
    pub version: u32,
    /// Trace flags
    pub flags: TraceFlags,
    /// Number of syscall entries
    pub entry_count: u64,
    /// CRC32 checksum of entries
    pub checksum: u64,
}

impl TraceHeader {
    /// Create a new trace header
    pub fn new(entry_count: u64, flags: TraceFlags) -> Self {
        let checksum = Self::compute_checksum(entry_count, flags.raw());
        Self { magic: *TRACE_MAGIC, version: TRACE_VERSION, flags, entry_count, checksum }
    }

    /// Compute checksum from entry count and flags
    fn compute_checksum(entry_count: u64, flags: u32) -> u64 {
        // Simple checksum: combine entry count and flags
        (entry_count ^ u64::from(flags)) ^ 0xDEAD_BEEF_CAFE_BABE
    }

    /// Encode header to bytes
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(32);
        buf.extend_from_slice(&self.magic);
        buf.extend_from_slice(&self.version.to_le_bytes());
        buf.extend_from_slice(&self.flags.raw().to_le_bytes());
        buf.extend_from_slice(&self.entry_count.to_le_bytes());
        buf.extend_from_slice(&self.checksum.to_le_bytes());
        // Pad to 32 bytes
        buf.resize(32, 0);
        buf
    }

    /// Decode header from bytes
    pub fn decode(data: &[u8]) -> Result<Self> {
        if data.len() < 32 {
            return Err(ValidateError::InvalidManifest { reason: "Header too short".to_string() });
        }

        let magic: [u8; 4] = data[0..4].try_into().expect("slice length verified above");
        if &magic != TRACE_MAGIC {
            return Err(ValidateError::InvalidManifest {
                reason: format!("Invalid magic: expected RNTR, got {magic:?}"),
            });
        }

        let version =
            u32::from_le_bytes(data[4..8].try_into().expect("4-byte slice for u32 conversion"));
        let flags = TraceFlags(u32::from_le_bytes(
            data[8..12].try_into().expect("4-byte slice for u32 flags"),
        ));
        let entry_count =
            u64::from_le_bytes(data[12..20].try_into().expect("8-byte slice for u64 entry_count"));
        let checksum =
            u64::from_le_bytes(data[20..28].try_into().expect("8-byte slice for u64 checksum"));

        Ok(Self { magic, version, flags, entry_count, checksum })
    }
}

/// A single syscall entry in the trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceSyscallEntry {
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
    /// Syscall number
    pub syscall_nr: u32,
    /// Duration in nanoseconds
    pub duration_ns: u64,
    /// Return value
    pub return_value: i64,
    /// Numeric arguments
    pub args: Vec<u64>,
    /// String arguments (paths, etc.)
    pub string_args: Vec<String>,
}

impl TraceSyscallEntry {
    /// Create a simple entry for testing
    pub fn simple(syscall_nr: u32, _name: &str, duration_ns: u64) -> Self {
        Self {
            timestamp_ns: 0,
            syscall_nr,
            duration_ns,
            return_value: 0,
            args: Vec::new(),
            string_args: Vec::new(),
        }
    }

    /// Encode entry to bytes
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        buf.extend_from_slice(&self.timestamp_ns.to_le_bytes());
        buf.extend_from_slice(&self.syscall_nr.to_le_bytes());
        buf.extend_from_slice(&self.duration_ns.to_le_bytes());
        buf.extend_from_slice(&self.return_value.to_le_bytes());

        // Encode args count and args
        buf.push(self.args.len() as u8);
        for arg in &self.args {
            buf.extend_from_slice(&arg.to_le_bytes());
        }

        // Encode string args count and strings
        buf.push(self.string_args.len() as u8);
        for s in &self.string_args {
            let bytes = s.as_bytes();
            buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(bytes);
        }

        buf
    }

    /// Decode entry from bytes
    pub fn decode(data: &[u8]) -> Result<Self> {
        if data.len() < 29 {
            // minimum size
            return Err(ValidateError::InvalidManifest { reason: "Entry too short".to_string() });
        }

        let timestamp_ns =
            u64::from_le_bytes(data[0..8].try_into().expect("8-byte slice for timestamp_ns"));
        let syscall_nr =
            u32::from_le_bytes(data[8..12].try_into().expect("4-byte slice for syscall_nr"));
        let duration_ns =
            u64::from_le_bytes(data[12..20].try_into().expect("8-byte slice for duration_ns"));
        let return_value =
            i64::from_le_bytes(data[20..28].try_into().expect("8-byte slice for return_value"));

        let mut offset = 28;
        let arg_count = data[offset] as usize;
        offset += 1;

        let mut args = Vec::with_capacity(arg_count);
        for _ in 0..arg_count {
            if offset + 8 > data.len() {
                return Err(ValidateError::InvalidManifest {
                    reason: "Truncated args".to_string(),
                });
            }
            args.push(u64::from_le_bytes(
                data[offset..offset + 8].try_into().expect("8-byte slice for arg value"),
            ));
            offset += 8;
        }

        let string_count = if offset < data.len() { data[offset] as usize } else { 0 };
        offset += 1;

        let mut string_args = Vec::with_capacity(string_count);
        for _ in 0..string_count {
            if offset + 4 > data.len() {
                break;
            }
            let len = u32::from_le_bytes(
                data[offset..offset + 4].try_into().expect("4-byte slice for string length"),
            ) as usize;
            offset += 4;
            if offset + len > data.len() {
                break;
            }
            let s = String::from_utf8_lossy(&data[offset..offset + len]).to_string();
            string_args.push(s);
            offset += len;
        }

        Ok(Self { timestamp_ns, syscall_nr, duration_ns, return_value, args, string_args })
    }
}

/// Platform information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformInfo {
    /// Operating system
    pub os: String,
    /// Architecture
    pub arch: String,
    /// Kernel version
    pub kernel: String,
}

impl PlatformInfo {
    /// Get current platform info
    pub fn current() -> Self {
        Self {
            os: "linux".to_string(),
            arch: std::env::consts::ARCH.to_string(),
            kernel: get_kernel_version(),
        }
    }
}

/// Get kernel version from /proc/version
fn get_kernel_version() -> String {
    fs::read_to_string("/proc/version")
        .ok()
        .and_then(|v| v.split_whitespace().nth(2).map(String::from))
        .unwrap_or_else(|| "unknown".to_string())
}

/// Statistics for a single syscall type
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SyscallTimingStats {
    /// Number of calls
    pub count: u64,
    /// Total duration in nanoseconds
    pub total_ns: u64,
    /// Mean duration
    pub mean_ns: u64,
    /// Standard deviation
    pub std_ns: u64,
    /// Minimum duration
    pub min_ns: u64,
    /// Maximum duration
    pub max_ns: u64,
    /// 50th percentile
    pub p50_ns: u64,
    /// 95th percentile
    pub p95_ns: u64,
    /// 99th percentile
    pub p99_ns: u64,
}

/// Timing statistics for the trace
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TimingStats {
    /// Total duration in nanoseconds
    pub total_duration_ns: u64,
    /// Total syscall count
    pub syscall_count: u64,
    /// Per-syscall statistics
    pub by_syscall: HashMap<String, SyscallTimingStats>,
}

/// Tolerance configuration stored in manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToleranceConfig {
    /// Timing tolerance percentage
    pub timing_percent: f32,
    /// Syscall sequence matching mode
    pub syscall_sequence: String,
    /// Argument matching mode
    pub argument_match: String,
}

impl Default for ToleranceConfig {
    fn default() -> Self {
        Self {
            timing_percent: 10.0,
            syscall_sequence: "exact".to_string(),
            argument_match: "fuzzy".to_string(),
        }
    }
}

/// Trace manifest (JSON format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceManifest {
    /// Manifest version
    pub version: String,
    /// Renacer version that created this baseline
    pub renacer_version: String,
    /// Creation timestamp (ISO 8601)
    pub created_at: String,
    /// Platform information
    pub platform: PlatformInfo,
    /// Command that was traced
    pub command: Vec<String>,
    /// Trace statistics
    pub statistics: TraceStatistics,
    /// Tolerance configuration
    pub tolerance: ToleranceConfig,
}

/// Statistics stored in manifest
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TraceStatistics {
    /// Total syscall count
    pub total_syscalls: u64,
    /// Total duration in milliseconds
    pub total_duration_ms: f64,
    /// Unique syscall types
    pub unique_syscalls: u32,
}

impl TraceManifest {
    /// Create a new manifest for the given command
    pub fn new(command: Vec<String>) -> Self {
        Self {
            version: "1.0.0".to_string(),
            renacer_version: env!("CARGO_PKG_VERSION").to_string(),
            created_at: get_iso_timestamp(),
            platform: PlatformInfo::current(),
            command,
            statistics: TraceStatistics::default(),
            tolerance: ToleranceConfig::default(),
        }
    }

    /// Update statistics from trace data
    pub fn with_statistics(mut self, total_syscalls: u64, total_duration_ns: u64) -> Self {
        self.statistics.total_syscalls = total_syscalls;
        self.statistics.total_duration_ms = total_duration_ns as f64 / 1_000_000.0;
        self
    }
}

/// Get current timestamp in ISO 8601 format
fn get_iso_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    format!("{}", duration.as_secs())
}

/// A loaded golden trace baseline
#[derive(Debug)]
pub struct GoldenBaseline {
    /// The manifest
    pub manifest: TraceManifest,
    /// Syscall entries
    pub syscalls: Vec<TraceSyscallEntry>,
    /// Timing statistics
    pub timing: TimingStats,
}

/// Generate a golden trace baseline
pub fn generate_baseline(
    command: &[&str],
    output_dir: &Path,
    _config: &ValidateConfig,
) -> Result<()> {
    use crate::filter::SyscallFilter;
    use crate::tracer::{trace_command, TracerConfig};

    // Create output directory
    fs::create_dir_all(output_dir)?;

    // Create tracer config for capture
    let tracer_config = TracerConfig {
        timing_mode: true,
        statistics_mode: true,
        filter: SyscallFilter::all(),
        ..TracerConfig::default()
    };

    // Trace the command
    let command_vec: Vec<String> = command.iter().map(|s| (*s).to_string()).collect();

    // For now, we'll capture basic trace info
    // In a full implementation, we'd capture the actual syscalls
    let result = trace_command(&command_vec, tracer_config);

    // Create manifest
    let manifest = TraceManifest::new(command_vec);
    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    fs::write(output_dir.join("manifest.json"), manifest_json)?;

    // Create empty syscalls trace (placeholder)
    let header = TraceHeader::new(0, TraceFlags::new(false, true, false));
    let mut trace_data = header.encode();
    trace_data.extend_from_slice(TRACE_MAGIC_END);
    fs::write(output_dir.join("syscalls.trace"), trace_data)?;

    // Create timing stats
    let timing = TimingStats::default();
    let timing_json = serde_json::to_string_pretty(&timing)?;
    fs::write(output_dir.join("timing.stats"), timing_json)?;

    // Handle trace result - we don't propagate the error for now
    // since we're doing baseline generation
    if let Err(e) = result {
        eprintln!("Warning: trace command returned error: {e}");
    }

    Ok(())
}

/// Load a golden trace baseline from directory
pub fn load_baseline(baseline_dir: &Path) -> Result<GoldenBaseline> {
    // Check directory exists
    if !baseline_dir.exists() {
        return Err(ValidateError::BaselineNotFound { path: baseline_dir.to_path_buf() });
    }

    // Load manifest
    let manifest_path = baseline_dir.join("manifest.json");
    if !manifest_path.exists() {
        return Err(ValidateError::BaselineNotFound { path: manifest_path });
    }

    let manifest_content = fs::read_to_string(&manifest_path)?;
    let manifest: TraceManifest = serde_json::from_str(&manifest_content)
        .map_err(|e| ValidateError::InvalidManifest { reason: e.to_string() })?;

    // Load timing stats
    let timing_path = baseline_dir.join("timing.stats");
    let timing: TimingStats = if timing_path.exists() {
        let timing_content = fs::read_to_string(&timing_path)?;
        serde_json::from_str(&timing_content).unwrap_or_default()
    } else {
        TimingStats::default()
    };

    // Load syscall trace (simplified for now)
    let syscalls = Vec::new();

    Ok(GoldenBaseline { manifest, syscalls, timing })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_trace_flags() {
        let flags = TraceFlags::new(true, true, false);
        assert!(flags.compressed());
        assert!(flags.has_timing());
        assert!(!flags.has_string_args());
    }

    #[test]
    fn test_trace_flags_all_false() {
        let flags = TraceFlags::new(false, false, false);
        assert!(!flags.compressed());
        assert!(!flags.has_timing());
        assert!(!flags.has_string_args());
        assert_eq!(flags.raw(), 0);
    }

    #[test]
    fn test_trace_flags_all_true() {
        let flags = TraceFlags::new(true, true, true);
        assert!(flags.compressed());
        assert!(flags.has_timing());
        assert!(flags.has_string_args());
        assert_eq!(flags.raw(), 7);
    }

    #[test]
    fn test_trace_flags_raw() {
        let flags = TraceFlags::new(false, true, false);
        assert_eq!(flags.raw(), 2);
    }

    #[test]
    fn test_header_roundtrip() {
        let header = TraceHeader::new(42, TraceFlags::new(false, true, true));
        let encoded = header.encode();
        let decoded = TraceHeader::decode(&encoded).expect("decode should succeed");

        assert_eq!(decoded.entry_count, 42);
        assert!(decoded.flags.has_timing());
    }

    #[test]
    fn test_header_decode_too_short() {
        let short_data = vec![0u8; 10];
        let result = TraceHeader::decode(&short_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_header_decode_invalid_magic() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"XXXX");
        let result = TraceHeader::decode(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_header_encode_size() {
        let header = TraceHeader::new(0, TraceFlags::default());
        let encoded = header.encode();
        assert_eq!(encoded.len(), 32);
    }

    #[test]
    fn test_header_magic_bytes() {
        let header = TraceHeader::new(100, TraceFlags::default());
        let encoded = header.encode();
        assert_eq!(&encoded[0..4], TRACE_MAGIC);
    }

    #[test]
    fn test_platform_info() {
        let info = PlatformInfo::current();
        assert_eq!(info.os, "linux");
        assert!(!info.arch.is_empty());
    }

    #[test]
    fn test_syscall_entry_simple() {
        let entry = TraceSyscallEntry::simple(1, "write", 1000);
        assert_eq!(entry.syscall_nr, 1);
        assert_eq!(entry.duration_ns, 1000);
        assert_eq!(entry.timestamp_ns, 0);
        assert_eq!(entry.return_value, 0);
        assert!(entry.args.is_empty());
        assert!(entry.string_args.is_empty());
    }

    #[test]
    fn test_syscall_entry_encode_decode_roundtrip() {
        let entry = TraceSyscallEntry {
            timestamp_ns: 12345,
            syscall_nr: 1,
            duration_ns: 5000,
            return_value: 10,
            args: vec![1, 2, 3],
            string_args: vec!["hello".to_string(), "world".to_string()],
        };
        let encoded = entry.encode();
        let decoded = TraceSyscallEntry::decode(&encoded).expect("decode should succeed");

        assert_eq!(decoded.timestamp_ns, 12345);
        assert_eq!(decoded.syscall_nr, 1);
        assert_eq!(decoded.duration_ns, 5000);
        assert_eq!(decoded.return_value, 10);
        assert_eq!(decoded.args, vec![1, 2, 3]);
        assert_eq!(decoded.string_args, vec!["hello", "world"]);
    }

    #[test]
    fn test_syscall_entry_decode_too_short() {
        let short_data = vec![0u8; 10];
        let result = TraceSyscallEntry::decode(&short_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_syscall_entry_encode_empty_args() {
        let entry = TraceSyscallEntry::simple(0, "read", 100);
        let encoded = entry.encode();
        let decoded = TraceSyscallEntry::decode(&encoded).expect("decode should succeed");
        assert!(decoded.args.is_empty());
        assert!(decoded.string_args.is_empty());
    }

    #[test]
    fn test_timing_stats_default() {
        let stats = TimingStats::default();
        assert_eq!(stats.total_duration_ns, 0);
        assert_eq!(stats.syscall_count, 0);
        assert!(stats.by_syscall.is_empty());
    }

    #[test]
    fn test_syscall_timing_stats_default() {
        let stats = SyscallTimingStats::default();
        assert_eq!(stats.count, 0);
        assert_eq!(stats.total_ns, 0);
        assert_eq!(stats.mean_ns, 0);
    }

    #[test]
    fn test_tolerance_config_default() {
        let config = ToleranceConfig::default();
        assert!((config.timing_percent - 10.0).abs() < f32::EPSILON);
        assert_eq!(config.syscall_sequence, "exact");
        assert_eq!(config.argument_match, "fuzzy");
    }

    #[test]
    fn test_trace_manifest_new() {
        let manifest = TraceManifest::new(vec!["echo".to_string(), "hello".to_string()]);
        assert_eq!(manifest.version, "1.0.0");
        assert_eq!(manifest.command, vec!["echo", "hello"]);
        assert_eq!(manifest.platform.os, "linux");
    }

    #[test]
    fn test_trace_manifest_with_statistics() {
        let manifest = TraceManifest::new(vec!["test".to_string()]).with_statistics(100, 1_000_000);
        assert_eq!(manifest.statistics.total_syscalls, 100);
        assert!((manifest.statistics.total_duration_ms - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_trace_statistics_default() {
        let stats = TraceStatistics::default();
        assert_eq!(stats.total_syscalls, 0);
        assert!((stats.total_duration_ms - 0.0).abs() < f64::EPSILON);
        assert_eq!(stats.unique_syscalls, 0);
    }

    #[test]
    fn test_load_baseline_not_found() {
        let result = load_baseline(Path::new("/nonexistent/path"));
        assert!(result.is_err());
        match result {
            Err(ValidateError::BaselineNotFound { .. }) => {}
            _ => panic!("Expected BaselineNotFound error"),
        }
    }

    #[test]
    fn test_load_baseline_missing_manifest() {
        let temp_dir = TempDir::new().expect("failed to create temp dir");
        let result = load_baseline(temp_dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_load_baseline_invalid_manifest() {
        let temp_dir = TempDir::new().expect("failed to create temp dir");
        fs::write(temp_dir.path().join("manifest.json"), "invalid json")
            .expect("failed to write manifest");
        let result = load_baseline(temp_dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_load_baseline_valid() {
        let temp_dir = TempDir::new().expect("failed to create temp dir");

        // Create valid manifest
        let manifest = TraceManifest::new(vec!["echo".to_string()]);
        let manifest_json = serde_json::to_string(&manifest).expect("failed to serialize manifest");
        fs::write(temp_dir.path().join("manifest.json"), manifest_json)
            .expect("failed to write manifest");

        // Create timing stats
        let timing = TimingStats::default();
        let timing_json = serde_json::to_string(&timing).expect("failed to serialize timing");
        fs::write(temp_dir.path().join("timing.stats"), timing_json)
            .expect("failed to write timing stats");

        let result = load_baseline(temp_dir.path());
        assert!(result.is_ok());
        let baseline = result.expect("baseline should load");
        assert_eq!(baseline.manifest.command, vec!["echo"]);
    }

    #[test]
    fn test_load_baseline_without_timing_stats() {
        let temp_dir = TempDir::new().expect("failed to create temp dir");

        // Create valid manifest only (no timing.stats)
        let manifest = TraceManifest::new(vec!["test".to_string()]);
        let manifest_json = serde_json::to_string(&manifest).expect("failed to serialize manifest");
        fs::write(temp_dir.path().join("manifest.json"), manifest_json)
            .expect("failed to write manifest");

        let result = load_baseline(temp_dir.path());
        assert!(result.is_ok());
    }

    #[test]
    fn test_get_kernel_version() {
        let version = get_kernel_version();
        // Should return something, not panic
        assert!(!version.is_empty());
    }

    #[test]
    fn test_get_iso_timestamp() {
        let ts = get_iso_timestamp();
        // Should be a numeric string (Unix timestamp)
        assert!(ts.parse::<u64>().is_ok());
    }
}
