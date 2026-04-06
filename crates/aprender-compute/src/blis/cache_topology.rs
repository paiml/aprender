//! Dynamic cache topology detection for BLIS blocking parameters.
//!
//! Contract: cgp-dynamic-cache-v1.yaml (C-CACHE-001 through C-CACHE-006)
//!
//! Reads `/sys/devices/system/cpu/cpu0/cache/` at runtime to determine
//! L1D, L2, L3 sizes. Computes optimal MC/KC/NC for the current CPU.
//! Falls back to hardcoded defaults if `/sys/` is not readable (C-CACHE-006).

use std::sync::OnceLock;

/// Detected cache sizes in bytes.
#[derive(Debug, Clone, Copy)]
pub struct CacheTopology {
    /// L1 data cache size in bytes (typically 32-48 KB)
    pub l1d_bytes: usize,
    /// L2 unified cache size in bytes (typically 256 KB - 2 MB)
    pub l2_bytes: usize,
    /// L3 unified cache size in bytes (typically 4-64 MB)
    pub l3_bytes: usize,
}

/// BLIS blocking parameters computed from cache topology.
#[derive(Debug, Clone, Copy)]
pub struct BlisBlocking {
    pub mr: usize,
    pub nr: usize,
    pub mc: usize,
    pub kc: usize,
    pub nc: usize,
    /// Whether these were computed from detected topology or are defaults
    pub dynamic: bool,
}

/// Hardcoded fallback for 8×16 path (small N, C-CACHE-006).
const DEFAULT_BLOCKING_8X16: BlisBlocking =
    BlisBlocking { mr: 8, nr: 16, mc: 64, kc: 256, nc: 1024, dynamic: false };

/// Cached topology — detected once, reused forever.
static TOPOLOGY: OnceLock<CacheTopology> = OnceLock::new();
static BLOCKING_8X32: OnceLock<BlisBlocking> = OnceLock::new();

/// Detect cache topology from /sys/. Returns None if not readable.
fn detect_from_sys() -> Option<CacheTopology> {
    let mut l1d = 0usize;
    let mut l2 = 0usize;
    let mut l3 = 0usize;

    for idx in 0..4 {
        let base = format!("/sys/devices/system/cpu/cpu0/cache/index{idx}");
        let size_str = std::fs::read_to_string(format!("{base}/size")).ok()?;
        let type_str = std::fs::read_to_string(format!("{base}/type")).ok()?;

        let size_str = size_str.trim();
        let type_str = type_str.trim();

        // Parse size: "32K" → 32768, "1024K" → 1048576
        let size_bytes = parse_cache_size(size_str)?;

        match (idx, type_str) {
            (0, "Data") => l1d = size_bytes,
            (1, _) => {} // L1 instruction, skip
            (2, "Unified") => l2 = size_bytes,
            (3, "Unified") => l3 = size_bytes,
            _ => {}
        }
    }

    if l1d > 0 && l2 > 0 {
        Some(CacheTopology { l1d_bytes: l1d, l2_bytes: l2, l3_bytes: l3 })
    } else {
        None
    }
}

/// Parse "32K" → 32768, "1024K" → 1048576, "32768K" → 33554432.
fn parse_cache_size(s: &str) -> Option<usize> {
    let s = s.trim();
    if let Some(kb) = s.strip_suffix('K') {
        kb.parse::<usize>().ok().map(|v| v * 1024)
    } else if let Some(mb) = s.strip_suffix('M') {
        mb.parse::<usize>().ok().map(|v| v * 1024 * 1024)
    } else {
        s.parse::<usize>().ok()
    }
}

/// Get detected cache topology (cached after first call).
pub fn topology() -> CacheTopology {
    *TOPOLOGY.get_or_init(|| {
        detect_from_sys().unwrap_or(CacheTopology {
            l1d_bytes: 32768,   // 32K default
            l2_bytes: 1048576,  // 1M default
            l3_bytes: 33554432, // 32M default
        })
    })
}

/// Compute BLIS blocking for 8×32 microkernel from cache topology.
///
/// Invariants (from contract cgp-dynamic-cache-v1.yaml):
/// - C-CACHE-001: mc * kc * 4 <= l2
/// - C-CACHE-002: kc * nr * 4 <= l1d
/// - C-CACHE-003: kc * nc * 4 <= l3 / 2
/// - C-CACHE-004: mc % mr == 0
/// - C-CACHE-005: nc % nr == 0
fn compute_blocking_8x32(topo: &CacheTopology) -> BlisBlocking {
    let mr = 8usize;
    let nr = 32usize;

    // C-CACHE-002: kc * nr * 4 <= l1d → kc <= l1d / (nr * 4)
    let kc_max = topo.l1d_bytes / (nr * 4);
    // Round down to power of 2, min 64
    let kc = kc_max.next_power_of_two().min(kc_max).max(64);

    // C-CACHE-001: mc * kc * 4 <= l2 → mc <= l2 / (kc * 4)
    let mc_max = topo.l2_bytes / (kc * 4);
    // Round down to multiple of MR.
    // EMPIRICAL: MC=96 outperforms MC=192/256 on Zen 4 (tested 2026-04-05).
    // Large MC increases A-packing overhead without proportional L2 benefit.
    // Cap at 12*MR=96 (matches empirically-tuned value).
    let mc = (mc_max / mr * mr).min(12 * mr).max(mr);

    // C-CACHE-003: kc * nc * 4 <= l3/2 → nc <= l3 / (2 * kc * 4)
    let nc_max = topo.l3_bytes / (2 * kc * 4);
    // Round down to multiple of NR, cap at 4096
    let nc = (nc_max / nr * nr).min(4096).max(nr);

    BlisBlocking { mr, nr, mc, kc, nc, dynamic: true }
}

/// Get optimal BLIS blocking for 8×32 microkernel (cached).
pub fn blocking_8x32() -> BlisBlocking {
    *BLOCKING_8X32.get_or_init(|| {
        let topo = topology();
        compute_blocking_8x32(&topo)
    })
}

/// Get optimal BLIS blocking for 8×48 codegen microkernel (cached).
/// 8×48: 24 accumulators, 24 FMAs/K-step (3× the 8×16 kernel).
/// KC is smaller (L1-limited at NR=48) but more FMAs per K-step may compensate.
pub fn blocking_8x48() -> BlisBlocking {
    static BLOCKING_8X48: OnceLock<BlisBlocking> = OnceLock::new();
    *BLOCKING_8X48.get_or_init(|| {
        let topo = topology();
        let mr = 8usize;
        let nr = 48usize;
        // KC: l1d / (nr * 4) = 32768 / 192 = 170, round down to power-of-2 = 128
        // Power-of-2 KC ensures aligned loop trips for vectorized packing.
        let kc_max = topo.l1d_bytes / (nr * 4);
        let kc = (kc_max.next_power_of_two() >> 1).max(64);
        let mc_max = topo.l2_bytes / (kc * 4);
        let mc = (mc_max / mr * mr).min(12 * mr).max(mr);
        let nc_max = topo.l3_bytes / (2 * kc * 4);
        let nc = (nc_max / nr * nr).min(4096).max(nr);
        BlisBlocking { mr, nr, mc, kc, nc, dynamic: true }
    })
}

/// Get optimal BLIS blocking for broadcast-B 64×6 microkernel (cached).
/// 64×6: 24 FMA accumulators — matches faer's nano-gemm register utilization.
/// NR=6 keeps B panel tiny → KC can be large (256-512).
pub fn blocking_64x6_bcast_b() -> BlisBlocking {
    static BLOCKING_64X6: OnceLock<BlisBlocking> = OnceLock::new();
    *BLOCKING_64X6.get_or_init(|| {
        let topo = topology();
        let mr = 64usize;
        let nr = 6usize;
        // KC: B panel = NR × KC × 4 = 24 × KC bytes. At KC=512: 12KB (fits L1 easily).
        // A panel = MR × KC × 4 = 256 × KC bytes. At KC=512: 128KB (fits L2).
        // Limit KC so A panel fits in L2: KC ≤ L2 / (MR × 4)
        let kc_max_l2 = topo.l2_bytes / (mr * 4);
        let kc_max_l1 = topo.l1d_bytes * 3 / 4 / (nr * 4); // B uses 3/4 of L1
        let kc = kc_max_l2.min(kc_max_l1).clamp(64, 512);
        // MC: number of rows per L2 tile. Since MR=64 is large, MC should be
        // a small multiple of MR.
        let mc_max = topo.l2_bytes / (kc * 4);
        let mc = (mc_max / mr * mr).min(4 * mr).max(mr);
        // NC: columns per L3 tile
        let nc_max = topo.l3_bytes / (2 * kc * 4);
        let nc = (nc_max / nr * nr).min(4096).max(nr);
        BlisBlocking { mr, nr, mc, kc, nc, dynamic: true }
    })
}

/// Get default blocking for 8×16 microkernel (hardcoded, used for small N).
pub fn blocking_8x16() -> BlisBlocking {
    DEFAULT_BLOCKING_8X16
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FALSIFY-CACHE-001: detect_cache_topology returns valid sizes on Linux.
    #[test]
    fn test_detect_topology() {
        let topo = topology();
        assert!(topo.l1d_bytes > 0, "L1D must be > 0");
        assert!(topo.l2_bytes > 0, "L2 must be > 0");
        // L3 may be 0 on some systems but should be > 0 on most
        assert!(
            topo.l1d_bytes <= topo.l2_bytes,
            "L1D ({}) must be <= L2 ({})",
            topo.l1d_bytes,
            topo.l2_bytes
        );
    }

    /// FALSIFY-CACHE-002: computed MC * KC * 4 fits in L2.
    #[test]
    fn test_mc_kc_fits_l2() {
        let topo = topology();
        let blk = compute_blocking_8x32(&topo);
        let packed_a_bytes = blk.mc * blk.kc * 4;
        assert!(
            packed_a_bytes <= topo.l2_bytes,
            "C-CACHE-001: packed A = {} bytes > L2 = {} bytes",
            packed_a_bytes,
            topo.l2_bytes
        );
    }

    /// Contract C-CACHE-002: KC * NR * 4 fits in L1D.
    #[test]
    fn test_kc_nr_fits_l1() {
        let topo = topology();
        let blk = compute_blocking_8x32(&topo);
        let b_panel_bytes = blk.kc * blk.nr * 4;
        assert!(
            b_panel_bytes <= topo.l1d_bytes,
            "C-CACHE-002: B panel = {} bytes > L1D = {} bytes",
            b_panel_bytes,
            topo.l1d_bytes
        );
    }

    /// Contract C-CACHE-004: MC is multiple of MR.
    #[test]
    fn test_mc_multiple_of_mr() {
        let blk = blocking_8x32();
        assert_eq!(blk.mc % blk.mr, 0, "C-CACHE-004: MC={} not multiple of MR={}", blk.mc, blk.mr);
    }

    /// Contract C-CACHE-005: NC is multiple of NR.
    #[test]
    fn test_nc_multiple_of_nr() {
        let blk = blocking_8x32();
        assert_eq!(blk.nc % blk.nr, 0, "C-CACHE-005: NC={} not multiple of NR={}", blk.nc, blk.nr);
    }

    /// Parse cache size strings correctly.
    #[test]
    fn test_parse_cache_size() {
        assert_eq!(parse_cache_size("32K"), Some(32768));
        assert_eq!(parse_cache_size("1024K"), Some(1048576));
        assert_eq!(parse_cache_size("32768K"), Some(33554432));
        assert_eq!(parse_cache_size("2M"), Some(2097152));
    }

    /// Computed blocking for Zen 4 (32K L1D, 1M L2, 32M L3).
    #[test]
    fn test_zen4_blocking() {
        let topo = CacheTopology { l1d_bytes: 32768, l2_bytes: 1048576, l3_bytes: 33554432 };
        let blk = compute_blocking_8x32(&topo);
        // KC: 32768 / (32 * 4) = 256
        assert_eq!(blk.kc, 256, "KC for Zen 4");
        // MC: 1048576 / (256 * 4) = 1024, capped at 256, rounded to 8
        assert!(blk.mc >= 64 && blk.mc <= 256, "MC={} for Zen 4", blk.mc);
        assert_eq!(blk.mc % 8, 0, "MC must be multiple of MR=8");
        // NC: 33554432 / (2 * 256 * 4) = 16384, capped at 4096
        assert_eq!(blk.nc, 4096, "NC for Zen 4");
        assert!(blk.dynamic);
    }
}
