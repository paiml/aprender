//! Training Monitor Grid Protocol (SPEC-TRAINMON-001)
//!
//! Integer-grid layout with provable cell occupancy. Inspired by rmedia's
//! mathematical precision: all positions are grid coordinates × cell size,
//! producing pixel-deterministic output testable via snapshot.
//!
//! Contract: `contracts/aprender/training-monitor-v1.yaml` equation `grid_occupancy`.

/// Grid dimensions (8 columns × 6 rows = 48 cells).
pub const GRID_COLS: u8 = 8;
pub const GRID_ROWS: u8 = 6;
pub const TOTAL_CELLS: usize = (GRID_COLS as usize) * (GRID_ROWS as usize);

/// Named region in the monitor dashboard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Region {
    Header,
    Loss,
    Throughput,
    Mfu,
    Gradient,
    GpuVram,
    GpuUtil,
    GpuTemp,
    LrSchedule,
    Config,
    Footer,
}

impl Region {
    /// Cell ranges for each region: (col_start, col_end, row_start, row_end).
    /// All ranges are half-open: [start, end).
    const fn bounds(&self) -> (u8, u8, u8, u8) {
        match self {
            Self::Header => (0, 8, 0, 1),
            Self::Loss => (0, 3, 1, 4),
            Self::Throughput => (3, 5, 1, 2),
            Self::Gradient => (3, 5, 2, 3),
            Self::Mfu => (3, 5, 3, 4),
            Self::GpuVram => (5, 8, 1, 2),
            Self::GpuUtil => (5, 8, 2, 3),
            Self::GpuTemp => (5, 8, 3, 4),
            Self::LrSchedule => (0, 3, 4, 5),
            Self::Config => (3, 8, 4, 5),
            Self::Footer => (0, 8, 5, 6),
        }
    }

    /// All 11 regions in assignment order.
    pub const ALL: [Region; 11] = [
        Self::Header,
        Self::Loss,
        Self::Throughput,
        Self::Gradient,
        Self::Mfu,
        Self::GpuVram,
        Self::GpuUtil,
        Self::GpuTemp,
        Self::LrSchedule,
        Self::Config,
        Self::Footer,
    ];
}

/// Pixel-space rectangle computed from grid coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: u16,
    pub y: u16,
    pub width: u16,
    pub height: u16,
}

/// Monitor grid with occupancy tracking.
///
/// Invariant: after `assign_all()`, exactly 48 cells are occupied with zero overlaps.
pub struct MonitorGrid {
    occupied: std::collections::HashSet<(u8, u8)>,
}

impl Default for MonitorGrid {
    fn default() -> Self {
        let mut grid = Self {
            occupied: std::collections::HashSet::with_capacity(TOTAL_CELLS),
        };
        grid.assign_all();
        grid
    }
}

impl MonitorGrid {
    /// Assign all 11 regions. Panics on overlap (Jidoka — stop the line).
    fn assign_all(&mut self) {
        for region in &Region::ALL {
            let (c0, c1, r0, r1) = region.bounds();
            for col in c0..c1 {
                for row in r0..r1 {
                    if !self.occupied.insert((col, row)) {
                        panic!(
                            "F-TM-LAYOUT: cell ({col}, {row}) already occupied — \
                             region {region:?} overlaps with another region"
                        );
                    }
                }
            }
        }
    }

    /// Total occupied cells (must equal 48).
    #[must_use]
    pub fn occupied_count(&self) -> usize {
        self.occupied.len()
    }

    /// All occupied cells as a vec (for testing).
    #[must_use]
    pub fn all_cells(&self) -> Vec<(u8, u8)> {
        let mut cells: Vec<_> = self.occupied.iter().copied().collect();
        cells.sort();
        cells
    }

    /// Compute pixel-space rectangle for a region given terminal dimensions.
    ///
    /// Integer division — zero floating-point error.
    #[must_use]
    pub fn rect(&self, region: Region, term_width: u16, term_height: u16) -> Rect {
        let cell_w = term_width / u16::from(GRID_COLS);
        let cell_h = term_height / u16::from(GRID_ROWS);
        let (c0, c1, r0, r1) = region.bounds();
        Rect {
            x: u16::from(c0) * cell_w,
            y: u16::from(r0) * cell_h,
            width: u16::from(c1 - c0) * cell_w,
            height: u16::from(r1 - r0) * cell_h,
        }
    }
}

/// Anomaly detection for training metrics (SPEC-TRAINMON-001 §6).
pub mod anomaly {
    /// Loss spike: loss > threshold × EMA, after warmup.
    ///
    /// Contract: `training-monitor-v1.yaml` equation `spike_detection`.
    #[must_use]
    pub fn is_spike(loss: f32, ema_loss: f32, step: u64, warmup_steps: u64) -> bool {
        step > warmup_steps && ema_loss > 0.0 && loss > 2.0 * ema_loss
    }

    /// NaN/Inf divergence detection.
    ///
    /// Contract: `training-monitor-v1.yaml` equation `nan_detection`.
    #[must_use]
    pub fn is_nan_divergence(loss: f32) -> bool {
        loss.is_nan() || loss.is_infinite()
    }

    /// Gradient explosion: gnorm > 100× EMA gnorm.
    #[must_use]
    pub fn is_gradient_explosion(gnorm: f32, ema_gnorm: f32) -> bool {
        ema_gnorm > 0.0 && gnorm > 100.0 * ema_gnorm
    }
}

/// Metric snapshot from training loop (data contract).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MetricSnapshot {
    pub step: u64,
    pub loss: f32,
    pub loss_ema: f32,
    pub lr: f32,
    pub throughput: f32,
    pub mfu: f32,
    pub gradient_norm: f32,
    pub gpu_vram_used: u64,
    pub gpu_vram_total: u64,
    pub gpu_utilization: f32,
    pub gpu_temperature: f32,
    pub val_ppl: Option<f32>,
    pub zclip_events: u32,
    pub elapsed_secs: f64,
    pub tokens_seen: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    // FALSIFY-TM-001: No cell overlap, all 48 cells assigned
    #[test]
    fn falsify_tm_001_no_overlap() {
        let grid = MonitorGrid::default();
        assert_eq!(grid.occupied_count(), TOTAL_CELLS, "not all 48 cells assigned");

        // Verify uniqueness (redundant with HashSet but explicit)
        let cells = grid.all_cells();
        let unique: std::collections::HashSet<(u8, u8)> = cells.iter().copied().collect();
        assert_eq!(cells.len(), unique.len(), "cell overlap detected");
    }

    // Verify all 11 regions present
    #[test]
    fn all_regions_assigned() {
        assert_eq!(Region::ALL.len(), 11);
        let grid = MonitorGrid::default();
        // Count cells per region
        let total: usize = Region::ALL.iter().map(|r| {
            let (c0, c1, r0, r1) = r.bounds();
            (c1 - c0) as usize * (r1 - r0) as usize
        }).sum();
        assert_eq!(total, TOTAL_CELLS);
        assert_eq!(grid.occupied_count(), total);
    }

    // Cell sizing: integer division
    #[test]
    fn cell_sizing_integer() {
        let grid = MonitorGrid::default();
        let rect = grid.rect(Region::Loss, 160, 48);
        // 160 / 8 = 20 per col, Loss is cols 0..3 = 60 wide
        assert_eq!(rect.x, 0);
        assert_eq!(rect.width, 60);
        // 48 / 6 = 8 per row, Loss is rows 1..4 = 24 high
        assert_eq!(rect.y, 8);
        assert_eq!(rect.height, 24);
    }

    // FALSIFY-TM-003: Spike detection boundary
    #[test]
    fn falsify_tm_003_spike_boundary() {
        let ema = 5.0;
        assert!(anomaly::is_spike(10.01, ema, 1000, 100)); // 2.002× — spike
        assert!(!anomaly::is_spike(9.99, ema, 1000, 100)); // 1.998× — not spike
        assert!(!anomaly::is_spike(10.01, ema, 50, 100));  // warmup — immune
        assert!(!anomaly::is_spike(10.01, ema, 100, 100)); // boundary — immune (step == warmup)
    }

    // FALSIFY-TM-004: NaN detection
    #[test]
    fn falsify_tm_004_nan_detection() {
        assert!(anomaly::is_nan_divergence(f32::NAN));
        assert!(anomaly::is_nan_divergence(f32::INFINITY));
        assert!(anomaly::is_nan_divergence(f32::NEG_INFINITY));
        assert!(!anomaly::is_nan_divergence(5.0));
        assert!(!anomaly::is_nan_divergence(0.0));
        assert!(!anomaly::is_nan_divergence(-1.0));
    }

    // Gradient explosion detection
    #[test]
    fn gradient_explosion_detection() {
        assert!(anomaly::is_gradient_explosion(1000.0, 1.0));  // 1000×
        assert!(!anomaly::is_gradient_explosion(99.0, 1.0));   // 99× — not explosion
        assert!(!anomaly::is_gradient_explosion(100.0, 0.0));  // ema=0 — no reference
    }

    // FALSIFY-TM-002 (proxy): Layout tiling — all regions tile without gap or overflow
    #[test]
    fn falsify_tm_002_layout_tiling() {
        let grid = MonitorGrid::default();
        let (w, h) = (160u16, 48u16);

        for region in &Region::ALL {
            let rect = grid.rect(*region, w, h);
            assert!(
                rect.x + rect.width <= w,
                "{region:?} exceeds width: x={} w={} term_w={w}",
                rect.x, rect.width
            );
            assert!(
                rect.y + rect.height <= h,
                "{region:?} exceeds height: y={} h={} term_h={h}",
                rect.y, rect.height
            );
        }

        // Header spans full width
        let header = grid.rect(Region::Header, w, h);
        assert_eq!(header.x, 0);
        assert_eq!(header.width, w);

        // Footer spans full width
        let footer = grid.rect(Region::Footer, w, h);
        assert_eq!(footer.x, 0);
        assert_eq!(footer.width, w);
    }

    // FALSIFY-TM-006 (proxy): Rect computation is O(1) — no layout engine overhead
    #[test]
    fn falsify_tm_006_rect_perf() {
        let grid = MonitorGrid::default();
        let start = std::time::Instant::now();
        for _ in 0..10_000 {
            for region in &Region::ALL {
                std::hint::black_box(grid.rect(*region, 160, 48));
            }
        }
        let elapsed = start.elapsed();
        assert!(elapsed.as_millis() < 50, "too slow: {elapsed:?} for 110K calls");
    }

    // FALSIFY-TM-005: JSON serialization round-trip
    #[test]
    fn falsify_tm_005_json_roundtrip() {
        let snap = MetricSnapshot {
            step: 100,
            loss: 5.07,
            loss_ema: 5.10,
            lr: 7.35e-5,
            throughput: 8994.0,
            mfu: 26.1,
            gradient_norm: 0.25,
            gpu_vram_used: 13_696_000_000,
            gpu_vram_total: 24_045_000_000,
            gpu_utilization: 89.0,
            gpu_temperature: 62.0,
            val_ppl: None,
            zclip_events: 2,
            elapsed_secs: 3600.0,
            tokens_seen: 51_200_000,
        };
        let json = serde_json::to_string(&snap).expect("serialize");
        let parsed: MetricSnapshot = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.step, 100);
        assert!((parsed.loss - 5.07).abs() < 1e-6);
        assert!(parsed.val_ppl.is_none());
    }
}
