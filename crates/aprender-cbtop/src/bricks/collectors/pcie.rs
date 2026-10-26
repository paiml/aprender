//! PCIe metrics collector
//!
//! Collects PCIe link speed from sysfs (Genchi Genbutsu)

use crate::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};
use crate::ring_buffer::RingBuffer;
use std::any::Any;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Legal PCIe link widths (x0 = down/unknown).
const VALID_WIDTHS: [u8; 8] = [0, 1, 2, 4, 8, 12, 16, 32];

#[derive(Debug, Clone)]
pub struct PcieMetrics {
    pub timestamp: Instant,
    /// Number of Gen4 devices
    pub gen4_count: usize,
    /// Number of Gen5 devices
    pub gen5_count: usize,
    /// Max link width found
    pub max_width: u8,
}

impl Default for PcieMetrics {
    fn default() -> Self {
        Self {
            timestamp: Instant::now(),
            gen4_count: 0,
            gen5_count: 0,
            max_width: 0,
        }
    }
}

pub struct PcieCollectorBrick {
    history: RingBuffer<PcieMetrics>,
    root: PathBuf,
}

impl PcieCollectorBrick {
    pub fn new() -> Self {
        Self {
            history: RingBuffer::new(120),
            root: PathBuf::from("/sys/bus/pci/devices"),
        }
    }

    /// Collector reading a different sysfs devices directory (tests, containers).
    pub fn with_root(root: PathBuf) -> Self {
        Self {
            root,
            ..Self::new()
        }
    }

    pub fn collect(&mut self) -> PcieMetrics {
        let metrics = self.read_pcie().unwrap_or_default();
        self.history.push(metrics.clone());
        metrics
    }

    fn read_pcie(&self) -> Result<PcieMetrics, std::io::Error> {
        Self::read_pcie_from(&self.root)
    }

    /// Scan a sysfs PCI devices directory (injectable so tests need no real hardware).
    fn read_pcie_from(root: &Path) -> Result<PcieMetrics, std::io::Error> {
        let mut m = PcieMetrics::default();
        let Ok(entries) = fs::read_dir(root) else {
            return Ok(m);
        };
        for path in entries.flatten().map(|e| e.path()) {
            match fs::read_to_string(path.join("current_link_speed")) {
                Ok(s) if s.contains("16.0 GT/s") => m.gen4_count += 1,
                Ok(s) if s.contains("32.0 GT/s") => m.gen5_count += 1,
                _ => {}
            }
            if let Some(w) = Self::read_width(&path) {
                m.max_width = m.max_width.max(w);
            }
        }
        m.timestamp = Instant::now();
        Ok(m)
    }

    /// Link width of one device; None when unreadable or an unknown-width sentinel.
    /// Some bridges report 63 (0x3f) or 255 (measured on kfocus: 4 devices at 63).
    fn read_width(dev: &Path) -> Option<u8> {
        let w = fs::read_to_string(dev.join("current_link_width"))
            .ok()?
            .trim()
            .parse::<u8>()
            .ok()?;
        VALID_WIDTHS.contains(&w).then_some(w)
    }

    /// The newest sample's max width is a legal PCIe width (vacuously true when empty).
    pub fn width_valid(&self) -> bool {
        self.history
            .back()
            .map_or(true, |m| VALID_WIDTHS.contains(&m.max_width))
    }

    pub fn history(&self) -> &RingBuffer<PcieMetrics> {
        &self.history
    }
}

impl Brick for PcieCollectorBrick {
    fn brick_name(&self) -> &'static str {
        "pcie_collector"
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![
            BrickAssertion::custom("width_valid", |b| {
                let s = b
                    .downcast_ref::<PcieCollectorBrick>()
                    .expect("brick MUST be PcieCollectorBrick");
                s.history
                    .back()
                    .map_or(true, |m| VALID_WIDTHS.contains(&m.max_width))
            }),
            BrickAssertion::max_latency_ms(10), // Filesystem scan can be slow
        ]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget {
            collect_ms: 10,
            layout_ms: 0,
            render_ms: 0,
        }
    }

    fn verify(&self) -> BrickVerification {
        let mut v = BrickVerification::new();
        for a in self.assertions() {
            v.check(&a);
        }
        v
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device(root: &Path, name: &str, speed: &str, width: &str) {
        let d = root.join(name);
        fs::create_dir_all(&d).expect("mkdir fixture device");
        fs::write(d.join("current_link_speed"), speed).expect("write speed");
        fs::write(d.join("current_link_width"), width).expect("write width");
    }

    #[test]
    fn test_pcie_collector() {
        // Real host width is hardware-dependent (a 32-lane link is legal): only the
        // PCIe width set {0,1,2,4,8,12,16,32} is invariant.
        let mut collector = PcieCollectorBrick::new();
        let metrics = collector.collect();
        assert!(VALID_WIDTHS.contains(&metrics.max_width));
    }

    #[test]
    fn test_pcie_read_from_fixture() {
        let root = std::env::temp_dir().join(format!("cbtop-pcie-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        device(&root, "0000:01:00.0", "16.0 GT/s PCIe\n", "16\n");
        device(&root, "0000:02:00.0", "32.0 GT/s PCIe\n", "32\n");
        device(&root, "0000:03:00.0", "8.0 GT/s PCIe\n", "4\n");
        // kfocus reports 63 (and 255 is the other sentinel) for unknown width: ignored.
        device(&root, "0000:04:00.0", "8.0 GT/s PCIe\n", "63\n");
        device(&root, "0000:05:00.0", "8.0 GT/s PCIe\n", "255\n");
        let m = PcieCollectorBrick::read_pcie_from(&root).expect("fixture read");
        fs::remove_dir_all(&root).expect("cleanup");
        assert_eq!((m.gen4_count, m.gen5_count, m.max_width), (1, 1, 32));
    }

    #[test]
    fn test_pcie_unknown_width_sentinels_ignored() {
        let root = std::env::temp_dir().join(format!("cbtop-pcie-s-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        device(&root, "0000:00:01.0", "8.0 GT/s PCIe\n", "63\n");
        device(&root, "0000:00:02.0", "8.0 GT/s PCIe\n", "4\n");
        let m = PcieCollectorBrick::read_pcie_from(&root).expect("fixture read");
        fs::remove_dir_all(&root).expect("cleanup");
        assert_eq!(m.max_width, 4);
    }

    #[test]
    fn test_pcie_missing_root_is_zero() {
        let m = PcieCollectorBrick::read_pcie_from(Path::new("/nonexistent-pcie-root"))
            .expect("missing root is not an error");
        assert_eq!(m.max_width, 0);
    }

    fn with_width(w: u8) -> PcieCollectorBrick {
        let mut c = PcieCollectorBrick::new();
        c.history.push(PcieMetrics {
            max_width: w,
            ..PcieMetrics::default()
        });
        c
    }

    #[test]
    fn test_width_valid_accepts_every_legal_width_including_x32() {
        for w in VALID_WIDTHS {
            assert!(with_width(w).width_valid(), "x{w} must verify");
        }
    }

    #[test]
    fn test_width_valid_rejects_invalid_and_accepts_empty_history() {
        assert!(!with_width(63).width_valid());
        assert!(!with_width(3).width_valid());
        assert!(PcieCollectorBrick::new().width_valid());
    }

    #[test]
    fn test_max_width_takes_the_largest_not_the_last() {
        let root = std::env::temp_dir().join(format!("cbtop-pcie-m-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        device(&root, "0000:00:01.0", "8.0 GT/s PCIe\n", "8\n");
        device(&root, "0000:00:02.0", "8.0 GT/s PCIe\n", "1\n");
        let m = PcieCollectorBrick::read_pcie_from(&root).expect("fixture read");
        fs::remove_dir_all(&root).expect("cleanup");
        assert_eq!(m.max_width, 8);
    }

    #[test]
    fn test_collect_reads_root_and_records_history() {
        let root = std::env::temp_dir().join(format!("cbtop-pcie-c-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        device(&root, "0000:00:01.0", "16.0 GT/s PCIe\n", "8\n");
        let mut c = PcieCollectorBrick::with_root(root.clone());
        let m = c.collect();
        fs::remove_dir_all(&root).expect("cleanup");
        assert_eq!((m.gen4_count, m.max_width), (1, 8));
        assert_eq!(c.history().back().map(|h| h.max_width), Some(8));
    }
}
