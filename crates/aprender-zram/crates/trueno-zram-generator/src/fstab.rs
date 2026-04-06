//! fstab entry generation for zram swap.

use crate::config::ZramConfig;

/// Generate an fstab entry for a zram swap device.
#[must_use]
#[allow(dead_code)]
pub fn generate_fstab_entry(config: &ZramConfig) -> String {
    format!(
        "/dev/zram{device}\tnone\tswap\tsw,pri={priority}\t0\t0",
        device = config.device,
        priority = config.priority,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fstab_entry() {
        let config = ZramConfig {
            device: 0,
            size: "4G".to_string(),
            algorithm: "lz4".to_string(),
            streams: 4,
            priority: 100,
        };

        let entry = generate_fstab_entry(&config);
        assert!(entry.contains("/dev/zram0"));
        assert!(entry.contains("pri=100"));
    }
}
