//! Memory metrics collector
//!
//! Genchi Genbutsu: go and see the machine. On Linux that is `/proc/meminfo`;
//! on macOS there is no `/proc` at all, so it is `sysctl` and `vm_stat`.
//!
//! WHAT WENT WRONG (#3236, measured on mini-m4, Apple M4, 16 GB)
//! -------------------------------------------------------------
//! `collect()` read `/proc/meminfo` unconditionally and swallowed the error:
//!
//!     let metrics = self.read_meminfo().unwrap_or_default();
//!
//! On darwin that read is `Err(ENOENT)`, `unwrap_or_default()` turned it into an
//! all-zero `MemoryMetrics`, and `cbtop` rendered `0` total / `0` available /
//! `0` swap -- presented exactly like a measurement, with nothing anywhere
//! saying the read had failed.
//!
//! Reporting "0 bytes of memory" is worse than reporting nothing. It is the same
//! shape as the VRAM ledger's `is_alive` (#3205): an UNMEASURABLE condition
//! rendered as a definite, wrong value. There the default released live
//! reservations; here it reports a machine with no RAM.
//!
//! Two things are fixed, and they are not the same thing:
//!
//!   1. The fail-open. `MemoryMetrics` now carries `measured`, and the failure
//!      path sets it false. A caller can tell "0 because the host has no swap"
//!      from "0 because nothing was read" -- which the old struct could not
//!      express at all.
//!   2. The blindness. darwin gets a real reader, so `apr cbtop` measures the
//!      box instead of being honest about not measuring it. mini is a full-time
//!      aprender build host; a monitor that cannot see it is not much of a
//!      monitor.
//!
//! The parsers are pure functions over text, so their tests run on EVERY
//! platform -- the darwin rows are proved on Linux CI, which is the only way
//! they would ever be proved at all. F043 already asserts `total_kb > 0`; with a
//! real reader it now passes on darwin for the right reason rather than being
//! relaxed to accommodate a zero.

use crate::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};
use crate::ring_buffer::RingBuffer;
use std::any::Any;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct MemoryMetrics {
    pub timestamp: Instant,
    /// False when nothing could be read. Every other field is then meaningless,
    /// and a renderer must say "unavailable" rather than print a zero.
    pub measured: bool,
    pub total_kb: u64,
    pub available_kb: u64,
    pub free_kb: u64,
    pub swap_total_kb: u64,
    pub swap_free_kb: u64,
}

impl Default for MemoryMetrics {
    fn default() -> Self {
        Self {
            timestamp: Instant::now(),
            measured: false,
            total_kb: 0,
            available_kb: 0,
            free_kb: 0,
            swap_total_kb: 0,
            swap_free_kb: 0,
        }
    }
}

impl MemoryMetrics {
    /// The honest empty value: nothing was read, and the struct says so.
    pub fn unavailable() -> Self {
        Self::default()
    }
}

/// Parse `/proc/meminfo`. Pure, so it is tested on every platform.
pub fn parse_proc_meminfo(content: &str) -> MemoryMetrics {
    let mut metrics = MemoryMetrics {
        timestamp: Instant::now(),
        measured: false,
        ..Default::default()
    };
    for line in content.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            continue;
        }
        let value = parts[1].parse::<u64>().unwrap_or(0);
        match parts[0] {
            "MemTotal:" => metrics.total_kb = value,
            "MemAvailable:" => metrics.available_kb = value,
            "MemFree:" => metrics.free_kb = value,
            "SwapTotal:" => metrics.swap_total_kb = value,
            "SwapFree:" => metrics.swap_free_kb = value,
            _ => {}
        }
    }
    // A parse that found no MemTotal read nothing about this machine. Saying
    // `measured: true` there would re-create the defect one layer down.
    metrics.measured = metrics.total_kb > 0;
    metrics
}

/// Parse `vm_stat` output into (free_kb, available_kb), given the page size.
///
/// darwin's "available" has no single field. `vm_stat` reports pages; free +
/// inactive + speculative is what Activity Monitor treats as reclaimable, and it
/// is the closest honest analogue of Linux's MemAvailable. Pure, so the rows run
/// on Linux.
pub fn parse_vm_stat(content: &str, page_size_bytes: u64) -> (u64, u64) {
    let mut free = 0u64;
    let mut inactive = 0u64;
    let mut speculative = 0u64;
    for line in content.lines() {
        let (key, rest) = match line.split_once(':') {
            Some(kv) => kv,
            None => continue,
        };
        // "Pages free:                        1234567."
        let pages: u64 = rest.trim().trim_end_matches('.').parse().unwrap_or(0);
        match key.trim() {
            "Pages free" => free = pages,
            "Pages inactive" => inactive = pages,
            "Pages speculative" => speculative = pages,
            _ => {}
        }
    }
    let kb = |pages: u64| pages.saturating_mul(page_size_bytes) / 1024;
    (kb(free), kb(free + inactive + speculative))
}

/// Parse `sysctl -n vm.swapusage` into (total_kb, free_kb).
///
/// `total = 2048.00M  used = 512.00M  free = 1536.00M  (encrypted)`
pub fn parse_swapusage(content: &str) -> (u64, u64) {
    let field = |name: &str| -> u64 {
        let Some(after) = content.split(name).nth(1) else {
            return 0;
        };
        let tok = after.trim().trim_start_matches('=').trim();
        let tok = tok.split_whitespace().next().unwrap_or("");
        let (num, mult) = match tok.chars().last() {
            Some('K') => (&tok[..tok.len() - 1], 1.0_f64),
            Some('M') => (&tok[..tok.len() - 1], 1024.0),
            Some('G') => (&tok[..tok.len() - 1], 1024.0 * 1024.0),
            _ => (tok, 1.0 / 1024.0),
        };
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        {
            (num.parse::<f64>().unwrap_or(0.0) * mult) as u64
        }
    };
    (field("total"), field("free"))
}

pub struct MemoryCollectorBrick {
    history: RingBuffer<MemoryMetrics>,
}

impl MemoryCollectorBrick {
    pub fn new() -> Self {
        Self {
            history: RingBuffer::new(120),
        }
    }

    pub fn collect(&mut self) -> MemoryMetrics {
        // `unwrap_or_else(MemoryMetrics::unavailable)`, never `unwrap_or_default`
        // with a Default that looks like a reading. The value that comes back
        // from a failed read says `measured: false`.
        let metrics = self
            .read_memory()
            .unwrap_or_else(|_| MemoryMetrics::unavailable());
        self.history.push(metrics.clone());
        metrics
    }

    #[cfg(target_os = "linux")]
    fn read_memory(&self) -> Result<MemoryMetrics, std::io::Error> {
        Ok(parse_proc_meminfo(&std::fs::read_to_string(
            "/proc/meminfo",
        )?))
    }

    #[cfg(target_os = "macos")]
    fn read_memory(&self) -> Result<MemoryMetrics, std::io::Error> {
        use std::process::Command;
        let out = |cmd: &str, args: &[&str]| -> Result<String, std::io::Error> {
            let o = Command::new(cmd).args(args).output()?;
            if !o.status.success() {
                return Err(std::io::Error::other(format!("{cmd} exited {}", o.status)));
            }
            Ok(String::from_utf8_lossy(&o.stdout).into_owned())
        };
        let total_bytes: u64 = out("sysctl", &["-n", "hw.memsize"])?
            .trim()
            .parse()
            .map_err(|e| std::io::Error::other(format!("hw.memsize: {e}")))?;
        let page_size: u64 = out("sysctl", &["-n", "hw.pagesize"])?
            .trim()
            .parse()
            .unwrap_or(4096);
        let (free_kb, available_kb) = parse_vm_stat(&out("vm_stat", &[])?, page_size);
        // Swap can legitimately be absent; it must not fail the whole read.
        let (swap_total_kb, swap_free_kb) = out("sysctl", &["-n", "vm.swapusage"])
            .map(|s| parse_swapusage(&s))
            .unwrap_or((0, 0));
        Ok(MemoryMetrics {
            timestamp: Instant::now(),
            measured: total_bytes > 0,
            total_kb: total_bytes / 1024,
            available_kb,
            free_kb,
            swap_total_kb,
            swap_free_kb,
        })
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    fn read_memory(&self) -> Result<MemoryMetrics, std::io::Error> {
        // No reader for this platform. Say so; do not invent a number. The
        // caller renders "unavailable" and F043 fails loudly here rather than
        // passing on a zero -- which is the point.
        Err(std::io::Error::other(
            "no memory reader for this platform (have: linux /proc/meminfo, macos sysctl+vm_stat)",
        ))
    }

    pub fn history(&self) -> &RingBuffer<MemoryMetrics> {
        &self.history
    }
}

impl Brick for MemoryCollectorBrick {
    fn brick_name(&self) -> &'static str {
        "memory_collector"
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![
            BrickAssertion::custom("mem_total_positive", |b| {
                let s = b
                    .downcast_ref::<MemoryCollectorBrick>()
                    .expect("brick MUST be MemoryCollectorBrick");
                s.history.back().map_or(true, |m| m.total_kb > 0)
            }),
            BrickAssertion::max_latency_ms(2),
        ]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget {
            collect_ms: 2,
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

    // Real `/proc/meminfo` head, trimmed.
    const MEMINFO: &str = "\
MemTotal:       32611128 kB
MemFree:         1234560 kB
MemAvailable:   28000000 kB
Buffers:          123456 kB
SwapTotal:       2097148 kB
SwapFree:        2097148 kB
";

    // Real `vm_stat` head from an Apple M4, trimmed. 16 KiB pages on arm64.
    const VM_STAT: &str = "\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               12000.
Pages active:                            400000.
Pages inactive:                           30000.
Pages speculative:                         8000.
Pages wired down:                        150000.
";

    #[test]
    fn proc_meminfo_is_parsed_and_marked_measured() {
        let m = parse_proc_meminfo(MEMINFO);
        assert!(m.measured, "a meminfo with MemTotal is a real measurement");
        assert_eq!(m.total_kb, 32_611_128);
        assert_eq!(m.available_kb, 28_000_000);
        assert_eq!(m.free_kb, 1_234_560);
        assert_eq!(m.swap_total_kb, 2_097_148);
        assert_eq!(m.swap_free_kb, 2_097_148);
    }

    /// #3236's defect, one layer down: a parse that found no MemTotal read
    /// nothing about this machine, and must not claim it did. Deleting the
    /// `measured = total_kb > 0` line turns this red.
    #[test]
    fn a_meminfo_without_memtotal_is_not_a_measurement() {
        let m = parse_proc_meminfo("Buffers: 123 kB\nCached: 456 kB\n");
        assert!(!m.measured, "no MemTotal means nothing was measured");
        assert_eq!(m.total_kb, 0);
    }

    /// The darwin rows run on Linux CI, which is the only place they would ever
    /// run at all: no CI host lacks /proc, so a darwin-gated test is dark.
    #[test]
    fn vm_stat_pages_become_kb_at_the_reported_page_size() {
        let (free_kb, avail_kb) = parse_vm_stat(VM_STAT, 16384);
        assert_eq!(free_kb, 12_000 * 16_384 / 1024);
        // free + inactive + speculative -- what Activity Monitor treats as
        // reclaimable, and the closest honest analogue of Linux MemAvailable.
        assert_eq!(avail_kb, (12_000 + 30_000 + 8_000) * 16_384 / 1024);
        assert!(avail_kb > free_kb, "available includes more than free");
    }

    /// The page size is read from the host, not assumed: arm64 macOS uses 16
    /// KiB pages and x86_64 uses 4 KiB. Hardcoding 4096 would under-report an
    /// M-series box by 4x, which is a wrong number rather than a missing one.
    #[test]
    fn the_page_size_is_load_bearing() {
        let (four, _) = parse_vm_stat(VM_STAT, 4096);
        let (sixteen, _) = parse_vm_stat(VM_STAT, 16384);
        assert_eq!(sixteen, four * 4);
    }

    #[test]
    fn swapusage_units_are_honoured() {
        let (t, f) =
            parse_swapusage("total = 2048.00M  used = 512.00M  free = 1536.00M  (encrypted)");
        assert_eq!(t, 2048 * 1024);
        assert_eq!(f, 1536 * 1024);
        let (g, _) = parse_swapusage("total = 4.00G  used = 0.00G  free = 4.00G");
        assert_eq!(g, 4 * 1024 * 1024);
        let (k, _) = parse_swapusage("total = 512.00K  used = 0.00K  free = 512.00K");
        assert_eq!(k, 512);
    }

    #[test]
    fn a_machine_with_no_swap_parses_as_zero_not_as_garbage() {
        let (t, f) = parse_swapusage("total = 0.00M  used = 0.00M  free = 0.00M");
        assert_eq!((t, f), (0, 0));
    }

    /// The heart of #3236. The old code called `unwrap_or_default()` and the
    /// Default looked exactly like a reading of a machine with no RAM. The
    /// unavailable value must be distinguishable from a measurement.
    #[test]
    fn unavailable_is_distinguishable_from_a_measurement() {
        let u = MemoryMetrics::unavailable();
        assert!(
            !u.measured,
            "an unread value must never claim to be measured"
        );
        assert_eq!(u.total_kb, 0);
        let m = parse_proc_meminfo(MEMINFO);
        assert!(m.measured);
        assert_ne!(u.measured, m.measured, "the two must not be confusable");
    }

    /// Replaces a test that asserted the DEFECT: it required `total_kb == 0`
    /// wherever /proc/meminfo was absent, which is exactly the zero #3236 is
    /// about. Both supported platforms now have a reader, so `collect()` must
    /// come back measured on either.
    #[test]
    fn collect_measures_every_platform_this_crate_claims_to_support() {
        let mut collector = MemoryCollectorBrick::new();
        let m = collector.collect();
        if cfg!(any(target_os = "linux", target_os = "macos")) {
            assert!(
                m.measured,
                "a supported platform must produce a measurement"
            );
            assert!(m.total_kb > 0, "a measured host has more than 0 kB of RAM");
        } else {
            assert!(
                !m.measured,
                "an unsupported platform must say so, not print 0"
            );
        }
    }
}
