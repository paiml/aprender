//! ttop run-loop pieces that can be tested without a terminal (#4511).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::Arc;
use std::time::Duration;

use presentar_terminal::ptop::PanelType;

/// Parse a `--explode` panel name. An unknown name is an error (exit 2 via clap),
/// never a silent full-screen fallback (#4511 D5).
pub fn parse_panel_type(name: &str) -> Result<PanelType, String> {
    match name.to_lowercase().as_str() {
        "cpu" => Ok(PanelType::Cpu),
        "memory" | "mem" => Ok(PanelType::Memory),
        "disk" => Ok(PanelType::Disk),
        "network" | "net" => Ok(PanelType::Network),
        "process" | "proc" | "processes" => Ok(PanelType::Process),
        "gpu" => Ok(PanelType::Gpu),
        "sensors" | "sensor" => Ok(PanelType::Sensors),
        "connections" | "conn" => Ok(PanelType::Connections),
        "psi" | "pressure" => Ok(PanelType::Psi),
        "files" | "file" => Ok(PanelType::Files),
        "battery" | "bat" => Ok(PanelType::Battery),
        "containers" | "container" | "docker" => Ok(PanelType::Containers),
        _ => Err(format!(
            "unknown panel '{name}'. Valid: cpu, memory, disk, network, process, gpu, sensors, connections, psi, files, battery, containers"
        )),
    }
}

/// Stops the collector thread when dropped — on return, on `?` error and on panic
/// unwinding alike (#4511 D6: an error path used to leave it running).
pub struct StopOnDrop(pub Arc<AtomicBool>);

impl Drop for StopOnDrop {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Relaxed);
    }
}

/// Spawn `collect` on a background thread every `interval`, delivering results over a
/// channel that holds at most ONE pending value (#4511 D3). The old unbounded channel
/// queued a full snapshot per tick whenever the UI stalled (XOFF, a stuck pty, a slow
/// ssh): RSS grew by one snapshot per tick for as long as the stall lasted.
/// When the slot is full the new value is dropped; the UI takes the pending one next.
pub fn spawn_bounded_collector<T, F>(
    interval: Duration,
    mut collect: F,
) -> (Receiver<T>, StopOnDrop)
where
    T: Send + 'static,
    F: FnMut() -> T + Send + 'static,
{
    let running = Arc::new(AtomicBool::new(true));
    let running_thread = Arc::clone(&running);
    let (tx, rx): (SyncSender<T>, Receiver<T>) = mpsc::sync_channel(1);
    std::thread::spawn(move || {
        while running_thread.load(Ordering::Relaxed) {
            match tx.try_send(collect()) {
                Ok(()) | Err(TrySendError::Full(_)) => {}
                Err(TrySendError::Disconnected(_)) => break,
            }
            std::thread::sleep(interval);
        }
    });
    (rx, StopOnDrop(running))
}

/// Resident set size of this process in KiB, from `/proc/self/status` (Linux only).
#[must_use]
pub fn rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status
        .lines()
        .find_map(|l| l.strip_prefix("VmRSS:"))
        .and_then(|v| v.trim().trim_end_matches("kB").trim().parse().ok())
}

/// Whether this frame must repaint every cell instead of diffing against the last one
/// (#4511 D4). The diff only rewrites cells drawn this frame, so after a resize the
/// cells outside the new layout kept their stale glyphs until something overdrew them.
pub fn needs_full_repaint(
    first_frame: bool,
    layout_mode_changed: bool,
    size: (u16, u16),
    last_size: (u16, u16),
) -> bool {
    first_frame || layout_mode_changed || size != last_size
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn unknown_panel_is_an_error_not_a_silent_fallback() {
        assert_eq!(parse_panel_type("CPU"), Ok(PanelType::Cpu));
        assert_eq!(parse_panel_type("docker"), Ok(PanelType::Containers));
        assert!(matches!(parse_panel_type("cpux"), Err(e) if e.contains("unknown panel 'cpux'")));
    }

    #[test]
    fn a_stalled_consumer_holds_at_most_one_pending_snapshot() {
        let ticks = Arc::new(AtomicUsize::new(0));
        let t = Arc::clone(&ticks);
        let (rx, stop) = spawn_bounded_collector(Duration::from_millis(1), move || {
            t.fetch_add(1, Ordering::Relaxed);
            vec![0u8; 1024]
        });
        // the UI "stalls": nobody reads while the collector ticks many times
        while ticks.load(Ordering::Relaxed) < 50 {
            std::thread::sleep(Duration::from_millis(2));
        }
        drop(stop);
        let pending = rx.try_iter().count();
        assert!(
            pending <= 1,
            "{pending} snapshots queued behind a stalled UI"
        );
    }

    #[test]
    fn dropping_the_guard_stops_the_collector() {
        let ticks = Arc::new(AtomicUsize::new(0));
        let t = Arc::clone(&ticks);
        let (rx, stop) = spawn_bounded_collector(Duration::from_millis(1), move || {
            t.fetch_add(1, Ordering::Relaxed);
        });
        std::thread::sleep(Duration::from_millis(10));
        drop(stop);
        // `rx` stays alive: a dropped receiver ends the loop on its own (Disconnected),
        // which would pass this test whether or not the guard stops anything
        std::thread::sleep(Duration::from_millis(10));
        let settled = ticks.load(Ordering::Relaxed);
        std::thread::sleep(Duration::from_millis(30));
        assert!(
            ticks.load(Ordering::Relaxed) <= settled + 1,
            "collector kept running after its guard dropped"
        );
        drop(rx);
    }

    #[test]
    fn a_resize_forces_a_full_repaint() {
        let (w, h) = (120, 40);
        assert!(
            !needs_full_repaint(false, false, (w, h), (w, h)),
            "steady frames diff"
        );
        assert!(
            needs_full_repaint(false, false, (w, h - 10), (w, h)),
            "shrunk height"
        );
        assert!(
            needs_full_repaint(false, false, (w + 1, h), (w, h)),
            "grown width"
        );
        assert!(
            needs_full_repaint(true, false, (w, h), (w, h)),
            "first frame"
        );
        assert!(
            needs_full_repaint(false, true, (w, h), (w, h)),
            "explode toggled"
        );
    }

    #[test]
    fn rss_is_readable_on_linux() {
        if cfg!(target_os = "linux") {
            assert!(rss_kib().is_some_and(|k| k > 0));
        }
    }
}
