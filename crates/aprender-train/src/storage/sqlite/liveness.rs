//! Process identity for run liveness (EXT-03, aprender#4385).
//!
//! A run is live only if the process that started it still exists, and "the same
//! process" means the full 4-tuple `(host, boot_id, pid, proc_start_time)`:
//! - `pid` alone is reused after a wrap or a reboot;
//! - `boot_id` separates boots that recycle the same pid;
//! - `proc_start_time` (clock ticks since boot, `/proc/<pid>/stat` field 22)
//!   separates a pid reused within one boot;
//! - `host` keeps a shared database from judging another machine's pids.
//!
//! Linux-only probes (`/proc`). Elsewhere `current()` is `None` and the run is
//! recorded without an identity.

use std::path::Path;

/// The identity of one process on one boot of one host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcIdentity {
    pub host: String,
    pub boot_id: String,
    pub pid: u32,
    pub start_ticks: u64,
}

impl ProcIdentity {
    /// The identity of the calling process, or `None` off Linux.
    pub fn current() -> Option<Self> {
        Self::of_pid_under(Path::new("/proc"), std::process::id())
    }

    /// The identity of `pid` on this host now, or `None` if no such process.
    pub fn of_pid(pid: u32) -> Option<Self> {
        Self::of_pid_under(Path::new("/proc"), pid)
    }

    fn of_pid_under(proc_root: &Path, pid: u32) -> Option<Self> {
        let stat = std::fs::read_to_string(proc_root.join(pid.to_string()).join("stat")).ok()?;
        Some(Self {
            host: host_name()?,
            boot_id: boot_id_under(proc_root)?,
            pid,
            start_ticks: parse_start_ticks(&stat)?,
        })
    }
}

/// This host's name, as recorded in `runs.host`.
pub fn host_name() -> Option<String> {
    hostname::get().ok()?.into_string().ok()
}

/// This boot's id, as recorded in `runs.boot_id`.
pub fn boot_id() -> Option<String> {
    boot_id_under(Path::new("/proc"))
}

fn boot_id_under(proc_root: &Path) -> Option<String> {
    let raw = std::fs::read_to_string(proc_root.join("sys/kernel/random/boot_id")).ok()?;
    let id = raw.trim();
    (!id.is_empty()).then(|| id.to_string())
}

/// Field 22 (`starttime`) of `/proc/<pid>/stat`.
///
/// Field 2 (`comm`) is parenthesised and may itself contain spaces or `)`, so
/// the split starts after the LAST `)`; the next field is field 3.
pub(crate) fn parse_start_ticks(stat: &str) -> Option<u64> {
    let rest = &stat[stat.rfind(')')? + 1..];
    rest.split_whitespace().nth(22 - 3)?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_ticks_survive_a_comm_with_spaces_and_parens() {
        // 52 fields; field 22 = 987654.
        let mut fields: Vec<String> = (3..=52).map(|i| i.to_string()).collect();
        fields[22 - 3] = "987654".into();
        let stat = format!("4242 (a) b) c) {}", fields.join(" "));
        assert_eq!(parse_start_ticks(&stat), Some(987_654));
    }

    #[test]
    fn start_ticks_refuse_a_truncated_stat() {
        assert_eq!(parse_start_ticks("4242 (x) R 1 2"), None);
        assert_eq!(parse_start_ticks("no paren at all"), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn current_identity_is_stable_and_matches_of_pid() {
        let me = ProcIdentity::current().expect("linux has /proc");
        assert_eq!(me.pid, std::process::id());
        assert_eq!(ProcIdentity::of_pid(me.pid), Some(me));
    }
}
