//! Attribute a *listening* TCP socket to the process that owns it.
//!
//! # Why this exists (#2606, second pass)
//!
//! `apr.serve` spawns a daemon and then has to answer one question honestly:
//! *is the URL I am about to hand back reachable **because of the child I
//! spawned**?* A TCP connect to `127.0.0.1:<port>` answers a strictly weaker
//! question — "is **anything** listening there?" — and the two come apart in
//! the exact case the tool exists to report on. An adversarial pass reproduced
//! the original fabricated-URL shape against the first #2606 fix by holding
//! the port with an unrelated process: the child never bound anything, the
//! probe succeeded anyway, and the tool reported
//! `{"ready":true,"url":"http://localhost:<port>"}` for a server that does not
//! exist. A liveness check on the child does not repair that: both halves —
//! "our child is alive" and "something answers on that port" — were true, and
//! their conjunction still is not the claim being made.
//!
//! So the probe has to be tightened from *reachability* to *attribution*:
//! the listening socket must be held by the spawned child or one of its
//! descendants.
//!
//! # What is and is not portable
//!
//! Mapping a listening socket back to a pid needs an OS interface. On Linux
//! `/proc/net/tcp{,6}` gives the socket inode for every listening port, and
//! `/proc/<pid>/fd/*` gives the inodes a process holds; intersecting them is
//! exact, needs no privileges for our own descendants, and shells out to
//! nothing. There is no equivalent that is portable across every unix — macOS
//! needs `libproc`/`lsof`, and neither is available here under
//! `unsafe_code = "forbid"` with no new dependencies.
//!
//! This module therefore reports [`PortOwner::Unknown`] rather than guessing
//! on platforms it cannot interrogate, and the caller degrades to a **stated,
//! weaker** guarantee there instead of pretending to the stronger one. See
//! `serve.rs` for how the two are kept distinguishable in the payload.

/// Who holds the listening socket on a port, as far as this platform can say.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortOwner {
    /// A listening socket on the port is held by the target pid or one of its
    /// descendants. A url naming that port is attributable to the child.
    Child,
    /// Something is listening (or the kernel tables say nothing at all is),
    /// but no file descriptor belonging to the target pid or its descendants
    /// holds a listening socket on that port. A url would name someone else's
    /// server.
    Foreign,
    /// This platform cannot map a listening socket to a pid. Nothing is
    /// claimed either way — callers must not read this as `Child`.
    Unknown,
}

/// Attribute the listening socket on `port` to `pid` (or a descendant of it).
///
/// Fails **closed**: anything this platform cannot prove is [`Foreign`] on a
/// platform that has the tables, and [`Unknown`] on one that does not. It is
/// never [`Child`] without a matching socket inode found on one of the child's
/// own file descriptors.
///
/// [`Foreign`]: PortOwner::Foreign
/// [`Child`]: PortOwner::Child
/// [`Unknown`]: PortOwner::Unknown
#[must_use]
pub fn owner_of_listening_port(port: u16, pid: u32) -> PortOwner {
    #[cfg(target_os = "linux")]
    {
        linux::owner_of_listening_port(port, pid)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (port, pid);
        PortOwner::Unknown
    }
}

/// Every process this platform can see holding a LISTEN socket on `port`,
/// as `(pid, comm)`.
///
/// Diagnostic only — the url decision is [`owner_of_listening_port`]. Naming
/// the squatter is the difference between "port 8080 is busy" and "port 8080
/// is held by pid 4242 (`apr`)", which is what a caller needs to unblock
/// themselves. Empty when the platform cannot say, or when the holder belongs
/// to another user (whose `/proc/<pid>/fd` is not readable).
#[must_use]
pub fn listening_pids(port: u16) -> Vec<(u32, String)> {
    #[cfg(target_os = "linux")]
    {
        linux::listening_pids(port)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = port;
        Vec::new()
    }
}

/// Whether this build can attribute a socket to a process at all.
///
/// Exposed so the tool can say *which* guarantee it is offering rather than
/// silently offering the weaker one.
#[must_use]
pub const fn attribution_available() -> bool {
    cfg!(target_os = "linux")
}

#[cfg(target_os = "linux")]
mod linux {
    use super::PortOwner;
    use std::collections::HashSet;

    /// TCP state code for `LISTEN` in `/proc/net/tcp` (hex, column 4).
    const TCP_LISTEN: &str = "0A";

    /// Cap on the ppid walk so a pathological (or racing) `/proc` cannot spin.
    const MAX_PROCS: usize = 65_536;

    pub fn owner_of_listening_port(port: u16, pid: u32) -> PortOwner {
        let Some(inodes) = listening_inodes(port) else {
            // Neither table readable: /proc is not mounted (container, chroot),
            // so this build cannot attribute after all.
            return PortOwner::Unknown;
        };
        if inodes.is_empty() {
            // Something answered a connect but no LISTEN row exists for the
            // port: a different netns, or a race with a closing socket.
            // Either way it is not attributable to the child.
            return PortOwner::Foreign;
        }
        for candidate in pid_and_descendants(pid) {
            if holds_any_inode(candidate, &inodes) {
                return PortOwner::Child;
            }
        }
        PortOwner::Foreign
    }

    pub fn listening_pids(port: u16) -> Vec<(u32, String)> {
        let Some(inodes) = listening_inodes(port) else {
            return Vec::new();
        };
        if inodes.is_empty() {
            return Vec::new();
        }
        let mut holders = Vec::new();
        let Ok(entries) = std::fs::read_dir("/proc") else {
            return holders;
        };
        for entry in entries.flatten().take(MAX_PROCS) {
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            let Ok(pid) = name.parse::<u32>() else {
                continue;
            };
            if holds_any_inode(pid, &inodes) {
                let comm = std::fs::read_to_string(format!("/proc/{pid}/comm"))
                    .map_or_else(|_| "?".to_string(), |c| c.trim().to_string());
                holders.push((pid, comm));
            }
        }
        holders
    }

    /// Socket inodes of every `LISTEN` socket bound to `port`, across IPv4 and
    /// IPv6. `None` when neither table could be read.
    fn listening_inodes(port: u16) -> Option<HashSet<u64>> {
        let mut found = HashSet::new();
        let mut readable = false;
        for table in ["/proc/net/tcp", "/proc/net/tcp6"] {
            let Ok(text) = std::fs::read_to_string(table) else {
                continue;
            };
            readable = true;
            for line in text.lines().skip(1) {
                if let Some(inode) = listening_inode_on_port(line, port) {
                    found.insert(inode);
                }
            }
        }
        readable.then_some(found)
    }

    /// Parse one `/proc/net/tcp{,6}` row; `Some(inode)` iff it is a `LISTEN`
    /// socket whose local port is `port`.
    ///
    /// Row shape (columns are whitespace separated, addresses are hex):
    /// `sl local_address rem_address st tx:rx tr:when retrnsmt uid timeout inode ...`
    fn listening_inode_on_port(line: &str, port: u16) -> Option<u64> {
        let cols: Vec<&str> = line.split_whitespace().collect();
        // Column 9 is the inode; anything shorter is not a socket row.
        if cols.len() < 10 {
            return None;
        }
        if cols[3] != TCP_LISTEN {
            return None;
        }
        let local_port_hex = cols[1].rsplit(':').next()?;
        let local_port = u16::from_str_radix(local_port_hex, 16).ok()?;
        if local_port != port {
            return None;
        }
        cols[9].parse::<u64>().ok()
    }

    /// `pid` followed by every transitive descendant of it, from `/proc`.
    ///
    /// A daemon that forks (or is wrapped by a shell) still counts as ours;
    /// nothing outside the subtree ever does.
    fn pid_and_descendants(pid: u32) -> Vec<u32> {
        let mut parents: Vec<(u32, u32)> = Vec::new(); // (child, parent)
        if let Ok(entries) = std::fs::read_dir("/proc") {
            for entry in entries.flatten().take(MAX_PROCS) {
                let name = entry.file_name();
                let Some(name) = name.to_str() else { continue };
                let Ok(candidate) = name.parse::<u32>() else {
                    continue;
                };
                if let Some(ppid) = parent_of(candidate) {
                    parents.push((candidate, ppid));
                }
            }
        }

        let mut subtree = vec![pid];
        let mut seen: HashSet<u32> = HashSet::from([pid]);
        let mut cursor = 0;
        while cursor < subtree.len() {
            let parent = subtree[cursor];
            cursor += 1;
            for &(child, child_parent) in &parents {
                if child_parent == parent && seen.insert(child) {
                    subtree.push(child);
                }
            }
        }
        subtree
    }

    /// PPID of `pid` from `/proc/<pid>/stat`.
    ///
    /// The `comm` field is parenthesised and may itself contain spaces and
    /// parentheses, so the fixed columns are counted from the LAST `)`.
    fn parent_of(pid: u32) -> Option<u32> {
        let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
        let after_comm = stat.rfind(')').map(|i| &stat[i + 1..])?;
        // After `)`: state, ppid, ...
        after_comm.split_whitespace().nth(1)?.parse::<u32>().ok()
    }

    /// Whether `pid` holds an open file descriptor for any of `inodes`.
    fn holds_any_inode(pid: u32, inodes: &HashSet<u64>) -> bool {
        let Ok(fds) = std::fs::read_dir(format!("/proc/{pid}/fd")) else {
            return false;
        };
        for fd in fds.flatten() {
            let Ok(target) = std::fs::read_link(fd.path()) else {
                continue;
            };
            let target = target.to_string_lossy();
            let Some(rest) = target.strip_prefix("socket:[") else {
                continue;
            };
            let Some(digits) = rest.strip_suffix(']') else {
                continue;
            };
            if digits.parse::<u64>().is_ok_and(|ino| inodes.contains(&ino)) {
                return true;
            }
        }
        false
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// A real `LISTEN` row from `/proc/net/tcp` (port 0x1F90 = 8080).
        const LISTEN_ROW: &str = "   0: 0100007F:1F90 00000000:0000 0A 00000000:00000000 \
                                  00:00000000 00000000  1000        0 4242424 1 0000000000000000 \
                                  100 0 0 10 0";
        /// The same socket in `ESTABLISHED` (01) rather than `LISTEN`.
        const ESTABLISHED_ROW: &str = "   1: 0100007F:1F90 0100007F:C000 01 00000000:00000000 \
                                       00:00000000 00000000  1000        0 4242425 1 \
                                       0000000000000000 20 0 0 10 -1";

        #[test]
        fn listen_row_on_the_asked_for_port_yields_its_inode() {
            assert_eq!(listening_inode_on_port(LISTEN_ROW, 8080), Some(4_242_424));
        }

        #[test]
        fn listen_row_on_a_different_port_is_not_matched() {
            assert_eq!(listening_inode_on_port(LISTEN_ROW, 8081), None);
        }

        /// The state column is the difference between "a server is waiting
        /// here" and "someone once connected here". Only the former is a
        /// listener; matching on port alone would attribute a stale
        /// established socket.
        #[test]
        fn established_row_is_not_a_listener() {
            assert_eq!(listening_inode_on_port(ESTABLISHED_ROW, 8080), None);
        }

        #[test]
        fn header_and_garbage_rows_are_ignored() {
            for junk in [
                "  sl  local_address rem_address   st tx_queue rx_queue",
                "",
                "   0: 0100007F:1F90 0A",
            ] {
                assert_eq!(listening_inode_on_port(junk, 8080), None, "junk: {junk:?}");
            }
        }

        /// `/proc/<pid>/stat`'s `comm` is attacker-controlled: a process can
        /// name itself `") 1 999999 ("`. Counting columns from the LAST `)`
        /// is what keeps the ppid honest.
        #[test]
        fn ppid_parse_survives_a_comm_containing_spaces_and_parens() {
            // Emulate the field layout: pid (comm) state ppid ...
            let stat = "1234 (evil ) 1 999999 (name) S 4321 1234 1234 0 -1 4194304";
            let after = stat.rfind(')').map(|i| &stat[i + 1..]).expect("has )");
            let ppid: u32 = after
                .split_whitespace()
                .nth(1)
                .and_then(|f| f.parse().ok())
                .expect("ppid parses");
            assert_eq!(ppid, 4321, "ppid must come from after the LAST )");
        }

        /// The live half: this test process holds a listening socket, so the
        /// kernel tables plus its own fds must attribute it to itself.
        #[test]
        fn a_listener_this_process_holds_is_attributed_to_this_process() {
            let listener = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
                .expect("bind ephemeral port");
            let port = listener.local_addr().expect("local_addr").port();
            assert_eq!(
                owner_of_listening_port(port, std::process::id()),
                PortOwner::Child,
                "the process that owns the socket must be recognised as owning it"
            );
            drop(listener);
        }

        /// The half that closes #2606's second pass: the *same* live listener,
        /// asked about a process that did not create it, must come back
        /// `Foreign`. A parent is not a descendant of its own child, so a
        /// spawned `sleep` is a sound stand-in for "an unrelated process".
        #[test]
        fn a_listener_held_by_someone_else_is_foreign() {
            let listener = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
                .expect("bind ephemeral port");
            let port = listener.local_addr().expect("local_addr").port();
            let mut other = std::process::Command::new("sleep")
                .arg("30")
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .expect("spawn sleep");
            assert_eq!(
                owner_of_listening_port(port, other.id()),
                PortOwner::Foreign,
                "a socket held by an unrelated process must NEVER be attributed to it"
            );
            let _ = other.kill();
            let _ = other.wait();
            drop(listener);
        }

        /// Nothing listening at all is not attributable either.
        #[test]
        fn a_port_with_no_listener_is_foreign() {
            let port = {
                let l = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
                    .expect("bind ephemeral port");
                l.local_addr().expect("local_addr").port()
            };
            assert_eq!(
                owner_of_listening_port(port, std::process::id()),
                PortOwner::Foreign
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The platform gate and the resolver must agree: a build that says it
    /// cannot attribute must never return `Child`.
    #[test]
    fn unavailable_attribution_never_claims_child() {
        if !attribution_available() {
            assert_eq!(
                owner_of_listening_port(8080, std::process::id()),
                PortOwner::Unknown,
                "a platform without socket->pid tables must say Unknown, not guess"
            );
        }
    }

    #[test]
    fn attribution_availability_matches_the_platform() {
        assert_eq!(attribution_available(), cfg!(target_os = "linux"));
    }
}
