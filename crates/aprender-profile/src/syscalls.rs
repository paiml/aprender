//! Syscall number to name mapping
//!
//! Sprint 3-4: Full syscall coverage
//! Supports x86_64 and aarch64 (Linux)
//!
//! Uses sorted static tables with binary search for O(log n) lookup,
//! keeping cyclomatic complexity minimal regardless of table size.

/// Resolve syscall number to name
///
/// Returns the syscall name, or "`unknown`" if unrecognized.
/// Syscall numbers differ by architecture.
pub fn syscall_name(num: i64) -> &'static str {
    lookup_syscall(num, syscall_table())
}

/// Binary-search a sorted `(number, name)` table.
fn lookup_syscall(num: i64, table: &[(i64, &'static str)]) -> &'static str {
    match table.binary_search_by_key(&num, |&(n, _)| n) {
        Ok(idx) => table[idx].1,
        Err(_) => "unknown",
    }
}

/// x86_64 syscall table (Linux, from arch/x86/entry/syscalls/syscall_64.tbl)
#[cfg(target_arch = "x86_64")]
static SYSCALL_TABLE_X86_64: &[(i64, &str)] = &[
    (0, "read"),
    (1, "write"),
    (2, "open"),
    (3, "close"),
    (4, "stat"),
    (5, "fstat"),
    (6, "lstat"),
    (7, "poll"),
    (8, "lseek"),
    (9, "mmap"),
    (10, "mprotect"),
    (11, "munmap"),
    (12, "brk"),
    (13, "rt_sigaction"),
    (14, "rt_sigprocmask"),
    (15, "rt_sigreturn"),
    (16, "ioctl"),
    (17, "pread64"),
    (18, "pwrite64"),
    (19, "readv"),
    (20, "writev"),
    (21, "access"),
    (22, "pipe"),
    (23, "select"),
    (24, "sched_yield"),
    (25, "mremap"),
    (26, "msync"),
    (27, "mincore"),
    (28, "madvise"),
    (29, "shmget"),
    (30, "shmat"),
    (31, "shmctl"),
    (32, "dup"),
    (33, "dup2"),
    (34, "pause"),
    (35, "nanosleep"),
    (36, "getitimer"),
    (37, "alarm"),
    (38, "setitimer"),
    (39, "getpid"),
    (40, "sendfile"),
    (41, "socket"),
    (42, "connect"),
    (43, "accept"),
    (44, "sendto"),
    (45, "recvfrom"),
    (46, "sendmsg"),
    (47, "recvmsg"),
    (48, "shutdown"),
    (49, "bind"),
    (50, "listen"),
    (51, "getsockname"),
    (52, "getpeername"),
    (53, "socketpair"),
    (54, "setsockopt"),
    (55, "getsockopt"),
    (56, "clone"),
    (57, "fork"),
    (58, "vfork"),
    (59, "execve"),
    (60, "exit"),
    (61, "wait4"),
    (62, "kill"),
    (63, "uname"),
    (72, "fcntl"),
    (73, "flock"),
    (74, "fsync"),
    (75, "fdatasync"),
    (76, "truncate"),
    (77, "ftruncate"),
    (78, "getdents"),
    (79, "getcwd"),
    (80, "chdir"),
    (81, "fchdir"),
    (82, "rename"),
    (83, "mkdir"),
    (84, "rmdir"),
    (85, "creat"),
    (86, "link"),
    (87, "unlink"),
    (88, "symlink"),
    (89, "readlink"),
    (90, "chmod"),
    (91, "fchmod"),
    (92, "chown"),
    (93, "fchown"),
    (94, "lchown"),
    (95, "umask"),
    (96, "gettimeofday"),
    (97, "getrlimit"),
    (98, "getrusage"),
    (99, "sysinfo"),
    (102, "getuid"),
    (104, "getgid"),
    (105, "setuid"),
    (107, "setgid"),
    (108, "geteuid"),
    (109, "getegid"),
    (110, "setpgid"),
    (111, "getppid"),
    (112, "getpgrp"),
    (113, "setsid"),
    (131, "sigaltstack"),
    (157, "prctl"),
    (158, "arch_prctl"),
    (186, "gettid"),
    (202, "futex"),
    (217, "getdents64"),
    (218, "set_tid_address"),
    (228, "clock_gettime"),
    // 229/230 were missing: the table jumped 228 -> 231, so clock_nanosleep
    // rendered as an unnamed syscall. sprint20_realtime_anomaly_tests
    // deliberately sleeps 50ms and asserts an anomaly is reported; the
    // detector never saw the sleep, and the test only passed when unrelated
    // microsecond jitter on `write` happened to exceed 3 sigma.
    (229, "clock_getres"),
    (230, "clock_nanosleep"),
    (231, "exit_group"),
    (257, "openat"),
    (262, "newfstatat"),
    (273, "set_robust_list"),
    (302, "pkey_mprotect"),
    (318, "getrandom"),
    (329, "pkey_mprotect"),
    (330, "pkey_alloc"),
    (331, "pkey_free"),
    (332, "statx"),
    (334, "rseq"),
    (435, "clone3"),
];

#[cfg(target_arch = "x86_64")]
fn syscall_table() -> &'static [(i64, &'static str)] {
    SYSCALL_TABLE_X86_64
}

/// aarch64 syscall table (Linux, from asm-generic/unistd.h)
///
/// aarch64 uses the "new" unified syscall numbering (no legacy x86 heritage).
/// Numbers sourced from Linux 6.8 include/uapi/asm-generic/unistd.h
#[cfg(target_arch = "aarch64")]
static SYSCALL_TABLE_AARCH64: &[(i64, &str)] = &[
    (17, "getcwd"),
    (23, "dup"),
    (24, "dup3"),
    (25, "fcntl"),
    (29, "ioctl"),
    (33, "mknodat"),
    (34, "mkdirat"),
    (35, "unlinkat"),
    (36, "symlinkat"),
    (37, "linkat"),
    (38, "renameat"),
    (43, "statfs"),
    (44, "fstatfs"),
    (45, "truncate"),
    (46, "ftruncate"),
    (48, "faccessat"),
    (49, "chdir"),
    (50, "fchdir"),
    (51, "chroot"),
    (52, "fchmod"),
    (53, "fchmodat"),
    (54, "fchownat"),
    (55, "fchown"),
    (56, "openat"),
    (57, "close"),
    (59, "pipe2"),
    (61, "getdents64"),
    (62, "lseek"),
    (63, "read"),
    (64, "write"),
    (65, "readv"),
    (66, "writev"),
    (67, "pread64"),
    (68, "pwrite64"),
    (72, "pselect6"),
    (73, "ppoll"),
    (78, "readlinkat"),
    (79, "newfstatat"),
    (80, "fstat"),
    (82, "fsync"),
    (83, "fdatasync"),
    (88, "utimensat"),
    (93, "exit"),
    (94, "exit_group"),
    (96, "set_tid_address"),
    (98, "futex"),
    (99, "set_robust_list"),
    (101, "nanosleep"),
    (103, "setitimer"),
    (104, "getitimer"),
    (113, "clock_gettime"),
    (116, "sched_yield"),
    (117, "sched_setscheduler"),
    (118, "sched_getscheduler"),
    (122, "sched_setaffinity"),
    (123, "sched_getaffinity"),
    (124, "sched_yield"),
    (129, "kill"),
    (131, "tgkill"),
    (132, "sigaltstack"),
    (134, "rt_sigaction"),
    (135, "rt_sigprocmask"),
    (139, "rt_sigreturn"),
    (153, "times"),
    (157, "prctl"),
    (160, "uname"),
    (166, "umask"),
    (167, "getrusage"),
    (172, "getpid"),
    (173, "getppid"),
    (174, "getuid"),
    (175, "geteuid"),
    (176, "getgid"),
    (177, "getegid"),
    (178, "gettid"),
    (179, "sysinfo"),
    (196, "shmget"),
    (198, "shmdt"),
    (200, "socket"),
    (201, "socketpair"),
    (202, "bind"),
    (203, "listen"),
    (204, "accept"),
    (205, "connect"),
    (206, "getsockname"),
    (207, "getpeername"),
    (208, "sendto"),
    (209, "recvfrom"),
    (210, "setsockopt"),
    (211, "getsockopt"),
    (212, "shutdown"),
    (213, "sendmsg"),
    (214, "recvmsg"),
    (215, "readahead"),
    (216, "brk"),
    (217, "munmap"),
    (218, "mremap"),
    (220, "clone"),
    (221, "execve"),
    (222, "mmap"),
    (226, "mprotect"),
    (227, "msync"),
    (228, "madvise"),
    (233, "mlock"),
    (234, "munlock"),
    (237, "sendfile"),
    (242, "accept4"),
    (260, "wait4"),
    (261, "prlimit64"),
    (262, "fanotify_init"),
    (268, "getrandom"),
    (276, "renameat2"),
    (278, "getrandom"),
    (280, "memfd_create"),
    (281, "bpf"),
    (291, "statx"),
    (293, "rseq"),
    (435, "clone3"),
];

#[cfg(target_arch = "aarch64")]
fn syscall_table() -> &'static [(i64, &'static str)] {
    SYSCALL_TABLE_AARCH64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_common_syscalls() {
        // These names must resolve on any architecture
        let common = ["read", "write", "close", "openat", "mmap", "brk", "clone3"];
        for name in common {
            // At least one number should map to this name
            let found = (-1..500).any(|n| syscall_name(n) == name);
            assert!(found, "syscall '{}' should exist in the table", name);
        }
    }

    #[test]
    fn test_unknown_syscall() {
        assert_eq!(syscall_name(9999), "unknown");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_x86_64_syscalls() {
        assert_eq!(syscall_name(0), "read");
        assert_eq!(syscall_name(1), "write");
        assert_eq!(syscall_name(2), "open");
        assert_eq!(syscall_name(3), "close");
        assert_eq!(syscall_name(257), "openat");
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_aarch64_syscalls() {
        assert_eq!(syscall_name(63), "read");
        assert_eq!(syscall_name(64), "write");
        assert_eq!(syscall_name(57), "close");
        assert_eq!(syscall_name(56), "openat");
        assert_eq!(syscall_name(222), "mmap");
    }

    #[test]
    fn test_syscall_name_never_panics() {
        for num in [-1000, -1, 0, 1, 100, 500, 1000, 9999, i64::MAX] {
            let name = syscall_name(num);
            assert!(!name.is_empty(), "Syscall name should never be empty for {}", num);
        }
    }

    #[test]
    fn test_syscall_name_always_returns_str() {
        for num in 0..500 {
            let name = syscall_name(num);
            assert!(!name.is_empty());
            assert!(name.chars().all(|c| c.is_alphanumeric() || c == '_'));
        }
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_syscall_name_never_panics(num in any::<i64>()) {
            let name = syscall_name(num);
            prop_assert!(!name.is_empty());
        }

        #[test]
        fn prop_syscall_name_deterministic(num in 0..500i64) {
            let name1 = syscall_name(num);
            let name2 = syscall_name(num);
            prop_assert_eq!(name1, name2);
        }

        #[test]
        fn prop_unknown_syscalls_return_unknown(num in 500..10000i64) {
            let name = syscall_name(num);
            // clone3 (435) is below our range, so all should be unknown
            prop_assert_eq!(name, "unknown");
        }
    }
}

#[cfg(test)]
mod table_completeness {
    //! The x86_64 table jumped 228 -> 231, so `clock_nanosleep` (230) rendered
    //! as an unnamed `syscall_230` and a deliberate 50ms sleep was invisible to
    //! the profiler. Nothing caught it: removing the entries again left all 107
    //! syscall tests green, because every existing test asserts a PROPERTY
    //! (determinism, non-empty) rather than a specific mapping.
    //!
    //! A property test over `0..500` cannot catch a missing entry -- "unknown"
    //! is a valid answer across most of that range. Only naming numbers can.

    use super::*;

    /// Syscalls this tracer must be able to name, numbers fixed by the x86_64
    /// ABI. These are the ones a profiler actually meets.
    const MUST_NAME: &[(i64, &str)] = &[
        (0, "read"),
        (1, "write"),
        (60, "exit"),
        (202, "futex"),
        (228, "clock_gettime"),
        (229, "clock_getres"),
        (230, "clock_nanosleep"),
        (231, "exit_group"),
        (257, "openat"),
    ];

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn every_common_syscall_resolves_to_its_abi_name() {
        let missing: Vec<String> = MUST_NAME
            .iter()
            .filter(|(n, want)| syscall_name(*n) != *want)
            .map(|(n, want)| format!("{n} should be {want}, got {}", syscall_name(*n)))
            .collect();
        assert!(missing.is_empty(), "syscall table is missing or wrong for: {missing:?}");
    }

    /// Non-vacuity. If `syscall_name` produced a name for anything asked of it,
    /// the assertion above would pass while proving nothing.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn an_unassigned_number_is_still_unknown() {
        assert_eq!(
            syscall_name(9_999),
            "unknown",
            "an unassigned syscall number resolved to a name, so the check above \
             cannot distinguish a populated table from a permissive one"
        );
    }

    /// The gap that produced the bug was contiguous. The clock family is where
    /// the tracer's timing anomalies live, so assert it is whole.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn the_clock_syscall_range_has_no_holes() {
        for n in 228..=231 {
            assert_ne!(
                syscall_name(n),
                "unknown",
                "syscall {n} is unnamed; the 228-231 range is where clock_nanosleep \
                 went missing and made a 50ms sleep invisible to the profiler"
            );
        }
    }
}
