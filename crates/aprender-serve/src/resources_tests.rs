//! MEAS-001 R1 (#4522): the resources{} block, as tests that can fail.

use super::*;

// A comm with a space and a ')' — the parser must count from the LAST ')'.
const STAT: &str =
    "4242 (apr (x) y) R 1 4242 4242 0 -1 4194304 1234 0 7 0 250 31 0 0 20 0 9 0 100 1000 200";

#[test]
fn proc_stat_counts_fields_from_the_last_paren() {
    let s = parse_proc_stat(STAT).expect("parses");
    assert_eq!(
        s,
        StatCounters {
            minor_faults: 1234,
            major_faults: 7,
            utime_ticks: 250,
            stime_ticks: 31,
        }
    );
}

#[test]
fn proc_stat_truncated_is_none_not_zero() {
    assert_eq!(parse_proc_stat("4242 (apr) R 1 2 3"), None);
    assert_eq!(parse_proc_stat("no paren at all"), None);
}

#[test]
fn status_field_converts_kb_and_keeps_plain_counts() {
    let s = "Name:\tapr\nVmHWM:\t  2048 kB\nThreads:\t9\nvoluntary_ctxt_switches:\t55\n";
    assert_eq!(status_field(s, "VmHWM"), Some(2048 * 1024));
    assert_eq!(status_field(s, "Threads"), Some(9));
    assert_eq!(status_field(s, "voluntary_ctxt_switches"), Some(55));
    // A prefix of another key is not that key.
    assert_eq!(status_field(s, "Thread"), None);
    assert_eq!(status_field(s, "VmRSS"), None);
}

#[test]
fn proc_io_needs_all_four_counters() {
    let io = "rchar: 10\nwchar: 20\nsyscr: 1\nsyscw: 2\nread_bytes: 4096\nwrite_bytes: 8192\ncancelled_write_bytes: 0\n";
    assert_eq!(
        parse_proc_io(io),
        Some(IoCounters {
            rchar: 10,
            wchar: 20,
            read_bytes: 4096,
            write_bytes: 8192,
        })
    );
    assert_eq!(parse_proc_io("rchar: 10\nwchar: 20\n"), None);
}

#[test]
fn compute_apps_sums_this_pid_only() {
    let t = "111, 500\n4242, 1000\n4242, 24\n9999, 7000\n";
    assert_eq!(parse_compute_apps(t, 4242), Some(1024 * 1024 * 1024));
    assert_eq!(parse_compute_apps(t, 5), None);
    assert_eq!(parse_compute_apps("", 4242), None);
    // "[N/A]" memory is not a number and is not counted as 0.
    assert_eq!(parse_compute_apps("4242, [N/A]\n", 4242), None);
}

#[test]
fn gpu_energy_is_joules_and_refuses_na() {
    assert_eq!(parse_gpu_energy_mj("1500\n2500\n"), Some(4.0));
    assert_eq!(parse_gpu_energy_mj("1500\n[N/A]\n"), None);
    assert_eq!(parse_gpu_energy_mj(""), None);
}

#[test]
fn rapl_delta_survives_one_wrap() {
    assert_eq!(rapl_delta_uj(100, 350, 1000), 250);
    assert_eq!(rapl_delta_uj(900, 50, 1000), 150);
}

/// Every field is either measured with a source, or null with a reason —
/// never both, never neither.
fn assert_every_field_accounted(r: &ResourceUsage) {
    let v = serde_json::to_value(r).expect("serialize");
    let obj = v.as_object().expect("object");
    for (k, val) in obj {
        if matches!(k.as_str(), "wall_s" | "sources" | "null_reasons") {
            continue;
        }
        let has_src = r.sources.contains_key(k);
        let has_why = r.null_reasons.contains_key(k);
        if val.is_null() {
            assert!(has_why && !has_src, "{k} is null without a reason: {r:?}");
        } else {
            assert!(has_src && !has_why, "{k} is set without a source: {r:?}");
        }
    }
}

#[cfg(target_os = "linux")]
#[test]
fn live_window_measures_rss_cpu_and_threads() {
    const MIB: usize = 1024 * 1024;
    let s = ResourceSampler::start();
    // Touch 64 MiB so it is resident, not just reserved.
    let mut buf = vec![0u8; 64 * MIB];
    for i in (0..buf.len()).step_by(4096) {
        buf[i] = (i % 251) as u8;
    }
    // Four extra threads alive across at least one 100 ms sample.
    let hs: Vec<_> = (0..4)
        .map(|_| std::thread::spawn(|| std::thread::sleep(Duration::from_millis(350))))
        .collect();
    // Burn CPU for ~300 ms of user time.
    let t = Instant::now();
    let mut x = 0u64;
    while t.elapsed() < Duration::from_millis(300) {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
    }
    std::hint::black_box(x);
    for h in hs {
        h.join().expect("join");
    }
    std::hint::black_box(&buf);
    let r = s.finish();

    assert_every_field_accounted(&r);
    assert!(r.peak_rss_bytes.expect("rss") >= 64 * MIB as u64, "{r:?}");
    assert!(r.cpu_user_s.expect("cpu") >= 0.1, "{r:?}");
    assert!(
        r.minor_faults.expect("faults") >= 1000,
        "64 MiB touched: {r:?}"
    );
    assert!(r.threads_peak.expect("threads") >= 5, "{r:?}");
    assert!(r.wall_s >= 0.3, "{r:?}");
    for k in [
        "peak_rss_bytes",
        "cpu_user_s",
        "threads_peak",
        "io_rchar_bytes",
    ] {
        assert!(r.sources.contains_key(k), "{k}: {r:?}");
    }
}

#[cfg(target_os = "linux")]
#[test]
fn syscall_io_is_counted() {
    let p = std::env::temp_dir().join(format!("meas001-io-{}", std::process::id()));
    let (_, r) = measure(|| {
        std::fs::write(&p, vec![7u8; 1 << 20]).expect("write");
        std::fs::read(&p).expect("read")
    });
    let _ = std::fs::remove_file(&p);
    assert!(r.io_wchar_bytes.expect("wchar") >= 1 << 20, "{r:?}");
    assert!(r.io_rchar_bytes.expect("rchar") >= 1 << 20, "{r:?}");
}

#[test]
fn json_keys_are_the_contract_r2_reads() {
    let v = serde_json::to_value(ResourceUsage::default()).expect("serialize");
    for k in [
        "peak_rss_bytes",
        "vram_peak_bytes",
        "cpu_user_s",
        "cpu_sys_s",
        "minor_faults",
        "major_faults",
        "voluntary_ctx_switches",
        "involuntary_ctx_switches",
        "threads_peak",
        "io_read_bytes",
        "io_write_bytes",
        "energy_j",
        "null_reasons",
        "sources",
    ] {
        assert!(v.get(k).is_some(), "missing {k}");
    }
}
