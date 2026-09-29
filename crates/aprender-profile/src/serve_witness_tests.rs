//! TR-07 witness tests. FALSIFY-WITNESS-001..006 in
//! `contracts/serve-external-witness-v1.yaml` name these.

use super::*;
use crate::json_output::{JsonOutput, JsonSyscall};
use proptest::prelude::*;

fn sc(name: &str, fd_arg: i64, result: i64, ts_us: Option<u64>) -> JsonSyscall {
    JsonSyscall {
        name: name.to_string(),
        args: vec![format!("{fd_arg:#x}"), "0x0".to_string(), "0x0".to_string()],
        result,
        duration_us: None,
        source: None,
        ts_us,
        tid: Some(7),
    }
}

fn accept(fd: i64, ts: u64) -> JsonSyscall {
    // accept4's first argument is the listener, and it returns the client fd.
    sc("accept4", 3, fd, Some(ts))
}

fn read(fd: i64, n: i64, ts: u64) -> JsonSyscall {
    sc("recvfrom", fd, n, Some(ts))
}

fn write(fd: i64, n: i64, ts: u64) -> JsonSyscall {
    sc("writev", fd, n, Some(ts))
}

fn trace(rows: Vec<JsonSyscall>) -> JsonOutput {
    let mut out = JsonOutput::new();
    for r in rows {
        out.add_syscall(r);
    }
    out
}

fn line(id: &str, stream: bool, ttft_ms: Option<f64>, total_ms: f64) -> String {
    let ttft = ttft_ms.map_or("null".to_string(), |v| v.to_string());
    format!(
        "[request] {{\"request_id\":\"{id}\",\"client_request_id\":null,\"model\":\"m\",\
         \"endpoint\":\"chat\",\"backend\":\"gpu\",\"stream\":{stream},\"prompt_n\":4,\
         \"predicted_n\":8,\"prefill_ms\":1.0,\"decode_ms\":2.0,\"ttft_ms\":{ttft},\
         \"total_ms\":{total_ms},\"finish_reason\":\"stop\"}}"
    )
}

/// One streamed request: accept at 1 ms, request read at 2 ms, first chunk
/// at 301 ms, last chunk at 1201 ms, so TTFT^ext = 300 ms and wall^ext = 1200 ms.
fn streamed() -> JsonOutput {
    trace(vec![
        accept(9, 1_000),
        read(9, 120, 2_000),
        read(9, -11, 2_100), // EAGAIN: not request bytes
        write(9, 40, 301_000),
        write(9, 40, 800_000),
        write(9, 40, 1_201_000),
        write(2, 300, 1_202_000), // the server's own stderr line: not fd 9
    ])
}

/// FALSIFY-WITNESS-001: an honest server is GREEN on both metrics.
#[test]
fn honest_streamed_request_is_green() {
    let log = line("chatcmpl-1", true, Some(295.0), 1190.0);
    let rows = witness(&streamed(), &log).expect("measured");
    assert_eq!(rows.len(), 1);
    let r = &rows[0];
    assert_eq!(r.schema, WITNESS_SCHEMA);
    assert_eq!(r.ttft_ext_ms, Some(300.0));
    assert_eq!(r.wall_ext_ms, Some(1200.0));
    assert_eq!(r.request_bytes, 120);
    assert_eq!(r.response_bytes, 120);
    assert_eq!(r.anchor, Anchor::Accept4);
    assert!(r.traced);
    assert_eq!((r.ttft_verdict, r.wall_verdict), (Verdict::Green, Verdict::Green));
    assert_eq!(run_verdict(&rows), Verdict::Green);
}

/// FALSIFY-WITNESS-002 (the TR-07 acceptance): the server's own numbers,
/// skewed +20%, go RED from the witness alone. No server-side check runs.
#[test]
fn planted_plus_20_percent_skew_is_red_from_the_witness_alone() {
    let honest = (300.0_f64, 1200.0_f64);
    let log = line("chatcmpl-1", true, Some(honest.0 * 1.2), honest.1 * 1.2);
    let rows = witness(&streamed(), &log).expect("measured");
    assert_eq!(rows[0].ttft_verdict, Verdict::Red, "{:?}", rows[0]);
    assert_eq!(rows[0].wall_verdict, Verdict::Red, "{:?}", rows[0]);
    assert_eq!(run_verdict(&rows), Verdict::Red);
    // Under-reporting is caught too: a server that claims to be faster.
    let log = line("chatcmpl-1", true, Some(honest.0 * 0.8), honest.1 * 0.8);
    assert_eq!(run_verdict(&witness(&streamed(), &log).expect("measured")), Verdict::Red);
}

/// FALSIFY-WITNESS-003: non-streaming TTFT is null with a reason, and the
/// wall time is still compared.
#[test]
fn non_streaming_ttft_is_null_with_reason() {
    let t = trace(vec![accept(9, 0), read(9, 80, 500), write(9, 900, 700_000)]);
    let rows = witness(&t, &line("cmpl-1", false, Some(650.0), 690.0)).expect("measured");
    let r = &rows[0];
    assert_eq!(r.ttft_ext_ms, None);
    assert_eq!(r.ttft_reason.as_deref(), Some("non-streaming"));
    assert_eq!(r.ttft_server_ms, None);
    assert_eq!(r.ttft_verdict, Verdict::NotMeasured);
    assert_eq!(r.wall_verdict, Verdict::Green);
    assert_eq!(r.verdict(), Verdict::Green);
}

/// FALSIFY-WITNESS-004: nothing that cannot be measured passes.
#[test]
fn untimed_trace_is_not_measured() {
    let t = trace(vec![sc("accept4", 3, 9, None)]);
    assert_eq!(
        witness(&t, "").unwrap_err(),
        NotMeasured::Untimed { syscall: "accept4".to_string() }
    );
}

#[test]
fn join_count_mismatch_is_not_measured() {
    let two = format!("{}\n{}", line("a", true, Some(300.0), 1200.0), line("b", true, None, 1.0));
    assert_eq!(
        witness(&streamed(), &two).unwrap_err(),
        NotMeasured::JoinCount { witnessed: 1, server: 2 }
    );
    assert_eq!(
        witness(&streamed(), "no request lines").unwrap_err(),
        NotMeasured::JoinCount { witnessed: 1, server: 0 }
    );
}

#[test]
fn no_connection_is_not_measured() {
    let t = trace(vec![read(0, 10, 5), write(1, 10, 6)]);
    assert_eq!(witness(&t, "").unwrap_err(), NotMeasured::NoConnections);
}

#[test]
fn no_write_after_read_is_null_and_not_measured() {
    let t = trace(vec![accept(9, 0), read(9, 80, 500)]);
    let rows = witness(&t, &line("a", true, Some(1.0), 2.0)).expect("measured");
    let r = &rows[0];
    assert_eq!(r.ttft_ext_ms, None);
    assert_eq!(r.ttft_reason.as_deref(), Some("no-write-after-read"));
    assert_eq!(r.wall_ext_ms, None);
    assert_eq!(r.verdict(), Verdict::NotMeasured);
    assert_eq!(run_verdict(&rows), Verdict::NotMeasured);
}

#[test]
fn keep_alive_second_request_is_anchored_at_its_read() {
    let t = trace(vec![
        accept(9, 0),
        read(9, 50, 100),
        write(9, 10, 200_000),
        read(9, 60, 500_000),
        write(9, 10, 900_000),
    ]);
    let reqs = witnessed_requests(&socket_events(&t).expect("timed"));
    assert_eq!(reqs.len(), 2);
    assert_eq!((reqs[0].anchor, reqs[0].wall_ext_ms()), (Anchor::Accept4, Some(200.0)));
    assert_eq!((reqs[1].anchor, reqs[1].wall_ext_ms()), (Anchor::Read, Some(400.0)));
    assert_eq!(reqs[1].request_bytes, 60);
}

#[test]
fn reused_fd_number_starts_a_new_connection() {
    let t = trace(vec![
        accept(9, 0),
        read(9, 50, 100),
        write(9, 10, 100_000),
        accept(9, 2_000_000),
        read(9, 50, 2_000_100),
        write(9, 10, 2_300_000),
        accept(10, 3_000_000), // probe that sent nothing: dropped
    ]);
    let reqs = witnessed_requests(&socket_events(&t).expect("timed"));
    assert_eq!(reqs.len(), 2);
    assert_eq!(reqs[1].anchor_us, 2_000_000);
    assert_eq!(reqs[1].wall_ext_ms(), Some(300.0));
}

#[test]
fn write_before_any_read_answers_nothing() {
    let t = trace(vec![accept(9, 0), write(9, 10, 50), read(9, 50, 100), write(9, 10, 400_000)]);
    let reqs = witnessed_requests(&socket_events(&t).expect("timed"));
    // The stray write must not open a keep-alive turn: that would re-anchor
    // the request at its read and shrink TTFT by the accept→read gap.
    assert_eq!(reqs.len(), 1);
    assert_eq!((reqs[0].anchor, reqs[0].ttft_ext_ms()), (Anchor::Accept4, Some(400.0)));
    assert_eq!(reqs[0].first_write_us, Some(400_000));
    assert_eq!(reqs[0].response_bytes, 10);
}

#[test]
fn events_are_ordered_by_timestamp_not_row_order() {
    // Two threads: the write's exit row was recorded before the read's.
    let t = trace(vec![accept(9, 0), write(9, 10, 900), read(9, 50, 100)]);
    let reqs = witnessed_requests(&socket_events(&t).expect("timed"));
    assert_eq!(reqs[0].first_write_us, Some(900));
}

/// FALSIFY-WITNESS-005: the tolerance is max(5%, 50 ms), both edges.
#[test]
fn tolerance_is_max_of_five_percent_and_fifty_ms() {
    assert!((tolerance_ms(100.0) - 50.0).abs() < 1e-9);
    assert!((tolerance_ms(2000.0) - 100.0).abs() < 1e-9);
    assert_eq!(compare(Some(150.0), Some(100.0)), Verdict::Green);
    assert_eq!(compare(Some(150.1), Some(100.0)), Verdict::Red);
    assert_eq!(compare(Some(2100.0), Some(2000.0)), Verdict::Green);
    assert_eq!(compare(Some(2100.5), Some(2000.0)), Verdict::Red);
    assert_eq!(compare(None, Some(1.0)), Verdict::NotMeasured);
    assert_eq!(compare(Some(f64::NAN), Some(1.0)), Verdict::NotMeasured);
}

#[test]
fn fd_argument_parses_hex_and_decimal() {
    assert_eq!(parse_fd_arg("0x1f"), Some(31));
    assert_eq!(parse_fd_arg("12"), Some(12));
    assert_eq!(parse_fd_arg("\"/tmp\""), None);
}

#[test]
fn server_lines_parse_from_a_mixed_log() {
    let log = format!(
        "INFO listening\n{}\n2026-09-29T15:00:00Z apr[1]: {}\n[request] {{broken",
        line("a", true, Some(1.0), 2.0),
        line("b", false, None, 3.0)
    );
    let s = server_requests(&log);
    assert_eq!(s.iter().map(|r| r.request_id.as_str()).collect::<Vec<_>>(), ["a", "b"]);
}

#[test]
fn untimed_json_row_keeps_the_old_shape() {
    let mut row = sc("read", 1, 1, None);
    row.tid = None;
    let json = serde_json::to_string(&row).expect("ser");
    assert!(!json.contains("ts_us") && !json.contains("tid"), "{json}");
    let back: JsonSyscall =
        serde_json::from_str(r#"{"name":"read","args":[],"result":0}"#).expect("old rows parse");
    assert_eq!((back.ts_us, back.tid), (None, None));
}

fn arb_event() -> impl Strategy<Value = (u8, i64, u64)> {
    // (kind 0=read 1=write 2=other fd, bytes -1..=64, gap µs)
    (0u8..3, -1i64..=64, 0u64..10_000)
}

proptest! {
    /// FALSIFY-WITNESS-006, the executable form of KANI-TRACE-WITNESS: for
    /// up to 8 socket syscalls on one fd with monotone timestamps,
    /// TTFT^ext ≤ wall^ext, and TTFT is null exactly when no write followed
    /// a read of request bytes.
    #[test]
    fn ttft_never_exceeds_wall(evs in prop::collection::vec(arb_event(), 0..=8)) {
        let mut rows = vec![accept(9, 0)];
        let mut ts = 0u64;
        let mut read_seen = false;
        let mut write_after_read = false;
        for (kind, bytes, gap) in evs {
            ts += gap;
            match kind {
                0 => { rows.push(read(9, bytes, ts)); read_seen |= bytes > 0; }
                1 => { rows.push(write(9, bytes, ts)); write_after_read |= read_seen && bytes > 0; }
                _ => rows.push(write(4, bytes, ts)),
            }
        }
        let reqs = witnessed_requests(&socket_events(&trace(rows)).expect("timed"));
        for r in &reqs {
            match (r.ttft_ext_ms(), r.wall_ext_ms()) {
                (Some(t), Some(w)) => prop_assert!(t <= w, "ttft {t} > wall {w}"),
                (None, None) => {}
                other => prop_assert!(false, "ttft/wall disagree on presence: {other:?}"),
            }
        }
        let first_has_ttft = reqs.first().is_some_and(|r| r.ttft_ext_ms().is_some());
        prop_assert_eq!(first_has_ttft, write_after_read);
    }
}
