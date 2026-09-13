//! e2e: the `http` transport, against the SHIPPED BINARY.
//!
//! Declared in the root `Cargo.toml` as
//! `[package.metadata.transports] http = { e2e = "e2e_http_serve_t", features = ["cli"] }`.
//!
//! WHAT IS EXERCISED, exactly: `apr serve run <fixture> --host 127.0.0.1
//! --port <p>` — the run mode of `apr serve` (the other mode, `apr serve plan`,
//! is a pre-flight report and binds no socket). The child is given the tracked
//! 1.07 KiB `golden_v2.apr`, which loads through the `AprTransformer` fallback
//! path; the test then polls `GET /health`, which the binary lists first in its
//! own startup banner, until it answers 200 and reports `"status":"healthy"`.
//! `/v1/models` is deliberately NOT probed: this server does not serve that
//! route (it answers 404 with the route list `/health, /v1/completions,
//! /v1/chat/completions, /api/chat, /api/generate, /api/tags`), and a test that
//! asserted it would be asserting about a different server.
//!
//! Hermetic: localhost only, a port taken by binding 0 and releasing it, a
//! model already in the repo, a deadline on every wait, and the child killed on
//! the way out — including on a panicking assertion.

use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

/// How long the server may take to bind and answer. Generous: this is a
/// debug-profile build doing a cold model load on a shared, deliberately
/// over-subscribed box.
const READY_TIMEOUT: Duration = Duration::from_secs(180);
const IO_TIMEOUT: Duration = Duration::from_secs(15);

/// Ask the kernel for a free port, then release it. There is an unavoidable
/// race between release and the child's bind; it is the same race every test
/// harness runs, and it is far smaller than the collision rate of a hardcoded
/// port on a box running many agents at once.
fn free_port() -> u16 {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("bind an ephemeral port");
    listener.local_addr().expect("read the bound port").port()
}

/// A child that is killed when it goes out of scope, panic or not.
struct ServerChild {
    child: Child,
    log: PathBuf,
}

impl ServerChild {
    fn log_tail(&self) -> String {
        std::fs::read_to_string(&self.log).unwrap_or_else(|e| format!("<unreadable log: {e}>"))
    }
}

impl Drop for ServerChild {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_file(&self.log);
    }
}

/// One HTTP/1.1 request, hand-written over a raw socket. `Connection: close`
/// makes the response self-delimiting, so `read_to_end` terminates without this
/// test having to parse chunked framing.
fn http_get(port: u16, path: &str) -> Result<String, String> {
    let addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, port));
    let mut sock =
        TcpStream::connect_timeout(&addr, IO_TIMEOUT).map_err(|e| format!("connect: {e}"))?;
    sock.set_read_timeout(Some(IO_TIMEOUT))
        .map_err(|e| format!("set_read_timeout: {e}"))?;
    sock.set_write_timeout(Some(IO_TIMEOUT))
        .map_err(|e| format!("set_write_timeout: {e}"))?;
    let req = format!(
        "GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nAccept: */*\r\nConnection: close\r\n\r\n"
    );
    sock.write_all(req.as_bytes())
        .map_err(|e| format!("write: {e}"))?;
    let mut raw = Vec::new();
    sock.read_to_end(&mut raw)
        .map_err(|e| format!("read: {e}"))?;
    Ok(String::from_utf8_lossy(&raw).into_owned())
}

fn spawn_server(port: u16, fixture: &PathBuf) -> ServerChild {
    // The child's stdout is a banner plus per-request logging. Piping it and
    // never draining it would wedge the server the moment the pipe filled, so
    // it goes to a file the assertions can quote back.
    let log = std::env::temp_dir().join(format!("apr-e2e-serve-{port}.log"));
    let out = std::fs::File::create(&log).expect("create the server log file");
    let err = out.try_clone().expect("clone the server log handle");
    let child = Command::new(env!("CARGO_BIN_EXE_apr"))
        .arg("serve")
        .arg("run")
        .arg(fixture)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::from(out))
        .stderr(Stdio::from(err))
        .spawn()
        .expect("spawn the shipped apr binary with `serve run`");
    ServerChild { child, log }
}

/// Poll `/health` until it answers, failing fast if the child dies first.
fn wait_for_health(server: &mut ServerChild, port: u16) -> String {
    let deadline = Instant::now() + READY_TIMEOUT;
    let mut last = String::from("never attempted");
    while Instant::now() < deadline {
        if let Some(status) = server.child.try_wait().expect("poll the serve child") {
            panic!(
                "`apr serve run` exited ({status}) before binding port {port}.\n\
                 last probe: {last}\n--- server log ---\n{}",
                server.log_tail()
            );
        }
        match http_get(port, "/health") {
            Ok(resp) => return resp,
            Err(e) => last = e,
        }
        thread::sleep(Duration::from_millis(200));
    }
    panic!(
        "`apr serve run` never answered GET /health on port {port} within \
         {READY_TIMEOUT:?}; last probe: {last}\n--- server log ---\n{}",
        server.log_tail()
    );
}

#[test]
fn serve_run_binds_localhost_and_answers_health() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("crates/apr-format/tests/fixtures/golden_v2.apr");
    assert!(
        fixture.is_file(),
        "fixture missing: {} — this test is hermetic and must not download a model",
        fixture.display()
    );

    let port = free_port();
    let mut server = spawn_server(port, &fixture);
    let resp = wait_for_health(&mut server, port);

    let (head, body) = resp
        .split_once("\r\n\r\n")
        .unwrap_or_else(|| panic!("GET /health returned no HTTP header/body split:\n{resp}"));
    assert!(
        head.starts_with("HTTP/1.1 200"),
        "GET /health did not return 200; status line was {:?}\n--- server log ---\n{}",
        head.lines().next().unwrap_or(""),
        server.log_tail()
    );
    assert!(
        body.contains("\"status\":\"healthy\""),
        "GET /health returned 200 but not a healthy body: {body:?}"
    );

    // The root route is the second thing this binary binds, and asserting a
    // SECOND route is what distinguishes "a socket is open" from "the HTTP
    // transport is serving": anything at all can accept a connection.
    let root = http_get(port, "/").unwrap_or_else(|e| panic!("GET / failed: {e}"));
    assert!(
        root.starts_with("HTTP/1.1 200"),
        "GET / did not return 200:\n{root}"
    );

    eprintln!("http: port={port} /health body={}", body.trim());

    // Explicit kill, so the assertion that the transport shut down is part of
    // the test rather than only a `Drop` side effect.
    server.child.kill().expect("kill the serve child");
    let status = server.child.wait().expect("reap the serve child");
    eprintln!("http: server exited with {status}");
}
