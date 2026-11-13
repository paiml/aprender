//! End-to-end (probar) test of `aprender-present bundle`: build the wasm bundle, serve it from
//! a scratch directory, load it in a real headless Chrome over CDP, drive every control, and
//! read the results back from the page. A curl of the HTML is not a receipt.
//!
//! Independent of page layout: it relies only on the documented ids `#canvas` and `#status`
//! (status text starts with "Ready" once the wasm app is up), and drives every interactive
//! element it finds (`button`, `input`, `select`, `a[href]`, `[data-control]`) by enumeration.
//!
//! Gated on `APR_PROBAR_CHROMIUM` (path to a Chrome/Chromium executable); the test needs
//! `wasm-pack` + the wasm32 target, so it never runs in the default unit-test lane. Set
//! `APR_PROBAR_REQUIRE=1` to make a missing prerequisite a failure instead of a skip.

use probar::{Browser, BrowserConfig};
use std::path::{Path, PathBuf};
use std::process::Command;

const CONTROLS: &str = "button, input, select, a[href], [data-control]";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root")
}

fn content_type(p: &Path) -> &'static str {
    match p.extension().and_then(|e| e.to_str()) {
        Some("html") => "text/html",
        Some("js") => "text/javascript",
        Some("wasm") => "application/wasm",
        Some("json") => "application/json",
        Some("css") => "text/css",
        _ => "application/octet-stream",
    }
}

/// Serve `dir` on an ephemeral 127.0.0.1 port; returns the port.
fn serve(dir: PathBuf) -> u16 {
    let server = tiny_http::Server::http("127.0.0.1:0").expect("bind");
    let port = server.server_addr().to_ip().expect("ip").port();
    std::thread::spawn(move || {
        for req in server.incoming_requests() {
            let rel = req
                .url()
                .split('?')
                .next()
                .unwrap_or("/")
                .trim_start_matches('/')
                .to_string();
            let rel = if rel.is_empty() {
                "index.html".to_string()
            } else {
                rel
            };
            let path = dir.join(&rel);
            if !rel.contains("..") && path.is_file() {
                let ct =
                    tiny_http::Header::from_bytes("Content-Type", content_type(&path)).unwrap();
                let r = tiny_http::Response::from_data(std::fs::read(&path).unwrap());
                let _ = req.respond(r.with_header(ct));
            } else {
                let _ = req
                    .respond(tiny_http::Response::from_string("not found").with_status_code(404));
            }
        }
    });
    port
}

async fn eval<T: serde::de::DeserializeOwned>(page: &probar::Page, js: &str) -> T {
    page.evaluate(js)
        .await
        .unwrap_or_else(|e| panic!("evaluate {js:?}: {e}"))
        .into_value()
        .unwrap_or_else(|e| panic!("decode {js:?}: {e}"))
}

#[test]
fn bundle_page_every_control_driven_in_real_chrome() {
    let Ok(chrome) = std::env::var("APR_PROBAR_CHROMIUM") else {
        assert!(
            std::env::var("APR_PROBAR_REQUIRE").is_err(),
            "APR_PROBAR_CHROMIUM unset"
        );
        eprintln!("SKIP: APR_PROBAR_CHROMIUM not set");
        return;
    };
    let root = repo_root();
    // `bundle --output` is resolved relative to the repo root (wasm-pack gets `../../<output>`),
    // so an absolute path lands in the wrong place; use a relative scratch dir under target/.
    let rel_out = PathBuf::from(format!("target/bundle-e2e-{}", std::process::id()));
    let out = root.join(&rel_out);
    let _ = std::fs::remove_dir_all(&out);

    // 1. Build the bundle with the shipped verb (no wasm-opt: size is not under test).
    let st = Command::new(env!("CARGO_BIN_EXE_aprender-present"))
        .args(["bundle", "--no-optimize", "--output"])
        .arg(&rel_out)
        .current_dir(&root)
        .status()
        .expect("run bundle");
    assert!(st.success(), "aprender-present bundle failed");
    assert!(out.join("pkg/presentar_bg.wasm").is_file(), "wasm missing");
    assert!(out.join("pkg/presentar.js").is_file(), "js glue missing");

    // If the verb did not copy a page (cwd-relative www/), use the repo's page and say so, so
    // the gap is visible rather than silently masked.
    if !out.join("index.html").is_file() {
        eprintln!(
            "NOTE: bundle emitted no index.html; using crates/aprender-present/www/index.html"
        );
        std::fs::copy(
            root.join("crates/aprender-present/www/index.html"),
            out.join("index.html"),
        )
        .expect("copy page");
    }
    let port = serve(out.clone());

    // 2. Real Chrome over CDP. (probar's `headless: false` selects chromiumoxide's headless
    //    default; `true` calls with_head(), i.e. a headed window.)
    let rt = tokio::runtime::Runtime::new().expect("rt");
    rt.block_on(async {
        let cfg = BrowserConfig {
            headless: false,
            sandbox: false,
            ..BrowserConfig::default()
        }
        .with_chromium_path(chrome);
        let browser = Browser::launch(cfg).await.expect("launch chrome");
        let mut page = browser.new_page().await.expect("new page");
        page.goto(&format!("http://127.0.0.1:{port}/index.html"))
            .await
            .expect("goto");

        // 3. Wait for the wasm app: #status starts with "Ready".
        let mut status = String::new();
        for _ in 0..150 {
            if let Ok(r) = page
                .evaluate("String((document.getElementById('status')||{}).textContent)")
                .await
            {
                status = r.into_value::<String>().unwrap_or_default();
            }
            if status.starts_with("Ready") {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        }
        assert!(status.starts_with("Ready"), "wasm never ready; #status = {status:?}");

        // 4. The wasm renderer actually painted the canvas.
        let painted: bool = eval(
            &page,
            "(() => { const c = document.getElementById('canvas'); \
             const d = c.getContext('2d').getImageData(0,0,c.width,c.height).data; \
             for (let i = 3; i < d.length; i += 4) { if (d[i] !== 0) return true; } return false; })()",
        )
        .await;
        assert!(painted, "canvas is blank after wasm init");

        // 5. Drive every control; trap page errors raised meanwhile.
        let _: bool = eval(
            &page,
            "window.__errs = []; \
             window.addEventListener('error', e => window.__errs.push(String(e.message))); \
             window.addEventListener('unhandledrejection', e => window.__errs.push(String(e.reason))); true",
        )
        .await;
        let n: u64 = eval(
            &page,
            &format!("document.querySelectorAll('{CONTROLS}').length"),
        )
        .await;
        for i in 0..n {
            let step = format!(
                "(() => {{ const el = document.querySelectorAll('{CONTROLS}')[{i}]; \
                 if (el.tagName === 'A') {{ el.addEventListener('click', e => e.preventDefault(), {{once: true}}); }} \
                 if (el.tagName === 'INPUT' && (el.type === 'checkbox' || el.type === 'radio')) {{ el.click(); }} \
                 else if (el.tagName === 'INPUT') {{ el.focus(); \
                   el.value = (el.type === 'number' || el.type === 'range') ? (el.min || '1') : 'probe'; \
                   el.dispatchEvent(new Event('input', {{bubbles: true}})); \
                   el.dispatchEvent(new Event('change', {{bubbles: true}})); }} \
                 else if (el.tagName === 'SELECT') {{ if (el.options.length) {{ \
                   el.selectedIndex = el.options.length - 1; \
                   el.dispatchEvent(new Event('change', {{bubbles: true}})); }} }} \
                 else {{ el.click(); }} return true; }})()"
            );
            let _: bool = eval(&page, &step).await;
        }
        // The canvas is a control too: click it through CDP's real input path.
        page.click("#canvas").await.expect("click #canvas");

        // 6. Read back: no page errors, still ready, still painted.
        let errs: Vec<String> = eval(&page, "window.__errs").await;
        assert!(errs.is_empty(), "page errors driving {n} controls: {errs:?}");
        let after: String = eval(&page, "document.getElementById('status').textContent").await;
        assert!(after.starts_with("Ready"), "status regressed: {after:?}");
        eprintln!("bundle e2e: status={after:?} controls_driven={n} canvas_click=ok");

        let _ = browser.close().await;
    });
    let _ = std::fs::remove_dir_all(&out);
}
