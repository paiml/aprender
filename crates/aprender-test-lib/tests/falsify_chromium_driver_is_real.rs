//! FALSIFY-PROBAR-DRIVER-001: `ChromiumDriver` must drive a real browser.
//!
//! Issue #2473 established that `ProbarDriver` had exactly one implementation,
//! `MockDriver`, so every layer built on the trait drove a mock. These tests
//! exist to make that condition detectable rather than arguable.
//!
//! Each assertion below is chosen so that **`MockDriver` fails it**. That is the
//! whole design constraint: a test that merely calls the trait would pass
//! identically against the mock and prove nothing. Where a value could be echoed
//! back from configuration, the test asserts something only a live JS engine and
//! renderer can produce — a computed sum, a laid-out bounding box, PNG bytes
//! whose header and dimensions come from the compositor.
//!
//! These require Chrome/Chromium on PATH. They are NOT on ci.yml's beat list,
//! because the clean-room image is not known to ship a browser; they belong on a
//! Chrome-equipped runner the way the GPU falsifiers belong on a CUDA one.
//! Running them without a browser FAILS — deliberately. There is no skip.

#![cfg(feature = "browser")]

use std::time::Duration;

use jugar_probar::{ChromiumDriver, DriverConfig, ProbarDriver};

/// A page with content whose layout and arithmetic the test can predict.
const PAGE: &str = "data:text/html,<html><body style='margin:0'>\
<div id='a' style='width:120px;height:40px'>alpha</div>\
<p class='item'>one</p><p class='item'>two</p><p class='item'>three</p>\
<input id='box' value=''>\
</body></html>";

fn config() -> DriverConfig {
    DriverConfig {
        headless: true,
        viewport_width: 800,
        viewport_height: 600,
        ..DriverConfig::default()
    }
}

async fn driver() -> ChromiumDriver {
    ChromiumDriver::launch(config()).await.unwrap_or_else(|e| {
        panic!(
            "could not launch a real browser: {e}\n\
             These tests assert probar drives Chrome. Install Chrome/Chromium \
             rather than making this test skip -- a skipped browser test is how \
             #2473 happened."
        )
    })
}

#[tokio::test]
async fn js_is_evaluated_by_a_real_engine() {
    let mut d = driver().await;
    d.navigate(PAGE).await.expect("navigate");

    // MockDriver returns a canned value here and cannot compute this.
    let sum = d
        .execute_js("[1,2,3,4,5,6,7,8].reduce((a,b)=>a+b,0)")
        .await
        .expect("execute_js");
    assert_eq!(
        sum.as_i64(),
        Some(36),
        "a real JS engine must fold this to 36"
    );

    // ...and the engine must be Chrome specifically, not any evaluator.
    let ua = d
        .execute_js("navigator.userAgent")
        .await
        .expect("execute_js");
    let ua = ua.as_str().unwrap_or_default();
    assert!(
        ua.contains("Chrome") || ua.contains("Chromium"),
        "user agent {ua:?} is not a Chromium browser"
    );

    // Non-vacuity: the two assertions above must be capable of disagreeing.
    // If execute_js returned one constant for every script they could not.
    let other = d.execute_js("'probar' + 1").await.expect("execute_js");
    assert_eq!(other.as_str(), Some("probar1"));
    assert_ne!(
        other.as_str().map(str::to_string),
        Some(ua.to_string()),
        "execute_js returns the same value regardless of script"
    );

    d.close().await.expect("close");
}

#[tokio::test]
async fn the_dom_is_queried_and_laid_out() {
    let mut d = driver().await;
    d.navigate(PAGE).await.expect("navigate");

    let el = d
        .query_selector("#a")
        .await
        .expect("query_selector")
        .expect("#a exists");
    assert_eq!(el.tag_name, "div");
    assert_eq!(el.text_content.as_deref(), Some("alpha"));

    // Layout is the part a mock cannot fake: these numbers come from Blink.
    let bb = el.bounding_box.expect("#a is visible so it has a box");
    assert!(
        (bb.width - 120.0).abs() < 1.0 && (bb.height - 40.0).abs() < 1.0,
        "expected the CSS 120x40, got {}x{} -- the box did not come from layout",
        bb.width,
        bb.height
    );

    let items = d.query_selector_all("p.item").await.expect("all");
    assert_eq!(items.len(), 3, "querySelectorAll must see all three <p>");
    let texts: Vec<_> = items
        .iter()
        .filter_map(|e| e.text_content.as_deref())
        .collect();
    assert_eq!(texts, vec!["one", "two", "three"]);

    // Excludes the outcome where every query returns the same canned element.
    let missing = d.query_selector("#does-not-exist").await.expect("query");
    assert!(
        missing.is_none(),
        "a selector matching nothing returned an element, so matches are fabricated"
    );

    d.close().await.expect("close");
}

#[tokio::test]
async fn typing_changes_real_dom_state() {
    let mut d = driver().await;
    d.navigate(PAGE).await.expect("navigate");

    let before = d
        .execute_js("document.getElementById('box').value")
        .await
        .expect("read");
    assert_eq!(before.as_str(), Some(""), "input starts empty");

    d.type_text("#box", "hola").await.expect("type_text");

    let after = d
        .execute_js("document.getElementById('box').value")
        .await
        .expect("read");
    assert_eq!(
        after.as_str(),
        Some("hola"),
        "typing did not reach the real input element"
    );
}

#[tokio::test]
async fn a_screenshot_is_real_png_pixels() {
    let mut d = driver().await;
    d.navigate(PAGE).await.expect("navigate");

    let shot = d.screenshot().await.expect("screenshot");

    // MockDriver errors here unless a canned image was set; a real one returns
    // PNG bytes from the compositor.
    assert_eq!(
        &shot.data[..8],
        b"\x89PNG\r\n\x1a\n",
        "not a PNG: the screenshot did not come from the renderer"
    );
    assert!(
        shot.data.len() > 1000,
        "PNG is {} bytes, too small to be a rendered 800x600 page",
        shot.data.len()
    );
    // Dimensions are read back from the page, not echoed from config, so a
    // browser that ignored the viewport would show up here.
    assert_eq!(
        (shot.width, shot.height),
        (800, 600),
        "reported viewport does not match the one the browser actually used"
    );

    d.close().await.expect("close");
}

#[tokio::test]
async fn navigation_and_metrics_reflect_the_live_page() {
    let mut d = driver().await;
    d.navigate(PAGE).await.expect("navigate");

    let url = d.current_url().await.expect("current_url");
    assert!(
        url.starts_with("data:text/html"),
        "current_url is {url:?}, not the page we navigated to"
    );

    let m = d.metrics().await.expect("metrics");
    // DOM node count is computed from the live document. The page has
    // html, body, div, 3x p, input plus head -- comfortably more than 5.
    let nodes = m.dom_nodes.expect("dom_nodes must be measured");
    assert!(
        nodes >= 6,
        "only {nodes} DOM nodes counted; the document was not walked"
    );
    assert!(
        m.dom_content_loaded_ms.is_some(),
        "no DOMContentLoaded timing: the Navigation Timing API was not read"
    );

    d.close().await.expect("close");
}

#[tokio::test]
async fn wait_for_selector_waits_and_then_times_out() {
    let mut d = driver().await;
    d.navigate(PAGE).await.expect("navigate");

    // An element that appears late: proves waiting, not just an immediate hit.
    d.execute_js(
        "setTimeout(() => { const s = document.createElement('span'); \
         s.id = 'late'; s.textContent = 'here'; document.body.appendChild(s); }, 300)",
    )
    .await
    .expect("schedule");

    let late = d
        .wait_for_selector("#late", Duration::from_secs(5))
        .await
        .expect("#late should appear within 5s");
    assert_eq!(late.text_content.as_deref(), Some("here"));

    // Excludes the outcome where wait_for_selector returns a handle for
    // anything: an element that never appears must TIME OUT.
    let err = d
        .wait_for_selector("#never", Duration::from_millis(400))
        .await
        .expect_err("a selector that never matches must time out");
    assert!(
        matches!(err, jugar_probar::ProbarError::Timeout { .. }),
        "expected Timeout, got {err:?}"
    );

    d.close().await.expect("close");
}

/// The anti-mock guard, and the only test here that needs no browser: a driver
/// pointed at a nonexistent executable must FAIL, not quietly hand back
/// something that answers questions it cannot know.
#[tokio::test]
async fn a_missing_browser_is_an_error_not_a_silent_mock() {
    let cfg = DriverConfig {
        headless: true,
        executable_path: Some("/nonexistent/definitely-not-chrome".to_string()),
        ..DriverConfig::default()
    };
    let result = ChromiumDriver::launch(cfg).await;
    assert!(
        result.is_err(),
        "launching with a bogus executable succeeded -- the driver fell back to \
         something that is not the browser it claims to be"
    );
}
