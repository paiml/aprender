//! FALSIFY-PROBAR-EXEC-001: playbook actions must reach a real browser.
//!
//! `ActionExecutor` is the trait playbooks execute through, and it had **no
//! production implementation at all**. The only two `impl ActionExecutor` in the
//! workspace were `MockExecutor`, both inside `#[cfg(test)]` modules. A playbook
//! could be authored, parsed, validated and "run" without anything reaching a
//! browser — the same shape as the `ProbarDriver`/`MockDriver` gap in #2473, one
//! layer up.
//!
//! Every assertion here is chosen so a mock executor fails it. Requires
//! Chrome/Chromium; fails rather than skips without one.

#![cfg(feature = "browser")]

use jugar_probar::playbook::chromium_executor::ChromiumExecutor;
use jugar_probar::playbook::executor::{ActionExecutor, ExecutorError};
use jugar_probar::playbook::schema::WaitCondition;
use jugar_probar::DriverConfig;

const PAGE: &str = "data:text/html,<html><body>\
<h1 id='title' data-kind='heading'>Probar</h1>\
<button id='go' onclick=\"document.getElementById('out').textContent='clicked'\">Go</button>\
<span id='out'></span>\
<div id='gone' style='display:none'>hidden</div>\
</body></html>";

fn executor() -> ChromiumExecutor {
    let dir = std::env::temp_dir().join(format!("probar-exec-{}", std::process::id()));
    ChromiumExecutor::launch(
        DriverConfig {
            headless: true,
            element_timeout: std::time::Duration::from_secs(3),
            ..DriverConfig::default()
        },
        dir,
    )
    .unwrap_or_else(|e| panic!("could not launch a browser-backed executor: {e}"))
}

#[test]
fn a_playbook_click_changes_the_real_page() {
    let mut x = executor();
    x.navigate(PAGE).expect("navigate");

    let before = x.get_text("#out").expect("read #out");
    assert_eq!(before, "", "the output span starts empty");

    x.click("#go").expect("click");

    // The button's own onclick wrote this. A mock executor that records the
    // click without dispatching it cannot produce it.
    let after = x.get_text("#out").expect("read #out");
    assert_eq!(
        after, "clicked",
        "the click did not reach the button's handler"
    );
}

#[test]
fn text_and_attributes_come_from_the_dom() {
    let mut x = executor();
    x.navigate(PAGE).expect("navigate");

    assert_eq!(x.get_text("#title").expect("text"), "Probar");
    assert_eq!(
        x.get_attribute("#title", "data-kind").expect("attr"),
        "heading"
    );
    assert!(x.element_exists("#go").expect("exists"));

    // Excludes the outcome where every query succeeds: a selector matching
    // nothing must be reported as missing, not as empty text.
    assert!(!x.element_exists("#nope").expect("exists"));
    assert!(
        matches!(
            x.get_text("#nope"),
            Err(ExecutorError::ElementNotFound { .. })
        ),
        "a missing element returned text instead of ElementNotFound"
    );
}

#[test]
fn evaluate_is_decided_by_the_page() {
    let mut x = executor();
    x.navigate(PAGE).expect("navigate");

    assert!(x.evaluate("1 + 1 === 2").expect("evaluate"));
    // ...and it must be able to say NO. An executor hardcoded to true passes
    // the line above and fails this one.
    assert!(!x.evaluate("1 + 1 === 3").expect("evaluate"));
    // A DOM-dependent expression, so this is the live document and not a bare
    // JS sandbox.
    assert!(x
        .evaluate("document.getElementById('title').textContent === 'Probar'")
        .expect("evaluate"));
}

#[test]
fn wait_conditions_observe_real_state() {
    let mut x = executor();
    x.navigate(PAGE).expect("navigate");

    // Already-hidden element: satisfied immediately.
    x.wait(&WaitCondition::Hidden {
        selector: "#gone".to_string(),
    })
    .expect("#gone is display:none");

    // An element appended later: proves waiting rather than an immediate hit.
    x.execute_script(
        "setTimeout(() => { const d = document.createElement('div'); \
         d.id = 'later'; document.body.appendChild(d); }, 300); true",
    )
    .expect("schedule");
    x.wait(&WaitCondition::Visible {
        selector: "#later".to_string(),
    })
    .map_or_else(|e| panic!("#later never became visible: {e}"), |()| ());

    // Excludes the outcome where wait always succeeds.
    // element_timeout is honoured, so this costs its configured budget rather
    // than a hardcoded 30s.
    let err = x.wait(&WaitCondition::Condition {
        expression: "false".to_string(),
    });
    assert!(
        matches!(err, Err(ExecutorError::Timeout)),
        "a condition that is never true did not time out"
    );
}

#[test]
fn screenshots_are_written_as_real_files() {
    let dir = std::env::temp_dir().join(format!("probar-shot-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut x = ChromiumExecutor::launch(
        DriverConfig {
            headless: true,
            ..DriverConfig::default()
        },
        &dir,
    )
    .expect("launch");
    x.navigate(PAGE).expect("navigate");

    x.screenshot("step-1").expect("screenshot");

    let path = dir.join("step-1.png");
    let bytes = std::fs::read(&path).expect("the screenshot file must exist");
    assert_eq!(
        &bytes[..8],
        b"\x89PNG\r\n\x1a\n",
        "written file is not a PNG"
    );
    assert!(
        bytes.len() > 1000,
        "PNG is {} bytes, too small",
        bytes.len()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Launching from inside an async context must be refused with a clear error
/// rather than panicking inside tokio several frames down.
#[tokio::test]
async fn launching_inside_a_runtime_is_refused_not_a_panic() {
    let result = ChromiumExecutor::launch(DriverConfig::default(), std::env::temp_dir());
    match result {
        Err(ExecutorError::ScriptError { message }) => {
            assert!(
                message.contains("async context"),
                "wrong error for a nested-runtime launch: {message}"
            );
        }
        Err(other) => panic!("expected a ScriptError about async context, got {other:?}"),
        Ok(_) => panic!("launching inside a runtime succeeded; it would deadlock"),
    }
}
