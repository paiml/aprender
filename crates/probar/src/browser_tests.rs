    use super::*;

    mod browser_config_tests {
        use super::*;

        #[test]
        fn test_default() {
            let config = BrowserConfig::default();
            assert!(config.headless);
            assert_eq!(config.viewport_width, 800);
            assert_eq!(config.viewport_height, 600);
            assert!(config.chromium_path.is_none());
            assert_eq!(config.debug_port, 0);
            assert!(config.user_agent.is_none());
            assert!(!config.devtools);
            assert!(config.sandbox);
        }

        #[test]
        fn test_with_viewport() {
            let config = BrowserConfig::default().with_viewport(1920, 1080);
            assert_eq!(config.viewport_width, 1920);
            assert_eq!(config.viewport_height, 1080);
        }

        #[test]
        fn test_with_headless() {
            let config = BrowserConfig::default().with_headless(false);
            assert!(!config.headless);
        }

        #[test]
        fn test_with_chromium_path() {
            let config = BrowserConfig::default().with_chromium_path("/usr/bin/chromium");
            assert_eq!(config.chromium_path, Some("/usr/bin/chromium".to_string()));
        }

        #[test]
        fn test_with_user_agent() {
            let config = BrowserConfig::default().with_user_agent("Custom UA");
            assert_eq!(config.user_agent, Some("Custom UA".to_string()));
        }

        #[test]
        fn test_with_no_sandbox() {
            let config = BrowserConfig::default().with_no_sandbox();
            assert!(!config.sandbox);
        }

        #[test]
        fn test_clone() {
            let config = BrowserConfig::default()
                .with_viewport(1024, 768)
                .with_headless(false);
            let cloned = config.clone();
            assert_eq!(config.viewport_width, cloned.viewport_width);
            assert_eq!(config.headless, cloned.headless);
        }

        #[test]
        fn test_debug() {
            let config = BrowserConfig::default();
            let debug = format!("{:?}", config);
            assert!(debug.contains("BrowserConfig"));
            assert!(debug.contains("headless"));
        }
    }

    #[cfg(not(feature = "browser"))]
    mod mock_browser_tests {
        use super::*;

        #[test]
        fn test_browser_launch() {
            let config = BrowserConfig::default();
            let browser = Browser::launch(config).unwrap();
            assert_eq!(browser.config().viewport_width, 800);
        }

        #[test]
        fn test_browser_new_page() {
            let config = BrowserConfig::default().with_viewport(1024, 768);
            let browser = Browser::launch(config).unwrap();
            let page = browser.new_page().unwrap();
            assert_eq!(page.width, 1024);
            assert_eq!(page.height, 768);
        }

        #[test]
        fn test_browser_debug() {
            let config = BrowserConfig::default();
            let browser = Browser::launch(config).unwrap();
            let debug = format!("{:?}", browser);
            assert!(debug.contains("Browser"));
        }
    }

    #[cfg(not(feature = "browser"))]
    mod mock_page_tests {
        use super::*;

        #[test]
        fn test_page_new() {
            let page = Page::new(800, 600);
            assert_eq!(page.width, 800);
            assert_eq!(page.height, 600);
            assert_eq!(page.url, "about:blank");
            assert!(!page.wasm_ready);
        }

        #[test]
        fn test_page_goto() {
            let mut page = Page::new(800, 600);
            page.goto("https://example.com").unwrap();
            assert_eq!(page.current_url(), "https://example.com");
        }

        #[test]
        fn test_page_wait_for_wasm_ready() {
            let mut page = Page::new(800, 600);
            assert!(!page.is_wasm_ready());
            page.wait_for_wasm_ready().unwrap();
            assert!(page.is_wasm_ready());
        }

        #[test]
        fn test_page_eval_wasm_error() {
            let page = Page::new(800, 600);
            let result: Result<String, _> = page.eval_wasm("test");
            assert!(result.is_err());
        }

        #[test]
        fn test_page_touch() {
            let page = Page::new(800, 600);
            let touch = crate::Touch {
                x: 100.0,
                y: 100.0,
                action: crate::TouchAction::Tap,
            };
            page.touch(touch).unwrap();
        }

        #[test]
        fn test_page_screenshot() {
            let page = Page::new(800, 600);
            let screenshot = page.screenshot().unwrap();
            assert!(screenshot.is_empty()); // Mock returns empty
        }

        #[test]
        fn test_page_debug() {
            let page = Page::new(800, 600);
            let debug = format!("{:?}", page);
            assert!(debug.contains("Page"));
        }
    }

    // =========================================================================
    // H₀ EXTREME TDD: Browser Tests (Feature F P0)
    // =========================================================================

    mod h0_browser_config_tests {
        use super::*;

        #[test]
        fn h0_browser_01_config_default_headless() {
            let config = BrowserConfig::default();
            assert!(config.headless);
        }

        #[test]
        fn h0_browser_02_config_default_viewport_width() {
            let config = BrowserConfig::default();
            assert_eq!(config.viewport_width, 800);
        }

        #[test]
        fn h0_browser_03_config_default_viewport_height() {
            let config = BrowserConfig::default();
            assert_eq!(config.viewport_height, 600);
        }

        #[test]
        fn h0_browser_04_config_default_no_chromium_path() {
            let config = BrowserConfig::default();
            assert!(config.chromium_path.is_none());
        }

        #[test]
        fn h0_browser_05_config_default_debug_port() {
            let config = BrowserConfig::default();
            assert_eq!(config.debug_port, 0);
        }

        #[test]
        fn h0_browser_06_config_default_no_user_agent() {
            let config = BrowserConfig::default();
            assert!(config.user_agent.is_none());
        }

        #[test]
        fn h0_browser_07_config_default_devtools_off() {
            let config = BrowserConfig::default();
            assert!(!config.devtools);
        }

        #[test]
        fn h0_browser_08_config_default_sandbox_on() {
            let config = BrowserConfig::default();
            assert!(config.sandbox);
        }

        #[test]
        fn h0_browser_09_config_with_viewport() {
            let config = BrowserConfig::default().with_viewport(1920, 1080);
            assert_eq!(config.viewport_width, 1920);
            assert_eq!(config.viewport_height, 1080);
        }

        #[test]
        fn h0_browser_10_config_with_headless_false() {
            let config = BrowserConfig::default().with_headless(false);
            assert!(!config.headless);
        }
    }

    mod h0_browser_config_builder_tests {
        use super::*;

        #[test]
        fn h0_browser_11_config_with_chromium_path() {
            let config = BrowserConfig::default().with_chromium_path("/path/to/chromium");
            assert_eq!(config.chromium_path, Some("/path/to/chromium".to_string()));
        }

        #[test]
        fn h0_browser_12_config_with_user_agent() {
            let config = BrowserConfig::default().with_user_agent("Test UA");
            assert_eq!(config.user_agent, Some("Test UA".to_string()));
        }

        #[test]
        fn h0_browser_13_config_with_no_sandbox() {
            let config = BrowserConfig::default().with_no_sandbox();
            assert!(!config.sandbox);
        }

        #[test]
        fn h0_browser_14_config_builder_chain() {
            let config = BrowserConfig::default()
                .with_viewport(1024, 768)
                .with_headless(false)
                .with_no_sandbox()
                .with_user_agent("Custom");
            assert_eq!(config.viewport_width, 1024);
            assert!(!config.headless);
            assert!(!config.sandbox);
            assert_eq!(config.user_agent, Some("Custom".to_string()));
        }

        #[test]
        fn h0_browser_15_config_clone() {
            let config = BrowserConfig::default().with_viewport(800, 600);
            let cloned = config;
            assert_eq!(cloned.viewport_width, 800);
        }

        #[test]
        fn h0_browser_16_config_string_conversion() {
            let config =
                BrowserConfig::default().with_chromium_path(String::from("/usr/bin/chrome"));
            assert!(config.chromium_path.is_some());
        }

        #[test]
        fn h0_browser_17_config_small_viewport() {
            let config = BrowserConfig::default().with_viewport(320, 240);
            assert_eq!(config.viewport_width, 320);
            assert_eq!(config.viewport_height, 240);
        }

        #[test]
        fn h0_browser_18_config_large_viewport() {
            let config = BrowserConfig::default().with_viewport(3840, 2160);
            assert_eq!(config.viewport_width, 3840);
        }

        #[test]
        fn h0_browser_19_config_debug_format() {
            let config = BrowserConfig::default();
            let debug = format!("{:?}", config);
            assert!(debug.contains("headless"));
        }

        #[test]
        fn h0_browser_20_config_user_agent_unicode() {
            let config = BrowserConfig::default().with_user_agent("UA/テスト");
            assert_eq!(config.user_agent, Some("UA/テスト".to_string()));
        }
    }

    #[cfg(not(feature = "browser"))]
    mod h0_mock_browser_tests {
        use super::*;

        #[test]
        fn h0_browser_21_launch() {
            let config = BrowserConfig::default();
            let browser = Browser::launch(config);
            assert!(browser.is_ok());
        }

        #[test]
        fn h0_browser_22_launch_config_preserved() {
            let config = BrowserConfig::default().with_viewport(1024, 768);
            let browser = Browser::launch(config).unwrap();
            assert_eq!(browser.config().viewport_width, 1024);
        }

        #[test]
        fn h0_browser_23_new_page() {
            let browser = Browser::launch(BrowserConfig::default()).unwrap();
            let page = browser.new_page();
            assert!(page.is_ok());
        }

        #[test]
        fn h0_browser_24_new_page_dimensions() {
            let config = BrowserConfig::default().with_viewport(1280, 720);
            let browser = Browser::launch(config).unwrap();
            let page = browser.new_page().unwrap();
            assert_eq!(page.width, 1280);
            assert_eq!(page.height, 720);
        }

        #[test]
        fn h0_browser_25_debug_format() {
            let browser = Browser::launch(BrowserConfig::default()).unwrap();
            let debug = format!("{:?}", browser);
            assert!(debug.contains("Browser"));
        }
    }

    #[cfg(not(feature = "browser"))]
    mod h0_mock_page_tests {
        use super::*;

        #[test]
        fn h0_browser_26_page_new() {
            let page = Page::new(800, 600);
            assert_eq!(page.width, 800);
        }

        #[test]
        fn h0_browser_27_page_initial_url() {
            let page = Page::new(800, 600);
            assert_eq!(page.url, "about:blank");
        }

        #[test]
        fn h0_browser_28_page_initial_wasm_not_ready() {
            let page = Page::new(800, 600);
            assert!(!page.wasm_ready);
        }

        #[test]
        fn h0_browser_29_page_goto() {
            let mut page = Page::new(800, 600);
            let result = page.goto("http://localhost:8080");
            assert!(result.is_ok());
        }

        #[test]
        fn h0_browser_30_page_goto_updates_url() {
            let mut page = Page::new(800, 600);
            page.goto("http://test.com").unwrap();
            assert_eq!(page.current_url(), "http://test.com");
        }

        #[test]
        fn h0_browser_31_page_wait_for_wasm() {
            let mut page = Page::new(800, 600);
            let result = page.wait_for_wasm_ready();
            assert!(result.is_ok());
        }

        #[test]
        fn h0_browser_32_page_wasm_ready_after_wait() {
            let mut page = Page::new(800, 600);
            page.wait_for_wasm_ready().unwrap();
            assert!(page.is_wasm_ready());
        }

        #[test]
        fn h0_browser_33_page_eval_wasm_fails() {
            let page = Page::new(800, 600);
            let result: Result<i32, _> = page.eval_wasm("1 + 1");
            assert!(result.is_err());
        }

        #[test]
        fn h0_browser_34_page_touch_tap() {
            let page = Page::new(800, 600);
            let touch = crate::Touch {
                x: 50.0,
                y: 50.0,
                action: crate::TouchAction::Tap,
            };
            assert!(page.touch(touch).is_ok());
        }

        #[test]
        fn h0_browser_35_page_screenshot_empty() {
            let page = Page::new(800, 600);
            let screenshot = page.screenshot().unwrap();
            assert!(screenshot.is_empty());
        }
    }

    #[cfg(not(feature = "browser"))]
    mod h0_mock_page_advanced_tests {
        use super::*;

        #[test]
        fn h0_browser_36_page_touch_swipe() {
            let page = Page::new(800, 600);
            let touch = crate::Touch {
                x: 100.0,
                y: 100.0,
                action: crate::TouchAction::Swipe {
                    end_x: 200.0,
                    end_y: 200.0,
                    duration_ms: 100,
                },
            };
            assert!(page.touch(touch).is_ok());
        }

        #[test]
        fn h0_browser_37_page_touch_hold() {
            let page = Page::new(800, 600);
            let touch = crate::Touch {
                x: 100.0,
                y: 100.0,
                action: crate::TouchAction::Hold { duration_ms: 500 },
            };
            assert!(page.touch(touch).is_ok());
        }

        #[test]
        fn h0_browser_38_page_debug() {
            let page = Page::new(800, 600);
            let debug = format!("{:?}", page);
            assert!(debug.contains("Page"));
        }

        #[test]
        fn h0_browser_39_page_current_url_method() {
            let page = Page::new(800, 600);
            assert_eq!(page.current_url(), "about:blank");
        }

        #[test]
        fn h0_browser_40_page_is_wasm_ready_method() {
            let page = Page::new(800, 600);
            assert!(!page.is_wasm_ready());
        }

        #[test]
        fn h0_browser_41_page_multiple_goto() {
            let mut page = Page::new(800, 600);
            page.goto("http://first.com").unwrap();
            page.goto("http://second.com").unwrap();
            assert_eq!(page.current_url(), "http://second.com");
        }

        #[test]
        fn h0_browser_42_page_zero_dimensions() {
            let page = Page::new(0, 0);
            assert_eq!(page.width, 0);
            assert_eq!(page.height, 0);
        }

        #[test]
        fn h0_browser_43_page_large_dimensions() {
            let page = Page::new(7680, 4320);
            assert_eq!(page.width, 7680);
        }

        #[test]
        fn h0_browser_44_config_overwrite_viewport() {
            let config = BrowserConfig::default()
                .with_viewport(800, 600)
                .with_viewport(1024, 768);
            assert_eq!(config.viewport_width, 1024);
        }

        #[test]
        fn h0_browser_45_config_overwrite_headless() {
            let config = BrowserConfig::default()
                .with_headless(false)
                .with_headless(true);
            assert!(config.headless);
        }
    }

    mod h0_browser_edge_cases {
        use super::*;

        #[test]
        fn h0_browser_46_config_empty_chromium_path() {
            let config = BrowserConfig::default().with_chromium_path("");
            assert_eq!(config.chromium_path, Some(String::new()));
        }

        #[test]
        fn h0_browser_47_config_empty_user_agent() {
            let config = BrowserConfig::default().with_user_agent("");
            assert_eq!(config.user_agent, Some(String::new()));
        }

        #[test]
        fn h0_browser_48_config_viewport_square() {
            let config = BrowserConfig::default().with_viewport(1000, 1000);
            assert_eq!(config.viewport_width, config.viewport_height);
        }

        #[test]
        fn h0_browser_49_config_viewport_portrait() {
            let config = BrowserConfig::default().with_viewport(600, 800);
            assert!(config.viewport_height > config.viewport_width);
        }

        #[test]
        fn h0_browser_50_config_viewport_landscape() {
            let config = BrowserConfig::default().with_viewport(1920, 1080);
            assert!(config.viewport_width > config.viewport_height);
        }
    }

    // =========================================================================
    // Console Capture Tests (Issue #8)
    // =========================================================================

    mod console_capture_tests {
        use super::*;

        #[test]
        fn test_browser_console_level_display() {
            assert_eq!(format!("{}", BrowserConsoleLevel::Log), "log");
            assert_eq!(format!("{}", BrowserConsoleLevel::Info), "info");
            assert_eq!(format!("{}", BrowserConsoleLevel::Warning), "warn");
            assert_eq!(format!("{}", BrowserConsoleLevel::Error), "error");
            assert_eq!(format!("{}", BrowserConsoleLevel::Debug), "debug");
        }

        #[test]
        fn test_browser_console_level_eq() {
            assert_eq!(BrowserConsoleLevel::Log, BrowserConsoleLevel::Log);
            assert_ne!(BrowserConsoleLevel::Log, BrowserConsoleLevel::Error);
        }

        #[test]
        fn test_browser_console_level_clone() {
            let level = BrowserConsoleLevel::Warning;
            let cloned = level;
            assert_eq!(level, cloned);
        }

        #[test]
        fn test_browser_console_level_debug() {
            let level = BrowserConsoleLevel::Error;
            let debug = format!("{:?}", level);
            assert!(debug.contains("Error"));
        }

        #[test]
        fn test_browser_console_message_create() {
            let msg = BrowserConsoleMessage {
                level: BrowserConsoleLevel::Log,
                text: "test message".to_string(),
                timestamp: 1234567890,
                source: Some("test.js".to_string()),
                line: Some(42),
            };
            assert_eq!(msg.level, BrowserConsoleLevel::Log);
            assert_eq!(msg.text, "test message");
            assert_eq!(msg.timestamp, 1234567890);
            assert_eq!(msg.source, Some("test.js".to_string()));
            assert_eq!(msg.line, Some(42));
        }

        #[test]
        fn test_browser_console_message_without_source() {
            let msg = BrowserConsoleMessage {
                level: BrowserConsoleLevel::Error,
                text: "error".to_string(),
                timestamp: 0,
                source: None,
                line: None,
            };
            assert!(msg.source.is_none());
            assert!(msg.line.is_none());
        }

        #[test]
        fn test_browser_console_message_clone() {
            let msg = BrowserConsoleMessage {
                level: BrowserConsoleLevel::Info,
                text: "info".to_string(),
                timestamp: 100,
                source: None,
                line: None,
            };
            let cloned = msg.clone();
            assert_eq!(msg.text, cloned.text);
            assert_eq!(msg.timestamp, cloned.timestamp);
        }

        #[test]
        fn test_browser_console_message_debug() {
            let msg = BrowserConsoleMessage {
                level: BrowserConsoleLevel::Debug,
                text: "debug msg".to_string(),
                timestamp: 0,
                source: None,
                line: None,
            };
            let debug = format!("{:?}", msg);
            assert!(debug.contains("BrowserConsoleMessage"));
            assert!(debug.contains("debug msg"));
        }
    }

    #[cfg(not(feature = "browser"))]
    mod mock_console_capture_tests {
        use super::*;

        #[test]
        fn test_page_enable_console_capture() {
            let mut page = Page::new(800, 600);
            assert!(!page.is_console_capture_enabled());
            page.enable_console_capture().unwrap();
            assert!(page.is_console_capture_enabled());
        }

        #[test]
        fn test_page_console_messages_empty() {
            let page = Page::new(800, 600);
            let messages = page.console_messages();
            assert!(messages.is_empty());
        }

        #[test]
        fn test_page_add_console_message() {
            let page = Page::new(800, 600);
            let msg = BrowserConsoleMessage {
                level: BrowserConsoleLevel::Log,
                text: "test".to_string(),
                timestamp: 123,
                source: None,
                line: None,
            };
            page.add_console_message(msg);
            let messages = page.console_messages();
            assert_eq!(messages.len(), 1);
            assert_eq!(messages[0].text, "test");
        }

        #[test]
        fn test_page_clear_console() {
            let page = Page::new(800, 600);
            page.add_console_message(BrowserConsoleMessage {
                level: BrowserConsoleLevel::Log,
                text: "msg".to_string(),
                timestamp: 0,
                source: None,
                line: None,
            });
            assert_eq!(page.console_messages().len(), 1);
            page.clear_console();
            assert!(page.console_messages().is_empty());
        }

        #[test]
        fn test_page_wait_for_console_found() {
            let page = Page::new(800, 600);
            page.add_console_message(BrowserConsoleMessage {
                level: BrowserConsoleLevel::Info,
                text: "ready".to_string(),
                timestamp: 100,
                source: None,
                line: None,
            });
            let result = page.wait_for_console(|m| m.text.contains("ready"), 1000);
            assert!(result.is_ok());
            assert_eq!(result.unwrap().text, "ready");
        }

        #[test]
        fn test_page_wait_for_console_not_found() {
            let page = Page::new(800, 600);
            let result = page.wait_for_console(|m| m.text.contains("missing"), 1000);
            assert!(result.is_err());
        }

        #[test]
        fn test_page_wait_for_console_by_level() {
            let page = Page::new(800, 600);
            page.add_console_message(BrowserConsoleMessage {
                level: BrowserConsoleLevel::Error,
                text: "error occurred".to_string(),
                timestamp: 0,
                source: None,
                line: None,
            });
            let result = page.wait_for_console(|m| m.level == BrowserConsoleLevel::Error, 1000);
            assert!(result.is_ok());
        }

        #[test]
        fn test_page_inject_console_capture() {
            let mut page = Page::new(800, 600);
            assert!(!page.is_console_capture_enabled());
            page.inject_console_capture().unwrap();
            assert!(page.is_console_capture_enabled());
        }

        #[test]
        fn test_page_fetch_console_messages() {
            let page = Page::new(800, 600);
            page.add_console_message(BrowserConsoleMessage {
                level: BrowserConsoleLevel::Warning,
                text: "warning".to_string(),
                timestamp: 0,
                source: None,
                line: None,
            });
            let result = page.fetch_console_messages();
            assert!(result.is_ok());
            let messages = result.unwrap();
            assert_eq!(messages.len(), 1);
            assert_eq!(messages[0].level, BrowserConsoleLevel::Warning);
        }

        #[test]
        fn test_page_multiple_console_messages() {
            let page = Page::new(800, 600);
            for i in 0..5 {
                page.add_console_message(BrowserConsoleMessage {
                    level: BrowserConsoleLevel::Log,
                    text: format!("message {i}"),
                    timestamp: i as u64,
                    source: None,
                    line: None,
                });
            }
            let messages = page.console_messages();
            assert_eq!(messages.len(), 5);
            assert_eq!(messages[0].text, "message 0");
            assert_eq!(messages[4].text, "message 4");
        }
    }

    // =========================================================================
    // Renacer Tracing Integration Tests (Issue #9)
    // =========================================================================

    mod renacer_tracing_tests {
        use super::*;

        #[test]
        fn test_browser_config_with_tracing() {
            let tracing_config = RenacerTracingConfig::new("test-service");
            let config = BrowserConfig::default().with_tracing(tracing_config);
            assert!(config.tracing_config.is_some());
            assert!(config.is_tracing_enabled());
        }

        #[test]
        fn test_browser_config_without_tracing() {
            let config = BrowserConfig::default();
            assert!(config.tracing_config.is_none());
            assert!(!config.is_tracing_enabled());
        }

        #[test]
        fn test_browser_config_disabled_tracing() {
            let tracing_config = RenacerTracingConfig::disabled();
            let config = BrowserConfig::default().with_tracing(tracing_config);
            assert!(config.tracing_config.is_some());
            assert!(!config.is_tracing_enabled());
        }
    }

    #[cfg(not(feature = "browser"))]
    mod mock_renacer_tracing_tests {
        use super::*;

        #[test]
        fn test_page_tracing_disabled_by_default() {
            let page = Page::new(800, 600);
            assert!(!page.is_tracing_enabled());
            assert!(page.traceparent().is_none());
            assert!(page.export_chrome_trace().is_none());
        }

        #[test]
        fn test_page_with_tracing_enabled() {
            let trace_collector = TraceCollector::new("test-service");
            let page = Page::new_with_tracing(800, 600, Some(trace_collector));
            assert!(page.is_tracing_enabled());
            assert!(page.traceparent().is_some());
        }

        #[test]
        fn test_page_traceparent_format() {
            let trace_collector = TraceCollector::new("test-service");
            let page = Page::new_with_tracing(800, 600, Some(trace_collector));
            let traceparent = page.traceparent().unwrap();
            assert!(traceparent.starts_with("00-"));
            let parts: Vec<&str> = traceparent.split('-').collect();
            assert_eq!(parts.len(), 4);
        }

        #[test]
        fn test_page_start_and_record_span() {
            let trace_collector = TraceCollector::new("test-service");
            let mut page = Page::new_with_tracing(800, 600, Some(trace_collector));

            let mut span = page.start_span("test-span", "browser").unwrap();
            span.add_attribute("key", "value");
            span.end();
            page.record_span(span);

            let chrome_trace = page.export_chrome_trace().unwrap();
            assert_eq!(chrome_trace.trace_events.len(), 1);
            assert_eq!(chrome_trace.trace_events[0].name, "test-span");
        }

        #[test]
        fn test_page_record_trace_console() {
            let trace_collector = TraceCollector::new("test-service");
            let mut page = Page::new_with_tracing(800, 600, Some(trace_collector));

            page.record_trace_console("test message");

            let chrome_trace = page.export_chrome_trace().unwrap();
            assert_eq!(chrome_trace.trace_events.len(), 1);
            assert_eq!(chrome_trace.trace_events[0].cat, "console");
        }

        #[test]
        fn test_page_export_trace_json() {
            let trace_collector = TraceCollector::new("test-service");
            let mut page = Page::new_with_tracing(800, 600, Some(trace_collector));

            let mut span = page.start_span("json-test", "browser").unwrap();
            span.end();
            page.record_span(span);

            let json = page.export_trace_json().unwrap().unwrap();
            assert!(json.contains("traceEvents"));
            assert!(json.contains("json-test"));
        }

        #[test]
        fn test_page_inject_trace_context() {
            let trace_collector = TraceCollector::new("test-service");
            let mut page = Page::new_with_tracing(800, 600, Some(trace_collector));
            // Mock implementation just returns Ok
            let result = page.inject_trace_context();
            assert!(result.is_ok());
        }

        #[test]
        fn test_browser_new_page_with_tracing() {
            let tracing_config = RenacerTracingConfig::new("test-service");
            let config = BrowserConfig::default().with_tracing(tracing_config);
            let browser = Browser::launch(config).unwrap();
            let page = browser.new_page().unwrap();
            assert!(page.is_tracing_enabled());
            assert!(page.traceparent().is_some());
        }

        #[test]
        fn test_browser_new_page_without_tracing() {
            let config = BrowserConfig::default();
            let browser = Browser::launch(config).unwrap();
            let page = browser.new_page().unwrap();
            assert!(!page.is_tracing_enabled());
            assert!(page.traceparent().is_none());
        }
    }

    // =========================================================================
    // CDP Coverage Integration Tests (Issue #10)
    // =========================================================================

    #[cfg(not(feature = "browser"))]
    mod mock_coverage_tests {
        use super::*;
        use crate::cdp_coverage::{CoverageConfig, CoverageRange, FunctionCoverage};

        #[test]
        fn test_coverage_disabled_by_default() {
            let page = Page::new(800, 600);
            assert!(!page.is_coverage_enabled());
        }

        #[test]
        fn test_start_coverage() {
            let mut page = Page::new(800, 600);
            assert!(page.start_coverage().is_ok());
            assert!(page.is_coverage_enabled());
        }

        #[test]
        fn test_take_coverage_without_start_fails() {
            let page = Page::new(800, 600);
            let result = page.take_coverage();
            assert!(result.is_err());
        }

        #[test]
        fn test_take_coverage_returns_report() {
            let mut page = Page::new(800, 600);
            page.goto("http://localhost:8080/test.html").unwrap();
            page.start_coverage().unwrap();

            let report = page.take_coverage().unwrap();
            assert_eq!(report.scripts.len(), 1);
            assert_eq!(report.scripts[0].url, "http://localhost:8080/test.html");
            assert!(report.timestamp_ms > 0);
        }

        #[test]
        fn test_stop_coverage_returns_report_and_disables() {
            let mut page = Page::new(800, 600);
            page.start_coverage().unwrap();

            let report = page.stop_coverage().unwrap();
            assert_eq!(report.scripts.len(), 1);
            assert!(!page.is_coverage_enabled());
        }

        #[test]
        fn test_add_mock_coverage() {
            let mut page = Page::new(800, 600);
            page.start_coverage().unwrap();

            page.add_mock_coverage(FunctionCoverage {
                function_name: "test_func".to_string(),
                ranges: vec![CoverageRange {
                    start_offset: 0,
                    end_offset: 100,
                    count: 5,
                }],
                is_block_coverage: false,
            });

            let report = page.take_coverage().unwrap();
            assert_eq!(report.scripts[0].functions.len(), 1);
            assert_eq!(report.scripts[0].functions[0].function_name, "test_func");
            assert_eq!(report.scripts[0].functions[0].ranges[0].count, 5);
        }

        #[test]
        fn test_clear_mock_coverage() {
            let mut page = Page::new(800, 600);
            page.start_coverage().unwrap();

            page.add_mock_coverage(FunctionCoverage {
                function_name: "func1".to_string(),
                ranges: vec![],
                is_block_coverage: false,
            });
            page.add_mock_coverage(FunctionCoverage {
                function_name: "func2".to_string(),
                ranges: vec![],
                is_block_coverage: false,
            });

            page.clear_mock_coverage();

            let report = page.take_coverage().unwrap();
            assert!(report.scripts[0].functions.is_empty());
        }

        #[test]
        fn test_coverage_with_config() {
            let mut page = Page::new(800, 600);
            let config = CoverageConfig {
                call_count: true,
                detailed: true,
                allow_triggered_updates: false,
            };
            assert!(page.start_coverage_with_config(config).is_ok());
            assert!(page.is_coverage_enabled());
        }

        #[test]
        fn test_multiple_coverage_sessions() {
            let mut page = Page::new(800, 600);

            // First session
            page.start_coverage().unwrap();
            page.add_mock_coverage(FunctionCoverage {
                function_name: "session1_func".to_string(),
                ranges: vec![],
                is_block_coverage: false,
            });
            page.stop_coverage().unwrap();

            // Second session
            page.start_coverage().unwrap();
            let report = page.take_coverage().unwrap();
            // Coverage data persists (mock behavior)
            assert_eq!(report.scripts[0].functions.len(), 1);
        }
    }

    // =========================================================================
    // Additional Comprehensive Coverage Tests
    // =========================================================================

    mod browser_console_level_comprehensive {
        use super::*;

        #[test]
        fn test_all_levels_display() {
            // Test Display for all variants
            let levels = [
                (BrowserConsoleLevel::Log, "log"),
                (BrowserConsoleLevel::Info, "info"),
                (BrowserConsoleLevel::Warning, "warn"),
                (BrowserConsoleLevel::Error, "error"),
                (BrowserConsoleLevel::Debug, "debug"),
            ];
            for (level, expected) in levels {
                assert_eq!(format!("{}", level), expected);
            }
        }

        #[test]
        fn test_level_copy_semantics() {
            let level = BrowserConsoleLevel::Warning;
            let copied = level;
            assert_eq!(level, copied);
            // Both should still be usable (Copy trait)
            assert_eq!(format!("{}", level), "warn");
            assert_eq!(format!("{}", copied), "warn");
        }

        #[test]
        fn test_level_equality_all_pairs() {
            let levels = [
                BrowserConsoleLevel::Log,
                BrowserConsoleLevel::Info,
                BrowserConsoleLevel::Warning,
                BrowserConsoleLevel::Error,
                BrowserConsoleLevel::Debug,
            ];
            // Each level should only equal itself
            for (i, level_a) in levels.iter().enumerate() {
                for (j, level_b) in levels.iter().enumerate() {
                    if i == j {
                        assert_eq!(level_a, level_b);
                    } else {
                        assert_ne!(level_a, level_b);
                    }
                }
            }
        }

        #[test]
        fn test_level_debug_all_variants() {
            assert!(format!("{:?}", BrowserConsoleLevel::Log).contains("Log"));
            assert!(format!("{:?}", BrowserConsoleLevel::Info).contains("Info"));
            assert!(format!("{:?}", BrowserConsoleLevel::Warning).contains("Warning"));
            assert!(format!("{:?}", BrowserConsoleLevel::Error).contains("Error"));
            assert!(format!("{:?}", BrowserConsoleLevel::Debug).contains("Debug"));
        }
    }

    mod browser_console_message_comprehensive {
        use super::*;

        #[test]
        fn test_message_all_fields() {
            let msg = BrowserConsoleMessage {
                level: BrowserConsoleLevel::Warning,
                text: "Test warning message".to_string(),
                timestamp: 9999999999,
                source: Some("file.js".to_string()),
                line: Some(123),
            };
            assert_eq!(msg.level, BrowserConsoleLevel::Warning);
            assert_eq!(msg.text, "Test warning message");
            assert_eq!(msg.timestamp, 9999999999);
            assert_eq!(msg.source.as_deref(), Some("file.js"));
            assert_eq!(msg.line, Some(123));
        }

        #[test]
        fn test_message_empty_text() {
            let msg = BrowserConsoleMessage {
                level: BrowserConsoleLevel::Log,
                text: String::new(),
                timestamp: 0,
                source: None,
                line: None,
            };
            assert!(msg.text.is_empty());
        }

        #[test]
        fn test_message_unicode_text() {
            let msg = BrowserConsoleMessage {
                level: BrowserConsoleLevel::Info,
                text: "Unicode: \u{1F600} \u{1F4BB}".to_string(),
                timestamp: 100,
                source: Some("/path/\u{65E5}\u{672C}\u{8A9E}.js".to_string()),
                line: Some(1),
            };
            assert!(msg.text.contains("\u{1F600}"));
            assert!(msg.source.as_ref().unwrap().contains("\u{65E5}"));
        }

        #[test]
        fn test_message_clone_deep() {
            let original = BrowserConsoleMessage {
                level: BrowserConsoleLevel::Error,
                text: "Error message".to_string(),
                timestamp: 12345,
                source: Some("source.js".to_string()),
                line: Some(42),
            };
            let cloned = original.clone();

            // Verify all fields match
            assert_eq!(original.level, cloned.level);
            assert_eq!(original.text, cloned.text);
            assert_eq!(original.timestamp, cloned.timestamp);
            assert_eq!(original.source, cloned.source);
            assert_eq!(original.line, cloned.line);
        }

        #[test]
        fn test_message_debug_format_comprehensive() {
            let msg = BrowserConsoleMessage {
                level: BrowserConsoleLevel::Debug,
                text: "debug text".to_string(),
                timestamp: 555,
                source: Some("test.js".to_string()),
                line: Some(10),
            };
            let debug = format!("{:?}", msg);
            assert!(debug.contains("BrowserConsoleMessage"));
            assert!(debug.contains("debug text"));
            assert!(debug.contains("555"));
            assert!(debug.contains("test.js"));
            assert!(debug.contains("10"));
        }

        #[test]
        fn test_message_max_values() {
            let msg = BrowserConsoleMessage {
                level: BrowserConsoleLevel::Log,
                text: "max".to_string(),
                timestamp: u64::MAX,
                source: None,
                line: Some(u32::MAX),
            };
            assert_eq!(msg.timestamp, u64::MAX);
            assert_eq!(msg.line, Some(u32::MAX));
        }
    }

    mod browser_config_comprehensive {
        use super::*;

        #[test]
        fn test_config_all_builder_methods() {
            let config = BrowserConfig::default()
                .with_viewport(1920, 1080)
                .with_headless(false)
                .with_chromium_path("/custom/path")
                .with_user_agent("Custom Agent")
                .with_no_sandbox();

            assert_eq!(config.viewport_width, 1920);
            assert_eq!(config.viewport_height, 1080);
            assert!(!config.headless);
            assert_eq!(config.chromium_path, Some("/custom/path".to_string()));
            assert_eq!(config.user_agent, Some("Custom Agent".to_string()));
            assert!(!config.sandbox);
        }

        #[test]
        fn test_config_tracing_enabled_check() {
            // Without tracing
            let config = BrowserConfig::default();
            assert!(!config.is_tracing_enabled());

            // With enabled tracing
            let tracing = RenacerTracingConfig::new("test");
            let config_with_tracing = BrowserConfig::default().with_tracing(tracing);
            assert!(config_with_tracing.is_tracing_enabled());

            // With disabled tracing
            let disabled_tracing = RenacerTracingConfig::disabled();
            let config_disabled = BrowserConfig::default().with_tracing(disabled_tracing);
            assert!(!config_disabled.is_tracing_enabled());
        }

        #[test]
        fn test_config_debug_format() {
            let config = BrowserConfig::default()
                .with_viewport(800, 600)
                .with_headless(true);
            let debug = format!("{:?}", config);
            assert!(debug.contains("BrowserConfig"));
            assert!(debug.contains("800"));
            assert!(debug.contains("600"));
            assert!(debug.contains("headless"));
        }

        #[test]
        fn test_config_clone_all_fields() {
            let tracing = RenacerTracingConfig::new("service");
            let config = BrowserConfig::default()
                .with_viewport(1024, 768)
                .with_headless(false)
                .with_chromium_path("/path")
                .with_user_agent("Agent")
                .with_no_sandbox()
                .with_tracing(tracing);

            let cloned = config.clone();
            assert_eq!(config.viewport_width, cloned.viewport_width);
            assert_eq!(config.viewport_height, cloned.viewport_height);
            assert_eq!(config.headless, cloned.headless);
            assert_eq!(config.chromium_path, cloned.chromium_path);
            assert_eq!(config.user_agent, cloned.user_agent);
            assert_eq!(config.sandbox, cloned.sandbox);
            assert!(cloned.tracing_config.is_some());
        }

        #[test]
        fn test_config_with_into_string() {
            // Test that Into<String> works with String
            let config = BrowserConfig::default()
                .with_chromium_path(String::from("/path"))
                .with_user_agent(String::from("UA"));
            assert!(config.chromium_path.is_some());
            assert!(config.user_agent.is_some());
        }

        #[test]
        fn test_config_default_values() {
            let config = BrowserConfig::default();
            assert!(config.headless);
            assert_eq!(config.viewport_width, 800);
            assert_eq!(config.viewport_height, 600);
            assert!(config.chromium_path.is_none());
            assert_eq!(config.debug_port, 0);
            assert!(config.user_agent.is_none());
            assert!(!config.devtools);
            assert!(config.sandbox);
            assert!(config.tracing_config.is_none());
        }
    }

    #[cfg(not(feature = "browser"))]
    mod mock_browser_comprehensive {
        use super::*;

        #[test]
        fn test_browser_launch_with_all_config() {
            let tracing = RenacerTracingConfig::new("test");
            let config = BrowserConfig::default()
                .with_viewport(1280, 720)
                .with_headless(false)
                .with_no_sandbox()
                .with_tracing(tracing);

            let browser = Browser::launch(config).unwrap();
            assert_eq!(browser.config().viewport_width, 1280);
            assert!(!browser.config().headless);
            assert!(!browser.config().sandbox);
        }

        #[test]
        fn test_browser_multiple_pages() {
            let browser = Browser::launch(BrowserConfig::default()).unwrap();
            let page1 = browser.new_page().unwrap();
            let page2 = browser.new_page().unwrap();
            assert_eq!(page1.width, 800);
            assert_eq!(page2.width, 800);
        }

        #[test]
        fn test_browser_config_accessor() {
            let config = BrowserConfig::default().with_viewport(1920, 1080);
            let browser = Browser::launch(config).unwrap();
            let returned_config = browser.config();
            assert_eq!(returned_config.viewport_width, 1920);
            assert_eq!(returned_config.viewport_height, 1080);
        }
    }

    #[cfg(not(feature = "browser"))]
    mod mock_page_comprehensive {
        use super::*;

        #[test]
        fn test_page_new_with_tracing() {
            let collector = TraceCollector::new("test");
            let page = Page::new_with_tracing(1024, 768, Some(collector));
            assert_eq!(page.width, 1024);
            assert_eq!(page.height, 768);
            assert!(page.is_tracing_enabled());
        }

        #[test]
        fn test_page_new_without_tracing() {
            let page = Page::new_with_tracing(800, 600, None);
            assert!(!page.is_tracing_enabled());
        }

        #[test]
        fn test_page_goto_various_urls() {
            let mut page = Page::new(800, 600);

            // HTTP
            page.goto("http://example.com").unwrap();
            assert_eq!(page.current_url(), "http://example.com");

            // HTTPS
            page.goto("https://secure.example.com").unwrap();
            assert_eq!(page.current_url(), "https://secure.example.com");

            // Localhost
            page.goto("http://localhost:8080/path").unwrap();
            assert_eq!(page.current_url(), "http://localhost:8080/path");

            // File URL
            page.goto("file:///path/to/file.html").unwrap();
            assert_eq!(page.current_url(), "file:///path/to/file.html");
        }

        #[test]
        fn test_page_wasm_ready_lifecycle() {
            let mut page = Page::new(800, 600);
            assert!(!page.is_wasm_ready());
            assert!(!page.wasm_ready);

            page.wait_for_wasm_ready().unwrap();
            assert!(page.is_wasm_ready());
            assert!(page.wasm_ready);
        }

        #[test]
        fn test_page_eval_wasm_error_message() {
            let page = Page::new(800, 600);
            let result: Result<String, _> = page.eval_wasm("window.test");
            let err = result.unwrap_err();
            let err_str = format!("{}", err);
            assert!(err_str.contains("Browser feature not enabled"));
        }

        #[test]
        fn test_page_all_touch_actions() {
            let page = Page::new(800, 600);

            // Tap
            let tap = crate::Touch {
                x: 100.0,
                y: 200.0,
                action: crate::TouchAction::Tap,
            };
            assert!(page.touch(tap).is_ok());

            // Swipe
            let swipe = crate::Touch {
                x: 50.0,
                y: 50.0,
                action: crate::TouchAction::Swipe {
                    end_x: 250.0,
                    end_y: 250.0,
                    duration_ms: 200,
                },
            };
            assert!(page.touch(swipe).is_ok());

            // Hold
            let hold = crate::Touch {
                x: 300.0,
                y: 300.0,
                action: crate::TouchAction::Hold { duration_ms: 1000 },
            };
            assert!(page.touch(hold).is_ok());
        }

        #[test]
        fn test_page_screenshot_returns_empty() {
            let page = Page::new(800, 600);
            let screenshot = page.screenshot().unwrap();
            assert!(screenshot.is_empty());
        }

        #[test]
        fn test_page_debug_includes_fields() {
            let mut page = Page::new(1024, 768);
            page.goto("http://test.com").unwrap();
            let debug = format!("{:?}", page);
            assert!(debug.contains("Page"));
            assert!(debug.contains("1024"));
            assert!(debug.contains("768"));
        }
    }

    #[cfg(not(feature = "browser"))]
    mod mock_console_comprehensive {
        use super::*;

        #[test]
        fn test_enable_console_capture_idempotent() {
            let mut page = Page::new(800, 600);
            page.enable_console_capture().unwrap();
            assert!(page.is_console_capture_enabled());
            // Enable again should still work
            page.enable_console_capture().unwrap();
            assert!(page.is_console_capture_enabled());
        }

        #[test]
        fn test_add_multiple_console_messages() {
            let page = Page::new(800, 600);
            for i in 0..10 {
                page.add_console_message(BrowserConsoleMessage {
                    level: BrowserConsoleLevel::Log,
                    text: format!("Message {}", i),
                    timestamp: i as u64 * 100,
                    source: None,
                    line: None,
                });
            }
            assert_eq!(page.console_messages().len(), 10);
        }

        #[test]
        fn test_clear_console_after_messages() {
            let page = Page::new(800, 600);
            page.add_console_message(BrowserConsoleMessage {
                level: BrowserConsoleLevel::Error,
                text: "Error 1".to_string(),
                timestamp: 0,
                source: None,
                line: None,
            });
            page.add_console_message(BrowserConsoleMessage {
                level: BrowserConsoleLevel::Error,
                text: "Error 2".to_string(),
                timestamp: 1,
                source: None,
                line: None,
            });
            assert_eq!(page.console_messages().len(), 2);

            page.clear_console();
            assert!(page.console_messages().is_empty());
        }

        #[test]
        fn test_wait_for_console_complex_predicate() {
            let page = Page::new(800, 600);
            page.add_console_message(BrowserConsoleMessage {
                level: BrowserConsoleLevel::Log,
                text: "startup complete".to_string(),
                timestamp: 100,
                source: Some("main.js".to_string()),
                line: Some(1),
            });
            page.add_console_message(BrowserConsoleMessage {
                level: BrowserConsoleLevel::Error,
                text: "error: failed".to_string(),
                timestamp: 200,
                source: Some("error.js".to_string()),
                line: Some(42),
            });

            // Find by multiple criteria
            let result = page.wait_for_console(
                |m| m.level == BrowserConsoleLevel::Error && m.text.contains("failed"),
                1000,
            );
            assert!(result.is_ok());
            let msg = result.unwrap();
            assert_eq!(msg.text, "error: failed");
        }

        #[test]
        fn test_inject_console_capture_sets_flag() {
            let mut page = Page::new(800, 600);
            assert!(!page.is_console_capture_enabled());
            page.inject_console_capture().unwrap();
            assert!(page.is_console_capture_enabled());
        }

        #[test]
        fn test_fetch_console_messages_returns_copy() {
            let page = Page::new(800, 600);
            page.add_console_message(BrowserConsoleMessage {
                level: BrowserConsoleLevel::Info,
                text: "info".to_string(),
                timestamp: 0,
                source: None,
                line: None,
            });

            let fetched = page.fetch_console_messages().unwrap();
            assert_eq!(fetched.len(), 1);

            // Original should still have messages
            assert_eq!(page.console_messages().len(), 1);
        }

        #[test]
        fn test_console_messages_with_all_levels() {
            let page = Page::new(800, 600);
            let levels = [
                BrowserConsoleLevel::Log,
                BrowserConsoleLevel::Info,
                BrowserConsoleLevel::Warning,
                BrowserConsoleLevel::Error,
                BrowserConsoleLevel::Debug,
            ];

            for level in levels {
                page.add_console_message(BrowserConsoleMessage {
                    level,
                    text: format!("{:?}", level),
                    timestamp: 0,
                    source: None,
                    line: None,
                });
            }

            let messages = page.console_messages();
            assert_eq!(messages.len(), 5);
        }
    }

    #[cfg(not(feature = "browser"))]
    mod mock_tracing_comprehensive {
        use super::*;

        #[test]
        fn test_traceparent_format_w3c() {
            let collector = TraceCollector::new("test");
            let page = Page::new_with_tracing(800, 600, Some(collector));
            let tp = page.traceparent().unwrap();

            // W3C traceparent format: version-traceid-spanid-flags
            let parts: Vec<&str> = tp.split('-').collect();
            assert_eq!(parts.len(), 4);
            assert_eq!(parts[0], "00"); // version
            assert_eq!(parts[1].len(), 32); // trace-id is 32 hex chars
            assert_eq!(parts[2].len(), 16); // span-id is 16 hex chars
        }

        #[test]
        fn test_start_span_without_tracing() {
            let mut page = Page::new(800, 600);
            let span = page.start_span("test", "category");
            assert!(span.is_none());
        }

        #[test]
        fn test_start_span_with_tracing() {
            let collector = TraceCollector::new("test");
            let mut page = Page::new_with_tracing(800, 600, Some(collector));
            let span = page.start_span("operation", "http");
            assert!(span.is_some());
            let mut span = span.unwrap();
            span.end();
            page.record_span(span);
        }

        #[test]
        fn test_record_span_without_tracing() {
            let mut page = Page::new(800, 600);
            // Create a span from a collector
            let mut collector = TraceCollector::new("temp");
            let mut span = collector.start_span("test", "cat");
            span.end();
            // Recording on page without tracing is a no-op
            page.record_span(span);
        }

        #[test]
        fn test_record_trace_console_without_tracing() {
            let mut page = Page::new(800, 600);
            // Should be a no-op
            page.record_trace_console("test message");
        }

        #[test]
        fn test_record_trace_console_with_tracing() {
            let collector = TraceCollector::new("test");
            let mut page = Page::new_with_tracing(800, 600, Some(collector));
            page.record_trace_console("console log entry");

            let trace = page.export_chrome_trace().unwrap();
            assert!(!trace.trace_events.is_empty());
        }

        #[test]
        fn test_export_chrome_trace_without_tracing() {
            let page = Page::new(800, 600);
            assert!(page.export_chrome_trace().is_none());
        }

        #[test]
        fn test_export_trace_json_without_tracing() {
            let page = Page::new(800, 600);
            let json = page.export_trace_json().unwrap();
            assert!(json.is_none());
        }

        #[test]
        fn test_export_trace_json_with_tracing() {
            let collector = TraceCollector::new("test");
            let mut page = Page::new_with_tracing(800, 600, Some(collector));

            let mut span = page.start_span("test-op", "test-cat").unwrap();
            span.end();
            page.record_span(span);

            let json = page.export_trace_json().unwrap();
            assert!(json.is_some());
            let json_str = json.unwrap();
            assert!(json_str.contains("traceEvents"));
        }

        #[test]
        fn test_inject_trace_context_mock() {
            let collector = TraceCollector::new("test");
            let mut page = Page::new_with_tracing(800, 600, Some(collector));
            // Mock just returns Ok
            assert!(page.inject_trace_context().is_ok());
        }

        #[test]
        fn test_inject_trace_context_without_tracing() {
            let mut page = Page::new(800, 600);
            assert!(page.inject_trace_context().is_ok());
        }
    }

    #[cfg(not(feature = "browser"))]
    mod mock_coverage_comprehensive {
        use super::*;
        use crate::cdp_coverage::{CoverageConfig, CoverageRange, FunctionCoverage};

        #[test]
        fn test_coverage_lifecycle_complete() {
            let mut page = Page::new(800, 600);
            assert!(!page.is_coverage_enabled());

            // Start
            page.start_coverage().unwrap();
            assert!(page.is_coverage_enabled());

            // Add data
            page.add_mock_coverage(FunctionCoverage {
                function_name: "testFn".to_string(),
                ranges: vec![CoverageRange {
                    start_offset: 0,
                    end_offset: 50,
                    count: 3,
                }],
                is_block_coverage: true,
            });

            // Take (doesn't stop)
            let report1 = page.take_coverage().unwrap();
            assert!(page.is_coverage_enabled());
            assert_eq!(report1.scripts[0].functions.len(), 1);

            // Stop
            let report2 = page.stop_coverage().unwrap();
            assert!(!page.is_coverage_enabled());
            assert_eq!(report2.scripts[0].functions.len(), 1);
        }

        #[test]
        fn test_coverage_config_options() {
            let mut page = Page::new(800, 600);

            // With detailed config
            let config = CoverageConfig {
                call_count: false,
                detailed: false,
                allow_triggered_updates: true,
            };
            page.start_coverage_with_config(config).unwrap();
            assert!(page.is_coverage_enabled());
        }

        #[test]
        fn test_coverage_report_url() {
            let mut page = Page::new(800, 600);
            page.goto("http://localhost:8080/app.html").unwrap();
            page.start_coverage().unwrap();

            let report = page.take_coverage().unwrap();
            assert_eq!(report.scripts[0].url, "http://localhost:8080/app.html");
        }

        #[test]
        fn test_coverage_report_timestamp() {
            let mut page = Page::new(800, 600);
            page.start_coverage().unwrap();

            let report = page.take_coverage().unwrap();
            // Timestamp should be non-zero (current time)
            assert!(report.timestamp_ms > 0);
        }

        #[test]
        fn test_clear_mock_coverage() {
            let mut page = Page::new(800, 600);
            page.start_coverage().unwrap();

            page.add_mock_coverage(FunctionCoverage {
                function_name: "fn1".to_string(),
                ranges: vec![],
                is_block_coverage: false,
            });
            page.add_mock_coverage(FunctionCoverage {
                function_name: "fn2".to_string(),
                ranges: vec![],
                is_block_coverage: false,
            });

            let report1 = page.take_coverage().unwrap();
            assert_eq!(report1.scripts[0].functions.len(), 2);

            page.clear_mock_coverage();

            let report2 = page.take_coverage().unwrap();
            assert!(report2.scripts[0].functions.is_empty());
        }

        #[test]
        fn test_coverage_multiple_functions() {
            let mut page = Page::new(800, 600);
            page.start_coverage().unwrap();

            for i in 0..5 {
                page.add_mock_coverage(FunctionCoverage {
                    function_name: format!("function_{}", i),
                    ranges: vec![CoverageRange {
                        start_offset: i * 100,
                        end_offset: (i + 1) * 100,
                        count: i + 1,
                    }],
                    is_block_coverage: i % 2 == 0,
                });
            }

            let report = page.take_coverage().unwrap();
            assert_eq!(report.scripts[0].functions.len(), 5);
        }

        #[test]
        fn test_take_coverage_error_when_disabled() {
            let page = Page::new(800, 600);
            let result = page.take_coverage();
            assert!(result.is_err());
            let err = result.unwrap_err();
            let err_str = format!("{}", err);
            assert!(err_str.contains("Coverage not enabled"));
        }

        #[test]
        fn test_stop_coverage_error_when_disabled() {
            let mut page = Page::new(800, 600);
            let result = page.stop_coverage();
            assert!(result.is_err());
        }
    }

    #[cfg(not(feature = "browser"))]
    mod mock_integration_tests {
        use super::*;

        #[test]
        fn test_full_page_lifecycle() {
            let config = BrowserConfig::default()
                .with_viewport(1280, 720)
                .with_headless(true);
            let browser = Browser::launch(config).unwrap();
            let mut page = browser.new_page().unwrap();

            // Navigate
            page.goto("http://localhost:8080").unwrap();
            assert_eq!(page.current_url(), "http://localhost:8080");

            // Wait for WASM
            page.wait_for_wasm_ready().unwrap();
            assert!(page.is_wasm_ready());

            // Console capture
            page.enable_console_capture().unwrap();
            page.add_console_message(BrowserConsoleMessage {
                level: BrowserConsoleLevel::Log,
                text: "ready".to_string(),
                timestamp: 100,
                source: None,
                line: None,
            });

            // Coverage
            page.start_coverage().unwrap();
            let report = page.stop_coverage().unwrap();
            assert!(!report.scripts.is_empty());

            // Screenshot
            let screenshot = page.screenshot().unwrap();
            assert!(screenshot.is_empty()); // Mock returns empty
        }

        #[test]
        fn test_browser_with_tracing_creates_traced_pages() {
            let tracing = RenacerTracingConfig::new("integration-test");
            let config = BrowserConfig::default().with_tracing(tracing);
            let browser = Browser::launch(config).unwrap();
            let mut page = browser.new_page().unwrap();

            assert!(page.is_tracing_enabled());
            let traceparent = page.traceparent();
            assert!(traceparent.is_some());

            // Start a span
            let mut span = page.start_span("test-op", "test-cat").unwrap();
            span.add_attribute("key", "value");
            span.end();
            page.record_span(span);

            // Export
            let json = page.export_trace_json().unwrap();
            assert!(json.is_some());
        }

        #[test]
        fn test_console_and_coverage_together() {
            let mut page = Page::new(800, 600);
            page.enable_console_capture().unwrap();
            page.start_coverage().unwrap();

            // Simulate console output
            page.add_console_message(BrowserConsoleMessage {
                level: BrowserConsoleLevel::Log,
                text: "Starting app...".to_string(),
                timestamp: 1,
                source: None,
                line: None,
            });

            // Simulate coverage
            page.add_mock_coverage(crate::cdp_coverage::FunctionCoverage {
                function_name: "init".to_string(),
                ranges: vec![crate::cdp_coverage::CoverageRange {
                    start_offset: 0,
                    end_offset: 100,
                    count: 1,
                }],
                is_block_coverage: false,
            });

            // Check console
            let messages = page.console_messages();
            assert_eq!(messages.len(), 1);

            // Check coverage
            let report = page.stop_coverage().unwrap();
            assert_eq!(report.scripts[0].functions.len(), 1);
        }
    }

    mod edge_cases {
        use super::*;

        #[test]
        fn test_browser_config_zero_viewport() {
            let config = BrowserConfig::default().with_viewport(0, 0);
            assert_eq!(config.viewport_width, 0);
            assert_eq!(config.viewport_height, 0);
        }

        #[test]
        fn test_browser_config_max_viewport() {
            let config = BrowserConfig::default().with_viewport(u32::MAX, u32::MAX);
            assert_eq!(config.viewport_width, u32::MAX);
            assert_eq!(config.viewport_height, u32::MAX);
        }

        #[test]
        fn test_browser_config_long_path() {
            let long_path = "a".repeat(10000);
            let config = BrowserConfig::default().with_chromium_path(&long_path);
            assert_eq!(config.chromium_path.as_ref().unwrap().len(), 10000);
        }

        #[test]
        fn test_browser_config_special_chars_user_agent() {
            let ua = "Mozilla/5.0 (Test) <script>alert(1)</script> \n\r\t";
            let config = BrowserConfig::default().with_user_agent(ua);
            assert_eq!(config.user_agent.as_ref().unwrap(), ua);
        }

        #[cfg(not(feature = "browser"))]
        #[test]
        fn test_page_empty_url() {
            let mut page = Page::new(800, 600);
            page.goto("").unwrap();
            assert_eq!(page.current_url(), "");
        }

        #[cfg(not(feature = "browser"))]
        #[test]
        fn test_page_touch_at_boundaries() {
            let page = Page::new(800, 600);

            // Touch at (0,0)
            let tap_origin = crate::Touch {
                x: 0.0,
                y: 0.0,
                action: crate::TouchAction::Tap,
            };
            assert!(page.touch(tap_origin).is_ok());

            // Touch at max coords
            let tap_max = crate::Touch {
                x: f32::MAX,
                y: f32::MAX,
                action: crate::TouchAction::Tap,
            };
            assert!(page.touch(tap_max).is_ok());

            // Negative coords
            let tap_neg = crate::Touch {
                x: -100.0,
                y: -100.0,
                action: crate::TouchAction::Tap,
            };
            assert!(page.touch(tap_neg).is_ok());
        }

        #[test]
        fn test_console_level_copy() {
            let level = BrowserConsoleLevel::Error;
            let copied: BrowserConsoleLevel = level;
            let another: BrowserConsoleLevel = level;
            assert_eq!(copied, another);
        }

        #[cfg(not(feature = "browser"))]
        #[test]
        fn test_wait_for_console_empty_predicate() {
            let page = Page::new(800, 600);
            // Predicate that never matches
            let result = page.wait_for_console(|_| false, 100);
            assert!(result.is_err());
        }

        #[cfg(not(feature = "browser"))]
        #[test]
        fn test_wait_for_console_always_matches() {
            let page = Page::new(800, 600);
            page.add_console_message(BrowserConsoleMessage {
                level: BrowserConsoleLevel::Log,
                text: "any".to_string(),
                timestamp: 0,
                source: None,
                line: None,
            });
            // Predicate that always matches
            let result = page.wait_for_console(|_| true, 100);
            assert!(result.is_ok());
        }
    }

    // =========================================================================
    // Additional Mock Coverage Tests
    // =========================================================================

    #[cfg(not(feature = "browser"))]
    mod additional_mock_coverage_tests {
        use super::*;
        use crate::cdp_coverage::{CoverageRange, FunctionCoverage};

        #[test]
        fn test_page_new_with_tracing_none_explicit() {
            // Explicitly test the None path for trace_collector
            let page = Page::new_with_tracing(640, 480, None);
            assert_eq!(page.width, 640);
            assert_eq!(page.height, 480);
            assert!(!page.is_tracing_enabled());
            assert!(page.traceparent().is_none());
            assert!(page.export_chrome_trace().is_none());
        }

        #[test]
        fn test_page_start_span_returns_none_without_tracing() {
            let mut page = Page::new(800, 600);
            let span = page.start_span("operation", "category");
            assert!(span.is_none());
        }

        #[test]
        fn test_page_record_span_noop_without_tracing() {
            let mut page = Page::new(800, 600);
            // Create a temporary collector to get a span
            let mut temp_collector = TraceCollector::new("temp");
            let mut span = temp_collector.start_span("test", "cat");
            span.end();
            // This should be a no-op
            page.record_span(span);
            // No assertion needed - just ensure no panic
        }

        #[test]
        fn test_page_record_trace_console_noop_without_tracing() {
            let mut page = Page::new(800, 600);
            // Should be a no-op
            page.record_trace_console("test message");
            // No assertion needed - just ensure no panic
        }

        #[test]
        fn test_page_export_trace_json_none_without_tracing() {
            let page = Page::new(800, 600);
            let result = page.export_trace_json();
            assert!(result.is_ok());
            assert!(result.unwrap().is_none());
        }

        #[test]
        fn test_page_inject_trace_context_ok_without_tracing() {
            let mut page = Page::new(800, 600);
            let result = page.inject_trace_context();
            assert!(result.is_ok());
        }

        #[test]
        fn test_browser_launch_preserves_all_config_fields() {
            let config = BrowserConfig {
                headless: false,
                viewport_width: 1920,
                viewport_height: 1080,
                chromium_path: Some("/usr/bin/chromium".to_string()),
                debug_port: 9222,
                user_agent: Some("TestAgent".to_string()),
                devtools: true,
                sandbox: false,
                tracing_config: Some(RenacerTracingConfig::new("test")),
            };
            let browser = Browser::launch(config).unwrap();
            let cfg = browser.config();
            assert!(!cfg.headless);
            assert_eq!(cfg.viewport_width, 1920);
            assert_eq!(cfg.viewport_height, 1080);
            assert_eq!(cfg.chromium_path, Some("/usr/bin/chromium".to_string()));
            assert_eq!(cfg.debug_port, 9222);
            assert_eq!(cfg.user_agent, Some("TestAgent".to_string()));
            assert!(cfg.devtools);
            assert!(!cfg.sandbox);
            assert!(cfg.tracing_config.is_some());
        }

        #[test]
        fn test_page_add_mock_coverage_multiple_ranges() {
            let mut page = Page::new(800, 600);
            page.start_coverage().unwrap();

            page.add_mock_coverage(FunctionCoverage {
                function_name: "multi_range_func".to_string(),
                ranges: vec![
                    CoverageRange {
                        start_offset: 0,
                        end_offset: 50,
                        count: 10,
                    },
                    CoverageRange {
                        start_offset: 50,
                        end_offset: 100,
                        count: 5,
                    },
                    CoverageRange {
                        start_offset: 100,
                        end_offset: 200,
                        count: 0,
                    },
                ],
                is_block_coverage: true,
            });

            let report = page.take_coverage().unwrap();
            assert_eq!(report.scripts[0].functions[0].ranges.len(), 3);
        }

        #[test]
        fn test_console_message_with_source_and_line() {
            let page = Page::new(800, 600);
            page.add_console_message(BrowserConsoleMessage {
                level: BrowserConsoleLevel::Error,
                text: "Error occurred".to_string(),
                timestamp: 12345678,
                source: Some("/path/to/script.js".to_string()),
                line: Some(42),
            });

            let messages = page.console_messages();
            assert_eq!(messages.len(), 1);
            assert_eq!(messages[0].source, Some("/path/to/script.js".to_string()));
            assert_eq!(messages[0].line, Some(42));
        }

        #[test]
        fn test_wait_for_console_predicate_by_timestamp() {
            let page = Page::new(800, 600);
            page.add_console_message(BrowserConsoleMessage {
                level: BrowserConsoleLevel::Log,
                text: "early".to_string(),
                timestamp: 100,
                source: None,
                line: None,
            });
            page.add_console_message(BrowserConsoleMessage {
                level: BrowserConsoleLevel::Log,
                text: "late".to_string(),
                timestamp: 500,
                source: None,
                line: None,
            });

            let result = page.wait_for_console(|m| m.timestamp > 200, 1000);
            assert!(result.is_ok());
            assert_eq!(result.unwrap().text, "late");
        }

        #[test]
        fn test_wait_for_console_predicate_by_source() {
            let page = Page::new(800, 600);
            page.add_console_message(BrowserConsoleMessage {
                level: BrowserConsoleLevel::Warning,
                text: "warning from main".to_string(),
                timestamp: 0,
                source: Some("main.js".to_string()),
                line: Some(10),
            });

            let result = page.wait_for_console(|m| m.source.as_deref() == Some("main.js"), 1000);
            assert!(result.is_ok());
        }

        #[test]
        fn test_coverage_restart_after_stop() {
            let mut page = Page::new(800, 600);

            // First session
            page.start_coverage().unwrap();
            assert!(page.is_coverage_enabled());
            page.stop_coverage().unwrap();
            assert!(!page.is_coverage_enabled());

            // Second session
            page.start_coverage().unwrap();
            assert!(page.is_coverage_enabled());
        }

        #[test]
        fn test_page_with_tracing_multiple_spans() {
            let collector = TraceCollector::new("multi-span-test");
            let mut page = Page::new_with_tracing(800, 600, Some(collector));

            for i in 0..5 {
                let mut span = page.start_span(format!("span_{}", i), "test").unwrap();
                span.add_attribute("index", i.to_string());
                span.end();
                page.record_span(span);
            }

            let trace = page.export_chrome_trace().unwrap();
            assert_eq!(trace.trace_events.len(), 5);
        }

        #[test]
        fn test_page_with_tracing_console_and_spans() {
            let collector = TraceCollector::new("mixed-test");
            let mut page = Page::new_with_tracing(800, 600, Some(collector));

            // Record console messages
            page.record_trace_console("Console 1");
            page.record_trace_console("Console 2");

            // Record spans
            let mut span = page.start_span("operation", "http").unwrap();
            span.end();
            page.record_span(span);

            let trace = page.export_chrome_trace().unwrap();
            // Should have 2 console + 1 span = 3 events
            assert_eq!(trace.trace_events.len(), 3);
        }

        #[test]
        fn test_coverage_with_about_blank_url() {
            let mut page = Page::new(800, 600);
            // Don't call goto - use default about:blank
            page.start_coverage().unwrap();

            let report = page.take_coverage().unwrap();
            assert_eq!(report.scripts[0].url, "about:blank");
        }

        #[test]
        fn test_page_current_url_after_multiple_navigations() {
            let mut page = Page::new(800, 600);
            assert_eq!(page.current_url(), "about:blank");

            let urls = ["http://first.com", "http://second.com", "http://third.com"];
            for url in &urls {
                page.goto(url).unwrap();
            }
            assert_eq!(page.current_url(), "http://third.com");
        }

        #[test]
        fn test_browser_disabled_tracing_creates_untraced_pages() {
            let disabled_tracing = RenacerTracingConfig::disabled();
            let config = BrowserConfig::default().with_tracing(disabled_tracing);
            let browser = Browser::launch(config).unwrap();
            let page = browser.new_page().unwrap();

            assert!(!page.is_tracing_enabled());
            assert!(page.traceparent().is_none());
        }

        #[test]
        fn test_all_touch_actions_mock() {
            let page = Page::new(800, 600);

            // Tap at various positions
            for x in [0.0f32, 400.0, 800.0] {
                for y in [0.0f32, 300.0, 600.0] {
                    let tap = crate::Touch {
                        x,
                        y,
                        action: crate::TouchAction::Tap,
                    };
                    assert!(page.touch(tap).is_ok());
                }
            }

            // Swipe with various durations
            for duration in [0u32, 100, 500, 1000] {
                let swipe = crate::Touch {
                    x: 100.0,
                    y: 100.0,
                    action: crate::TouchAction::Swipe {
                        end_x: 200.0,
                        end_y: 200.0,
                        duration_ms: duration,
                    },
                };
                assert!(page.touch(swipe).is_ok());
            }

            // Hold with various durations
            for duration in [0u32, 100, 500, 2000] {
                let hold = crate::Touch {
                    x: 300.0,
                    y: 300.0,
                    action: crate::TouchAction::Hold {
                        duration_ms: duration,
                    },
                };
                assert!(page.touch(hold).is_ok());
            }
        }

        #[test]
        fn test_console_messages_empty_initially() {
            let page = Page::new(800, 600);
            let fetched = page.fetch_console_messages().unwrap();
            assert!(fetched.is_empty());
        }

        #[test]
        fn test_clear_console_when_empty() {
            let page = Page::new(800, 600);
            // Should not panic
            page.clear_console();
            assert!(page.console_messages().is_empty());
        }

        #[test]
        fn test_clear_mock_coverage_when_empty() {
            let mut page = Page::new(800, 600);
            page.start_coverage().unwrap();
            // Should not panic
            page.clear_mock_coverage();
            let report = page.take_coverage().unwrap();
            assert!(report.scripts[0].functions.is_empty());
        }
    }

    // =========================================================================
    // Error Type and Display Tests
    // =========================================================================

    mod error_display_tests {
        use super::*;

        #[test]
        fn test_browser_console_level_display_matches_expected() {
            // Verify exact string output
            assert_eq!(BrowserConsoleLevel::Log.to_string(), "log");
            assert_eq!(BrowserConsoleLevel::Info.to_string(), "info");
            assert_eq!(BrowserConsoleLevel::Warning.to_string(), "warn");
            assert_eq!(BrowserConsoleLevel::Error.to_string(), "error");
            assert_eq!(BrowserConsoleLevel::Debug.to_string(), "debug");
        }

        #[test]
        fn test_browser_console_level_debug_format() {
            assert_eq!(format!("{:?}", BrowserConsoleLevel::Log), "Log");
            assert_eq!(format!("{:?}", BrowserConsoleLevel::Info), "Info");
            assert_eq!(format!("{:?}", BrowserConsoleLevel::Warning), "Warning");
            assert_eq!(format!("{:?}", BrowserConsoleLevel::Error), "Error");
            assert_eq!(format!("{:?}", BrowserConsoleLevel::Debug), "Debug");
        }

        #[test]
        fn test_browser_console_message_debug_format_complete() {
            let msg = BrowserConsoleMessage {
                level: BrowserConsoleLevel::Error,
                text: "test error".to_string(),
                timestamp: 999,
                source: Some("test.js".to_string()),
                line: Some(99),
            };
            let debug_str = format!("{:?}", msg);
            assert!(debug_str.contains("BrowserConsoleMessage"));
            assert!(debug_str.contains("Error"));
            assert!(debug_str.contains("test error"));
            assert!(debug_str.contains("999"));
            assert!(debug_str.contains("test.js"));
            assert!(debug_str.contains("99"));
        }

        #[test]
        fn test_browser_config_debug_format_complete() {
            let config = BrowserConfig::default()
                .with_viewport(1280, 720)
                .with_chromium_path("/path/to/chrome")
                .with_user_agent("Test UA")
                .with_no_sandbox();
            let debug_str = format!("{:?}", config);
            assert!(debug_str.contains("BrowserConfig"));
            assert!(debug_str.contains("1280"));
            assert!(debug_str.contains("720"));
            assert!(debug_str.contains("/path/to/chrome"));
            assert!(debug_str.contains("Test UA"));
        }

        #[cfg(not(feature = "browser"))]
        #[test]
        fn test_browser_debug_format() {
            let browser = Browser::launch(BrowserConfig::default()).unwrap();
            let debug_str = format!("{:?}", browser);
            assert!(debug_str.contains("Browser"));
        }

        #[cfg(not(feature = "browser"))]
        #[test]
        fn test_page_debug_format() {
            let page = Page::new(1024, 768);
            let debug_str = format!("{:?}", page);
            assert!(debug_str.contains("Page"));
            assert!(debug_str.contains("1024"));
            assert!(debug_str.contains("768"));
        }
    }

    // =========================================================================
    // Property-based-like comprehensive tests
    // =========================================================================

    mod comprehensive_property_tests {
        use super::*;

        #[test]
        fn test_browser_config_viewport_dimensions_preserved() {
            for (w, h) in [
                (100u32, 100u32),
                (800, 600),
                (1920, 1080),
                (3840, 2160),
                (1, 1),
            ] {
                let config = BrowserConfig::default().with_viewport(w, h);
                assert_eq!(config.viewport_width, w);
                assert_eq!(config.viewport_height, h);
            }
        }

        #[test]
        fn test_browser_config_headless_toggle() {
            let config_headless = BrowserConfig::default().with_headless(true);
            assert!(config_headless.headless);

            let config_not_headless = BrowserConfig::default().with_headless(false);
            assert!(!config_not_headless.headless);
        }

        #[test]
        fn test_browser_console_level_equality_reflexive() {
            let levels = [
                BrowserConsoleLevel::Log,
                BrowserConsoleLevel::Info,
                BrowserConsoleLevel::Warning,
                BrowserConsoleLevel::Error,
                BrowserConsoleLevel::Debug,
            ];
            for level in levels {
                assert_eq!(level, level);
            }
        }

        #[test]
        fn test_browser_console_level_clone_equals_original() {
            let levels = [
                BrowserConsoleLevel::Log,
                BrowserConsoleLevel::Info,
                BrowserConsoleLevel::Warning,
                BrowserConsoleLevel::Error,
                BrowserConsoleLevel::Debug,
            ];
            for level in levels {
                let cloned = level;
                assert_eq!(level, cloned);
            }
        }

        #[cfg(not(feature = "browser"))]
        #[test]
        fn test_page_screenshot_always_returns_empty() {
            // Multiple calls should all return empty
            let page = Page::new(800, 600);
            for _ in 0..5 {
                let result = page.screenshot();
                assert!(result.is_ok());
                assert!(result.unwrap().is_empty());
            }
        }

        #[cfg(not(feature = "browser"))]
        #[test]
        fn test_page_eval_wasm_always_fails() {
            let page = Page::new(800, 600);
            let expressions = ["1 + 1", "window.test", "document.body", ""];
            for expr in expressions {
                let result: Result<String, _> = page.eval_wasm(expr);
                assert!(result.is_err());
            }
        }

        #[cfg(not(feature = "browser"))]
        #[test]
        fn test_page_goto_accepts_any_string() {
            let mut page = Page::new(800, 600);
            let urls = [
                "",
                "http://example.com",
                "https://secure.example.com",
                "file:///path/to/file",
                "about:blank",
                "javascript:void(0)",
                "data:text/html,<h1>Test</h1>",
            ];
            for url in urls {
                let result = page.goto(url);
                assert!(result.is_ok());
                assert_eq!(page.current_url(), url);
            }
        }

        #[cfg(not(feature = "browser"))]
        #[test]
        fn test_page_wasm_ready_idempotent() {
            let mut page = Page::new(800, 600);
            assert!(!page.is_wasm_ready());

            for _ in 0..5 {
                page.wait_for_wasm_ready().unwrap();
                assert!(page.is_wasm_ready());
            }
        }
    }

    // =========================================================================
    // BrowserConfig Tracing Tests
    // =========================================================================

    mod browser_config_tracing_tests {
        use super::*;

        #[test]
        fn test_is_tracing_enabled_none() {
            let config = BrowserConfig::default();
            assert!(!config.is_tracing_enabled());
        }

        #[test]
        fn test_is_tracing_enabled_some_enabled() {
            let tracing = RenacerTracingConfig::new("test-service");
            let config = BrowserConfig::default().with_tracing(tracing);
            assert!(config.is_tracing_enabled());
        }

        #[test]
        fn test_is_tracing_enabled_some_disabled() {
            let tracing = RenacerTracingConfig::disabled();
            let config = BrowserConfig::default().with_tracing(tracing);
            assert!(!config.is_tracing_enabled());
        }

        #[test]
        fn test_with_tracing_replaces_previous() {
            let tracing1 = RenacerTracingConfig::new("service1");
            let tracing2 = RenacerTracingConfig::new("service2");

            let config = BrowserConfig::default()
                .with_tracing(tracing1)
                .with_tracing(tracing2);

            assert!(config.is_tracing_enabled());
            let service_name = &config.tracing_config.as_ref().unwrap().service_name;
            assert_eq!(service_name, "service2");
        }
    }

    // =========================================================================
    // Additional Mock Coverage Tests for 99%+ Coverage
    // =========================================================================

    #[cfg(not(feature = "browser"))]
    mod mock_coverage_99_percent {
        use super::*;
        use crate::cdp_coverage::{CoverageConfig, CoverageRange, FunctionCoverage};

        // =====================================================================
        // Page::new() and Page::new_with_tracing() edge cases
        // =====================================================================

        #[test]
        fn test_page_new_default_console_capture_disabled() {
            let page = Page::new(800, 600);
            assert!(!page.is_console_capture_enabled());
        }

        #[test]
        fn test_page_new_default_coverage_disabled() {
            let page = Page::new(800, 600);
            assert!(!page.is_coverage_enabled());
        }

        #[test]
        fn test_page_new_with_tracing_default_fields() {
            let collector = TraceCollector::new("test");
            let page = Page::new_with_tracing(640, 480, Some(collector));
            assert_eq!(page.url, "about:blank");
            assert!(!page.wasm_ready);
            assert!(!page.is_console_capture_enabled());
            assert!(!page.is_coverage_enabled());
        }

        #[test]
        fn test_page_new_with_tracing_none_all_defaults() {
            let page = Page::new_with_tracing(320, 240, None);
            assert_eq!(page.width, 320);
            assert_eq!(page.height, 240);
            assert_eq!(page.url, "about:blank");
            assert!(!page.wasm_ready);
            assert!(!page.is_console_capture_enabled());
            assert!(!page.is_tracing_enabled());
            assert!(!page.is_coverage_enabled());
        }

        // =====================================================================
        // Console message methods - comprehensive edge cases
        // =====================================================================

        #[test]
        fn test_console_messages_returns_empty_vec_initially() {
            let page = Page::new(800, 600);
            let messages = page.console_messages();
            assert!(messages.is_empty());
            assert_eq!(messages.len(), 0);
        }

        #[test]
        fn test_fetch_console_messages_empty() {
            let page = Page::new(800, 600);
            let result = page.fetch_console_messages();
            assert!(result.is_ok());
            assert!(result.unwrap().is_empty());
        }

        #[test]
        fn test_add_console_message_then_fetch() {
            let page = Page::new(800, 600);
            page.add_console_message(BrowserConsoleMessage {
                level: BrowserConsoleLevel::Log,
                text: "test".to_string(),
                timestamp: 100,
                source: None,
                line: None,
            });
            let fetched = page.fetch_console_messages().unwrap();
            assert_eq!(fetched.len(), 1);
            assert_eq!(fetched[0].text, "test");
        }

        #[test]
        fn test_clear_console_empties_messages() {
            let page = Page::new(800, 600);
            page.add_console_message(BrowserConsoleMessage {
                level: BrowserConsoleLevel::Error,
                text: "error".to_string(),
                timestamp: 0,
                source: None,
                line: None,
            });
            assert_eq!(page.console_messages().len(), 1);
            page.clear_console();
            assert_eq!(page.console_messages().len(), 0);
        }

        #[test]
        fn test_wait_for_console_finds_first_match() {
            let page = Page::new(800, 600);
            page.add_console_message(BrowserConsoleMessage {
                level: BrowserConsoleLevel::Log,
                text: "first".to_string(),
                timestamp: 1,
                source: None,
                line: None,
            });
            page.add_console_message(BrowserConsoleMessage {
                level: BrowserConsoleLevel::Log,
                text: "second".to_string(),
                timestamp: 2,
                source: None,
                line: None,
            });
            let result = page.wait_for_console(|m| m.text == "first", 1000);
            assert!(result.is_ok());
            assert_eq!(result.unwrap().text, "first");
        }

        #[test]
        fn test_wait_for_console_no_match_returns_timeout_error() {
            let page = Page::new(800, 600);
            page.add_console_message(BrowserConsoleMessage {
                level: BrowserConsoleLevel::Log,
                text: "exists".to_string(),
                timestamp: 0,
                source: None,
                line: None,
            });
            let result = page.wait_for_console(|m| m.text == "does_not_exist", 100);
            assert!(result.is_err());
            let err = result.unwrap_err();
            let err_str = format!("{}", err);
            assert!(err_str.contains("No matching console message"));
        }

        #[test]
        fn test_enable_console_capture_returns_ok() {
            let mut page = Page::new(800, 600);
            let result = page.enable_console_capture();
            assert!(result.is_ok());
            assert!(page.is_console_capture_enabled());
        }

        #[test]
        fn test_inject_console_capture_returns_ok() {
            let mut page = Page::new(800, 600);
            let result = page.inject_console_capture();
            assert!(result.is_ok());
            assert!(page.is_console_capture_enabled());
        }

        // =====================================================================
        // Tracing methods - comprehensive edge cases
        // =====================================================================

        #[test]
        fn test_traceparent_returns_none_without_collector() {
            let page = Page::new(800, 600);
            assert!(page.traceparent().is_none());
        }

        #[test]
        fn test_traceparent_returns_some_with_collector() {
            let collector = TraceCollector::new("test");
            let page = Page::new_with_tracing(800, 600, Some(collector));
            let tp = page.traceparent();
            assert!(tp.is_some());
            let traceparent = tp.unwrap();
            // Verify W3C format: version-trace_id-parent_id-flags
            let parts: Vec<&str> = traceparent.split('-').collect();
            assert_eq!(parts.len(), 4);
            assert_eq!(parts[0], "00");
            assert_eq!(parts[1].len(), 32);
            assert_eq!(parts[2].len(), 16);
            assert_eq!(parts[3].len(), 2);
        }

        #[test]
        fn test_start_span_returns_none_without_collector() {
            let mut page = Page::new(800, 600);
            let span = page.start_span("test-span", "test-category");
            assert!(span.is_none());
        }

        #[test]
        fn test_start_span_returns_some_with_collector() {
            let collector = TraceCollector::new("test");
            let mut page = Page::new_with_tracing(800, 600, Some(collector));
            let span = page.start_span("test-span", "test-category");
            assert!(span.is_some());
            let span = span.unwrap();
            assert_eq!(span.name, "test-span");
            assert_eq!(span.category, "test-category");
        }

        #[test]
        fn test_record_span_noop_without_collector() {
            let mut page = Page::new(800, 600);
            // Create a span from a temporary collector
            let mut temp = TraceCollector::new("temp");
            let mut span = temp.start_span("span", "cat");
            span.end();
            // This should be a no-op, not panic
            page.record_span(span);
        }

        #[test]
        fn test_record_span_with_collector() {
            let collector = TraceCollector::new("test");
            let mut page = Page::new_with_tracing(800, 600, Some(collector));
            let mut span = page.start_span("recorded-span", "browser").unwrap();
            span.end();
            page.record_span(span);

            let trace = page.export_chrome_trace().unwrap();
            assert!(!trace.trace_events.is_empty());
            assert_eq!(trace.trace_events[0].name, "recorded-span");
        }

        #[test]
        fn test_record_trace_console_noop_without_collector() {
            let mut page = Page::new(800, 600);
            // Should be a no-op, not panic
            page.record_trace_console("test message");
        }

        #[test]
        fn test_record_trace_console_with_collector() {
            let collector = TraceCollector::new("test");
            let mut page = Page::new_with_tracing(800, 600, Some(collector));
            page.record_trace_console("console message 1");
            page.record_trace_console("console message 2");

            let trace = page.export_chrome_trace().unwrap();
            assert_eq!(trace.trace_events.len(), 2);
        }

        #[test]
        fn test_export_chrome_trace_returns_none_without_collector() {
            let page = Page::new(800, 600);
            assert!(page.export_chrome_trace().is_none());
        }

        #[test]
        fn test_export_chrome_trace_returns_some_with_collector() {
            let collector = TraceCollector::new("test");
            let page = Page::new_with_tracing(800, 600, Some(collector));
            let trace = page.export_chrome_trace();
            assert!(trace.is_some());
        }

        #[test]
        fn test_export_trace_json_returns_ok_none_without_collector() {
            let page = Page::new(800, 600);
            let result = page.export_trace_json();
            assert!(result.is_ok());
            assert!(result.unwrap().is_none());
        }

        #[test]
        fn test_export_trace_json_returns_ok_some_with_collector() {
            let collector = TraceCollector::new("test");
            let mut page = Page::new_with_tracing(800, 600, Some(collector));
            let mut span = page.start_span("json-span", "cat").unwrap();
            span.end();
            page.record_span(span);

            let result = page.export_trace_json();
            assert!(result.is_ok());
            let json = result.unwrap();
            assert!(json.is_some());
            let json_str = json.unwrap();
            assert!(json_str.contains("traceEvents"));
            assert!(json_str.contains("json-span"));
        }

        #[test]
        fn test_inject_trace_context_returns_ok() {
            let mut page = Page::new(800, 600);
            let result = page.inject_trace_context();
            assert!(result.is_ok());
        }

        #[test]
        fn test_inject_trace_context_with_tracing_returns_ok() {
            let collector = TraceCollector::new("test");
            let mut page = Page::new_with_tracing(800, 600, Some(collector));
            let result = page.inject_trace_context();
            assert!(result.is_ok());
        }

        // =====================================================================
        // Coverage methods - comprehensive edge cases
        // =====================================================================

        #[test]
        fn test_start_coverage_enables_coverage() {
            let mut page = Page::new(800, 600);
            assert!(!page.is_coverage_enabled());
            let result = page.start_coverage();
            assert!(result.is_ok());
            assert!(page.is_coverage_enabled());
        }

        #[test]
        fn test_start_coverage_with_config_enables_coverage() {
            let mut page = Page::new(800, 600);
            let config = CoverageConfig {
                call_count: false,
                detailed: true,
                allow_triggered_updates: true,
            };
            let result = page.start_coverage_with_config(config);
            assert!(result.is_ok());
            assert!(page.is_coverage_enabled());
        }

        #[test]
        fn test_take_coverage_error_when_not_enabled() {
            let page = Page::new(800, 600);
            let result = page.take_coverage();
            assert!(result.is_err());
            let err = result.unwrap_err();
            let err_str = format!("{}", err);
            assert!(err_str.contains("Coverage not enabled"));
        }

        #[test]
        fn test_take_coverage_returns_report_when_enabled() {
            let mut page = Page::new(800, 600);
            page.goto("http://test.com").unwrap();
            page.start_coverage().unwrap();
            let result = page.take_coverage();
            assert!(result.is_ok());
            let report = result.unwrap();
            assert_eq!(report.scripts.len(), 1);
            assert_eq!(report.scripts[0].url, "http://test.com");
            assert!(report.timestamp_ms > 0);
        }

        #[test]
        fn test_stop_coverage_error_when_not_enabled() {
            let mut page = Page::new(800, 600);
            let result = page.stop_coverage();
            assert!(result.is_err());
        }

        #[test]
        fn test_stop_coverage_returns_report_and_disables() {
            let mut page = Page::new(800, 600);
            page.start_coverage().unwrap();
            assert!(page.is_coverage_enabled());

            let result = page.stop_coverage();
            assert!(result.is_ok());
            assert!(!page.is_coverage_enabled());
        }

        #[test]
        fn test_add_mock_coverage_adds_function() {
            let mut page = Page::new(800, 600);
            page.start_coverage().unwrap();

            page.add_mock_coverage(FunctionCoverage {
                function_name: "myFunction".to_string(),
                ranges: vec![CoverageRange {
                    start_offset: 0,
                    end_offset: 100,
                    count: 5,
                }],
                is_block_coverage: true,
            });

            let report = page.take_coverage().unwrap();
            assert_eq!(report.scripts[0].functions.len(), 1);
            assert_eq!(report.scripts[0].functions[0].function_name, "myFunction");
        }

        #[test]
        fn test_clear_mock_coverage_clears_functions() {
            let mut page = Page::new(800, 600);
            page.start_coverage().unwrap();

            page.add_mock_coverage(FunctionCoverage {
                function_name: "fn1".to_string(),
                ranges: vec![],
                is_block_coverage: false,
            });
            page.add_mock_coverage(FunctionCoverage {
                function_name: "fn2".to_string(),
                ranges: vec![],
                is_block_coverage: false,
            });

            let report = page.take_coverage().unwrap();
            assert_eq!(report.scripts[0].functions.len(), 2);

            page.clear_mock_coverage();

            let report2 = page.take_coverage().unwrap();
            assert!(report2.scripts[0].functions.is_empty());
        }

        // =====================================================================
        // Browser methods
        // =====================================================================

        #[test]
        fn test_browser_launch_returns_ok() {
            let config = BrowserConfig::default();
            let result = Browser::launch(config);
            assert!(result.is_ok());
        }

        #[test]
        fn test_browser_config_accessor() {
            let config = BrowserConfig::default()
                .with_viewport(1920, 1080)
                .with_headless(false);
            let browser = Browser::launch(config).unwrap();
            let cfg = browser.config();
            assert_eq!(cfg.viewport_width, 1920);
            assert_eq!(cfg.viewport_height, 1080);
            assert!(!cfg.headless);
        }

        #[test]
        fn test_browser_new_page_returns_page_with_config_dimensions() {
            let config = BrowserConfig::default().with_viewport(1280, 720);
            let browser = Browser::launch(config).unwrap();
            let page = browser.new_page().unwrap();
            assert_eq!(page.width, 1280);
            assert_eq!(page.height, 720);
        }

        #[test]
        fn test_browser_new_page_with_tracing_enabled() {
            let tracing = RenacerTracingConfig::new("test-service");
            let config = BrowserConfig::default().with_tracing(tracing);
            let browser = Browser::launch(config).unwrap();
            let page = browser.new_page().unwrap();
            assert!(page.is_tracing_enabled());
        }

        #[test]
        fn test_browser_new_page_with_tracing_disabled() {
            let tracing = RenacerTracingConfig::disabled();
            let config = BrowserConfig::default().with_tracing(tracing);
            let browser = Browser::launch(config).unwrap();
            let page = browser.new_page().unwrap();
            assert!(!page.is_tracing_enabled());
        }

        #[test]
        fn test_browser_new_page_without_tracing_config() {
            let config = BrowserConfig::default();
            let browser = Browser::launch(config).unwrap();
            let page = browser.new_page().unwrap();
            assert!(!page.is_tracing_enabled());
        }

        // =====================================================================
        // Page basic methods
        // =====================================================================

        #[test]
        fn test_page_goto_returns_ok() {
            let mut page = Page::new(800, 600);
            let result = page.goto("http://example.com");
            assert!(result.is_ok());
        }

        #[test]
        fn test_page_goto_updates_url() {
            let mut page = Page::new(800, 600);
            page.goto("http://new-url.com").unwrap();
            assert_eq!(page.current_url(), "http://new-url.com");
            assert_eq!(page.url, "http://new-url.com");
        }

        #[test]
        fn test_page_wait_for_wasm_ready_returns_ok() {
            let mut page = Page::new(800, 600);
            let result = page.wait_for_wasm_ready();
            assert!(result.is_ok());
        }

        #[test]
        fn test_page_wait_for_wasm_ready_sets_flag() {
            let mut page = Page::new(800, 600);
            assert!(!page.wasm_ready);
            page.wait_for_wasm_ready().unwrap();
            assert!(page.wasm_ready);
            assert!(page.is_wasm_ready());
        }

        #[test]
        fn test_page_eval_wasm_returns_error() {
            let page = Page::new(800, 600);
            let result: Result<i32, _> = page.eval_wasm("expression");
            assert!(result.is_err());
            let err = result.unwrap_err();
            let err_str = format!("{}", err);
            assert!(err_str.contains("Browser feature not enabled"));
        }

        #[test]
        fn test_page_touch_returns_ok() {
            let page = Page::new(800, 600);
            let touch = crate::Touch {
                x: 100.0,
                y: 100.0,
                action: crate::TouchAction::Tap,
            };
            let result = page.touch(touch);
            assert!(result.is_ok());
        }

        #[test]
        fn test_page_screenshot_returns_empty_bytes() {
            let page = Page::new(800, 600);
            let result = page.screenshot();
            assert!(result.is_ok());
            let bytes = result.unwrap();
            assert!(bytes.is_empty());
        }

        #[test]
        fn test_page_current_url_returns_url() {
            let page = Page::new(800, 600);
            assert_eq!(page.current_url(), "about:blank");
        }

        #[test]
        fn test_page_is_wasm_ready_returns_bool() {
            let page = Page::new(800, 600);
            assert!(!page.is_wasm_ready());
        }

        // =====================================================================
        // Integration scenarios
        // =====================================================================

        #[test]
        fn test_full_mock_page_workflow() {
            // Create browser with tracing
            let tracing = RenacerTracingConfig::new("integration-test");
            let config = BrowserConfig::default()
                .with_viewport(1920, 1080)
                .with_tracing(tracing);
            let browser = Browser::launch(config).unwrap();
            let mut page = browser.new_page().unwrap();

            // Navigate
            page.goto("http://localhost:8080/app").unwrap();
            assert_eq!(page.current_url(), "http://localhost:8080/app");

            // Wait for WASM
            page.wait_for_wasm_ready().unwrap();
            assert!(page.is_wasm_ready());

            // Enable console capture
            page.enable_console_capture().unwrap();
            assert!(page.is_console_capture_enabled());

            // Add console messages
            page.add_console_message(BrowserConsoleMessage {
                level: BrowserConsoleLevel::Log,
                text: "App started".to_string(),
                timestamp: 1000,
                source: Some("main.js".to_string()),
                line: Some(10),
            });

            // Start coverage
            page.start_coverage().unwrap();
            assert!(page.is_coverage_enabled());

            // Add coverage data
            page.add_mock_coverage(FunctionCoverage {
                function_name: "init".to_string(),
                ranges: vec![CoverageRange {
                    start_offset: 0,
                    end_offset: 100,
                    count: 1,
                }],
                is_block_coverage: true,
            });

            // Start a trace span
            let mut span = page.start_span("test-operation", "test").unwrap();
            span.add_attribute("key", "value");
            span.end();
            page.record_span(span);

            // Record console in trace
            page.record_trace_console("Trace console message");

            // Get trace JSON
            let trace_json = page.export_trace_json().unwrap();
            assert!(trace_json.is_some());
            let json_str = trace_json.unwrap();
            assert!(json_str.contains("traceEvents"));

            // Get coverage report
            let report = page.stop_coverage().unwrap();
            assert!(!page.is_coverage_enabled());
            assert!(!report.scripts.is_empty());

            // Verify console messages
            let messages = page.console_messages();
            assert_eq!(messages.len(), 1);

            // Take screenshot
            let screenshot = page.screenshot().unwrap();
            assert!(screenshot.is_empty());
        }

        #[test]
        fn test_coverage_report_script_id() {
            let mut page = Page::new(800, 600);
            page.start_coverage().unwrap();
            let report = page.take_coverage().unwrap();
            assert_eq!(report.scripts[0].script_id, "mock-script-1");
        }

        #[test]
        fn test_coverage_report_empty_functions() {
            let mut page = Page::new(800, 600);
            page.start_coverage().unwrap();
            let report = page.take_coverage().unwrap();
            assert!(report.scripts[0].functions.is_empty());
        }

        #[test]
        fn test_multiple_console_message_sources() {
            let page = Page::new(800, 600);

            page.add_console_message(BrowserConsoleMessage {
                level: BrowserConsoleLevel::Log,
                text: "from main".to_string(),
                timestamp: 1,
                source: Some("main.js".to_string()),
                line: Some(10),
            });
            page.add_console_message(BrowserConsoleMessage {
                level: BrowserConsoleLevel::Warning,
                text: "from utils".to_string(),
                timestamp: 2,
                source: Some("utils.js".to_string()),
                line: Some(20),
            });
            page.add_console_message(BrowserConsoleMessage {
                level: BrowserConsoleLevel::Error,
                text: "no source".to_string(),
                timestamp: 3,
                source: None,
                line: None,
            });

            let messages = page.console_messages();
            assert_eq!(messages.len(), 3);
            assert_eq!(messages[0].source, Some("main.js".to_string()));
            assert_eq!(messages[1].source, Some("utils.js".to_string()));
            assert!(messages[2].source.is_none());
        }

        #[test]
        fn test_wait_for_console_by_line() {
            let page = Page::new(800, 600);
            page.add_console_message(BrowserConsoleMessage {
                level: BrowserConsoleLevel::Error,
                text: "error on line 42".to_string(),
                timestamp: 0,
                source: Some("app.js".to_string()),
                line: Some(42),
            });

            let result = page.wait_for_console(|m| m.line == Some(42), 1000);
            assert!(result.is_ok());
            assert_eq!(result.unwrap().line, Some(42));
        }

        #[test]
        fn test_tracing_span_with_multiple_attributes() {
            let collector = TraceCollector::new("test");
            let mut page = Page::new_with_tracing(800, 600, Some(collector));

            let mut span = page.start_span("complex-span", "http").unwrap();
            span.add_attribute("method", "GET");
            span.add_attribute("url", "http://api.example.com/data");
            span.add_attribute("status", "200");
            span.end();
            page.record_span(span);

            let trace = page.export_chrome_trace().unwrap();
            assert_eq!(trace.trace_events.len(), 1);
            assert_eq!(trace.trace_events[0].name, "complex-span");
        }

        #[test]
        fn test_coverage_with_multiple_ranges() {
            let mut page = Page::new(800, 600);
            page.start_coverage().unwrap();

            page.add_mock_coverage(FunctionCoverage {
                function_name: "complex_function".to_string(),
                ranges: vec![
                    CoverageRange {
                        start_offset: 0,
                        end_offset: 50,
                        count: 10,
                    },
                    CoverageRange {
                        start_offset: 50,
                        end_offset: 100,
                        count: 5,
                    },
                    CoverageRange {
                        start_offset: 100,
                        end_offset: 150,
                        count: 0, // Uncovered branch
                    },
                ],
                is_block_coverage: true,
            });

            let report = page.take_coverage().unwrap();
            let func = &report.scripts[0].functions[0];
            assert_eq!(func.ranges.len(), 3);
            assert_eq!(func.ranges[2].count, 0);
        }

        #[test]
        fn test_browser_debug_format() {
            let config = BrowserConfig::default();
            let browser = Browser::launch(config).unwrap();
            let debug_str = format!("{:?}", browser);
            assert!(debug_str.contains("Browser"));
            assert!(debug_str.contains("config"));
        }

        #[test]
        fn test_page_debug_format_comprehensive() {
            let mut page = Page::new(1024, 768);
            page.goto("http://test.com").unwrap();
            page.enable_console_capture().unwrap();
            page.start_coverage().unwrap();

            let debug_str = format!("{:?}", page);
            assert!(debug_str.contains("Page"));
            assert!(debug_str.contains("1024"));
            assert!(debug_str.contains("768"));
        }

        #[test]
        fn test_start_coverage_then_start_again() {
            let mut page = Page::new(800, 600);
            page.start_coverage().unwrap();
            assert!(page.is_coverage_enabled());

            // Starting again should still work (override)
            let config = CoverageConfig {
                call_count: false,
                detailed: false,
                allow_triggered_updates: true,
            };
            page.start_coverage_with_config(config).unwrap();
            assert!(page.is_coverage_enabled());
        }

        #[test]
        fn test_page_touch_all_variants() {
            let page = Page::new(800, 600);

            // Tap
            assert!(page
                .touch(crate::Touch {
                    x: 0.0,
                    y: 0.0,
                    action: crate::TouchAction::Tap
                })
                .is_ok());

            // Swipe
            assert!(page
                .touch(crate::Touch {
                    x: 0.0,
                    y: 0.0,
                    action: crate::TouchAction::Swipe {
                        end_x: 100.0,
                        end_y: 100.0,
                        duration_ms: 200
                    }
                })
                .is_ok());

            // Hold
            assert!(page
                .touch(crate::Touch {
                    x: 50.0,
                    y: 50.0,
                    action: crate::TouchAction::Hold { duration_ms: 1000 }
                })
                .is_ok());
        }

        #[test]
        fn test_console_capture_enabled_after_inject() {
            let mut page = Page::new(800, 600);
            assert!(!page.is_console_capture_enabled());
            page.inject_console_capture().unwrap();
            assert!(page.is_console_capture_enabled());

            // Inject again should still be enabled
            page.inject_console_capture().unwrap();
            assert!(page.is_console_capture_enabled());
        }

        #[test]
        fn test_console_capture_enabled_after_enable() {
            let mut page = Page::new(800, 600);
            assert!(!page.is_console_capture_enabled());
            page.enable_console_capture().unwrap();
            assert!(page.is_console_capture_enabled());

            // Enable again should still be enabled
            page.enable_console_capture().unwrap();
            assert!(page.is_console_capture_enabled());
        }

        #[test]
        fn test_coverage_timestamp_is_current_time() {
            let mut page = Page::new(800, 600);
            page.start_coverage().unwrap();

            let before = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;

            let report = page.take_coverage().unwrap();

            let after = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;

            // Timestamp should be between before and after
            assert!(report.timestamp_ms >= before);
            assert!(report.timestamp_ms <= after);
        }
    }
