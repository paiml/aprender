
    #[test]
    fn test_tab_titles() {
        let titles = Tab::titles();
        assert_eq!(titles.len(), 4);
        assert!(titles[0].contains("Overview"));
        assert!(titles[1].contains("Tensors"));
        assert!(titles[2].contains("Stats"));
        assert!(titles[3].contains("Help"));
    }

    #[test]
    fn test_tab_index() {
        assert_eq!(Tab::Overview.index(), 0);
        assert_eq!(Tab::Tensors.index(), 1);
        assert_eq!(Tab::Stats.index(), 2);
        assert_eq!(Tab::Help.index(), 3);
    }

    #[test]
    fn test_app_new_no_file() {
        let app = App::new(None);
        assert!(app.file_path.is_none());
        assert!(app.reader.is_none());
        assert!(app.tensors.is_empty());
        assert!(!app.should_quit);
    }

    #[test]
    fn test_app_next_tab() {
        let mut app = App::new(None);
        assert_eq!(app.current_tab, Tab::Overview);
        app.next_tab();
        assert_eq!(app.current_tab, Tab::Tensors);
        app.next_tab();
        assert_eq!(app.current_tab, Tab::Stats);
        app.next_tab();
        assert_eq!(app.current_tab, Tab::Help);
        app.next_tab();
        assert_eq!(app.current_tab, Tab::Overview);
    }

    #[test]
    fn test_app_prev_tab() {
        let mut app = App::new(None);
        assert_eq!(app.current_tab, Tab::Overview);
        app.prev_tab();
        assert_eq!(app.current_tab, Tab::Help);
        app.prev_tab();
        assert_eq!(app.current_tab, Tab::Stats);
        app.prev_tab();
        assert_eq!(app.current_tab, Tab::Tensors);
        app.prev_tab();
        assert_eq!(app.current_tab, Tab::Overview);
    }

    #[test]
    fn test_truncate_name_short() {
        assert_eq!(truncate_name("short", 10), "short");
    }

    #[test]
    fn test_truncate_name_long() {
        assert_eq!(
            truncate_name("this_is_a_very_long_tensor_name", 20),
            "this_is_a_very_lo..."
        );
    }

    #[test]
    fn test_tensor_selection_empty() {
        let mut app = App::new(None);
        app.select_next_tensor();
        assert!(app.tensor_list_state.selected().is_none());
        app.select_prev_tensor();
        assert!(app.tensor_list_state.selected().is_none());
    }

    // TUI frame capture tests using ratatui's TestBackend
    // Converts to probar TuiFrame for assertions
    mod tui_frame_tests {
        use super::*;
        use jugar_probar::tui::{
            expect_frame, FrameSequence, SnapshotManager, TuiFrame, TuiSnapshot,
        };
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        /// Helper to render app to a test backend and capture frame
        fn render_frame(app: &mut App, width: u16, height: u16) -> TuiFrame {
            let backend = TestBackend::new(width, height);
            let mut terminal = Terminal::new(backend).unwrap();
            terminal.draw(|f| ui(f, app)).unwrap();
            // Convert ratatui buffer to probar TuiFrame
            TuiFrame::from_buffer(terminal.backend().buffer(), 0)
        }

        #[test]
        fn test_tui_overview_no_file() {
            let mut app = App::new(None);
            let frame = render_frame(&mut app, 80, 24);

            // Should show "No model loaded" message
            assert!(
                frame.contains("No model loaded"),
                "Frame should show no model message:\n{}",
                frame.as_text()
            );
        }

        #[test]
        fn test_tui_tabs_displayed() {
            let mut app = App::new(None);
            let frame = render_frame(&mut app, 80, 24);

            // Should show all tab titles
            assert!(frame.contains("Overview"), "Should show Overview tab");
            assert!(frame.contains("Tensors"), "Should show Tensors tab");
            assert!(frame.contains("Stats"), "Should show Stats tab");
            assert!(frame.contains("Help"), "Should show Help tab");
        }

        #[test]
        fn test_tui_title_bar() {
            let mut app = App::new(None);
            let frame = render_frame(&mut app, 80, 24);

            // Should show title
            assert!(
                frame.contains("APR Model Inspector"),
                "Should show title bar"
            );
        }

        #[test]
        fn test_tui_help_view() {
            let mut app = App::new(None);
            app.current_tab = Tab::Help;
            let frame = render_frame(&mut app, 80, 24);

            // Should show help content
            assert!(
                frame.contains("Keyboard Shortcuts"),
                "Help should show keyboard shortcuts"
            );
            assert!(frame.contains("q / Esc"), "Help should show quit shortcut");
        }

        #[test]
        fn test_tui_stats_empty() {
            let mut app = App::new(None);
            app.current_tab = Tab::Stats;
            let frame = render_frame(&mut app, 80, 24);

            // Should show no data message
            assert!(
                frame.contains("No tensor data"),
                "Stats should show no data message"
            );
        }

        #[test]
        fn test_tui_tensors_empty() {
            let mut app = App::new(None);
            app.current_tab = Tab::Tensors;
            let frame = render_frame(&mut app, 80, 24);

            // Should show tensor browser (even if empty)
            assert!(
                frame.contains("Tensor") || frame.contains("navigate"),
                "Tensors view should be shown"
            );
        }

        #[test]
        fn test_tui_frame_dimensions() {
            let mut app = App::new(None);
            let frame = render_frame(&mut app, 100, 30);

            assert_eq!(frame.width(), 100, "Frame width should match");
            assert_eq!(frame.height(), 30, "Frame height should match");
        }

        // =========================================================================
        // ADVANCED PROBAR TESTS: Playwright-style assertions
        // =========================================================================

        #[test]
        fn test_probar_chained_assertions() {
            let mut app = App::new(None);
            let frame = render_frame(&mut app, 80, 24);

            // Playwright-style chained assertions
            let mut assertion = expect_frame(&frame);
            let r1 = assertion.to_contain_text("APR Model Inspector");
            assert!(r1.is_ok(), "Should contain title");

            let r2 = assertion.to_contain_text("Overview");
            assert!(r2.is_ok(), "Should contain Overview");

            let r3 = assertion.to_contain_text("Navigation");
            assert!(r3.is_ok(), "Should contain Navigation");

            let r4 = assertion.not_to_contain_text("ERROR");
            assert!(r4.is_ok(), "Should not contain ERROR");
        }

        #[test]
        fn test_probar_soft_assertions() {
            let mut app = App::new(None);
            let frame = render_frame(&mut app, 80, 24);

            // Soft assertions collect all failures instead of failing fast
            let mut assertion = expect_frame(&frame).soft();
            let _ = assertion.to_contain_text("APR Model Inspector");
            let _ = assertion.to_contain_text("Overview");
            let _ = assertion.to_contain_text("Help");
            let _ = assertion.to_have_size(80, 24);

            assert!(assertion.errors().is_empty(), "No soft assertion errors");
            assert!(assertion.finalize().is_ok(), "All soft assertions passed");
        }

        #[test]
        fn test_probar_regex_matching() {
            let mut app = App::new(None);
            app.current_tab = Tab::Help;
            let frame = render_frame(&mut app, 80, 24);

            // Regex pattern matching for dynamic content
            let mut assertion = expect_frame(&frame);
            let result = assertion.to_match(r"Version: \d+\.\d+\.\d+");

            assert!(result.is_ok(), "Should match version pattern");
        }

        // =========================================================================
        // SNAPSHOT TESTING: Golden file comparisons
        // =========================================================================

        #[test]
        fn test_snapshot_creation_and_matching() {
            let mut app = App::new(None);
            let frame = render_frame(&mut app, 80, 24);

            // Create snapshot from frame
            let snapshot = TuiSnapshot::from_frame("overview_no_file", &frame);

            assert_eq!(snapshot.name, "overview_no_file");
            assert_eq!(snapshot.width, 80);
            assert_eq!(snapshot.height, 24);
            assert!(!snapshot.hash.is_empty(), "Hash should be computed");

            // Snapshots with same content should match
            let frame2 = render_frame(&mut app, 80, 24);
            let snapshot2 = TuiSnapshot::from_frame("overview_no_file_2", &frame2);

            assert!(
                snapshot.matches(&snapshot2),
                "Identical frames should have matching snapshots"
            );
        }

        #[test]
        fn test_snapshot_with_metadata() {
            let mut app = App::new(None);
            let frame = render_frame(&mut app, 80, 24);

            let snapshot = TuiSnapshot::from_frame("test", &frame)
                .with_metadata("test_name", "overview_no_file")
                .with_metadata("tab", "overview")
                .with_metadata("model_loaded", "false");

            assert_eq!(
                snapshot.metadata.get("test_name"),
                Some(&"overview_no_file".to_string())
            );
            assert_eq!(snapshot.metadata.get("tab"), Some(&"overview".to_string()));
        }

        #[test]
        fn test_snapshot_different_tabs_differ() {
            let mut app = App::new(None);

            // Overview tab
            let frame_overview = render_frame(&mut app, 80, 24);
            let snap_overview = TuiSnapshot::from_frame("overview", &frame_overview);

            // Help tab
            app.current_tab = Tab::Help;
            let frame_help = render_frame(&mut app, 80, 24);
            let snap_help = TuiSnapshot::from_frame("help", &frame_help);

            // Different tabs should NOT match
            assert!(
                !snap_overview.matches(&snap_help),
                "Different tabs should have different snapshots"
            );
        }

        // =========================================================================
        // FRAME SEQUENCE: Animation / state transition testing
        // =========================================================================

        #[test]
        fn test_frame_sequence_tab_navigation() {
            let mut app = App::new(None);
            let mut sequence = FrameSequence::new("tab_navigation");

            // Capture frame for each tab (simulating user navigation)
            app.current_tab = Tab::Overview;
            sequence.add_frame(&render_frame(&mut app, 80, 24));

            app.current_tab = Tab::Tensors;
            sequence.add_frame(&render_frame(&mut app, 80, 24));

            app.current_tab = Tab::Stats;
            sequence.add_frame(&render_frame(&mut app, 80, 24));

            app.current_tab = Tab::Help;
            sequence.add_frame(&render_frame(&mut app, 80, 24));

            assert_eq!(sequence.len(), 4, "Should have 4 frames");
            assert!(!sequence.is_empty());

            // First and last frames should differ
            let first = sequence.first().unwrap();
            let last = sequence.last().unwrap();
            assert!(!first.matches(last), "First and last frames should differ");
        }

        #[test]
        fn test_frame_sequence_diff_detection() {
            let mut app = App::new(None);

            // Create two sequences with mostly same content
            let mut seq1 = FrameSequence::new("seq1");
            let mut seq2 = FrameSequence::new("seq2");

            // Same: Overview
            app.current_tab = Tab::Overview;
            seq1.add_frame(&render_frame(&mut app, 80, 24));
            seq2.add_frame(&render_frame(&mut app, 80, 24));

            // Different: Tensors vs Stats
            app.current_tab = Tab::Tensors;
            seq1.add_frame(&render_frame(&mut app, 80, 24));
            app.current_tab = Tab::Stats;
            seq2.add_frame(&render_frame(&mut app, 80, 24));

            // Same: Help
            app.current_tab = Tab::Help;
            seq1.add_frame(&render_frame(&mut app, 80, 24));
            seq2.add_frame(&render_frame(&mut app, 80, 24));

            // Find differing frames
            let diffs = seq1.diff_frames(&seq2);
            assert_eq!(diffs, vec![1], "Only frame index 1 should differ");
        }

        // =========================================================================
        // SNAPSHOT MANAGER: Persistent golden file testing
        // =========================================================================

        #[test]
        fn test_snapshot_manager_workflow() {
            use tempfile::TempDir;

            let temp_dir = TempDir::new().unwrap();
            let manager = SnapshotManager::new(temp_dir.path());

            let mut app = App::new(None);
            let frame = render_frame(&mut app, 80, 24);

            // First run: creates snapshot
            let result = manager.assert_snapshot("tui_overview", &frame);
            assert!(result.is_ok(), "First snapshot should be created");
            assert!(manager.exists("tui_overview"), "Snapshot file should exist");

            // Second run: matches existing
            let result2 = manager.assert_snapshot("tui_overview", &frame);
            assert!(result2.is_ok(), "Same frame should match snapshot");

            // List snapshots
            let list = manager.list().unwrap();
            assert!(list.contains(&"tui_overview".to_string()));
        }

        #[test]
        fn test_snapshot_manager_detects_changes() {
            use tempfile::TempDir;

            let temp_dir = TempDir::new().unwrap();
            let manager = SnapshotManager::new(temp_dir.path());

            let mut app = App::new(None);

            // Create snapshot with Overview tab
            app.current_tab = Tab::Overview;
            let frame1 = render_frame(&mut app, 80, 24);
            manager.assert_snapshot("test_snap", &frame1).unwrap();

            // Try to assert with Help tab (should fail)
            app.current_tab = Tab::Help;
            let frame2 = render_frame(&mut app, 80, 24);
            let result = manager.assert_snapshot("test_snap", &frame2);

            assert!(result.is_err(), "Changed frame should fail snapshot");
        }

        #[test]
        fn test_snapshot_manager_update_mode() {
            use tempfile::TempDir;

            let temp_dir = TempDir::new().unwrap();
            let manager = SnapshotManager::new(temp_dir.path()).with_update_mode(true);

            let mut app = App::new(None);

            // Create initial snapshot
            app.current_tab = Tab::Overview;
            let frame1 = render_frame(&mut app, 80, 24);
            manager.assert_snapshot("updatable", &frame1).unwrap();

            // Update with different content (update mode allows this)
            app.current_tab = Tab::Help;
            let frame2 = render_frame(&mut app, 80, 24);
            let result = manager.assert_snapshot("updatable", &frame2);

            assert!(result.is_ok(), "Update mode should allow changes");

            // Verify snapshot was updated
            let loaded = manager.load("updatable").unwrap();
            assert!(
                loaded.content.iter().any(|l| l.contains("Keyboard")),
                "Snapshot should now contain Help content"
            );
        }

        // =========================================================================
        // PIXEL-LEVEL LINE ASSERTIONS
        // =========================================================================

        #[test]
        fn test_line_level_assertions() {
            let mut app = App::new(None);
            app.current_tab = Tab::Help;
            let frame = render_frame(&mut app, 80, 24);

            let mut assertion = expect_frame(&frame);
            let result = assertion.to_contain_text("Keyboard Shortcuts");
            assert!(result.is_ok());
        }

        #[test]
        fn test_frame_identical_comparison() {
            let mut app = App::new(None);

            let frame1 = render_frame(&mut app, 80, 24);
            let frame2 = render_frame(&mut app, 80, 24);

            let mut assertion = expect_frame(&frame1);
            let result = assertion.to_be_identical_to(&frame2);
            assert!(result.is_ok(), "Same state should produce identical frames");
        }

        #[test]
        fn test_frame_non_identical_detection() {
            let mut app = App::new(None);

            app.current_tab = Tab::Overview;
            let frame1 = render_frame(&mut app, 80, 24);

            app.current_tab = Tab::Help;
            let frame2 = render_frame(&mut app, 80, 24);

            let mut assertion = expect_frame(&frame1);
            let result = assertion.to_be_identical_to(&frame2);
            assert!(result.is_err(), "Different tabs should not be identical");
        }
    }
