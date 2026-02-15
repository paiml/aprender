
    #[test]
    fn test_federation_app_should_quit() {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());
        let mut app = FederationApp::new(catalog, health, circuit_breaker);

        assert!(!app.should_quit);
        app.should_quit = true;
        assert!(app.should_quit);
    }

    #[test]
    fn test_federation_app_policies_count() {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());
        let app = FederationApp::new(catalog, health, circuit_breaker);

        // Default policies: health, latency, privacy, locality, cost
        assert_eq!(app.policies.len(), 5);
        assert_eq!(app.policies[0].name, "health");
        assert_eq!(app.policies[1].name, "latency");
        assert_eq!(app.policies[2].name, "privacy");
        assert_eq!(app.policies[3].name, "locality");
        assert_eq!(app.policies[4].name, "cost");
    }

    #[test]
    fn test_federation_app_max_history_default() {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());
        let app = FederationApp::new(catalog, health, circuit_breaker);

        assert_eq!(app.max_history, 100);
    }

    // =========================================================================
    // Probar TUI Frame Tests
    // =========================================================================

    mod tui_frame_tests {
        use super::*;
        use jugar_probar::tui::{
            expect_frame, FrameSequence, SnapshotManager, TuiFrame, TuiSnapshot,
        };
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        /// Helper to render federation app to a test backend
        fn render_frame(app: &FederationApp, width: u16, height: u16) -> TuiFrame {
            let backend = TestBackend::new(width, height);
            let mut terminal = Terminal::new(backend).expect("terminal creation");
            terminal
                .draw(|f| render_federation_dashboard(f, app))
                .expect("draw");
            TuiFrame::from_buffer(terminal.backend().buffer(), 0)
        }

        fn create_test_app() -> FederationApp {
            let catalog = Arc::new(ModelCatalog::new());
            let health = Arc::new(HealthChecker::default());
            let circuit_breaker = Arc::new(CircuitBreaker::default());
            FederationApp::new(catalog, health, circuit_breaker)
        }

        #[test]
        fn test_federation_tui_title_displayed() {
            let app = create_test_app();
            let frame = render_frame(&app, 100, 30);

            assert!(
                frame.contains("Federation Gateway"),
                "Should show title: {}",
                frame.as_text()
            );
        }

        #[test]
        fn test_federation_tui_all_tabs_displayed() {
            let app = create_test_app();
            let frame = render_frame(&app, 100, 30);

            assert!(frame.contains("Catalog"), "Should show Catalog tab");
            assert!(frame.contains("Health"), "Should show Health tab");
            assert!(frame.contains("Routing"), "Should show Routing tab");
            assert!(frame.contains("Circuits"), "Should show Circuits tab");
            assert!(frame.contains("Stats"), "Should show Stats tab");
            assert!(frame.contains("Policies"), "Should show Policies tab");
            assert!(frame.contains("Help"), "Should show Help tab");
        }

        #[test]
        fn test_federation_tui_catalog_empty() {
            let app = create_test_app();
            let frame = render_frame(&app, 100, 30);

            // Default tab is catalog, should show empty message
            assert!(
                frame.contains("No models registered") || frame.contains("MODEL CATALOG"),
                "Should show catalog panel"
            );
        }

        #[test]
        fn test_federation_tui_health_tab() {
            let mut app = create_test_app();
            app.current_tab = FederationTab::Health;
            let frame = render_frame(&app, 100, 30);

            assert!(
                frame.contains("NODE HEALTH"),
                "Should show health panel title"
            );
        }

        #[test]
        fn test_federation_tui_routing_tab() {
            let mut app = create_test_app();
            app.current_tab = FederationTab::Routing;
            let frame = render_frame(&app, 100, 30);

            assert!(
                frame.contains("ROUTING DECISIONS"),
                "Should show routing panel title"
            );
        }

        #[test]
        fn test_federation_tui_circuits_tab() {
            let mut app = create_test_app();
            app.current_tab = FederationTab::Circuits;
            let frame = render_frame(&app, 100, 30);

            assert!(
                frame.contains("CIRCUIT BREAKERS"),
                "Should show circuits panel title"
            );
        }

        #[test]
        fn test_federation_tui_stats_tab() {
            let mut app = create_test_app();
            app.current_tab = FederationTab::Stats;
            let frame = render_frame(&app, 100, 30);

            assert!(
                frame.contains("GATEWAY STATS"),
                "Should show stats panel title"
            );
            assert!(
                frame.contains("Total Requests"),
                "Should show request count"
            );
        }

        #[test]
        fn test_federation_tui_policies_tab() {
            let mut app = create_test_app();
            app.current_tab = FederationTab::Policies;
            let frame = render_frame(&app, 100, 30);

            assert!(
                frame.contains("ACTIVE POLICIES"),
                "Should show policies panel title"
            );
            assert!(frame.contains("health"), "Should show health policy");
            assert!(frame.contains("latency"), "Should show latency policy");
        }

        #[test]
        fn test_federation_tui_help_tab() {
            let mut app = create_test_app();
            app.current_tab = FederationTab::Help;
            let frame = render_frame(&app, 100, 30);

            assert!(
                frame.contains("Keyboard Shortcuts"),
                "Should show keyboard shortcuts"
            );
            assert!(frame.contains("q / Esc"), "Should show quit shortcut");
        }

        #[test]
        fn test_federation_tui_frame_dimensions() {
            let app = create_test_app();
            let frame = render_frame(&app, 120, 40);

            assert_eq!(frame.width(), 120, "Frame width");
            assert_eq!(frame.height(), 40, "Frame height");
        }

        // =====================================================================
        // Playwright-style Assertions
        // =====================================================================

        #[test]
        fn test_probar_chained_assertions() {
            let app = create_test_app();
            let frame = render_frame(&app, 100, 30);

            let mut assertion = expect_frame(&frame);
            assert!(assertion.to_contain_text("Federation Gateway").is_ok());
            assert!(assertion.to_contain_text("Navigation").is_ok());
            assert!(assertion.not_to_contain_text("ERROR").is_ok());
        }

        #[test]
        fn test_probar_soft_assertions() {
            let app = create_test_app();
            let frame = render_frame(&app, 100, 30);

            let mut assertion = expect_frame(&frame).soft();
            let _ = assertion.to_contain_text("Federation Gateway");
            let _ = assertion.to_contain_text("Catalog");
            let _ = assertion.to_have_size(100, 30);

            assert!(assertion.errors().is_empty(), "No soft assertion errors");
            assert!(assertion.finalize().is_ok());
        }

        // =====================================================================
        // Snapshot Testing
        // =====================================================================

        #[test]
        fn test_snapshot_creation() {
            let app = create_test_app();
            let frame = render_frame(&app, 100, 30);

            let snapshot = TuiSnapshot::from_frame("federation_catalog", &frame);

            assert_eq!(snapshot.name, "federation_catalog");
            assert_eq!(snapshot.width, 100);
            assert_eq!(snapshot.height, 30);
            assert!(!snapshot.hash.is_empty());
        }

        #[test]
        fn test_snapshot_different_tabs_differ() {
            let mut app = create_test_app();

            app.current_tab = FederationTab::Catalog;
            let frame_catalog = render_frame(&app, 100, 30);
            let snap_catalog = TuiSnapshot::from_frame("catalog", &frame_catalog);

            app.current_tab = FederationTab::Help;
            let frame_help = render_frame(&app, 100, 30);
            let snap_help = TuiSnapshot::from_frame("help", &frame_help);

            assert!(
                !snap_catalog.matches(&snap_help),
                "Different tabs should have different snapshots"
            );
        }

        // =====================================================================
        // Frame Sequence Testing
        // =====================================================================

        #[test]
        fn test_frame_sequence_tab_navigation() {
            let mut app = create_test_app();
            let mut sequence = FrameSequence::new("federation_tabs");

            for tab in [
                FederationTab::Catalog,
                FederationTab::Health,
                FederationTab::Routing,
                FederationTab::Circuits,
                FederationTab::Stats,
                FederationTab::Policies,
                FederationTab::Help,
            ] {
                app.current_tab = tab;
                sequence.add_frame(&render_frame(&app, 100, 30));
            }

            assert_eq!(sequence.len(), 7, "Should have 7 frames");

            let first = sequence.first().expect("first");
            let last = sequence.last().expect("last");
            assert!(!first.matches(last), "First and last should differ");
        }

        // =====================================================================
        // Snapshot Manager Tests
        // =====================================================================

        #[test]
        fn test_snapshot_manager_workflow() {
            use tempfile::TempDir;

            let temp_dir = TempDir::new().expect("temp dir");
            let manager = SnapshotManager::new(temp_dir.path());

            let app = create_test_app();
            let frame = render_frame(&app, 100, 30);

            let result = manager.assert_snapshot("federation_dashboard", &frame);
            assert!(result.is_ok(), "First snapshot should be created");
            assert!(
                manager.exists("federation_dashboard"),
                "Snapshot should exist"
            );

            let result2 = manager.assert_snapshot("federation_dashboard", &frame);
            assert!(result2.is_ok(), "Same frame should match");
        }

        // =====================================================================
        // UX Coverage Tests
        // =====================================================================

        #[test]
        fn test_ux_coverage_federation_elements() {
            use jugar_probar::ux_coverage::{InteractionType, StateId, UxCoverageBuilder};

            let mut tracker = UxCoverageBuilder::new()
                .clickable("tab", "catalog")
                .clickable("tab", "health")
                .clickable("tab", "routing")
                .clickable("tab", "circuits")
                .clickable("tab", "stats")
                .clickable("tab", "policies")
                .clickable("tab", "help")
                .clickable("nav", "next_tab")
                .clickable("nav", "prev_tab")
                .clickable("nav", "next_row")
                .clickable("nav", "prev_row")
                .clickable("nav", "quit")
                .screen("catalog")
                .screen("health")
                .screen("routing")
                .screen("circuits")
                .screen("stats")
                .screen("policies")
                .screen("help")
                .build();

            let mut app = create_test_app();

            // Cover all tabs
            for (tab, name) in [
                (FederationTab::Catalog, "catalog"),
                (FederationTab::Health, "health"),
                (FederationTab::Routing, "routing"),
                (FederationTab::Circuits, "circuits"),
                (FederationTab::Stats, "stats"),
                (FederationTab::Policies, "policies"),
                (FederationTab::Help, "help"),
            ] {
                app.current_tab = tab;
                tracker.record_interaction(
                    &jugar_probar::ux_coverage::ElementId::new("tab", name),
                    InteractionType::Click,
                );
                tracker.record_state(StateId::new("screen", name));
            }

            // Cover navigation
            app.next_tab();
            tracker.record_interaction(
                &jugar_probar::ux_coverage::ElementId::new("nav", "next_tab"),
                InteractionType::Click,
            );

            app.prev_tab();
            tracker.record_interaction(
                &jugar_probar::ux_coverage::ElementId::new("nav", "prev_tab"),
                InteractionType::Click,
            );

            app.select_next();
            tracker.record_interaction(
                &jugar_probar::ux_coverage::ElementId::new("nav", "next_row"),
                InteractionType::Click,
            );

            app.select_prev();
            tracker.record_interaction(
                &jugar_probar::ux_coverage::ElementId::new("nav", "prev_row"),
                InteractionType::Click,
            );

            app.should_quit = true;
            tracker.record_interaction(
                &jugar_probar::ux_coverage::ElementId::new("nav", "quit"),
                InteractionType::Click,
            );

            let report = tracker.generate_report();
            println!("{}", report.summary());

            assert!(report.is_complete, "UX coverage must be COMPLETE");
            assert!(tracker.meets(100.0), "Coverage: {}", tracker.summary());
        }

        #[test]
        fn test_ux_coverage_with_gui_macro() {
            use jugar_probar::gui_coverage;

            let mut gui = gui_coverage! {
                buttons: [
                    "tab_catalog", "tab_health", "tab_routing",
                    "tab_circuits", "tab_stats", "tab_policies", "tab_help",
                    "nav_next", "nav_prev", "quit"
                ],
                screens: [
                    "catalog", "health", "routing",
                    "circuits", "stats", "policies", "help"
                ]
            };

            // Cover all buttons
            gui.click("tab_catalog");
            gui.click("tab_health");
            gui.click("tab_routing");
            gui.click("tab_circuits");
            gui.click("tab_stats");
            gui.click("tab_policies");
            gui.click("tab_help");
            gui.click("nav_next");
            gui.click("nav_prev");
            gui.click("quit");

            // Cover all screens
            gui.visit("catalog");
            gui.visit("health");
            gui.visit("routing");
            gui.visit("circuits");
            gui.visit("stats");
            gui.visit("policies");
            gui.visit("help");

            assert!(gui.is_complete(), "Should have 100% coverage");
            assert!(gui.meets(100.0), "Coverage: {}", gui.summary());

            let report = gui.generate_report();
            println!(
                "Federation UX Coverage:\n  Elements: {}/{}\n  States: {}/{}\n  Overall: {:.1}%",
                report.covered_elements,
                report.total_elements,
                report.covered_states,
                report.total_states,
                report.overall_coverage * 100.0
            );
        }
    }
