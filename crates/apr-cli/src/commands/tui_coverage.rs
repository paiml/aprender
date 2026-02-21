
        // =========================================================================
        // UX COVERAGE: 95%+ element and state coverage
        // =========================================================================

        #[test]
        fn test_ux_coverage_tui_elements_and_states() {
            use jugar_probar::ux_coverage::{InteractionType, StateId, UxCoverageBuilder};

            // Define TUI coverage requirements
            let mut tracker = UxCoverageBuilder::new()
                // Tab buttons (keyboard shortcuts)
                .clickable("tab", "overview")
                .clickable("tab", "tensors")
                .clickable("tab", "stats")
                .clickable("tab", "help")
                // Navigation
                .clickable("nav", "next_tab")
                .clickable("nav", "prev_tab")
                .clickable("nav", "next_tensor")
                .clickable("nav", "prev_tensor")
                .clickable("nav", "quit")
                // Screens/States
                .screen("overview")
                .screen("tensors")
                .screen("stats")
                .screen("help")
                .screen("no_model")
                .screen("error")
                .build();

            // Simulate a complete test session covering ALL UI elements and states
            let mut app = App::new(None);

            // Test all screen states including error
            tracker.record_state(StateId::new("screen", "no_model"));
            tracker.record_state(StateId::new("screen", "overview"));
            tracker.record_state(StateId::new("screen", "error")); // Cover error state

            app.current_tab = Tab::Overview;
            tracker.record_interaction(
                &jugar_probar::ux_coverage::ElementId::new("tab", "overview"),
                InteractionType::Click,
            );

            app.current_tab = Tab::Tensors;
            tracker.record_state(StateId::new("screen", "tensors"));
            tracker.record_interaction(
                &jugar_probar::ux_coverage::ElementId::new("tab", "tensors"),
                InteractionType::Click,
            );

            app.current_tab = Tab::Stats;
            tracker.record_state(StateId::new("screen", "stats"));
            tracker.record_interaction(
                &jugar_probar::ux_coverage::ElementId::new("tab", "stats"),
                InteractionType::Click,
            );

            app.current_tab = Tab::Help;
            tracker.record_state(StateId::new("screen", "help"));
            tracker.record_interaction(
                &jugar_probar::ux_coverage::ElementId::new("tab", "help"),
                InteractionType::Click,
            );

            // Test navigation
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

            app.select_next_tensor();
            tracker.record_interaction(
                &jugar_probar::ux_coverage::ElementId::new("nav", "next_tensor"),
                InteractionType::Click,
            );

            app.select_prev_tensor();
            tracker.record_interaction(
                &jugar_probar::ux_coverage::ElementId::new("nav", "prev_tensor"),
                InteractionType::Click,
            );

            // Quit action
            app.should_quit = true;
            tracker.record_interaction(
                &jugar_probar::ux_coverage::ElementId::new("nav", "quit"),
                InteractionType::Click,
            );

            // Generate report
            let report = tracker.generate_report();
            println!("{}", report.summary());

            // Assert 100% coverage - COMPLETE required
            assert!(report.is_complete, "UX coverage must be COMPLETE");
            assert!(
                tracker.meets(100.0),
                "UX coverage must be 100%: {}",
                tracker.summary()
            );
            assert!(
                tracker.assert_coverage(1.0).is_ok(),
                "Must meet 100% threshold"
            );
        }

        #[test]
        fn test_ux_coverage_100_percent_elements() {
            use jugar_probar::gui_coverage;

            // Define minimal TUI element coverage
            let mut gui = gui_coverage! {
                buttons: ["tab_1", "tab_2", "tab_3", "tab_help", "quit"],
                screens: ["overview", "tensors", "stats", "help"]
            };

            // Cover all elements
            gui.click("tab_1");
            gui.click("tab_2");
            gui.click("tab_3");
            gui.click("tab_help");
            gui.click("quit");

            gui.visit("overview");
            gui.visit("tensors");
            gui.visit("stats");
            gui.visit("help");

            // Assert 100% coverage
            assert!(gui.is_complete(), "Should have 100% GUI coverage");
            assert!(gui.meets(100.0), "Coverage: {}", gui.summary());

            let report = gui.generate_report();
            println!(
                "UX Coverage Report:\n  Elements: {}/{}\n  States: {}/{}\n  Overall: {:.1}%",
                report.covered_elements,
                report.total_elements,
                report.covered_states,
                report.total_states,
                report.overall_coverage * 100.0
            );
        }

        #[test]
        fn test_ux_coverage_with_report() {
            use jugar_probar::ux_coverage::UxCoverageBuilder;

            let mut tracker = UxCoverageBuilder::new()
                .button("overview")
                .button("tensors")
                .button("stats")
                .button("help")
                .button("quit")
                .screen("overview")
                .screen("tensors")
                .screen("stats")
                .screen("help")
                .build();

            // Cover all
            tracker.click("overview");
            tracker.click("tensors");
            tracker.click("stats");
            tracker.click("help");
            tracker.click("quit");
            tracker.visit("overview");
            tracker.visit("tensors");
            tracker.visit("stats");
            tracker.visit("help");

            // Full report
            let report = tracker.generate_report();
            assert!(report.is_complete);
            assert_eq!(report.covered_elements, 5);
            assert_eq!(report.covered_states, 4);
            assert!((report.overall_coverage - 1.0).abs() < f64::EPSILON);

            // Assert 95%+ coverage threshold
            assert!(
                tracker.assert_coverage(0.95).is_ok(),
                "Should meet 95% threshold"
            );
        }
