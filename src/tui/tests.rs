//! Integration tests for TUI module.
//!
//! These tests verify the TUI app modules work correctly together.

use super::orbit_app::OrbitApp;
use super::tsp_app::TspApp;
use crossterm::event::KeyCode;

#[test]
fn test_orbit_app_lifecycle() {
    let mut app = OrbitApp::new();

    // Initial state
    assert!(!app.paused);
    assert!(!app.should_quit);
    assert_eq!(app.frame_count, 0);

    // Run a few frames
    for _ in 0..5 {
        app.update();
    }
    assert_eq!(app.frame_count, 5);

    // Pause
    app.handle_key(KeyCode::Char(' '));
    assert!(app.paused);

    // Update while paused (should not advance)
    let frame_before = app.frame_count;
    app.update();
    assert_eq!(app.frame_count, frame_before);

    // Reset
    app.handle_key(KeyCode::Char('r'));
    assert_eq!(app.frame_count, 0);

    // Quit
    app.handle_key(KeyCode::Char('q'));
    assert!(app.should_quit);
}

#[test]
fn test_tsp_app_lifecycle() {
    let mut app = TspApp::new(15, 42);

    // Initial state
    assert!(app.auto_run);
    assert!(!app.paused);
    assert!(!app.should_quit());
    assert_eq!(app.demo.n, 15);

    // Run a few steps
    for _ in 0..5 {
        app.step();
    }
    assert_eq!(app.frame_count, 5);
    assert!(app.convergence_history.len() >= 5);

    // Pause (toggle auto_run)
    app.handle_key(KeyCode::Char(' '));
    assert!(!app.auto_run);
    assert!(app.paused);

    // Single step
    let restarts_before = app.demo.restarts;
    app.handle_key(KeyCode::Char('g'));
    assert!(app.demo.restarts > restarts_before);

    // Reset
    app.handle_key(KeyCode::Char('r'));
    assert_eq!(app.frame_count, 0);

    // Quit
    app.handle_key(KeyCode::Char('q'));
    assert!(app.should_quit());
}

#[test]
fn test_orbit_jidoka_status_preserved() {
    let mut app = OrbitApp::new();
    // Run at least one update to initialize jidoka properly
    app.update();
    let status = app.jidoka_status();

    // After running, finite check should pass
    assert!(status.finite_ok);
}

#[test]
fn test_tsp_equation_verification() {
    let mut app = TspApp::new(20, 42);

    // Run several iterations
    for _ in 0..10 {
        app.step();
    }

    // Equation should still verify
    assert!(app.verify_equation());

    // Gap should be reasonable (< 100% for a good heuristic)
    let gap = app.optimality_gap();
    assert!(gap < 1.0);
}

#[test]
fn test_orbit_energy_conservation() {
    let mut app = OrbitApp::new();
    let initial_energy = app.total_energy();

    // Run simulation
    for _ in 0..10 {
        app.update();
    }

    let final_energy = app.total_energy();

    // Energy should be conserved within Jidoka tolerance
    let relative_error = ((final_energy - initial_energy) / initial_energy).abs();
    assert!(relative_error < 0.01, "Energy drift: {relative_error:.6}");
}
