//! trueno-monitor - TUI system and ML workload monitor.
//!
//! A btop-like terminal monitor with Sovereign AI Stack integration.

use trueno_viz::monitor::{App, Config};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load configuration
    let config = Config::load_or_default(
        dirs::config_dir().map(|p| p.join("trueno-monitor/config.yaml")).unwrap_or_default(),
    );

    // Run the application
    let mut app = App::new(config);
    app.run()?;

    Ok(())
}
