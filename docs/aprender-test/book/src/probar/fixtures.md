# Fixtures

> **Toyota Way**: Heijunka (Level Loading) - Consistent test environments

Manage test fixtures for setup and teardown with dependency injection and ordered lifecycle management.

## Running the Example

```bash
cargo run --example basic_test
```

## Quick Start

```rust
use probar::{Fixture, FixtureManager, ProbarResult};

// Define a fixture
struct DatabaseFixture {
    connection: Option<String>,
}

impl Fixture for DatabaseFixture {
    fn setup(&mut self) -> ProbarResult<()> {
        self.connection = Some("db://test".to_string());
        println!("Database connected");
        Ok(())
    }

    fn teardown(&mut self) -> ProbarResult<()> {
        self.connection = None;
        println!("Database disconnected");
        Ok(())
    }
}

// Use fixtures
let mut manager = FixtureManager::new();
manager.register(DatabaseFixture { connection: None });
manager.setup_all()?;
// Run tests...
manager.teardown_all()?;
```

## The Fixture Trait

```rust
use probar::{Fixture, ProbarResult};

// Full fixture implementation
struct BrowserFixture {
    browser_id: Option<u32>,
    headless: bool,
}

impl Fixture for BrowserFixture {
    fn setup(&mut self) -> ProbarResult<()> {
        // Launch browser
        self.browser_id = Some(42);
        println!("Browser launched (headless: {})", self.headless);
        Ok(())
    }

    fn teardown(&mut self) -> ProbarResult<()> {
        // Close browser
        if let Some(id) = self.browser_id.take() {
            println!("Browser {} closed", id);
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "BrowserFixture"
    }

    fn priority(&self) -> i32 {
        10  // Higher priority = setup first, teardown last
    }
}
```

## Fixture State

```rust
use probar::FixtureState;

// Fixture lifecycle states
let states = [
    FixtureState::Registered, // Just registered
    FixtureState::SetUp,      // Setup completed
    FixtureState::TornDown,   // Teardown completed
    FixtureState::Failed,     // Setup or teardown failed
];

// Check fixture state
fn describe_state(state: FixtureState) {
    match state {
        FixtureState::Registered => println!("Ready to set up"),
        FixtureState::SetUp => println!("Active and ready"),
        FixtureState::TornDown => println!("Cleaned up"),
        FixtureState::Failed => println!("Error occurred"),
    }
}
```

## Fixture Manager

```rust
use probar::{FixtureManager, Fixture, ProbarResult};

// Create manager
let mut manager = FixtureManager::new();

// Register fixtures
// manager.register(DatabaseFixture::new());
// manager.register(BrowserFixture::new());
// manager.register(CacheFixture::new());

// Check registration
// assert!(manager.is_registered::<DatabaseFixture>());

// Setup all fixtures (ordered by priority)
manager.setup_all()?;

// Run tests...

// Teardown all fixtures (reverse order)
manager.teardown_all()?;

// Get fixture count
println!("Registered fixtures: {}", manager.fixture_count());
```

## Fixture Priority

```rust
use probar::{Fixture, ProbarResult};

// Infrastructure fixtures (setup first)
struct NetworkFixture;
impl Fixture for NetworkFixture {
    fn setup(&mut self) -> ProbarResult<()> { Ok(()) }
    fn teardown(&mut self) -> ProbarResult<()> { Ok(()) }
    fn priority(&self) -> i32 { 100 }  // Highest
}

// Database fixture (depends on network)
struct DatabaseFixture;
impl Fixture for DatabaseFixture {
    fn setup(&mut self) -> ProbarResult<()> { Ok(()) }
    fn teardown(&mut self) -> ProbarResult<()> { Ok(()) }
    fn priority(&self) -> i32 { 50 }  // Medium
}

// Application fixtures (depends on database)
struct AppFixture;
impl Fixture for AppFixture {
    fn setup(&mut self) -> ProbarResult<()> { Ok(()) }
    fn teardown(&mut self) -> ProbarResult<()> { Ok(()) }
    fn priority(&self) -> i32 { 10 }  // Lower
}

// Setup order: Network → Database → App
// Teardown order: App → Database → Network
```

## Fixture Scope

```rust
use probar::FixtureScope;

// Different fixture scopes
let scopes = [
    FixtureScope::Test,     // Per test
    FixtureScope::Suite,    // Per test suite
    FixtureScope::Session,  // Entire session
];

// Scope affects when setup/teardown runs
fn describe_scope(scope: FixtureScope) {
    match scope {
        FixtureScope::Test => {
            println!("Setup before each test, teardown after");
        }
        FixtureScope::Suite => {
            println!("Setup once per suite, teardown at end");
        }
        FixtureScope::Session => {
            println!("Setup once, teardown at session end");
        }
    }
}
```

## Fixture Builder

```rust
use probar::{FixtureBuilder, Fixture, ProbarResult};

// Build fixtures with configuration
let fixture = FixtureBuilder::new("TestServer")
    .with_priority(50)
    .with_scope(probar::FixtureScope::Suite)
    .on_setup(|| {
        println!("Starting server...");
        Ok(())
    })
    .on_teardown(|| {
        println!("Stopping server...");
        Ok(())
    })
    .build();
```

## Simple Fixture

```rust
use probar::{SimpleFixture, ProbarResult};

// Quick fixture without full trait implementation
let fixture = SimpleFixture::new(
    "TempDir",
    || {
        // Setup: create temp directory
        println!("Creating temp dir");
        Ok(())
    },
    || {
        // Teardown: remove temp directory
        println!("Removing temp dir");
        Ok(())
    },
);
```

## Error Handling

```rust
use probar::{Fixture, FixtureManager, ProbarResult, ProbarError};

struct FlakeyFixture {
    fail_setup: bool,
}

impl Fixture for FlakeyFixture {
    fn setup(&mut self) -> ProbarResult<()> {
        if self.fail_setup {
            Err(ProbarError::FixtureSetupFailed {
                name: "FlakeyFixture".to_string(),
                reason: "Simulated failure".to_string(),
            })
        } else {
            Ok(())
        }
    }

    fn teardown(&mut self) -> ProbarResult<()> {
        Ok(())
    }
}

// Handle setup failures
let mut manager = FixtureManager::new();
// manager.register(FlakeyFixture { fail_setup: true });

match manager.setup_all() {
    Ok(()) => println!("All fixtures ready"),
    Err(e) => {
        eprintln!("Fixture setup failed: {}", e);
        // Attempt cleanup of already-setup fixtures
        let _ = manager.teardown_all();
    }
}
```

## Fixture Dependencies

```rust
use probar::{Fixture, ProbarResult};

// Fixtures with explicit dependencies
struct WebServerFixture {
    port: u16,
    // db: DatabaseHandle, // Would hold reference to DB fixture
}

impl WebServerFixture {
    fn new(port: u16) -> Self {
        Self { port }
    }

    // Access database through dependency
    // fn with_database(mut self, db: &DatabaseFixture) -> Self {
    //     self.db = db.connection().clone();
    //     self
    // }
}

impl Fixture for WebServerFixture {
    fn setup(&mut self) -> ProbarResult<()> {
        println!("Starting web server on port {}", self.port);
        Ok(())
    }

    fn teardown(&mut self) -> ProbarResult<()> {
        println!("Stopping web server");
        Ok(())
    }

    fn priority(&self) -> i32 {
        20  // Lower than database
    }
}
```

## Test Integration

```rust
use probar::{FixtureManager, TestHarness, TestSuite};

fn run_with_fixtures() {
    // Setup fixtures
    let mut fixtures = FixtureManager::new();
    // fixtures.register(DatabaseFixture::new());
    // fixtures.register(BrowserFixture::new());

    // Setup all
    if fixtures.setup_all().is_err() {
        eprintln!("Fixture setup failed");
        return;
    }

    // Run tests
    let harness = TestHarness::new();
    let suite = TestSuite::new("integration_tests");
    let results = harness.run(&suite);

    // Always teardown, even if tests fail
    let teardown_result = fixtures.teardown_all();

    // Report results
    println!("Tests: {} passed, {} failed",
        results.passed_count(),
        results.failed_count());

    if teardown_result.is_err() {
        eprintln!("Warning: fixture teardown had errors");
    }
}
```

## Best Practices

1. **Clear Priority**: Set explicit priorities for predictable ordering
2. **Always Teardown**: Ensure cleanup runs even on test failures
3. **Independent Setup**: Each fixture should be self-contained
4. **Fast Setup**: Keep fixture setup quick for rapid test iteration
5. **Idempotent Teardown**: Teardown should handle partial setup states
6. **Logging**: Add logging to track fixture lifecycle
7. **Resource Limits**: Consider memory/connection limits in fixtures
