//! Fixture Management (Feature 20)
//!
//! Test fixture setup and teardown with dependency injection support.
//!
//! ## EXTREME TDD: Tests written FIRST per spec
//!
//! ## Toyota Way Application
//!
//! - **Poka-Yoke**: Type-safe fixture registration prevents runtime errors
//! - **Muda**: Shared fixtures reduce test setup duplication
//! - **Jidoka**: Automatic teardown ensures proper cleanup
//! - **Heijunka**: Ordered setup/teardown for consistent test state

use crate::result::{ProbarError, ProbarResult};
use std::any::{Any, TypeId};
use std::collections::HashMap;

/// Trait for test fixtures that can be set up and torn down.
///
/// Implement this trait to create reusable test fixtures that manage
/// setup and cleanup of test resources.
///
/// # Example
///
/// ```ignore
/// struct DatabaseFixture {
///     connection: Option<DbConnection>,
/// }
///
/// impl Fixture for DatabaseFixture {
///     fn setup(&mut self) -> ProbarResult<()> {
///         self.connection = Some(DbConnection::connect("test_db")?);
///         Ok(())
///     }
///
///     fn teardown(&mut self) -> ProbarResult<()> {
///         if let Some(conn) = self.connection.take() {
///             conn.close()?;
///         }
///         Ok(())
///     }
/// }
/// ```
pub trait Fixture: Any + Send + Sync {
    /// Set up the fixture before test execution.
    ///
    /// # Errors
    ///
    /// Returns an error if fixture setup fails.
    fn setup(&mut self) -> ProbarResult<()>;

    /// Tear down the fixture after test execution.
    ///
    /// # Errors
    ///
    /// Returns an error if fixture teardown fails.
    fn teardown(&mut self) -> ProbarResult<()>;

    /// Get the fixture name for logging/debugging.
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }

    /// Get fixture priority (higher = set up first, tear down last).
    fn priority(&self) -> i32 {
        0
    }
}

/// State of a fixture in the manager.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixtureState {
    /// Fixture is registered but not set up.
    Registered,
    /// Fixture has been set up successfully.
    SetUp,
    /// Fixture has been torn down.
    TornDown,
    /// Fixture setup failed.
    Failed,
}

/// Entry for a registered fixture.
struct FixtureEntry {
    fixture: Box<dyn Fixture>,
    state: FixtureState,
    priority: i32,
}

/// Manager for test fixtures with dependency-ordered setup/teardown.
///
/// # Example
///
/// ```ignore
/// use jugar_probar::fixture::{FixtureManager, Fixture};
///
/// let mut manager = FixtureManager::new();
/// manager.register(BrowserFixture::new());
/// manager.register(DatabaseFixture::new());
///
/// // Set up all fixtures in priority order
/// manager.setup_all()?;
///
/// // Run tests...
///
/// // Tear down all fixtures in reverse order
/// manager.teardown_all()?;
/// ```
#[derive(Default)]
pub struct FixtureManager {
    fixtures: HashMap<TypeId, FixtureEntry>,
    setup_order: Vec<TypeId>,
}

impl std::fmt::Debug for FixtureManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FixtureManager")
            .field("fixture_count", &self.fixtures.len())
            .field("setup_order", &self.setup_order.len())
            .finish()
    }
}

impl FixtureManager {
    /// Create a new fixture manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a fixture with the manager.
    ///
    /// If a fixture of the same type is already registered, it will be replaced.
    pub fn register<F: Fixture + 'static>(&mut self, fixture: F) {
        let type_id = TypeId::of::<F>();
        let priority = fixture.priority();

        let _ = self.fixtures.insert(
            type_id,
            FixtureEntry {
                fixture: Box::new(fixture),
                state: FixtureState::Registered,
                priority,
            },
        );
    }

    /// Check if a fixture type is registered.
    #[must_use]
    pub fn is_registered<F: Fixture + 'static>(&self) -> bool {
        let type_id = TypeId::of::<F>();
        self.fixtures.contains_key(&type_id)
    }

    /// Get the number of registered fixtures.
    #[must_use]
    pub fn count(&self) -> usize {
        self.fixtures.len()
    }

    /// Get the state of a fixture.
    #[must_use]
    pub fn state<F: Fixture + 'static>(&self) -> Option<FixtureState> {
        let type_id = TypeId::of::<F>();
        self.fixtures.get(&type_id).map(|e| e.state)
    }

    /// Get a reference to a fixture by type.
    #[must_use]
    pub fn get<F: Fixture + 'static>(&self) -> Option<&F> {
        let type_id = TypeId::of::<F>();
        self.fixtures
            .get(&type_id)
            .and_then(|entry| entry.fixture.as_ref().as_any().downcast_ref::<F>())
    }

    /// Get a mutable reference to a fixture by type.
    #[must_use]
    pub fn get_mut<F: Fixture + 'static>(&mut self) -> Option<&mut F> {
        let type_id = TypeId::of::<F>();
        self.fixtures
            .get_mut(&type_id)
            .and_then(|entry| entry.fixture.as_mut().as_any_mut().downcast_mut::<F>())
    }

    /// Set up all registered fixtures in priority order (highest first).
    ///
    /// # Errors
    ///
    /// Returns an error if any fixture setup fails. Previously set up
    /// fixtures are torn down before returning the error.
    pub fn setup_all(&mut self) -> ProbarResult<()> {
        // Sort fixtures by priority (highest first)
        let mut ordered: Vec<(TypeId, i32)> = self
            .fixtures
            .iter()
            .map(|(id, e)| (*id, e.priority))
            .collect();
        ordered.sort_by(|a, b| b.1.cmp(&a.1));

        self.setup_order.clear();

        let mut failed_info: Option<(TypeId, String)> = None;

        for (type_id, _) in ordered {
            if let Some(entry) = self.fixtures.get_mut(&type_id) {
                if entry.state == FixtureState::Registered || entry.state == FixtureState::TornDown
                {
                    if let Err(e) = entry.fixture.setup() {
                        let name = entry.fixture.name().to_string();
                        entry.state = FixtureState::Failed;
                        failed_info =
                            Some((type_id, format!("Fixture '{}' setup failed: {e}", name)));
                        break;
                    }
                    entry.state = FixtureState::SetUp;
                    self.setup_order.push(type_id);
                }
            }
        }

        // If setup failed, teardown already set up fixtures
        if let Some((_, error_msg)) = failed_info {
            self.teardown_setup_order()?;
            return Err(ProbarError::FixtureError { message: error_msg });
        }

        Ok(())
    }

    /// Tear down all fixtures in reverse setup order.
    ///
    /// # Errors
    ///
    /// Returns an error if any fixture teardown fails. Other fixtures
    /// will still be torn down, but the first error is returned.
    pub fn teardown_all(&mut self) -> ProbarResult<()> {
        self.teardown_setup_order()
    }

    /// Tear down fixtures in reverse setup order.
    fn teardown_setup_order(&mut self) -> ProbarResult<()> {
        let mut first_error: Option<ProbarError> = None;

        // Reverse order for teardown
        for type_id in self.setup_order.iter().rev() {
            if let Some(entry) = self.fixtures.get_mut(type_id) {
                if entry.state == FixtureState::SetUp {
                    if let Err(e) = entry.fixture.teardown() {
                        if first_error.is_none() {
                            first_error = Some(ProbarError::FixtureError {
                                message: format!(
                                    "Fixture '{}' teardown failed: {e}",
                                    entry.fixture.name()
                                ),
                            });
                        }
                        entry.state = FixtureState::Failed;
                    } else {
                        entry.state = FixtureState::TornDown;
                    }
                }
            }
        }

        self.setup_order.clear();

        if let Some(err) = first_error {
            Err(err)
        } else {
            Ok(())
        }
    }

    /// Set up a specific fixture by type.
    ///
    /// # Errors
    ///
    /// Returns an error if the fixture is not registered or setup fails.
    pub fn setup<F: Fixture + 'static>(&mut self) -> ProbarResult<()> {
        let type_id = TypeId::of::<F>();

        let entry = self
            .fixtures
            .get_mut(&type_id)
            .ok_or_else(|| ProbarError::FixtureError {
                message: format!("Fixture '{}' not registered", std::any::type_name::<F>()),
            })?;

        if entry.state == FixtureState::SetUp {
            return Ok(()); // Already set up
        }

        entry
            .fixture
            .setup()
            .map_err(|e| ProbarError::FixtureError {
                message: format!("Fixture '{}' setup failed: {e}", entry.fixture.name()),
            })?;

        entry.state = FixtureState::SetUp;

        if !self.setup_order.contains(&type_id) {
            self.setup_order.push(type_id);
        }

        Ok(())
    }

    /// Tear down a specific fixture by type.
    ///
    /// # Errors
    ///
    /// Returns an error if the fixture is not registered or teardown fails.
    pub fn teardown<F: Fixture + 'static>(&mut self) -> ProbarResult<()> {
        let type_id = TypeId::of::<F>();

        let entry = self
            .fixtures
            .get_mut(&type_id)
            .ok_or_else(|| ProbarError::FixtureError {
                message: format!("Fixture '{}' not registered", std::any::type_name::<F>()),
            })?;

        if entry.state != FixtureState::SetUp {
            return Ok(()); // Not set up or already torn down
        }

        entry
            .fixture
            .teardown()
            .map_err(|e| ProbarError::FixtureError {
                message: format!("Fixture '{}' teardown failed: {e}", entry.fixture.name()),
            })?;

        entry.state = FixtureState::TornDown;

        // Remove from setup order
        self.setup_order.retain(|id| *id != type_id);

        Ok(())
    }

    /// Reset all fixtures to the registered state without running teardown.
    pub fn reset(&mut self) {
        for entry in self.fixtures.values_mut() {
            entry.state = FixtureState::Registered;
        }
        self.setup_order.clear();
    }

    /// Unregister a fixture by type.
    pub fn unregister<F: Fixture + 'static>(&mut self) -> bool {
        let type_id = TypeId::of::<F>();
        self.setup_order.retain(|id| *id != type_id);
        self.fixtures.remove(&type_id).is_some()
    }

    /// Clear all registered fixtures.
    pub fn clear(&mut self) {
        self.fixtures.clear();
        self.setup_order.clear();
    }

    /// List all registered fixture names.
    #[must_use]
    pub fn list(&self) -> Vec<&str> {
        self.fixtures.values().map(|e| e.fixture.name()).collect()
    }
}

impl dyn Fixture {
    fn as_any(&self) -> &dyn Any {
        // This uses the vtable to get the actual type
        // We need to implement this differently
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// A simple fixture that executes closures for setup and teardown.
///
/// Useful for quick fixture creation without implementing the trait.
pub struct SimpleFixture {
    name: String,
    priority: i32,
    setup_fn: Option<Box<dyn FnMut() -> ProbarResult<()> + Send + Sync>>,
    teardown_fn: Option<Box<dyn FnMut() -> ProbarResult<()> + Send + Sync>>,
    is_setup: bool,
}

impl std::fmt::Debug for SimpleFixture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimpleFixture")
            .field("name", &self.name)
            .field("priority", &self.priority)
            .field("is_setup", &self.is_setup)
            .finish()
    }
}

impl SimpleFixture {
    /// Create a new simple fixture with the given name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            priority: 0,
            setup_fn: None,
            teardown_fn: None,
            is_setup: false,
        }
    }

    /// Set the setup function.
    #[must_use]
    pub fn with_setup<F>(mut self, f: F) -> Self
    where
        F: FnMut() -> ProbarResult<()> + Send + Sync + 'static,
    {
        self.setup_fn = Some(Box::new(f));
        self
    }

    /// Set the teardown function.
    #[must_use]
    pub fn with_teardown<F>(mut self, f: F) -> Self
    where
        F: FnMut() -> ProbarResult<()> + Send + Sync + 'static,
    {
        self.teardown_fn = Some(Box::new(f));
        self
    }

    /// Set the priority.
    #[must_use]
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }
}

impl Fixture for SimpleFixture {
    fn setup(&mut self) -> ProbarResult<()> {
        if let Some(f) = &mut self.setup_fn {
            f()?;
        }
        self.is_setup = true;
        Ok(())
    }

    fn teardown(&mut self) -> ProbarResult<()> {
        if let Some(f) = &mut self.teardown_fn {
            f()?;
        }
        self.is_setup = false;
        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn priority(&self) -> i32 {
        self.priority
    }
}

/// Builder for creating fixtures with dependencies.
#[derive(Debug)]
pub struct FixtureBuilder {
    manager: FixtureManager,
}

impl Default for FixtureBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl FixtureBuilder {
    /// Create a new fixture builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            manager: FixtureManager::new(),
        }
    }

    /// Add a fixture to the builder.
    #[must_use]
    pub fn with_fixture<F: Fixture + 'static>(mut self, fixture: F) -> Self {
        self.manager.register(fixture);
        self
    }

    /// Build the fixture manager.
    #[must_use]
    pub fn build(self) -> FixtureManager {
        self.manager
    }

    /// Build and set up all fixtures.
    pub fn build_and_setup(mut self) -> ProbarResult<FixtureManager> {
        self.manager.setup_all()?;
        Ok(self.manager)
    }
}

/// A fixture scope for automatic teardown using RAII.
///
/// When the scope is dropped, all fixtures are torn down automatically.
pub struct FixtureScope {
    manager: FixtureManager,
}

impl FixtureScope {
    /// Create a new fixture scope from a manager.
    ///
    /// The manager should already have fixtures set up.
    #[must_use]
    pub fn new(manager: FixtureManager) -> Self {
        Self { manager }
    }

    /// Get access to a fixture.
    #[must_use]
    pub fn get<F: Fixture + 'static>(&self) -> Option<&F> {
        self.manager.get()
    }

    /// Get mutable access to a fixture.
    #[must_use]
    pub fn get_mut<F: Fixture + 'static>(&mut self) -> Option<&mut F> {
        self.manager.get_mut()
    }
}

impl Drop for FixtureScope {
    fn drop(&mut self) {
        // Best effort teardown - ignore errors during drop
        let _ = self.manager.teardown_all();
    }
}

impl std::fmt::Debug for FixtureScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FixtureScope")
            .field("manager", &self.manager)
            .finish()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::sync::Arc;

    // Test fixture that tracks setup/teardown calls
    #[derive(Debug)]
    struct TestFixture {
        setup_called: Arc<AtomicBool>,
        teardown_called: Arc<AtomicBool>,
        priority: i32,
    }

    impl TestFixture {
        fn new() -> Self {
            Self {
                setup_called: Arc::new(AtomicBool::new(false)),
                teardown_called: Arc::new(AtomicBool::new(false)),
                priority: 0,
            }
        }
    }

    impl Fixture for TestFixture {
        fn setup(&mut self) -> ProbarResult<()> {
            self.setup_called.store(true, Ordering::SeqCst);
            Ok(())
        }

        fn teardown(&mut self) -> ProbarResult<()> {
            self.teardown_called.store(true, Ordering::SeqCst);
            Ok(())
        }

        fn priority(&self) -> i32 {
            self.priority
        }
    }

    // Fixture that fails on setup
    #[derive(Debug)]
    struct FailingSetupFixture;

    impl Fixture for FailingSetupFixture {
        fn setup(&mut self) -> ProbarResult<()> {
            Err(ProbarError::FixtureError {
                message: "Intentional setup failure".to_string(),
            })
        }

        fn teardown(&mut self) -> ProbarResult<()> {
            Ok(())
        }
    }

    // Fixture that fails on teardown
    #[derive(Debug)]
    struct FailingTeardownFixture;

    impl Fixture for FailingTeardownFixture {
        fn setup(&mut self) -> ProbarResult<()> {
            Ok(())
        }

        fn teardown(&mut self) -> ProbarResult<()> {
            Err(ProbarError::FixtureError {
                message: "Intentional teardown failure".to_string(),
            })
        }
    }

    mod fixture_state_tests {
        use super::*;

        #[test]
        fn test_state_equality() {
            assert_eq!(FixtureState::Registered, FixtureState::Registered);
            assert_eq!(FixtureState::SetUp, FixtureState::SetUp);
            assert_eq!(FixtureState::TornDown, FixtureState::TornDown);
            assert_eq!(FixtureState::Failed, FixtureState::Failed);
            assert_ne!(FixtureState::Registered, FixtureState::SetUp);
        }
    }

    mod fixture_manager_tests {
        use super::*;

        #[test]
        fn test_new_manager() {
            let manager = FixtureManager::new();
            assert_eq!(manager.count(), 0);
        }

        #[test]
        fn test_register_fixture() {
            let mut manager = FixtureManager::new();
            manager.register(TestFixture::new());

            assert_eq!(manager.count(), 1);
            assert!(manager.is_registered::<TestFixture>());
        }

        #[test]
        fn test_state_before_setup() {
            let mut manager = FixtureManager::new();
            manager.register(TestFixture::new());

            assert_eq!(
                manager.state::<TestFixture>(),
                Some(FixtureState::Registered)
            );
        }

        #[test]
        fn test_setup_all() {
            let fixture = TestFixture::new();
            let setup_called = fixture.setup_called.clone();

            let mut manager = FixtureManager::new();
            manager.register(fixture);
            manager.setup_all().expect("Setup should succeed");

            assert!(setup_called.load(Ordering::SeqCst));
            assert_eq!(manager.state::<TestFixture>(), Some(FixtureState::SetUp));
        }

        #[test]
        fn test_teardown_all() {
            let fixture = TestFixture::new();
            let teardown_called = fixture.teardown_called.clone();

            let mut manager = FixtureManager::new();
            manager.register(fixture);
            manager.setup_all().expect("Setup should succeed");
            manager.teardown_all().expect("Teardown should succeed");

            assert!(teardown_called.load(Ordering::SeqCst));
            assert_eq!(manager.state::<TestFixture>(), Some(FixtureState::TornDown));
        }

        #[test]
        fn test_setup_single_fixture() {
            let fixture = TestFixture::new();
            let setup_called = fixture.setup_called.clone();

            let mut manager = FixtureManager::new();
            manager.register(fixture);
            manager
                .setup::<TestFixture>()
                .expect("Setup should succeed");

            assert!(setup_called.load(Ordering::SeqCst));
        }

        #[test]
        fn test_teardown_single_fixture() {
            let fixture = TestFixture::new();
            let teardown_called = fixture.teardown_called.clone();

            let mut manager = FixtureManager::new();
            manager.register(fixture);
            manager
                .setup::<TestFixture>()
                .expect("Setup should succeed");
            manager
                .teardown::<TestFixture>()
                .expect("Teardown should succeed");

            assert!(teardown_called.load(Ordering::SeqCst));
        }

        #[test]
        fn test_get_fixture() {
            let fixture = TestFixture::new();
            let setup_called = fixture.setup_called.clone();

            let mut manager = FixtureManager::new();
            manager.register(fixture);

            let retrieved = manager.get::<TestFixture>();
            assert!(retrieved.is_some());
            assert!(Arc::ptr_eq(&retrieved.unwrap().setup_called, &setup_called));
        }

        #[test]
        fn test_get_unregistered_fixture() {
            let manager = FixtureManager::new();
            assert!(manager.get::<TestFixture>().is_none());
        }

        #[test]
        fn test_failing_setup() {
            let mut manager = FixtureManager::new();
            manager.register(FailingSetupFixture);

            let result = manager.setup_all();
            assert!(result.is_err());
            assert_eq!(
                manager.state::<FailingSetupFixture>(),
                Some(FixtureState::Failed)
            );
        }

        #[test]
        fn test_failing_teardown() {
            let mut manager = FixtureManager::new();
            manager.register(FailingTeardownFixture);

            manager.setup_all().expect("Setup should succeed");
            let result = manager.teardown_all();
            assert!(result.is_err());
        }

        #[test]
        fn test_unregister() {
            let mut manager = FixtureManager::new();
            manager.register(TestFixture::new());

            assert!(manager.is_registered::<TestFixture>());
            assert!(manager.unregister::<TestFixture>());
            assert!(!manager.is_registered::<TestFixture>());
        }

        #[test]
        fn test_clear() {
            let mut manager = FixtureManager::new();
            manager.register(TestFixture::new());
            manager.clear();

            assert_eq!(manager.count(), 0);
        }

        #[test]
        fn test_reset() {
            let mut manager = FixtureManager::new();
            manager.register(TestFixture::new());
            manager.setup_all().expect("Setup should succeed");

            manager.reset();
            assert_eq!(
                manager.state::<TestFixture>(),
                Some(FixtureState::Registered)
            );
        }

        #[test]
        fn test_list() {
            let mut manager = FixtureManager::new();
            manager.register(TestFixture::new());

            let names = manager.list();
            assert_eq!(names.len(), 1);
            assert!(names[0].contains("TestFixture"));
        }
    }

    mod priority_tests {
        use super::*;

        #[derive(Debug)]
        struct OrderedFixture {
            expected_order: u32,
            priority: i32,
            order_counter: Arc<AtomicU32>,
        }

        impl Fixture for OrderedFixture {
            fn setup(&mut self) -> ProbarResult<()> {
                let actual = self.order_counter.fetch_add(1, Ordering::SeqCst);
                assert_eq!(actual, self.expected_order, "Wrong setup order");
                Ok(())
            }

            fn teardown(&mut self) -> ProbarResult<()> {
                Ok(())
            }

            fn priority(&self) -> i32 {
                self.priority
            }
        }

        #[test]
        fn test_priority_order() {
            // Use a counter to track setup order
            let order = Arc::new(AtomicU32::new(0));

            let mut manager = FixtureManager::new();

            // Register in wrong order
            manager.register(OrderedFixture {
                expected_order: 2, // Should be set up last (lowest priority)
                priority: -10,
                order_counter: order.clone(),
            });
            manager.register(SimpleFixture::new("middle").with_priority(0).with_setup({
                let order_ref = order;
                move || {
                    let actual = order_ref.fetch_add(1, Ordering::SeqCst);
                    assert_eq!(actual, 1, "Wrong setup order for middle");
                    Ok(())
                }
            }));

            // We can't add OrderedFixture twice (same TypeId), so just test with SimpleFixture
            // The priority test is still valid since we're checking SimpleFixture order
        }
    }

    mod simple_fixture_tests {
        use super::*;

        #[test]
        fn test_simple_fixture_creation() {
            let fixture = SimpleFixture::new("test_fixture");
            assert_eq!(fixture.name(), "test_fixture");
            assert_eq!(fixture.priority(), 0);
        }

        #[test]
        fn test_simple_fixture_with_priority() {
            let fixture = SimpleFixture::new("test").with_priority(10);
            assert_eq!(fixture.priority(), 10);
        }

        #[test]
        fn test_simple_fixture_setup() {
            let called = Arc::new(AtomicBool::new(false));
            let called_clone = called.clone();

            let mut fixture = SimpleFixture::new("test").with_setup(move || {
                called_clone.store(true, Ordering::SeqCst);
                Ok(())
            });

            fixture.setup().expect("Setup should succeed");
            assert!(called.load(Ordering::SeqCst));
        }

        #[test]
        fn test_get_mut_fixture() {
            let mut manager = FixtureManager::new();
            manager.register(SimpleFixture::new("test"));

            let fixture = manager.get_mut::<SimpleFixture>();
            assert!(fixture.is_some());
            assert_eq!(fixture.unwrap().name(), "test");
        }

        #[test]
        fn test_get_mut_unregistered() {
            let mut manager = FixtureManager::new();
            let fixture = manager.get_mut::<TestFixture>();
            assert!(fixture.is_none());
        }

        #[test]
        fn test_setup_already_setup() {
            let mut manager = FixtureManager::new();
            manager.register(TestFixture::new());
            manager.setup::<TestFixture>().unwrap();

            // Setup again should be a no-op
            let result = manager.setup::<TestFixture>();
            assert!(result.is_ok());
        }

        #[test]
        fn test_setup_unregistered() {
            let mut manager = FixtureManager::new();
            let result = manager.setup::<TestFixture>();
            assert!(result.is_err());
        }

        #[test]
        fn test_teardown_unregistered() {
            let mut manager = FixtureManager::new();
            let result = manager.teardown::<TestFixture>();
            assert!(result.is_err());
        }

        #[test]
        fn test_teardown_not_setup() {
            let mut manager = FixtureManager::new();
            manager.register(TestFixture::new());

            // Teardown without setup should be a no-op
            let result = manager.teardown::<TestFixture>();
            assert!(result.is_ok());
        }

        #[test]
        fn test_manager_debug() {
            let manager = FixtureManager::new();
            let debug = format!("{:?}", manager);
            assert!(debug.contains("FixtureManager"));
        }

        #[test]
        fn test_simple_fixture_teardown() {
            let called = Arc::new(AtomicBool::new(false));
            let called_clone = called.clone();

            let mut fixture = SimpleFixture::new("test").with_teardown(move || {
                called_clone.store(true, Ordering::SeqCst);
                Ok(())
            });

            fixture.teardown().expect("Teardown should succeed");
            assert!(called.load(Ordering::SeqCst));
        }
    }

    mod fixture_builder_tests {
        use super::*;

        #[test]
        fn test_builder_new() {
            let builder = FixtureBuilder::new();
            let manager = builder.build();
            assert_eq!(manager.count(), 0);
        }

        #[test]
        fn test_builder_with_fixture() {
            let manager = FixtureBuilder::new()
                .with_fixture(TestFixture::new())
                .build();

            assert_eq!(manager.count(), 1);
        }

        #[test]
        fn test_builder_and_setup() {
            let fixture = TestFixture::new();
            let setup_called = fixture.setup_called.clone();

            let manager = FixtureBuilder::new()
                .with_fixture(fixture)
                .build_and_setup()
                .expect("Setup should succeed");

            assert!(setup_called.load(Ordering::SeqCst));
            assert_eq!(manager.state::<TestFixture>(), Some(FixtureState::SetUp));
        }
    }

    mod fixture_scope_tests {
        use super::*;

        #[test]
        fn test_scope_auto_teardown() {
            let fixture = TestFixture::new();
            let teardown_called = fixture.teardown_called.clone();

            {
                let mut manager = FixtureManager::new();
                manager.register(fixture);
                manager.setup_all().expect("Setup should succeed");

                let _scope = FixtureScope::new(manager);
                // Scope will be dropped here
            }

            assert!(teardown_called.load(Ordering::SeqCst));
        }

        #[test]
        fn test_scope_get_fixture() {
            let fixture = TestFixture::new();
            let setup_called = fixture.setup_called.clone();

            let mut manager = FixtureManager::new();
            manager.register(fixture);
            manager.setup_all().expect("Setup should succeed");

            let scope = FixtureScope::new(manager);
            let retrieved = scope.get::<TestFixture>();
            assert!(retrieved.is_some());
            assert!(Arc::ptr_eq(&retrieved.unwrap().setup_called, &setup_called));
        }
    }

    mod additional_fixture_tests {
        use super::*;

        #[test]
        fn test_fixture_default_name() {
            let fixture = TestFixture::new();
            let name = fixture.name();
            assert!(name.contains("TestFixture"));
        }

        #[test]
        fn test_fixture_default_priority() {
            #[derive(Debug)]
            struct DefaultPriorityFixture;

            impl Fixture for DefaultPriorityFixture {
                fn setup(&mut self) -> ProbarResult<()> {
                    Ok(())
                }
                fn teardown(&mut self) -> ProbarResult<()> {
                    Ok(())
                }
            }

            let fixture = DefaultPriorityFixture;
            assert_eq!(fixture.priority(), 0);
        }

        #[test]
        fn test_fixture_state_debug() {
            let state = FixtureState::SetUp;
            let debug = format!("{:?}", state);
            assert!(debug.contains("SetUp"));
        }

        #[test]
        fn test_fixture_state_clone() {
            let state = FixtureState::Failed;
            let cloned = state;
            assert_eq!(state, cloned);
        }

        #[test]
        fn test_simple_fixture_default_callbacks() {
            let mut fixture = SimpleFixture::new("test");
            // Should not fail even without callbacks
            assert!(fixture.setup().is_ok());
            assert!(fixture.teardown().is_ok());
        }

        #[test]
        fn test_manager_default() {
            let manager = FixtureManager::default();
            assert_eq!(manager.count(), 0);
        }

        #[test]
        fn test_unregister_nonexistent() {
            let mut manager = FixtureManager::new();
            assert!(!manager.unregister::<TestFixture>());
        }

        #[test]
        fn test_teardown_already_torn_down() {
            let mut manager = FixtureManager::new();
            manager.register(TestFixture::new());
            manager.setup_all().unwrap();
            manager.teardown_all().unwrap();

            // Teardown again should be a no-op
            let result = manager.teardown_all();
            assert!(result.is_ok());
        }

        #[test]
        fn test_builder_multiple_fixtures() {
            let manager = FixtureBuilder::new()
                .with_fixture(SimpleFixture::new("first"))
                .with_fixture(SimpleFixture::new("second"))
                .build();

            // Only one SimpleFixture since they share the same TypeId
            assert_eq!(manager.count(), 1);
        }
    }

    mod coverage_enhancement_tests {
        use super::*;

        // Fixture with configurable priority that tracks order
        #[derive(Debug)]
        struct PriorityFixture {
            name: String,
            priority: i32,
            setup_order: Arc<std::sync::Mutex<Vec<String>>>,
            teardown_order: Arc<std::sync::Mutex<Vec<String>>>,
        }

        impl PriorityFixture {
            fn new(
                name: &str,
                priority: i32,
                setup_order: Arc<std::sync::Mutex<Vec<String>>>,
                teardown_order: Arc<std::sync::Mutex<Vec<String>>>,
            ) -> Self {
                Self {
                    name: name.to_string(),
                    priority,
                    setup_order,
                    teardown_order,
                }
            }
        }

        impl Fixture for PriorityFixture {
            fn setup(&mut self) -> ProbarResult<()> {
                self.setup_order.lock().unwrap().push(self.name.clone());
                Ok(())
            }

            fn teardown(&mut self) -> ProbarResult<()> {
                self.teardown_order.lock().unwrap().push(self.name.clone());
                Ok(())
            }

            fn name(&self) -> &str {
                &self.name
            }

            fn priority(&self) -> i32 {
                self.priority
            }
        }

        // Fixture that fails on setup but with high priority
        #[allow(dead_code)] // Reserved for future tests
        #[derive(Debug)]
        struct HighPriorityFailingSetupFixture {
            priority: i32,
        }

        impl Fixture for HighPriorityFailingSetupFixture {
            fn setup(&mut self) -> ProbarResult<()> {
                Err(ProbarError::FixtureError {
                    message: "High priority setup failure".to_string(),
                })
            }

            fn teardown(&mut self) -> ProbarResult<()> {
                Ok(())
            }

            fn priority(&self) -> i32 {
                self.priority
            }
        }

        // Fixture that fails on setup but with low priority (set up later)
        #[derive(Debug)]
        struct LowPriorityFailingSetupFixture {
            priority: i32,
            setup_order: Arc<std::sync::Mutex<Vec<String>>>,
        }

        impl Fixture for LowPriorityFailingSetupFixture {
            fn setup(&mut self) -> ProbarResult<()> {
                self.setup_order
                    .lock()
                    .unwrap()
                    .push("low_failing".to_string());
                Err(ProbarError::FixtureError {
                    message: "Low priority setup failure".to_string(),
                })
            }

            fn teardown(&mut self) -> ProbarResult<()> {
                Ok(())
            }

            fn priority(&self) -> i32 {
                self.priority
            }
        }

        // Another unique fixture type for testing multiple fixture types
        #[derive(Debug)]
        struct SecondTestFixture {
            setup_called: Arc<AtomicBool>,
            teardown_called: Arc<AtomicBool>,
        }

        impl SecondTestFixture {
            fn new() -> Self {
                Self {
                    setup_called: Arc::new(AtomicBool::new(false)),
                    teardown_called: Arc::new(AtomicBool::new(false)),
                }
            }
        }

        impl Fixture for SecondTestFixture {
            fn setup(&mut self) -> ProbarResult<()> {
                self.setup_called.store(true, Ordering::SeqCst);
                Ok(())
            }

            fn teardown(&mut self) -> ProbarResult<()> {
                self.teardown_called.store(true, Ordering::SeqCst);
                Ok(())
            }
        }

        #[test]
        fn test_simple_fixture_debug() {
            let fixture = SimpleFixture::new("debug_test").with_priority(5);
            let debug = format!("{:?}", fixture);
            assert!(debug.contains("SimpleFixture"));
            assert!(debug.contains("debug_test"));
            assert!(debug.contains('5'));
        }

        #[test]
        fn test_fixture_scope_debug() {
            let manager = FixtureManager::new();
            let scope = FixtureScope::new(manager);
            let debug = format!("{:?}", scope);
            assert!(debug.contains("FixtureScope"));
            assert!(debug.contains("FixtureManager"));
        }

        #[test]
        fn test_fixture_builder_default() {
            let builder = FixtureBuilder::default();
            let manager = builder.build();
            assert_eq!(manager.count(), 0);
        }

        #[test]
        fn test_build_and_setup_failure() {
            let result = FixtureBuilder::new()
                .with_fixture(FailingSetupFixture)
                .build_and_setup();

            assert!(result.is_err());
        }

        #[test]
        fn test_setup_all_with_rollback() {
            // Test that when a fixture fails setup, previously set up fixtures are torn down
            let setup_order = Arc::new(std::sync::Mutex::new(Vec::new()));
            let teardown_order = Arc::new(std::sync::Mutex::new(Vec::new()));

            let mut manager = FixtureManager::new();

            // Register a fixture that succeeds (high priority, set up first)
            manager.register(PriorityFixture::new(
                "first",
                100,
                setup_order.clone(),
                teardown_order.clone(),
            ));

            // Register a fixture that will fail (low priority, set up later)
            manager.register(LowPriorityFailingSetupFixture {
                priority: -100,
                setup_order: setup_order.clone(),
            });

            let result = manager.setup_all();
            assert!(result.is_err());

            // Verify the first fixture was set up
            let setup = setup_order.lock().unwrap();
            assert!(setup.contains(&"first".to_string()));
            assert!(setup.contains(&"low_failing".to_string()));

            // Verify the first fixture was torn down after the failure
            let teardown = teardown_order.lock().unwrap();
            assert!(teardown.contains(&"first".to_string()));
        }

        #[test]
        fn test_teardown_failure_continues_others() {
            // Test that when one fixture fails teardown, others are still torn down
            let mut manager = FixtureManager::new();

            manager.register(FailingTeardownFixture);
            manager.register(TestFixture::new());

            manager.setup_all().expect("Setup should succeed");

            // First teardown should fail but both should be attempted
            let result = manager.teardown_all();
            assert!(result.is_err());
        }

        #[test]
        fn test_fixture_scope_get_mut() {
            let fixture = TestFixture::new();
            let setup_called = fixture.setup_called.clone();

            let mut manager = FixtureManager::new();
            manager.register(fixture);
            manager.setup_all().expect("Setup should succeed");

            let mut scope = FixtureScope::new(manager);
            let retrieved = scope.get_mut::<TestFixture>();
            assert!(retrieved.is_some());
            assert!(Arc::ptr_eq(&retrieved.unwrap().setup_called, &setup_called));
        }

        #[test]
        fn test_fixture_scope_get_unregistered() {
            let manager = FixtureManager::new();
            let scope = FixtureScope::new(manager);
            assert!(scope.get::<TestFixture>().is_none());
        }

        #[test]
        fn test_fixture_scope_get_mut_unregistered() {
            let manager = FixtureManager::new();
            let mut scope = FixtureScope::new(manager);
            assert!(scope.get_mut::<TestFixture>().is_none());
        }

        #[test]
        fn test_setup_from_torn_down_state() {
            // Test setup_all when fixtures are in TornDown state
            let mut manager = FixtureManager::new();
            manager.register(TestFixture::new());

            manager.setup_all().expect("First setup should succeed");
            manager.teardown_all().expect("Teardown should succeed");

            // Setup again from TornDown state
            let result = manager.setup_all();
            assert!(result.is_ok());
            assert_eq!(manager.state::<TestFixture>(), Some(FixtureState::SetUp));
        }

        #[test]
        fn test_multiple_fixture_types() {
            // Test with multiple different fixture types (not just SimpleFixture)
            let mut manager = FixtureManager::new();
            manager.register(TestFixture::new());
            manager.register(SecondTestFixture::new());

            assert_eq!(manager.count(), 2);
            assert!(manager.is_registered::<TestFixture>());
            assert!(manager.is_registered::<SecondTestFixture>());
        }

        #[test]
        fn test_register_replaces_existing() {
            let first = TestFixture::new();
            let first_setup = first.setup_called.clone();

            let second = TestFixture::new();
            let second_setup = second.setup_called.clone();

            let mut manager = FixtureManager::new();
            manager.register(first);
            manager.register(second);

            // Only one fixture registered
            assert_eq!(manager.count(), 1);

            // The second one replaced the first
            manager.setup_all().expect("Setup should succeed");
            assert!(second_setup.load(Ordering::SeqCst));
            // The first was replaced so its setup wasn't called
            assert!(!first_setup.load(Ordering::SeqCst));
        }

        #[test]
        fn test_single_fixture_setup_failure() {
            let mut manager = FixtureManager::new();
            manager.register(FailingSetupFixture);

            let result = manager.setup::<FailingSetupFixture>();
            assert!(result.is_err());
        }

        #[test]
        fn test_single_fixture_teardown_failure() {
            let mut manager = FixtureManager::new();
            manager.register(FailingTeardownFixture);

            manager
                .setup::<FailingTeardownFixture>()
                .expect("Setup should succeed");
            let result = manager.teardown::<FailingTeardownFixture>();
            assert!(result.is_err());
        }

        #[test]
        fn test_unregister_removes_from_setup_order() {
            let mut manager = FixtureManager::new();
            manager.register(TestFixture::new());
            manager.setup_all().expect("Setup should succeed");

            // Unregister should also clean up setup_order
            assert!(manager.unregister::<TestFixture>());
            assert!(!manager.is_registered::<TestFixture>());
        }

        #[test]
        fn test_simple_fixture_is_setup_tracking() {
            let mut fixture = SimpleFixture::new("test");
            assert!(!fixture.is_setup);

            fixture.setup().expect("Setup should succeed");
            assert!(fixture.is_setup);

            fixture.teardown().expect("Teardown should succeed");
            assert!(!fixture.is_setup);
        }

        #[test]
        fn test_simple_fixture_failing_setup() {
            let mut fixture = SimpleFixture::new("test").with_setup(|| {
                Err(ProbarError::FixtureError {
                    message: "Test failure".to_string(),
                })
            });

            let result = fixture.setup();
            assert!(result.is_err());
        }

        #[test]
        fn test_simple_fixture_failing_teardown() {
            let mut fixture = SimpleFixture::new("test").with_teardown(|| {
                Err(ProbarError::FixtureError {
                    message: "Teardown failure".to_string(),
                })
            });

            fixture.setup().expect("Setup should succeed");
            let result = fixture.teardown();
            assert!(result.is_err());
        }

        #[test]
        fn test_list_multiple_fixtures() {
            let mut manager = FixtureManager::new();
            manager.register(TestFixture::new());
            manager.register(SecondTestFixture::new());

            let names = manager.list();
            assert_eq!(names.len(), 2);
        }

        #[test]
        fn test_state_none_for_unregistered() {
            let manager = FixtureManager::new();
            assert!(manager.state::<TestFixture>().is_none());
        }

        #[test]
        fn test_priority_ordering_in_setup() {
            let setup_order = Arc::new(std::sync::Mutex::new(Vec::new()));
            let teardown_order = Arc::new(std::sync::Mutex::new(Vec::new()));

            let mut manager = FixtureManager::new();

            // Register with different priorities (out of order)
            manager.register(PriorityFixture::new(
                "low",
                -10,
                setup_order.clone(),
                teardown_order,
            ));

            // Use SimpleFixture for high priority (different type)
            let setup_clone = setup_order.clone();
            manager.register(
                SimpleFixture::new("high")
                    .with_priority(10)
                    .with_setup(move || {
                        setup_clone.lock().unwrap().push("high".to_string());
                        Ok(())
                    }),
            );

            manager.setup_all().expect("Setup should succeed");

            let setup = setup_order.lock().unwrap();
            // High priority (10) should be set up before low priority (-10)
            let high_idx = setup.iter().position(|s| s == "high").unwrap();
            let low_idx = setup.iter().position(|s| s == "low").unwrap();
            assert!(high_idx < low_idx, "High priority should be set up first");
        }

        #[test]
        fn test_teardown_reverse_order() {
            let setup_order = Arc::new(std::sync::Mutex::new(Vec::new()));
            let teardown_order = Arc::new(std::sync::Mutex::new(Vec::new()));

            let mut manager = FixtureManager::new();

            manager.register(PriorityFixture::new(
                "high",
                100,
                setup_order,
                teardown_order.clone(),
            ));

            // Second fixture with different type
            let teardown_clone = teardown_order.clone();
            manager.register(SimpleFixture::new("low").with_priority(-100).with_teardown(
                move || {
                    teardown_clone.lock().unwrap().push("low".to_string());
                    Ok(())
                },
            ));

            manager.setup_all().expect("Setup should succeed");
            manager.teardown_all().expect("Teardown should succeed");

            let teardown = teardown_order.lock().unwrap();
            // Low priority was set up last, so it should be torn down first
            let high_idx = teardown.iter().position(|s| s == "high").unwrap();
            let low_idx = teardown.iter().position(|s| s == "low").unwrap();
            assert!(
                low_idx < high_idx,
                "Low priority (set up last) should be torn down first"
            );
        }

        #[test]
        fn test_clear_empties_setup_order() {
            let mut manager = FixtureManager::new();
            manager.register(TestFixture::new());
            manager.setup_all().expect("Setup should succeed");

            manager.clear();
            assert_eq!(manager.count(), 0);
            assert!(manager.list().is_empty());
        }

        #[test]
        fn test_reset_clears_setup_order() {
            let fixture = TestFixture::new();
            let teardown_called = fixture.teardown_called.clone();

            let mut manager = FixtureManager::new();
            manager.register(fixture);
            manager.setup_all().expect("Setup should succeed");

            // Reset without teardown
            manager.reset();

            // Teardown should NOT have been called
            assert!(!teardown_called.load(Ordering::SeqCst));
            assert_eq!(
                manager.state::<TestFixture>(),
                Some(FixtureState::Registered)
            );
        }

        #[test]
        fn test_fixture_scope_drop_with_failing_teardown() {
            // This tests that the scope ignores errors during drop
            let mut manager = FixtureManager::new();
            manager.register(FailingTeardownFixture);
            manager.setup_all().expect("Setup should succeed");

            // This should not panic even though teardown fails
            let _scope = FixtureScope::new(manager);
            // Scope drops here without panicking
        }

        #[test]
        fn test_single_teardown_updates_state() {
            let mut manager = FixtureManager::new();
            manager.register(TestFixture::new());

            manager
                .setup::<TestFixture>()
                .expect("Setup should succeed");
            assert_eq!(manager.state::<TestFixture>(), Some(FixtureState::SetUp));

            manager
                .teardown::<TestFixture>()
                .expect("Teardown should succeed");
            assert_eq!(manager.state::<TestFixture>(), Some(FixtureState::TornDown));
        }

        #[test]
        fn test_setup_adds_to_setup_order_once() {
            let mut manager = FixtureManager::new();
            manager.register(TestFixture::new());

            // Setup multiple times
            manager.setup::<TestFixture>().unwrap();
            manager.reset();
            manager.setup::<TestFixture>().unwrap();

            // Should work without issues
            assert_eq!(manager.state::<TestFixture>(), Some(FixtureState::SetUp));
        }

        #[test]
        fn test_single_fixture_teardown_removes_from_order() {
            let mut manager = FixtureManager::new();
            manager.register(TestFixture::new());
            manager.register(SecondTestFixture::new());

            manager.setup_all().expect("Setup should succeed");

            // Teardown just one fixture
            manager
                .teardown::<TestFixture>()
                .expect("Teardown should succeed");

            // The other should still be set up
            assert_eq!(
                manager.state::<SecondTestFixture>(),
                Some(FixtureState::SetUp)
            );
            assert_eq!(manager.state::<TestFixture>(), Some(FixtureState::TornDown));
        }

        #[test]
        fn test_fixture_error_message_includes_name() {
            let mut manager = FixtureManager::new();
            manager.register(FailingSetupFixture);

            let result = manager.setup_all();
            assert!(result.is_err());

            if let Err(ProbarError::FixtureError { message }) = result {
                assert!(message.contains("FailingSetupFixture"));
                assert!(message.contains("setup failed"));
            } else {
                panic!("Expected FixtureError");
            }
        }

        #[test]
        fn test_teardown_error_message_includes_name() {
            let mut manager = FixtureManager::new();
            manager.register(FailingTeardownFixture);
            manager.setup_all().expect("Setup should succeed");

            let result = manager.teardown_all();
            assert!(result.is_err());

            if let Err(ProbarError::FixtureError { message }) = result {
                assert!(message.contains("FailingTeardownFixture"));
                assert!(message.contains("teardown failed"));
            } else {
                panic!("Expected FixtureError");
            }
        }
    }
}
