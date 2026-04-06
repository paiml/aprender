//! Framework types: Category, TestResult, FalsificationTest, FalsificationReport

use crate::parser::{PtxModule, SourceLocation};

/// Test category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Category {
    /// Syntax validity (F001-F010)
    SyntaxValidity,
    /// Type safety (F011-F020)
    TypeSafety,
    /// Address space (F021-F035)
    AddressSpace,
    /// Barrier safety (F036-F050)
    BarrierSafety,
    /// Memory model (F051-F060)
    MemoryModel,
    /// Control flow (F061-F070)
    ControlFlow,
    /// Data flow (F071-F080)
    DataFlow,
    /// Known bugs (F081-F090)
    KnownBugs,
    /// Performance (F091-F095)
    Performance,
    /// Instrumentation (F096-F100)
    Instrumentation,
}

impl Category {
    /// Get all categories
    pub fn all() -> &'static [Category] {
        &[
            Category::SyntaxValidity,
            Category::TypeSafety,
            Category::AddressSpace,
            Category::BarrierSafety,
            Category::MemoryModel,
            Category::ControlFlow,
            Category::DataFlow,
            Category::KnownBugs,
            Category::Performance,
            Category::Instrumentation,
        ]
    }
}

impl std::fmt::Display for Category {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Category::SyntaxValidity => write!(f, "Syntax Validity"),
            Category::TypeSafety => write!(f, "Type Safety"),
            Category::AddressSpace => write!(f, "Address Space"),
            Category::BarrierSafety => write!(f, "Barrier Safety"),
            Category::MemoryModel => write!(f, "Memory Model"),
            Category::ControlFlow => write!(f, "Control Flow"),
            Category::DataFlow => write!(f, "Data Flow"),
            Category::KnownBugs => write!(f, "Known Bugs"),
            Category::Performance => write!(f, "Performance"),
            Category::Instrumentation => write!(f, "Instrumentation"),
        }
    }
}

/// Result of a falsification test
#[derive(Debug, Clone)]
pub enum TestResult {
    /// Test passed (hypothesis not refuted)
    Pass,
    /// Test failed (hypothesis refuted)
    Fail {
        /// Evidence of failure
        evidence: String,
        /// Location of failure
        location: Option<SourceLocation>,
    },
    /// Test not applicable
    NotApplicable,
}

impl TestResult {
    /// Check if the test passed
    pub fn is_pass(&self) -> bool {
        matches!(self, TestResult::Pass)
    }

    /// Check if the test failed
    pub fn is_fail(&self) -> bool {
        matches!(self, TestResult::Fail { .. })
    }
}

/// A single falsification test
pub struct FalsificationTest {
    /// Test ID (e.g., "F001")
    pub id: String,
    /// Category
    pub category: Category,
    /// Description
    pub description: String,
    /// Points for passing
    pub points: u8,
    /// Test function
    pub(crate) test_fn: Box<dyn Fn(&PtxModule) -> TestResult + Send + Sync>,
}

impl FalsificationTest {
    /// Create a new falsification test
    pub fn new<F>(id: &str, category: Category, description: &str, points: u8, test_fn: F) -> Self
    where
        F: Fn(&PtxModule) -> TestResult + Send + Sync + 'static,
    {
        Self {
            id: id.to_string(),
            category,
            description: description.to_string(),
            points,
            test_fn: Box::new(test_fn),
        }
    }

    /// Run the test
    pub fn run(&self, module: &PtxModule) -> TestResult {
        (self.test_fn)(module)
    }
}

impl std::fmt::Debug for FalsificationTest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FalsificationTest")
            .field("id", &self.id)
            .field("category", &self.category)
            .field("description", &self.description)
            .field("points", &self.points)
            .finish()
    }
}

impl Clone for FalsificationTest {
    fn clone(&self) -> Self {
        // Cannot clone the test function, so create a stub
        Self {
            id: self.id.clone(),
            category: self.category,
            description: self.description.clone(),
            points: self.points,
            test_fn: Box::new(|_| TestResult::NotApplicable),
        }
    }
}

/// Falsification report
#[derive(Debug, Clone)]
pub struct FalsificationReport {
    /// Test results
    pub results: Vec<(String, Category, String, TestResult)>,
    /// Score (0-100)
    pub score: f64,
    /// Earned points
    pub earned_points: u32,
    /// Total points
    pub total_points: u32,
    /// Confidence (0-1)
    pub confidence: f64,
}

impl FalsificationReport {
    /// Get categories with all tests passed
    pub fn categories_with_all_tests_passed(&self) -> usize {
        Category::all()
            .iter()
            .filter(|&cat| {
                self.results
                    .iter()
                    .filter(|(_, c, _, _)| c == cat)
                    .all(|(_, _, _, r)| r.is_pass() || matches!(r, TestResult::NotApplicable))
            })
            .count()
    }

    /// Check if all critical bugs are absent
    pub fn critical_bugs_absent(&self) -> bool {
        // F082 is the remaining critical correctness test
        self.results
            .iter()
            .filter(|(id, _, _, _)| id == "F082")
            .all(|(_, _, _, r)| r.is_pass())
    }

    /// Check if any critical bugs were detected
    pub fn has_critical_bugs(&self) -> bool {
        !self.critical_bugs_absent()
    }

    /// Get failed tests
    pub fn failed_tests(&self) -> Vec<&(String, Category, String, TestResult)> {
        self.results
            .iter()
            .filter(|(_, _, _, r)| r.is_fail())
            .collect()
    }
}
