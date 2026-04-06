//! Playbook Testing: State Machine Verification
//!
//! This module implements YAML-driven state machine testing with:
//! - SCXML-inspired state definitions
//! - Transition-based assertions
//! - O(n) complexity verification via curve fitting
//! - M1-M5 mutation testing for falsification
//!
//! # References
//! - W3C SCXML: <https://www.w3.org/TR/scxml/>
//! - Lamport, "Specifying Systems" (TLA+)
//! - Goldsmith et al., "Measuring Empirical Computational Complexity" (ESEC/FSE 2007)
//! - Fabbri et al., "Mutation Testing Applied to Statecharts" (ISSRE 1999)
//!
//! # Example
//!
//! ```yaml
//! version: "1.0"
//! machine:
//!   id: "login_flow"
//!   initial: "logged_out"
//!   states:
//!     logged_out:
//!       id: "logged_out"
//!       invariants:
//!         - description: "Login button visible"
//!           condition: "document.querySelector('#login') !== null"
//!     logged_in:
//!       id: "logged_in"
//!       final_state: true
//!   transitions:
//!     - id: "do_login"
//!       from: "logged_out"
//!       to: "logged_in"
//!       event: "login_success"
//!       assertions:
//!         - type: element_exists
//!           selector: "#welcome"
//! ```

pub mod complexity;
pub mod executor;
pub mod mutation;
pub mod runner;
pub mod schema;
pub mod state_machine;

// Re-export primary types
pub use complexity::{check_complexity_violation, ComplexityAnalyzer, ComplexityResult};
pub use executor::{
    ActionExecutor, AssertionFailure, ExecutionResult, ExecutorError, PlaybookExecutor,
};
pub use mutation::{
    calculate_mutation_score, MutantResult, MutationClass, MutationGenerator, MutationScore,
};
pub use runner::{to_svg, AssertionCheckResult, PlaybookRunResult, PlaybookRunner, StepResult};
pub use schema::{
    Action, ActionSpec, Assertion, ComplexityAssertion, ComplexityClass, FalsificationConfig,
    ForbiddenTransition, Invariant, MutationDef, OutputAssertion, PathAssertion, PerformanceBudget,
    Playbook, PlaybookAction, PlaybookAssertions, PlaybookError, PlaybookStep, PlaybookSteps,
    State, StateMachine, Transition, VariableCapture, WaitCondition,
};
pub use state_machine::{
    to_dot, DeterminismInfo, IssueSeverity, ReachabilityInfo, StateMachineValidator,
    ValidationIssue, ValidationResult,
};
