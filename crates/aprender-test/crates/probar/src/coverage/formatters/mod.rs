//! Coverage Report Formatters (Feature 11, 12, 13)
//!
//! LCOV, HTML, and Cobertura XML format generators for CI integration.
//!
//! ## EXTREME TDD: Tests written FIRST per spec

mod cobertura;
mod html;
mod lcov;

pub use cobertura::CoberturaFormatter;
pub use html::{HtmlFormatter, HtmlReportConfig, Theme};
pub use lcov::LcovFormatter;
