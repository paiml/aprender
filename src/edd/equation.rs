//! Governing equation definitions and traits for EDD.
//!
//! Every simulation in the EDD framework must be grounded in a mathematically
//! verified governing equation. This module provides the traits and types
//! for defining, documenting, and validating these equations.

use std::fmt;

/// Classification of equation types for domain organization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EquationClass {
    /// Conservation laws (mass, energy, momentum)
    Conservation,
    /// Queueing theory equations
    Queueing,
    /// Statistical mechanics / thermodynamics
    Statistical,
    /// Inventory and supply chain
    Inventory,
    /// Optimization and control
    Optimization,
    /// Machine learning / inference
    MachineLearning,
}

impl fmt::Display for EquationClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Conservation => write!(f, "Conservation"),
            Self::Queueing => write!(f, "Queueing Theory"),
            Self::Statistical => write!(f, "Statistical Mechanics"),
            Self::Inventory => write!(f, "Inventory Management"),
            Self::Optimization => write!(f, "Optimization"),
            Self::MachineLearning => write!(f, "Machine Learning"),
        }
    }
}

/// A variable in a governing equation with its metadata.
#[derive(Debug, Clone)]
pub struct EquationVariable {
    /// Symbol used in the equation (e.g., "λ", "L", "W")
    pub symbol: String,
    /// Human-readable name
    pub name: String,
    /// Physical units (e.g., "items/time", "seconds")
    pub units: String,
    /// Description of what this variable represents
    pub description: String,
    /// Valid range constraints
    pub constraints: Option<VariableConstraints>,
}

/// Constraints on variable values.
#[derive(Debug, Clone)]
pub struct VariableConstraints {
    /// Minimum value (inclusive)
    pub min: Option<f64>,
    /// Maximum value (inclusive)
    pub max: Option<f64>,
    /// Must be strictly positive
    pub positive: bool,
    /// Must be an integer
    pub integer: bool,
}

impl EquationVariable {
    /// Create a new equation variable.
    #[must_use]
    pub fn new(symbol: &str, name: &str, units: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            name: name.to_string(),
            units: units.to_string(),
            description: String::new(),
            constraints: None,
        }
    }

    /// Add a description.
    #[must_use]
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    /// Add constraints.
    #[must_use]
    pub fn with_constraints(mut self, constraints: VariableConstraints) -> Self {
        self.constraints = Some(constraints);
        self
    }
}

/// A citation for the governing equation's source.
#[derive(Debug, Clone)]
pub struct Citation {
    /// List of author names
    pub authors: Vec<String>,
    /// Publication title
    pub title: String,
    /// Journal or conference name
    pub venue: String,
    /// Publication year
    pub year: u32,
    /// DOI if available
    pub doi: Option<String>,
    /// Page numbers if applicable
    pub pages: Option<String>,
}

impl Citation {
    /// Create a new citation.
    #[must_use]
    pub fn new(authors: &[&str], venue: &str, year: u32) -> Self {
        Self {
            authors: authors.iter().map(|s| (*s).to_string()).collect(),
            title: String::new(),
            venue: venue.to_string(),
            year,
            doi: None,
            pages: None,
        }
    }

    /// Add the publication title.
    #[must_use]
    pub fn with_title(mut self, title: &str) -> Self {
        self.title = title.to_string();
        self
    }

    /// Add DOI.
    #[must_use]
    pub fn with_doi(mut self, doi: &str) -> Self {
        self.doi = Some(doi.to_string());
        self
    }

    /// Add page numbers.
    #[must_use]
    pub fn with_pages(mut self, pages: &str) -> Self {
        self.pages = Some(pages.to_string());
        self
    }
}

impl fmt::Display for Citation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let authors = self.authors.join(", ");
        write!(
            f,
            "{authors} ({}) \"{}\", {}",
            self.year, self.title, self.venue
        )?;
        if let Some(ref pages) = self.pages {
            write!(f, ", pp. {pages}")?;
        }
        if let Some(ref doi) = self.doi {
            write!(f, ", doi:{doi}")?;
        }
        Ok(())
    }
}

/// Trait for governing equations that ground simulations.
///
/// Every simulation in the EDD framework must implement this trait
/// to provide mathematical foundation and falsifiability.
pub trait GoverningEquation {
    /// Get the LaTeX representation of the equation.
    fn latex(&self) -> &str;

    /// Get the equation's classification.
    fn class(&self) -> EquationClass;

    /// Get the primary citation for this equation.
    fn citation(&self) -> Citation;

    /// Get all variables in this equation.
    fn variables(&self) -> Vec<EquationVariable>;

    /// Get a human-readable description.
    fn description(&self) -> &str;

    /// Get the equation name.
    fn name(&self) -> &'static str;

    /// Validate that a set of values satisfies the equation within tolerance.
    ///
    /// Returns `Ok(())` if the equation holds, `Err` with explanation otherwise.
    ///
    /// # Errors
    /// Returns error message if the values don't satisfy the equation within tolerance.
    fn validate_consistency(&self, values: &[(&str, f64)], tolerance: f64) -> Result<(), String>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equation_variable_builder() {
        let var = EquationVariable::new("λ", "arrival_rate", "items/hour")
            .with_description("Rate at which items arrive to the system")
            .with_constraints(VariableConstraints {
                min: Some(0.0),
                max: None,
                positive: true,
                integer: false,
            });

        assert_eq!(var.symbol, "λ");
        assert_eq!(var.name, "arrival_rate");
        assert_eq!(var.units, "items/hour");
        assert!(!var.description.is_empty());
        assert!(var.constraints.is_some());
    }

    #[test]
    fn test_citation_builder() {
        let cite = Citation::new(&["Little, J.D.C."], "Operations Research", 1961)
            .with_title("A Proof for the Queuing Formula: L = λW")
            .with_doi("10.1287/opre.9.3.383");

        assert_eq!(cite.authors.len(), 1);
        assert_eq!(cite.year, 1961);
        assert!(cite.doi.is_some());
    }

    #[test]
    fn test_equation_class_display() {
        assert_eq!(EquationClass::Queueing.to_string(), "Queueing Theory");
        assert_eq!(EquationClass::Conservation.to_string(), "Conservation");
    }

    #[test]
    fn test_all_equation_classes_display() {
        assert_eq!(
            EquationClass::Statistical.to_string(),
            "Statistical Mechanics"
        );
        assert_eq!(EquationClass::Inventory.to_string(), "Inventory Management");
        assert_eq!(EquationClass::Optimization.to_string(), "Optimization");
        assert_eq!(
            EquationClass::MachineLearning.to_string(),
            "Machine Learning"
        );
    }

    #[test]
    fn test_citation_display() {
        let cite = Citation::new(&["Author A", "Author B"], "Test Journal", 2020)
            .with_title("Test Title")
            .with_pages("1-10");
        let display = format!("{cite}");
        assert!(display.contains("Author A"));
        assert!(display.contains("Author B"));
        assert!(display.contains("2020"));
        assert!(display.contains("Test Title"));
        assert!(display.contains("1-10"));
    }

    #[test]
    fn test_citation_display_with_doi() {
        let cite = Citation::new(&["Author"], "Journal", 2021)
            .with_title("Title")
            .with_doi("10.1234/test");
        let display = format!("{cite}");
        assert!(display.contains("doi:10.1234/test"));
    }

    #[test]
    fn test_citation_with_pages() {
        let cite = Citation::new(&["Test"], "Venue", 2022).with_pages("100-200");
        assert!(cite.pages.is_some());
        assert_eq!(cite.pages.as_deref(), Some("100-200"));
    }

    #[test]
    fn test_variable_constraints() {
        let constraints = VariableConstraints {
            min: Some(-5.0),
            max: Some(5.0),
            positive: false,
            integer: true,
        };
        assert_eq!(constraints.min, Some(-5.0));
        assert_eq!(constraints.max, Some(5.0));
        assert!(!constraints.positive);
        assert!(constraints.integer);
    }

    #[test]
    fn test_equation_variable_without_constraints() {
        let var = EquationVariable::new("x", "variable", "units");
        assert!(var.constraints.is_none());
        assert!(var.description.is_empty());
    }
}
