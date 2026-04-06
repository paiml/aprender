//! Grammar-constrained generation for structured output
//!
//! Implements GBNF-style grammar constraints for LLM generation.
//! Supports JSON schema validation and custom grammar rules.
//!
//! Reference: llama.cpp grammar implementation
//! - GBNF format: Backus-Naur Form with extensions
//! - Token masking: Efficiently filter invalid tokens
//! - State machine: Track grammar state during generation

use crate::error::{RealizarError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// =============================================================================
// GRAMMAR RULE TYPES
// =============================================================================

/// Grammar rule element types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GrammarElement {
    /// Literal character
    Char(char),
    /// Character range [a-z]
    CharRange(char, char),
    /// Reference to another rule
    RuleRef(String),
    /// Negated character set [^...]
    CharNot(Vec<char>),
    /// Any character
    Any,
    /// End of rule
    End,
}

/// Alternative production in a rule
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrammarAlternative {
    /// Sequence of elements in this alternative
    pub elements: Vec<GrammarElement>,
}

impl GrammarAlternative {
    /// Create new alternative from elements
    pub fn new(elements: Vec<GrammarElement>) -> Self {
        Self { elements }
    }

    /// Create single-character alternative
    pub fn char(c: char) -> Self {
        Self {
            elements: vec![GrammarElement::Char(c)],
        }
    }

    /// Check if this alternative is empty (epsilon production)
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }
}

/// Grammar rule with one or more alternatives
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrammarRule {
    /// Rule name
    pub name: String,
    /// Alternative productions
    pub alternatives: Vec<GrammarAlternative>,
}

impl GrammarRule {
    /// Create new rule with given alternatives
    pub fn new(name: impl Into<String>, alternatives: Vec<GrammarAlternative>) -> Self {
        Self {
            name: name.into(),
            alternatives,
        }
    }

    /// Create rule with single alternative
    pub fn single(name: impl Into<String>, elements: Vec<GrammarElement>) -> Self {
        Self {
            name: name.into(),
            alternatives: vec![GrammarAlternative::new(elements)],
        }
    }
}

// =============================================================================
// GRAMMAR DEFINITION
// =============================================================================

/// Complete grammar definition
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Grammar {
    /// Rules by name
    rules: HashMap<String, GrammarRule>,
    /// Root rule name
    root: String,
}

impl Grammar {
    /// Create empty grammar
    pub fn new() -> Self {
        Self {
            rules: HashMap::new(),
            root: String::new(),
        }
    }

    /// Create grammar with root rule
    pub fn with_root(root: impl Into<String>) -> Self {
        Self {
            rules: HashMap::new(),
            root: root.into(),
        }
    }

    /// Add a rule to the grammar
    pub fn add_rule(&mut self, rule: GrammarRule) {
        if self.root.is_empty() {
            self.root.clone_from(&rule.name);
        }
        self.rules.insert(rule.name.clone(), rule);
    }

    /// Get rule by name
    pub fn get_rule(&self, name: &str) -> Option<&GrammarRule> {
        self.rules.get(name)
    }

    /// Get root rule name
    pub fn root(&self) -> &str {
        &self.root
    }

    /// Set root rule
    pub fn set_root(&mut self, root: impl Into<String>) {
        self.root = root.into();
    }

    /// Get all rule names
    pub fn rule_names(&self) -> impl Iterator<Item = &String> {
        self.rules.keys()
    }

    /// Number of rules
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// Check if grammar is empty
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Validate grammar has required rules
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfiguration` if:
    /// - Grammar has no root rule
    /// - Root rule is not defined in the grammar
    /// - Any rule references an undefined rule
    pub fn validate(&self) -> Result<()> {
        if self.root.is_empty() {
            return Err(RealizarError::InvalidConfiguration(
                "Grammar has no root rule".to_string(),
            ));
        }

        if !self.rules.contains_key(&self.root) {
            return Err(RealizarError::InvalidConfiguration(format!(
                "Root rule '{}' not found in grammar",
                self.root
            )));
        }

        // Check all rule references are valid
        for rule in self.rules.values() {
            for alt in &rule.alternatives {
                for elem in &alt.elements {
                    if let GrammarElement::RuleRef(ref_name) = elem {
                        if !self.rules.contains_key(ref_name) {
                            return Err(RealizarError::InvalidConfiguration(format!(
                                "Rule '{}' references undefined rule '{}'",
                                rule.name, ref_name
                            )));
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// =============================================================================
// GRAMMAR STATE MACHINE
// =============================================================================

/// State in the grammar state machine
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GrammarState {
    /// Current rule being matched
    pub rule: String,
    /// Alternative index within rule
    pub alt_idx: usize,
    /// Position within alternative
    pub elem_idx: usize,
    /// Stack of parent states (for rule references)
    pub stack: Vec<(String, usize, usize)>,
}

impl GrammarState {
    /// Create initial state from root rule
    pub fn initial(root: &str) -> Self {
        Self {
            rule: root.to_string(),
            alt_idx: 0,
            elem_idx: 0,
            stack: Vec::new(),
        }
    }

    /// Check if state is at end of rule
    pub fn is_complete(&self, grammar: &Grammar) -> bool {
        if let Some(rule) = grammar.get_rule(&self.rule) {
            if self.alt_idx < rule.alternatives.len() {
                let alt = &rule.alternatives[self.alt_idx];
                return self.elem_idx >= alt.elements.len() && self.stack.is_empty();
            }
        }
        false
    }

    /// Get current element being matched
    pub fn current_element<'a>(&self, grammar: &'a Grammar) -> Option<&'a GrammarElement> {
        grammar.get_rule(&self.rule).and_then(|rule| {
            rule.alternatives
                .get(self.alt_idx)
                .and_then(|alt| alt.elements.get(self.elem_idx))
        })
    }
}

/// Grammar state machine for tracking generation progress
#[derive(Debug, Clone)]
pub struct GrammarStateMachine {
    /// The grammar being enforced
    grammar: Grammar,
    /// Current possible states (NFA-style)
    states: Vec<GrammarState>,
    /// Generated characters so far
    generated: String,
}

include!("grammar_state_machine.rs");
include!("mod_add_schema.rs");
include!("tool.rs");
include!("grammar.rs");
