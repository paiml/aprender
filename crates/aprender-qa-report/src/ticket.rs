//! Upstream Ticket Generator
//!
//! Creates tickets for aprender repository when bugs are found.
//! Generates structured issue reports with reproduction steps.

use aprender_qa_runner::{classify_failure, Evidence, Outcome};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::popperian::PopperianScore;

/// Ticket priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TicketPriority {
    /// Critical - blocks release
    P0,
    /// High - should fix before release
    P1,
    /// Medium - fix in next release
    P2,
    /// Low - nice to have
    P3,
}

/// Display ticket priority as human-readable label
impl std::fmt::Display for TicketPriority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::P0 => write!(f, "P0-Critical"),
            Self::P1 => write!(f, "P1-High"),
            Self::P2 => write!(f, "P2-Medium"),
            Self::P3 => write!(f, "P3-Low"),
        }
    }
}

/// Ticket category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TicketCategory {
    /// Bug in functionality
    Bug,
    /// Performance issue
    Performance,
    /// Crash or instability
    Crash,
    /// Compatibility issue
    Compatibility,
    /// Edge case handling
    EdgeCase,
    /// Regression from previous version
    Regression,
}

/// Display ticket category as lowercase kebab-case label
impl std::fmt::Display for TicketCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bug => write!(f, "bug"),
            Self::Performance => write!(f, "performance"),
            Self::Crash => write!(f, "crash"),
            Self::Compatibility => write!(f, "compatibility"),
            Self::EdgeCase => write!(f, "edge-case"),
            Self::Regression => write!(f, "regression"),
        }
    }
}

/// Upstream ticket for aprender
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpstreamTicket {
    /// Ticket title
    pub title: String,
    /// Ticket body (markdown)
    pub body: String,
    /// Priority
    pub priority: TicketPriority,
    /// Category
    pub category: TicketCategory,
    /// Labels for GitHub
    pub labels: Vec<String>,
    /// Related gate ID
    pub gate_id: String,
    /// Model that triggered this
    pub model_id: String,
    /// Is this a black swan event?
    pub is_black_swan: bool,
    /// Upstream fixture path for reproduction (§3.5)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upstream_fixture: Option<String>,
    /// Pygmy builder function name (§3.5)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pygmy_builder: Option<String>,
}

/// Methods for generating upstream ticket CLI commands
impl UpstreamTicket {
    /// Generate GitHub CLI command to create this ticket
    #[must_use]
    pub fn to_gh_command(&self, repo: &str) -> String {
        let labels = self.labels.join(",");
        format!(
            r#"gh issue create --repo {} --title "{}" --body "{}" --label "{}""#,
            repo,
            self.title.replace('"', r#"\""#),
            self.body.replace('"', r#"\""#).replace('\n', "\\n"),
            labels
        )
    }
}

/// Gate ID patterns → priority mapping. First match wins.
/// The P0 entry handles the "-P0-" substring; the "G" prefix is checked separately.
const PRIORITY_RULES: &[(&str, TicketPriority)] = &[
    ("-P0-", TicketPriority::P0),
    ("-P1-", TicketPriority::P1),
    ("-P2-", TicketPriority::P2),
];

/// Gate ID substring → ticket category mapping. First match wins.
const CATEGORY_RULES: &[(&[&str], TicketCategory)] = &[
    (&["PERF"], TicketCategory::Performance),
    (&["STAB", "CRASH"], TicketCategory::Crash),
    (&["COMP"], TicketCategory::Compatibility),
    (&["EDGE"], TicketCategory::EdgeCase),
    (&["REGR"], TicketCategory::Regression),
];

/// Ticket generator
#[derive(Debug, Default)]
pub struct TicketGenerator {
    /// Repository to create tickets in
    repo: String,
    /// Minimum occurrences before creating ticket
    min_occurrences: usize,
    /// Only create tickets for black swans
    black_swans_only: bool,
}

/// Ticket generation methods for evidence and Popperian analysis
impl TicketGenerator {
    /// Create a new ticket generator
    #[must_use]
    pub fn new(repo: impl Into<String>) -> Self {
        Self {
            repo: repo.into(),
            min_occurrences: 1,
            black_swans_only: false,
        }
    }

    /// Set minimum occurrences before creating ticket
    #[must_use]
    pub fn with_min_occurrences(mut self, min: usize) -> Self {
        self.min_occurrences = min;
        self
    }

    /// Only create tickets for black swan events
    #[must_use]
    pub fn black_swans_only(mut self) -> Self {
        self.black_swans_only = true;
        self
    }

    /// Generate tickets from evidence
    #[must_use]
    pub fn generate_from_evidence(&self, evidence: &[Evidence]) -> Vec<UpstreamTicket> {
        let mut tickets = Vec::new();

        // Group failures by gate_id
        let mut failure_groups: std::collections::HashMap<String, Vec<&Evidence>> =
            std::collections::HashMap::new();

        for e in evidence {
            if e.outcome.is_fail() {
                failure_groups.entry(e.gate_id.clone()).or_default().push(e);
            }
        }

        for (_gate_id, failures) in failure_groups {
            if failures.len() < self.min_occurrences {
                continue;
            }

            let first = failures[0];
            let is_black_swan = first.outcome == Outcome::Crashed;

            if self.black_swans_only && !is_black_swan {
                continue;
            }

            let ticket = self.create_ticket(first, failures.len(), is_black_swan);
            tickets.push(ticket);
        }

        tickets
    }

    /// Generate tickets from Popperian analysis
    #[must_use]
    pub fn generate_from_popperian(&self, popperian: &PopperianScore) -> Vec<UpstreamTicket> {
        let mut tickets = Vec::new();

        for falsification in &popperian.falsifications {
            if falsification.occurrence_count < self.min_occurrences {
                continue;
            }

            if self.black_swans_only && !falsification.is_black_swan {
                continue;
            }

            let priority = if falsification.is_black_swan {
                TicketPriority::P0
            } else if falsification.severity >= 4 {
                TicketPriority::P1
            } else if falsification.severity >= 3 {
                TicketPriority::P2
            } else {
                TicketPriority::P3
            };

            let category = self.determine_category(&falsification.gate_id);

            let title = format!(
                "[QA] {}: {}",
                falsification.gate_id, falsification.hypothesis
            );

            let body = format!(
                r#"## Summary

Automated QA testing discovered a falsification of the hypothesis: **{}**

## Details

- **Gate ID**: `{}`
- **Model**: `{}`
- **Severity**: {}/5
- **Occurrences**: {}
- **Black Swan**: {}

## Evidence

```
{}
```

## Reproduction

This issue was detected by `apr-model-qa-playbook` during automated qualification testing.

## Labels

- `{}` (priority)
- `{}` (category)
- `qa-automated`
"#,
                falsification.hypothesis,
                falsification.gate_id,
                popperian.model_id,
                falsification.severity,
                falsification.occurrence_count,
                if falsification.is_black_swan {
                    "Yes"
                } else {
                    "No"
                },
                falsification.evidence,
                priority,
                category,
            );

            let mut labels = vec![
                format!("priority:{}", priority),
                category.to_string(),
                "qa-automated".to_string(),
            ];

            if falsification.is_black_swan {
                labels.push("black-swan".to_string());
            }

            tickets.push(UpstreamTicket {
                title,
                body,
                priority,
                category,
                labels,
                gate_id: falsification.gate_id.clone(),
                model_id: popperian.model_id.clone(),
                is_black_swan: falsification.is_black_swan,
                upstream_fixture: None,
                pygmy_builder: None,
            });
        }

        tickets
    }

    /// Create a ticket from evidence
    fn create_ticket(
        &self,
        evidence: &Evidence,
        occurrence_count: usize,
        is_black_swan: bool,
    ) -> UpstreamTicket {
        let priority = self.determine_priority(evidence, is_black_swan);
        let category = self.determine_category(&evidence.gate_id);

        let title = format!(
            "[QA] {}: {} failure in {} mode",
            evidence.gate_id,
            match evidence.outcome {
                Outcome::Crashed => "Crash",
                Outcome::Falsified => "Assertion",
                Outcome::Timeout => "Timeout",
                _ => "Test",
            },
            evidence.scenario.modality
        );

        let body = format!(
            r#"## Summary

Automated QA testing discovered a failure in `apr-cli`.

## Details

- **Gate ID**: `{}`
- **Model**: `{}`
- **Modality**: `{}`
- **Backend**: `{}`
- **Format**: `{}`
- **Occurrences**: {}
- **Black Swan**: {}

## Scenario

```
Prompt: {}
Seed: {}
Temperature: {}
Max Tokens: {}
```

## Output

```
{}
```

## Error

```
{}
```

## Reproduction

```bash
{}
```

## Environment

- **Host**: `{}`
- **OS**: `{}`
- **APR Version**: `{}`

## Labels

- `{}` (priority)
- `{}` (category)
- `qa-automated`
"#,
            evidence.gate_id,
            evidence.scenario.model,
            evidence.scenario.modality,
            evidence.scenario.backend,
            evidence.scenario.format,
            occurrence_count,
            if is_black_swan { "Yes" } else { "No" },
            evidence.scenario.prompt,
            evidence.scenario.seed,
            evidence.scenario.temperature,
            evidence.scenario.max_tokens,
            evidence.output,
            evidence.stderr.as_deref().unwrap_or("N/A"),
            evidence.scenario.to_command("model.gguf"),
            evidence.host.hostname,
            evidence.host.os,
            evidence.host.apr_version,
            priority,
            category,
        );

        let mut labels = vec![
            format!("priority:{}", priority),
            category.to_string(),
            "qa-automated".to_string(),
            format!("modality:{}", evidence.scenario.modality),
            format!("backend:{}", evidence.scenario.backend),
        ];

        if is_black_swan {
            labels.push("black-swan".to_string());
        }

        UpstreamTicket {
            title,
            body,
            priority,
            category,
            labels,
            gate_id: evidence.gate_id.clone(),
            model_id: evidence.scenario.model.to_string(),
            is_black_swan,
            upstream_fixture: None,
            pygmy_builder: None,
        }
    }

    /// Determine priority from evidence
    fn determine_priority(&self, evidence: &Evidence, is_black_swan: bool) -> TicketPriority {
        if is_black_swan || evidence.outcome == Outcome::Crashed {
            return TicketPriority::P0;
        }
        // G0-G4 are the only gateway gates; use explicit prefix checks to avoid
        // false positives from future gate IDs that start with 'G' (e.g. GOLDEN-*)
        let gid = &evidence.gate_id;
        if gid.starts_with("G0-")
            || gid.starts_with("G1-")
            || gid.starts_with("G2-")
            || gid.starts_with("G3-")
            || gid.starts_with("G4-")
        {
            return TicketPriority::P0;
        }
        PRIORITY_RULES
            .iter()
            .find(|&&(pattern, _)| gid.contains(pattern))
            .map_or(TicketPriority::P3, |&(_, priority)| priority)
    }

    /// Determine category from gate ID
    fn determine_category(&self, gate_id: &str) -> TicketCategory {
        CATEGORY_RULES
            .iter()
            .find(|&&(keywords, _)| keywords.iter().any(|kw| gate_id.contains(kw)))
            .map_or(TicketCategory::Bug, |&(_, cat)| cat)
    }

    /// Get repository name
    #[must_use]
    pub fn repo(&self) -> &str {
        &self.repo
    }
}

include!("ticket_generation_impl.rs");
