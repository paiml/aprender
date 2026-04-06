// Contract validation for InferenceTracer (P15-06)
//
// Validates trace events against tracing-observability-v1.yaml invariants.
// Contract 5: Monotonic event IDs
// Contract 6: No orphan spans, no dangling entries
//
// Reference: realizr#202, candle-vs-apr P15-06

impl InferenceTracer {
    /// Validate tracing-observability-v1 contract invariants.
    ///
    /// Checks:
    /// 1. No orphan spans — every TaskStateExited.previous_event_id references
    ///    a valid TaskStateEntered event
    /// 2. Monotonic event IDs — IDs are strictly increasing
    /// 3. No dangling entries — every TaskStateEntered has a corresponding
    ///    TaskStateExited (no unclosed spans)
    ///
    /// Contract: tracing-observability-v1.yaml (provable-contracts)
    /// Reference: realizr#202, candle-vs-apr P15-06
    pub fn validate_observability_contracts(
        &self,
    ) -> Vec<(crate::brick::profiler::ContractSeverity, String)> {
        use crate::brick::profiler::ContractSeverity;
        let mut violations = Vec::new();

        if self.events.is_empty() {
            return violations;
        }

        // CONTRACT 5: Monotonic event IDs
        // "t1 < t2 => counter(t1) <= counter(t2)"
        let mut prev_id = 0u64;
        for event in &self.events {
            if event.id <= prev_id && prev_id > 0 {
                violations.push((
                    ContractSeverity::Error,
                    format!(
                        "tracing-observability-v1 MONOTONIC_COUNTER: event ID {} <= previous ID {}. \
                         IDs must be strictly increasing.",
                        event.id, prev_id
                    ),
                ));
                break; // One violation is enough to flag
            }
            prev_id = event.id;
        }

        // CONTRACT 6a: No orphan spans
        // "span.parent_id = None OR span.parent_id in active_spans"
        let entered_ids: std::collections::HashSet<u64> = self
            .events
            .iter()
            .filter(|e| e.event_type == AwsEventType::TaskStateEntered)
            .map(|e| e.id)
            .collect();

        for event in &self.events {
            if event.event_type == AwsEventType::TaskStateExited {
                if let Some(parent_id) = event.previous_event_id {
                    if !entered_ids.contains(&parent_id) {
                        violations.push((
                            ContractSeverity::Error,
                            format!(
                                "tracing-observability-v1 ORPHAN_SPAN: TaskStateExited (id={}) \
                                 references previous_event_id={} which has no matching \
                                 TaskStateEntered. Orphan span detected.",
                                event.id, parent_id
                            ),
                        ));
                    }
                }
            }
        }

        // CONTRACT 6b: No dangling entries (unclosed spans)
        let exited_parents: std::collections::HashSet<u64> = self
            .events
            .iter()
            .filter(|e| e.event_type == AwsEventType::TaskStateExited)
            .filter_map(|e| e.previous_event_id)
            .collect();

        for event in &self.events {
            if event.event_type == AwsEventType::TaskStateEntered
                && !exited_parents.contains(&event.id)
            {
                violations.push((
                    ContractSeverity::Warning,
                    format!(
                        "tracing-observability-v1 DANGLING_ENTRY: TaskStateEntered (id={}, step={}) \
                         has no corresponding TaskStateExited. Unclosed span.",
                        event.id, event.step.name()
                    ),
                ));
            }
        }

        violations
    }
}
