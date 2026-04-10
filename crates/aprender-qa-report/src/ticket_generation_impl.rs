/// Generate structured tickets from evidence using the defect-fixture map (§3.6)
///
/// Groups failures by root cause (`ConversionFailureType`), deduplicates,
/// and renders each group as a single ticket with the corresponding fixture template.
#[must_use]
pub fn generate_structured_tickets<S: ::std::hash::BuildHasher>(
    evidence: &[Evidence],
    defect_map: &HashMap<String, crate::defect_map::DefectFixtureEntry, S>,
) -> Vec<UpstreamTicket> {
    // Step 1: Filter to failures only
    let failures: Vec<&Evidence> = evidence.iter().filter(|e| e.outcome.is_fail()).collect();

    if failures.is_empty() {
        return Vec::new();
    }

    // Step 2: Classify each failure and group by root cause key
    let mut groups: HashMap<String, Vec<&Evidence>> = HashMap::new();
    for ev in &failures {
        let stderr = ev.stderr.as_deref().unwrap_or("");
        let exit_code = ev.exit_code.unwrap_or(1);
        let ft = classify_failure(stderr, exit_code);
        let key = ft.key().to_string();
        groups.entry(key).or_default().push(ev);
    }

    // Step 3: One ticket per root cause
    let mut tickets = Vec::new();
    for (key, group) in &groups {
        let first = group[0];
        let is_black_swan = first.outcome == Outcome::Crashed;

        let gid = &first.gate_id;
        let priority = if is_black_swan {
            TicketPriority::P0
        } else if gid.contains("-P0-")
            || gid.starts_with("G0-")
            || gid.starts_with("G1-")
            || gid.starts_with("G2-")
            || gid.starts_with("G3-")
            || gid.starts_with("G4-")
        {
            TicketPriority::P0
        } else {
            TicketPriority::P1
        };

        let (upstream_fixture, pygmy_builder, body) = if let Some(entry) = defect_map.get(key) {
            let mut fields = HashMap::new();
            fields.insert("model_id".to_string(), first.scenario.model.to_string());
            fields.insert(
                "exit_code".to_string(),
                format!("{}", first.exit_code.unwrap_or(1)),
            );
            fields.insert(
                "stderr".to_string(),
                first.stderr.clone().unwrap_or_default(),
            );
            fields.insert("occurrences".to_string(), group.len().to_string());

            let rendered =
                crate::defect_map::render_ticket_template(&entry.ticket_template, &fields);
            (
                Some(entry.upstream_fixture.clone()),
                Some(entry.pygmy_builder.clone()),
                rendered,
            )
        } else {
            let body = format!(
                "## Conversion Failure\n\n- **Type**: `{key}`\n- **Model**: `{}`\n- **Occurrences**: {}\n\n```\n{}\n```",
                first.scenario.model,
                group.len(),
                first.stderr.as_deref().unwrap_or("N/A"),
            );
            (None, None, body)
        };

        let title = format!(
            "[QA] {}: {} ({} occurrence{})",
            first.gate_id,
            key,
            group.len(),
            if group.len() == 1 { "" } else { "s" },
        );

        let labels = vec![
            format!("priority:{priority}"),
            "qa-automated".to_string(),
            format!("failure-type:{key}"),
        ];

        tickets.push(UpstreamTicket {
            title,
            body,
            priority,
            category: TicketCategory::Bug,
            labels,
            gate_id: first.gate_id.clone(),
            model_id: first.scenario.model.to_string(),
            is_black_swan,
            upstream_fixture,
            pygmy_builder,
        });
    }

    tickets
}


#[cfg(test)]
#[path = "ticket_tests.rs"]
mod ticket_tests;
