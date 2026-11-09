//! The part of `crux_serve_routes.py` the judge reads: which comparator
//! route answers a served route's question.

use crate::pyval::Val;

/// `GENERATION`: route -> kind (the judge never reads `modes`).
const GENERATION: &[(&str, &str)] = &[
    ("POST /v1/chat/completions", "chat_messages"),
    ("POST /v1/chat/completions/stream", "chat_messages"),
    ("POST /v1/completions", "text_prompt"),
    ("POST /generate", "raw_generate"),
    ("POST /stream/generate", "raw_sse"),
    ("POST /realize/generate", "raw_sse"),
    ("POST /batch/generate", "raw_batch"),
    ("POST /realize/batch", "raw_batch"),
    ("POST /v1/batch/completions", "raw_batch"),
    ("POST /api/chat", "ollama_chat"),
    ("POST /api/generate", "ollama_generate"),
];

/// `ORACLE_ROUTE_BY_KIND`.
const ORACLE_ROUTE_BY_KIND: &[(&str, &str)] = &[
    ("chat_messages", "POST /v1/chat/completions"),
    ("ollama_chat", "POST /v1/chat/completions"),
    ("ollama_generate", "POST /v1/chat/completions"),
    ("text_prompt", "POST /v1/completions"),
    ("raw_generate", "POST /v1/completions"),
    ("raw_sse", "POST /v1/completions"),
    ("raw_batch", "POST /v1/completions"),
];

/// `oracle_route(route)`: the comparator route that answers `route`'s
/// question, or None when none is mapped (a non-str route is in no table).
pub fn oracle_route(route: &Val) -> Option<&'static str> {
    let route = route.as_str()?;
    let kind = GENERATION.iter().find(|(r, _)| *r == route)?.1;
    ORACLE_ROUTE_BY_KIND
        .iter()
        .find(|(k, _)| *k == kind)
        .map(|(_, r)| *r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_generation_kind_has_an_oracle_route() {
        for (route, _) in GENERATION {
            assert!(oracle_route(&Val::str(*route)).is_some(), "{route}");
        }
        assert_eq!(
            oracle_route(&Val::str("POST /api/generate")),
            Some("POST /v1/chat/completions")
        );
        assert_eq!(oracle_route(&Val::str("GET /health")), None);
        assert_eq!(oracle_route(&Val::int(1)), None);
    }
}
