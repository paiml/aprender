// `http-api-v1` algorithm-level PARTIAL discharge for
// FALSIFY-HTTP-001..004.
//
// Contract: `contracts/http-api-v1.yaml`.
//
// Pure-Rust verdicts for the 4 falsification gates:
//   HTTP-001: 200 response is valid JSON with content-type=application/json
//   HTTP-002: 400 error response carries `{error: {message, type}}` envelope
//   HTTP-003: --no-cors removes CORS headers entirely (all-or-nothing)
//   HTTP-004: 404 unknown-endpoint also uses the JSON error envelope
//
// We model the response as a tiny POD struct so the verdicts are pure
// — no live HTTP, no tokio, no serde dep. The contract's full
// behaviour-level test would still be the live integration test the
// YAML enumerates; PARTIAL_ALGORITHM_LEVEL captures the decision rule.

/// Required Content-Type for healthy 200 responses.
pub const AC_HTTP_JSON_CONTENT_TYPE: &str = "application/json";
/// Bad-request status for HTTP-002.
pub const AC_HTTP_BAD_REQUEST: u16 = 400;
/// Not-found status for HTTP-004.
pub const AC_HTTP_NOT_FOUND: u16 = 404;
/// CORS header name probed by HTTP-003.
pub const AC_HTTP_CORS_HEADER: &str = "access-control-allow-origin";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpVerdict {
    Pass,
    Fail,
}

/// Trivial test scaffolding: a header-name list (lowercased) plus a
/// status, plus a body that's been pre-classified as
/// `parse_status` (Valid / Invalid / MissingFields).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyParseStatus {
    /// Body parsed as JSON AND has every required envelope field.
    ValidJson,
    /// Body is valid JSON but missing required envelope fields.
    JsonMissingFields,
    /// Body did not parse as JSON.
    NotJson,
}

/// HTTP-001: 200 response is valid JSON with `Content-Type: application/json`.
#[must_use]
pub fn verdict_from_completions_response(
    status: u16,
    content_type: &str,
    body: BodyParseStatus,
) -> HttpVerdict {
    if status != 200 {
        return HttpVerdict::Fail;
    }
    if !content_type.eq_ignore_ascii_case(AC_HTTP_JSON_CONTENT_TYPE) {
        return HttpVerdict::Fail;
    }
    if body != BodyParseStatus::ValidJson {
        return HttpVerdict::Fail;
    }
    HttpVerdict::Pass
}

/// HTTP-002: 400 error response carries `{error: {message, type}}` envelope.
///
/// `body == ValidJson` here means the body parsed AND the
/// `error.message` + `error.type` fields are both present and string-typed.
#[must_use]
pub fn verdict_from_error_envelope(status: u16, body: BodyParseStatus) -> HttpVerdict {
    if status != AC_HTTP_BAD_REQUEST {
        return HttpVerdict::Fail;
    }
    if body != BodyParseStatus::ValidJson {
        return HttpVerdict::Fail;
    }
    HttpVerdict::Pass
}

/// HTTP-003: `--no-cors` removes CORS headers entirely.
///
/// Pass iff `no_cors_flag == true` AND the response headers list does
/// NOT contain `access-control-allow-origin` (case-insensitive).
#[must_use]
pub fn verdict_from_cors_disabled(no_cors_flag: bool, header_names_lower: &[&str]) -> HttpVerdict {
    if !no_cors_flag {
        return HttpVerdict::Fail;
    }
    if header_names_lower
        .iter()
        .any(|h| h.eq_ignore_ascii_case(AC_HTTP_CORS_HEADER))
    {
        return HttpVerdict::Fail;
    }
    HttpVerdict::Pass
}

/// HTTP-004: 404 unknown endpoint uses the JSON error envelope.
#[must_use]
pub fn verdict_from_404_envelope(status: u16, body: BodyParseStatus) -> HttpVerdict {
    if status != AC_HTTP_NOT_FOUND {
        return HttpVerdict::Fail;
    }
    if body != BodyParseStatus::ValidJson {
        return HttpVerdict::Fail;
    }
    HttpVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // Section 1: Provenance pin.
    // -----------------------------------------------------------------
    #[test]
    fn provenance_json_content_type() {
        assert_eq!(AC_HTTP_JSON_CONTENT_TYPE, "application/json");
    }

    #[test]
    fn provenance_bad_request_400() {
        assert_eq!(AC_HTTP_BAD_REQUEST, 400);
    }

    #[test]
    fn provenance_not_found_404() {
        assert_eq!(AC_HTTP_NOT_FOUND, 404);
    }

    #[test]
    fn provenance_cors_header_name() {
        assert_eq!(AC_HTTP_CORS_HEADER, "access-control-allow-origin");
    }

    // -----------------------------------------------------------------
    // Section 2: HTTP-001 completions response.
    // -----------------------------------------------------------------
    #[test]
    fn fhttp001_pass_canonical() {
        let v = verdict_from_completions_response(200, "application/json", BodyParseStatus::ValidJson);
        assert_eq!(v, HttpVerdict::Pass);
    }

    #[test]
    fn fhttp001_pass_content_type_case_insensitive() {
        let v = verdict_from_completions_response(200, "Application/JSON", BodyParseStatus::ValidJson);
        assert_eq!(v, HttpVerdict::Pass);
    }

    #[test]
    fn fhttp001_fail_non_200_status() {
        let v = verdict_from_completions_response(500, "application/json", BodyParseStatus::ValidJson);
        assert_eq!(v, HttpVerdict::Fail);
    }

    #[test]
    fn fhttp001_fail_text_html_content_type() {
        let v = verdict_from_completions_response(200, "text/html", BodyParseStatus::ValidJson);
        assert_eq!(v, HttpVerdict::Fail);
    }

    #[test]
    fn fhttp001_fail_unparseable_body() {
        let v = verdict_from_completions_response(200, "application/json", BodyParseStatus::NotJson);
        assert_eq!(v, HttpVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 3: HTTP-002 error envelope.
    // -----------------------------------------------------------------
    #[test]
    fn fhttp002_pass_400_with_envelope() {
        let v = verdict_from_error_envelope(400, BodyParseStatus::ValidJson);
        assert_eq!(v, HttpVerdict::Pass);
    }

    #[test]
    fn fhttp002_fail_200_status() {
        let v = verdict_from_error_envelope(200, BodyParseStatus::ValidJson);
        assert_eq!(v, HttpVerdict::Fail);
    }

    #[test]
    fn fhttp002_fail_envelope_missing_fields() {
        let v = verdict_from_error_envelope(400, BodyParseStatus::JsonMissingFields);
        assert_eq!(v, HttpVerdict::Fail);
    }

    #[test]
    fn fhttp002_fail_html_body() {
        let v = verdict_from_error_envelope(400, BodyParseStatus::NotJson);
        assert_eq!(v, HttpVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 4: HTTP-003 CORS disabled.
    // -----------------------------------------------------------------
    #[test]
    fn fhttp003_pass_no_cors_no_header() {
        let v = verdict_from_cors_disabled(true, &["content-type", "x-server"]);
        assert_eq!(v, HttpVerdict::Pass);
    }

    #[test]
    fn fhttp003_fail_no_cors_but_header_present() {
        let v = verdict_from_cors_disabled(
            true,
            &["content-type", "access-control-allow-origin"],
        );
        assert_eq!(v, HttpVerdict::Fail);
    }

    #[test]
    fn fhttp003_fail_flag_off() {
        let v = verdict_from_cors_disabled(false, &["content-type"]);
        assert_eq!(v, HttpVerdict::Fail);
    }

    #[test]
    fn fhttp003_pass_no_cors_empty_headers() {
        let v = verdict_from_cors_disabled(true, &[]);
        assert_eq!(v, HttpVerdict::Pass);
    }

    #[test]
    fn fhttp003_fail_case_variant_header_leaks() {
        // Server case-mismatch must still trip the gate.
        let v = verdict_from_cors_disabled(true, &["Access-Control-Allow-Origin"]);
        assert_eq!(v, HttpVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 5: HTTP-004 404 envelope.
    // -----------------------------------------------------------------
    #[test]
    fn fhttp004_pass_404_with_envelope() {
        let v = verdict_from_404_envelope(404, BodyParseStatus::ValidJson);
        assert_eq!(v, HttpVerdict::Pass);
    }

    #[test]
    fn fhttp004_fail_200_status() {
        let v = verdict_from_404_envelope(200, BodyParseStatus::ValidJson);
        assert_eq!(v, HttpVerdict::Fail);
    }

    #[test]
    fn fhttp004_fail_html_404_page() {
        let v = verdict_from_404_envelope(404, BodyParseStatus::NotJson);
        assert_eq!(v, HttpVerdict::Fail);
    }

    #[test]
    fn fhttp004_fail_envelope_missing_message() {
        let v = verdict_from_404_envelope(404, BodyParseStatus::JsonMissingFields);
        assert_eq!(v, HttpVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 6: Mutation surveys.
    // -----------------------------------------------------------------
    #[test]
    fn mutation_survey_001_status_codes_around_200() {
        for status in [199_u16, 200, 201, 204, 301, 400, 500] {
            let v = verdict_from_completions_response(
                status,
                "application/json",
                BodyParseStatus::ValidJson,
            );
            let expected = if status == 200 {
                HttpVerdict::Pass
            } else {
                HttpVerdict::Fail
            };
            assert_eq!(v, expected, "status={status}");
        }
    }

    #[test]
    fn mutation_survey_002_status_codes_around_400() {
        for status in [200_u16, 399, 400, 401, 500] {
            let v = verdict_from_error_envelope(status, BodyParseStatus::ValidJson);
            let expected = if status == AC_HTTP_BAD_REQUEST {
                HttpVerdict::Pass
            } else {
                HttpVerdict::Fail
            };
            assert_eq!(v, expected, "status={status}");
        }
    }

    #[test]
    fn mutation_survey_004_status_codes_around_404() {
        for status in [200_u16, 403, 404, 405, 500] {
            let v = verdict_from_404_envelope(status, BodyParseStatus::ValidJson);
            let expected = if status == AC_HTTP_NOT_FOUND {
                HttpVerdict::Pass
            } else {
                HttpVerdict::Fail
            };
            assert_eq!(v, expected, "status={status}");
        }
    }

    // -----------------------------------------------------------------
    // Section 7: Realistic.
    // -----------------------------------------------------------------
    #[test]
    fn realistic_healthy_server_passes_all_4() {
        // Canonical apr serve --no-cors run.
        let v1 = verdict_from_completions_response(200, "application/json", BodyParseStatus::ValidJson);
        let v2 = verdict_from_error_envelope(400, BodyParseStatus::ValidJson);
        let v3 = verdict_from_cors_disabled(true, &["content-type"]);
        let v4 = verdict_from_404_envelope(404, BodyParseStatus::ValidJson);
        assert_eq!(v1, HttpVerdict::Pass);
        assert_eq!(v2, HttpVerdict::Pass);
        assert_eq!(v3, HttpVerdict::Pass);
        assert_eq!(v4, HttpVerdict::Pass);
    }

    #[test]
    fn realistic_pre_fix_all_4_failures() {
        // Pre-fix regression: text/plain bodies, no envelopes, CORS leak.
        let v1 = verdict_from_completions_response(500, "text/plain", BodyParseStatus::NotJson);
        let v2 = verdict_from_error_envelope(400, BodyParseStatus::NotJson);
        let v3 = verdict_from_cors_disabled(true, &["access-control-allow-origin"]);
        let v4 = verdict_from_404_envelope(404, BodyParseStatus::NotJson);
        assert_eq!(v1, HttpVerdict::Fail);
        assert_eq!(v2, HttpVerdict::Fail);
        assert_eq!(v3, HttpVerdict::Fail);
        assert_eq!(v4, HttpVerdict::Fail);
    }
}
