// `apr-serve-v1` algorithm-level PARTIAL discharge for the 4
// inference-server falsifiers (health-readiness, graceful shutdown,
// 404 routing, concurrent inference isolation).
//
// Contract: `contracts/apr-serve-v1.yaml`.

/// Default graceful-shutdown timeout (30 seconds) per server_lifecycle
/// invariant.
pub const AC_APRSERVE_SHUTDOWN_TIMEOUT_S: u32 = 30;

/// Server lifecycle states per the contract's state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerState {
    Init,
    Binding,
    Loading,
    Ready,
    Draining,
    Stopped,
}

// =============================================================================
// FALSIFY-SRV-001 — health returns 200 only when ready
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthReadinessVerdict {
    /// /health returns 200 only in Ready state; 503 in all others.
    Pass,
    /// /health returned 200 in non-Ready state — health-lying regression.
    Fail,
}

#[must_use]
pub fn verdict_from_health_readiness(state: ServerState, http_status: u16) -> HealthReadinessVerdict {
    match (state, http_status) {
        (ServerState::Ready, 200) => HealthReadinessVerdict::Pass,
        // 200 in any non-Ready state is the regression class.
        (_, 200) => HealthReadinessVerdict::Fail,
        // Any non-200 in Ready is also wrong (probably 503 by the test).
        (ServerState::Ready, _) => HealthReadinessVerdict::Fail,
        // Non-Ready state returning 503 (or any non-200) is correct.
        _ => HealthReadinessVerdict::Pass,
    }
}

// =============================================================================
// FALSIFY-SRV-002 — graceful shutdown completes in-flight
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GracefulShutdownVerdict {
    /// In-flight count == 0 before exit AND exit_code == 0 AND shutdown
    /// duration ≤ timeout.
    Pass,
    /// In-flight requests dropped (connection reset), or timeout exceeded.
    Fail,
}

#[must_use]
pub fn verdict_from_graceful_shutdown(
    in_flight_at_exit: u32,
    exit_code: i32,
    shutdown_duration_s: u32,
) -> GracefulShutdownVerdict {
    if in_flight_at_exit > 0 {
        return GracefulShutdownVerdict::Fail;
    }
    if exit_code != 0 {
        return GracefulShutdownVerdict::Fail;
    }
    if shutdown_duration_s > AC_APRSERVE_SHUTDOWN_TIMEOUT_S {
        return GracefulShutdownVerdict::Fail;
    }
    GracefulShutdownVerdict::Pass
}

// =============================================================================
// FALSIFY-SRV-003 — unknown path returns 404 (not 500 / HTML)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnknownPathVerdict {
    /// Status 404 AND Content-Type: application/json.
    Pass,
    /// 500, 200, or HTML response — fell through to default Actix handler.
    Fail,
}

#[must_use]
pub fn verdict_from_unknown_path(http_status: u16, content_type: &str) -> UnknownPathVerdict {
    if http_status != 404 {
        return UnknownPathVerdict::Fail;
    }
    let lower = content_type.to_lowercase();
    if !lower.contains("application/json") {
        return UnknownPathVerdict::Fail;
    }
    UnknownPathVerdict::Pass
}

// =============================================================================
// FALSIFY-SRV-004 — concurrent inference isolation
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConcurrentIsolationVerdict {
    /// All N concurrent identical requests at temperature=0 return
    /// identical output (no KV-cache contamination).
    Pass,
    /// At least 2 distinct outputs across requests.
    Fail,
}

#[must_use]
pub fn verdict_from_concurrent_isolation(responses: &[&str]) -> ConcurrentIsolationVerdict {
    if responses.is_empty() {
        return ConcurrentIsolationVerdict::Fail;
    }
    let first = responses[0];
    for r in responses.iter().skip(1) {
        if *r != first {
            return ConcurrentIsolationVerdict::Fail;
        }
    }
    ConcurrentIsolationVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_shutdown_timeout_30s() {
        assert_eq!(AC_APRSERVE_SHUTDOWN_TIMEOUT_S, 30);
    }

    #[test]
    fn provenance_state_machine_six_states() {
        // Smoke: count of valid lifecycle states matches contract enumeration.
        let all = [
            ServerState::Init,
            ServerState::Binding,
            ServerState::Loading,
            ServerState::Ready,
            ServerState::Draining,
            ServerState::Stopped,
        ];
        assert_eq!(all.len(), 6);
    }

    // -------------------------------------------------------------------------
    // Section 2: SRV-001 health readiness.
    // -------------------------------------------------------------------------
    #[test]
    fn fs001_pass_ready_returns_200() {
        assert_eq!(
            verdict_from_health_readiness(ServerState::Ready, 200),
            HealthReadinessVerdict::Pass
        );
    }

    #[test]
    fn fs001_pass_loading_returns_503() {
        assert_eq!(
            verdict_from_health_readiness(ServerState::Loading, 503),
            HealthReadinessVerdict::Pass
        );
    }

    #[test]
    fn fs001_pass_init_returns_503() {
        assert_eq!(
            verdict_from_health_readiness(ServerState::Init, 503),
            HealthReadinessVerdict::Pass
        );
    }

    #[test]
    fn fs001_pass_draining_returns_503() {
        assert_eq!(
            verdict_from_health_readiness(ServerState::Draining, 503),
            HealthReadinessVerdict::Pass
        );
    }

    #[test]
    fn fs001_fail_loading_returns_200() {
        // The exact regression: health-lying during model load.
        assert_eq!(
            verdict_from_health_readiness(ServerState::Loading, 200),
            HealthReadinessVerdict::Fail
        );
    }

    #[test]
    fn fs001_fail_ready_returns_503() {
        // Ready state must return 200, not 503.
        assert_eq!(
            verdict_from_health_readiness(ServerState::Ready, 503),
            HealthReadinessVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: SRV-002 graceful shutdown.
    // -------------------------------------------------------------------------
    #[test]
    fn fs002_pass_clean_shutdown() {
        assert_eq!(
            verdict_from_graceful_shutdown(0, 0, 5),
            GracefulShutdownVerdict::Pass
        );
    }

    #[test]
    fn fs002_pass_at_timeout() {
        assert_eq!(
            verdict_from_graceful_shutdown(0, 0, 30),
            GracefulShutdownVerdict::Pass
        );
    }

    #[test]
    fn fs002_fail_in_flight_dropped() {
        assert_eq!(
            verdict_from_graceful_shutdown(3, 0, 5),
            GracefulShutdownVerdict::Fail
        );
    }

    #[test]
    fn fs002_fail_nonzero_exit() {
        assert_eq!(
            verdict_from_graceful_shutdown(0, 1, 5),
            GracefulShutdownVerdict::Fail
        );
    }

    #[test]
    fn fs002_fail_timeout_exceeded() {
        assert_eq!(
            verdict_from_graceful_shutdown(0, 0, 31),
            GracefulShutdownVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: SRV-003 unknown path 404 routing.
    // -------------------------------------------------------------------------
    #[test]
    fn fs003_pass_404_json() {
        assert_eq!(
            verdict_from_unknown_path(404, "application/json"),
            UnknownPathVerdict::Pass
        );
    }

    #[test]
    fn fs003_pass_404_json_charset() {
        assert_eq!(
            verdict_from_unknown_path(404, "application/json; charset=utf-8"),
            UnknownPathVerdict::Pass
        );
    }

    #[test]
    fn fs003_fail_500_internal_error() {
        // The regression: routing failure surfaced as 500.
        assert_eq!(
            verdict_from_unknown_path(500, "application/json"),
            UnknownPathVerdict::Fail
        );
    }

    #[test]
    fn fs003_fail_404_html() {
        // Actix default — HTML 404 is the regression class.
        assert_eq!(
            verdict_from_unknown_path(404, "text/html"),
            UnknownPathVerdict::Fail
        );
    }

    #[test]
    fn fs003_fail_200_returned() {
        assert_eq!(
            verdict_from_unknown_path(200, "application/json"),
            UnknownPathVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: SRV-004 concurrent inference isolation.
    // -------------------------------------------------------------------------
    #[test]
    fn fs004_pass_10_identical() {
        let r = ["The capital of France is Paris."; 10];
        assert_eq!(
            verdict_from_concurrent_isolation(&r),
            ConcurrentIsolationVerdict::Pass
        );
    }

    #[test]
    fn fs004_pass_single_request() {
        let r = ["A response"];
        assert_eq!(
            verdict_from_concurrent_isolation(&r),
            ConcurrentIsolationVerdict::Pass
        );
    }

    #[test]
    fn fs004_fail_kv_cache_contamination() {
        // The regression: req 7 leaked content from req 6.
        let r = [
            "Paris.", "Paris.", "Paris.", "Paris.", "Paris.",
            "Paris.", "Paris.", "Lyon.", "Paris.", "Paris.",
        ];
        assert_eq!(
            verdict_from_concurrent_isolation(&r),
            ConcurrentIsolationVerdict::Fail
        );
    }

    #[test]
    fn fs004_fail_empty() {
        let r: [&str; 0] = [];
        assert_eq!(
            verdict_from_concurrent_isolation(&r),
            ConcurrentIsolationVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: Realistic — full healthy server passes all 4.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_healthy_server_passes_all_4() {
        // Ready + 200, in-flight==0 + exit 0, 404 JSON, 5 identical responses.
        assert_eq!(
            verdict_from_health_readiness(ServerState::Ready, 200),
            HealthReadinessVerdict::Pass
        );
        assert_eq!(
            verdict_from_graceful_shutdown(0, 0, 8),
            GracefulShutdownVerdict::Pass
        );
        assert_eq!(
            verdict_from_unknown_path(404, "application/json"),
            UnknownPathVerdict::Pass
        );
        let r = ["4", "4", "4", "4", "4"];
        assert_eq!(
            verdict_from_concurrent_isolation(&r),
            ConcurrentIsolationVerdict::Pass
        );
    }

    #[test]
    fn realistic_pre_fix_all_4_failures() {
        // 001: health lies during loading.
        assert_eq!(
            verdict_from_health_readiness(ServerState::Loading, 200),
            HealthReadinessVerdict::Fail
        );
        // 002: SIGTERM dropped 5 in-flight requests.
        assert_eq!(
            verdict_from_graceful_shutdown(5, 0, 1),
            GracefulShutdownVerdict::Fail
        );
        // 003: HTML 404 from Actix default handler.
        assert_eq!(
            verdict_from_unknown_path(404, "text/html"),
            UnknownPathVerdict::Fail
        );
        // 004: KV-cache contaminated request 8.
        let r = ["A", "A", "B"];
        assert_eq!(
            verdict_from_concurrent_isolation(&r),
            ConcurrentIsolationVerdict::Fail
        );
    }
}
