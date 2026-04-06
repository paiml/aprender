//! Tests for Banco application state.

use super::state::BancoStateInner;
use crate::serve::backends::PrivacyTier;

// ============================================================================
// BANCO_STA_001: Default state creation
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_BANCO_STA_001_with_defaults() {
    let state = BancoStateInner::with_defaults();
    assert_eq!(state.privacy_tier, PrivacyTier::Standard);
}

// ============================================================================
// BANCO_STA_002: Privacy tier state creation
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_BANCO_STA_002_with_sovereign_privacy() {
    let state = BancoStateInner::with_privacy(PrivacyTier::Sovereign);
    assert_eq!(state.privacy_tier, PrivacyTier::Sovereign);
}

#[test]
#[allow(non_snake_case)]
fn test_BANCO_STA_002_with_private_privacy() {
    let state = BancoStateInner::with_privacy(PrivacyTier::Private);
    assert_eq!(state.privacy_tier, PrivacyTier::Private);
}

// ============================================================================
// BANCO_STA_003: Health status
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_BANCO_STA_003_health_status() {
    let state = BancoStateInner::with_defaults();
    let health = state.health_status();
    assert_eq!(health.status, "ok");
    assert_eq!(health.circuit_breaker_state, "closed");
    // Uptime should be very small (just created)
    assert!(health.uptime_secs < 2);
}

// ============================================================================
// BANCO_STA_004: List models
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_BANCO_STA_004_list_models_standard() {
    let state = BancoStateInner::with_defaults();
    let models = state.list_models();
    assert_eq!(models.object, "list");
    assert!(!models.data.is_empty());
    // All entries should have the expected shape
    for model in &models.data {
        assert_eq!(model.object, "model");
        assert_eq!(model.owned_by, "batuta");
    }
}

#[test]
#[allow(non_snake_case)]
fn test_BANCO_STA_004_list_models_sovereign() {
    let state = BancoStateInner::with_privacy(PrivacyTier::Sovereign);
    let models = state.list_models();
    // Sovereign should only recommend local backends
    for model in &models.data {
        assert!(model.local, "sovereign model {:?} should be local", model.id);
    }
}

// ============================================================================
// BANCO_STA_005: System info
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_BANCO_STA_005_system_info() {
    let state = BancoStateInner::with_defaults();
    let info = state.system_info();
    assert_eq!(info.privacy_tier, "Standard");
    assert!(!info.backends.is_empty());
    assert!(!info.version.is_empty());
}

#[test]
#[allow(non_snake_case)]
fn test_BANCO_STA_005_system_info_sovereign() {
    let state = BancoStateInner::with_privacy(PrivacyTier::Sovereign);
    let info = state.system_info();
    assert_eq!(info.privacy_tier, "Sovereign");
}

// ============================================================================
// BANCO_STA_006: Circuit breaker reflected in health
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_BANCO_STA_006_circuit_breaker_in_health() {
    let state = BancoStateInner::with_defaults();
    // Spend entire budget to open circuit breaker
    state.circuit_breaker.record(10.0);
    let health = state.health_status();
    assert_eq!(health.circuit_breaker_state, "open");
}
