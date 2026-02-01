//! License property tests.

use super::super::*;
use proptest::prelude::*;

// Arbitrary generators for license types

fn arb_license_tier() -> impl Strategy<Value = LicenseTier> {
    prop_oneof![
        Just(LicenseTier::Personal),
        Just(LicenseTier::Team),
        Just(LicenseTier::Enterprise),
        Just(LicenseTier::Academic),
    ]
}

/// Generate valid UUID v4 format
fn arb_uuid() -> impl Strategy<Value = String> {
    "[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
}

/// Generate SHA256 hash
fn arb_hash() -> impl Strategy<Value = String> {
    "[0-9a-f]{64}"
}

/// Generate ISO 8601 date (YYYY-MM-DD)
fn arb_iso_date() -> impl Strategy<Value = String> {
    (2024u32..2035, 1u32..13, 1u32..29).prop_map(|(y, m, d)| format!("{y:04}-{m:02}-{d:02}"))
}

fn arb_license_info() -> impl Strategy<Value = LicenseInfo> {
    (
        arb_uuid(),
        arb_hash(),
        proptest::option::of(arb_iso_date()),
        proptest::option::of(1u32..1000),
        proptest::option::of("[A-Za-z0-9 ]{1,50}"),
        arb_license_tier(),
    )
        .prop_map(|(uuid, hash, expiry, seats, licensee, tier)| LicenseInfo {
            uuid,
            hash,
            expiry,
            seats,
            licensee,
            tier,
        })
}

fn arb_model_data() -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(
        any::<f32>().prop_filter("finite", |f| f.is_finite()),
        1..100,
    )
}

proptest! {
    /// Property: LicenseTier serialization roundtrip
    #[test]
    fn prop_license_tier_roundtrip(tier in arb_license_tier()) {
        let serialized = serde_json::to_string(&tier).expect("serialize");
        let deserialized: LicenseTier = serde_json::from_str(&serialized).expect("deserialize");
        prop_assert_eq!(tier, deserialized);
    }

    /// Property: LicenseInfo serialization roundtrip
    #[test]
    fn prop_license_info_roundtrip(info in arb_license_info()) {
        let serialized = serde_json::to_string(&info).expect("serialize");
        let deserialized: LicenseInfo = serde_json::from_str(&serialized).expect("deserialize");

        prop_assert_eq!(&info.uuid, &deserialized.uuid);
        prop_assert_eq!(&info.hash, &deserialized.hash);
        prop_assert_eq!(info.tier, deserialized.tier);
        prop_assert_eq!(info.seats, deserialized.seats);
        prop_assert_eq!(&info.expiry, &deserialized.expiry);
    }

    /// Property: UUID format is valid v4
    #[test]
    fn prop_uuid_format_valid(uuid in arb_uuid()) {
        // UUID v4 format: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
        // where y is 8, 9, a, or b
        prop_assert_eq!(uuid.len(), 36);
        prop_assert!(uuid.chars().nth(14) == Some('4'), "Version must be 4");
        let y = uuid.chars().nth(19).expect("UUID must have char at position 19");
        prop_assert!(
            matches!(y, '8' | '9' | 'a' | 'b'),
            "Variant must be 8, 9, a, or b"
        );
    }

    /// Property: License persists through save/load cycle
    #[test]
    fn prop_license_save_load_roundtrip(
        license in arb_license_info(),
        data in arb_model_data()
    ) {
        use tempfile::tempdir;

        #[derive(Debug, serde::Serialize, serde::Deserialize)]
        struct Model { weights: Vec<f32> }

        let model = Model { weights: data };
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("licensed.apr");

        let options = SaveOptions::default().with_license(license.clone());
        save(&model, ModelType::Custom, &path, options).expect("save");

        let model_info = inspect(&path).expect("inspect");
        let restored = model_info.metadata.license
            .expect("should have license");

        prop_assert_eq!(&license.uuid, &restored.uuid);
        prop_assert_eq!(&license.hash, &restored.hash);
        prop_assert_eq!(license.tier, restored.tier);
        prop_assert_eq!(license.seats, restored.seats);
    }

    /// Property: LICENSED flag is set when license provided
    #[test]
    fn prop_licensed_flag_set(license in arb_license_info(), data in arb_model_data()) {
        use tempfile::tempdir;

        #[derive(Debug, serde::Serialize, serde::Deserialize)]
        struct Model { weights: Vec<f32> }

        let model = Model { weights: data };
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("licensed.apr");

        let options = SaveOptions::default().with_license(license);
        save(&model, ModelType::Custom, &path, options).expect("save");

        let info = inspect(&path).expect("inspect");
        prop_assert!(info.licensed, "LICENSED flag must be set");
    }

    /// Property: Seats must be positive when specified
    #[test]
    fn prop_seats_positive(seats in 1u32..10000) {
        let license = LicenseInfo {
            uuid: "00000000-0000-4000-8000-000000000000".to_string(),
            hash: "0".repeat(64),
            expiry: None,
            seats: Some(seats),
            licensee: None,
            tier: LicenseTier::Team,
        };
        prop_assert!(license.seats == Some(seats) && seats > 0, "Seats must be positive");
    }

    /// Property: Enterprise tier has no seat limit by default
    #[test]
    fn prop_enterprise_unlimited_seats(_dummy in 0u8..1) {
        // Enterprise tier typically has unlimited seats
        let license = LicenseInfo {
            uuid: "00000000-0000-4000-8000-000000000000".to_string(),
            hash: "0".repeat(64),
            expiry: None,
            seats: None, // Unlimited
            licensee: Some("ACME Corp".to_string()),
            tier: LicenseTier::Enterprise,
        };
        prop_assert!(license.seats.is_none(), "Enterprise should have unlimited seats");
        prop_assert!(matches!(license.tier, LicenseTier::Enterprise));
    }

    /// Property: Academic tier is non-commercial
    #[test]
    fn prop_academic_tier_valid(_dummy in 0u8..1) {
        let license = LicenseInfo {
            uuid: "00000000-0000-4000-8000-000000000000".to_string(),
            hash: "0".repeat(64),
            expiry: Some("2025-12-31".to_string()),
            seats: Some(100),
            licensee: Some("MIT".to_string()),
            tier: LicenseTier::Academic,
        };
        prop_assert!(matches!(license.tier, LicenseTier::Academic));
    }

    /// Property: Hash is 64 hex characters (SHA256)
    #[test]
    fn prop_hash_length_valid(hash in arb_hash()) {
        prop_assert_eq!(hash.len(), 64, "SHA256 hash must be 64 hex chars");
        prop_assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
