//! Commercial licensing support for .ald format (§9)
//!
//! Provides license blocks for commercial dataset distribution with:
//! - Expiration enforcement
//! - Seat limits
//! - Query limits
//! - Revocation support

use crate::error::{Error, Result};

/// License block size (fixed portion, 71 bytes)
pub const LICENSE_BLOCK_FIXED_SIZE: usize = 71;

/// License flags (§9.2)
pub mod flags {
    /// Limit concurrent installations
    pub const SEATS_ENFORCED: u8 = 0b0000_0001;
    /// Dataset stops loading after expires_at
    pub const EXPIRATION_ENFORCED: u8 = 0b0000_0010;
    /// Count-based usage cap
    pub const QUERY_LIMITED: u8 = 0b0000_0100;
    /// Contains buyer-specific fingerprint (watermarked)
    pub const WATERMARKED: u8 = 0b0000_1000;
    /// Can be remotely revoked (requires network)
    pub const REVOCABLE: u8 = 0b0001_0000;
    /// License can be resold
    pub const TRANSFERABLE: u8 = 0b0010_0000;
}

/// Commercial license block
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LicenseBlock {
    /// Unique license identifier (UUID as 16 bytes)
    pub license_id: [u8; 16],
    /// SHA-256 hash of licensee identifier
    pub licensee_hash: [u8; 32],
    /// Issue timestamp (Unix epoch seconds)
    pub issued_at: u64,
    /// Expiration timestamp (Unix epoch seconds, 0 = never)
    pub expires_at: u64,
    /// License flags
    pub flags: u8,
    /// Maximum concurrent seats (0 = unlimited)
    pub seat_limit: u16,
    /// Maximum queries (0 = unlimited)
    pub query_limit: u32,
    /// Custom terms (optional, serialized)
    pub custom_terms: Vec<u8>,
}

impl LicenseBlock {
    /// Create a new license block
    #[must_use]
    pub fn new(license_id: [u8; 16], licensee_hash: [u8; 32]) -> Self {
        Self {
            license_id,
            licensee_hash,
            issued_at: current_unix_time(),
            expires_at: 0,
            flags: 0,
            seat_limit: 0,
            query_limit: 0,
            custom_terms: Vec::new(),
        }
    }

    /// Set expiration timestamp
    #[must_use]
    pub fn with_expiration(mut self, expires_at: u64) -> Self {
        self.expires_at = expires_at;
        self.flags |= flags::EXPIRATION_ENFORCED;
        self
    }

    /// Set seat limit
    #[must_use]
    pub fn with_seat_limit(mut self, limit: u16) -> Self {
        self.seat_limit = limit;
        self.flags |= flags::SEATS_ENFORCED;
        self
    }

    /// Set query limit
    #[must_use]
    pub fn with_query_limit(mut self, limit: u32) -> Self {
        self.query_limit = limit;
        self.flags |= flags::QUERY_LIMITED;
        self
    }

    /// Mark as watermarked
    #[must_use]
    pub fn with_watermark(mut self) -> Self {
        self.flags |= flags::WATERMARKED;
        self
    }

    /// Mark as revocable
    #[must_use]
    pub fn with_revocable(mut self) -> Self {
        self.flags |= flags::REVOCABLE;
        self
    }

    /// Mark as transferable
    #[must_use]
    pub fn with_transferable(mut self) -> Self {
        self.flags |= flags::TRANSFERABLE;
        self
    }

    /// Set custom terms
    #[must_use]
    pub fn with_custom_terms(mut self, terms: Vec<u8>) -> Self {
        self.custom_terms = terms;
        self
    }

    /// Total serialized size
    #[must_use]
    pub fn size(&self) -> usize {
        LICENSE_BLOCK_FIXED_SIZE + self.custom_terms.len()
    }

    /// Serialize to bytes
    ///
    /// Format:
    /// - license_id (16 bytes)
    /// - licensee_hash (32 bytes)
    /// - issued_at (8 bytes, LE)
    /// - expires_at (8 bytes, LE)
    /// - flags (1 byte)
    /// - seat_limit (2 bytes, LE)
    /// - query_limit (4 bytes, LE)
    /// - custom_terms_len (4 bytes, LE) -- added for deserialization
    /// - custom_terms (variable)
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.size() + 4); // +4 for custom_terms_len

        // license_id (16 bytes)
        buf.extend_from_slice(&self.license_id);

        // licensee_hash (32 bytes)
        buf.extend_from_slice(&self.licensee_hash);

        // issued_at (8 bytes)
        buf.extend_from_slice(&self.issued_at.to_le_bytes());

        // expires_at (8 bytes)
        buf.extend_from_slice(&self.expires_at.to_le_bytes());

        // flags (1 byte)
        buf.push(self.flags);

        // seat_limit (2 bytes)
        buf.extend_from_slice(&self.seat_limit.to_le_bytes());

        // query_limit (4 bytes)
        buf.extend_from_slice(&self.query_limit.to_le_bytes());

        // custom_terms_len (4 bytes)
        // Note: terms > 4GB are not supported (reasonable for license terms)
        #[allow(clippy::cast_possible_truncation)]
        let terms_len = self.custom_terms.len() as u32;
        buf.extend_from_slice(&terms_len.to_le_bytes());

        // custom_terms (variable)
        buf.extend_from_slice(&self.custom_terms);

        buf
    }

    /// Deserialize from bytes
    ///
    /// # Errors
    ///
    /// Returns error if buffer is too small or malformed.
    pub fn from_bytes(buf: &[u8]) -> Result<Self> {
        // Need at least fixed size + 4 bytes for terms_len
        const MIN_SIZE: usize = LICENSE_BLOCK_FIXED_SIZE + 4;

        if buf.len() < MIN_SIZE {
            return Err(Error::Format(format!(
                "License block too small: {} bytes, expected at least {}",
                buf.len(),
                MIN_SIZE
            )));
        }

        let mut offset = 0;

        // license_id (16 bytes)
        let mut license_id = [0u8; 16];
        license_id.copy_from_slice(&buf[offset..offset + 16]);
        offset += 16;

        // licensee_hash (32 bytes)
        let mut licensee_hash = [0u8; 32];
        licensee_hash.copy_from_slice(&buf[offset..offset + 32]);
        offset += 32;

        // issued_at (8 bytes)
        let issued_at = u64::from_le_bytes([
            buf[offset],
            buf[offset + 1],
            buf[offset + 2],
            buf[offset + 3],
            buf[offset + 4],
            buf[offset + 5],
            buf[offset + 6],
            buf[offset + 7],
        ]);
        offset += 8;

        // expires_at (8 bytes)
        let expires_at = u64::from_le_bytes([
            buf[offset],
            buf[offset + 1],
            buf[offset + 2],
            buf[offset + 3],
            buf[offset + 4],
            buf[offset + 5],
            buf[offset + 6],
            buf[offset + 7],
        ]);
        offset += 8;

        // flags (1 byte)
        let flags = buf[offset];
        offset += 1;

        // seat_limit (2 bytes)
        let seat_limit = u16::from_le_bytes([buf[offset], buf[offset + 1]]);
        offset += 2;

        // query_limit (4 bytes)
        let query_limit = u32::from_le_bytes([
            buf[offset],
            buf[offset + 1],
            buf[offset + 2],
            buf[offset + 3],
        ]);
        offset += 4;

        // custom_terms_len (4 bytes)
        let terms_len = u32::from_le_bytes([
            buf[offset],
            buf[offset + 1],
            buf[offset + 2],
            buf[offset + 3],
        ]) as usize;
        offset += 4;

        // custom_terms (variable)
        if buf.len() < offset + terms_len {
            return Err(Error::Format(format!(
                "License block truncated: expected {} bytes for custom terms",
                terms_len
            )));
        }

        let custom_terms = buf[offset..offset + terms_len].to_vec();

        Ok(Self {
            license_id,
            licensee_hash,
            issued_at,
            expires_at,
            flags,
            seat_limit,
            query_limit,
            custom_terms,
        })
    }

    /// Check if expiration is enforced
    #[must_use]
    pub const fn is_expiration_enforced(&self) -> bool {
        self.flags & flags::EXPIRATION_ENFORCED != 0
    }

    /// Check if seat limit is enforced
    #[must_use]
    pub const fn is_seats_enforced(&self) -> bool {
        self.flags & flags::SEATS_ENFORCED != 0
    }

    /// Check if query limit is enforced
    #[must_use]
    pub const fn is_query_limited(&self) -> bool {
        self.flags & flags::QUERY_LIMITED != 0
    }

    /// Check if watermarked
    #[must_use]
    pub const fn is_watermarked(&self) -> bool {
        self.flags & flags::WATERMARKED != 0
    }

    /// Check if revocable
    #[must_use]
    pub const fn is_revocable(&self) -> bool {
        self.flags & flags::REVOCABLE != 0
    }

    /// Check if transferable
    #[must_use]
    pub const fn is_transferable(&self) -> bool {
        self.flags & flags::TRANSFERABLE != 0
    }

    /// Verify the license is currently valid
    ///
    /// # Errors
    ///
    /// Returns error if license has expired.
    pub fn verify(&self) -> Result<()> {
        if self.is_expiration_enforced() && self.expires_at > 0 {
            let now = current_unix_time();
            if now > self.expires_at {
                return Err(Error::LicenseExpired {
                    expired_at: self.expires_at,
                    current_time: now,
                });
            }
        }
        Ok(())
    }

    /// Get license ID as UUID string
    #[must_use]
    pub fn license_id_string(&self) -> String {
        format!(
            "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
            u32::from_be_bytes([
                self.license_id[0],
                self.license_id[1],
                self.license_id[2],
                self.license_id[3]
            ]),
            u16::from_be_bytes([self.license_id[4], self.license_id[5]]),
            u16::from_be_bytes([self.license_id[6], self.license_id[7]]),
            u16::from_be_bytes([self.license_id[8], self.license_id[9]]),
            u64::from_be_bytes([
                0,
                0,
                self.license_id[10],
                self.license_id[11],
                self.license_id[12],
                self.license_id[13],
                self.license_id[14],
                self.license_id[15]
            ])
        )
    }
}

/// Get current Unix timestamp in seconds
fn current_unix_time() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};

    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Generate a random UUID v4 for license IDs
///
/// # Errors
///
/// Returns error if random number generation fails.
#[cfg(feature = "format-encryption")]
pub fn generate_license_id() -> Result<[u8; 16]> {
    let mut id = [0u8; 16];
    getrandom::getrandom(&mut id).map_err(|e| Error::Format(format!("RNG error: {e}")))?;

    // Set version 4 (random) and variant bits
    id[6] = (id[6] & 0x0F) | 0x40; // Version 4
    id[8] = (id[8] & 0x3F) | 0x80; // Variant 1

    Ok(id)
}

/// Hash licensee identifier using SHA-256
///
/// This creates a consistent hash for identifying the licensee without
/// storing the raw identifier.
#[cfg(feature = "format-encryption")]
pub fn hash_licensee(identifier: &str) -> [u8; 32] {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    hasher.update(identifier.as_bytes());
    let result = hasher.finalize();

    let mut hash = [0u8; 32];
    hash.copy_from_slice(&result);
    hash
}

/// License builder for convenient construction
#[derive(Debug)]
pub struct LicenseBuilder {
    license_id: [u8; 16],
    licensee_hash: [u8; 32],
    expires_at: Option<u64>,
    seat_limit: Option<u16>,
    query_limit: Option<u32>,
    watermarked: bool,
    revocable: bool,
    transferable: bool,
    custom_terms: Option<Vec<u8>>,
}

impl LicenseBuilder {
    /// Create a new license builder
    #[must_use]
    pub fn new(license_id: [u8; 16], licensee_hash: [u8; 32]) -> Self {
        Self {
            license_id,
            licensee_hash,
            expires_at: None,
            seat_limit: None,
            query_limit: None,
            watermarked: false,
            revocable: false,
            transferable: false,
            custom_terms: None,
        }
    }

    /// Set expiration (Unix timestamp)
    #[must_use]
    pub fn expires_at(mut self, timestamp: u64) -> Self {
        self.expires_at = Some(timestamp);
        self
    }

    /// Set expiration relative to now (in seconds)
    #[must_use]
    pub fn expires_in(mut self, seconds: u64) -> Self {
        self.expires_at = Some(current_unix_time() + seconds);
        self
    }

    /// Set seat limit
    #[must_use]
    pub fn seat_limit(mut self, limit: u16) -> Self {
        self.seat_limit = Some(limit);
        self
    }

    /// Set query limit
    #[must_use]
    pub fn query_limit(mut self, limit: u32) -> Self {
        self.query_limit = Some(limit);
        self
    }

    /// Mark as watermarked
    #[must_use]
    pub fn watermarked(mut self) -> Self {
        self.watermarked = true;
        self
    }

    /// Mark as revocable
    #[must_use]
    pub fn revocable(mut self) -> Self {
        self.revocable = true;
        self
    }

    /// Mark as transferable
    #[must_use]
    pub fn transferable(mut self) -> Self {
        self.transferable = true;
        self
    }

    /// Set custom terms
    #[must_use]
    pub fn custom_terms(mut self, terms: Vec<u8>) -> Self {
        self.custom_terms = Some(terms);
        self
    }

    /// Build the license block
    #[must_use]
    pub fn build(self) -> LicenseBlock {
        let mut license = LicenseBlock::new(self.license_id, self.licensee_hash);

        if let Some(expires) = self.expires_at {
            license = license.with_expiration(expires);
        }

        if let Some(seats) = self.seat_limit {
            license = license.with_seat_limit(seats);
        }

        if let Some(queries) = self.query_limit {
            license = license.with_query_limit(queries);
        }

        if self.watermarked {
            license = license.with_watermark();
        }

        if self.revocable {
            license = license.with_revocable();
        }

        if self.transferable {
            license = license.with_transferable();
        }

        if let Some(terms) = self.custom_terms {
            license = license.with_custom_terms(terms);
        }

        license
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_license_block_roundtrip() {
        let license_id = [1u8; 16];
        let licensee_hash = [2u8; 32];

        let license = LicenseBlock::new(license_id, licensee_hash)
            .with_expiration(1_800_000_000)
            .with_seat_limit(5)
            .with_query_limit(10_000)
            .with_watermark()
            .with_custom_terms(b"Custom license terms".to_vec());

        let bytes = license.to_bytes();
        let restored = LicenseBlock::from_bytes(&bytes).expect("parse failed");

        assert_eq!(restored.license_id, license_id);
        assert_eq!(restored.licensee_hash, licensee_hash);
        assert_eq!(restored.expires_at, 1_800_000_000);
        assert_eq!(restored.seat_limit, 5);
        assert_eq!(restored.query_limit, 10_000);
        assert!(restored.is_expiration_enforced());
        assert!(restored.is_seats_enforced());
        assert!(restored.is_query_limited());
        assert!(restored.is_watermarked());
        assert!(!restored.is_revocable());
        assert!(!restored.is_transferable());
        assert_eq!(restored.custom_terms, b"Custom license terms");
    }

    #[test]
    fn test_license_block_minimal() {
        let license_id = [0u8; 16];
        let licensee_hash = [0u8; 32];

        let license = LicenseBlock::new(license_id, licensee_hash);
        let bytes = license.to_bytes();
        let restored = LicenseBlock::from_bytes(&bytes).expect("parse failed");

        assert_eq!(restored.flags, 0);
        assert_eq!(restored.seat_limit, 0);
        assert_eq!(restored.query_limit, 0);
        assert!(restored.custom_terms.is_empty());
    }

    #[test]
    fn test_license_expiration_check() {
        let license_id = [1u8; 16];
        let licensee_hash = [2u8; 32];

        // Non-expired license
        let future_time = current_unix_time() + 3600; // 1 hour from now
        let valid_license =
            LicenseBlock::new(license_id, licensee_hash).with_expiration(future_time);
        assert!(valid_license.verify().is_ok());

        // Expired license
        let past_time = current_unix_time() - 3600; // 1 hour ago
        let expired_license =
            LicenseBlock::new(license_id, licensee_hash).with_expiration(past_time);
        assert!(expired_license.verify().is_err());
    }

    #[test]
    fn test_license_no_expiration() {
        let license_id = [1u8; 16];
        let licensee_hash = [2u8; 32];

        // License without expiration enforcement should always be valid
        let license = LicenseBlock::new(license_id, licensee_hash);
        assert!(license.verify().is_ok());
    }

    #[test]
    fn test_license_builder() {
        let license_id = [3u8; 16];
        let licensee_hash = [4u8; 32];

        let license = LicenseBuilder::new(license_id, licensee_hash)
            .expires_in(86400) // 24 hours
            .seat_limit(10)
            .query_limit(50_000)
            .watermarked()
            .revocable()
            .build();

        assert!(license.is_expiration_enforced());
        assert!(license.is_seats_enforced());
        assert!(license.is_query_limited());
        assert!(license.is_watermarked());
        assert!(license.is_revocable());
        assert!(!license.is_transferable());
        assert_eq!(license.seat_limit, 10);
        assert_eq!(license.query_limit, 50_000);
    }

    #[test]
    fn test_license_id_string() {
        let license_id = [
            0x12, 0x34, 0x56, 0x78, // time_low
            0x9A, 0xBC, // time_mid
            0xDE, 0xF0, // time_hi_and_version
            0x12, 0x34, // clock_seq
            0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, // node
        ];
        let licensee_hash = [0u8; 32];

        let license = LicenseBlock::new(license_id, licensee_hash);
        let uuid_str = license.license_id_string();

        assert_eq!(uuid_str, "12345678-9abc-def0-1234-56789abcdef0");
    }

    #[test]
    fn test_all_flags() {
        let license_id = [5u8; 16];
        let licensee_hash = [6u8; 32];

        let license = LicenseBlock::new(license_id, licensee_hash)
            .with_expiration(1_900_000_000)
            .with_seat_limit(1)
            .with_query_limit(1)
            .with_watermark()
            .with_revocable()
            .with_transferable();

        let expected_flags = flags::EXPIRATION_ENFORCED
            | flags::SEATS_ENFORCED
            | flags::QUERY_LIMITED
            | flags::WATERMARKED
            | flags::REVOCABLE
            | flags::TRANSFERABLE;

        assert_eq!(license.flags, expected_flags);

        // Verify all flag checks
        assert!(license.is_expiration_enforced());
        assert!(license.is_seats_enforced());
        assert!(license.is_query_limited());
        assert!(license.is_watermarked());
        assert!(license.is_revocable());
        assert!(license.is_transferable());
    }

    #[test]
    fn test_buffer_too_small() {
        let small_buf = [0u8; 10];
        let result = LicenseBlock::from_bytes(&small_buf);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too small"));
    }

    #[cfg(feature = "format-encryption")]
    #[test]
    fn test_generate_license_id() {
        let id1 = generate_license_id().expect("generate failed");
        let id2 = generate_license_id().expect("generate failed");

        // IDs should be different
        assert_ne!(id1, id2);

        // Version should be 4
        assert_eq!((id1[6] >> 4) & 0x0F, 4);
        assert_eq!((id2[6] >> 4) & 0x0F, 4);

        // Variant should be 1 (bits 10xx)
        assert_eq!((id1[8] >> 6) & 0x03, 2);
        assert_eq!((id2[8] >> 6) & 0x03, 2);
    }

    #[cfg(feature = "format-encryption")]
    #[test]
    fn test_hash_licensee() {
        let hash1 = hash_licensee("user@example.com");
        let hash2 = hash_licensee("user@example.com");
        let hash3 = hash_licensee("other@example.com");

        // Same input produces same hash
        assert_eq!(hash1, hash2);

        // Different input produces different hash
        assert_ne!(hash1, hash3);

        // Hash is 32 bytes
        assert_eq!(hash1.len(), 32);
    }
}
