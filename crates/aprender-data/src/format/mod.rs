//! Alimentar Dataset Format (.ald)
//!
//! A binary format for secure, verifiable dataset distribution.
//! See `docs/specifications/dataset-format-spec.md` for full specification.
//!
//! # Format Structure
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │ Header (32 bytes, fixed)                │
//! ├─────────────────────────────────────────┤
//! │ Metadata (variable, MessagePack)        │
//! ├─────────────────────────────────────────┤
//! │ Schema (variable, Arrow IPC)            │
//! ├─────────────────────────────────────────┤
//! │ Payload (variable, Arrow IPC + zstd)    │
//! ├─────────────────────────────────────────┤
//! │ Checksum (4 bytes, CRC32)               │
//! └─────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use alimentar::format::{save, load, SaveOptions, DatasetType};
//!
//! // Save dataset
//! save(&dataset, DatasetType::Tabular, "data.ald", SaveOptions::default())?;
//!
//! // Load dataset
//! let dataset = load("data.ald")?;
//! ```

mod crc;
#[cfg(feature = "format-encryption")]
pub mod encryption;
pub mod license;
pub mod piracy;
#[cfg(feature = "format-signing")]
pub mod signing;
#[cfg(feature = "format-streaming")]
pub mod streaming;

pub use crc::crc32;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Magic bytes: "ALDF" (0x414C4446)
pub const MAGIC: [u8; 4] = [0x41, 0x4C, 0x44, 0x46];

/// Current format version major number
pub const FORMAT_VERSION_MAJOR: u8 = 1;
/// Current format version minor number
pub const FORMAT_VERSION_MINOR: u8 = 2;

/// Header size in bytes (fixed)
pub const HEADER_SIZE: usize = 32;

/// Header flags (bit positions)
pub mod flags {
    /// Payload encrypted (AES-256-GCM)
    pub const ENCRYPTED: u8 = 0b0000_0001;
    /// Has digital signature (Ed25519)
    pub const SIGNED: u8 = 0b0000_0010;
    /// Supports chunked/mmap loading (native only)
    pub const STREAMING: u8 = 0b0000_0100;
    /// Has commercial license block
    pub const LICENSED: u8 = 0b0000_1000;
    /// 64-byte aligned arrays for zero-copy SIMD (native only)
    pub const TRUENO_NATIVE: u8 = 0b0001_0000;
}

/// Dataset type identifiers (§3.1)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u16)]
pub enum DatasetType {
    // Structured Data (0x0001-0x000F)
    /// Generic columnar data
    Tabular = 0x0001,
    /// Temporal sequences
    TimeSeries = 0x0002,
    /// Node/edge structures
    Graph = 0x0003,
    /// Geospatial coordinates
    Spatial = 0x0004,

    // Text & NLP (0x0010-0x001F)
    /// Raw text documents
    TextCorpus = 0x0010,
    /// Labeled text
    TextClassification = 0x0011,
    /// Sentence pairs (NLI, STS)
    TextPairs = 0x0012,
    /// Token-level labels (NER)
    SequenceLabeling = 0x0013,
    /// QA datasets (SQuAD-style)
    QuestionAnswering = 0x0014,
    /// Document + summary pairs
    Summarization = 0x0015,
    /// Parallel corpora
    Translation = 0x0016,

    // Computer Vision (0x0020-0x002F)
    /// Images + class labels
    ImageClassification = 0x0020,
    /// Images + bounding boxes
    ObjectDetection = 0x0021,
    /// Images + pixel masks
    Segmentation = 0x0022,
    /// Image matching/similarity
    ImagePairs = 0x0023,
    /// Sequential frames
    Video = 0x0024,

    // Audio (0x0030-0x003F)
    /// Audio + class labels
    AudioClassification = 0x0030,
    /// Audio + transcripts (ASR)
    SpeechRecognition = 0x0031,
    /// Audio + speaker labels
    SpeakerIdentification = 0x0032,

    // Recommender Systems (0x0040-0x004F)
    /// Collaborative filtering
    UserItemRatings = 0x0040,
    /// Click/view interactions
    ImplicitFeedback = 0x0041,
    /// Session-based sequences
    SequentialRecs = 0x0042,

    // Multimodal (0x0050-0x005F)
    /// Image captioning/VQA
    ImageText = 0x0050,
    /// Speech-to-text pairs
    AudioText = 0x0051,
    /// Video descriptions
    VideoText = 0x0052,

    // Special
    /// User extensions
    Custom = 0x00FF,
}

impl DatasetType {
    /// Convert from u16 value
    #[must_use]
    pub fn from_u16(value: u16) -> Option<Self> {
        match value {
            0x0001 => Some(Self::Tabular),
            0x0002 => Some(Self::TimeSeries),
            0x0003 => Some(Self::Graph),
            0x0004 => Some(Self::Spatial),
            0x0010 => Some(Self::TextCorpus),
            0x0011 => Some(Self::TextClassification),
            0x0012 => Some(Self::TextPairs),
            0x0013 => Some(Self::SequenceLabeling),
            0x0014 => Some(Self::QuestionAnswering),
            0x0015 => Some(Self::Summarization),
            0x0016 => Some(Self::Translation),
            0x0020 => Some(Self::ImageClassification),
            0x0021 => Some(Self::ObjectDetection),
            0x0022 => Some(Self::Segmentation),
            0x0023 => Some(Self::ImagePairs),
            0x0024 => Some(Self::Video),
            0x0030 => Some(Self::AudioClassification),
            0x0031 => Some(Self::SpeechRecognition),
            0x0032 => Some(Self::SpeakerIdentification),
            0x0040 => Some(Self::UserItemRatings),
            0x0041 => Some(Self::ImplicitFeedback),
            0x0042 => Some(Self::SequentialRecs),
            0x0050 => Some(Self::ImageText),
            0x0051 => Some(Self::AudioText),
            0x0052 => Some(Self::VideoText),
            0x00FF => Some(Self::Custom),
            _ => None,
        }
    }

    /// Convert to u16 value
    #[must_use]
    pub const fn as_u16(self) -> u16 {
        self as u16
    }
}

/// Compression algorithm identifiers (§3.3)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[repr(u8)]
pub enum Compression {
    /// No compression (debugging)
    None = 0x00,
    /// Zstd level 3 (standard distribution)
    #[default]
    ZstdL3 = 0x01,
    /// Zstd level 19 (archival, max compression)
    ZstdL19 = 0x02,
    /// LZ4 (high-throughput streaming)
    Lz4 = 0x03,
}

impl Compression {
    /// Convert from u8 value
    #[must_use]
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0x00 => Some(Self::None),
            0x01 => Some(Self::ZstdL3),
            0x02 => Some(Self::ZstdL19),
            0x03 => Some(Self::Lz4),
            _ => None,
        }
    }

    /// Convert to u8 value
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }
}

/// File header (32 bytes, fixed)
///
/// | Offset | Size | Field |
/// |--------|------|-------|
/// | 0 | 4 | magic |
/// | 4 | 2 | format_version (major.minor) |
/// | 6 | 2 | dataset_type |
/// | 8 | 4 | metadata_size |
/// | 12 | 4 | payload_size (compressed) |
/// | 16 | 4 | uncompressed_size |
/// | 20 | 1 | compression |
/// | 21 | 1 | flags |
/// | 22 | 2 | schema_size |
/// | 24 | 8 | num_rows |
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    /// Format version (major, minor)
    pub version: (u8, u8),
    /// Dataset type identifier
    pub dataset_type: DatasetType,
    /// Metadata block size in bytes
    pub metadata_size: u32,
    /// Compressed payload size in bytes
    pub payload_size: u32,
    /// Uncompressed payload size in bytes
    pub uncompressed_size: u32,
    /// Compression algorithm
    pub compression: Compression,
    /// Feature flags
    pub flags: u8,
    /// Schema block size in bytes
    pub schema_size: u16,
    /// Total row count
    pub num_rows: u64,
}

impl Header {
    /// Create a new header with default values
    #[must_use]
    pub fn new(dataset_type: DatasetType) -> Self {
        // Contract: configuration-v1.yaml precondition (pv codegen)
        contract_pre_configuration!();
        Self {
            version: (FORMAT_VERSION_MAJOR, FORMAT_VERSION_MINOR),
            dataset_type,
            metadata_size: 0,
            payload_size: 0,
            uncompressed_size: 0,
            compression: Compression::default(),
            flags: 0,
            schema_size: 0,
            num_rows: 0,
        }
    }

    /// Serialize header to 32 bytes
    #[must_use]
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        // Contract: serialization-v1.yaml precondition (pv codegen)
        contract_pre_configuration!();
        let mut buf = [0u8; HEADER_SIZE];

        // Magic (0-3)
        buf[0..4].copy_from_slice(&MAGIC);

        // Version (4-5)
        buf[4] = self.version.0;
        buf[5] = self.version.1;

        // Dataset type (6-7)
        let dt = self.dataset_type.as_u16().to_le_bytes();
        buf[6..8].copy_from_slice(&dt);

        // Metadata size (8-11)
        buf[8..12].copy_from_slice(&self.metadata_size.to_le_bytes());

        // Payload size (12-15)
        buf[12..16].copy_from_slice(&self.payload_size.to_le_bytes());

        // Uncompressed size (16-19)
        buf[16..20].copy_from_slice(&self.uncompressed_size.to_le_bytes());

        // Compression (20)
        buf[20] = self.compression.as_u8();

        // Flags (21)
        buf[21] = self.flags;

        // Schema size (22-23)
        buf[22..24].copy_from_slice(&self.schema_size.to_le_bytes());

        // Num rows (24-31)
        buf[24..32].copy_from_slice(&self.num_rows.to_le_bytes());

        buf
    }

    /// Deserialize header from bytes
    ///
    /// # Errors
    ///
    /// Returns error if magic is invalid, version is unsupported, or types are
    /// unknown.
    pub fn from_bytes(buf: &[u8]) -> Result<Self> {
        if buf.len() < HEADER_SIZE {
            return Err(Error::Format(format!(
                "Header too short: {} bytes, expected {}",
                buf.len(),
                HEADER_SIZE
            )));
        }

        // Validate magic
        if buf[0..4] != MAGIC {
            return Err(Error::Format(format!(
                "Invalid magic: expected {:?}, got {:?}",
                MAGIC,
                &buf[0..4]
            )));
        }

        // Contract: serialization-v1.yaml precondition (pv codegen)
        contract_pre_configuration!(buf);

        // Version
        let version = (buf[4], buf[5]);
        if version.0 > FORMAT_VERSION_MAJOR {
            return Err(Error::Format(format!(
                "Unsupported version: {}.{}, max supported: {}.{}",
                version.0, version.1, FORMAT_VERSION_MAJOR, FORMAT_VERSION_MINOR
            )));
        }

        // Dataset type
        let dt_value = u16::from_le_bytes([buf[6], buf[7]]);
        let dataset_type = DatasetType::from_u16(dt_value)
            .ok_or_else(|| Error::Format(format!("Unknown dataset type: 0x{:04X}", dt_value)))?;

        // Metadata size
        let metadata_size = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);

        // Payload size
        let payload_size = u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]);

        // Uncompressed size
        let uncompressed_size = u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]);

        // Compression
        let compression = Compression::from_u8(buf[20])
            .ok_or_else(|| Error::Format(format!("Unknown compression: 0x{:02X}", buf[20])))?;

        // Flags
        let flags = buf[21];

        // Schema size
        let schema_size = u16::from_le_bytes([buf[22], buf[23]]);

        // Num rows
        let num_rows = u64::from_le_bytes([
            buf[24], buf[25], buf[26], buf[27], buf[28], buf[29], buf[30], buf[31],
        ]);

        Ok(Self {
            version,
            dataset_type,
            metadata_size,
            payload_size,
            uncompressed_size,
            compression,
            flags,
            schema_size,
            num_rows,
        })
    }

    /// Check if encrypted flag is set
    #[must_use]
    pub const fn is_encrypted(&self) -> bool {
        self.flags & flags::ENCRYPTED != 0
    }

    /// Check if signed flag is set
    #[must_use]
    pub const fn is_signed(&self) -> bool {
        self.flags & flags::SIGNED != 0
    }

    /// Check if streaming flag is set
    #[must_use]
    pub const fn is_streaming(&self) -> bool {
        self.flags & flags::STREAMING != 0
    }

    /// Check if licensed flag is set
    #[must_use]
    pub const fn is_licensed(&self) -> bool {
        self.flags & flags::LICENSED != 0
    }

    /// Check if trueno-native flag is set
    #[must_use]
    pub const fn is_trueno_native(&self) -> bool {
        self.flags & flags::TRUENO_NATIVE != 0
    }
}

/// Dataset metadata (MessagePack-encoded)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Metadata {
    /// Human-readable name
    pub name: Option<String>,
    /// Version (semver)
    pub version: Option<String>,
    /// SPDX license identifier
    pub license: Option<String>,
    /// Searchable tags
    #[serde(default)]
    pub tags: Vec<String>,
    /// Markdown description
    pub description: Option<String>,
    /// Citation (BibTeX)
    pub citation: Option<String>,
    /// Creation timestamp (RFC 3339)
    pub created_at: Option<String>,
    /// SHA-256 hash of the payload data (hex string, 64 chars)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}

/// Computes SHA-256 hash of data and returns it as a hex string.
///
/// # Example
///
/// ```
/// use alimentar::format::sha256_hex;
///
/// let hash = sha256_hex(b"Hello, World!");
/// assert_eq!(hash.len(), 64); // 256 bits = 32 bytes = 64 hex chars
/// ```
#[cfg(feature = "provenance")]
#[must_use]
pub fn sha256_hex(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();

    // Convert to hex string
    result.iter().fold(String::with_capacity(64), |mut s, b| {
        use std::fmt::Write;
        let _ = write!(s, "{b:02x}");
        s
    })
}

/// Options for saving datasets
#[derive(Debug, Clone)]
pub struct SaveOptions {
    /// Compression algorithm to use
    pub compression: Compression,
    /// Optional metadata to include
    pub metadata: Option<Metadata>,
    /// Encryption parameters (requires `format-encryption` feature)
    #[cfg(feature = "format-encryption")]
    pub encryption: Option<encryption::EncryptionParams>,
    /// Signing key pair (requires `format-signing` feature)
    #[cfg(feature = "format-signing")]
    pub signing_key: Option<signing::SigningKeyPair>,
    /// License block for commercial distribution
    pub license: Option<license::LicenseBlock>,
}

impl Default for SaveOptions {
    fn default() -> Self {
        Self {
            compression: Compression::ZstdL3,
            metadata: None,
            #[cfg(feature = "format-encryption")]
            encryption: None,
            #[cfg(feature = "format-signing")]
            signing_key: None,
            license: None,
        }
    }
}

impl SaveOptions {
    /// Set compression algorithm
    #[must_use]
    pub fn with_compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
        self
    }

    /// Set metadata
    #[must_use]
    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Set password-based encryption (requires `format-encryption` feature)
    #[cfg(feature = "format-encryption")]
    #[must_use]
    pub fn with_password(mut self, password: impl Into<String>) -> Self {
        self.encryption = Some(encryption::EncryptionParams::password(password));
        self
    }

    /// Set recipient-based encryption (requires `format-encryption` feature)
    #[cfg(feature = "format-encryption")]
    #[must_use]
    pub fn with_recipient(mut self, public_key: [u8; 32]) -> Self {
        self.encryption = Some(encryption::EncryptionParams::recipient(public_key));
        self
    }

    /// Set signing key (requires `format-signing` feature)
    #[cfg(feature = "format-signing")]
    #[must_use]
    pub fn with_signing_key(mut self, key: signing::SigningKeyPair) -> Self {
        self.signing_key = Some(key);
        self
    }

    /// Set license block
    #[must_use]
    pub fn with_license(mut self, license: license::LicenseBlock) -> Self {
        self.license = Some(license);
        self
    }
}

/// Options for loading datasets
#[derive(Debug, Clone, Default)]
pub struct LoadOptions {
    /// Decryption parameters (required if dataset is encrypted)
    #[cfg(feature = "format-encryption")]
    pub decryption: Option<encryption::DecryptionParams>,
    /// Trusted public keys for signature verification (if empty, skip
    /// verification)
    #[cfg(feature = "format-signing")]
    pub trusted_keys: Vec<[u8; 32]>,
    /// Whether to verify license expiration
    pub verify_license: bool,
}

impl LoadOptions {
    /// Set password for decryption
    #[cfg(feature = "format-encryption")]
    #[must_use]
    pub fn with_password(mut self, password: impl Into<String>) -> Self {
        self.decryption = Some(encryption::DecryptionParams::password(password));
        self
    }

    /// Set private key for decryption
    #[cfg(feature = "format-encryption")]
    #[must_use]
    pub fn with_private_key(mut self, key: [u8; 32]) -> Self {
        self.decryption = Some(encryption::DecryptionParams::private_key(key));
        self
    }

    /// Add trusted public key for signature verification
    #[cfg(feature = "format-signing")]
    #[must_use]
    pub fn with_trusted_key(mut self, key: [u8; 32]) -> Self {
        self.trusted_keys.push(key);
        self
    }

    /// Enable license verification
    #[must_use]
    pub fn verify_license(mut self) -> Self {
        self.verify_license = true;
        self
    }
}

/// Save an Arrow dataset to the .ald format
///
/// # Errors
///
/// Returns error if serialization or I/O fails.
#[allow(clippy::cast_possible_truncation, clippy::too_many_lines)]
pub fn save<W: std::io::Write>(
    writer: &mut W,
    batches: &[arrow::array::RecordBatch],
    dataset_type: DatasetType,
    options: &SaveOptions,
) -> Result<()> {
    use arrow::ipc::writer::StreamWriter;

    if batches.is_empty() {
        return Err(Error::EmptyDataset);
    }

    let schema = batches[0].schema();

    // Serialize schema via Arrow IPC
    let mut schema_buf = Vec::new();
    {
        let mut schema_writer =
            StreamWriter::try_new(&mut schema_buf, &schema).map_err(Error::Arrow)?;
        schema_writer.finish().map_err(Error::Arrow)?;
    }

    // Serialize payload via Arrow IPC
    let mut payload_buf = Vec::new();
    {
        let mut payload_writer =
            StreamWriter::try_new(&mut payload_buf, &schema).map_err(Error::Arrow)?;
        for batch in batches {
            payload_writer.write(batch).map_err(Error::Arrow)?;
        }
        payload_writer.finish().map_err(Error::Arrow)?;
    }

    let uncompressed_size = payload_buf.len() as u32;

    // Compress payload if needed
    let compressed_payload = compress_payload(payload_buf, options.compression)?;

    // Build flags
    let mut header_flags: u8 = 0;

    // Encryption: build block, split into header and ciphertext payload
    #[cfg(feature = "format-encryption")]
    let (final_payload, encryption_header) = if let Some(ref enc_params) = options.encryption {
        header_flags |= flags::ENCRYPTED;
        let block = build_encryption_block(&compressed_payload, enc_params)?;
        let hdr_size = encryption_block_header_size(block[0]);
        (block[hdr_size..].to_vec(), block[..hdr_size].to_vec())
    } else {
        (compressed_payload, Vec::new())
    };
    #[cfg(not(feature = "format-encryption"))]
    let (final_payload, encryption_header): (Vec<u8>, Vec<u8>) = (compressed_payload, Vec::new());

    // Signing setup
    #[cfg(feature = "format-signing")]
    if options.signing_key.is_some() {
        header_flags |= flags::SIGNED;
    }

    // License setup
    if options.license.is_some() {
        header_flags |= flags::LICENSED;
    }

    // Serialize metadata
    let metadata_buf = if let Some(ref meta) = options.metadata {
        rmp_serde::to_vec(meta).map_err(|e| Error::Format(e.to_string()))?
    } else {
        rmp_serde::to_vec(&Metadata::default()).map_err(|e| Error::Format(e.to_string()))?
    };

    // Count total rows
    let num_rows: u64 = batches.iter().map(|b| b.num_rows() as u64).sum();

    // Build header
    let header = Header {
        version: (FORMAT_VERSION_MAJOR, FORMAT_VERSION_MINOR),
        dataset_type,
        metadata_size: metadata_buf.len() as u32,
        payload_size: final_payload.len() as u32,
        uncompressed_size,
        compression: options.compression,
        flags: header_flags,
        schema_size: schema_buf.len() as u16,
        num_rows,
    };

    // Build all data for checksum and signature
    let mut all_data = Vec::new();
    let header_bytes = header.to_bytes();
    all_data.extend_from_slice(&header_bytes);
    all_data.extend_from_slice(&metadata_buf);
    all_data.extend_from_slice(&schema_buf);
    all_data.extend_from_slice(&encryption_header);
    all_data.extend_from_slice(&final_payload);

    // Add signature block if signing
    #[cfg(feature = "format-signing")]
    let signature_block: Option<[u8; signing::SignatureBlock::SIZE]> =
        if let Some(ref key) = options.signing_key {
            let sig_block = signing::SignatureBlock::sign(&all_data, key);
            let sig_bytes = sig_block.to_bytes();
            all_data.extend_from_slice(&sig_bytes);
            Some(sig_bytes)
        } else {
            None
        };
    #[cfg(not(feature = "format-signing"))]
    let signature_block: Option<[u8; 96]> = None;

    // Add license block if present
    let license_bytes: Option<Vec<u8>> = if let Some(ref lic) = options.license {
        let lic_bytes = lic.to_bytes();
        all_data.extend_from_slice(&lic_bytes);
        Some(lic_bytes)
    } else {
        None
    };

    // Calculate checksum over all preceding data
    let checksum = crc32(&all_data);

    // Write everything
    writer.write_all(&header_bytes).map_err(Error::io_no_path)?;
    writer.write_all(&metadata_buf).map_err(Error::io_no_path)?;
    writer.write_all(&schema_buf).map_err(Error::io_no_path)?;
    writer
        .write_all(&encryption_header)
        .map_err(Error::io_no_path)?;
    writer
        .write_all(&final_payload)
        .map_err(Error::io_no_path)?;

    if let Some(ref sig) = signature_block {
        writer.write_all(sig).map_err(Error::io_no_path)?;
    }

    if let Some(ref lic) = license_bytes {
        writer.write_all(lic).map_err(Error::io_no_path)?;
    }

    writer
        .write_all(&checksum.to_le_bytes())
        .map_err(Error::io_no_path)?;

    Ok(())
}

/// Compress a payload buffer using the specified compression method.
fn compress_payload(payload: Vec<u8>, compression: Compression) -> Result<Vec<u8>> {
    match compression {
        Compression::None => Ok(payload),
        Compression::ZstdL3 => zstd::encode_all(payload.as_slice(), 3).map_err(Error::io_no_path),
        Compression::ZstdL19 => zstd::encode_all(payload.as_slice(), 19).map_err(Error::io_no_path),
        Compression::Lz4 => {
            let mut encoder = lz4_flex::frame::FrameEncoder::new(Vec::new());
            std::io::Write::write_all(&mut encoder, &payload).map_err(Error::io_no_path)?;
            encoder
                .finish()
                .map_err(|e| Error::Format(format!("LZ4 compression error: {e}")))
        }
    }
}

/// Return the header size for an encryption block based on its mode byte.
#[cfg(feature = "format-encryption")]
fn encryption_block_header_size(mode: u8) -> usize {
    if mode == encryption::mode::PASSWORD {
        1 + 16 + 12 // mode + salt + nonce
    } else {
        1 + 32 + 12 // mode + ephemeral_pub + nonce
    }
}

/// Build encryption block: mode + key_material + nonce + ciphertext
#[cfg(feature = "format-encryption")]
fn build_encryption_block(
    plaintext: &[u8],
    params: &encryption::EncryptionParams,
) -> Result<Vec<u8>> {
    match &params.mode {
        encryption::EncryptionMode::Password(password) => {
            let (mode, salt, nonce, ciphertext) =
                encryption::encrypt_password(plaintext, password)?;
            let mut block = Vec::with_capacity(1 + 16 + 12 + ciphertext.len());
            block.push(mode);
            block.extend_from_slice(&salt);
            block.extend_from_slice(&nonce);
            block.extend_from_slice(&ciphertext);
            Ok(block)
        }
        encryption::EncryptionMode::Recipient {
            recipient_public_key,
        } => {
            let (mode, ephemeral_pub, nonce, ciphertext) =
                encryption::encrypt_recipient(plaintext, recipient_public_key)?;
            let mut block = Vec::with_capacity(1 + 32 + 12 + ciphertext.len());
            block.push(mode);
            block.extend_from_slice(&ephemeral_pub);
            block.extend_from_slice(&nonce);
            block.extend_from_slice(&ciphertext);
            Ok(block)
        }
    }
}

/// Loaded dataset from .ald format
#[derive(Debug)]
pub struct LoadedDataset {
    /// Parsed header
    pub header: Header,
    /// Dataset metadata
    pub metadata: Metadata,
    /// Arrow record batches
    pub batches: Vec<arrow::array::RecordBatch>,
    /// License block (if present)
    pub license: Option<license::LicenseBlock>,
    /// Signer public key (if signed and verified)
    pub signer_public_key: Option<[u8; 32]>,
}

/// Load an Arrow dataset from the .ald format (unencrypted only)
///
/// For encrypted or signed datasets, use `load_with_options`.
///
/// # Errors
///
/// Returns error if dataset is encrypted, or if deserialization,
/// decompression, or checksum validation fails.
pub fn load<R: std::io::Read>(reader: &mut R) -> Result<LoadedDataset> {
    load_with_options(reader, &LoadOptions::default())
}

/// Load an Arrow dataset with decryption and verification options
///
/// # Errors
///
/// Returns error if deserialization, decompression, decryption,
/// signature verification, license validation, or checksum validation fails.
#[allow(clippy::too_many_lines)]
pub fn load_with_options<R: std::io::Read>(
    reader: &mut R,
    options: &LoadOptions,
) -> Result<LoadedDataset> {
    use arrow::ipc::reader::StreamReader;

    // Read all data
    let mut all_data = Vec::new();
    reader
        .read_to_end(&mut all_data)
        .map_err(Error::io_no_path)?;

    if all_data.len() < HEADER_SIZE + 4 {
        return Err(Error::Format("File too small".to_string()));
    }

    // Split off checksum (last 4 bytes)
    let checksum_offset = all_data.len() - 4;
    let stored_checksum = u32::from_le_bytes([
        all_data[checksum_offset],
        all_data[checksum_offset + 1],
        all_data[checksum_offset + 2],
        all_data[checksum_offset + 3],
    ]);

    // Verify checksum
    let computed_checksum = crc32(&all_data[..checksum_offset]);
    if stored_checksum != computed_checksum {
        return Err(Error::ChecksumMismatch {
            expected: stored_checksum,
            actual: computed_checksum,
        });
    }

    // Parse header
    let header = Header::from_bytes(&all_data[..HEADER_SIZE])?;

    // Parse metadata
    let metadata_start = HEADER_SIZE;
    let metadata_end = metadata_start + header.metadata_size as usize;
    let metadata: Metadata = rmp_serde::from_slice(&all_data[metadata_start..metadata_end])
        .map_err(|e| Error::Format(format!("Metadata parse error: {e}")))?;

    // Skip schema (embedded in payload IPC stream)
    let schema_end = metadata_end + header.schema_size as usize;

    // Determine encryption header size
    let encryption_header_size = determine_encryption_header_size(&header, &all_data, schema_end)?;

    let payload_start = schema_end + encryption_header_size;
    let payload_end = payload_start + header.payload_size as usize;

    if payload_end > checksum_offset {
        return Err(Error::Format("Payload extends beyond data".to_string()));
    }

    // Extract and decrypt payload if encrypted
    let compressed_payload: Vec<u8> = if header.is_encrypted() {
        #[cfg(feature = "format-encryption")]
        {
            let enc_header = &all_data[schema_end..payload_start];
            let ciphertext = &all_data[payload_start..payload_end];

            let decryption_params = options.decryption.as_ref().ok_or_else(|| {
                Error::Format("Dataset is encrypted but no decryption params provided".to_string())
            })?;

            decrypt_payload(enc_header, ciphertext, decryption_params)?
        }
        #[cfg(not(feature = "format-encryption"))]
        {
            return Err(Error::Format(
                "Dataset is encrypted but format-encryption feature is not enabled".to_string(),
            ));
        }
    } else {
        all_data[payload_start..payload_end].to_vec()
    };

    // Parse trailing blocks (signature, license)
    let (signer_public_key, license_block) =
        parse_trailing_blocks(&header, &all_data, payload_end, checksum_offset, options)?;

    // Decompress payload
    let decompressed_payload = decompress_payload(compressed_payload, header.compression)?;

    // Parse Arrow IPC stream
    let cursor = std::io::Cursor::new(decompressed_payload);
    let stream_reader = StreamReader::try_new(cursor, None).map_err(Error::Arrow)?;

    let batches: Vec<_> = stream_reader
        .into_iter()
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(Error::Arrow)?;

    Ok(LoadedDataset {
        header,
        metadata,
        batches,
        license: license_block,
        signer_public_key,
    })
}

/// Determine the encryption header size from the mode byte.
fn determine_encryption_header_size(
    header: &Header,
    all_data: &[u8],
    schema_end: usize,
) -> Result<usize> {
    if !header.is_encrypted() {
        return Ok(0);
    }

    if all_data.len() <= schema_end {
        return Err(Error::Format("Missing encryption header".to_string()));
    }

    #[cfg(feature = "format-encryption")]
    {
        Ok(encryption_block_header_size(all_data[schema_end]))
    }
    #[cfg(not(feature = "format-encryption"))]
    {
        Err(Error::Format(
            "Dataset is encrypted but format-encryption feature is not enabled".to_string(),
        ))
    }
}

/// Parse trailing blocks (signature, license) after the payload.
fn parse_trailing_blocks(
    header: &Header,
    all_data: &[u8],
    payload_end: usize,
    checksum_offset: usize,
    options: &LoadOptions,
) -> Result<(Option<[u8; 32]>, Option<license::LicenseBlock>)> {
    #[allow(unused_mut)]
    let mut trailing_offset = payload_end;
    #[allow(unused_mut)]
    let mut signer_public_key: Option<[u8; 32]> = None;
    let mut license_block: Option<license::LicenseBlock> = None;

    if header.is_signed() {
        #[cfg(feature = "format-signing")]
        {
            let sig_end = trailing_offset + signing::SignatureBlock::SIZE;
            if sig_end > checksum_offset {
                return Err(Error::Format(
                    "Signature block extends beyond data".to_string(),
                ));
            }

            let sig_block =
                signing::SignatureBlock::from_bytes(&all_data[trailing_offset..sig_end])?;

            if !options.trusted_keys.is_empty() {
                let signed_data = &all_data[..trailing_offset];
                if !options.trusted_keys.contains(&sig_block.public_key) {
                    return Err(Error::Format("Signer not in trusted keys list".to_string()));
                }
                sig_block.verify(signed_data)?;
            }

            signer_public_key = Some(sig_block.public_key);
            trailing_offset = sig_end;
        }
        #[cfg(not(feature = "format-signing"))]
        {
            return Err(Error::Format(
                "Dataset is signed but format-signing feature is not enabled".to_string(),
            ));
        }
    }

    if header.is_licensed() {
        if trailing_offset >= checksum_offset {
            return Err(Error::Format("Missing license block".to_string()));
        }
        let lic = license::LicenseBlock::from_bytes(&all_data[trailing_offset..checksum_offset])?;
        if options.verify_license {
            lic.verify()?;
        }
        license_block = Some(lic);
    }

    Ok((signer_public_key, license_block))
}

/// Decompress a payload buffer using the specified compression method.
fn decompress_payload(payload: Vec<u8>, compression: Compression) -> Result<Vec<u8>> {
    match compression {
        Compression::None => Ok(payload),
        Compression::ZstdL3 | Compression::ZstdL19 => zstd::decode_all(payload.as_slice())
            .map_err(|e| Error::Format(format!("Zstd decompression error: {e}"))),
        Compression::Lz4 => {
            let mut decoder = lz4_flex::frame::FrameDecoder::new(payload.as_slice());
            let mut decompressed = Vec::new();
            std::io::Read::read_to_end(&mut decoder, &mut decompressed)
                .map_err(|e| Error::Format(format!("LZ4 decompression error: {e}")))?;
            Ok(decompressed)
        }
    }
}

/// Decrypt payload using the provided parameters
#[cfg(feature = "format-encryption")]
fn decrypt_payload(
    enc_header: &[u8],
    ciphertext: &[u8],
    params: &encryption::DecryptionParams,
) -> Result<Vec<u8>> {
    if enc_header.is_empty() {
        return Err(Error::Format("Empty encryption header".to_string()));
    }

    let mode = enc_header[0];

    match (mode, params) {
        (encryption::mode::PASSWORD, encryption::DecryptionParams::Password(password)) => {
            if enc_header.len() < 1 + 16 + 12 {
                return Err(Error::Format(
                    "Invalid password encryption header".to_string(),
                ));
            }
            let mut salt = [0u8; 16];
            let mut nonce = [0u8; 12];
            salt.copy_from_slice(&enc_header[1..17]);
            nonce.copy_from_slice(&enc_header[17..29]);

            encryption::decrypt_password(ciphertext, password, &salt, &nonce)
        }
        (encryption::mode::RECIPIENT, encryption::DecryptionParams::PrivateKey(private_key)) => {
            if enc_header.len() < 1 + 32 + 12 {
                return Err(Error::Format(
                    "Invalid recipient encryption header".to_string(),
                ));
            }
            let mut ephemeral_pub = [0u8; 32];
            let mut nonce = [0u8; 12];
            ephemeral_pub.copy_from_slice(&enc_header[1..33]);
            nonce.copy_from_slice(&enc_header[33..45]);

            encryption::decrypt_recipient(ciphertext, private_key, &ephemeral_pub, &nonce)
        }
        (encryption::mode::PASSWORD, encryption::DecryptionParams::PrivateKey(_)) => Err(
            Error::Format("Dataset encrypted with password but private key provided".to_string()),
        ),
        (encryption::mode::RECIPIENT, encryption::DecryptionParams::Password(_)) => Err(
            Error::Format("Dataset encrypted for recipient but password provided".to_string()),
        ),
        _ => Err(Error::Format(format!("Unknown encryption mode: {mode}"))),
    }
}

/// Save dataset to a file path
///
/// # Errors
///
/// Returns error if file creation or serialization fails.
pub fn save_to_file<P: AsRef<std::path::Path>>(
    path: P,
    batches: &[arrow::array::RecordBatch],
    dataset_type: DatasetType,
    options: &SaveOptions,
) -> Result<()> {
    let file = std::fs::File::create(path.as_ref())
        .map_err(|e| Error::io(e, path.as_ref().to_path_buf()))?;
    let mut writer = std::io::BufWriter::new(file);
    save(&mut writer, batches, dataset_type, options)
}

/// Load dataset from a file path
///
/// # Errors
///
/// Returns error if file reading or deserialization fails.
pub fn load_from_file<P: AsRef<std::path::Path>>(path: P) -> Result<LoadedDataset> {
    load_from_file_with_options(path, &LoadOptions::default())
}

/// Load dataset from a file path with decryption and verification options
///
/// # Errors
///
/// Returns error if file reading, decryption, verification, or deserialization
/// fails.
pub fn load_from_file_with_options<P: AsRef<std::path::Path>>(
    path: P,
    options: &LoadOptions,
) -> Result<LoadedDataset> {
    let file = std::fs::File::open(path.as_ref())
        .map_err(|e| Error::io(e, path.as_ref().to_path_buf()))?;
    let mut reader = std::io::BufReader::new(file);
    load_with_options(&mut reader, options)
}

#[cfg(test)]
mod tests;
