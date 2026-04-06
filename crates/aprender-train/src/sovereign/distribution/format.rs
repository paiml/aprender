//! Distribution format options

use serde::{Deserialize, Serialize};

/// Distribution format options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum DistributionFormat {
    /// Bootable ISO with NixOS
    Iso,
    /// OCI container image
    Oci,
    /// Nix flake
    Nix,
    /// Flatpak bundle
    Flatpak,
    /// Simple tar.gz
    #[default]
    Tarball,
}

impl DistributionFormat {
    /// Get file extension for the format
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Iso => "iso",
            Self::Oci => "tar",
            Self::Nix => "nix",
            Self::Flatpak => "flatpak",
            Self::Tarball => "tar.gz",
        }
    }

    /// Get MIME type for the format
    pub fn mime_type(&self) -> &'static str {
        match self {
            Self::Iso => "application/x-iso9660-image",
            Self::Oci => "application/vnd.oci.image.layer.v1.tar",
            Self::Nix => "text/plain",
            Self::Flatpak => "application/vnd.flatpak",
            Self::Tarball => "application/gzip",
        }
    }
}

impl std::fmt::Display for DistributionFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Iso => write!(f, "ISO"),
            Self::Oci => write!(f, "OCI"),
            Self::Nix => write!(f, "Nix"),
            Self::Flatpak => write!(f, "Flatpak"),
            Self::Tarball => write!(f, "Tarball"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distribution_format_default() {
        assert_eq!(DistributionFormat::default(), DistributionFormat::Tarball);
    }

    #[test]
    fn test_distribution_format_extension_iso() {
        assert_eq!(DistributionFormat::Iso.extension(), "iso");
    }

    #[test]
    fn test_distribution_format_extension_oci() {
        assert_eq!(DistributionFormat::Oci.extension(), "tar");
    }

    #[test]
    fn test_distribution_format_extension_nix() {
        assert_eq!(DistributionFormat::Nix.extension(), "nix");
    }

    #[test]
    fn test_distribution_format_extension_flatpak() {
        assert_eq!(DistributionFormat::Flatpak.extension(), "flatpak");
    }

    #[test]
    fn test_distribution_format_extension_tarball() {
        assert_eq!(DistributionFormat::Tarball.extension(), "tar.gz");
    }

    #[test]
    fn test_distribution_format_mime_type_iso() {
        assert_eq!(DistributionFormat::Iso.mime_type(), "application/x-iso9660-image");
    }

    #[test]
    fn test_distribution_format_mime_type_oci() {
        assert_eq!(DistributionFormat::Oci.mime_type(), "application/vnd.oci.image.layer.v1.tar");
    }

    #[test]
    fn test_distribution_format_mime_type_nix() {
        assert_eq!(DistributionFormat::Nix.mime_type(), "text/plain");
    }

    #[test]
    fn test_distribution_format_mime_type_flatpak() {
        assert_eq!(DistributionFormat::Flatpak.mime_type(), "application/vnd.flatpak");
    }

    #[test]
    fn test_distribution_format_mime_type_tarball() {
        assert_eq!(DistributionFormat::Tarball.mime_type(), "application/gzip");
    }

    #[test]
    fn test_distribution_format_display_iso() {
        assert_eq!(DistributionFormat::Iso.to_string(), "ISO");
    }

    #[test]
    fn test_distribution_format_display_oci() {
        assert_eq!(DistributionFormat::Oci.to_string(), "OCI");
    }

    #[test]
    fn test_distribution_format_display_nix() {
        assert_eq!(DistributionFormat::Nix.to_string(), "Nix");
    }

    #[test]
    fn test_distribution_format_display_flatpak() {
        assert_eq!(DistributionFormat::Flatpak.to_string(), "Flatpak");
    }

    #[test]
    fn test_distribution_format_display_tarball() {
        assert_eq!(DistributionFormat::Tarball.to_string(), "Tarball");
    }

    #[test]
    fn test_distribution_format_clone() {
        let fmt = DistributionFormat::Oci;
        let cloned = fmt;
        assert_eq!(fmt, cloned);
    }

    #[test]
    fn test_distribution_format_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(DistributionFormat::Iso);
        set.insert(DistributionFormat::Iso);
        assert_eq!(set.len(), 1);
        set.insert(DistributionFormat::Oci);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_distribution_format_serde() {
        let fmt = DistributionFormat::Flatpak;
        let json = serde_json::to_string(&fmt).expect("JSON serialization should succeed");
        let deserialized: DistributionFormat =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
        assert_eq!(fmt, deserialized);
    }

    #[test]
    fn test_distribution_format_debug() {
        assert_eq!(format!("{:?}", DistributionFormat::Iso), "Iso");
        assert_eq!(format!("{:?}", DistributionFormat::Oci), "Oci");
        assert_eq!(format!("{:?}", DistributionFormat::Nix), "Nix");
        assert_eq!(format!("{:?}", DistributionFormat::Flatpak), "Flatpak");
        assert_eq!(format!("{:?}", DistributionFormat::Tarball), "Tarball");
    }

    #[test]
    fn test_distribution_format_eq() {
        assert_eq!(DistributionFormat::Iso, DistributionFormat::Iso);
        assert_ne!(DistributionFormat::Iso, DistributionFormat::Oci);
    }
}
