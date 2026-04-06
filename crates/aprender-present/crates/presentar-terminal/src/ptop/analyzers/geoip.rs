//! Embedded Geo-IP lookup - Sovereign AI, no external dependencies (GAP-CONN-002).
//!
//! Lightweight IP-to-country mapping using hardcoded ranges for:
//! - Major cloud providers (AWS, GCP, Azure, DO, etc.)
//! - CDNs (Cloudflare, Akamai, Fastly)
//! - Well-known services (Google, Facebook, Apple, Microsoft)
//! - Private/local ranges
//! - Major country allocations (rough approximations)
//!
//! Parity with trueno-viz/crates/ttop/src/analyzers/geoip.rs

use std::net::Ipv4Addr;

/// Country info with flag emoji
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CountryInfo {
    pub code: &'static str,
    pub flag: &'static str,
    pub name: &'static str,
}

impl CountryInfo {
    const fn new(code: &'static str, flag: &'static str, name: &'static str) -> Self {
        Self { code, flag, name }
    }
}

// Common countries
const US: CountryInfo = CountryInfo::new("US", "🇺🇸", "United States");
const DE: CountryInfo = CountryInfo::new("DE", "🇩🇪", "Germany");
const GB: CountryInfo = CountryInfo::new("GB", "🇬🇧", "United Kingdom");
const FR: CountryInfo = CountryInfo::new("FR", "🇫🇷", "France");
const NL: CountryInfo = CountryInfo::new("NL", "🇳🇱", "Netherlands");
const JP: CountryInfo = CountryInfo::new("JP", "🇯🇵", "Japan");
const SG: CountryInfo = CountryInfo::new("SG", "🇸🇬", "Singapore");
const AU: CountryInfo = CountryInfo::new("AU", "🇦🇺", "Australia");
const BR: CountryInfo = CountryInfo::new("BR", "🇧🇷", "Brazil");
const IN: CountryInfo = CountryInfo::new("IN", "🇮🇳", "India");
const CN: CountryInfo = CountryInfo::new("CN", "🇨🇳", "China");
const RU: CountryInfo = CountryInfo::new("RU", "🇷🇺", "Russia");
const KR: CountryInfo = CountryInfo::new("KR", "🇰🇷", "South Korea");
const IE: CountryInfo = CountryInfo::new("IE", "🇮🇪", "Ireland");

// Special designations
const LOCAL: CountryInfo = CountryInfo::new("LO", "🏠", "Local");
const PRIVATE: CountryInfo = CountryInfo::new("PR", "🔒", "Private");

/// IP range with associated country
struct IpRange {
    start: u32,
    end: u32,
    country: CountryInfo,
}

impl IpRange {
    const fn new(start: u32, end: u32, country: CountryInfo) -> Self {
        Self {
            start,
            end,
            country,
        }
    }

    fn contains(&self, ip: u32) -> bool {
        ip >= self.start && ip <= self.end
    }
}

/// Convert IP to u32 for range comparison
fn ip_to_u32(ip: Ipv4Addr) -> u32 {
    let octets = ip.octets();
    ((octets[0] as u32) << 24)
        | ((octets[1] as u32) << 16)
        | ((octets[2] as u32) << 8)
        | (octets[3] as u32)
}

/// Convert CIDR notation to range
const fn cidr_to_range(a: u8, b: u8, c: u8, d: u8, prefix: u8) -> (u32, u32) {
    let ip = ((a as u32) << 24) | ((b as u32) << 16) | ((c as u32) << 8) | (d as u32);
    let mask = if prefix == 0 {
        0
    } else {
        !0u32 << (32 - prefix)
    };
    let start = ip & mask;
    let end = start | !mask;
    (start, end)
}

// Macro for cleaner range definitions
macro_rules! range {
    ($a:expr, $b:expr, $c:expr, $d:expr, $prefix:expr, $country:expr) => {{
        let (start, end) = cidr_to_range($a, $b, $c, $d, $prefix);
        IpRange::new(start, end, $country)
    }};
}

/// Get embedded IP ranges
/// Ordered by specificity (more specific ranges first)
fn get_ranges() -> Vec<IpRange> {
    vec![
        // === Private/Local Ranges ===
        range!(127, 0, 0, 0, 8, LOCAL),      // Loopback
        range!(10, 0, 0, 0, 8, PRIVATE),     // Private Class A
        range!(172, 16, 0, 0, 12, PRIVATE),  // Private Class B
        range!(192, 168, 0, 0, 16, PRIVATE), // Private Class C
        range!(169, 254, 0, 0, 16, LOCAL),   // Link-local
        // === Major Cloud Providers ===
        // Cloudflare (anycast, but HQ in US)
        range!(104, 16, 0, 0, 13, US),    // Cloudflare
        range!(104, 24, 0, 0, 14, US),    // Cloudflare
        range!(172, 64, 0, 0, 13, US),    // Cloudflare
        range!(173, 245, 48, 0, 20, US),  // Cloudflare
        range!(141, 101, 64, 0, 18, US),  // Cloudflare
        range!(108, 162, 192, 0, 18, US), // Cloudflare
        range!(162, 158, 0, 0, 15, US),   // Cloudflare
        // Google (US-based, global anycast)
        range!(8, 8, 8, 0, 24, US),      // Google DNS
        range!(8, 8, 4, 0, 24, US),      // Google DNS
        range!(34, 64, 0, 0, 10, US),    // Google Cloud
        range!(34, 128, 0, 0, 10, US),   // Google Cloud
        range!(35, 184, 0, 0, 13, US),   // Google Cloud
        range!(35, 192, 0, 0, 12, US),   // Google Cloud
        range!(142, 250, 0, 0, 15, US),  // Google
        range!(172, 217, 0, 0, 16, US),  // Google
        range!(216, 58, 192, 0, 19, US), // Google
        // Amazon AWS - Regional ranges (specific before general)
        range!(3, 248, 0, 0, 13, IE),  // AWS eu-west-1 (Ireland)
        range!(52, 16, 0, 0, 12, IE),  // AWS eu-west-1
        range!(54, 72, 0, 0, 13, IE),  // AWS eu-west-1
        range!(3, 64, 0, 0, 10, DE),   // AWS eu-central-1 (Frankfurt)
        range!(52, 28, 0, 0, 14, DE),  // AWS eu-central-1
        range!(13, 112, 0, 0, 12, JP), // AWS ap-northeast-1 (Tokyo)
        range!(52, 68, 0, 0, 14, JP),  // AWS ap-northeast-1
        range!(13, 228, 0, 0, 14, SG), // AWS ap-southeast-1 (Singapore)
        range!(52, 74, 0, 0, 15, SG),  // AWS ap-southeast-1
        range!(13, 236, 0, 0, 14, AU), // AWS ap-southeast-2 (Sydney)
        range!(52, 62, 0, 0, 15, AU),  // AWS ap-southeast-2
        // AWS US (general fallback)
        range!(3, 0, 0, 0, 8, US),    // AWS (mostly US)
        range!(18, 128, 0, 0, 9, US), // AWS US
        range!(52, 0, 0, 0, 8, US),   // AWS (global fallback)
        range!(54, 64, 0, 0, 10, US), // AWS US
        // Microsoft Azure
        range!(13, 64, 0, 0, 10, US),  // Azure US
        range!(20, 0, 0, 0, 8, US),    // Azure (global, mostly US)
        range!(40, 64, 0, 0, 10, US),  // Azure US
        range!(104, 40, 0, 0, 13, US), // Azure US
        // DigitalOcean
        range!(45, 55, 0, 0, 16, US),   // DigitalOcean
        range!(104, 131, 0, 0, 16, US), // DigitalOcean
        range!(138, 197, 0, 0, 16, US), // DigitalOcean
        range!(159, 65, 0, 0, 16, US),  // DigitalOcean
        // Hetzner (Germany)
        range!(5, 9, 0, 0, 16, DE),     // Hetzner
        range!(78, 46, 0, 0, 15, DE),   // Hetzner
        range!(88, 198, 0, 0, 15, DE),  // Hetzner
        range!(136, 243, 0, 0, 16, DE), // Hetzner
        range!(148, 251, 0, 0, 16, DE), // Hetzner
        range!(176, 9, 0, 0, 16, DE),   // Hetzner
        // OVH (France)
        range!(5, 39, 0, 0, 16, FR),   // OVH
        range!(51, 68, 0, 0, 14, FR),  // OVH
        range!(51, 77, 0, 0, 16, FR),  // OVH
        range!(51, 91, 0, 0, 16, FR),  // OVH
        range!(91, 121, 0, 0, 16, FR), // OVH
        range!(137, 74, 0, 0, 15, FR), // OVH
        // Linode
        range!(45, 33, 0, 0, 16, US),   // Linode
        range!(139, 162, 0, 0, 15, US), // Linode
        range!(172, 104, 0, 0, 13, US), // Linode
        // === Major Services ===
        // Apple
        range!(17, 0, 0, 0, 8, US), // Apple (entire /8)
        // Microsoft (non-Azure)
        range!(40, 76, 0, 0, 14, US),   // Microsoft
        range!(65, 52, 0, 0, 14, US),   // Microsoft
        range!(131, 107, 0, 0, 16, US), // Microsoft
        // GitHub (Microsoft)
        range!(140, 82, 112, 0, 20, US),  // GitHub
        range!(143, 55, 64, 0, 20, US),   // GitHub
        range!(185, 199, 108, 0, 22, US), // GitHub
        // Akamai
        range!(23, 0, 0, 0, 11, US),   // Akamai
        range!(23, 32, 0, 0, 11, US),  // Akamai
        range!(104, 64, 0, 0, 10, US), // Akamai
        // Fastly
        range!(151, 101, 0, 0, 16, US), // Fastly
        range!(199, 232, 0, 0, 16, US), // Fastly
        // Netflix
        range!(45, 57, 0, 0, 16, US),    // Netflix
        range!(108, 175, 32, 0, 19, US), // Netflix
        // Discord
        range!(162, 159, 128, 0, 17, US), // Discord
        // === Major Country Allocations (rough) ===
        // China (APNIC)
        range!(1, 0, 0, 0, 8, CN),
        range!(14, 0, 0, 0, 8, CN),
        range!(27, 0, 0, 0, 8, CN),
        range!(36, 0, 0, 0, 8, CN),
        range!(58, 0, 0, 0, 8, CN),
        range!(59, 0, 0, 0, 8, CN),
        range!(60, 0, 0, 0, 8, CN),
        range!(61, 0, 0, 0, 8, CN),
        range!(110, 0, 0, 0, 8, CN),
        range!(111, 0, 0, 0, 8, CN),
        range!(112, 0, 0, 0, 8, CN),
        range!(113, 0, 0, 0, 8, CN),
        range!(114, 0, 0, 0, 8, CN),
        range!(115, 0, 0, 0, 8, CN),
        range!(116, 0, 0, 0, 8, CN),
        range!(117, 0, 0, 0, 8, CN),
        range!(118, 0, 0, 0, 8, CN),
        range!(119, 0, 0, 0, 8, CN),
        range!(120, 0, 0, 0, 8, CN),
        range!(121, 0, 0, 0, 8, CN),
        range!(122, 0, 0, 0, 8, CN),
        range!(123, 0, 0, 0, 8, CN),
        range!(124, 0, 0, 0, 8, CN),
        range!(125, 0, 0, 0, 8, CN),
        range!(180, 0, 0, 0, 8, CN),
        range!(182, 0, 0, 0, 8, CN),
        range!(183, 0, 0, 0, 8, CN),
        range!(218, 0, 0, 0, 8, CN),
        range!(219, 0, 0, 0, 8, CN),
        range!(220, 0, 0, 0, 8, CN),
        range!(221, 0, 0, 0, 8, CN),
        range!(222, 0, 0, 0, 8, CN),
        range!(223, 0, 0, 0, 8, CN),
        // Russia
        range!(5, 8, 0, 0, 13, RU),
        range!(31, 40, 0, 0, 13, RU),
        range!(46, 138, 0, 0, 15, RU),
        range!(77, 72, 0, 0, 13, RU),
        range!(78, 24, 0, 0, 13, RU),
        range!(81, 16, 0, 0, 12, RU),
        range!(85, 192, 0, 0, 11, RU),
        range!(87, 224, 0, 0, 12, RU),
        range!(89, 208, 0, 0, 12, RU),
        range!(93, 80, 0, 0, 12, RU),
        // India
        range!(14, 139, 0, 0, 16, IN),
        range!(14, 192, 0, 0, 11, IN),
        range!(27, 48, 0, 0, 12, IN),
        range!(43, 224, 0, 0, 12, IN),
        range!(45, 64, 0, 0, 12, IN),
        range!(45, 112, 0, 0, 12, IN),
        range!(49, 32, 0, 0, 11, IN),
        range!(59, 88, 0, 0, 13, IN),
        range!(61, 0, 0, 0, 10, IN),
        range!(103, 0, 0, 0, 10, IN),
        range!(106, 0, 0, 0, 10, IN),
        range!(117, 192, 0, 0, 10, IN),
        range!(182, 64, 0, 0, 10, IN),
        // Brazil
        range!(131, 0, 0, 0, 10, BR),
        range!(177, 0, 0, 0, 8, BR),
        range!(179, 0, 0, 0, 8, BR),
        range!(187, 0, 0, 0, 8, BR),
        range!(189, 0, 0, 0, 8, BR),
        range!(191, 0, 0, 0, 9, BR),
        range!(200, 0, 0, 0, 9, BR),
        range!(201, 0, 0, 0, 8, BR),
        // South Korea
        range!(1, 208, 0, 0, 12, KR),
        range!(14, 32, 0, 0, 11, KR),
        range!(27, 96, 0, 0, 11, KR),
        range!(58, 72, 0, 0, 13, KR),
        range!(59, 0, 0, 0, 10, KR),
        range!(112, 160, 0, 0, 11, KR),
        range!(118, 32, 0, 0, 11, KR),
        range!(121, 128, 0, 0, 10, KR),
        range!(175, 192, 0, 0, 10, KR),
        // Japan
        range!(42, 96, 0, 0, 11, JP),
        range!(49, 212, 0, 0, 14, JP),
        range!(59, 128, 0, 0, 10, JP),
        range!(101, 128, 0, 0, 9, JP),
        range!(110, 64, 0, 0, 10, JP),
        range!(126, 0, 0, 0, 9, JP),
        range!(133, 0, 0, 0, 11, JP),
        range!(150, 0, 0, 0, 10, JP),
        range!(153, 0, 0, 0, 10, JP),
        range!(157, 0, 0, 0, 10, JP),
        range!(175, 0, 0, 0, 11, JP),
        range!(202, 0, 0, 0, 11, JP),
        range!(210, 128, 0, 0, 10, JP),
        range!(211, 0, 0, 0, 10, JP),
        // UK
        range!(2, 16, 0, 0, 14, GB),
        range!(5, 64, 0, 0, 11, GB),
        range!(31, 48, 0, 0, 12, GB),
        range!(37, 128, 0, 0, 12, GB),
        range!(46, 32, 0, 0, 11, GB),
        range!(51, 36, 0, 0, 14, GB),
        range!(77, 96, 0, 0, 12, GB),
        range!(78, 128, 0, 0, 10, GB),
        range!(79, 64, 0, 0, 10, GB),
        range!(80, 0, 0, 0, 11, GB),
        range!(81, 128, 0, 0, 10, GB),
        range!(82, 0, 0, 0, 11, GB),
        range!(86, 0, 0, 0, 11, GB),
        range!(193, 0, 0, 0, 10, GB),
        range!(194, 0, 0, 0, 10, GB),
        // Germany
        range!(2, 200, 0, 0, 13, DE),
        range!(31, 0, 0, 0, 12, DE),
        range!(37, 0, 0, 0, 12, DE),
        range!(46, 0, 0, 0, 12, DE),
        range!(62, 0, 0, 0, 12, DE),
        range!(77, 0, 0, 0, 11, DE),
        range!(78, 0, 0, 0, 11, DE),
        range!(79, 192, 0, 0, 10, DE),
        range!(80, 128, 0, 0, 10, DE),
        range!(83, 0, 0, 0, 11, DE),
        range!(84, 128, 0, 0, 10, DE),
        range!(85, 128, 0, 0, 10, DE),
        range!(217, 0, 0, 0, 10, DE),
        // France
        range!(2, 0, 0, 0, 11, FR),
        range!(5, 32, 0, 0, 11, FR),
        range!(46, 192, 0, 0, 10, FR),
        range!(62, 192, 0, 0, 10, FR),
        range!(77, 128, 0, 0, 10, FR),
        range!(80, 64, 0, 0, 10, FR),
        range!(81, 0, 0, 0, 11, FR),
        range!(82, 192, 0, 0, 10, FR),
        range!(83, 128, 0, 0, 10, FR),
        range!(84, 0, 0, 0, 11, FR),
        range!(85, 64, 0, 0, 10, FR),
        range!(87, 0, 0, 0, 11, FR),
        range!(88, 0, 0, 0, 11, FR),
        range!(89, 64, 0, 0, 10, FR),
        range!(90, 0, 0, 0, 11, FR),
        // Netherlands
        range!(2, 56, 0, 0, 13, NL),
        range!(31, 160, 0, 0, 11, NL),
        range!(37, 32, 0, 0, 11, NL),
        range!(46, 64, 0, 0, 10, NL),
        range!(51, 0, 0, 0, 11, NL),
        range!(62, 64, 0, 0, 10, NL),
        range!(77, 160, 0, 0, 11, NL),
        range!(78, 64, 0, 0, 10, NL),
        range!(79, 128, 0, 0, 10, NL),
        range!(84, 192, 0, 0, 10, NL),
        range!(87, 192, 0, 0, 10, NL),
        range!(88, 192, 0, 0, 10, NL),
        range!(89, 128, 0, 0, 10, NL),
        range!(94, 64, 0, 0, 10, NL),
    ]
}

/// Lookup country for an IPv4 address
#[must_use]
pub fn lookup(ip: Ipv4Addr) -> Option<CountryInfo> {
    let ip_u32 = ip_to_u32(ip);
    let ranges = get_ranges();

    // Check ranges (more specific ones are listed first, so first match wins)
    for range in ranges {
        if range.contains(ip_u32) {
            return Some(range.country);
        }
    }

    None
}

/// Get flag emoji for an IP, or "🌐" for unknown
#[must_use]
pub fn get_flag(ip: Ipv4Addr) -> &'static str {
    lookup(ip).map(|c| c.flag).unwrap_or("🌐")
}

/// Get country code for an IP, or "??" for unknown
#[must_use]
pub fn get_country_code(ip: Ipv4Addr) -> &'static str {
    lookup(ip).map(|c| c.code).unwrap_or("??")
}

/// Get country name for an IP, or "Unknown" for unknown
#[must_use]
pub fn get_country_name(ip: Ipv4Addr) -> &'static str {
    lookup(ip).map(|c| c.name).unwrap_or("Unknown")
}

/// Format location string for display (flag + code)
#[must_use]
pub fn format_location(ip: Ipv4Addr) -> String {
    let flag = get_flag(ip);
    let code = get_country_code(ip);
    format!("{} {}", flag, code)
}

/// Parse IP string and lookup location
#[must_use]
pub fn lookup_str(ip_str: &str) -> Option<CountryInfo> {
    ip_str.parse::<Ipv4Addr>().ok().and_then(lookup)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_localhost() {
        let ip: Ipv4Addr = "127.0.0.1".parse().unwrap();
        let info = lookup(ip).unwrap();
        assert_eq!(info.code, "LO");
        assert_eq!(info.flag, "🏠");
        assert_eq!(info.name, "Local");
    }

    #[test]
    fn test_private_10() {
        let ip: Ipv4Addr = "10.0.0.1".parse().unwrap();
        let info = lookup(ip).unwrap();
        assert_eq!(info.code, "PR");
        assert_eq!(info.flag, "🔒");
    }

    #[test]
    fn test_private_172() {
        let ip: Ipv4Addr = "172.16.1.1".parse().unwrap();
        let info = lookup(ip).unwrap();
        assert_eq!(info.code, "PR");
    }

    #[test]
    fn test_private_192() {
        let ip: Ipv4Addr = "192.168.1.1".parse().unwrap();
        let info = lookup(ip).unwrap();
        assert_eq!(info.code, "PR");
    }

    #[test]
    fn test_link_local() {
        let ip: Ipv4Addr = "169.254.1.1".parse().unwrap();
        let info = lookup(ip).unwrap();
        assert_eq!(info.code, "LO");
    }

    #[test]
    fn test_google_dns() {
        let ip: Ipv4Addr = "8.8.8.8".parse().unwrap();
        let info = lookup(ip).unwrap();
        assert_eq!(info.code, "US");
        assert_eq!(info.flag, "🇺🇸");
    }

    #[test]
    fn test_cloudflare_104() {
        let ip: Ipv4Addr = "104.16.0.1".parse().unwrap();
        let info = lookup(ip).unwrap();
        assert_eq!(info.code, "US");
    }

    #[test]
    fn test_unknown_ip() {
        let ip: Ipv4Addr = "224.0.0.1".parse().unwrap(); // Multicast
        let flag = get_flag(ip);
        assert_eq!(flag, "🌐");
    }

    #[test]
    fn test_flag_helper() {
        let ip: Ipv4Addr = "8.8.8.8".parse().unwrap();
        assert_eq!(get_flag(ip), "🇺🇸");
    }

    #[test]
    fn test_country_code_helper() {
        let ip: Ipv4Addr = "8.8.8.8".parse().unwrap();
        assert_eq!(get_country_code(ip), "US");
    }

    #[test]
    fn test_country_name_helper() {
        let ip: Ipv4Addr = "8.8.8.8".parse().unwrap();
        assert_eq!(get_country_name(ip), "United States");
    }

    #[test]
    fn test_hetzner_germany() {
        let ip: Ipv4Addr = "148.251.1.1".parse().unwrap();
        let info = lookup(ip).unwrap();
        assert_eq!(info.code, "DE");
        assert_eq!(info.flag, "🇩🇪");
    }

    #[test]
    fn test_ovh_france() {
        let ip: Ipv4Addr = "51.77.1.1".parse().unwrap();
        let info = lookup(ip).unwrap();
        assert_eq!(info.code, "FR");
        assert_eq!(info.flag, "🇫🇷");
    }

    #[test]
    fn test_apple_17() {
        let ip: Ipv4Addr = "17.0.0.1".parse().unwrap();
        let info = lookup(ip).unwrap();
        assert_eq!(info.code, "US");
    }

    #[test]
    fn test_aws_ireland() {
        let ip: Ipv4Addr = "52.16.1.1".parse().unwrap();
        let info = lookup(ip).unwrap();
        assert_eq!(info.code, "IE");
        assert_eq!(info.flag, "🇮🇪");
    }

    #[test]
    fn test_format_location() {
        let ip: Ipv4Addr = "8.8.8.8".parse().unwrap();
        let loc = format_location(ip);
        assert!(loc.contains("🇺🇸"));
        assert!(loc.contains("US"));
    }

    #[test]
    fn test_format_location_unknown() {
        let ip: Ipv4Addr = "224.0.0.1".parse().unwrap();
        let loc = format_location(ip);
        assert!(loc.contains("🌐"));
        assert!(loc.contains("??"));
    }

    #[test]
    fn test_lookup_str_valid() {
        let info = lookup_str("8.8.8.8").unwrap();
        assert_eq!(info.code, "US");
    }

    #[test]
    fn test_lookup_str_invalid() {
        assert!(lookup_str("not-an-ip").is_none());
    }

    #[test]
    fn test_lookup_str_localhost() {
        let info = lookup_str("127.0.0.1").unwrap();
        assert_eq!(info.code, "LO");
    }

    #[test]
    fn test_country_info_debug() {
        let info = CountryInfo::new("US", "🇺🇸", "United States");
        let debug = format!("{:?}", info);
        assert!(debug.contains("US"));
    }

    #[test]
    fn test_country_info_clone() {
        let info = CountryInfo::new("US", "🇺🇸", "United States");
        let cloned = info;
        assert_eq!(cloned.code, info.code);
    }

    #[test]
    fn test_country_info_eq() {
        let info1 = CountryInfo::new("US", "🇺🇸", "United States");
        let info2 = CountryInfo::new("US", "🇺🇸", "United States");
        assert_eq!(info1, info2);
    }

    #[test]
    fn test_digital_ocean() {
        let ip: Ipv4Addr = "104.131.1.1".parse().unwrap();
        let info = lookup(ip).unwrap();
        assert_eq!(info.code, "US");
    }

    #[test]
    fn test_linode() {
        let ip: Ipv4Addr = "139.162.1.1".parse().unwrap();
        let info = lookup(ip).unwrap();
        assert_eq!(info.code, "US");
    }

    #[test]
    fn test_github() {
        let ip: Ipv4Addr = "140.82.112.1".parse().unwrap();
        let info = lookup(ip).unwrap();
        assert_eq!(info.code, "US");
    }

    #[test]
    fn test_fastly() {
        let ip: Ipv4Addr = "151.101.1.1".parse().unwrap();
        let info = lookup(ip).unwrap();
        assert_eq!(info.code, "US");
    }

    #[test]
    fn test_discord() {
        let ip: Ipv4Addr = "162.159.128.1".parse().unwrap();
        let info = lookup(ip).unwrap();
        assert_eq!(info.code, "US");
    }
}
