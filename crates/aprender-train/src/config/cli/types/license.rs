//! License type for CLI commands.

/// License for CLI
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum LicenseArg {
    #[default]
    CcBy4,
    CcBySa4,
    Cc0,
    Mit,
    Apache2,
    Gpl3,
    Bsd3,
}

impl std::str::FromStr for LicenseArg {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().replace(['-', '.'], "").as_str() {
            "ccby4" | "ccby40" => Ok(LicenseArg::CcBy4),
            "ccbysa4" | "ccbysa40" => Ok(LicenseArg::CcBySa4),
            "cc0" => Ok(LicenseArg::Cc0),
            "mit" => Ok(LicenseArg::Mit),
            "apache2" | "apache20" => Ok(LicenseArg::Apache2),
            "gpl3" | "gplv3" => Ok(LicenseArg::Gpl3),
            "bsd3" | "bsd3clause" => Ok(LicenseArg::Bsd3),
            _ => Err(format!(
                "Unknown license: {s}. Valid licenses: CC-BY-4.0, CC-BY-SA-4.0, CC0, MIT, Apache-2.0, GPL-3.0, BSD-3-Clause"
            )),
        }
    }
}

impl std::fmt::Display for LicenseArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LicenseArg::CcBy4 => write!(f, "CC-BY-4.0"),
            LicenseArg::CcBySa4 => write!(f, "CC-BY-SA-4.0"),
            LicenseArg::Cc0 => write!(f, "CC0"),
            LicenseArg::Mit => write!(f, "MIT"),
            LicenseArg::Apache2 => write!(f, "Apache-2.0"),
            LicenseArg::Gpl3 => write!(f, "GPL-3.0"),
            LicenseArg::Bsd3 => write!(f, "BSD-3-Clause"),
        }
    }
}
