/// CPU governor types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuGovernor {
    /// Performance governor (max frequency)
    Performance,
    /// Powersave governor (min frequency)
    Powersave,
    /// Ondemand governor (dynamic)
    Ondemand,
    /// Conservative governor (gradual)
    Conservative,
    /// Schedutil governor (scheduler-based)
    Schedutil,
    /// Userspace governor (manual)
    Userspace,
    /// Unknown governor
    Unknown,
}

/// Governor spec: (variant, string name).
const GOVERNOR_SPECS: &[(CpuGovernor, &str)] = &[
    (CpuGovernor::Performance, "performance"),
    (CpuGovernor::Powersave, "powersave"),
    (CpuGovernor::Ondemand, "ondemand"),
    (CpuGovernor::Conservative, "conservative"),
    (CpuGovernor::Schedutil, "schedutil"),
    (CpuGovernor::Userspace, "userspace"),
];

impl CpuGovernor {
    /// Get governor name
    pub fn name(&self) -> &'static str {
        GOVERNOR_SPECS.iter().find(|(v, _)| v == self).map(|(_, n)| *n).unwrap_or("unknown")
    }

    /// Parse from string
    pub fn parse(s: &str) -> Self {
        let lower = s.trim().to_lowercase();
        GOVERNOR_SPECS.iter().find(|(_, n)| *n == lower).map(|(v, _)| *v).unwrap_or(Self::Unknown)
    }

    /// Check if deterministic (fixed frequency)
    pub fn is_deterministic(&self) -> bool {
        matches!(self, Self::Performance | Self::Powersave | Self::Userspace)
    }
}
