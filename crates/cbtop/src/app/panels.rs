//! Active panel definitions for the cbtop TUI.

/// Active panel in the UI
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ActivePanel {
    #[default]
    Overview,
    Cpu,
    Gpu,
    Pcie,
    Memory,
    Thermal,
    Load,
    Config,
    Help,
}

impl ActivePanel {
    /// Get panel from key (1-9)
    pub fn from_key(key: char) -> Option<Self> {
        match key {
            '1' => Some(Self::Overview),
            '2' => Some(Self::Cpu),
            '3' => Some(Self::Gpu),
            '4' => Some(Self::Pcie),
            '5' => Some(Self::Memory),
            '6' => Some(Self::Thermal),
            '7' => Some(Self::Load),
            '8' => Some(Self::Config),
            '9' => Some(Self::Help),
            _ => None,
        }
    }

    /// Panel title
    pub fn title(&self) -> &'static str {
        match self {
            Self::Overview => "Overview",
            Self::Cpu => "CPU",
            Self::Gpu => "GPU",
            Self::Pcie => "PCIe",
            Self::Memory => "Memory",
            Self::Thermal => "Thermal",
            Self::Load => "Load",
            Self::Config => "Config",
            Self::Help => "Help",
        }
    }

    /// All panels for tab bar rendering (UI-10)
    pub fn all() -> &'static [Self] {
        &[
            Self::Overview,
            Self::Cpu,
            Self::Gpu,
            Self::Pcie,
            Self::Memory,
            Self::Thermal,
            Self::Load,
            Self::Config,
            Self::Help,
        ]
    }

    /// Key number for this panel (1-9)
    pub fn key_number(&self) -> char {
        match self {
            Self::Overview => '1',
            Self::Cpu => '2',
            Self::Gpu => '3',
            Self::Pcie => '4',
            Self::Memory => '5',
            Self::Thermal => '6',
            Self::Load => '7',
            Self::Config => '8',
            Self::Help => '9',
        }
    }
}
