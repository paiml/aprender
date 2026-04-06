//! Shell type for CLI completions.

/// Shell type for completions
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ShellType {
    #[default]
    Bash,
    Zsh,
    Fish,
    PowerShell,
}

impl std::str::FromStr for ShellType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "bash" => Ok(ShellType::Bash),
            "zsh" => Ok(ShellType::Zsh),
            "fish" => Ok(ShellType::Fish),
            "powershell" | "ps" => Ok(ShellType::PowerShell),
            _ => Err(format!("Unknown shell: {s}. Valid shells: bash, zsh, fish, powershell")),
        }
    }
}

impl std::fmt::Display for ShellType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShellType::Bash => write!(f, "bash"),
            ShellType::Zsh => write!(f, "zsh"),
            ShellType::Fish => write!(f, "fish"),
            ShellType::PowerShell => write!(f, "powershell"),
        }
    }
}
