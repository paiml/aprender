//! The projections a LoRA run adapts, and the slot each adapter takes
//! (contract `lora-target-selection-v1`, FALSIFY-LORA_TARGET_SELECTION_V1_004).

use std::fmt;

use super::LoRAConfig;
use crate::{Error, Result};

/// A linear projection of a decoder layer that a LoRA adapter can target.
///
/// The variants are in slot order: within a layer, adapters are laid out
/// q, k, v, o, gate, up, down.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LoraTarget {
    Q,
    K,
    V,
    O,
    Gate,
    Up,
    Down,
}

impl LoraTarget {
    /// Every target, in slot order.
    pub const ALL: [Self; 7] =
        [Self::Q, Self::K, Self::V, Self::O, Self::Gate, Self::Up, Self::Down];

    /// The module name HF and PEFT checkpoints use, `q_proj` … `down_proj`.
    pub fn module_name(self) -> &'static str {
        match self {
            Self::Q => "q_proj",
            Self::K => "k_proj",
            Self::V => "v_proj",
            Self::O => "o_proj",
            Self::Gate => "gate_proj",
            Self::Up => "up_proj",
            Self::Down => "down_proj",
        }
    }

    /// The target a whole module name names: `q_proj` is `Q`, `qkv_proj` is none.
    pub fn from_module_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|t| t.module_name() == name)
    }
}

impl fmt::Display for LoraTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.module_name())
    }
}

/// The targets of one run: never empty, in slot order, without repeats.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LoraTargets(Vec<LoraTarget>);

impl LoraTargets {
    /// Parse module names, or one of the shorthands `all_linear`, `attention`,
    /// `qv` and `mlp` given alone ([`LoRAConfig::expand_shorthand`]).
    ///
    /// # Errors
    /// `Error::ConfigError` naming every name that is not one of the seven
    /// projections, or for an empty list.
    pub fn parse<S: AsRef<str>>(names: &[S]) -> Result<Self> {
        let names: Vec<String> = names.iter().map(|n| n.as_ref().to_string()).collect();
        let names = LoRAConfig::expand_shorthand(&names);
        let unknown: Vec<String> = names
            .iter()
            .filter(|n| LoraTarget::from_module_name(n).is_none())
            .map(|n| format!("{n:?}"))
            .collect();
        if !unknown.is_empty() {
            return Err(Error::ConfigError(format!(
                "LoRA targets: unknown {}; use q_proj, k_proj, v_proj, o_proj, gate_proj, \
                 up_proj or down_proj, or one of all_linear, attention, qv, mlp alone",
                unknown.join(", ")
            )));
        }
        let mut targets: Vec<LoraTarget> =
            names.iter().filter_map(|n| LoraTarget::from_module_name(n)).collect();
        targets.sort_unstable();
        targets.dedup();
        if targets.is_empty() {
            return Err(Error::ConfigError("LoRA targets: the list is empty".into()));
        }
        Ok(Self(targets))
    }

    /// The targets, in slot order.
    pub fn as_slice(&self) -> &[LoraTarget] {
        &self.0
    }

    /// How many adapters each layer gets.
    pub fn per_layer(&self) -> usize {
        self.0.len()
    }

    /// Whether `target` is selected.
    pub fn contains(&self, target: LoraTarget) -> bool {
        self.0.contains(&target)
    }

    /// The slot of the adapter for `target` in `layer`, `per_layer() · layer +`
    /// the position of `target`, or none if `target` is not selected.
    pub fn slot(&self, layer: usize, target: LoraTarget) -> Option<usize> {
        let position = self.0.iter().position(|&t| t == target)?;
        Some(layer * self.0.len() + position)
    }
}

impl Default for LoraTargets {
    /// `q_proj` and `v_proj`, the targets of the LoRA paper.
    fn default() -> Self {
        Self(vec![LoraTarget::Q, LoraTarget::V])
    }
}

impl fmt::Display for LoraTargets {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<&str> = self.0.iter().map(|t| t.module_name()).collect();
        f.write_str(&names.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::LoraTarget::{Down, Gate, Up, K, O, Q, V};
    use super::*;

    #[test]
    fn falsify_lora_target_selection_v1_004_parse_sorts_dedups_and_expands() {
        let cases: [(&[&str], &[LoraTarget]); 7] = [
            (&["v_proj", "q_proj", "v_proj"], &[Q, V]),
            (&["qv"], &[Q, V]),
            (&["attention"], &[Q, K, V, O]),
            (&["mlp"], &[Gate, Up, Down]),
            (&["all_linear"], &LoraTarget::ALL),
            (&["down_proj", "k_proj"], &[K, Down]),
            (&["q_proj"], &[Q]),
        ];
        for (names, want) in cases {
            let targets = LoraTargets::parse(names).expect("known names must parse");
            assert_eq!(targets.as_slice(), want, "{names:?}");
        }
        assert_eq!(LoraTargets::default().as_slice(), [Q, V]);
        assert_eq!(LoraTargets::default().to_string(), "q_proj, v_proj");
    }

    #[test]
    fn falsify_lora_target_selection_v1_004_parse_refuses_unknown_names_by_name() {
        // A shorthand expands only alone; an empty name is a trailing comma.
        let cases: [&[&str]; 4] = [
            &["q_proj", "nonexistent_proj"],
            &["qkv_proj"],
            &["all_linear", "q_proj"],
            &["v_proj", ""],
        ];
        for names in cases {
            let err = LoraTargets::parse(names).expect_err("an unknown name must refuse");
            for bad in names.iter().filter(|n| LoraTarget::from_module_name(n).is_none()) {
                assert!(err.to_string().contains(&format!("{bad:?}")), "names {bad:?}: {err}");
            }
        }
        let err = LoraTargets::parse::<&str>(&[]).expect_err("no targets must refuse");
        assert!(err.to_string().contains("empty"), "{err}");
    }

    #[test]
    fn falsify_lora_target_selection_v1_004_slot_is_layer_times_count_plus_position() {
        // The default map is the one the forward, the CUDA blocks and their
        // sync hard-code: Q at 2·layer, V at 2·layer + 1.
        let default = LoraTargets::default();
        let all = LoraTargets::parse(&["all_linear"]).expect("all_linear must parse");
        for layer in 0..4 {
            assert_eq!(default.slot(layer, Q), Some(2 * layer));
            assert_eq!(default.slot(layer, V), Some(2 * layer + 1));
            for target in [K, O, Gate, Up, Down] {
                assert_eq!(default.slot(layer, target), None, "{target}");
            }
            for (position, target) in LoraTarget::ALL.into_iter().enumerate() {
                assert_eq!(all.slot(layer, target), Some(7 * layer + position), "{target}");
            }
        }
        let mlp = LoraTargets::parse(&["mlp"]).expect("mlp must parse");
        assert_eq!(mlp.slot(2, Down), Some(3 * 2 + 2));
        assert_eq!(mlp.slot(2, Q), None);
    }
}
