//! Capability-based access control for agent tools.
//!
//! Implements the Poka-Yoke (mistake-proofing) pattern from Toyota
//! Production System. Each tool declares its required capability;
//! the agent manifest grants capabilities. Mismatch → denied.
//!
//! See: arXiv:2406.09187 (`GuardAgent`), arXiv:2509.22256 (access control).

use serde::{Deserialize, Serialize};

/// Capability grants for tools (Poka-Yoke pattern).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Capability {
    /// Access RAG pipeline for document retrieval.
    Rag,
    /// Read/write agent memory.
    Memory,
    /// Execute shell commands (sandboxed).
    Shell {
        /// Allowed command prefixes. `["*"]` allows all.
        allowed_commands: Vec<String>,
    },
    /// Launch headless browser via jugar-probar.
    Browser,
    /// Invoke sub-inference on a different model.
    Inference,
    /// Read files from the filesystem.
    FileRead {
        /// Allowed path prefixes. `["*"]` allows all.
        allowed_paths: Vec<String>,
    },
    /// Write or edit files on the filesystem.
    FileWrite {
        /// Allowed path prefixes. `["*"]` allows all.
        allowed_paths: Vec<String>,
    },
    /// Submit work to repartir compute pool.
    Compute,
    /// Network egress (blocked in Sovereign tier).
    Network {
        /// Allowed hostnames. `["*"]` allows all.
        allowed_hosts: Vec<String>,
    },
    /// MCP tool from external server (agents-mcp feature).
    Mcp {
        /// MCP server name.
        server: String,
        /// Tool name on that server. `"*"` matches all.
        tool: String,
    },
    /// Spawn sub-agents (bounded by `max_depth`).
    Spawn {
        /// Maximum recursion depth (0 = no sub-spawning).
        max_depth: u32,
    },
}

/// Check if granted capabilities satisfy a required capability.
///
/// Returns `true` if at least one granted capability matches the
/// required capability. Wildcard (`"*"`) matching is supported
/// for Shell commands, Network hosts, and MCP tools.
#[cfg_attr(
    feature = "agents-contracts",
    provable_contracts_macros::contract("agent-loop-v1", equation = "capability_match")
)]
pub fn capability_matches(granted: &[Capability], required: &Capability) -> bool {
    granted.iter().any(|g| single_match(g, required))
}

fn single_match(granted: &Capability, required: &Capability) -> bool {
    match (granted, required) {
        (Capability::Rag, Capability::Rag)
        | (Capability::Memory, Capability::Memory)
        | (Capability::Browser, Capability::Browser)
        | (Capability::Inference, Capability::Inference)
        | (Capability::Compute, Capability::Compute) => true,

        (Capability::FileRead { allowed_paths: g }, Capability::FileRead { allowed_paths: r }) => {
            r.iter().all(|p| g.contains(p) || g.iter().any(|gp| gp == "*"))
        }

        (
            Capability::FileWrite { allowed_paths: g },
            Capability::FileWrite { allowed_paths: r },
        ) => r.iter().all(|p| g.contains(p) || g.iter().any(|gp| gp == "*")),

        (Capability::Spawn { max_depth: g }, Capability::Spawn { max_depth: r }) => g >= r,

        (Capability::Shell { allowed_commands: g }, Capability::Shell { allowed_commands: r }) => {
            r.iter().all(|cmd| g.contains(cmd) || g.iter().any(|p| p == "*"))
        }

        (Capability::Network { allowed_hosts: g }, Capability::Network { allowed_hosts: r }) => {
            r.iter().all(|h| g.contains(h) || g.iter().any(|p| p == "*"))
        }

        (Capability::Mcp { server: gs, tool: gt }, Capability::Mcp { server: rs, tool: rt }) => {
            (gs == rs || gs == "*") && (gt == rt || gt == "*")
        }

        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_match_simple() {
        assert!(capability_matches(&[Capability::Rag], &Capability::Rag));
        assert!(capability_matches(&[Capability::Memory], &Capability::Memory));
        assert!(capability_matches(&[Capability::Browser], &Capability::Browser));
        assert!(capability_matches(&[Capability::Inference], &Capability::Inference));
        assert!(capability_matches(&[Capability::Compute], &Capability::Compute));
    }

    #[test]
    fn test_mismatch_denied() {
        assert!(!capability_matches(&[Capability::Rag], &Capability::Memory));
        assert!(!capability_matches(&[Capability::Browser], &Capability::Compute));
        assert!(!capability_matches(&[], &Capability::Rag));
    }

    #[test]
    fn test_shell_wildcard() {
        let granted = Capability::Shell { allowed_commands: vec!["*".into()] };
        let required = Capability::Shell { allowed_commands: vec!["ls".into(), "cat".into()] };
        assert!(capability_matches(&[granted], &required));
    }

    #[test]
    fn test_shell_specific() {
        let granted = Capability::Shell { allowed_commands: vec!["ls".into()] };
        let required = Capability::Shell { allowed_commands: vec!["ls".into()] };
        assert!(capability_matches(&[granted.clone()], &required));

        let denied = Capability::Shell { allowed_commands: vec!["rm".into()] };
        assert!(!capability_matches(&[granted], &denied));
    }

    #[test]
    fn test_network_wildcard() {
        let granted = Capability::Network { allowed_hosts: vec!["*".into()] };
        let required = Capability::Network { allowed_hosts: vec!["api.example.com".into()] };
        assert!(capability_matches(&[granted], &required));
    }

    #[test]
    fn test_network_specific() {
        let granted = Capability::Network { allowed_hosts: vec!["localhost".into()] };
        let required = Capability::Network { allowed_hosts: vec!["localhost".into()] };
        assert!(capability_matches(&[granted.clone()], &required));

        let denied = Capability::Network { allowed_hosts: vec!["evil.com".into()] };
        assert!(!capability_matches(&[granted], &denied));
    }

    #[test]
    fn test_mcp_exact() {
        let granted = Capability::Mcp { server: "fs".into(), tool: "read".into() };
        let required = Capability::Mcp { server: "fs".into(), tool: "read".into() };
        assert!(capability_matches(&[granted], &required));
    }

    #[test]
    fn test_mcp_tool_wildcard() {
        let granted = Capability::Mcp { server: "fs".into(), tool: "*".into() };
        let required = Capability::Mcp { server: "fs".into(), tool: "read".into() };
        assert!(capability_matches(&[granted], &required));
    }

    #[test]
    fn test_mcp_server_mismatch() {
        let granted = Capability::Mcp { server: "fs".into(), tool: "*".into() };
        let required = Capability::Mcp { server: "db".into(), tool: "query".into() };
        assert!(!capability_matches(&[granted], &required));
    }

    #[test]
    fn test_multiple_granted_any_match() {
        let granted = vec![Capability::Rag, Capability::Memory, Capability::Browser];
        assert!(capability_matches(&granted, &Capability::Memory));
        assert!(!capability_matches(&granted, &Capability::Compute));
    }

    #[test]
    fn test_spawn_capability() {
        let granted = Capability::Spawn { max_depth: 3 };
        let required = Capability::Spawn { max_depth: 2 };
        assert!(capability_matches(&[granted], &required));

        let too_deep = Capability::Spawn { max_depth: 5 };
        let shallow = Capability::Spawn { max_depth: 1 };
        assert!(!capability_matches(&[shallow], &too_deep));

        assert!(!capability_matches(&[Capability::Compute], &Capability::Spawn { max_depth: 1 },));
    }

    #[test]
    fn test_serialization_roundtrip() {
        let caps = vec![
            Capability::Rag,
            Capability::Shell { allowed_commands: vec!["ls".into()] },
            Capability::Mcp { server: "s".into(), tool: "t".into() },
        ];
        for cap in &caps {
            let json = serde_json::to_string(cap).expect("serialize failed");
            let back: Capability = serde_json::from_str(&json).expect("deserialize failed");
            assert_eq!(*cap, back);
        }
    }

    // ════════════════════════════════════════════
    // PROPERTY TESTS — capability matching invariants
    // ════════════════════════════════════════════

    mod prop {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// INV-003: Empty grants deny everything.
            #[test]
            fn prop_empty_grants_deny_all(
                depth in 1u32..10,
            ) {
                let required = Capability::Spawn { max_depth: depth };
                prop_assert!(
                    !capability_matches(&[], &required),
                    "empty grants must deny all capabilities"
                );
            }

            /// A capability always matches itself.
            #[test]
            fn prop_self_match(depth in 1u32..10) {
                let cap = Capability::Spawn { max_depth: depth };
                prop_assert!(
                    capability_matches(&[cap.clone()], &cap),
                    "capability must match itself"
                );
            }

            /// Network wildcard matches any host.
            #[test]
            fn prop_network_wildcard_matches_all(
                host in "[a-z]{3,10}\\.[a-z]{2,4}",
            ) {
                let granted = Capability::Network {
                    allowed_hosts: vec!["*".into()],
                };
                let required = Capability::Network {
                    allowed_hosts: vec![host],
                };
                prop_assert!(
                    capability_matches(&[granted], &required),
                    "wildcard must match any host"
                );
            }

            /// Shell wildcard matches any command.
            #[test]
            fn prop_shell_wildcard_matches_all(
                cmd in "[a-z]{2,10}",
            ) {
                let granted = Capability::Shell {
                    allowed_commands: vec!["*".into()],
                };
                let required = Capability::Shell {
                    allowed_commands: vec![cmd],
                };
                prop_assert!(
                    capability_matches(&[granted], &required),
                    "wildcard must match any command"
                );
            }

            /// Spawn depth: granted max_depth must be >= required.
            #[test]
            fn prop_spawn_depth_requires_sufficient_grant(
                granted_depth in 1u32..20,
                required_depth in 1u32..20,
            ) {
                let granted = Capability::Spawn { max_depth: granted_depth };
                let required = Capability::Spawn { max_depth: required_depth };
                let result = capability_matches(&[granted], &required);

                if granted_depth >= required_depth {
                    prop_assert!(result, "depth {granted_depth} >= {required_depth} must match");
                } else {
                    prop_assert!(!result, "depth {granted_depth} < {required_depth} must deny");
                }
            }

            /// Proof obligation: capability_matches is pure (idempotent).
            #[test]
            fn prop_capability_match_idempotent(depth in 1u32..10) {
                let granted = vec![Capability::Spawn { max_depth: depth }];
                let required = Capability::Spawn { max_depth: depth };
                let r1 = capability_matches(&granted, &required);
                let r2 = capability_matches(&granted, &required);
                prop_assert_eq!(r1, r2, "capability_matches must be pure");
            }
        }
    }
}
