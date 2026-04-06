//! Multi-agent orchestration pool.
//!
//! Manages concurrent agent instances with message passing
//! and fan-out/fan-in patterns. Each agent runs its own
//! perceive-reason-act loop in a separate tokio task.
//!
//! # Toyota Production System Principles
//!
//! - **Heijunka**: Load-level work across agents
//! - **Jidoka**: Each agent has its own `LoopGuard`
//! - **Muda**: Bounded concurrency prevents resource waste

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::mpsc;
use tokio::task::JoinSet;
use tracing::{debug, info, warn};

use super::driver::{LlmDriver, StreamEvent};
use super::manifest::AgentManifest;
use super::memory::{InMemorySubstrate, MemorySubstrate};
use super::result::{AgentError, AgentLoopResult};
use super::tool::ToolRegistry;

/// Unique identifier for a spawned agent.
pub type AgentId = u64;

/// Message sent between agents in the pool.
#[derive(Debug, Clone)]
pub struct AgentMessage {
    /// Source agent ID (0 = external/supervisor).
    pub from: AgentId,
    /// Target agent ID.
    pub to: AgentId,
    /// Message payload.
    pub content: String,
}

/// Configuration for a spawned agent.
pub struct SpawnConfig {
    /// Agent manifest.
    pub manifest: AgentManifest,
    /// Query to execute.
    pub query: String,
}

/// Routes messages between agents in a pool.
///
/// Each agent gets an inbox (bounded `mpsc` channel). The router
/// holds senders keyed by `AgentId`, so any agent can send to any
/// other agent via the shared router reference.
#[derive(Clone)]
pub struct MessageRouter {
    inboxes: Arc<std::sync::RwLock<HashMap<AgentId, mpsc::Sender<AgentMessage>>>>,
    inbox_capacity: usize,
}

impl MessageRouter {
    /// Create a new message router.
    pub fn new(inbox_capacity: usize) -> Self {
        Self { inboxes: Arc::new(std::sync::RwLock::new(HashMap::new())), inbox_capacity }
    }

    /// Register an agent inbox, returning the receiver.
    pub fn register(&self, agent_id: AgentId) -> mpsc::Receiver<AgentMessage> {
        let (tx, rx) = mpsc::channel(self.inbox_capacity);
        let mut inboxes = self.inboxes.write().expect("message router lock");
        inboxes.insert(agent_id, tx);
        rx
    }

    /// Unregister an agent (removes its inbox sender).
    pub fn unregister(&self, agent_id: AgentId) {
        let mut inboxes = self.inboxes.write().expect("message router lock");
        inboxes.remove(&agent_id);
    }

    /// Send a message to a target agent.
    ///
    /// Returns `Err` if target agent is not registered or inbox
    /// is full (bounded channel protects against backpressure).
    pub async fn send(&self, msg: AgentMessage) -> Result<(), String> {
        let tx = {
            let inboxes = self.inboxes.read().expect("message router lock");
            inboxes
                .get(&msg.to)
                .cloned()
                .ok_or_else(|| format!("agent {} not registered", msg.to))?
        };
        tx.send(msg).await.map_err(|e| format!("inbox closed: {e}"))
    }

    /// Number of registered agents.
    pub fn agent_count(&self) -> usize {
        let inboxes = self.inboxes.read().expect("message router lock");
        inboxes.len()
    }
}

/// Function that builds a `ToolRegistry` from a manifest.
pub type ToolBuilder = Arc<dyn Fn(&AgentManifest) -> ToolRegistry + Send + Sync>;

/// Multi-agent orchestration pool.
///
/// Manages concurrent agent instances, each running its own
/// perceive-reason-act loop. Supports fan-out (spawn many) and
/// fan-in (collect results) patterns.
///
/// ```rust,ignore
/// let mut pool = AgentPool::new(driver, 4);
/// pool.spawn(config1).await?;
/// pool.spawn(config2).await?;
/// let results = pool.join_all().await;
/// ```
pub struct AgentPool {
    driver: Arc<dyn LlmDriver>,
    memory: Arc<dyn MemorySubstrate>,
    next_id: AgentId,
    max_concurrent: usize,
    join_set: JoinSet<(AgentId, String, Result<AgentLoopResult, String>)>,
    stream_tx: Option<mpsc::Sender<StreamEvent>>,
    router: MessageRouter,
    tool_builder: Option<ToolBuilder>,
}

impl AgentPool {
    /// Create a new agent pool with bounded concurrency.
    pub fn new(driver: Arc<dyn LlmDriver>, max_concurrent: usize) -> Self {
        Self {
            driver,
            memory: Arc::new(InMemorySubstrate::new()),
            next_id: 1,
            max_concurrent,
            join_set: JoinSet::new(),
            stream_tx: None,
            router: MessageRouter::new(32),
            tool_builder: None,
        }
    }

    /// Access the message router for inter-agent messaging.
    pub fn router(&self) -> &MessageRouter {
        &self.router
    }

    /// Set a shared memory substrate for all agents.
    #[must_use]
    pub fn with_memory(mut self, memory: Arc<dyn MemorySubstrate>) -> Self {
        self.memory = memory;
        self
    }

    /// Set a stream event channel for pool-level events.
    #[must_use]
    pub fn with_stream(mut self, tx: mpsc::Sender<StreamEvent>) -> Self {
        self.stream_tx = Some(tx);
        self
    }

    /// Set a tool builder for spawned agents.
    ///
    /// When set, each spawned agent gets tools built from its
    /// manifest rather than an empty registry.
    #[must_use]
    pub fn with_tool_builder(mut self, builder: ToolBuilder) -> Self {
        self.tool_builder = Some(builder);
        self
    }

    /// Number of currently active agents.
    pub fn active_count(&self) -> usize {
        self.join_set.len()
    }

    /// Maximum concurrent agents allowed.
    pub fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }

    /// Spawn a new agent in the pool.
    ///
    /// Returns the `AgentId` assigned to this agent.
    /// Returns error if pool is at capacity.
    pub fn spawn(&mut self, config: SpawnConfig) -> Result<AgentId, AgentError> {
        if self.join_set.len() >= self.max_concurrent {
            return Err(AgentError::CircuitBreak(format!(
                "agent pool at capacity ({}/{})",
                self.join_set.len(),
                self.max_concurrent
            )));
        }

        let id = self.next_id;
        self.next_id += 1;

        let name = config.manifest.name.clone();
        let driver = Arc::clone(&self.driver);
        let memory = Arc::clone(&self.memory);
        let stream_tx = self.stream_tx.clone();

        // Register agent inbox for inter-agent messaging
        let _inbox_rx = self.router.register(id);
        let router = self.router.clone();

        info!(
            agent_id = id,
            name = %name,
            query_len = config.query.len(),
            "spawning agent"
        );

        let tool_builder = self.tool_builder.clone();

        self.join_set.spawn(async move {
            let tools = match tool_builder {
                Some(builder) => builder(&config.manifest),
                None => ToolRegistry::new(),
            };
            let result = super::runtime::run_agent_loop(
                &config.manifest,
                &config.query,
                driver.as_ref(),
                &tools,
                memory.as_ref(),
                stream_tx,
            )
            .await;

            // Unregister agent from router on completion
            router.unregister(id);

            // Map error to String to avoid Clone requirement
            let mapped = result.map_err(|e| e.to_string());
            (id, name, mapped)
        });

        Ok(id)
    }

    /// Fan-out: spawn multiple agents concurrently.
    ///
    /// Returns a list of `AgentId`s for the spawned agents.
    pub fn fan_out(&mut self, configs: Vec<SpawnConfig>) -> Result<Vec<AgentId>, AgentError> {
        let mut ids = Vec::with_capacity(configs.len());
        for config in configs {
            ids.push(self.spawn(config)?);
        }
        Ok(ids)
    }

    /// Fan-in: wait for all active agents to complete.
    ///
    /// Returns results keyed by `AgentId`. Agents that error
    /// are included with their error string.
    pub async fn join_all(&mut self) -> HashMap<AgentId, Result<AgentLoopResult, String>> {
        let mut results = HashMap::new();

        while let Some(outcome) = self.join_set.join_next().await {
            match outcome {
                Ok((id, name, result)) => {
                    debug!(
                        agent_id = id,
                        name = %name,
                        ok = result.is_ok(),
                        "agent completed"
                    );
                    results.insert(id, result);
                }
                Err(e) => {
                    warn!(error = %e, "agent task panicked");
                }
            }
        }

        results
    }

    /// Wait for the next agent to complete.
    ///
    /// Returns `None` if no agents are active.
    pub async fn join_next(&mut self) -> Option<(AgentId, Result<AgentLoopResult, String>)> {
        match self.join_set.join_next().await {
            Some(Ok((id, _name, result))) => Some((id, result)),
            Some(Err(e)) => {
                warn!(error = %e, "agent task panicked");
                None
            }
            None => None,
        }
    }

    /// Abort all running agents.
    pub fn abort_all(&mut self) {
        self.join_set.abort_all();
        info!("all agents aborted");
    }
}

#[cfg(test)]
#[path = "pool_tests.rs"]
mod tests;
