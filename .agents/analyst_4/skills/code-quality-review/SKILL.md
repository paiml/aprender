---
name: code-quality-review
description: >-
  Use this skill to perform a comprehensive code quality and architecture review of the current project or repository,
  identifying code smells, architectural debt, performance bottlenecks, testing gaps,
  and logic issues. Trigger this whenever the user asks for a comprehensive code review
  or quality audit.
---

# Code Quality Review

This skill orchestrates a parallel audit of a codebase using a quorum of highly specialized subagents. It is designed to work out-of-the-box on any PAIML project.

## Workflow

1.  **Spawn Subagents**: Use the `invoke_subagent` tool to spawn a quorum of specialized subagents. Assign each subagent a highly specialized role.
    Ensure that the roles cover a wide range of engineering concerns. Example roles must include, but are not limited to:
    - Quantitative PMAT Auditor (Must use the `pmat` CLI to review quantitative quality and formal verification metrics)
    - Architecture Auditor: Dependencies
    - Architecture Auditor: State Management
    - Performance: Hot Paths
    - Performance: IO and Async
    - Performance: Data Structures
    - Testing: Unit Coverage
    - API Design Ergonomics
    - Build System Auditor
    - Documentation Auditor
    - Scalability & Large Data
    - Code Smell Auditor
    - Error Handling Auditor

2.  **Define Prompts**: For each subagent, provide a highly specific prompt instructing them to review the current project (or the specific repositories the user requested) for issues matching their domain. Instruct them to use fast tools (like `grep_search` or `view_file`) and return concise, high-impact findings (e.g. top 1-3 critical issues) to avoid overwhelming the context. Ensure the PMAT Auditor is instructed to run `pmat`.

3.  **Wait for Reports**: Pause your execution and wait for all subagents to report back. Do not poll. The system will automatically wake you up and notify you as messages arrive.

4.  **Create Quality Report Epic**: Once all subagents have submitted their findings, assimilate the entire report into a comprehensive Quality Report Epic.
    Use the `gh` CLI (via `run_command` executing `gh issue create`) to create and populate this Epic on the GitHub repository.
    
    Ensure the resulting Epic includes:
    *   A categorized breakdown of all findings and architectural debt.
    *   A roster of the names and roles of the agents that participated.
    *   Individual feedback and findings attributed to each specific agent.
