# paiml-agy-delegate memory

- [Quorum schema verdict-enum mismatch](feedback_quorum_schema_verdict_enum_mismatch.md) — pinned enum is PASS|FAIL|do-not-implement-as-written; inject a DESIGN_VERDICT= mapping, and `--mode plan` is the identity wrapper when `mode` is omitted
- [Lanes need a cd wrapper script](feedback_lanes_need_a_cd_wrapper_script.md) — agy-lane.sh execs agy and inherits cwd; fan-out via bash launch.sh that cd's to repo_root, capture per-lane rc
- [Read `response` when findings are labels](feedback_read_response_when_findings_are_labels.md) — a lane may put every citation in `response` and leave `findings` as bare axis headings
- [plan-mode lanes never run commands](feedback_plan_mode_lanes_do_not_run_commands.md) — num_turns=1 means zero tool calls; downgrade every `measured` grounding and name the unrun commands
- [Detach lanes from the Bash-tool timeout](feedback_detach_lanes_from_bash_tool_timeout.md) — a foreground launcher dies at 120s and agy reports status=ERROR "timeout waiting for response"; setsid nohup + poll, and `bash launch.sh --dry-run` still launches for real
- [Verify the pivotal citation before reporting consensus](feedback_verify_the_pivotal_citation_before_reporting_consensus.md) — check the one file:line every chain hinges on; cargo feature implication runs one way only; read the obligation's scope before passing a reject up
- [Goal-mode lanes DO run tools](feedback_goal_mode_lanes_do_run_tools.md) — num_turns=1 is not zero tool calls outside plan mode (use input_tokens); goal+quorum-schema works if the OUTPUT CONTRACT block comes first; probe --model eligibility
