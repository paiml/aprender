# #7307: on a pull_request x86-main, the workspace-test shards and determinism start only after a green pr-cheap; push and merge_group unchanged. Case table over a fixture and over ci.yml itself (its first row is the clean scan). bash + awk only.
bash scripts/check_pr_cheap_gates_heavy.sh --self-test
