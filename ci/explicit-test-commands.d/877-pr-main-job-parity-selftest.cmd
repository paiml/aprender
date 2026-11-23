# Guard: every push-to-main job also runs on pull_request, and no job passes a PR before its heavy checks (red-main parity). Self-test table.
bash scripts/check_pr_main_job_parity.sh --self-test
