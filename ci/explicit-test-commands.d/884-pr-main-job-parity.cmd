# Guard: no job runs on a push to main that no pull_request or merge_group run covers (red-main parity). Self-test, then the scan.
bash scripts/check_pr_main_job_parity.sh --self-test
bash scripts/check_pr_main_job_parity.sh
