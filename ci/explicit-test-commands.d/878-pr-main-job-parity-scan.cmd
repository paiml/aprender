# Guard: the scan of .github/workflows (RED on a push-to-main job skipped on pull_request, or an aggregate that passes a PR early).
bash scripts/check_pr_main_job_parity.sh
