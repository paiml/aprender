# Drafts skip the heavy CI jobs and the required checks still report; each removed guard must turn --self-test RED.
bash scripts/check_ci_draft_skip.sh --self-test
