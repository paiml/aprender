# The three required jobs that were arch-neutral pin X64 (gx10 ARM runners die at job start); the self-test checks the real ci.yml is clean and each removed X64 must turn it RED.
bash scripts/check_ci_required_x64.sh --self-test
