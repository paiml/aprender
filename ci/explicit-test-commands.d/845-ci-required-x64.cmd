# The three required jobs that were arch-neutral pin X64 (gx10 ARM runners die at job start); each removed X64 must turn --self-test RED.
bash scripts/check_ci_required_x64.sh
bash scripts/check_ci_required_x64.sh --self-test
