# check_runner_labels.sh case table: a label named only in a trailing comment must not discriminate; mq (kfocus-mq seats) does (AC-6475)
out=$(bash scripts/check_runner_labels.sh --self-test 2>&1); rc=$?; printf '%s\n' "$out"; [ "$rc" -eq 0 ] && ! printf '%s\n' "$out" | grep -q '^FAIL' && printf '%s\n' "$out" | grep -q ' 0 failed$'
