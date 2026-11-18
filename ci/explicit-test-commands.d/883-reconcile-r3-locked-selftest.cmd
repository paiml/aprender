# #7229 R3B: check_reconcile.sh self-test (R3 exempts release/** and single-level nightly/<name>; nested nightly/a/b, releasex/, fix/nightly/ still listed)
out=$(bash scripts/check_reconcile.sh --self-test 2>&1); rc=$?; printf '%s\n' "$out"; [ "$rc" -eq 0 ] && [[ "$out" == *"self-test OK: 33 case(s)."* ]]
