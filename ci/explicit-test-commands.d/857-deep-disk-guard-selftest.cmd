# deep-nightly disk guard (#6126): floor check goes RED on an impossible floor and passes on 0; fails on a missing PASS line as well as the exit status
out=$(bash scripts/release/deep_disk_guard.sh --self-test 2>&1); rc=$?; printf "%s\n" "$out"; [ "$rc" -eq 0 ] && printf "%s\n" "$out" | grep -q "self-test: PASS$"
