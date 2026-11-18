# #7158 lab stop rule: lab_stop.sh case table (gh stub) and mutants; nightly_greens.sh case table incl. lab rows (need 7, notgreen=)
out=$(bash scripts/release/lab_stop.sh --self-test 2>&1); rc=$?; printf '%s\n' "$out"; [ "$rc" -eq 0 ] && [[ "$out" == *"--- 17/17 rows ---"* ]] && bash scripts/release/lab_stop.sh --mutants && bash scripts/release/nightly_greens.sh --selftest
