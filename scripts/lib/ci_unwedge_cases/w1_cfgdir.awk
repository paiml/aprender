# W1 (check_ci_unwedge.sh --self-test) over .github/workflows/ci-unwedge.yml:
# each of the two steps that run code sets its OWN uncommented GH_CONFIG_DIR
# under runner.temp, and the two dirs differ. Prints "A|B|distinct" when so.
/^[[:space:]]*#/ { next }
/^[[:space:]]*- (name|uses):/ { step = $0; sub(/^[[:space:]]*- (name|uses):[[:space:]]*/, "", step) }
/^[[:space:]]+GH_CONFIG_DIR:[[:space:]]*\$\{\{ runner\.temp \}\}\/[A-Za-z0-9_.-]+[[:space:]]*$/ {
    n[step]++; dir[step] = $NF
}
END {
    a = "Predicate falsifier (must pass before anything is cancelled)"; b = "Scan and free"
    printf "%s|%s|%s", (n[a] == 1 ? "Predicate falsifier" : "missing"),
        (n[b] == 1 ? "Scan and free" : "missing"),
        ((n[a] == 1 && n[b] == 1 && dir[a] != dir[b]) ? "distinct" : "same")
}
