# proc_resources.sh -- per-process resource snapshots for speed receipts
# (MEAS-001, aprender#4522, folded under #5322).
#
# Sourceable:  . scripts/lib/proc_resources.sh || exit 1
#
# A speed receipt that carries only timing cannot say WHAT the speed cost. Two
# runtimes at the same tok/s with 2x the RSS or 3x the CPU-seconds are not at
# parity, and a ratio published without that column hides it.
#
#   proc_snapshot PID         one JSON object on stdout: the process's own
#                             counters at this instant, read from /proc
#   proc_window BEFORE AFTER  one JSON object: what the process spent between
#                             two snapshots (CPU seconds, faults, IO bytes) and
#                             VmHWM at AFTER: the lifetime peak RSS so far, which
#                             includes every earlier window, not this one alone
#
# What it does NOT claim:
#   - energy is the CPU PACKAGE's (RAPL), host-wide, never the process's. It is
#     recorded as `package_energy_j` with that scope, or null with the reason
#     the counter was unreadable (it is root-only on most current kernels).
#   - vram_mib is the process's own row in nvidia-smi --query-compute-apps, or
#     null with a reason. A host with no nvidia-smi says so; it does not say 0.
#
# Option-neutral: a sourced file must not `set` the caller's shell options
# (scripts/check_sourced_libs_option_neutral.sh). Failures are return codes.

# proc_snapshot PID
proc_snapshot() {
    local pid="$1" stat rest io_r io_w hwm rss tck energy energy_why vram vram_why t_ns up
    [ -n "$pid" ] && [ -r "/proc/$pid/stat" ] || return 1
    stat=$(cat "/proc/$pid/stat" 2>/dev/null) || return 1
    # Field 2 (comm) may contain spaces and parentheses; everything after the
    # LAST ')' is space-separated and starts at field 3.
    rest=${stat##*) }
    # shellcheck disable=SC2086
    set -- $rest
    # field N is positional $((N-2)): minflt 10, majflt 12, utime 14, stime 15, threads 20
    local minflt="${8}" majflt="${10}" utime="${12}" stime="${13}" threads="${18}"
    hwm=$(awk '/^VmHWM:/{print $2}' "/proc/$pid/status" 2>/dev/null)
    rss=$(awk '/^VmRSS:/{print $2}' "/proc/$pid/status" 2>/dev/null)
    io_r=$(awk '/^read_bytes:/{print $2}' "/proc/$pid/io" 2>/dev/null)
    io_w=$(awk '/^write_bytes:/{print $2}' "/proc/$pid/io" 2>/dev/null)
    tck=$(getconf CLK_TCK 2>/dev/null)
    energy=null; energy_why=""
    if [ -r /sys/class/powercap/intel-rapl:0/energy_uj ]; then
        energy=$(cat /sys/class/powercap/intel-rapl:0/energy_uj 2>/dev/null) || energy=null
        [ -n "$energy" ] || energy=null
    else
        energy_why="rapl energy_uj not readable by this user"
    fi
    vram=null; vram_why=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        vram=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
            | awk -F', *' -v p="$pid" '$1 == p {print $2; exit}')
        if [ -z "$vram" ]; then vram=null; vram_why="pid not in nvidia-smi compute apps"; fi
        case $vram in
            null|*[!0-9]*)
                [ "$vram" = null ] || vram_why="nvidia-smi reported a non-numeric value: $vram"
                vram=null ;;
        esac
    else
        vram_why="no nvidia-smi on this host"
    fi
    # t_ns only ever feeds wall_s (b - a), so a clock that differs between two reads
    # does. EPOCHREALTIME (bash >= 5, wall clock like the old `date`) keeps this free
    # of a `date` call (bashrs DET002); non-digits are stripped because the radix
    # follows the locale (a comma under en_DK). Older bash falls back to
    # /proc/uptime, which is monotonic but 10 ms coarse.
    if [ -n "${EPOCHREALTIME:-}" ]; then
        t_ns="${EPOCHREALTIME//[!0-9]/}000"
    else
        read -r up _ < /proc/uptime
        t_ns="${up/./}0000000"
    fi
    jq -cn --argjson pid "$pid" --argjson t_ns "$t_ns" --argjson tck "${tck:-100}" \
        --argjson utime "$utime" --argjson stime "$stime" \
        --argjson minflt "$minflt" --argjson majflt "$majflt" --argjson threads "$threads" \
        --argjson hwm "${hwm:-null}" --argjson rss "${rss:-null}" \
        --argjson io_r "${io_r:-null}" --argjson io_w "${io_w:-null}" \
        --argjson energy "$energy" --arg energy_why "$energy_why" \
        --argjson vram "$vram" --arg vram_why "$vram_why" \
        '{pid: $pid, t_ns: $t_ns, clk_tck: $tck, utime_ticks: $utime, stime_ticks: $stime,
          minflt: $minflt, majflt: $majflt, threads: $threads,
          vm_hwm_kb: $hwm, vm_rss_kb: $rss, read_bytes: $io_r, write_bytes: $io_w,
          package_energy_uj: $energy, package_energy_why: $energy_why,
          vram_mib: $vram, vram_why: $vram_why}'
}

# proc_window BEFORE_JSON AFTER_JSON
proc_window() {
    jq -cn --argjson a "$1" --argjson b "$2" '
      def d(k): if ($a[k] == null or $b[k] == null) then null else $b[k] - $a[k] end;
      if $a.pid != $b.pid then error("snapshots are of two different pids") else . end
      | {
          pid: $b.pid,
          wall_s: (($b.t_ns - $a.t_ns) / 1e9),
          cpu_user_s: (d("utime_ticks") / $b.clk_tck),
          cpu_sys_s: (d("stime_ticks") / $b.clk_tck),
          minflt: d("minflt"), majflt: d("majflt"),
          read_bytes: d("read_bytes"), write_bytes: d("write_bytes"),
          vm_hwm_mib: (if $b.vm_hwm_kb == null then null else ($b.vm_hwm_kb / 1024 | floor) end),
          vm_hwm_scope: "process-lifetime high-water at window end (VmHWM), cumulative over earlier replicates, not this window alone",
          rss_end_mib: (if $b.vm_rss_kb == null then null else ($b.vm_rss_kb / 1024 | floor) end),
          threads: $b.threads,
          vram_mib: $b.vram_mib, vram_why: $b.vram_why,
          package_energy_j: (if d("package_energy_uj") == null or d("package_energy_uj") < 0
                             then null else d("package_energy_uj") / 1e6 end),
          package_energy_scope: "cpu package, host-wide (RAPL), not this process",
          package_energy_why: $b.package_energy_why
        }
      | .cpu_s = ((.cpu_user_s // 0) + (.cpu_sys_s // 0))
      | .cpu_util = (if .wall_s > 0 then (.cpu_s / .wall_s * 1000 | round / 1000) else null end)'
}

# res_begin PID / res_end PID FILE -- bracket one measured window. res_end
# always writes FILE: the window, or {"absent": "<why>"} when either snapshot
# failed (a server that died mid-replicate has no /proc entry to read). Both
# return 0, so a caller under `set -e` never loses a replicate to the sampler.
RES_BEFORE=""
res_begin() {
    RES_BEFORE=$(proc_snapshot "$1" 2>/dev/null) || RES_BEFORE=""
    return 0
}
res_end() {
    local after win
    after=$(proc_snapshot "$1" 2>/dev/null) || after=""
    if [ -n "$RES_BEFORE" ] && [ -n "$after" ] && win=$(proc_window "$RES_BEFORE" "$after" 2>/dev/null); then
        printf '%s\n' "$win" > "$2"
    else
        jq -cn --arg pid "${1:-}" '{absent: "no /proc snapshot of server pid \($pid) at the start or end of the window"}' > "$2"
    fi
    RES_BEFORE=""
    return 0
}
