# lib_rehearsal.sh — what the release scripts read in place of a dispatch, under RELEASE_REHEARSAL=1
# (APR-071 spec row B1, H10; E1 #3998). Sourced only when RELEASE_REHEARSAL=1.
#
# A rehearsal pushes no tag, so nothing can be dispatched on one. The steps that dispatch a producer on
# release day (models' GPU-host legs, clean-room, b2-gpu, the asset build) read instead the result that
# night's producer measured on main's head C -- the train bundle scripts/release/nightly_train.sh wrote
# into the rehearsal's state dir (rehearse.sh's `lanes` stage). The bump commit's parent is C: the bump
# changes the version and the CHANGELOG, nothing a producer measured.
#
# A lane reads green only when the bundle is for C and the lane's own row is green with run_head C.
# Anything else -- no bundle, two bundles, a bundle for another commit, a missing row, a red or
# not_measured row, a green on another head -- is not green, and the caller stops on it.
#
# SOURCED, so option-neutral: no `set` here (CLAUDE.md, scripts/check_sourced_libs_option_neutral.sh).

# rehearsal_lane LANE -> stdout the producer's run id, rc 0, when the night's bundle reads LANE green
# on C; else stdout "<state>: <reason>" and rc 1. Needs RELEASE_REHEARSAL_TRAIN and RELEASE_REHEARSAL_C.
rehearsal_lane() {
    local lane=${1:?} train=${RELEASE_REHEARSAL_TRAIN:-} c=${RELEASE_REHEARSAL_C:-} f n=0 b="" bc row
    [ -n "$train" ] && [ -n "$c" ] || { echo "not_measured: RELEASE_REHEARSAL_TRAIN or RELEASE_REHEARSAL_C is unset"; return 1; }
    for f in "$train"/*/bundle.tsv; do
        [ -f "$f" ] || continue
        n=$((n + 1)); b=$f
    done
    [ "$n" = 1 ] || { echo "not_measured: $n train bundles under $train, not exactly 1"; return 1; }
    bc=$(awk -F '\t' '$1 == "# C" { print $2 }' "$b")
    [ "$bc" = "$c" ] || { echo "not_measured: the train bundle is for ${bc:-no commit}, not C ${c:0:10}"; return 1; }
    # columns by the header's names, never by position
    row=$(awk -F '\t' -v L="$lane" '
        NR == 1 { for (i = 1; i <= NF; i++) H[$i] = i; next }
        /^#/ { next }
        $H["lane"] == L { print $H["state"] "\t" $H["run_id"] "\t" $H["run_head"] "\t" $H["reason"]; exit }' "$b")
    [ -n "$row" ] || { echo "not_measured: the train bundle has no $lane row"; return 1; }
    local state run head why
    IFS=$'\t' read -r state run head why <<< "$row"
    [ "$state" = green ] || { echo "$state: ${why:--}"; return 1; }
    [ "$head" = "$c" ] || { echo "not_measured: $lane read green on ${head:-?}, not C ${c:0:10}"; return 1; }
    [[ $run =~ ^[0-9]+$ ]] || { echo "not_measured: $lane read green with no run id (${run:-empty})"; return 1; }
    printf '%s\n' "$run"
}
