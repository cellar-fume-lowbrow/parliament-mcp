#!/usr/bin/env bash
# Resumable month-by-month backfill of Hansard and parliamentary questions.
#
# Usage (from the repo root, with the containers running):
#   scripts/backfill.sh 2025-10 2026-09            # both sources, newest month first
#   scripts/backfill.sh 2025-10 2026-09 hansard    # one source only
#
# - Each month/source is loaded separately, so a failure costs one month, not the run.
# - Completed months are recorded in .backfill/done/ and skipped on re-run,
#   so re-running the same command resumes where it left off.
# - Failed months are retried (MAX_TRIES), then skipped and listed at the end.
# - Keeps the Mac awake while running (caffeinate).
# - Per-month logs go to .backfill/logs/.
#
# Written for macOS's bash 3.2: no associative arrays or GNU date.

set -uo pipefail

usage() {
  echo "Usage: $0 FROM_YYYY-MM TO_YYYY-MM [hansard|parliamentary-questions]" >&2
  exit 1
}

[ $# -ge 2 ] || usage
FROM="$1"
TO="$2"
SOURCES="${3:-hansard parliamentary-questions}"
MAX_TRIES="${MAX_TRIES:-3}"
RETRY_PAUSE="${RETRY_PAUSE:-300}"  # Parliament API errors tend to be load-related; give it time
LOAD_CMD="${LOAD_CMD:-docker compose exec -T mcp-server uv run parliament-mcp --log-level WARNING load-data}"

for ym in "$FROM" "$TO"; do
  echo "$ym" | grep -Eq '^[0-9]{4}-(0[1-9]|1[0-2])$' || { echo "Bad month: $ym (expected YYYY-MM)" >&2; usage; }
done
for s in $SOURCES; do
  case "$s" in hansard|parliamentary-questions) ;; *) echo "Unknown source: $s" >&2; usage ;; esac
done

# Keep the machine awake: re-exec under caffeinate once (macOS only).
if [ -z "${BACKFILL_CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  export BACKFILL_CAFFEINATED=1
  exec caffeinate -i "$0" "$@"
fi

STATE_DIR=".backfill"
mkdir -p "$STATE_DIR/done" "$STATE_DIR/logs"

days_in_month() { # year month(no leading zero)
  case "$2" in
    1|3|5|7|8|10|12) echo 31 ;;
    4|6|9|11) echo 30 ;;
    2) if [ $(($1 % 4)) -eq 0 ] && { [ $(($1 % 100)) -ne 0 ] || [ $(($1 % 400)) -eq 0 ]; }; then echo 29; else echo 28; fi ;;
  esac
}

# Build the month list newest-first, so the most useful data lands soonest.
from_y=${FROM%-*}; from_m=$((10#${FROM#*-}))
y=${TO%-*};        m=$((10#${TO#*-}))
MONTHS=""
while [ "$y" -gt "$from_y" ] || { [ "$y" -eq "$from_y" ] && [ "$m" -ge "$from_m" ]; }; do
  MONTHS="$MONTHS $(printf '%04d-%02d' "$y" "$m")"
  m=$((m - 1)); if [ "$m" -eq 0 ]; then m=12; y=$((y - 1)); fi
done
[ -n "$MONTHS" ] || { echo "FROM is after TO" >&2; exit 1; }

FAILED=""
run_start=$(date +%s)

for ym in $MONTHS; do
  yy=${ym%-*}; mm=$((10#${ym#*-}))
  start="$ym-01"
  end="$ym-$(days_in_month "$yy" "$mm")"
  for source in $SOURCES; do
    marker="$STATE_DIR/done/$source-$ym"
    if [ -f "$marker" ]; then
      echo "[skip] $source $ym (already done)"
      continue
    fi
    log="$STATE_DIR/logs/$source-$ym.log"
    try=1
    while :; do
      echo "[load] $source $ym ($start to $end), attempt $try/$MAX_TRIES ..."
      t0=$(date +%s)
      $LOAD_CMD "$source" --from-date "$start" --to-date "$end" >>"$log" 2>&1
      rc=$?
      secs=$(($(date +%s) - t0))
      if [ "$rc" -eq 0 ]; then
        date '+%Y-%m-%d %H:%M:%S' >"$marker"
        echo "[done] $source $ym in $((secs / 60))m$((secs % 60))s"
        break
      fi
      echo "[fail] $source $ym (exit $rc after ${secs}s) - see $log"
      if [ "$try" -ge "$MAX_TRIES" ]; then
        FAILED="$FAILED $source:$ym"
        break
      fi
      try=$((try + 1))
      sleep "$RETRY_PAUSE"
    done
  done
done

total=$(($(date +%s) - run_start))
echo
echo "Finished in $((total / 3600))h$(((total % 3600) / 60))m."
if [ -n "$FAILED" ]; then
  echo "Months that still failed:$FAILED"
  echo "Re-run the same command to retry just those."
  exit 1
fi
echo "All months loaded."
