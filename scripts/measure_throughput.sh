#!/usr/bin/env bash
# Measures training throughput (env steps per second) at several actor counts.
#
#   scripts/measure_throughput.sh [rounds] [actor counts...]    default: 2 rounds at 1 2 4
#
# For each count it starts a fresh learner (its checkpoints are deleted) and that
# many actors, with full-length rounds (LEARNER_N_STEPS from TrainingConfig),
# waits for the given number of rounds after the first, then shuts down. The
# first round is skipped: it includes the games booting. Results are printed
# and saved to run/throughput/<timestamp>.txt.
set -euo pipefail
cd "$(dirname "$0")/.."

ROUNDS=${1:-2}
shift || true
COUNTS=${*:-1 2 4}
ROUND_TIMEOUT=${ROUND_TIMEOUT:-1800}  # seconds; a dead actor stalls a round forever
COMPOSE=(docker compose -f compose.yaml -f compose.norender.yaml)
export LEARNER_N_STEPS=${LEARNER_N_STEPS:-}  # empty: full rounds. Set it for a quick try.

mkdir -p run/actors run/learner run/throughput
OUT="run/throughput/$(date +%Y%m%d-%H%M%S).txt"

report() { echo "$*" | tee -a "$OUT"; }
round_lines() { docker compose logs learner 2>/dev/null | grep -o 'round for v.*' || true; }

report "throughput, $ROUNDS measured round(s) per count, actor counts: $COUNTS, $(nproc) cores"
for n in $COUNTS; do
  "${COMPOSE[@]}" down >/dev/null 2>&1 || true
  rm -rf run/learner/checkpoints
  NUM_ACTORS=$n "${COMPOSE[@]}" up -d learner actor >/dev/null 2>&1
  report ""
  report "== $n actor(s), started $(date +%T)"

  want=$((ROUNDS + 1))
  deadline=$((SECONDS + ROUND_TIMEOUT * want))
  while [ "$(round_lines | wc -l)" -lt "$want" ]; do
    if [ "$SECONDS" -gt "$deadline" ]; then
      report "TIMED OUT waiting for $want rounds; actor errors:"
      docker compose logs actor 2>/dev/null | grep -E "Traceback|Error" | tail -5 | tee -a "$OUT"
      break
    fi
    sleep 15
  done

  docker stats --no-stream --format '   {{.Name}} cpu={{.CPUPerc}} mem={{.MemUsage}}' | grep slay-the-spire | tee -a "$OUT"
  round_lines | head -n "$want" | sed 's/^/   /' | tee -a "$OUT"
  # Mean over the measured rounds (all but the first).
  round_lines | head -n "$want" | tail -n +2 | awk -v n="$n" '
    { for (i = 1; i <= NF; i++) if ($(i + 1) == "steps/s") { rate += $i; count++ } }
    END { if (count) printf "   => %d actor(s): %.2f steps/s overall, %.2f per actor (mean of %d rounds)\n",
                            n, rate / count, rate / count / n, count }' | tee -a "$OUT"
  "${COMPOSE[@]}" down >/dev/null 2>&1
done
report ""
report "saved to $OUT"
