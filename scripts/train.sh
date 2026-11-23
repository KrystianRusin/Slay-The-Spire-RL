#!/usr/bin/env bash
# A long-running training stack that development can't disturb.
#
#   scripts/train.sh start     start (or resume) training; builds the frozen image the first time
#   scripts/train.sh update    rebuild the frozen image from the current code, then restart on it
#   scripts/train.sh stop      stop training (checkpoints are kept)
#   scripts/train.sh status    containers, and the last few training rounds
#   scripts/train.sh logs      follow the learner
#   scripts/train.sh watch     watch its newest policy play the real game (see docs/runbook.md)
#
# It's compose.yaml run as its own project (sts2-train), with:
#   STS2_TAG=train            images sts2-game:train and sts-coordinator:train, which only
#                             `update` rebuilds; dev builds retag :dev and leave these alone
#   STS2_RUN_DIR=./run/train  its own checkpoints and actor folders
# so `docker compose up --build ...` for development starts a separate stack
# (project sts2, tag dev, ./run) next to it.
#
# NUM_ACTORS [8] leaves most of the CPU free for development. STS2_ASCENSION [0].
# Changing NUM_ACTORS between runs is fine; changing settings in
# model/policy.py makes the learner refuse to resume (move run/train/sts2-learner
# aside to start over).
set -euo pipefail
cd "$(dirname "$0")/.."

export STS2_TAG=train STS2_RUN_DIR=./run/train
export NUM_ACTORS=${NUM_ACTORS:-8}
COMPOSE=(docker compose -p sts2-train)

case "${1:-}" in
  start)
    mkdir -p run/train/sts2-actors run/train/sts2-learner
    if ! docker image inspect sts2-game:train >/dev/null 2>&1; then
      "${COMPOSE[@]}" build learner coordinator
    fi
    "${COMPOSE[@]}" up -d learner actor
    echo "Training with $NUM_ACTORS actors. Follow it with: scripts/train.sh logs"
    ;;
  update)
    "${COMPOSE[@]}" build learner coordinator
    "${COMPOSE[@]}" up -d learner actor
    ;;
  stop)
    "${COMPOSE[@]}" down
    ;;
  status)
    "${COMPOSE[@]}" ps --format '{{.Name}}  {{.Status}}' | sort -V | head -5
    echo "... $("${COMPOSE[@]}" ps -q actor | wc -l) actors"
    "${COMPOSE[@]}" logs learner 2>/dev/null | grep -E "round for|trained" | tail -4 | cut -c1-200
    ;;
  logs)
    "${COMPOSE[@]}" logs -f learner
    ;;
  watch)
    shift
    "${COMPOSE[@]}" run --rm --no-deps live "$@"
    ;;
  *)
    sed -n '2,9p' "$0"
    exit 1
    ;;
esac
