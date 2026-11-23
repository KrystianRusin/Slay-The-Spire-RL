#!/usr/bin/env bash
# Check the mounted game and the built host belong together, then run the
# command given (by default the random bot).
#
#   /game       the Slay the Spire 2 install, read-only
#   /sts2-host  the builder's output, read-only
#   /instance   this container's logs and outputs
set -euo pipefail
source /opt/sts2/game-version.sh

GAME=${STS2_GAME_DIR:-/game}
HOST_DIR=${STS2_HOST_DIR:-/sts2-host}

require_pinned_game "$GAME"

# The host was compiled against one copy of sts2.dll. If Steam updated the game
# since, the patched DLLs in the host folder are from the old build: rebuild.
stamp=$HOST_DIR/BUILT_FOR.json
if [ ! -f "$stamp" ]; then
  echo "No built host in $HOST_DIR. Build it first:" >&2
  echo "  docker compose -f compose.sts2.yaml run --rm sts2-build" >&2
  exit 1
fi
built=$(release_field "$stamp" main_assembly_hash)
mounted=$(release_field "$GAME/release_info.json" main_assembly_hash)
if [ "$built" != "$mounted" ]; then
  echo "The host was built for assembly hash $built, but the mounted game is $mounted." >&2
  echo "The game changed since the build. Rebuild with sts2-build." >&2
  exit 1
fi

export STS2_HOST_DLL=$HOST_DIR/sts2-cli/src/Sts2Headless/bin/Debug/net9.0/Sts2Headless.dll
# The host writes its save files relative to the working directory. With
# STS2_INSTANCE_ROOT set, each container gets <root>/<hostname>, so scaled-up
# actors sharing one mount keep their files apart (the hostname is also the
# actor id in the learner's log).
if [ -n "${STS2_INSTANCE_ROOT:-}" ]; then
  INST=$STS2_INSTANCE_ROOT/$HOSTNAME
else
  INST=${STS2_INSTANCE_DIR:-/instance}
fi
mkdir -p "$INST"
cd "$INST"

if [ $# -eq 0 ]; then
  set -- python3 /opt/sts/app/bots/sts2_random_bot.py
fi
exec "$@"
