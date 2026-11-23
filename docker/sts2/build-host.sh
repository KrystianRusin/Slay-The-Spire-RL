#!/usr/bin/env bash
# Build the headless STS2 host from the mounted game, into /out.
#
#   /game  the Slay the Spire 2 install, read-only
#   /out   where the built host goes (a host folder; never part of an image)
#
# Steps: check the game is the pinned build, copy the sts2-cli source to /out,
# then run its setup.sh there. setup.sh copies the game's .NET DLLs, IL-patches
# sts2.dll so its async waits don't deadlock without Godot's main loop,
# extracts the text tables from the .pck, and compiles the host. Finally a
# copy of the game's release_info.json is stamped next to it, so the runtime
# can tell if the game changed after this build.
set -euo pipefail
source /opt/sts2/game-version.sh

GAME=${STS2_GAME_DIR:-/game}
OUT=${STS2_HOST_DIR:-/out}
DATA=$GAME/data_sts2_linuxbsd_x86_64

require_pinned_game "$GAME"
[ -f "$DATA/sts2.dll" ] || { echo "No $DATA/sts2.dll: expected the Linux build of the game." >&2; exit 1; }

# Start clean: a half-finished earlier build must not leave stale DLLs behind.
rm -rf "$OUT/sts2-cli" "$OUT/BUILT_FOR.json"
cp -a /src/sts2-cli "$OUT/sts2-cli"
# setup.sh finds the .pck by searching the folder it's given and its parent,
# so handing it the data folder finds /game/SlayTheSpire2.pck too.
"$OUT/sts2-cli/setup.sh" "$DATA"

HOST_DLL=$OUT/sts2-cli/src/Sts2Headless/bin/Debug/net9.0/Sts2Headless.dll
[ -f "$HOST_DLL" ] || { echo "Build finished but $HOST_DLL is missing." >&2; exit 1; }
cp "$GAME/release_info.json" "$OUT/BUILT_FOR.json"
echo "Built $HOST_DLL for $(release_field "$GAME/release_info.json" version)."
