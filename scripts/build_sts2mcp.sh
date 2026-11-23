#!/usr/bin/env bash
# Build the STS2MCP mod (github.com/Gennadiyev/STS2MCP, MIT) against the
# installed game, into run/sts2-mod/. play_live.py drives the real game
# through it.
#
#   scripts/build_sts2mcp.sh            # then copy the two files it names into the game's mods/ folder
#
# Its last release (0.4.0) predates game v0.107, so this builds a pinned
# commit from source instead, compiled against your copy of sts2.dll. Two
# multiplayer-lobby members it reads don't exist in v0.111.1; they're patched
# out below (singleplayer doesn't touch them).
#
# The build runs in the sts2-builder image (docker compose build sts2-build),
# with the game mounted read-only. Needs network for the git clone.
set -euo pipefail
cd "$(dirname "$0")/.."

COMMIT=55e0648  # "Fix game API compatibility with STS2 v0.107 (#123)"
GAME=${STS2_GAME_DIR:-$HOME/.var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Slay the Spire 2}
OUT=run/sts2-mod
mkdir -p "$OUT"

docker run --rm --user "$(id -u):$(id -g)" --entrypoint bash \
  -v "$GAME":/game:ro -v "$PWD/$OUT":/out sts2-builder:dev -c "
    set -euo pipefail
    git clone -q https://github.com/Gennadiyev/STS2MCP.git /tmp/src
    cd /tmp/src && git checkout -q $COMMIT
    sed -i 's/\[\"max_players\"\] = lobby.MaxPlayers,/[\"max_players\"] = 0,/; s/lobby\.ConnectedPlayerIds/((System.Collections.Generic.HashSet<ulong>?)null)/g' McpMod.StateBuilder.cs
    dotnet build -c Release -p:STS2GameDir=/game -o /tmp/build 2>&1 | grep -E ' error |Error\(s\)' | sort -u
    cp /tmp/build/STS2_MCP.dll /out/
    cp mod_manifest.json /out/STS2_MCP.json
  "
echo
echo "Built $OUT/STS2_MCP.dll and $OUT/STS2_MCP.json. Install them with:"
echo "  mkdir -p \"$GAME/mods\" && cp $OUT/STS2_MCP.dll $OUT/STS2_MCP.json \"$GAME/mods/\""
