#!/usr/bin/env bash
# Shared by the build and the runtime entrypoint: which game build we support,
# and a check that the mounted game is that build.
#
# The game is in Early Access and patches often. The headless host patches
# sts2.dll's IL and reaches into its internals, so a different build can fail
# in quiet ways (hangs, wrong rules) rather than loudly. We pin one build and
# refuse to run anything else unless asked to.
#
# release_info.json ships in the game folder, e.g.
#   {"commit": "cc1f9462", "version": "v0.111.1", ..., "main_assembly_hash": 1176378353}
# main_assembly_hash identifies sts2.dll itself, so it catches a re-released
# build that kept the same version string.

STS2_PINNED_VERSION=v0.111.1
STS2_PINNED_ASSEMBLY_HASH=1176378353

# Print one field of a release_info.json.
release_field() {  # file field
  python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"
}

# Exit with a message unless the game at $1 is the pinned build.
# STS2_ALLOW_ANY_VERSION=1 turns the refusal into a warning, for trying a new build.
require_pinned_game() {  # game_dir
  local info=$1/release_info.json version hash
  if [ ! -f "$info" ]; then
    echo "No $info. Is Slay the Spire 2 mounted at $1?" >&2
    exit 1
  fi
  version=$(release_field "$info" version)
  hash=$(release_field "$info" main_assembly_hash)
  if [ "$version" = "$STS2_PINNED_VERSION" ] && [ "$hash" = "$STS2_PINNED_ASSEMBLY_HASH" ]; then
    return 0
  fi
  echo "Game is $version (assembly hash $hash), but this setup is pinned to" >&2
  echo "$STS2_PINNED_VERSION (hash $STS2_PINNED_ASSEMBLY_HASH)." >&2
  if [ "${STS2_ALLOW_ANY_VERSION:-0}" = 1 ]; then
    echo "STS2_ALLOW_ANY_VERSION=1: continuing anyway." >&2
    return 0
  fi
  echo "Install the pinned build, or set STS2_ALLOW_ANY_VERSION=1 to try this one." >&2
  exit 1
}
