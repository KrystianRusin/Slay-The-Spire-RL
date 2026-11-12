#!/usr/bin/env bash
# Set up one game instance from the read-only game mount, start a virtual
# display, and launch ModTheSpire. CommunicationMod then spawns $STS_BOT_CMD and
# talks to it over stdin/stdout.
#
# Env (defaults in brackets):
#   STS_BOT_CMD   command CommunicationMod runs [python3 /opt/sts/app/bots/random_bot.py]
#   STS_ACTOR_CMD optional process to run next to the game (the RL actor). When
#                 set, the game runs in the background and this becomes the main
#                 process, so the container stops when the actor does.
#   STS_W/STS_H   resolution [640x360]      STS_FPS   frame cap [24]
#   STS_HEAP      Java max heap [512m]      JAVA_OPTS extra JVM flags
#   STS_INSTANCE_ROOT  if set, the instance folder is $STS_INSTANCE_ROOT/<hostname>,
#                 so scaled-up containers sharing one mount each get their own
#                 (the hostname is also the actor id in the learner's log).
#                 Otherwise it's STS_INSTANCE_DIR [/instance].
#   STS_MODS      mod ids passed to --mods [basemod,CommunicationMod]
#                 Extra mod jars (e.g. NoRender) can be mounted at /extra-mods.
set -euo pipefail

GAME=${STS_GAME_DIR:-/game}
if [ -n "${STS_INSTANCE_ROOT:-}" ]; then
  INST=$STS_INSTANCE_ROOT/$HOSTNAME
else
  INST=${STS_INSTANCE_DIR:-/instance}
fi
BOT_CMD=${STS_BOT_CMD:-python3 /opt/sts/app/bots/random_bot.py}
W=${STS_W:-640}; H=${STS_H:-360}; FPS=${STS_FPS:-24}
JAVA=$GAME/jre-linux/bin/java
# Lean JVM: live heap is ~125 MB, and default GC/JIT thread counts are sized for the whole host CPU.
JVM_TUNING=${JVM_TUNING:--Xms64m -XX:+UseSerialGC -XX:TieredStopAtLevel=1 -XX:CICompilerCount=1 -XX:ReservedCodeCacheSize=64m -Xss512k}

[ -x "$JAVA" ] || { echo "No Java 8 at $JAVA. Is the game mounted at $GAME?" >&2; exit 1; }

# --- instance folder: shared game files are symlinked, per-run state is copied ---
mkdir -p "$INST/home/.config/ModTheSpire/CommunicationMod" "$INST/mods"
cd "$INST"
for f in desktop-1.0.jar ModTheSpire.jar; do ln -sfn "$GAME/$f" "$f"; done
for j in "$GAME"/mods/*.jar /extra-mods/*.jar; do
  # if, not `[ ] &&`: an unmatched glob would end the loop non-zero and trip set -e
  if [ -e "$j" ]; then ln -sfn "$j" "mods/$(basename "$j")"; fi
done
# ModTheSpire runs helper processes with ./jre/bin/java.
ln -sfn "$GAME/jre-linux" jre
if [ ! -d saves ]; then
  cp -a "$GAME/saves" saves
  # Progress lives in betaPreferences on the Windows copy; seed both names so either branch finds it.
  cp -a "$GAME/betaPreferences" betaPreferences
  cp -a "$GAME/betaPreferences" preferences
fi
# Gameplay settings for bot play, applied to this instance's copy only:
#   Fast Mode on           shorter built-in action durations
#   Particles/shake off    purely visual effects, and some of them have timers the game waits on
#   Upload Data off        don't send every bot run's stats to Mega Crit
# STS_KEEP_SETTINGS=1 leaves the copied settings exactly as they were.
if [ "${STS_KEEP_SETTINGS:-0}" != 1 ]; then
  python3 - betaPreferences/STSGameplaySettings preferences/STSGameplaySettings <<'PY'
import json, sys
for path in sys.argv[1:]:
    try:
        with open(path) as f:
            settings = json.load(f)
    except FileNotFoundError:
        continue
    settings.update({"Fast Mode": "true", "Particle Effects": "false",
                     "Screen Shake": "false", "Upload Data": "false"})
    with open(path, "w") as f:
        json.dump(settings, f, indent=2)
PY
fi
# width, height, fps, fullscreen, windowed-fullscreen, vsync
printf '%s\n%s\n%s\nfalse\nfalse\nfalse\n' "$W" "$H" "$FPS" > info.displayconfig
cat > home/.config/ModTheSpire/CommunicationMod/config.properties <<EOF
command=$BOT_CMD
runAtGameStart=true
verbose=false
maxInitializationTimeout=60
EOF

# --- virtual display ---
DISP=:99
rm -f /tmp/.X99-lock
Xvfb "$DISP" -screen 0 1024x768x24 -nolisten tcp > "$INST/xvfb.log" 2>&1 &
for _ in $(seq 50); do [ -e /tmp/.X11-unix/X99 ] && break; sleep 0.1; done

echo "Starting Slay the Spire (bot: $BOT_CMD)"
export HOME=$INST/home DISPLAY=$DISP
GAME_CMD=("$JAVA" -Xmx"${STS_HEAP:-512m}" $JVM_TUNING ${JAVA_OPTS:-} -Duser.home="$HOME"
  -jar ModTheSpire.jar --skip-launcher --skip-intro --mods "${STS_MODS:-basemod,CommunicationMod}")

if [ -z "${STS_ACTOR_CMD:-}" ]; then
  exec "${GAME_CMD[@]}"
fi
# Game output goes to a file so the container log is the actor's. The actor's
# output is also kept in actor.log, so a crash's traceback outlives the container.
"${GAME_CMD[@]}" > "$INST/game.log" 2>&1 &
echo "Game log: $INST/game.log. Starting actor: $STS_ACTOR_CMD"
exec > >(tee -a "$INST/actor.log") 2>&1
exec $STS_ACTOR_CMD
