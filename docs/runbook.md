# Runbook: tests, live runs and fault tests

All commands run from the repository root. Docker does everything; nothing
needs installing on the host beyond Docker itself.

## 0. One-time setup

Slay the Spire 2 installed through Steam (Linux build). Set it to update only
when launched (Properties > Updates): the setup is pinned to one game build
(`docker/sts2/game-version.sh`) and refuses to run on another.

```bash
# Only if your install isn't Flatpak Steam's default library:
export STS2_GAME_DIR=~/.local/share/Steam/steamapps/common/"Slay the Spire 2"
mkdir -p run/sts2-host run/sts2-random run/sts2-actors run/sts2-learner   # else Docker creates them owned by root

# Build the headless game host from your game files (again after a game update)
docker compose run --rm --build sts2-build
```

## 1. Automated tests

```bash
# Go: the coordinator's own tests, with the race detector
docker build -f docker/coordinator/Dockerfile --target test .

# Python: rebuild the images first so they contain your changes
docker compose build learner coordinator
docker run --rm --user 0 --entrypoint bash \
  -v ./tests:/opt/sts/app/tests:ro -v ./pytest.ini:/opt/sts/app/pytest.ini:ro \
  -w /opt/sts/app sts2-game:dev \
  -c "pip install -q pytest >/dev/null 2>&1; python3 -m pytest -q tests"
```

To run one file or one test, replace the final `tests` with e.g.
`tests/test_learner_restart.py` or `tests/test_rpc.py::test_a_full_round_trains_and_publishes_the_next_version`.

The Go build printing an image id (`sha256:...`) and nothing else means the
tests passed; a failure prints the failing test and stops the build.

## 2. A live run

```bash
# Start fresh: set the old checkpoints aside (a run with other settings
# refuses to resume from them)
mv run/sts2-learner/checkpoints run/sts2-learner/checkpoints.$(date +%s) 2>/dev/null

# 2 actors, short 256-step rounds. -d = in the background.
NUM_ACTORS=2 LEARNER_N_STEPS=256 docker compose up -d --build learner actor

# A real training run: about one actor per two cores.
NUM_ACTORS=24 docker compose up -d --build learner actor
```

The headless game starts in about a second, so the first round arrives quickly. Then:

```bash
docker compose ps                                  # what's running, and restarts
docker compose logs -f learner                     # follow the learner (Ctrl-C stops following, not the run)
docker compose logs -f coordinator
docker compose logs -f actor                       # both actors, interleaved
docker compose logs -t --since 2m learner          # -t adds timestamps; --since takes 30s, 2m, 1h
tail -f run/sts2-actors/*/episodes.jsonl          # one line per finished run: character, floor, reward
```

### What healthy looks like

| Log line | What to check |
|---|---|
| `learner: trained v3 -> v4 ... first \|ratio-1\| mean=0.000000 max=0.000000` | Must be 0: actors played with exactly the weights the learner trains. Anything else is a bug. |
| `learner: round for v3 took 116.5s, 422.02 steps/s overall, rollouts arrived over 12.2s` | Throughput (24 actors), and how far apart actors finished. |
| `... returns mean=0.21 std=1.27, explained_variance=0.556` | Value function learning: explained variance should rise above 0 over rounds. |
| `publisher: published v4, 0.89 MB` | The learner handed v4 to the coordinator. |
| `actor ...: loaded weights v4, ... waited 0.3s` | An actor got v4. |
| `REJECTED` (learner) / `unreachable` (actor) | Fine during a fault test; not in a quiet run. |

Quick summaries:

```bash
docker compose logs learner | grep -E "round for|ratio" | tail      # recent rounds
docker compose logs | grep -cE "REJECTED|Traceback"                  # should be 0 in a quiet run
```

### Stopping

```bash
docker compose down
```

Checkpoints in `run/sts2-learner/checkpoints` survive `down`; the next `up`
resumes from them.

How far runs get, by version (each line of episodes.jsonl is one run):

```bash
cat run/sts2-actors/*/episodes.jsonl | python3 -c "
import sys, json
runs = [json.loads(l) for l in sys.stdin]
recent = runs[-500:]
print('last 500 runs: mean floor', sum(r.get('floor') or 0 for r in recent) / len(recent),
      ' wins', sum(r['outcome'] == 'victory' for r in recent))"
```

## 3. Fault tests

The point is to **predict first**. Before each one, write down:
1. which process notices first, and how (an error? a timeout? a heartbeat?)
2. which log lines you expect, in order
3. how long until a new `trained v...` line appears
4. whether anything is lost (rejected rollouts, a replayed rollout, a stuck actor)

Then run it, read the logs with `docker compose logs -t --since 3m`, and
compare. Wrong predictions are the useful part.

### Crash the learner

```bash
docker compose exec learner pkill -9 -f learner_server.py
```

`pkill -9` sends SIGKILL to the Python process inside the container: no
cleanup, like a real crash. Docker's restart policy (`restart: unless-stopped`
in compose.yaml) then starts it again, and it resumes from its last checkpoint.

Don't use `docker compose kill learner` for this: Docker treats that as you
stopping it on purpose, so it does **not** restart it.

### Take the coordinator down

```bash
docker compose kill coordinator      # SIGKILL: everything it held in memory is gone
# ... wait as long as you want the outage to be ...
docker compose start coordinator
```

The coordinator image has no shell, so it can't be crashed from inside like
the learner. `kill` then `start` does the same and lets you choose how long
it stays down. With the actors mid-rollout or waiting for weights, try
both an instant restart and ~40 s.

### Kill one actor

```bash
docker compose ps actor                            # names: sts2-actor-1, -2
docker kill sts2-actor-1
docker compose up -d --no-recreate actor           # bring it back
```

Until membership is in place, a dead actor stalls the round: the learner
waits for a rollout that never comes. Expect that.

### When is a good moment?

- **Mid-rollout:** a few seconds after a `loaded weights` line; actors are playing.
- **Between rounds:** right after `round for v... took`; actors are fetching.
- **While one actor waits:** one actor has submitted (`round 1/2`) and the
  other hasn't. This is where lost or stuck rollouts show up.

## 4. Measuring throughput

```bash
scripts/measure_throughput.sh 2 1 2 4     # 2 rounds each at 1, 2 and 4 actors
```

Full-length rounds; takes a while. Results land in `run/throughput/`. It
deletes the learner's checkpoints between counts.

## 5. Watch the policy play the real game

The trained policy can play the real, rendered game. The STS2MCP mod
(github.com/Gennadiyev/STS2MCP, MIT) runs inside the game and serves its
state and actions on localhost:15526; `play_live.py` reads the newest
checkpoint and plays through it with the same observation code as training.

One-time: build the mod against your game and install it.

```bash
scripts/build_sts2mcp.sh                  # -> run/sts2-mod/STS2_MCP.dll + STS2_MCP.json
GAME=~/.var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/"Slay the Spire 2"
mkdir -p "$GAME/mods" && cp run/sts2-mod/STS2_MCP.dll run/sts2-mod/STS2_MCP.json "$GAME/mods/"
```

Start the game from Steam. The first time, it asks whether to load mods:
accept, then in Settings > Mods make sure STS2 MCP is enabled, and restart the
game if it asks. Check the mod is up:

```bash
curl -s http://127.0.0.1:15526/api/v1/singleplayer | head -c 200; echo
```

Then, from the main menu (or mid-run):

```bash
docker compose run --rm live                       # one run as a random character, then stops
docker compose run --rm live --character Defect --runs 3 --delay 1 --greedy
```

The terminal shows each decision: what the policy saw (HP, energy, hand,
enemies and their intents, or the options on offer), what it chose, and the
value it expects from there.
Ctrl-C stops it; the game stays where it is. Training can keep running: the
demo only reads checkpoints, so restart it to pick up a newer one.

Screens the headless trainer never shows the policy are clicked through by
fixed rules in `sts2/live.py`: menus, claiming gold/potion/relic rewards,
treasure chests, Ancient dialogue, boss relic picks, the Crystal Sphere.

Mods don't affect the headless build: it copies only the game's own DLLs, and
the version pin checks release_info.json and sts2.dll, not mods/.
