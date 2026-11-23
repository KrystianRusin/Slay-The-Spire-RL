# Slay the Spire 2 PPO Agent

A reinforcement learning agent that plays *Slay the Spire 2*, trained with
Maskable PPO across many game instances at once. One policy plays all five
characters (Ironclad, Silent, Defect, Regent, Necrobinder).

The original *Slay the Spire* (STS1) version, which played through
CommunicationMod and a rendered game under Xvfb, is on the
[`legacy/sts-1`](../../tree/legacy/sts-1) branch.

## How it works

```
 actor containers (x N)                         learner              coordinator (Go)
 ┌───────────────────────────────┐   rollouts   ┌───────────┐ weights ┌───────────┐
 │ headless STS2 ── JSON lines ──│─── gRPC ────>│ PPO update│───────> │ latest     │
 │ (sts2-cli host)   policy      │              └───────────┘         │ version    │
 │                 <─────────────│──────────── new weights ───────────│            │
 └───────────────────────────────┘                                    └───────────┘
```

- **The game runs headless.** [sts2-cli](https://github.com/wuhao21/sts2-cli)
  (MIT) loads the game's own logic (`sts2.dll`) in a .NET process with the
  Godot engine stubbed out: no rendering, no Steam client, no network. It talks
  JSON over stdin/stdout, about 170 decisions per second per game, in ~65 MB.
  It's built from your installed copy of the game; no game files are in any
  image (`docker/sts2/`).
- **Actors** (`run_actor.py`, `environment/run_env.py`) play runs with the
  current policy and send each 2048-step rollout to the learner.
- **The learner** (`learner_server.py`) trains on one rollout per actor, then
  publishes the new weights to the **coordinator** (`coordinator/`, Go), which
  hands them to the actors.
- **To watch it play**, `play_live.py` drives the real, rendered game through
  the [STS2MCP](https://github.com/Gennadiyev/STS2MCP) mod (MIT) with the same
  observation code as training.

## The environment (`sts2/`)

**Observations** (`sts2/env.py`): fixed-size groups of rows, each row with id
columns and numeric columns: the player and run (`global`), the hand, enemies
(with intents and powers), player powers, Defect's orbs, potions, relics, the
deck, and the options of the current non-combat decision (map nodes, reward
cards, shop items, event and rest options). Character-specific mechanics have
their own features (orbs, Regent's stars, Necrobinder's Osty), zero for the
other characters.

**Network** (`sts2/features.py`): every id (card, monster, relic, power...)
goes through a learned embedding shared everywhere that vocabulary appears;
each row becomes a fixed-width vector; slot-ordered groups (hand, enemies,
options) stay in order, because actions point at slots, and sets (deck,
relics, powers) are mean- and max-pooled. Then 256x256 policy and value heads.

**Vocabularies** (`sts2/vocab.json`): generated from the game's own text
tables by `scripts/generate_sts2_vocab.py`, so every card, relic, potion,
power, monster, orb, enchantment, affliction and event option has an id.
Append-only, so a game patch adds names without renumbering a trained policy.

**Actions**: one flat, masked discrete space of 148: end turn, play hand card
h at target t, use or discard potion p, choose option k, skip/leave, confirm.
Multi-card selections are one pick per step. The mask comes from the payload
(`can_play`, reachable map nodes, locked options, affordability); the game
refuses about 1 action in 3,000, and a refused action is retried without it.

**Rewards**: +1 per floor, +3 per act, +10 for a win, -1 for dying; +1 per
fight won and the share of enemy HP removed; -3x the share of HP lost, +1x
healed; -0.005 per action.

## Running it

Everything runs in Docker. See [docs/runbook.md](docs/runbook.md) for setup,
tests, training, fault tests and watching the policy play the real game. In
short:

```bash
mkdir -p run/sts2-host run/sts2-random run/sts2-actors run/sts2-learner
docker compose run --rm --build sts2-build              # build the headless game from your install
NUM_ACTORS=24 docker compose up -d --build learner actor
docker compose run --rm live                            # watch it play (needs the mod, see the runbook)
```

Requires *Slay the Spire 2* on Steam (Linux build, pinned to v0.111.1 in
`docker/sts2/game-version.sh`) and Docker.
