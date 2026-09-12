# Game-state fixtures

Communication Mod payloads, exactly as they arrive over the socket: the outer
envelope (`available_commands`, `ready_for_command`, `in_game`) wrapping a
`game_state`. They let the observation pipeline, the action mask and the reward
function be exercised with no copy of Slay the Spire running.

| Fixture | `screen_type` | Contains |
| --- | --- | --- |
| `combat_with_monsters.json` | `NONE` | Mid-combat: two live monsters with powers and intents, a seven-card hand holding an X-cost card and an unplayable curse, player block and a debuff |
| `combat_reward.json` | `COMBAT_REWARD` | The gold/potion/card rewards after that same combat, on the same floor |
| `card_reward.json` | `CARD_REWARD` | Three cards on offer, skip available |
| `map_screen.json` | `MAP` | Current node and the nodes it connects to |
| `shop_screen.json` | `SHOP_SCREEN` | Cards, relics and potions with prices, plus a purge on offer |
| `rest_site.json` | `REST` | Rest and smith on offer, not yet rested |
| `game_over.json` | `GAME_OVER` | A death on floor 14 |
| `main_menu.json` | - | Before a run starts: no `game_state` key at all |

Every fixture carries a full 57-node act map spanning `y` 0 to 14, and a deck
the combat piles add up to exactly.

## Provenance

**These are hand-built to the Communication Mod's schema, not recorded from a
live game.** Field names, nesting and value types follow the mod's output; the
values are a plausible Ironclad run rather than a real one. If a real capture
ever contradicts one of these files, the real capture wins - replacing any of
them with a genuine capture of the same screen should need no test changes.

One value to confirm against a capture: **card `cost`**. These use the game's
own encoding, where an X-cost card reports `-1` and an unplayable card reports
`-2`, while `encode_cost` in `observations/observation_processing.py` tests for
`None` and the string `"X"`.

## Capturing more

`middleman_process.py` writes every state it forwards when `STS_CAPTURE_DIR` is
set:

```bash
export STS_CAPTURE_DIR=/tmp/sts-captures   # PowerShell: $env:STS_CAPTURE_DIR = "C:\sts-captures"
```

Then start the game with the Communication Mod pointed at the middleman as
usual and play through the screens you want. Files are named
`<screen_type>_<timestamp>_<sequence>.json`, so a directory listing shows which
screens you have hit. Copy the ones you want into this directory, give them a
descriptive name, and add the name to `SCREEN_COVERAGE` in `tests/conftest.py`
(`tests/test_fixtures.py` fails if a fixture is added without it).

Captures contain nothing but game state - no account or system information.
