# Game-state fixtures

Communication Mod payloads, exactly as they arrive over the socket: the outer
envelope (`available_commands`, `ready_for_command`, `in_game`) wrapping a
`game_state`. They let the observation pipeline, the action mask and the reward
function be exercised with no copy of Slay the Spire running.

| Fixture | `screen_type` | Covers |
| --- | --- | --- |
| `combat_with_monsters.json` | `NONE` | Mid-combat: two live monsters with powers and intents, a seven-card hand holding an X-cost card and an unplayable curse, player block and a debuff |
| `combat_reward.json` | `COMBAT_REWARD` | The gold/potion/card rewards after that same combat. Pairs with the fixture above as a combat-ending transition |
| `card_reward.json` | `CARD_REWARD` | Three cards on offer, skip available |
| `map_screen.json` | `MAP` | Current node and the nodes it connects to |
| `shop_screen.json` | `SHOP_SCREEN` | Cards, relics and potions with prices, plus a purge on offer |
| `rest_site.json` | `REST` | Rest and smith on offer, not yet rested |
| `game_over.json` | `GAME_OVER` | A death on floor 14 |
| `main_menu.json` | - | Before a run starts: no `game_state` key at all |

## Provenance

**These are hand-built to the Communication Mod's schema, not recorded from a
live game.** The agent's own capture path exists (see below) but needs the game
running, which was not available when the fixtures were written. Field names,
nesting and value types follow the mod's output; the values are a plausible
Ironclad run rather than a real one.

That is good enough for what tickets 03-07 need - shapes, bounds, screen
dispatch and mask legality do not care whose run it was - but if a real capture
ever contradicts one of these files, the real capture wins. Replacing any of
them with a genuine capture of the same screen is strictly an improvement and
should need no test changes.

One value to confirm against a real capture: **card `cost`**. These fixtures
use the game's own encoding, where an X-cost card reports `-1` and an
unplayable card reports `-2`. The observation code instead tests for `None` and
the string `"X"` (`observations/hand_observations.py`, `deck_observations.py`),
so on this reading both of its special cases are dead and every such card falls
through to `float(cost)`. Tickets 03 and 05 should settle it against a capture
before re-encoding anything.

Other deliberate choices worth knowing about:

- Multi-word card names (`Body Slam`, `A Thousand Cuts`, `The Bomb`, `Shrug It
  Off`) appear throughout, because ticket 03 is about names that collide on
  their first word.
- The combat hand carries an X-cost `Whirlwind` and an unplayable `Clumsy`, so
  the negative sentinel costs ticket 05 must re-encode appear in live data and
  not only in the deck, which is dead until ticket 04 lands.
- Every fixture carries a full 57-node act map spanning `y` 0 to 14. The `map`
  component declares a high of 10, so a stubby map would have hidden a bound
  ticket 05 has to fix.
- Relic counters include `-1` (the mod's "no counter" value) and a real count,
  because `-1` violates the declared lower bound today.
- The potion list mixes a targeted potion, an untargeted one and an empty slot,
  which are the three cases the action mask branches on.
- In the combat fixture the hand, draw pile and discard pile add up to exactly
  the deck, as they would in a real run.
- `combat_with_monsters` and `combat_reward` share a floor number so they read
  as one transition.

Every card, relic, monster and potion name here is in the hard-coded
vocabulary, so the fixtures never take the unknown-name path. That path is
ticket 03's, and `test_an_unlisted_card_name_does_not_crash_the_pipeline`
covers it by editing a fixture in memory - committing a fixture that crashes
the pipeline would take the whole suite down with it.

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
