"""The Slay the Spire 2 environment: what the policy sees, what it can do, and
what it is rewarded for.

The game (the headless host, or the real game through an adapter) sends one
decision payload at a time, e.g. {"decision": "combat_play", "hand": [...],
"enemies": [...], "player": {...}}. This module turns a payload into:

- an observation: a dict of fixed-size arrays (see GROUPS), the same shape for
  every decision, with absent things left as zero rows;
- an action mask: which of the N_ACTIONS slots are legal right now;
- for each legal slot, the command to send back to the host;
- after the next payload arrives, the reward the action earned.

One policy plays all five characters. The character is part of the
observation, and character-specific mechanics (Defect's orbs, Regent's stars,
Necrobinder's Osty) have their own features, zero for everyone else.

The action space is flat and fixed:
    0           end turn
    1..60       play hand card h (0-9) at target t: t=0 for no target, t=1..5 for enemy t-1
    61..90      use potion in slot p (0-4) at target t (same targets)
    91..95      discard the potion in slot p
    96..145     choose option k (0-49) of the current non-combat decision: a map
                node, a reward card, a card to select, an event or rest option,
                a shop item
    146         skip / leave / proceed (skip a card reward, leave a shop, ...)
    147         confirm the cards selected so far
Picking several cards ("choose 2 of these") is one pick per step: each option
pick adds a card, and the selection is sent once it is full or confirmed.
"""
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sts2.vocab import KEYWORDS, VOCABS

MAX_HAND = 10
MAX_ENEMIES = 5
MAX_POTIONS = 5
MAX_OPTIONS = 50
MAX_DECK = 60
MAX_RELICS = 30
MAX_PLAYER_POWERS = 16
ENEMY_POWERS = 6
ENEMY_INTENTS = 2
MAX_ORBS = 10
TARGETS = 1 + MAX_ENEMIES  # "no target" + one per enemy slot

END_TURN = 0
PLAY_BASE = 1
POTION_BASE = PLAY_BASE + MAX_HAND * TARGETS
DISCARD_BASE = POTION_BASE + MAX_POTIONS * TARGETS
OPTION_BASE = DISCARD_BASE + MAX_POTIONS
SKIP = OPTION_BASE + MAX_OPTIONS
CONFIRM = SKIP + 1
N_ACTIONS = CONFIRM + 1

ENEMY_TARGETED = {"AnyEnemy"}


@dataclass(frozen=True)
class Group:
    """One observation component: `rows` rows, each with id columns and numeric columns.

    It becomes two arrays, <name>_ids (rows x len(ids), each column numbered by
    the vocabulary named in ids) and <name>_num (rows x num). Numeric column 0
    is always "this row is present". pooled groups are sets whose order means
    nothing (the deck, relics); the rest keep their slot order, because
    actions refer to slots (hand card 3, enemy 1, option 7).
    """
    rows: int
    ids: tuple
    num: tuple
    pooled: bool = False
    width: int = 32  # size each row is encoded to by the network

    @property
    def n_num(self):
        return len(self.num)


CARD_NUM = ("present", "cost", "star_cost", "upgraded", "damage", "block", "other_stat",
            *[f"kw_{k.lower()}" for k in KEYWORDS])

GROUPS = {
    "global": Group(1, ("character", "decision", "room_type"), (
        "present", "hp_frac", "hp", "max_hp", "block", "energy", "max_energy", "gold", "act", "floor",
        "ascension", "round", "draw_pile", "discard_pile", "deck_size", "stars", "orb_slots",
        "osty_present", "osty_alive", "osty_hp_frac", "osty_block", "incoming_damage", "enemies",
        "potions", "select_count", "select_min", "select_max", "can_skip", "options"), width=64),
    "hand": Group(MAX_HAND, ("card", "card_type", "target_type", "enchantment", "affliction"),
                  (*CARD_NUM, "can_play", "enchant_amount", "affliction_amount",
                   *[f"damage_vs_{t}" for t in range(MAX_ENEMIES)]), width=48),
    "enemies": Group(MAX_ENEMIES, ("monster", *["intent"] * ENEMY_INTENTS, *["power"] * ENEMY_POWERS),
                     ("present", "hp_frac", "hp", "max_hp", "block", "intends_attack", "intent_damage",
                      "intent_hits", *[f"power_amount_{i}" for i in range(ENEMY_POWERS)]), width=48),
    "player_powers": Group(MAX_PLAYER_POWERS, ("power",), ("present", "amount"), pooled=True),
    "orbs": Group(MAX_ORBS, ("orb",), ("present", "passive", "evoke"), width=16),
    "potions": Group(MAX_POTIONS, ("potion", "target_type"), ("present",), width=16),
    "relics": Group(MAX_RELICS, ("relic",), ("present",), pooled=True),
    "deck": Group(MAX_DECK, ("card", "card_type", "enchantment"), CARD_NUM, pooled=True, width=48),
    "options": Group(MAX_OPTIONS, ("option_kind", "card", "card_type", "relic", "potion", "map_node",
                                   "event_option", "rest_option"),
                     ("present", "enabled", "price", *CARD_NUM[1:], "selected", "on_sale", "col", "row")),
}


def _f(value, scale=1.0):
    """value / scale as a float, with None and junk as 0."""
    try:
        return float(value or 0) / scale
    except (TypeError, ValueError):
        return 0.0


def _vid(vocab, name):
    return VOCABS[vocab].id_of(name)


def _card_name(card):
    # Upgraded cards may carry a trailing "+"; the vocabulary has base names.
    return str(card.get("name") or "").rstrip("+").strip()


def card_features(card):
    """The numeric features shared by every place a card appears (CARD_NUM)."""
    stats = card.get("stats") or {}
    damage = stats.get("damage", stats.get("ostydamage", 0))
    block = stats.get("block", 0)
    other = sum(_f(v) for k, v in stats.items() if k not in ("damage", "ostydamage", "block"))
    keywords = {str(k).casefold() for k in card.get("keywords") or []}
    cost = card.get("card_cost", card.get("cost"))  # shop entries: "cost" is the gold price
    upgraded = bool(card.get("upgraded")) or str(card.get("name") or "").endswith("+")
    return [1.0, _f(cost, 5), _f(card.get("star_cost"), 5), float(upgraded), _f(damage, 30),
            _f(block, 30), _f(other, 10), *[float(k.casefold() in keywords) for k in KEYWORDS]]


class Option:
    """One row of the options list: its features and the command that picks it."""

    def __init__(self, kind, command, enabled=True, card=None, relic=None, potion=None, map_node=None,
                 event_option=None, rest_option=None, price=0, col=0, row=0, on_sale=False, select_index=None):
        self.kind, self.command, self.enabled = kind, command, enabled
        self.card, self.select_index = card, select_index
        self.ids = [_vid("option_kind", kind),
                    _vid("card", _card_name(card)) if card else 0,
                    _vid("card_type", card.get("type")) if card else 0,
                    _vid("relic", relic), _vid("potion", potion), _vid("map_node", map_node),
                    _vid("event_option", event_option), _vid("rest_option", rest_option)]
        card_num = card_features(card)[1:] if card else [0.0] * (len(CARD_NUM) - 1)
        self.num = [1.0, float(enabled), _f(price, 300), *card_num, 0.0, float(bool(on_sale)),
                    _f(col, 7), _f(row, 20)]


def action(name, **args):
    command = {"cmd": "action", "action": name}
    if args:
        command["args"] = args
    return command


def build_options(state):
    """The rows of the options list for a non-combat decision."""
    decision = state.get("decision")
    gold = (state.get("player") or {}).get("gold", 0) or 0
    options = []
    if decision == "map_select":
        for c in state.get("choices") or []:
            options.append(Option("map_node", action("select_map_node", col=c["col"], row=c["row"]),
                                  map_node=c.get("type"), col=c.get("col"), row=c.get("row")))
    elif decision == "card_reward":
        for c in state.get("cards") or []:
            options.append(Option("reward_card", action("select_card_reward", card_index=c["index"]), card=c))
    elif decision == "card_select":
        for c in state.get("cards") or []:
            options.append(Option("select_card", None, card=c, select_index=c["index"]))
    elif decision == "bundle_select":
        for i, bundle in enumerate(state.get("bundles") or []):
            cards = bundle.get("cards") or [{}]
            options.append(Option("bundle", action("select_bundle", bundle_index=bundle.get("index", i)),
                                  card=cards[0]))
    elif decision == "event_choice":
        for o in state.get("options") or []:
            # Ancient (Neow-style) options are relics; the title names it.
            options.append(Option("event_option", action("choose_option", option_index=o["index"]),
                                  enabled=not o.get("is_locked"), event_option=o.get("text_key"),
                                  relic=o.get("title")))
    elif decision == "rest_site":
        for o in state.get("options") or []:
            options.append(Option("rest_option", action("choose_option", option_index=o["index"]),
                                  enabled=o.get("is_enabled", True), rest_option=o.get("option_id")))
    elif decision == "shop":
        def affordable(item):
            return bool(item.get("is_stocked", True)) and (item.get("cost") or 0) <= gold
        for c in state.get("cards") or []:
            options.append(Option("shop_card", action("buy_card", card_index=c["index"]), card=c,
                                  enabled=affordable(c), price=c.get("cost"), on_sale=c.get("on_sale")))
        for r in state.get("relics") or []:
            options.append(Option("shop_relic", action("buy_relic", relic_index=r["index"]), relic=r.get("name"),
                                  enabled=affordable(r), price=r.get("cost")))
        for p in state.get("potions") or []:
            options.append(Option("shop_potion", action("buy_potion", potion_index=p["index"]),
                                  potion=p.get("name"), enabled=affordable(p), price=p.get("cost")))
        removal = state.get("card_removal_cost")
        if removal is not None:
            options.append(Option("shop_removal", action("remove_card"), enabled=removal <= gold, price=removal))
    return options[:MAX_OPTIONS]


def live_enemies(state):
    return [e for e in state.get("enemies") or [] if (e.get("hp") or 0) > 0][:MAX_ENEMIES]


def progress(state):
    """(act, floor) the payload is at: from its context, else the top level (some payloads have both)."""
    context = state.get("context") or {}
    return (int(context.get("act") or state.get("act") or 0), int(context.get("floor") or state.get("floor") or 0))


class Sts2Env(gym.Env):
    """Encodes payloads and translates actions. Never steps the game itself:
    the actor loop sends the commands and feeds the replies back in."""

    def __init__(self, character="Ironclad", ascension=0):
        super().__init__()
        self.action_space = spaces.Discrete(N_ACTIONS)
        space = {}
        for name, group in GROUPS.items():
            id_high = np.array([[VOCABS[v].size - 1 for v in group.ids]] * group.rows, dtype=np.float32)
            space[f"{name}_ids"] = spaces.Box(0, id_high, shape=(group.rows, len(group.ids)), dtype=np.float32)
            space[f"{name}_num"] = spaces.Box(-np.inf, np.inf, shape=(group.rows, group.n_num), dtype=np.float32)
        self.observation_space = spaces.Dict(space)
        self.begin_run(character, ascension)

    # --- run state the payloads don't carry -----------------------------------

    def begin_run(self, character, ascension=0):
        self.character, self.ascension = character, ascension
        self.state = None
        self._new_decision()

    def _new_decision(self):
        """A new payload arrived: forget the selection and refusals of the last one."""
        self.selected = []   # card_select: indices picked so far
        self.refused = set()  # actions the host refused for this payload
        self._options = None

    def update_game_state(self, state):
        self.state = state
        self._new_decision()

    @property
    def options(self):
        if self._options is None:
            self._options = build_options(self.state or {})
        return self._options

    # --- observation ------------------------------------------------------------

    def observe(self, state=None):
        state = self.state if state is None else state
        player = state.get("player") or {}
        decision = state.get("decision")
        obs = {}

        def put(name, rows):
            group = GROUPS[name]
            ids = np.zeros((group.rows, len(group.ids)), dtype=np.float32)
            num = np.zeros((group.rows, group.n_num), dtype=np.float32)
            for i, (row_ids, row_num) in enumerate(rows[:group.rows]):
                ids[i], num[i] = row_ids, row_num
            obs[f"{name}_ids"], obs[f"{name}_num"] = ids, num

        enemies = live_enemies(state) if decision == "combat_play" else []
        osty = state.get("osty")
        incoming = sum(_f(intent.get("total_damage", intent.get("damage")))
                       for e in enemies for intent in e.get("intents") or [])
        act, floor = progress(state)
        max_hp = max(_f(player.get("max_hp")), 1.0)
        select = decision == "card_select"
        put("global", [([_vid("character", self.character), _vid("decision", decision),
                         _vid("room_type", (state.get("context") or {}).get("room_type"))],
                        [1.0, _f(player.get("hp")) / max_hp, _f(player.get("hp"), 100), _f(player.get("max_hp"), 100),
                         _f(player.get("block"), 30), _f(state.get("energy"), 5), _f(state.get("max_energy"), 5),
                         _f(player.get("gold"), 300), _f(act, 4), _f(floor, 20), _f(self.ascension, 20),
                         _f(state.get("round"), 20), _f(state.get("draw_pile_count"), 30),
                         _f(state.get("discard_pile_count"), 30), _f(player.get("deck_size"), 40),
                         _f(state.get("stars"), 10), _f(state.get("orb_slots"), 10),
                         float(osty is not None), float(bool(osty and osty.get("alive"))),
                         _f(osty.get("hp")) / max(_f(osty.get("max_hp")), 1.0) if osty else 0.0,
                         _f(osty.get("block"), 20) if osty else 0.0, _f(incoming, 50), _f(len(enemies), 5),
                         _f(len(player.get("potions") or []), 5), _f(len(self.selected) if select else 0, 5),
                         _f(state.get("min_select") if select else 0, 5),
                         _f(state.get("max_select") if select else 0, 5),
                         float(self._skip_command() is not None), _f(len(self.options), MAX_OPTIONS)])])

        hand = (state.get("hand") or []) if decision == "combat_play" else []
        enemy_slot = {e.get("index"): slot for slot, e in enumerate(enemies)}
        rows = []
        for card in hand:
            vs = [0.0] * MAX_ENEMIES
            for d in card.get("damage_by_target") or []:
                slot = enemy_slot.get(d.get("target_index"))
                if slot is not None:
                    vs[slot] = _f(d.get("total_damage", d.get("damage")), 30)
            rows.append(([_vid("card", _card_name(card)), _vid("card_type", card.get("type")),
                          _vid("target_type", card.get("target_type")), _vid("enchantment", card.get("enchantment")),
                          _vid("affliction", card.get("affliction"))],
                         [*card_features(card), float(bool(card.get("can_play"))),
                          _f(card.get("enchantment_amount"), 10), _f(card.get("affliction_amount"), 10), *vs]))
        put("hand", rows)

        rows = []
        for e in enemies:
            intents = (e.get("intents") or [])[:ENEMY_INTENTS]
            powers = (e.get("powers") or [])[:ENEMY_POWERS]
            damage = sum(_f(i.get("total_damage", i.get("damage"))) for i in intents)
            hits = max([_f(i.get("hits") or (1 if i.get("damage") else 0)) for i in intents] or [0])
            rows.append(([_vid("monster", e.get("name")),
                          *[_vid("intent", i.get("type")) for i in intents] + [0] * (ENEMY_INTENTS - len(intents)),
                          *[_vid("power", p.get("name")) for p in powers] + [0] * (ENEMY_POWERS - len(powers))],
                         [1.0, _f(e.get("hp")) / max(_f(e.get("max_hp")), 1.0), _f(e.get("hp"), 100),
                          _f(e.get("max_hp"), 100), _f(e.get("block"), 30), float(bool(e.get("intends_attack"))),
                          _f(damage, 30), _f(hits, 5),
                          *[_f(p.get("amount"), 10) for p in powers] + [0.0] * (ENEMY_POWERS - len(powers))]))
        put("enemies", rows)

        put("player_powers", [([_vid("power", p.get("name"))], [1.0, _f(p.get("amount"), 10)])
                              for p in state.get("player_powers") or []])
        put("orbs", [([_vid("orb", o.get("name"))], [1.0, _f(o.get("passive"), 10), _f(o.get("evoke"), 10)])
                     for o in state.get("orbs") or []])

        rows = [([0, 0], [0.0])] * MAX_POTIONS
        for p in player.get("potions") or []:
            slot = p.get("index", 0)
            if 0 <= slot < MAX_POTIONS:
                rows[slot] = ([_vid("potion", p.get("name")), _vid("target_type", p.get("target_type"))], [1.0])
        put("potions", rows)

        put("relics", [([_vid("relic", r.get("name"))], [1.0]) for r in player.get("relics") or []])
        put("deck", [([_vid("card", _card_name(c)), _vid("card_type", c.get("type")),
                       _vid("enchantment", c.get("enchantment"))], card_features(c))
                     for c in player.get("deck") or []])

        rows = []
        for option in self.options:
            num = list(option.num)
            num[GROUPS["options"].num.index("selected")] = float(option.select_index in self.selected)
            rows.append((option.ids, num))
        put("options", rows)
        return obs

    # --- actions ----------------------------------------------------------------

    def _skip_command(self):
        state = self.state or {}
        decision = state.get("decision")
        if decision == "card_reward":
            return action("skip_card_reward") if state.get("can_skip", True) else None
        if decision == "card_select":
            return action("skip_select") if (state.get("min_select") or 0) == 0 and not self.selected else None
        if decision == "shop":
            return action("leave_room")
        if decision in ("event_choice", "rest_site"):
            # Only when nothing else is possible: leaving early skips the room.
            return action("leave_room") if not any(o.enabled for o in self.options) else None
        if decision in ("unknown", None):
            return action("proceed")
        return None

    def commands(self):
        """{action index: command} for every legal action in the current payload.

        A command of None means the action is handled here, without the host
        (adding a card to a selection that isn't full yet).
        """
        state = self.state or {}
        decision = state.get("decision")
        legal = {}
        if decision == "combat_play":
            legal[END_TURN] = action("end_turn")
            enemies = live_enemies(state)
            targets = range(1, len(enemies) + 1)
            for h, card in enumerate((state.get("hand") or [])[:MAX_HAND]):
                if not card.get("can_play"):
                    continue
                if card.get("target_type") in ENEMY_TARGETED:
                    for t in targets:
                        legal[PLAY_BASE + h * TARGETS + t] = action(
                            "play_card", card_index=card["index"], target_index=enemies[t - 1]["index"])
                else:
                    legal[PLAY_BASE + h * TARGETS] = action("play_card", card_index=card["index"])
            for p in (state.get("player") or {}).get("potions") or []:
                slot = p.get("index", 0)
                if not 0 <= slot < MAX_POTIONS:
                    continue
                if p.get("target_type") in ENEMY_TARGETED:
                    for t in targets:
                        legal[POTION_BASE + slot * TARGETS + t] = action(
                            "use_potion", potion_index=slot, target_index=enemies[t - 1]["index"])
                else:
                    legal[POTION_BASE + slot * TARGETS] = action("use_potion", potion_index=slot)
                legal[DISCARD_BASE + slot] = action("discard_potion", potion_index=slot)
        elif decision == "card_select":
            high = min(state.get("max_select") or 1, len(self.options))
            low = state.get("min_select") or 0
            for k, option in enumerate(self.options):
                if option.select_index in self.selected:
                    continue
                if len(self.selected) + 1 >= high:
                    picked = sorted(self.selected + [option.select_index])
                    legal[OPTION_BASE + k] = action("select_cards", indices=",".join(map(str, picked)))
                else:
                    legal[OPTION_BASE + k] = None
            if self.selected and len(self.selected) >= low:
                legal[CONFIRM] = action("select_cards", indices=",".join(map(str, sorted(self.selected))))
        else:
            for k, option in enumerate(self.options):
                if option.enabled:
                    legal[OPTION_BASE + k] = option.command
        skip = self._skip_command()
        if skip is not None:
            legal[SKIP] = skip
        for refused in self.refused:
            legal.pop(refused, None)
        if not legal:
            # Everything listed was refused (or nothing was listed): "proceed"
            # is the host's generic way past a screen.
            legal[SKIP] = action("proceed")
        return legal

    def action_mask(self):
        mask = np.zeros(N_ACTIONS, dtype=bool)
        mask[list(self.commands())] = True
        return mask

    def apply_internal(self, action_index):
        """Carry out an action that needs no host command (a selection pick)."""
        option = self.options[action_index - OPTION_BASE]
        self.selected.append(option.select_index)

    # --- reward -------------------------------------------------------------------

    def calculate_reward(self, previous, current):
        """Reward for the action that took the game from payload previous to current.

        Mostly progress, with combat shaping so a fresh policy has something
        to climb before it ever clears a floor:
          +1 per floor climbed, +3 per act cleared, +10 for a win, -1 for dying
          +1 for winning a fight, + the share of enemy HP removed
          -3 x the share of max HP lost, +1 x the share healed
          -0.005 per action, so dithering costs something
        """
        reward = -0.005
        if previous is None or current is None:
            return reward
        (prev_act, prev_floor), (act, floor) = progress(previous), progress(current)
        if act > prev_act:
            reward += 3.0
        elif act == prev_act and floor > prev_floor:
            reward += float(floor - prev_floor)

        prev_player, player = previous.get("player") or {}, current.get("player") or {}
        max_hp = max(_f(player.get("max_hp")), 1.0)
        if prev_player and player:
            delta = _f(player.get("hp")) - _f(prev_player.get("hp"))
            reward += 3.0 * delta / max_hp if delta < 0 else 1.0 * delta / max_hp

        was_combat = previous.get("decision") == "combat_play"
        if was_combat and current.get("decision") == "combat_play" and (act, floor) == (prev_act, prev_floor):
            before = live_enemies(previous)
            removed = sum(_f(e.get("hp")) for e in before) - sum(_f(e.get("hp")) for e in live_enemies(current))
            total = sum(_f(e.get("max_hp")) for e in before)
            if removed > 0 and total > 0:
                reward += removed / total
        if was_combat and current.get("decision") not in ("combat_play", "game_over"):
            reward += 1.0

        if current.get("decision") == "game_over":
            reward += 10.0 if current.get("victory") else -1.0
        return reward

    @staticmethod
    def is_done(state):
        return (state or {}).get("decision") == "game_over"
