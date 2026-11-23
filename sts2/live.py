"""Drive the real, rendered game through the STS2MCP mod's HTTP API, with the
same observation and action code the policy was trained with.

The policy was trained on the headless host's payloads (sts2/host.py). The mod
reports the same game in its own shapes, so LiveGame translates both ways:

    mod state  --to_payload()-->  host-style payload  --> Sts2Env --> policy
    policy's host-style command  --to_request()-->  mod action

Some screens exist in the real game but never reach the policy in training,
because the headless host resolves them itself: the rewards screen (gold,
potions and relics are claimed for you there), treasure chests, Ancient
dialogue, menus, the game-over screen. LiveGame clicks through those with
fixed rules, so the policy only sees decisions it was trained on.

Mod API (localhost:15526): GET /api/v1/singleplayer for the state, POST the
same path with {"action": ...}. See its docs/raw-full.md.
"""
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from sts2.vocab import CHARACTERS

URL = "http://127.0.0.1:15526/api/v1/singleplayer"
COMBAT_STATES = {"monster", "elite", "boss"}
ROOM_TYPES = {"monster": "Monster", "elite": "Elite", "boss": "Boss", "event": "Event", "shop": "Shop",
              "rest_site": "RestSite", "treasure": "Treasure", "map": "Map"}
REST_IDS = {"rest": "HEAL", "heal": "HEAL", "smith": "SMITH"}


def _int(text, default=0):
    """'2' -> 2, 'X' or None -> default."""
    match = re.search(r"-?\d+", str(text or ""))
    return int(match.group()) if match else default


def _stats_from_description(description):
    """The training payloads carry card numbers as stats; the mod only has text. Read the common ones back."""
    text = re.sub(r"\[[^\]]*\]", "", str(description or ""))
    stats = {}
    damage = re.search(r"Deal (\d+) damage", text)
    block = re.search(r"Gain (\d+) Block", text)
    if damage:
        stats["damage"] = int(damage.group(1))
    if block:
        stats["block"] = int(block.group(1))
    return stats


def _card(card, index=None):
    """A mod card object in the host's card shape."""
    return {
        "index": card.get("index", index),
        "name": card.get("name") or card.get("card_name"),
        "cost": _int(card.get("cost", card.get("card_cost"))),
        "star_cost": _int(card.get("star_cost", card.get("card_star_cost"))) or None,
        "type": card.get("type") or card.get("card_type"),
        "rarity": card.get("rarity") or card.get("card_rarity"),
        "target_type": card.get("target_type"),
        "can_play": card.get("can_play"),
        "upgraded": bool(card.get("is_upgraded")),
        "keywords": [k.get("name") for k in card.get("keywords") or [] if isinstance(k, dict)],
        "stats": _stats_from_description(card.get("description") or card.get("card_description")),
    }


def _intent(intent):
    """Mod intents carry the damage as a label like '11' or '5x2'."""
    out = {"type": intent.get("type")}
    if str(intent.get("type", "")).startswith("Attack") or intent.get("type") == "DeathBlow":
        numbers = [int(n) for n in re.findall(r"\d+", str(intent.get("label") or ""))]
        if numbers:
            out["damage"] = numbers[0]
            if len(numbers) > 1:
                out["hits"] = numbers[1]
                out["total_damage"] = numbers[0] * numbers[1]
    return out


class EventKeys:
    """Event options are numbered by their text key (NEOW.pages.INITIAL.options.X) in
    training; the mod gives the event id and the option's title. The text tables map
    one to the other."""

    def __init__(self, events_table=None):
        self.by_event = {}
        path = Path(events_table) if events_table else None
        if path and path.exists():
            table = json.loads(path.read_text(encoding="utf-8"))
            for key, title in table.items():
                if key.endswith(".title") and ".options." in key:
                    event = key.split(".", 1)[0]
                    self.by_event.setdefault(event, {})[str(title).strip().casefold()] = key[: -len(".title")]

    def key(self, event_id, title):
        return self.by_event.get(str(event_id or ""), {}).get(str(title or "").strip().casefold())


class LiveGame:
    def __init__(self, url=URL, delay=0.4, events_table=None, log=print):
        self.url, self.delay, self.log = url, delay, log
        self.events = EventKeys(events_table)
        self.deck = []          # last deck seen in combat (the mod only lists piles there)
        self.act_start = {}     # act -> the total floor count when it began
        self.translate = {}     # what the last payload's host-style indices mean to the mod
        self.picked = None      # character clicked on the character select screen

    # --- HTTP ------------------------------------------------------------------

    def state(self):
        with urllib.request.urlopen(self.url, timeout=10) as response:
            return json.loads(response.read())

    def post(self, body):
        request = urllib.request.Request(self.url, data=json.dumps(body).encode(), method="POST",
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                reply = json.loads(response.read())
        except urllib.error.HTTPError as e:
            reply = {"status": "error", "error": f"HTTP {e.code}: {e.read()[:200]!r}"}
        time.sleep(self.delay)
        return reply

    def wait_for_change(self, before, timeout=5.0):
        """Poll until the state differs from before: the game has applied the last action."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            now = self.state()
            if now != before:
                return now
            time.sleep(0.1)
        return before

    # --- screens the policy never sees --------------------------------------------

    def handle_automatically(self, s, character):
        """Click through a screen the headless host resolves on its own. Returns True if it did."""
        kind = s.get("state_type")
        if kind == "menu":
            options = [o if isinstance(o, str) else o.get("name") for o in s.get("options") or []]
            options = [o for o in options if o]
            screen = s.get("menu_screen")
            want = {"main": "singleplayer", "singleplayer": "standard", "tutorial_prompt": "no"}.get(screen)
            if screen == "character_select":
                # The state doesn't say which character is highlighted, so
                # remember our own click: pick the character, then embark.
                want = "embark" if self.picked == character else character.upper()
                self.picked = character
            else:
                self.picked = None
            if want is None or want not in [o.lower() for o in options] + [o.upper() for o in options]:
                want = "back" if "back" in options else (options[0] if options else None)
            if want:
                self.log(f"  [menu] {screen}: {want}")
                self.post({"action": "menu_select", "option": want})
            return True
        if kind == "game_over":
            self.post({"action": "menu_select", "option": "main_menu"})
            return True
        if kind == "rewards":
            items = (s.get("rewards") or {}).get("items") or []
            potions_full = len((s.get("player") or {}).get("potions") or []) >= \
                (s.get("player") or {}).get("max_potion_slots", 3)
            for item in items:
                if item.get("type") == "potion" and potions_full:
                    continue
                if item.get("type") in ("gold", "potion", "relic", "card", "special_card", "card_removal"):
                    self.log(f"  [rewards] claim {item.get('type')}")
                    self.post({"action": "claim_reward", "index": item["index"]})
                    return True
            self.post({"action": "proceed"})
            return True
        if kind == "treasure":
            relics = (s.get("treasure") or {}).get("relics") or []
            if relics:
                self.post({"action": "claim_treasure_relic", "index": relics[0]["index"]})
            elif (s.get("treasure") or {}).get("can_proceed"):
                self.post({"action": "proceed"})
            else:
                time.sleep(0.3)
            return True
        if kind == "event" and (s.get("event") or {}).get("in_dialogue"):
            self.post({"action": "advance_dialogue"})
            return True
        if kind == "rest_site":
            rest = s.get("rest_site") or {}
            if not any(o.get("is_enabled") for o in rest.get("options") or []) and rest.get("can_proceed"):
                self.post({"action": "proceed"})
                return True
        if kind == "relic_select":
            relics = (s.get("relic_select") or {}).get("relics") or []
            self.post({"action": "select_relic", "index": relics[0]["index"]} if relics
                      else {"action": "skip_relic_selection"})
            return True
        if kind == "crystal_sphere":
            sphere = s.get("crystal_sphere") or {}
            cells = sphere.get("clickable_cells") or []
            self.post({"action": "crystal_sphere_click_cell", "x": cells[0]["x"], "y": cells[0]["y"]} if cells
                      else {"action": "crystal_sphere_proceed"})
            return True
        if kind == "fake_merchant":
            self.post({"action": "proceed"})
            return True
        if kind in COMBAT_STATES:
            battle = s.get("battle") or {}
            if not battle.get("is_play_phase") or battle.get("turn") != "player":
                time.sleep(0.2)  # enemy turn or cards resolving: wait
                return True
        if kind in ("overlay", "unknown") or (kind == "shop" and (s.get("shop") or {}).get("error")):
            time.sleep(0.5)
            return True
        return False

    # --- mod state -> host-style payload ---------------------------------------------

    def _context(self, s, room):
        run = s.get("run") or {}
        act, total = int(run.get("act") or 1), int(run.get("floor") or 0)
        self.act_start.setdefault(act, max(total - 1, 0))
        return {"act": act, "floor": total - self.act_start[act], "room_type": room}

    def _player(self, s):
        p = s.get("player") or {}
        return {
            "hp": p.get("hp"), "max_hp": p.get("max_hp"), "block": p.get("block"), "gold": p.get("gold"),
            "relics": [{"name": r.get("name")} for r in p.get("relics") or []],
            "potions": [{"index": q.get("slot"), "name": q.get("name"), "target_type": q.get("target_type")}
                        for q in p.get("potions") or []],
            "deck": self.deck, "deck_size": len(self.deck),
        }

    def to_payload(self, s):
        """The host-style payload for a decision screen, or None if it isn't one."""
        kind = s.get("state_type")
        payload = {"type": "decision", "context": self._context(s, ROOM_TYPES.get(kind)), "player": self._player(s)}
        self.translate = {"kind": kind}
        p = s.get("player") or {}

        if kind in COMBAT_STATES:
            battle = s.get("battle") or {}
            hand = [_card(c, i) for i, c in enumerate(p.get("hand") or [])]
            piles = (p.get("hand") or []) + (p.get("draw_pile") or []) + (p.get("discard_pile") or []) + \
                (p.get("exhaust_pile") or [])
            self.deck = [_card(c) for c in piles]
            enemies = [e for e in battle.get("enemies") or [] if (e.get("hp") or 0) > 0]
            self.translate["targets"] = {i: e.get("entity_id") for i, e in enumerate(enemies)}
            pets = p.get("pets") or []
            payload.update({
                "decision": "combat_play", "round": battle.get("round"), "energy": p.get("energy"),
                "max_energy": p.get("max_energy"), "hand": hand,
                "enemies": [{"index": i, "name": e.get("name"), "hp": e.get("hp"), "max_hp": e.get("max_hp"),
                             "block": e.get("block"),
                             "intents": [_intent(x) for x in e.get("intents") or []],
                             "intends_attack": any(str(x.get("type", "")).startswith("Attack")
                                                   for x in e.get("intents") or []),
                             "powers": [{"name": w.get("name"), "amount": w.get("amount")}
                                        for w in e.get("status") or []]}
                            for i, e in enumerate(enemies)],
                "player_powers": [{"name": w.get("name"), "amount": w.get("amount")} for w in p.get("status") or []],
                "draw_pile_count": p.get("draw_pile_count"), "discard_pile_count": p.get("discard_pile_count"),
                "stars": p.get("stars"), "orb_slots": p.get("orb_slots"),
                "orbs": [{"name": o.get("name"), "passive": o.get("passive_val"), "evoke": o.get("evoke_val")}
                         for o in p.get("orbs") or []],
            })
            if pets:
                payload["osty"] = {k: pets[0].get(k) for k in ("hp", "max_hp", "block", "alive")}
            payload["player"]["deck"], payload["player"]["deck_size"] = self.deck, len(self.deck)
            return payload

        if kind == "map":
            options = (s.get("map") or {}).get("next_options") or []
            self.translate["nodes"] = {(o["col"], o["row"]): o["index"] for o in options}
            payload.update({"decision": "map_select",
                            "choices": [{"col": o["col"], "row": o["row"], "type": o.get("type")} for o in options]})
            return payload

        if kind == "card_reward":
            reward = s.get("card_reward") or {}
            payload.update({"decision": "card_reward", "cards": [_card(c) for c in reward.get("cards") or []],
                            "can_skip": reward.get("can_skip", True)})
            return payload

        if kind in ("card_select", "hand_select"):
            select = s.get(kind) or {}
            count = _int(re.search(r"\d+", select.get("prompt") or "") and
                         re.search(r"\d+", select.get("prompt") or "").group(), 1)
            payload.update({"decision": "card_select", "cards": [_card(c) for c in select.get("cards") or []],
                            "min_select": count, "max_select": count})
            return payload

        if kind == "bundle_select":
            bundles = (s.get("bundle_select") or {}).get("bundles") or []
            payload.update({"decision": "bundle_select",
                            "bundles": [{"index": b["index"], "cards": [_card(c) for c in b.get("cards") or []]}
                                        for b in bundles]})
            return payload

        if kind == "event":
            event = s.get("event") or {}
            payload.update({"decision": "event_choice", "event_name": event.get("event_name"),
                            "options": [{"index": o["index"], "title": o.get("relic_name") or o.get("title"),
                                         "text_key": self.events.key(event.get("event_id"), o.get("title")),
                                         "is_locked": o.get("is_locked")} for o in event.get("options") or []]})
            return payload

        if kind == "rest_site":
            options = (s.get("rest_site") or {}).get("options") or []
            payload.update({"decision": "rest_site",
                            "options": [{"index": o["index"],
                                         "option_id": REST_IDS.get(str(o.get("id")).lower(), str(o.get("id")).upper()),
                                         "is_enabled": o.get("is_enabled")} for o in options]})
            return payload

        if kind == "shop":
            items = (s.get("shop") or {}).get("items") or []
            by_kind = {"card": [], "relic": [], "potion": []}
            flat = {}
            removal = None
            for item in items:
                category = item.get("category")
                if category == "card_removal":
                    if item.get("is_stocked", True):
                        removal = item.get("price")
                        flat[("removal", 0)] = item["index"]
                    continue
                if category not in by_kind:
                    continue
                i = len(by_kind[category])
                flat[(category, i)] = item["index"]
                entry = {"index": i, "cost": item.get("price"), "is_stocked": item.get("is_stocked", True),
                         "on_sale": item.get("on_sale"),
                         "name": item.get(f"{category}_name")}
                if category == "card":
                    entry.update({k: v for k, v in _card(item).items() if k in ("type", "rarity", "stats", "keywords")})
                    entry["card_cost"] = _int(item.get("card_cost"))
                by_kind[category].append(entry)
            self.translate["shop"] = flat
            payload.update({"decision": "shop", "cards": by_kind["card"], "relics": by_kind["relic"],
                            "potions": by_kind["potion"], "card_removal_cost": removal})
            return payload
        return None

    # --- host-style command -> mod actions -------------------------------------------

    def to_requests(self, command):
        """The mod requests that carry out one host-style command, in order."""
        name, args = command.get("action"), command.get("args") or {}
        kind = self.translate.get("kind")
        targets = self.translate.get("targets", {})

        def target(index):
            return {"target": targets[index]} if index is not None and index in targets else {}

        if name == "play_card":
            return [{"action": "play_card", "card_index": args["card_index"], **target(args.get("target_index"))}]
        if name == "use_potion":
            return [{"action": "use_potion", "slot": args["potion_index"], **target(args.get("target_index"))}]
        if name == "discard_potion":
            return [{"action": "discard_potion", "slot": args["potion_index"]}]
        if name == "end_turn":
            return [{"action": "end_turn"}]
        if name == "select_map_node":
            return [{"action": "choose_map_node", "index": self.translate["nodes"][(args["col"], args["row"])]}]
        if name == "select_card_reward":
            return [{"action": "select_card_reward", "card_index": args["card_index"]}]
        if name == "skip_card_reward":
            return [{"action": "skip_card_reward"}]
        if name == "select_cards":
            picks = [int(i) for i in str(args["indices"]).split(",") if i != ""]
            if kind == "hand_select":
                return [{"action": "combat_select_card", "card_index": i} for i in picks] + \
                       [{"action": "combat_confirm_selection"}]
            return [{"action": "select_card", "index": i} for i in picks] + [{"action": "confirm_selection"}]
        if name == "skip_select":
            return [{"action": "cancel_selection"}]
        if name == "select_bundle":
            return [{"action": "select_bundle", "index": args["bundle_index"]}, {"action": "confirm_bundle_selection"}]
        if name == "choose_option":
            verb = "choose_rest_option" if kind == "rest_site" else "choose_event_option"
            return [{"action": verb, "index": args["option_index"]}]
        if name in ("buy_card", "buy_relic", "buy_potion"):
            category = name.split("_")[1]
            index = args[f"{category}_index"]
            return [{"action": "shop_purchase", "index": self.translate["shop"][(category, index)]}]
        if name == "remove_card":
            return [{"action": "shop_purchase", "index": self.translate["shop"][("removal", 0)]}]
        if name in ("leave_room", "proceed"):
            return [{"action": "proceed"}]
        raise ValueError(f"No mod action for {command}")

    @staticmethod
    def character_for(run_number):
        return CHARACTERS[run_number % len(CHARACTERS)]
