"""Integer ids for the names that appear in a game payload.

Each vocabulary is a fixed list of names numbered from 1, with 0 reserved for
anything the list does not have. The big ones (cards, relics, monsters, ...)
are generated from the game's text tables into vocab.json by
scripts/generate_sts2_vocab.py; the small enums (card types, map node types,
...) are listed here, since the text tables don't name them.

Ids are positions, so lists only ever grow at the end: reordering or removing
a name renumbers the rest and invalidates a trained policy.
"""
import json
from pathlib import Path

UNKNOWN_ID = 0


class Vocabulary:
    def __init__(self, label, names):
        self.label = label
        self.names = []
        self._ids = {}
        for name in names:
            key = self._key(name)
            if key not in self._ids:
                self.names.append(name)
                self._ids[key] = len(self.names)

    @staticmethod
    def _key(name):
        # Case- and spacing-insensitive: "DebuffStrong", "DEBUFF_STRONG" and
        # "debuff strong" are one name.
        return "".join(ch for ch in str(name).casefold() if ch.isalnum())

    def id_of(self, name):
        if name is None:
            return UNKNOWN_ID
        return self._ids.get(self._key(name), UNKNOWN_ID)

    def __contains__(self, name):
        return self.id_of(name) != UNKNOWN_ID

    @property
    def size(self):
        """Number of ids, counting the unknown id 0: what an embedding table needs."""
        return len(self.names) + 1


# The five playable characters, in the order the start_run command names them.
CHARACTERS = ["Ironclad", "Silent", "Defect", "Regent", "Necrobinder"]

# The decisions the host can ask for. "unknown" is its catch-all.
DECISIONS = ["combat_play", "map_select", "card_reward", "card_select", "bundle_select",
             "event_choice", "rest_site", "shop", "game_over", "unknown"]

SMALL = {
    "character": CHARACTERS,
    "decision": DECISIONS,
    "room_type": ["Monster", "Elite", "Boss", "Event", "Shop", "RestSite", "Treasure", "Map", "Ancient"],
    "card_type": ["Attack", "Skill", "Power", "Status", "Curse", "Quest"],
    "target_type": ["None", "Self", "AnyEnemy", "AllEnemies", "RandomEnemy", "AnyAlly", "AllAllies",
                    "AnyPlayer", "Osty", "TargetedNoCreature"],
    "rarity": ["Basic", "Common", "Uncommon", "Rare", "Special", "Curse", "Status", "Token", "Event", "Ancient"],
    "map_node": ["Monster", "Elite", "Boss", "Unknown", "Shop", "RestSite", "Treasure", "Ancient"],
    "intent": ["Attack", "Buff", "CardDebuff", "DeathBlow", "Debuff", "DebuffStrong", "Defend", "Escape",
               "Heal", "Sleep", "Status", "Stun", "Summon", "Unknown", "Hidden"],
    "rest_option": ["HEAL", "SMITH", "LIFT", "TOKE", "DIG", "RECALL", "COOK", "CLEAR", "MEND", "HATCH"],
    # What one row of the options list is.
    "option_kind": ["map_node", "reward_card", "select_card", "bundle", "event_option", "rest_option",
                    "shop_card", "shop_relic", "shop_potion", "shop_removal"],
}

# Keywords that change how a card plays, as yes/no features.
KEYWORDS = ["Exhaust", "Ethereal", "Retain", "Innate", "Unplayable", "Sly", "Eternal"]


def _load():
    generated = json.loads((Path(__file__).parent / "vocab.json").read_text(encoding="utf-8"))
    all_lists = {**generated, **SMALL}
    return {label: Vocabulary(label, names) for label, names in all_lists.items()}


VOCABS = _load()
