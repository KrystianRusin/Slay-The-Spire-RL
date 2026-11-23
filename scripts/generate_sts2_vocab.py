"""Build sts2/vocab.json, the name lists the observation encoder numbers, from
the game's own text tables.

The headless host reports cards, relics, potions, powers, monsters, orbs,
enchantments and afflictions by their English display name, and event options
by their text key. Those names come from the localization tables the host
build extracts from the game (run/sts2-host/sts2-cli/localization_eng/), so the
same tables give us every name the game can send, without playing to find them.

Ids are positions in each list (see sts2/vocab.py), and a trained policy has
learned what each position means. So an existing vocab.json is only ever
appended to: names already listed keep their place, and names new in this game
build go on the end, sorted. A name the game dropped stays, unused.

Usage:
    python3 scripts/generate_sts2_vocab.py [localization_eng dir] [output]
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_TABLES = REPO / "run" / "sts2-host" / "sts2-cli" / "localization_eng"
DEFAULT_OUTPUT = REPO / "sts2" / "vocab.json"

# vocabulary -> (table file, which keys hold a name)
SOURCES = {
    "card": ("cards.json", ".title"),
    "relic": ("relics.json", ".title"),
    "potion": ("potions.json", ".title"),
    "power": ("powers.json", ".title"),
    "monster": ("monsters.json", ".name"),
    "orb": ("orbs.json", ".title"),
    "enchantment": ("enchantments.json", ".title"),
    "affliction": ("afflictions.json", ".title"),
}


def names_from_table(path, suffix):
    """The values of keys like 'STRIKE_IRONCLAD.title': one entry's display name.

    Only top-level entries count: 'AEONGLASS.name' is a monster, but
    'AEONGLASS.moves.EBB.title' is one of its moves.
    """
    table = json.loads(path.read_text(encoding="utf-8"))
    return {value.strip() for key, value in table.items()
            if key.endswith(suffix) and key.count(".") == 1 and value.strip()}


def event_option_keys(path):
    """Event options are sent with their text_key, e.g. 'NEOW.pages.INITIAL.options.KALEIDOSCOPE'.

    The table has '<that key>.title' for each one; the encoder looks the key up.
    """
    table = json.loads(path.read_text(encoding="utf-8"))
    return {key[: -len(".title")] for key in table
            if key.endswith(".title") and ".options." in key}


def merge(existing, found):
    """existing in its order, then whatever is new in found, sorted."""
    known = {name.casefold() for name in existing}
    return list(existing) + sorted(name for name in found if name.casefold() not in known)


def main():
    tables = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TABLES
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT
    if not tables.is_dir():
        sys.exit(f"No text tables at {tables}. Build the host first (sts2-build).")

    vocab = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    for name, (file, suffix) in SOURCES.items():
        vocab[name] = merge(vocab.get(name, []), names_from_table(tables / file, suffix))
    vocab["event_option"] = merge(vocab.get("event_option", []), event_option_keys(tables / "events.json"))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(vocab, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {output}: " + ", ".join(f"{k} {len(v)}" for k, v in vocab.items()))


if __name__ == "__main__":
    main()
