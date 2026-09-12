"""The vocabulary maps that turn game names into integer ids."""

import pytest

from util.vocabularies import ALL_VOCABULARIES, UNKNOWN_ID, Vocabulary

VOCABULARIES = sorted(ALL_VOCABULARIES.items())
VOCABULARY_IDS = [name for name, _ in VOCABULARIES]


@pytest.fixture(params=[vocab for _, vocab in VOCABULARIES], ids=VOCABULARY_IDS)
def vocabulary(request):
    return request.param


# Strike and Defend appear once per character class, and one power is listed
# twice. Any other drop means a typo crept into a list.
EXPECTED_DUPLICATES = {
    "card": ["Defend", "Strike", "Defend", "Strike", "Strike"],
    "power": ["CorpseExplosionPower"],
}


def test_no_name_is_dropped_except_the_known_repeats(vocabulary):
    assert vocabulary.duplicates == EXPECTED_DUPLICATES.get(vocabulary.label, [])


def test_every_name_in_the_source_list_resolves(vocabulary):
    """Goes through the list as written, not through the deduplicated one."""
    source = vocabulary.names + vocabulary.duplicates

    unresolved = [name for name in source if vocabulary.id_of(name) == UNKNOWN_ID]

    assert not unresolved, f"{vocabulary.label} cannot find {unresolved}"


def test_ids_are_contiguous_and_leave_the_unknown_id_free(vocabulary):
    ids = sorted(vocabulary.id_of(name) for name in vocabulary.names)

    assert ids == list(range(1, vocabulary.max_id + 1))
    assert UNKNOWN_ID == 0


def test_an_unrecognised_name_returns_the_unknown_id(vocabulary):
    assert vocabulary.id_of("Sozu's Revenge") == UNKNOWN_ID


def test_a_missing_or_blank_name_returns_the_unknown_id(vocabulary):
    assert vocabulary.id_of(None) == UNKNOWN_ID
    assert vocabulary.id_of("") == UNKNOWN_ID


def test_lookup_ignores_surrounding_space(vocabulary):
    name = vocabulary.names[0]

    assert vocabulary.id_of(f"  {name} ") == vocabulary.id_of(name)


def test_lookup_ignores_case():
    cards = ALL_VOCABULARIES["card"]

    assert cards.id_of("bOdY sLaM") == cards.id_of("Body Slam")
    assert cards.id_of("BODY SLAM") == cards.id_of("Body Slam")


@pytest.mark.parametrize("vocabulary_name, names", [
    ("card", ["Empty Body", "Empty Fist", "Empty Mind"]),
    ("card", ["Dagger Spray", "Dagger Throw"]),
    ("card", ["Secret Technique", "Secret Weapon"]),
    ("relic", ["Bottled Flame", "Bottled Lightning", "Bottled Tornado"]),
    ("event_id", ["The Joust", "The Library", "The Mausoleum", "The Moai Head"]),
    ("power", ["Draw", "Draw Card", "Draw Reduction"]),
])
def test_names_sharing_a_first_word_do_not_collide(vocabulary_name, names):
    vocabulary = ALL_VOCABULARIES[vocabulary_name]
    ids = [vocabulary.id_of(name) for name in names]

    assert UNKNOWN_ID not in ids, f"{names} are not all in the {vocabulary_name} vocabulary"
    assert len(set(ids)) == len(names)


def test_a_name_is_one_id_not_a_sequence_of_word_ids():
    cards = ALL_VOCABULARIES["card"]

    assert cards.id_of("A Thousand Cuts") != cards.id_of("Adrenaline")
    assert cards.id_of("A") == UNKNOWN_ID, "single words are not names"
    assert cards.id_of("Thousand") == UNKNOWN_ID


@pytest.mark.parametrize("vocabulary_name, name", [
    ("monster_id", "WritingMass"),
    ("monster_id", "FuzzyLouseDefensive"),
    ("monster_id", "FuzzyLouseNormal"),
    ("power", "Duplication Power"),
    ("power", "Echo Form"),
    ("potion", "Potion Slot"),
    ("map_symbol", "B"),
])
def test_names_a_malformed_list_entry_used_to_hide_resolve(vocabulary_name, name):
    assert ALL_VOCABULARIES[vocabulary_name].id_of(name) != UNKNOWN_ID


def test_punctuation_is_part_of_the_name():
    cards = ALL_VOCABULARIES["card"]

    assert cards.id_of("J.A.X.") != UNKNOWN_ID
    assert cards.id_of("JAX") == UNKNOWN_ID


def test_map_symbols_are_ids_not_stripped_punctuation():
    symbols = ALL_VOCABULARIES["map_symbol"]

    ids = {symbol: symbols.id_of(symbol) for symbol in ["?", "$", "T", "M", "E", "R"]}

    assert UNKNOWN_ID not in ids.values()
    assert len(set(ids.values())) == len(ids)


def test_a_duplicated_name_keeps_its_first_id():
    """The card list repeats Strike and Defend once per character class."""
    vocabulary = Vocabulary("test", ["Strike", "Defend", "Strike"])

    assert vocabulary.names == ["Strike", "Defend"]
    assert vocabulary.duplicates == ["Strike"]
    assert vocabulary.id_of("Strike") == 1
    assert vocabulary.max_id == 2


def test_ids_do_not_move_when_a_name_is_appended():
    """Ids are positional, so a later addition cannot renumber earlier names."""
    before = Vocabulary("test", ["Strike", "Defend"])
    after = Vocabulary("test", ["Strike", "Defend", "Bash"])

    assert after.id_of("Strike") == before.id_of("Strike")
    assert after.id_of("Defend") == before.id_of("Defend")
