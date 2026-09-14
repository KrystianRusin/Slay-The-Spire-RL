"""Game instances registering where they listen, and actors claiming one each."""

import multiprocessing
from datetime import datetime, timedelta

import pytest

from db.game_registry import EXPIRY_SECONDS, ClaimLost, GameRegistry, wait_for_game

from tests.test_data_layer import database, shared_database  # noqa: F401

START = datetime(2026, 9, 13, 12, 0, 0)


class Clock:
    def __init__(self):
        self.now = START

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += timedelta(seconds=seconds)


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def registry(database, clock):
    return GameRegistry(clock=clock)


def test_each_actor_is_assigned_a_different_game(registry):
    registry.register("game-a", "localhost", 5001)
    registry.register("game-b", "localhost", 5002)

    first = registry.claim("actor-0")
    second = registry.claim("actor-1")

    assert {first, second} == {"game-a", "game-b"}
    assert registry.claim("actor-2") is None


@pytest.mark.parametrize("game_order", [["game-a", "game-b"], ["game-b", "game-a"]])
def test_startup_order_does_not_change_that_every_actor_reaches_its_own_game(registry, game_order):
    ports = {"game-a": 5001, "game-b": 5002}
    assert registry.claim("actor-0") is None

    for game_id in game_order:
        registry.register(game_id, "localhost", ports[game_id])
    claims = {actor_id: registry.claim(actor_id) for actor_id in ("actor-0", "actor-1")}

    assert sorted(claims.values()) == ["game-a", "game-b"]
    for actor_id, game_id in claims.items():
        assert registry.address(game_id, actor_id) == ("localhost", ports[game_id])


def test_a_restarted_middleman_reports_its_new_port_and_the_claim_survives(registry):
    registry.register("game-a", "localhost", 5001)
    registry.claim("actor-0")

    registry.register("game-a", "localhost", 6001)

    assert registry.address("game-a", "actor-0") == ("localhost", 6001)
    assert registry.claim("actor-1") is None


def test_a_game_whose_middleman_stopped_reporting_has_no_address_and_cannot_be_claimed(registry, clock):
    registry.register("game-a", "localhost", 5001)
    registry.claim("actor-0")

    clock.advance(EXPIRY_SECONDS + 1)
    registry.renew("game-a", "actor-0")

    assert registry.address("game-a", "actor-0") is None
    assert registry.claim("actor-1") is None


def test_a_claim_the_actor_keeps_renewing_is_not_taken(registry, clock):
    registry.register("game-a", "localhost", 5001)
    registry.claim("actor-0")

    for _ in range(3):
        clock.advance(EXPIRY_SECONDS - 1)
        registry.register("game-a", "localhost", 5001)
        registry.renew("game-a", "actor-0")

    assert registry.claim("actor-1") is None


def test_the_claim_of_an_actor_that_stopped_renewing_passes_to_another(registry, clock):
    registry.register("game-a", "localhost", 5001)
    registry.claim("actor-0")

    clock.advance(EXPIRY_SECONDS + 1)
    registry.register("game-a", "localhost", 5001)

    assert registry.claim("actor-1") == "game-a"
    with pytest.raises(ClaimLost):
        registry.address("game-a", "actor-0")
    with pytest.raises(ClaimLost):
        registry.renew("game-a", "actor-0")


def test_a_released_game_can_be_claimed_at_once(registry):
    registry.register("game-a", "localhost", 5001)
    registry.claim("actor-0")

    registry.release("game-a", "actor-0")

    assert registry.claim("actor-1") == "game-a"


def test_releasing_a_claim_already_taken_leaves_the_new_owner_in_place(registry, clock):
    registry.register("game-a", "localhost", 5001)
    registry.claim("actor-0")
    clock.advance(EXPIRY_SECONDS + 1)
    registry.register("game-a", "localhost", 5001)
    registry.claim("actor-1")

    registry.release("game-a", "actor-0")

    assert registry.address("game-a", "actor-1") == ("localhost", 5001)


def test_a_restarted_actor_takes_back_the_game_it_had_claimed(registry):
    registry.register("game-a", "localhost", 5001)
    registry.register("game-b", "localhost", 5002)
    assert registry.claim("actor-0") == "game-a"

    assert registry.claim("actor-0") == "game-a"


def test_deregistering_an_old_middleman_leaves_its_replacement_registered(registry):
    registry.register("game-a", "localhost", 5001)
    registry.register("game-a", "localhost", 6001)

    registry.deregister("game-a", 5001)
    assert registry.claim("actor-0") == "game-a"

    registry.deregister("game-a", 6001)
    assert registry.address("game-a", "actor-0") is None


def test_waiting_for_a_game_polls_until_one_registers(registry):
    polls = []

    def sleep(seconds):
        polls.append(seconds)
        if len(polls) == 2:
            registry.register("game-a", "localhost", 5001)

    assert wait_for_game(registry, "actor-0", poll_seconds=5, sleep=sleep) == "game-a"
    assert polls == [5, 5]


def claim_as(actor_id):
    return GameRegistry().claim(actor_id)


def test_actors_claiming_at_once_never_share_a_game(shared_database):
    games, actors = 3, 6
    registry = GameRegistry()
    for game in range(games):
        registry.register(f"game-{game}", "localhost", 5000 + game)

    with multiprocessing.get_context("spawn").Pool(actors) as pool:
        claims = pool.map(claim_as, [f"actor-{actor}" for actor in range(actors)])

    claimed = [game_id for game_id in claims if game_id is not None]
    assert sorted(claimed) == [f"game-{game}" for game in range(games)]
