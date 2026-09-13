"""Game records in the database, written by one actor or by several at once."""

import logging
import multiprocessing
import os

import pytest
from sqlalchemy import create_engine, inspect, select

import db.session
from db.models import Base, CardPerformance, CardPicked, Game
from db.session import get_engine, init_db, session_scope
from tests.conftest import load_payload
from util.card_tracking import track_card_performance
from util.data_processor import process_game_state
from util.game_over_tracking import update_game_stats_on_game_over

START = "START IRONCLAD 0"


def boss_reward_payload():
    payload = load_payload("combat_reward")
    payload["game_state"]["screen_type"] = "BOSS_REWARD"
    return payload


def start_game():
    return process_game_state(load_payload("main_menu"), START, None)


def games():
    with session_scope() as session:
        return {game.game_id: game for game in session.scalars(select(Game))}


def card_picks():
    with session_scope() as session:
        return session.scalars(select(CardPicked)).all()


def card_performance():
    with session_scope() as session:
        return {row.card_id: row for row in session.scalars(select(CardPerformance))}


def sqlite_url(tmp_path):
    return f"sqlite:///{(tmp_path / 'games.db').as_posix()}"


def use_database(monkeypatch, url):
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setattr(db.session, "_engine", None)
    monkeypatch.setattr(db.session, "_session_factory", None)
    init_db()


@pytest.fixture
def database(tmp_path, monkeypatch):
    use_database(monkeypatch, sqlite_url(tmp_path))
    yield
    get_engine().dispose()


def postgres_url():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL is not set")
    engine = create_engine(url)
    try:
        existing = set(inspect(engine).get_table_names()) & set(Base.metadata.tables)
    finally:
        engine.dispose()
    if existing:
        pytest.fail(f"TEST_DATABASE_URL must point at a scratch database; it already has {', '.join(sorted(existing))}")
    return url


@pytest.fixture(params=["sqlite", "postgresql"])
def shared_database(request, tmp_path, monkeypatch):
    if request.param == "sqlite":
        url = sqlite_url(tmp_path)
    else:
        url = postgres_url()
    use_database(monkeypatch, url)
    yield
    engine = get_engine()
    if request.param == "postgresql":
        Base.metadata.drop_all(engine)
    engine.dispose()


def test_starting_a_game_returns_the_id_the_database_assigned(database):
    game_id = start_game()

    assert list(games()) == [game_id]
    game = games()[game_id]
    assert game.agent_class == "IRONCLAD"
    assert game.reward == 0.0


def test_each_started_game_gets_its_own_id(database):
    assert len({start_game() for _ in range(3)}) == 3


def test_a_card_pick_is_recorded_against_the_game_it_was_made_in(database):
    start_game()
    second = start_game()

    returned = process_game_state(load_payload("card_reward"), "CHOOSE 1", second)

    assert returned == second
    [pick] = card_picks()
    assert pick.game_id == second
    offered = load_payload("card_reward")["game_state"]["screen_state"]["cards"]
    assert pick.card_id == offered[1]["id"]


def test_game_over_updates_the_game_it_was_given(database):
    first = start_game()
    second = start_game()

    update_game_stats_on_game_over(load_payload("game_over"), second, 42.5)

    recorded = games()
    assert recorded[second].reward == 42.5
    assert recorded[second].floors_reached == load_payload("game_over")["game_state"]["floor"]
    assert recorded[second].end_time is not None
    assert recorded[first].end_time is None


def test_choosing_a_boss_relic_counts_a_defeated_boss(database):
    game_id = start_game()

    process_game_state(boss_reward_payload(), "CHOOSE 0", game_id)

    assert games()[game_id].bosses_defeated == 1


def test_lingering_on_the_boss_reward_screen_does_not_count_the_boss_again(database):
    game_id = start_game()

    process_game_state(boss_reward_payload(), "RETURN", game_id)
    process_game_state(boss_reward_payload(), "CHOOSE 0", game_id)

    assert games()[game_id].bosses_defeated == 1


def test_nothing_is_recorded_against_a_game_that_was_never_started(database, caplog):
    process_game_state(load_payload("card_reward"), "CHOOSE 0", None)

    assert card_picks() == []
    assert "No game has been recorded" in caplog.text


def test_a_failed_write_is_logged_and_releases_its_connection(database, caplog):
    Base.metadata.drop_all(get_engine())

    with caplog.at_level(logging.ERROR):
        assert start_game() is None

    [record] = caplog.records
    assert record.exc_info is not None
    assert get_engine().pool.checkedout() == 0


def test_card_performance_accumulates_across_games(database):
    game_state = load_payload("game_over")["game_state"]

    track_card_performance(game_state, 10, won=True)
    track_card_performance(game_state, 20, won=False)

    performance = card_performance()
    deck = [card for card in game_state["deck"] if card["name"] not in ("Strike", "Defend")]
    assert set(performance) == {card["id"] for card in deck}
    for card_id, row in performance.items():
        copies = sum(card["id"] == card_id for card in deck)
        assert row.games_featured_in == 2
        assert row.times_picked == 2 * copies
        assert row.average_floor_reached == pytest.approx(15.0)
        assert row.win_rate == pytest.approx(0.5)


class _ErrorCollector(logging.Handler):
    def __init__(self):
        super().__init__(logging.ERROR)
        self.messages = []

    def emit(self, record):
        self.messages.append(self.format(record))


def play_games(worker, count):
    """One actor's worth of games, as a separate process would record them."""
    errors = _ErrorCollector()
    logging.getLogger().addHandler(errors)

    rewards = {}
    game_over = load_payload("game_over")
    for game in range(count):
        game_id = start_game()
        process_game_state(load_payload("card_reward"), "CHOOSE 0", game_id)
        process_game_state(boss_reward_payload(), "CHOOSE 0", game_id)
        rewards[game_id] = worker * 1000 + game
        update_game_stats_on_game_over(game_over, game_id, rewards[game_id])
        track_card_performance(game_over["game_state"], game_over["game_state"]["floor"], won=False)
    return rewards, errors.messages


def test_concurrent_actors_each_record_their_own_games(shared_database):
    workers, games_each = 4, 10

    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        results = pool.starmap(play_games, [(worker, games_each) for worker in range(workers)])

    errors = [message for _, messages in results for message in messages]
    assert errors == []

    rewards = {}
    for worker_rewards, _ in results:
        assert not rewards.keys() & worker_rewards.keys(), "two actors were handed the same game id"
        rewards.update(worker_rewards)
    assert len(rewards) == workers * games_each

    recorded = games()
    assert set(recorded) == set(rewards)
    for game_id, game in recorded.items():
        assert game.reward == rewards[game_id]
        assert game.bosses_defeated == 1

    picks_per_game = {}
    for pick in card_picks():
        picks_per_game[pick.game_id] = picks_per_game.get(pick.game_id, 0) + 1
    assert picks_per_game == {game_id: 1 for game_id in rewards}

    for row in card_performance().values():
        assert row.games_featured_in == workers * games_each
