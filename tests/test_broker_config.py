"""Broker connection details, read from the environment."""

import pytest

from broker.config import load_broker_config


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in ["KAFKA_BOOTSTRAP_SERVERS", "KAFKA_REPLICATION_FACTOR"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("broker.config.load_dotenv", lambda: None)


def test_bootstrap_servers_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "kafka-0:9092,kafka-1:9092")

    assert load_broker_config().bootstrap_servers == "kafka-0:9092,kafka-1:9092"


def test_missing_bootstrap_servers_fail_loudly():
    with pytest.raises(RuntimeError, match="KAFKA_BOOTSTRAP_SERVERS"):
        load_broker_config()


def test_replication_factor_can_be_raised_for_a_multi_broker_cluster(monkeypatch):
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    monkeypatch.setenv("KAFKA_REPLICATION_FACTOR", "3")

    assert load_broker_config().replication_factor == 3
