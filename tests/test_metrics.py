"""The metrics actors and the learner export, read back as Prometheus would scrape them."""

import pytest

from observability.metrics import ActorMetrics


def sample(metrics, name, **labels):
    return metrics.registry.get_sample_value(name, labels)


def test_the_rolling_mean_reward_follows_the_last_ten_episodes():
    metrics = ActorMetrics("3")
    means = []
    for reward in range(1, 13):
        metrics.episode_finished(reward=float(reward), length=5)
        means.append(sample(metrics, "sts_actor_episode_reward_rolling_mean", actor_id="3"))

    assert means[:3] == [1.0, 1.5, 2.0]
    assert means[-1] == pytest.approx(7.5)
    assert sample(metrics, "sts_actor_episode_reward", actor_id="3") == 12.0
    assert sample(metrics, "sts_actor_episodes_total", actor_id="3") == 12
    assert sample(metrics, "sts_actor_episode_length_steps", actor_id="3") == 5


def test_a_published_rollout_records_how_far_the_learner_had_moved_past_its_policy():
    metrics = ActorMetrics("0")
    metrics.running_policy(4)
    for _ in range(3):
        metrics.step()
    metrics.rollout_published(collected_under=4, learner_version=7)

    assert sample(metrics, "sts_actor_steps_total", actor_id="0") == 3
    assert sample(metrics, "sts_actor_rollouts_published_total", actor_id="0") == 1
    assert sample(metrics, "sts_actor_policy_lag_versions", actor_id="0") == 3
    assert sample(metrics, "sts_actor_policy_version", actor_id="0") == 4
