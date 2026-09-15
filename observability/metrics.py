"""The Prometheus metrics each process exports, and serving them for scraping.

Each metrics object keeps its own registry, served by serve(port). The
metrics and why they are measured where they are is recorded in docs/adr/0006.
"""

import time
from collections import deque

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server

ROLLING_EPISODES = 10
ROLLOUT_AGE_BUCKETS = (5, 15, 30, 60, 120, 300, 600, 1200, 1800, 3600, 7200)


class LearnerMetrics:
    """Rollout age, training progress and skipped rollouts for the learner."""

    def __init__(self, clock=time.time):
        self.registry = CollectorRegistry()
        self._clock = clock
        self._rollout_age = Histogram(
            "sts_learner_rollout_age_seconds",
            "Time from an actor publishing a rollout to the learner training on it",
            buckets=ROLLOUT_AGE_BUCKETS,
            registry=self.registry,
        )
        self._last_rollout_age = Gauge(
            "sts_learner_last_rollout_age_seconds", "Age of the rollout the learner most recently trained on", registry=self.registry,
        )
        self._updates = Counter("sts_learner_updates", "PPO updates applied", registry=self.registry)
        self._policy_version = Gauge("sts_learner_policy_version", "Newest policy version the learner has", registry=self.registry)
        self._steps = Gauge("sts_learner_steps", "Steps trained, counting those before a restart", registry=self.registry)
        self._consumer_lag = Gauge(
            "sts_learner_consumer_lag_rollouts", "Rollouts on the rollout topic the learner has not committed",
            labelnames=["partition"], registry=self.registry,
        )
        self._skipped = Counter(
            "sts_learner_rollouts_skipped", "Rollouts committed without training on them",
            labelnames=["reason"], registry=self.registry,
        )

    def consumer_lag(self, lag):
        """Record lag given as rollouts not yet committed, by partition number, or None when it could not be measured."""
        self._consumer_lag.clear()
        for partition, rollouts in (lag or {}).items():
            self._consumer_lag.labels(str(partition)).set(rollouts)

    def record_progress(self, progress):
        self._policy_version.set(progress.policy_version)
        self._steps.set(progress.steps)

    def training_on(self, published_at):
        """Record the age of a rollout about to be trained on, given when it was published in seconds since the epoch, if known."""
        if published_at is not None:
            age = max(0.0, self._clock() - published_at)
            self._rollout_age.observe(age)
            self._last_rollout_age.set(age)

    def updated(self, progress):
        self._updates.inc()
        self.record_progress(progress)

    def skipped(self, reason):
        self._skipped.labels(reason).inc()

    def serve(self, port):
        """Serve these metrics over HTTP at /metrics on port, from a background thread."""
        start_http_server(port, registry=self.registry)


class ActorMetrics:
    """Throughput, policy lag and episode rewards for one actor, labelled with its id."""

    def __init__(self, actor_id):
        self.registry = CollectorRegistry()
        labels = {"labelnames": ["actor_id"], "registry": self.registry}
        self._steps = Counter("sts_actor_steps", "Actions taken in the game", **labels).labels(actor_id)
        self._rollouts = Counter("sts_actor_rollouts_published", "Rollouts published for the learner", **labels).labels(actor_id)
        self._policy_version = Gauge("sts_actor_policy_version", "Policy version the actor is playing", **labels).labels(actor_id)
        self._policy_lag = Gauge(
            "sts_actor_policy_lag_versions",
            "Versions the learner had published past the policy the last published rollout was collected under",
            **labels,
        ).labels(actor_id)
        self._episodes = Counter("sts_actor_episodes", "Episodes finished", **labels).labels(actor_id)
        self._length = Gauge("sts_actor_episode_length_steps", "Steps in the last finished episode", **labels).labels(actor_id)
        self._reward = Gauge("sts_actor_episode_reward", "Total reward of the last finished episode", **labels).labels(actor_id)
        self._rolling_reward = Gauge(
            "sts_actor_episode_reward_rolling_mean",
            f"Mean total reward of the last {ROLLING_EPISODES} finished episodes",
            **labels,
        ).labels(actor_id)
        self._recent_rewards = deque(maxlen=ROLLING_EPISODES)

    def step(self):
        self._steps.inc()

    def running_policy(self, version):
        self._policy_version.set(version)

    def rollout_published(self, collected_under, learner_version):
        self._rollouts.inc()
        self._policy_lag.set(learner_version - collected_under)

    def episode_finished(self, reward, length):
        self._episodes.inc()
        self._length.set(length)
        self._reward.set(reward)
        self._recent_rewards.append(reward)
        self._rolling_reward.set(sum(self._recent_rewards) / len(self._recent_rewards))

    def serve(self, port):
        """Serve these metrics over HTTP at /metrics on port, from a background thread."""
        start_http_server(port, registry=self.registry)
