"""An actor keeping its policy on the newest weights the learner has published."""

import torch as th
from sb3_contrib.ppo_mask import MaskablePPO

from model.policy_codec import encode_policy
from model.policy_follower import PolicyFollower
from slay_the_spire_env import SlayTheSpireEnv

from tests.test_learner import make_model, weights


class FakeSource:
    """The policy topic as an actor reads it: each read returns only the newest message since the last."""

    def __init__(self):
        self.published = []
        self.empty_reads_before_first = 0

    def publish(self, value):
        self.published.append(value)

    def newest(self, timeout=0.0):
        if self.empty_reads_before_first:
            self.empty_reads_before_first -= 1
            return None
        if not self.published:
            return None
        newest = self.published[-1]
        self.published.clear()
        return newest


def learner_at(version):
    model = make_model()
    with th.no_grad():
        for parameter in model.policy.parameters():
            parameter.fill_(version)
    return model


def runs(follower, learner):
    return all(th.equal(old, new) for old, new in zip(weights(learner), follower.policy.parameters()))


def test_an_actor_waits_for_the_first_published_version():
    source = FakeSource()
    source.empty_reads_before_first = 2
    learner = learner_at(3)
    source.publish(encode_policy(learner.policy, version=3))
    follower = PolicyFollower(make_model().policy, source)

    follower.wait_for_first()

    assert follower.version == 3
    assert runs(follower, learner)


def following(version):
    source = FakeSource()
    source.publish(encode_policy(learner_at(version).policy, version=version))
    follower = PolicyFollower(make_model().policy, source)
    follower.wait_for_first()
    return follower, source


def test_an_update_switches_to_the_newest_version_published_since_the_last():
    follower, source = following(1)
    for version in (2, 3, 4):
        newest = learner_at(version)
        source.publish(encode_policy(newest.policy, version=version))

    switched = follower.update()

    assert switched
    assert follower.version == follower.learner_version == 4
    assert runs(follower, newest)


def test_weights_that_cannot_be_loaded_leave_the_actor_on_its_current_version():
    follower, source = following(1)
    current = learner_at(1)
    source.publish(encode_policy(learner_at(2).policy, version=2)[:-100])

    switched = follower.update()

    assert not switched
    assert follower.version == 1
    assert runs(follower, current)


def test_a_version_the_actor_already_runs_is_not_loaded_again():
    follower, source = following(5)
    source.publish(encode_policy(learner_at(5).policy, version=5))

    assert not follower.update()
    assert not follower.update()
    assert follower.version == 5


def test_the_learner_version_is_known_even_when_its_weights_cannot_be_loaded():
    follower, source = following(1)
    wider = MaskablePPO("MultiInputPolicy", SlayTheSpireEnv({}), device="cpu", policy_kwargs={"net_arch": [64, 32]})
    source.publish(encode_policy(wider.policy, version=2))

    assert not follower.update()
    assert follower.version == 1
    assert follower.learner_version == 2
