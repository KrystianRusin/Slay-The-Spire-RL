"""Policy weights as bytes, published by the learner and loaded by actors."""

import pytest
import torch as th
from sb3_contrib.ppo_mask import MaskablePPO

from model.policy_codec import decode_policy, encode_policy
from slay_the_spire_env import SlayTheSpireEnv

from tests.test_learner import make_model, weights


def test_weights_arrive_with_the_version_they_were_published_as():
    learner, actor = make_model(), make_model()
    with th.no_grad():
        for parameter in learner.policy.parameters():
            parameter.add_(1.0)

    published = decode_policy(encode_policy(learner.policy, version=7))
    published.apply_to(actor.policy)

    assert published.version == 7
    assert all(th.equal(old, new) for old, new in zip(weights(learner), actor.policy.parameters()))


@pytest.mark.parametrize("end", [0, 3, 12, 100, 100_000, -1])
def test_a_partially_transferred_payload_is_rejected(end):
    encoded = encode_policy(make_model().policy, version=1)

    with pytest.raises(ValueError):
        decode_policy(encoded[:end])


def test_a_corrupted_payload_is_rejected():
    encoded = bytearray(encode_policy(make_model().policy, version=1))
    encoded[len(encoded) // 2] ^= 0xFF

    with pytest.raises(ValueError, match="digest"):
        decode_policy(bytes(encoded))


def test_weights_that_do_not_fit_the_policy_leave_it_untouched():
    wider = MaskablePPO("MultiInputPolicy", SlayTheSpireEnv({}), device="cpu", seed=1, policy_kwargs={"net_arch": [64, 32]})
    actor = make_model()
    before = weights(actor)

    with pytest.raises(ValueError, match="does not fit"):
        decode_policy(encode_policy(wider.policy, version=1)).apply_to(actor.policy)

    assert all(th.equal(old, new) for old, new in zip(before, actor.policy.parameters()))
