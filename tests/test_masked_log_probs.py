"""The actor records log-probs under the masked policy, and the learner recomputes
the same ones, so the first PPO ratio of an update is exactly 1."""

import numpy as np
import torch as th

from environment.run_env import act
from model.policy import TrainingConfig, make_model
from tests.rollouts import make_buffer

LEGAL = [3, 17, 40]


def random_obs(observation_space, generator):
    return {key: th.rand(1, *space.shape, generator=generator)
            for key, space in observation_space.spaces.items()}


def only(actions, n=141):
    mask = np.zeros(n, dtype=bool)
    mask[actions] = True
    return mask


def test_the_actor_only_ever_picks_legal_actions():
    model = make_model(TrainingConfig())
    obs = random_obs(model.observation_space, th.Generator().manual_seed(0))

    picked = {act(model, obs, only(LEGAL))[0] for _ in range(200)}

    assert picked <= set(LEGAL)


def test_the_recorded_log_prob_is_over_the_legal_actions_only():
    model = make_model(TrainingConfig())
    obs = random_obs(model.observation_space, th.Generator().manual_seed(0))

    action, _, log_prob = act(model, obs, only(LEGAL))

    with th.no_grad():
        logits = model.policy.get_distribution(obs).distribution.logits[0]
    masked = th.log_softmax(logits[LEGAL], dim=0)[LEGAL.index(action)]
    unmasked = th.log_softmax(logits, dim=0)[action]
    assert th.isclose(log_prob.squeeze(), masked, atol=1e-5)
    assert not th.isclose(log_prob.squeeze(), unmasked, atol=1e-3)


def test_the_learner_recomputes_the_actors_log_probs_exactly():
    """What makes the first ratio exp(new - old) equal 1 before any update."""
    model = make_model(TrainingConfig())
    buffer = make_buffer(model.observation_space, size=32)
    generator = th.Generator().manual_seed(1)
    for step in range(32):
        obs = random_obs(model.observation_space, generator)
        mask = only(sorted({step % 141, (7 * step) % 141, 100}))
        action, value, log_prob = act(model, obs, mask)
        buffer.add(obs, th.tensor(action), 0.0, False, value, log_prob, mask)

    batch = next(buffer.get())
    with th.no_grad():
        _, new_log_prob, _ = model.policy.evaluate_actions(
            batch["observations"], batch["actions"].long().flatten(), action_masks=batch["action_masks"])

    ratio = th.exp(new_log_prob - batch["log_probs"])
    assert th.allclose(ratio, th.ones_like(ratio), atol=1e-5)
