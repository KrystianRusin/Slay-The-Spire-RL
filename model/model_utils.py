from typing import NamedTuple

import numpy as np
import torch as th


class PPOLoss(NamedTuple):
    total: th.Tensor
    policy: th.Tensor
    value: th.Tensor
    entropy: th.Tensor
    ratio: th.Tensor


class TrainStats(NamedTuple):
    steps: int
    policy_loss: float
    value_loss: float
    entropy_loss: float
    approx_kl: float
    # |new_prob / old_prob - 1| on the first minibatch, before any optimizer step.
    # About 0 only if the learner sees what the actor saw: same weights, observations and masks.
    first_ratio_error_mean: float
    first_ratio_error_max: float


def ppo_loss(policy, batch, clip_range, ent_coef, vf_coef):
    """Clipped PPO loss for one minibatch from CustomRolloutBuffer.get.

    log_prob and entropy come from the masked distribution, the one the actor
    sampled from, so the ratio compares like with like.
    """
    actions = batch["actions"].long().flatten()
    values, log_prob, entropy = policy.evaluate_actions(batch["observations"], actions,
                                                        action_masks=batch["action_masks"])
    values = values.flatten()

    advantages = batch["advantages"]
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    ratio = th.exp(log_prob - batch["log_probs"])
    policy_loss_1 = advantages * ratio
    policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

    value_loss = th.nn.functional.mse_loss(batch["returns"], values)
    entropy_loss = -th.mean(entropy)
    total = policy_loss + ent_coef * entropy_loss + vf_coef * value_loss
    return PPOLoss(total, policy_loss, value_loss, entropy_loss, ratio)


def train(model, buffer, progress_remaining=1.0):
    """One PPO update over buffer: n_epochs of shuffled minibatches.

    Errors propagate, so a broken update can't pass for a working one.
    Returns TrainStats; the losses and KL are averages over the last epoch.
    """
    clip_range = model.clip_range(progress_remaining)
    first_ratio = None

    for _ in range(model.n_epochs):
        epoch = []
        for batch in buffer.get(model.batch_size):
            loss = ppo_loss(model.policy, batch, clip_range, model.ent_coef, model.vf_coef)
            ratio = loss.ratio.detach()
            if first_ratio is None:
                first_ratio = ratio  # computed before any optimizer step

            model.policy.optimizer.zero_grad()
            loss.total.backward()
            th.nn.utils.clip_grad_norm_(model.policy.parameters(), model.max_grad_norm)
            model.policy.optimizer.step()

            approx_kl = th.mean(ratio - 1 - th.log(ratio))
            epoch.append([loss.policy.item(), loss.value.item(), loss.entropy.item(), approx_kl.item()])

    policy_loss, value_loss, entropy_loss, approx_kl = np.mean(epoch, axis=0)
    first_error = th.abs(first_ratio - 1)
    return TrainStats(buffer.pos, float(policy_loss), float(value_loss), float(entropy_loss), float(approx_kl),
                      first_error.mean().item(), first_error.max().item())
