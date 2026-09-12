import torch as th
import time
from typing import NamedTuple


class PPOLoss(NamedTuple):
    total: th.Tensor
    policy: th.Tensor
    value: th.Tensor
    entropy: th.Tensor


def ppo_loss(policy, batch, clip_range, ent_coef, vf_coef):
    """Clipped PPO loss for one minibatch from CustomRolloutBuffer.get."""
    actions = batch["actions"].long().flatten()
    values, log_prob, entropy = policy.evaluate_actions(batch["observations"], actions)
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
    return PPOLoss(total, policy_loss, value_loss, entropy_loss)


def update_model(model, rollout_buffer, current_step, total_steps):
    n_epochs = 10
    batch_size = 64
    progress_remaining = 1 - (current_step / total_steps)
    clip_range = model.clip_range(progress_remaining)

    for epoch in range(n_epochs):
        for rollout_data in rollout_buffer.get(batch_size):
            try:
                loss = ppo_loss(model.policy, rollout_data, clip_range, model.ent_coef, model.vf_coef)

                model.policy.optimizer.zero_grad()
                loss.total.backward()
                th.nn.utils.clip_grad_norm_(model.policy.parameters(), model.max_grad_norm)
                model.policy.optimizer.step()

            except Exception as e:
                with open("model_update_log.txt", "a") as log_file:
                    log_file.write(f"Error during model update at {time.strftime('%Y-%m-%d %H:%M:%S')}: {str(e)}\n")
                print(f"Error during model update: {e}")

        with open("model_update_log.txt", "a") as log_file:
            log_file.write(f"Model was updated at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        print("Model Updated and logged.")
