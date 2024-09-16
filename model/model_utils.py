import torch as th
import time 

def update_model(model, rollout_buffer, current_step, total_steps):
    n_epochs = 10
    batch_size = 64
    progress_remaining = 1 - (current_step / total_steps)

    for epoch in range(n_epochs):
        for rollout_data in rollout_buffer.get(batch_size):
            try:
                actions = th.tensor(rollout_data["actions"], dtype=th.long).flatten().to(model.device)
                observations = rollout_data["observations"]
                observations_tensor = {key: th.tensor(value).to(model.device) for key, value in observations.items()}

                values, log_prob, entropy = model.policy.evaluate_actions(observations_tensor, actions)
                advantages = rollout_data["advantages"].to(model.device)
                log_probs_old = rollout_data["log_probs"].to(model.device)
                returns = rollout_data["returns"].to(model.device)

                ratio = th.exp(log_prob - log_probs_old)
                clip_range = model.clip_range(progress_remaining)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                value_loss = th.nn.functional.mse_loss(returns, values)
                entropy_loss = -th.mean(entropy)
                loss = policy_loss + model.ent_coef * entropy_loss + model.vf_coef * value_loss

                model.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(model.policy.parameters(), model.max_grad_norm)
                model.policy.optimizer.step()

            except Exception as e:
                with open("model_update_log.txt", "a") as log_file:
                    log_file.write(f"Error during model update at {time.strftime('%Y-%m-%d %H:%M:%S')}: {str(e)}\n")
                print(f"Error during model update: {e}")

        with open("model_update_log.txt", "a") as log_file:
            log_file.write(f"Model was updated at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        print("Model Updated and logged.")
