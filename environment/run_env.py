"""The actor loop: play STS2 runs on the headless host with the current policy,
and hand each full rollout off for training.

Each run gets its own host process (the host can't start a second run). The
loop asks the Actor for a command, sends it, and tells the Actor how the host
answered: with the next payload (accepted), with an error (refused), or not at
all (the host died). A pick that adds a card to a selection that isn't full
yet needs no host at all, and is handled by the Actor alone.
"""
import copy
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import torch as th

from model.custom_rollout_buffer import CustomRolloutBuffer
from model.policy import TrainingConfig, make_model
from sts2.env import Sts2Env
from sts2.host import Host, HostError
from sts2.vocab import CHARACTERS

# A run that hasn't ended after this many decisions is cut off: a policy that
# found a loop (buy, refuse, buy...) mustn't stall its actor forever.
MAX_RUN_STEPS = int(os.environ.get("STS2_MAX_RUN_STEPS", "4000"))
# Refusals in a row on one payload before the run is abandoned as stuck.
MAX_REFUSALS = 20


def hand_off_rollout(rollout_buffer, experience_queue):
    """Queue a completed rollout for the learner and clear the buffer for the next one."""
    # Queue.put may pickle on a background thread, so it must get a copy the reset cannot reach.
    experience_queue.put(copy.deepcopy(rollout_buffer))
    rollout_buffer.reset()


def act(model, obs_tensor, action_mask, deterministic=False):
    """Sample an action from the masked policy. Returns (action, value, log_prob).

    log_prob is taken from the same masked distribution the action was drawn
    from, which is what the learner recomputes for the PPO ratio.
    """
    mask = th.as_tensor(action_mask, dtype=th.bool, device=model.device).reshape(1, -1)
    with th.no_grad():
        distribution = model.policy.get_distribution(obs_tensor, action_masks=mask)
        action = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(action)
        value = model.policy.predict_values(obs_tensor)
    return int(action.item()), value, log_prob


def to_tensors(obs, device):
    return {key: th.as_tensor(value, dtype=th.float32).unsqueeze(0).to(device) for key, value in obs.items()}


class Actor:
    """Chooses actions for the current payload and records PPO steps.

    An action is only finished once the game answers it: the reply says what
    reward it earned and whether it ended the run. So the chosen step is held
    as the candidate until accepted() (or internal()) records it; refused()
    drops it and the next decide() picks again without it.

    on_rollout_full() is called each time the buffer fills, after the returns
    are computed and before the next action is chosen, so it can hand the
    rollout off and swap in new weights.
    """

    def __init__(self, env, model, buffer, on_rollout_full, reward_scale=1.0):
        self.env = env
        self.model = model
        self.buffer = buffer
        self.on_rollout_full = on_rollout_full
        self.reward_scale = reward_scale
        self.candidate = None  # (obs_tensor, action, value, log_prob, mask)
        self.episode_reward = 0.0
        self.episode_steps = 0

    def begin_run(self, character, ascension, first_state):
        self.env.begin_run(character, ascension)
        self.env.update_game_state(first_state)
        self.episode_reward, self.episode_steps = 0.0, 0

    def decide(self):
        """Pick an action for the current payload. Returns (action, command); command None = no host needed."""
        obs_tensor = to_tensors(self.env.observe(), self.model.device)
        commands = self.env.commands()
        mask = self.env.action_mask()
        action, value, log_prob = act(self.model, obs_tensor, mask)
        self.candidate = (obs_tensor, action, value, log_prob, mask)
        return action, commands[action]

    def accepted(self, reply):
        """The host carried out the candidate and answered with reply."""
        previous = self.env.state
        self.env.update_game_state(reply)
        reward = self.env.calculate_reward(previous, reply)
        self._record(reward, self.env.is_done(reply))

    def internal(self):
        """The candidate needs no host: apply it here (a selection pick)."""
        self.env.apply_internal(self.candidate[1])
        self._record(self.env.calculate_reward(None, None), False)

    def refused(self):
        """The host refused the candidate. Don't record it, and don't offer it again for this payload."""
        self.env.refused.add(self.candidate[1])
        self.candidate = None

    def cut_off(self):
        """The run ends without a game over (host died, stuck, too long): mark its last step as the end."""
        self.candidate = None
        if self.episode_steps > 0 and self.buffer.pos > 0:
            self.buffer.dones[self.buffer.pos - 1] = 1.0

    def _record(self, reward, done):
        obs_tensor, action, value, log_prob, mask = self.candidate
        self.candidate = None
        self.episode_reward += reward  # unscaled, for the logs
        self.episode_steps += 1
        self.buffer.add(obs_tensor, th.tensor(action, dtype=th.long), reward * self.reward_scale, done,
                        value, log_prob, mask)
        if len(self.buffer) >= self.buffer.buffer_size:
            with th.no_grad():
                next_obs = to_tensors(self.env.observe(), self.model.device)
                last_value = self.model.policy.predict_values(next_obs)
            self.buffer.compute_returns_and_advantage(last_values=last_value, dones=done)
            self.on_rollout_full()


def launch(make_host, character, seed, ascension=0):
    """A fresh host with a run started on it. Returns (host, first payload)."""
    host = make_host()
    try:
        return host, host.start_run(character, seed, ascension)
    except BaseException:
        host.close()
        raise


def play_run(actor, host, character, seed, ascension=0, first_state=None):
    """Play one run on host to the end. Returns a result dict for the logs.

    first_state is the start_run reply, if the run was already started (see launch).
    """
    started = time.monotonic()
    state = first_state if first_state is not None else host.start_run(character, seed, ascension)
    if state.get("type") == "error":
        raise HostError(f"start_run failed: {state.get('message')}")
    actor.begin_run(character, ascension, state)
    refusals = 0
    outcome = "cut_off"
    while actor.episode_steps < MAX_RUN_STEPS:
        if actor.env.is_done(actor.env.state):
            outcome = "victory" if actor.env.state.get("victory") else "defeat"
            break
        action, command = actor.decide()
        if command is None:
            actor.internal()
            continue
        reply = host.send(command)
        if reply.get("type") == "error":
            if reply.get("error_code") == "combat_turn_timeout" or refusals >= MAX_REFUSALS:
                raise HostError(f"stuck at {actor.env.state.get('decision')}: {reply.get('message')}")
            refusals += 1
            actor.refused()
            continue
        refusals = 0
        actor.accepted(reply)
    else:
        actor.cut_off()
    state = actor.env.state or {}
    act_reached = state.get("act") or (state.get("context") or {}).get("act")
    floor = state.get("floor") or (state.get("context") or {}).get("floor")
    return {"character": character, "seed": seed, "outcome": outcome, "act": act_reached, "floor": floor,
            "steps": actor.episode_steps, "reward": round(actor.episode_reward, 2),
            "seconds": round(time.monotonic() - started, 1)}


def run_environment(env_id, experience_queue, config=None, initial_state_dict=None, next_weights=None,
                    make_host=None, ascension=0, episode_log=None, rng=None):
    """Play runs forever, filling rollouts and handing them to experience_queue.

    config is the TrainingConfig to build the policy from (the learner's, sent
    with its weights); defaults to model/policy.py. initial_state_dict, if
    given, replaces the policy's random initial weights. next_weights, if
    given, is called after each rollout is handed off and returns the
    state_dict to play the next one with. make_host builds a fresh Host per
    run. episode_log, if given, is a path each run's result is appended to as
    one JSON line.
    """
    config = config or TrainingConfig()
    rng = rng or random.Random()
    make_host = make_host or (lambda: Host(log_file=open(os.devnull, "w")))
    env = Sts2Env()
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model = make_model(config, env=env, device=device)
    if initial_state_dict is not None:
        model.policy.load_state_dict(initial_state_dict)

    rollout_buffer = CustomRolloutBuffer(
        buffer_size=config.n_steps,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=1,
    )

    def on_rollout_full():
        hand_off_rollout(rollout_buffer, experience_queue)
        if next_weights is not None:
            model.policy.load_state_dict(next_weights())

    actor = Actor(env, model, rollout_buffer, on_rollout_full, reward_scale=config.reward_scale)

    # Starting a host and its run takes about a second, as long as a short
    # run itself. So the next run's host is started on a background thread
    # while this one plays (it's a separate process; the thread only waits).
    starter = ThreadPoolExecutor(max_workers=1)

    def start_next():
        character = rng.choice(CHARACTERS)
        seed = f"{env_id}-{rng.getrandbits(48):012x}"
        return character, seed, starter.submit(launch, make_host, character, seed, ascension)

    upcoming = start_next()
    episode = 0
    while True:
        character, seed, started = upcoming
        upcoming = start_next()
        host = None
        try:
            host, first_state = started.result()
            result = play_run(actor, host, character, seed, ascension, first_state=first_state)
        except HostError as e:
            actor.cut_off()
            result = {"character": character, "seed": seed, "outcome": "host_error", "error": str(e)[:300],
                      "steps": actor.episode_steps, "reward": round(actor.episode_reward, 2)}
        finally:
            if host is not None:
                host.close()
        result["episode"] = episode
        print(f"Environment {env_id}: episode {episode} {result['outcome']}: {character} "
              f"act={result.get('act')} floor={result.get('floor')} reward={result['reward']:.1f} "
              f"steps={result['steps']}" + (f" ({result['error']})" if "error" in result else ""), flush=True)
        if episode_log:
            with open(episode_log, "a") as f:
                f.write(json.dumps(result) + "\n")
        episode += 1
