import torch as th
import socket
import os
from collections import deque
from sb3_contrib.ppo_mask import MaskablePPO
from slay_the_spire_env import SlayTheSpireEnv
from model.custom_rollout_buffer import CustomRolloutBuffer

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = SlayTheSpireEnv({})
        env.seed(seed + rank)
        return env
    return _init