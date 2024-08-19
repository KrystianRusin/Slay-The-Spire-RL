import numpy as np

def get_default_observation(self):
    player_observation = np.zeros((24,), dtype=np.float32)  # Adjusted to (24,) for 4 stats + 20 powers
    hand_observation = np.zeros((10, 8), dtype=np.float32)
    monster_observation = np.zeros((5, 30), dtype=np.float32)
    map_observation = np.zeros((100, 4), dtype=np.float32)
    relic_observation = np.zeros((30, 2), dtype=np.float32)
    extra_info = np.zeros((5,), dtype=np.float32)

    return {
        "player": player_observation,
        "hand": hand_observation,
        "monsters": monster_observation,
        "map": map_observation,
        "relics": relic_observation,
        "extra_info": extra_info
    }