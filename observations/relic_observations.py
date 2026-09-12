import numpy as np
from util.vocabularies import relic_vocab

def get_relic_observation(game_state):
    relics = game_state.get("relics", [])
    max_relics = 30
    relic_observation = []

    for relic in relics[:max_relics]:
        relic_token = relic_vocab.id_of(relic["name"])
        relic_observation.append([float(relic_token), float(relic.get("counter", -1))])

    while len(relic_observation) < max_relics:
        relic_observation.append([0.0, 0.0])

    return np.array(relic_observation, dtype=np.float32)
