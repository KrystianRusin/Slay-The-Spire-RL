import numpy as np
from tokenizers import relic_tokenizer

def get_relic_observation(game_state):
    relics = game_state.get("relics", [])
    max_relics = 30
    relic_observation = []

    for relic in relics[:max_relics]:
        relic_token = relic_tokenizer.texts_to_sequences([relic["name"]])[0][0] if relic_tokenizer.texts_to_sequences([relic["name"]]) else 0
        relic_observation.append([float(relic_token), float(relic.get("counter", -1))])

    while len(relic_observation) < max_relics:
        relic_observation.append([0.0, 0.0])

    return np.array(relic_observation, dtype=np.float32)
