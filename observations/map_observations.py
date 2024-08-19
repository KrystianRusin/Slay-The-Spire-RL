import numpy as np
from tokenizers import map_symbol_tokenizer

def get_map_observation(game_state):
    map_state = game_state.get("map", [])
    max_map_nodes = 100
    map_observation = []

    for node in map_state[:max_map_nodes]:
        symbol_sequence = map_symbol_tokenizer.texts_to_sequences([node["symbol"]])
        map_symbol_token = symbol_sequence[0][0] if symbol_sequence and symbol_sequence[0] else 0
        node_observation = [map_symbol_token, node["x"], node["y"], len(node.get("children", []))]
        map_observation.append(node_observation)

    while len(map_observation) < max_map_nodes:
        map_observation.append([0.0] * 4)

    return np.array(map_observation, dtype=np.float32)
