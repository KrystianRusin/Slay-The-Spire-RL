import json
import numpy as np
from slay_the_spire_env import SlayTheSpireEnv, MaskedSlayTheSpireEnv
from masked_dqn_agent import MaskedDQNAgent
from build_model import build_model
from prepare_state_inputs import prepare_state_inputs

def main():
    with open("game_state.json", 'r') as f:
        game_state = json.load(f)

    env = SlayTheSpireEnv(game_state)
    env = MaskedSlayTheSpireEnv(env)

    nb_actions = env.action_space.n

    model = build_model(env.observation_space, nb_actions)
    model.summary()

    state = env.reset()
    invalid_action_mask = env.get_invalid_action_mask()
    
    # Prepare the state inputs for the model
    state_inputs = prepare_state_inputs(state)

    # Example of choosing an action
    action_values = model.predict(state_inputs)
    chosen_action = np.argmax(action_values[0])
    print("Chosen action index:", chosen_action)
    print("Chosen action:", env.actions[chosen_action])

    state, reward, done, info = env.step(chosen_action, game_state)
    print("State after chosen action:", state)
    print("Reward after chosen action:", reward)
    print("Done after chosen action:", done)
    print("Info after chosen action:", info)

if __name__ == "__main__":
    main()
