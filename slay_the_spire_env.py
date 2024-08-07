import gymnasium as gym
import json
import numpy as np
from gymnasium import spaces
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers.legacy import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

#TODO Game loop

class SlayTheSpireEnv(gym.Env):
    def __init__(self, initial_state):
        super(SlayTheSpireEnv, self).__init__()

        self.state = initial_state
        self.previous_state = None
        self.previous_action = None
        self.curr_action = None

        # Define the available commands
        self.commands = ['start', 'potion', 'play', 'end', 'proceed', 'return', 'choose']

        # Define all possible actions
        self.action_space, self.actions = self.create_action_space()

        # Define observation space
        self.observation_space = spaces.Dict({
            'ready_for_command': spaces.Discrete(2),  # True or False
            'in_game': spaces.Discrete(2),  # True or False
            'screen_type': spaces.Discrete(10),  # Assuming 10 different screen types
            'game_state': spaces.Dict({
                'choice_list': spaces.Sequence(spaces.Text(max_length=100)),  # Assuming a sequence of strings
                'combat_state': spaces.Dict({
                    'draw_pile': spaces.Sequence(spaces.Dict({
                        'exhausts': spaces.Discrete(2),
                        'is_playable': spaces.Discrete(2),
                        'cost': spaces.Discrete(7),  # 0-5 + 1 special value for "X"
                        'name': spaces.Text(max_length=100),  # Assuming max length of name
                        'id': spaces.Text(max_length=100),  # Assuming max length of id
                        'type': spaces.Discrete(5),  # "ATTACK", "SKILL", "POWER", "CURSE", "STATUS"
                        'ethereal': spaces.Discrete(2),
                        'uuid': spaces.Text(max_length=100),  # UUID length
                        'upgrades': spaces.Discrete(2),  # 0 or 1
                        'rarity': spaces.Discrete(2),  # "BASIC", "SPECIAL"
                        'has_target': spaces.Discrete(2)
                    })),
                    'discard_pile': spaces.Sequence(spaces.Dict({
                        'exhausts': spaces.Discrete(2),
                        'is_playable': spaces.Discrete(2),
                        'cost': spaces.Discrete(7),  # 0-5 + 1 special value for "X"
                        'name': spaces.Text(max_length=100),  # Assuming max length of name
                        'id': spaces.Text(max_length=100),  # Assuming max length of id
                        'type': spaces.Discrete(5),  # "ATTACK", "SKILL", "POWER", "CURSE", "STATUS"
                        'ethereal': spaces.Discrete(2),
                        'uuid': spaces.Text(max_length=100),  # UUID length
                        'upgrades': spaces.Discrete(2),  # 0 or 1
                        'rarity': spaces.Discrete(2),  # "BASIC", "SPECIAL"
                        'has_target': spaces.Discrete(2)
                    })),
                    'exhaust_pile': spaces.Sequence(spaces.Dict({
                        'exhausts': spaces.Discrete(2),
                        'is_playable': spaces.Discrete(2),
                        'cost': spaces.Discrete(7),  # 0-5 + 1 special value for "X"
                        'name': spaces.Text(max_length=100),  # Assuming max length of name
                        'id': spaces.Text(max_length=100),  # Assuming max length of id
                        'type': spaces.Discrete(5),  # "ATTACK", "SKILL", "POWER", "CURSE", "STATUS"
                        'ethereal': spaces.Discrete(2),
                        'uuid': spaces.Text(max_length=100),  # UUID length
                        'upgrades': spaces.Discrete(2),  # 0 or 1
                        'rarity': spaces.Discrete(2),  # "BASIC", "SPECIAL"
                        'has_target': spaces.Discrete(2)
                    })),
                    'cards_discarded_this_turn': spaces.Box(low=0, high=np.inf, shape=()),
                    'times_damaged': spaces.Box(low=0, high=np.inf, shape=()),
                    'monsters': spaces.Sequence(spaces.Dict({
                        'is_gone': spaces.Discrete(2),
                        'move_hits': spaces.Box(low=0, high=np.inf, shape=()),
                        'move_base_damage': spaces.Box(low=0, high=np.inf, shape=()),
                        'half_dead': spaces.Discrete(2),
                        'move_adjusted_damage': spaces.Box(low=-np.inf, high=np.inf, shape=()),
                        'max_hp': spaces.Box(low=0, high=np.inf, shape=()),
                        'intent': spaces.Discrete(10),  # Assuming 10 different intents
                        'move_id': spaces.Box(low=0, high=np.inf, shape=()),
                        'name': spaces.Text(max_length=100),
                        'current_hp': spaces.Box(low=0, high=np.inf, shape=()),
                        'block': spaces.Box(low=0, high=np.inf, shape=()),
                        'id': spaces.Text(max_length=100),
                        'powers': spaces.Sequence(spaces.Dict({
                            'name': spaces.Text(max_length=100),
                            'amount': spaces.Box(low=0, high=np.inf, shape=())
                        }))
                    })),
                    'turn': spaces.Box(low=0, high=np.inf, shape=()),
                    'limbo': spaces.Sequence(spaces.Dict({
                        'exhausts': spaces.Discrete(2),
                        'is_playable': spaces.Discrete(2),
                        'cost': spaces.Discrete(7),  # 0-5 + 1 special value for "X"
                        'name': spaces.Text(max_length=100),  # Assuming max length of name
                        'id': spaces.Text(max_length=100),  # Assuming max length of id
                        'type': spaces.Discrete(5),  # "ATTACK", "SKILL", "POWER", "CURSE", "STATUS"
                        'ethereal': spaces.Discrete(2),
                        'uuid': spaces.Text(max_length=100),  # UUID length
                        'upgrades': spaces.Discrete(2),  # 0 or 1
                        'rarity': spaces.Discrete(2),  # "BASIC", "SPECIAL"
                        'has_target': spaces.Discrete(2)
                    })),
                    'hand': spaces.Sequence(spaces.Dict({
                        'exhausts': spaces.Discrete(2),
                        'is_playable': spaces.Discrete(2),
                        'cost': spaces.Discrete(7),  # 0-5 + 1 special value for "X"
                        'name': spaces.Text(max_length=100),  # Assuming max length of name
                        'id': spaces.Text(max_length=100),  # Assuming max length of id
                        'type': spaces.Discrete(5),  # "ATTACK", "SKILL", "POWER", "CURSE", "STATUS"
                        'ethereal': spaces.Discrete(2),
                        'uuid': spaces.Text(max_length=100),  # UUID length
                        'upgrades': spaces.Discrete(2),  # 0 or 1
                        'rarity': spaces.Discrete(2),  # "BASIC", "SPECIAL"
                        'has_target': spaces.Discrete(2)
                    })),
                    'player': spaces.Dict({
                        'orbs': spaces.Sequence(spaces.Dict({
                            'name': spaces.Text(max_length=100),
                            'amount': spaces.Box(low=0, high=np.inf, shape=())
                        })),
                        'current_hp': spaces.Box(low=0, high=100, shape=()),  # Example HP range
                        'block': spaces.Box(low=0, high=100, shape=()),
                        'max_hp': spaces.Box(low=0, high=100, shape=()),
                        'powers': spaces.Sequence(spaces.Dict({
                            'name': spaces.Text(max_length=100),
                            'amount': spaces.Box(low=0, high=np.inf, shape=())
                        })),
                        'energy': spaces.Box(low=0, high=10, shape=())  # Example energy range
                    })
                }),
                'deck': spaces.Sequence(spaces.Dict({
                    'exhausts': spaces.Discrete(2),
                    'is_playable': spaces.Discrete(2),
                    'cost': spaces.Discrete(7),  # 0-5 + 1 special value for "X"
                    'name': spaces.Text(max_length=100),  # Assuming max length of name
                    'id': spaces.Text(max_length=100),  # Assuming max length of id
                    'type': spaces.Discrete(5),  # "ATTACK", "SKILL", "POWER", "CURSE", "STATUS"
                    'ethereal': spaces.Discrete(2),
                    'uuid': spaces.Text(max_length=100),  # UUID length
                    'upgrades': spaces.Discrete(2),  # 0 or 1
                    'rarity': spaces.Discrete(2),  # "BASIC", "SPECIAL"
                    'has_target': spaces.Discrete(2)
                })),
                'relics': spaces.Sequence(spaces.Dict({
                    'name': spaces.Text(max_length=100),
                    'id': spaces.Text(max_length=100),
                    'counter': spaces.Box(low=-np.inf, high=np.inf, shape=())
                })),
                'max_hp': spaces.Box(low=0, high=100, shape=()),
                'current_hp': spaces.Box(low=0, high=100, shape=()),
                'gold': spaces.Box(low=0, high=np.inf, shape=()),
                'floor': spaces.Box(low=0, high=100, shape=()),
                'ascension_level': spaces.Box(low=0, high=20, shape=()),
                'act_boss': spaces.Discrete(5),  # Assuming 5 different act bosses
                'action_phase': spaces.Discrete(5),  # Assuming 5 different action phases
                'act': spaces.Discrete(3),  # Assuming 3 acts
                'screen_name': spaces.Discrete(5),  # Assuming 5 different screen names
                'room_phase': spaces.Discrete(5),  # Assuming 5 different room phases
                'is_screen_up': spaces.Discrete(2),  # True or False
                'potions': spaces.Sequence(spaces.Dict({
                    'name': spaces.Text(max_length=100),
                    'requires_target': spaces.Discrete(2),
                    'can_use': spaces.Discrete(2),
                    'can_discard': spaces.Discrete(2)
                })),
                'class': spaces.Discrete(4),  # Assuming 4 classes
                'map': spaces.Sequence(spaces.Dict({
                    'symbol': spaces.Text(max_length=100),  # Assuming 1-2 character symbols
                    'children': spaces.Sequence(spaces.Dict({
                        'x': spaces.Box(low=0, high=20, shape=()),
                        'y': spaces.Box(low=0, high=20, shape=())
                    })),
                    'x': spaces.Box(low=0, high=20, shape=()),
                    'y': spaces.Box(low=0, high=20, shape=()),
                    'parents': spaces.Sequence(spaces.Dict({
                        'x': spaces.Box(low=0, high=20, shape=()),
                        'y': spaces.Box(low=0, high=20, shape=())
                    }))
                }))
            })
        })

    def create_action_space(self):
        actions = []

        # Add START commands with string player classes
        player_classes = ['IRONCLAD', 'SILENT', 'DEFECT', 'WATCHER']
        for player_class in player_classes:
            actions.append(f'START {player_class} 0')

        # Add POTION commands
        for use_discard in range(2):  # Use or Discard
            for potion_slot in range(5):  # 5 potion slots
                for target_index in range(8):  # 8 possible targets
                    actions.append(f'POTION {use_discard} {potion_slot} {target_index}')

        for use_discard in range(2):
            for potion_slot in range(5):
                actions.append(f'POTION {use_discard} {potion_slot}')

        # Add PLAY commands
        for card_index in range(1, 10):  # 10 possible cards
            for target_index in range(8):  # 8 possible targets
                actions.append(f'PLAY {card_index} {target_index}')

        for card_index in range(1, 10):
            actions.append(f'PLAY {card_index}')

        # Add simple commands with no arguments
        actions.extend(['END', 'PROCEED', 'RETURN'])

        # Add CHOOSE commands with a reasonable limit on choices
        for choice_index in range(20):  # 20 possible choices
            actions.append(f'CHOOSE {choice_index}')

        return spaces.Discrete(len(actions)), actions

    def reset(self):
        self.current_command = None
        self.current_args = {}
        self.previous_state = None
        return self.state

    def step(self, action, game_state):
        # Update the game state with the new game_state provided
        self.previous_state = self.state
        self.previous_action = self.curr_action

        self.curr_action = action
        self.state = game_state

        # Get the command based on the chosen action
        command_str = self.actions[self.curr_action]

        reward = self.calculate_reward()
        done = self.check_if_done()

        # Reset for the next command
        self.current_command = None
        self.current_args = {}

        return self.state, reward, done, {'command': command_str}

    def calculate_reward(self):
        reward = 0
        if self.previous_state and self.previous_action is not None:

            # Reward for making a choice
            previous_choices = self.previous_state['game_state'].get('choice_list', [])
            current_choices = self.state['game_state'].get('choice_list', [])
            
            if previous_choices != current_choices:
                reward += 5 
            
            # Reward for damaging monster and defeating a monster
            previous_monsters = self.previous_state['game_state']['combat_state'].get('monsters', [])
            current_monsters = self.state['game_state']['combat_state'].get('monsters', [])
            for prev_monster, curr_monster in zip(previous_monsters, current_monsters):
                if prev_monster['current_hp'] > curr_monster['current_hp']:
                    reward += (prev_monster['current_hp'] - curr_monster['current_hp'])  # Reward for reducing monster health
                if prev_monster['current_hp'] > 0 and curr_monster['current_hp'] <= 0:
                    reward += 10

            # Penalty for taking damage
            previous_hp = self.previous_state['game_state']['player']['current_hp']
            current_hp = self.state['game_state']['player']['current_hp']
            if current_hp < previous_hp:
                reward -= (previous_hp - current_hp)

            # Reward for progressing to the next floor
            if self.state['game_state']['floor'] > self.previous_state['game_state']['floor']:
                reward += 20  # Reward for progressing to the next floor

            # Reward for using a potion
            if self.actions[self.previous_action].startswith('POTION'):
                reward += 5  # Small reward for using a potion

        return reward
    
    def check_if_done(self):
        if self.state['game_state'].get('screen_type') == "GAME_OVER":
            return True
        return False

    def get_invalid_action_mask(self):
        # Initialize the mask to all False (all actions are initially valid)
        invalid_action_mask = np.zeros(len(self.actions), dtype=bool)

        available_commands = self.state.get('available_commands', [])

        # Mask out unavailable commands
        for i, action in enumerate(self.actions):
            command = action.split()[0].lower()
            if command not in available_commands:
                invalid_action_mask[i] = True

        # Mask out unavailable potion actions based on game state
        potions = self.state['game_state'].get('potions', [])
        for i, action in enumerate(self.actions):
            parts = action.split()
            if parts[0].lower() == 'potion':
                use_discard = int(parts[1])
                potion_slot = int(parts[2])

                if potion_slot >= len(potions):
                    # If potion slot is out of range, mask the action
                    invalid_action_mask[i] = True
                    continue
                
                potion = potions[potion_slot]
                if not potion['can_use'] and use_discard == 0:
                    # If potion cannot be used, mask the use action
                    invalid_action_mask[i] = True
                    continue
                if not potion['can_discard'] and use_discard == 1:
                    # If potion cannot be discarded, mask the discard action
                    invalid_action_mask[i] = True
                    continue
                if potion['requires_target'] and len(parts) < 4:
                    # If potion requires target but action doesn't provide one, mask the action
                    invalid_action_mask[i] = True
                    continue
                if not potion['requires_target'] and len(parts) == 4:
                    # If potion doesn't require target but action provides one, mask the action
                    invalid_action_mask[i] = True

        # Mask out unavailable play actions based on game state
        combat_state = self.state['game_state'].get('combat_state')
        if not combat_state:
            # If combat_state does not exist, mask all play commands
            for i, action in enumerate(self.actions):
                if action.startswith('PLAY'):
                    invalid_action_mask[i] = True
        else:
            hand = combat_state.get('hand', [])
            monsters = combat_state.get('monsters', [])
            valid_monster_indices = [i for i, monster in enumerate(monsters) if not monster['is_gone']]
            
            for i, action in enumerate(self.actions):
                parts = action.split()
                if parts[0].lower() == 'play':
                    card_index = int(parts[1]) - 1  # Convert to 0-indexed

                    if card_index >= len(hand):
                        # If card index is out of range, mask the action
                        invalid_action_mask[i] = True
                        continue
                    
                    card = hand[card_index]
                    if not card['is_playable']:
                        # If card is not playable, mask the action
                        invalid_action_mask[i] = True
                        continue
                    if card['has_target'] and len(parts) < 3:
                        # If card requires target but action doesn't provide one, mask the action
                        invalid_action_mask[i] = True
                        continue
                    if not card['has_target'] and len(parts) == 3:
                        # If card doesn't require target but action provides one, mask the action
                        invalid_action_mask[i] = True
                        continue
                    if len(parts) == 3:
                        target_index = int(parts[2])
                        if target_index not in valid_monster_indices:
                            # If the specified target is invalid, mask the action
                            invalid_action_mask[i] = True

        # Mask out unavailable choose actions based on game state
        choice_list = self.state['game_state'].get('choice_list', [])
        for i, action in enumerate(self.actions):
            parts = action.split()
            if parts[0].lower() == 'choose':
                choice_index = int(parts[1])

                if choice_index >= len(choice_list):
                    # If choice index is out of range, mask the action
                    invalid_action_mask[i] = True

        return invalid_action_mask

    def get_valid_actions(self):
        invalid_action_mask = self.get_invalid_action_mask()
        valid_actions = [action for idx, action in enumerate(self.actions) if not invalid_action_mask[idx]]
        print("Valid actions (filtered):", valid_actions)  # Debug print
        return valid_actions

    def apply_invalid_action_mask(self, logits):
        invalid_action_mask = self.get_invalid_action_mask()
        adjusted_logits = np.where(invalid_action_mask, -1e8, logits)
        return adjusted_logits
    
    def print_action_space_with_validity(self):
        invalid_action_mask = self.get_invalid_action_mask()
        for idx, action in enumerate(self.actions):
            valid = not invalid_action_mask[idx]
            print(f"{action} - Valid: {valid}")

class MaskedSlayTheSpireEnv(gym.Wrapper):
    def __init__(self, env):
        super(MaskedSlayTheSpireEnv, self).__init__(env)
        self.action_space = spaces.Discrete(env.action_space.n)
        self.observation_space = env.observation_space

    def step(self, action, game_state):
        state, reward, done, info = self.env.step(action, game_state)
        invalid_action_mask = self.env.get_invalid_action_mask()
        state['invalid_action_mask'] = invalid_action_mask
        return state, reward, done, info

class MaskedDQNAgent(DQNAgent):
    def forward(self, observation):
        state = observation[0]
        masked_env = observation[1]
        invalid_action_mask = masked_env.get('invalid_action_mask')
        logits = self.model.predict(state[None])
        masked_logits = masked_env.apply_invalid_action_mask(logits)
        action = np.argmax(masked_logits)
        return action

def build_model(state_shape, nb_actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + state_shape))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    return model

def main():
    with open("game_state.json", 'r') as f:
        game_state = json.load(f)

    env = SlayTheSpireEnv(game_state)
    env = MaskedSlayTheSpireEnv(env)

    # Convert hierarchical observation space to a flat one
    flat_state_shape = (sum([np.prod(space.shape) if isinstance(space, spaces.Box) else 1 for space in env.observation_space.spaces.values()]),)

    nb_actions = env.action_space.n

    model = build_model(flat_state_shape, nb_actions)
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = MaskedDQNAgent(model=model, nb_actions=nb_actions, memory=memory, policy=policy,
                         nb_steps_warmup=10, target_model_update=1e-2)

    dqn.compile(Adam(lr=1e-3), metrics=['mae'])


if __name__ == "__main__":
    main()
