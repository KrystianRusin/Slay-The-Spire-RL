import gymnasium as gym
import json
import numpy as np
from gymnasium import spaces
import torch
from torch.distributions.categorical import Categorical

# TODO Update step function so that it takes in the action and remove all the other bullshit from previous attempts at implementation
# TODO Implement reward function

class SlayTheSpireEnv(gym.Env):
    def __init__(self, initial_state):
        super(SlayTheSpireEnv, self).__init__()

        self.state = initial_state
        self.previous_state = None
        self.current_command = None
        self.current_args = {}

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
        final_command = None
        action_str = self.actions[action]

        # Update the game state with the new game_state provided
        self.previous_state = self.state
        self.state = game_state

        reward = self.calculate_reward()
        done = self.check_if_done()

        # Reset for the next command
        self.current_command = None
        self.current_args = {}

        return self.state, reward, done, {'final_command': final_command}

    def calculate_reward(self):
        # Calculate the reward for the current state
        # This is a placeholder; implement the actual reward calculation logic
        return 0

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
        adjusted_logits = torch.where(torch.tensor(invalid_action_mask, dtype=torch.bool), torch.tensor(-1e+8), logits)
        return adjusted_logits
    
    def print_action_space_with_validity(self):
        invalid_action_mask = self.get_invalid_action_mask()
        for idx, action in enumerate(self.actions):
            valid = not invalid_action_mask[idx]
            print(f"{action} - Valid: {valid}")

def main():
    with open("game_state.json", 'r') as f:
        game_state = json.load(f)

    env = SlayTheSpireEnv(game_state)

    print("Full action space with validity:")
    env.print_action_space_with_validity()

    # Example usage:
    logits = torch.tensor([1.] * len(env.actions), requires_grad=True)
    adjusted_logits = env.apply_invalid_action_mask(logits)
    print("Adjusted logits:", adjusted_logits)

    # Sample an action from the adjusted logits
    action_dist = Categorical(logits=adjusted_logits)
    action = action_dist.sample()
    print("Selected action:", env.actions[action])

if __name__ == "__main__":
    main()
