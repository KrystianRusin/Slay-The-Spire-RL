import gymnasium as gym
import numpy as np
from gymnasium import spaces

class SlayTheSpireEnv(gym.Env):
    def __init__(self, initial_state):
        super(SlayTheSpireEnv, self).__init__()
        self.state = initial_state
        self.previous_state = None
        self.previous_action = None
        self.curr_action = None
        self.commands = ['start', 'potion', 'play', 'end', 'proceed', 'return', 'choose']
        self.action_space, self.actions = self.create_action_space()

        player_space = spaces.Dict({
            "current_hp": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "max_hp": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "block": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "energy": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "orbs": spaces.MultiBinary(10),
        })

        card_space = spaces.Dict({
            "exhausts": spaces.Discrete(2),
            "is_playable": spaces.Discrete(2),
            "cost": spaces.Box(low=0, high=5, shape=(1,), dtype=np.float32),
            "name": spaces.Discrete(378), 
            "type": spaces.Discrete(5),  
            "ethereal": spaces.Discrete(2),
            "upgrades": spaces.Box(low=0, high=5, shape=(1,), dtype=np.float32),
            "rarity": spaces.Discrete(6), 
            "has_target": spaces.Discrete(2)
        })

        hand_space = spaces.Tuple([card_space] * 10)

        self.observation_space = spaces.Dict({
            "player": player_space,
            "hand": spaces.Tuple([hand_space] * 10),
            "deck": spaces.Sequence(card_space),
            "draw_pile": spaces.Sequence(card_space),
            "discard_pile": spaces.Sequence(card_space),
            "exhaust_pile": spaces.Sequence(card_space),
            "screen_type": spaces.Discrete(10),
            "monsters": spaces.Tuple([spaces.Dict({
                "is_gone": spaces.Discrete(2),
                "move_hits": spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32),
                "move_base_damage": spaces.Box(low=0, high=50, shape=(1,), dtype=np.float32),
                "half_dead": spaces.Discrete(2),
                "move_adjusted_damage": spaces.Box(low=0, high=50, shape=(1,), dtype=np.float32),
                "max_hp": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "intent": spaces.Discrete(13),
                "current_hp": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "block": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "id": spaces.Discrete(53),
            })] * 5),
            "screen_type": spaces.Discrete(14)
        })

    
    def create_action_space(self):
        actions = []
        player_classes = ['IRONCLAD']
        for player_class in player_classes:
            actions.append(f'START {player_class} 0')
        for use_discard in ['Use', 'Discard']:
            for potion_slot in range(5):
                for target_index in range(5):
                    actions.append(f'POTION {use_discard} {potion_slot} {target_index}')
        for use_discard in ['Use', 'Discard']:
            for potion_slot in range(5):
                actions.append(f'POTION {use_discard} {potion_slot}')
        for card_index in range(1, 10):
            for target_index in range(5):
                actions.append(f'PLAY {card_index} {target_index}')
        for card_index in range(1, 10):
            actions.append(f'PLAY {card_index}')
        actions.extend(['END', 'PROCEED', 'RETURN'])
        for choice_index in range(20):
            actions.append(f'CHOOSE {choice_index}')
        return spaces.Discrete(len(actions)), actions


    def reset(self, seed=None, options=None):
        self.current_command = None
        self.current_args = {}
        self.previous_state = None
        self.state = self.state  # Initialize the state to the initial state
        invalid_action_mask = self.get_invalid_action_mask()
        self.state['invalid_action_mask'] = invalid_action_mask
        return self.state

    def step(self, action, game_state):
        # Store the current state and action as the previous ones for reference
        self.previous_state = self.state
        self.previous_action = self.curr_action
        self.curr_action = action

        # Attempt to update the state with the new game state
        if 'error' in game_state:
            # If the new state is an error, retain the previous state
            print("Error in game state detected. Retaining previous state.")
            self.state = self.previous_state
        else:
            # Update to the new game state
            self.state = game_state

        # Get the command string corresponding to the current action
        command_str = self.actions[self.curr_action]

        # Calculate the reward based on the action taken and state transition
        reward = self.calculate_reward()

        # Check if the episode is done
        done = self.check_if_done()

        # Clear the current command and arguments for the next step
        self.current_command = None
        self.current_args = {}

        # Generate and update the invalid action mask for the current state
        invalid_action_mask = self.get_invalid_action_mask()
        self.state['invalid_action_mask'] = invalid_action_mask

        # Return the updated state, reward, done flag, and the command string
        return self.state, reward, done, {command_str}


    def calculate_reward(self):
        reward = 0
        previous_game_state = self.previous_state.get('game_state')
        current_game_state = self.state.get('game_state')
        
        if previous_game_state and current_game_state:
            if self.previous_action is not None:
                # Check for choice list changes
                previous_choices = previous_game_state.get('choice_list', [])
                current_choices = current_game_state.get('choice_list', [])
                if previous_choices != current_choices:
                    reward += 5

                # Check for combat state changes (monsters' health)
                previous_combat_state = previous_game_state.get('combat_state', {})
                current_combat_state = current_game_state.get('combat_state', {})
                previous_monsters = previous_combat_state.get('monsters', [])
                current_monsters = current_combat_state.get('monsters', [])
                
                for prev_monster, curr_monster in zip(previous_monsters, current_monsters):
                    if prev_monster.get('current_hp', 0) > curr_monster.get('current_hp', 0):
                        reward += (prev_monster['current_hp'] - curr_monster['current_hp'])
                    if prev_monster.get('current_hp', 0) > 0 and curr_monster.get('current_hp', 0) <= 0:
                        reward += 10

                # Check for player HP changes
                previous_hp = previous_game_state.get('player', {}).get('current_hp', 0)
                current_hp = current_game_state.get('player', {}).get('current_hp', 0)
                if current_hp < previous_hp:
                    reward -= (previous_hp - current_hp)
                
                # Check for floor progression
                if current_game_state.get('floor', 0) > previous_game_state.get('floor', 0):
                    reward += 20
                
                # Additional reward for potion use
                if self.actions[self.previous_action].startswith('POTION Use'):
                    reward += 5

        return reward


    def check_if_done(self):
        game_state = self.state.get("game_state", None)
        if not game_state:
            return False
        if self.state['game_state'].get('screen_type') == "GAME_OVER":
            return True
        return False

    def get_invalid_action_mask(self):
        invalid_action_mask = np.zeros(len(self.actions), dtype=bool)

        # Process available commands
        available_commands = self.state.get('available_commands', [])
        for i, action in enumerate(self.actions):
            command = action.split()[0].lower()
            if command not in available_commands:
                invalid_action_mask[i] = True

        # Check if 'game_state' exists
        game_state = self.state.get('game_state', None)
        if not game_state:
            # If game_state doesn't exist, all actions are invalid
            return np.ones(len(self.actions), dtype=bool)

        potions = game_state.get('potions', [])
        for i, action in enumerate(self.actions):
            parts = action.split()
            if parts[0].lower() == 'potion':
                use_discard = 0 if parts[1].lower() == 'use' else 1
                potion_slot = int(parts[2])
                
                if potion_slot >= len(potions):
                    invalid_action_mask[i] = True
                    continue

                potion = potions[potion_slot]

                if not potion['can_use'] and use_discard == 0:
                    invalid_action_mask[i] = True
                    continue
                if not potion['can_discard'] and use_discard == 1:
                    invalid_action_mask[i] = True
                    continue

                if potion['requires_target'] and len(parts) < 4:
                    invalid_action_mask[i] = True
                    continue
                if not potion['requires_target'] and len(parts) == 4:
                    invalid_action_mask[i] = True

         # Handle choice-related actions
        choice_list = game_state.get('choice_list', [])
        if len(choice_list) != 0:
            for i, action in enumerate(self.actions):
                parts = action.split()
                if parts[0].lower() == 'choose':
                    choice_index = int(parts[1])
                    if choice_index >= len(choice_list):
                        invalid_action_mask[i] = True

             # Check if 'combat_state' exists
        combat_state = game_state.get('combat_state', None)
        if not combat_state:
            # If combat_state doesn't exist, restrict combat-related actions
            for i, action in enumerate(self.actions):
                if action.startswith('PLAY'):
                    invalid_action_mask[i] = True
            return invalid_action_mask

        # Handle combat-related actions
        hand = combat_state.get('hand', [])
        monsters = combat_state.get('monsters', [])
        valid_monster_indices = [i for i, monster in enumerate(monsters) if not monster['is_gone']]
        for i, action in enumerate(self.actions):
            parts = action.split()
            if parts[0].lower() == 'play':
                card_index = int(parts[1]) - 1
                if card_index >= len(hand):
                    invalid_action_mask[i] = True
                    continue
                card = hand[card_index]
                if not card['is_playable']:
                    invalid_action_mask[i] = True
                    continue
                if card['has_target'] and len(parts) < 3:
                    invalid_action_mask[i] = True
                    continue
                if not card['has_target'] and len(parts) == 3:
                    invalid_action_mask[i] = True
                    continue
                if len(parts) == 3:
                    target_index = int(parts[2])
                    if target_index not in valid_monster_indices:
                        invalid_action_mask[i] = True

        return invalid_action_mask

    def get_valid_actions(self):
        invalid_action_mask = self.get_invalid_action_mask()
        valid_actions = [action for idx, action in enumerate(self.actions) if not invalid_action_mask[idx]]
        print("Valid actions (filtered):", valid_actions)
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