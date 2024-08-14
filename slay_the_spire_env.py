import gymnasium as gym
import numpy as np
from gymnasium import spaces
from tokenizers import power_tokenizer, relic_tokenizer

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
        self.commands = ['start', 'potion', 'play', 'end', 'proceed', 'return', 'choose', 'confirm', "leave"]
        self.action_space, self.actions = self.create_action_space()

        power_space = spaces.Dict({
            "amount": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            "just_applied": spaces.Discrete(2),
            "id": spaces.Discrete(len(power_tokenizer.word_index) + 1)
        })

        relic_space = spaces.Dict({
            "name": spaces.Discrete(len(relic_tokenizer.word_index) + 1),
            "counter": spaces.Box(low=-1, high=100, shape=(1,), dtype=np.float32)
        })

        player_space = spaces.Dict({
            "current_hp": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "max_hp": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "block": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "energy": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "orbs": spaces.MultiBinary(10),
            "powers": spaces.Tuple([power_space] * 10),
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

        map_node_space = spaces.Dict({
            "symbol": spaces.Discrete(6),  # ?, $, T, M, E, R represented by 0-5
            "x": spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32),
            "y": spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32),
            "children": spaces.Tuple([spaces.Tuple([spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32), 
                                                   spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32)])] * 3)  # Max 3 children nodes per node
        })

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
                "powers": spaces.Tuple([power_space] * 10),
            })] * 5),
            "map": spaces.Tuple([map_node_space] * 51),
            "screen_type": spaces.Discrete(14),
            "relics": spaces.Sequence(relic_space)
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
        actions.extend(['END', 'PROCEED', 'RETURN', 'CONFIRM', "LEAVE"])
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

                # Reward for making a choice
                previous_choices = previous_game_state.get('choice_list', [])
                current_choices = current_game_state.get('choice_list', [])
                if previous_choices != current_choices:
                    reward += 5

                # Check for combat state changes (monsters' health)
                previous_combat_state = previous_game_state.get('combat_state', {})
                current_combat_state = current_game_state.get('combat_state', {})
                previous_monsters = previous_combat_state.get('monsters', [])
                current_monsters = current_combat_state.get('monsters', [])
                
                # Reward for doing damage and defeating a monster
                for prev_monster, curr_monster in zip(previous_monsters, current_monsters):
                    if prev_monster.get('current_hp', 0) > curr_monster.get('current_hp', 0):
                        reward += (prev_monster['current_hp'] - curr_monster['current_hp'])
                    if prev_monster.get('current_hp', 0) > 0 and curr_monster.get('current_hp', 0) <= 0:
                        reward += 50

                # Penalty for taking damage
                previous_hp = previous_game_state.get('player', {}).get('current_hp', 0)
                current_hp = current_game_state.get('player', {}).get('current_hp', 0)
                if current_hp < previous_hp:
                    reward -= (previous_hp - current_hp)
                
                # Check for floor progression
                if current_game_state.get('floor', 0) > previous_game_state.get('floor', 0):
                    reward += 100
                
                # Additional reward for potion use
                if self.actions[self.previous_action].startswith('POTION Use'):
                    reward += 5

                if self.actions[self.previous_action].startswith('POTION Discard'):
                    reward -= 5

                # Penalize for ending the turn with playable cards
                if self.actions[self.previous_action] == 'END':
                    hand = previous_combat_state.get('hand', [])
                    playable_cards = [card for card in hand if card.get('is_playable')]
                    if playable_cards:
                        reward -= 10  # Apply a small penalty for ending the turn with playable cards

                # Reward for acquiring a relic
                previous_relics = previous_game_state.get('relics', [])
                current_relics = current_game_state.get('relics', [])
                if len(current_relics) > len(previous_relics):
                    reward += 30  # Adjust the reward value as you see fit

                # Reward/Penalty for gold changes
                previous_gold = previous_game_state.get('gold', 0)
                current_gold = current_game_state.get('gold', 0)
                gold_difference = current_gold - previous_gold
                if gold_difference > 0:
                    reward += (gold_difference / 10)  # 1 point for each 10 gold gained
                elif gold_difference < 0:
                    reward += (gold_difference * 0.05)  # -0.05 points for each gold lost

                # Reward for adding a card to the deck
                previous_deck = previous_game_state.get('deck', [])
                current_deck = current_game_state.get('deck', [])
                if len(current_deck) > len(previous_deck):
                    new_card = current_deck[-1]  # Assuming the new card is added at the end
                    rarity = new_card.get('rarity', 'COMMON').upper()  # Default to 'COMMON' if rarity is not found
                    if rarity == 'COMMON':
                        reward += 3
                    elif rarity == 'UNCOMMON':
                        reward += 4.3
                    elif rarity == 'RARE':
                        reward += 10

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

        # Handle Potion actions
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
                if potion['requires_target']:
                    if len(parts) < 4:
                        invalid_action_mask[i] = True
                        continue
                    
                    target_index = int(parts[3])
                    monsters = game_state.get('combat_state', {}).get('monsters', [])
                    valid_monster_indices = [idx for idx, monster in enumerate(monsters) if not monster.get('is_gone', False)]
                    
                    if target_index not in valid_monster_indices:
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
        
        # Prevent "RETURN" action immediately after "PROCEED"
        if self.previous_action is not None:
            previous_command = self.actions[self.previous_action].split()[0].lower()
            if previous_command == 'proceed':
                for i, action in enumerate(self.actions):
                    if action.split()[0].lower() == 'return':
                        invalid_action_mask[i] = True
                    if action.split()[0].lower() == 'leave':
                        invalid_action_mask[i] = True

        # Handle combat-related actions
        combat_state = game_state.get('combat_state', None)
        if not combat_state:
            # If combat_state doesn't exist, restrict combat-related actions
            for i, action in enumerate(self.actions):
                if action.startswith('PLAY'):
                    invalid_action_mask[i] = True
            return invalid_action_mask

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