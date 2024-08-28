import gymnasium as gym
import copy
import numpy as np
from gymnasium import spaces
from tokenizers import screen_type_tokenizer

from observations.player_observations import get_player_observation
from observations.hand_observations import get_hand_observation
from observations.monster_observations import get_monster_observation
from observations.map_observations import get_map_observation
from observations.potion_observations import get_potion_observation
from observations.relic_observations import get_relic_observation
from observations.extra_info_observations import get_extra_info_observation
from observations.deck_observations import get_deck_observation
from observations.screen_observations import get_screen_observation

class SlayTheSpireEnv(gym.Env):
    def __init__(self, initial_state):
        super(SlayTheSpireEnv, self).__init__()
        self.state = initial_state
        self.previous_state = None
        self.previous_action = None
        self.curr_action = None
        self.action_taken = False  # Define the available commands and action space permutations
        self.recent_actions = []  # Store recent actions to avoid loops
        self.recent_action_limit = 5
        self.commands = ['start', 'potion', 'play', 'end', 'proceed', 'return', 'choose', 'confirm', "leave"]
        self.action_space, self.actions = self.create_action_space()

        # Define observation space (preserving the structure you provided)
        self.observation_space = self.create_observation_space()

    def create_action_space(self):
        actions = []
        player_classes = ['IRONCLAD', 'SILENT']
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

    def create_observation_space(self):

        # Player observation space (current_hp, max_hp, block, energy, powers)
        player_space = spaces.Box(low=0, high=1, shape=(4 + 20,), dtype=np.float32)

        # Hand observation space (max 10 cards, with 8 attributes per card)
        hand_space = spaces.Box(low=0, high=1, shape=(10, 8), dtype=np.float32)

        # Monster observation space (max 5 monsters, with 10 attributes per monster + 20 powers)
        monster_space = spaces.Box(low=0, high=1, shape=(5, 10 + 20), dtype=np.float32)

        # Map observation space (max 100 map nodes, with 4 attributes per node)
        map_space = spaces.Box(low=0, high=10, shape=(100, 4), dtype=np.float32)

        # Relics observation space (max 30 relics, with 2 attributes per relic)
        relic_space = spaces.Box(low=0, high=1, shape=(30, 2), dtype=np.float32)

        deck_space = spaces.Box(low=0, high=1, shape=(100, 8), dtype=np.float32)

        potion_space = spaces.Box(low=0, high=1, shape=(5, 4), dtype=np.float32)

        screen_space = spaces.Box(low=0, high=1, shape=(50,), dtype=np.float32)

        # Additional game state information (screen_type, deck size, etc.)
        extra_info_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0]),  # Lower bounds for each field
            high=np.array([100, 100, 1000, 20, len(screen_type_tokenizer.word_index)]),  # Upper bounds
            shape=(5,),
            dtype=np.float32
        )

        # Combine all the spaces into a single observation space
        combined_space = spaces.Dict({
            "player": player_space,
            "hand": hand_space,
            "monsters": monster_space,
            "deck": deck_space,
            "potion": potion_space,
            "map": map_space,
            "relics": relic_space,
            "screen": screen_space,
            "extra_info": extra_info_space
        })

        return combined_space

    def flatten_observation(self, state):
        if "game_state" not in state:
            player_observation = np.zeros((24,), dtype=np.float32)  # Adjusted to (24,) for 4 stats + 20 powers
            hand_observation = np.zeros((10, 8), dtype=np.float32)
            monster_observation = np.zeros((5, 30), dtype=np.float32)
            map_observation = np.zeros((100, 4), dtype=np.float32)
            relic_observation = np.zeros((30, 2), dtype=np.float32)
            extra_info = np.zeros((5,), dtype=np.float32)
            potion_observation = np.zeros((5, 4), dtype=np.float32)
            deck_observation = np.zeros((100, 8), dtype=np.float32) 
            screen_observation = np.zeros((50,), dtype=np.float32)

            return {
                "player": player_observation,
                "hand": hand_observation,
                "monsters": monster_observation,
                "map": map_observation,
                "relics": relic_observation,
                "deck": deck_observation,
                "potion": potion_observation,
                "screen": screen_observation, 
                "extra_info": extra_info
            }

        game_state = state["game_state"]
        combat_state = game_state.get("combat_state", None)

        player_observation = get_player_observation(game_state)
        potion_observation = get_potion_observation(game_state)
        monster_observation = get_monster_observation(game_state)
        map_observation = get_map_observation(game_state)
        relic_observation = get_relic_observation(game_state)
        extra_info_observation = get_extra_info_observation(game_state)
        hand_observation = get_hand_observation(combat_state)
        deck_observation = get_deck_observation(state)
        screen_observation = get_screen_observation(state)

        # Combine them into a full observation
        return {
            "player": player_observation,
            "hand": hand_observation,
            "potion": potion_observation,
            "deck": deck_observation,
            "monsters": monster_observation,
            "map": map_observation,
            "relics": relic_observation,
            "screen": screen_observation,
            "extra_info": extra_info_observation,
        }

    def reset(self, seed=None, options=None):
        # Reset internal variables
        self.current_command = None
        self.current_args = {}
        self.previous_state = None
        self.curr_action = None
        self.action_taken = False

        # Expect that the initial state is passed in via an external process
        if self.state is None:
            raise ValueError("Initial state must be provided by the external process.")

        observation = self.flatten_observation(self.state)

        return observation

    def step(self, action):
        self.previous_action = self.curr_action
        self.curr_action = action

        # Add the action to the recent action list
        if len(self.recent_actions) >= self.recent_action_limit:
            self.recent_actions.pop(0)
        self.recent_actions.append(self.actions[action])

        # Calculate the reward based on the action taken and state transition
        reward = self.calculate_reward()

        # Check if the episode is done
        done = self.check_if_done()

        # Clear the current command and arguments for the next step
        self.current_command = None
        self.current_args = {}

        # Flatten the observation based on the new game state
        observation = self.flatten_observation(self.state)

        return observation, reward, done, {}

    def get_valid_actions(self):
        available_commands = self.state.get('available_commands', [])
        valid_actions = [i for i, action in enumerate(self.actions) if action.split()[0].lower() in available_commands]
        return valid_actions

    def update_game_state(self, state):
        self.previous_state = copy.deepcopy(self.state) 
        self.state = state

    def calculate_reward(self):
        reward = 0
        invalid_action_mask = self.get_invalid_action_mask(self.previous_state)
        # Check if previous_state and current state exist
        if self.previous_state is None or self.state is None or self.previous_action is None:
            return reward

        previous_game_state = self.previous_state.get('game_state', None)
        current_game_state = self.state.get('game_state', None)

        # Check if game states exist
        if previous_game_state is None or current_game_state is None:
            return reward

        previous_combat_state = previous_game_state.get('combat_state', {})
        current_combat_state = current_game_state.get('combat_state', {})
        previous_monsters = previous_combat_state.get('monsters', [])
        current_monsters = current_combat_state.get('monsters', [])

        if len(previous_combat_state) > 0:
            for prev_monster, curr_monster in zip(previous_monsters, current_monsters):
                if curr_monster.get('current_hp', 0) < prev_monster.get('current_hp', 0):
                    print("Monster Damaged Reward ", self.actions[self.previous_action])
                    health_diff = prev_monster.get('current_hp', 0) - curr_monster.get('current_hp', 0)
                    reward += health_diff
                    if prev_monster.get('current_hp', 0) == 0:
                        print("Monster Defeated Reward", self.actions[self.previous_action])
                        reward += 20

        if self.previous_state.get("screen_type", None) == "NONE" and self.state.get("screen_type", None) == "COMBAT_REWARD":
            print("Combat Ended Reward ")
            reward += 40

        # Penalty for taking damage
        previous_hp = previous_game_state.get('current_hp', 0)
        current_hp = current_game_state.get('current_hp', 0)
        if current_hp < previous_hp:
            print("HP Damage Penalty ", self.actions[self.previous_action])
            reward -= (previous_hp - current_hp) * 3
                
        # Check for floor progression
        if current_game_state.get('floor', 0) > previous_game_state.get('floor', 0):
            print("Floor Climbing Reward ", self.actions[self.previous_action])
            reward += 10
                
        # Additional reward for potion use
        if self.actions[self.previous_action].startswith('POTION Use'):
            print("Potion Use Reward ", self.actions[self.previous_action])
            reward += 10

        if self.actions[self.previous_action].startswith('POTION Discard'):
            print("Potion Discard Penalty ", self.actions[self.previous_action])
            reward -= 20

        # Reward for acquiring a relic
        previous_relics = previous_game_state.get('relics', [])
        current_relics = current_game_state.get('relics', [])
        if len(current_relics) > len(previous_relics):
            print("Relic taken reward ", self.actions[self.previous_action])
            reward += 50  # Adjust the reward value as you see fit

        # Reward/Penalty for gold changes
        previous_gold = previous_game_state.get('gold', 0)
        current_gold = current_game_state.get('gold', 0)
        gold_difference = current_gold - previous_gold
        if gold_difference > 0:
            print("Gold Gained Reward ", self.actions[self.previous_action])
            reward += (gold_difference / 10)  # 1 point for each 10 gold gained
        elif gold_difference < 0:
            print("Gold Lost Penalty ", self.actions[self.previous_action])
            reward += (gold_difference * 0.05)  # -0.05 points for each gold lost

        # Reward for adding a card to the deck
        previous_deck = previous_game_state.get('deck', [])
        current_deck = current_game_state.get('deck', [])
        if len(current_deck) > len(previous_deck):
            new_card = current_deck[-1]  # Assuming the new card is added at the end
            rarity = new_card.get('rarity', 'COMMON').upper()  # Default to 'COMMON' if rarity is not found
            if rarity == 'COMMON':
                print("Common Card Reward ", self.actions[self.previous_action])
                reward += 3
            elif rarity == 'UNCOMMON':
                print("Uncommon Card Reward", self.actions[self.previous_action])
                reward += 4.3
            elif rarity == 'RARE':
                print("Rare Card Reward ", self.actions[self.previous_action])
                reward += 10

        # Reward for removing CURSE cards from the deck
        previous_curse_count = sum(1 for card in previous_deck if card.get('rarity', '').upper() == 'CURSE')
        current_curse_count = sum(1 for card in current_deck if card.get('rarity', '').upper() == 'CURSE')
        if current_curse_count < previous_curse_count:
            print("Curse Removal Reward ", self.actions[self.previous_action])
            reward += 15

        # Return the calculated reward
        return reward
    
    def get_invalid_action_mask(self, state):
        invalid_action_mask = np.zeros(len(self.actions), dtype=bool)

        # Process available commands
        available_commands = state.get('available_commands', [])
        for i, action in enumerate(self.actions):
            command = action.split()[0].lower()
            if command not in available_commands:
                invalid_action_mask[i] = True

        # Check if 'game_state' exists
        game_state = state.get('game_state', None)
        if not game_state:
            # If game_state doesn't exist, assume that only START commands are valid
            for i, action in enumerate(self.actions):
                if action not in ["START IRONCLAD 0", "START SILENT 0"]:
                    invalid_action_mask[i] = True
            return ~invalid_action_mask  # Invert mask before returning

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
            if previous_command == 'PROCEED' or previous_command == "CHOOSE" or previous_command == "RETURN":
                for i, action in enumerate(self.actions):
                    if action.split()[0].lower() == 'RETURN':
                        invalid_action_mask[i] = True
            if previous_command == 'leave':
                for i, action in enumerate(self.actions):
                    if action.startswith('CHOOSE'):
                        invalid_action_mask[i] = True
                        
        # Handle combat-related actions
        combat_state = game_state.get('combat_state', None)
        if not combat_state:
            # If combat_state doesn't exist, restrict combat-related actions
            for i, action in enumerate(self.actions):
                if action.startswith('PLAY'):
                    invalid_action_mask[i] = True
            return ~invalid_action_mask  # Invert mask and return
        
        hand = combat_state.get('hand', [])
        monsters = combat_state.get('monsters', [])
        has_playable_cards = any(card.get('is_playable') for card in hand)

        # If no action has been taken and there are playable cards, invalidate "END"
        if not self.action_taken and has_playable_cards:
            for i, action in enumerate(self.actions):
                if action.lower() == 'end':
                    invalid_action_mask[i] = True

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

        return ~invalid_action_mask


    def check_if_done(self):
        game_state = self.state.get("game_state", None)
        if not game_state:
            return False
        return self.state['game_state'].get('screen_type') == "GAME_OVER"
