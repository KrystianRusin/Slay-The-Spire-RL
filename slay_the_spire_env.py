import gymnasium as gym
import copy
import numpy as np
from gymnasium import spaces
from util.vocabularies import screen_type_vocab
import random
from typing import NamedTuple, Optional

from observations.player_observations import get_player_observation
from observations.hand_observations import get_hand_observation
from observations.monster_observations import get_monster_observation
from observations.map_observations import get_map_observation
from observations.potion_observations import get_potion_observation
from observations.relic_observations import get_relic_observation
from observations.extra_info_observations import get_extra_info_observation
from observations.deck_observations import get_deck_observation
from observations.screen_observations import get_screen_observation

# After any of these, RETURN would step straight back to the screen just left.
LOOP_BACK_AFTER = {"proceed", "choose", "return"}


class Action(NamedTuple):
    """One entry of the action table, parsed from its command text."""
    text: str
    command: str
    use: Optional[bool] = None
    index: Optional[int] = None
    target: Optional[int] = None


def parse_action(text):
    """Parse a command such as 'POTION Use 0 1' or 'PLAY 3' into an Action.

    Card numbers are 1-based in the command and 0-based in the record.
    """
    words = text.split()
    command = words[0].lower()
    if command == "potion":
        target = int(words[3]) if len(words) == 4 else None
        return Action(text, command, use=words[1] == "Use", index=int(words[2]), target=target)
    if command == "play":
        target = int(words[2]) if len(words) == 3 else None
        return Action(text, command, index=int(words[1]) - 1, target=target)
    if command == "choose":
        return Action(text, command, index=int(words[1]))
    return Action(text, command)


def combat_turn(state):
    """The combat turn number in a socket payload, or None outside combat."""
    game_state = (state or {}).get("game_state") or {}
    combat_state = game_state.get("combat_state") or {}
    return combat_state.get("turn")


def potion_is_legal(action, game_state, live_targets):
    potions = game_state.get("potions", [])
    if action.index >= len(potions):
        return False
    potion = potions[action.index]
    if not action.use:
        return bool(potion["can_discard"]) and action.target is None
    if not potion["can_use"]:
        return False
    if potion["requires_target"]:
        return action.target in live_targets
    return action.target is None


def play_is_legal(action, combat_state, live_targets):
    hand = combat_state.get("hand", [])
    if action.index >= len(hand):
        return False
    card = hand[action.index]
    if not card["is_playable"]:
        return False
    if card["has_target"]:
        return action.target in live_targets
    return action.target is None


def choice_is_legal(action, game_state):
    choices = game_state.get("choice_list", [])
    if not choices:
        return True
    if action.index >= len(choices):
        return False
    if game_state.get("screen_type") == "COMBAT_REWARD" and "potion" in choices[action.index].lower():
        potions = game_state.get("potions", [])
        return any(potion["id"] == "Potion Slot" for potion in potions)
    return True


class SlayTheSpireEnv(gym.Env):
    def __init__(self, initial_state):
        super(SlayTheSpireEnv, self).__init__()
        self.state = initial_state
        self.previous_state = None
        self.previous_action = None
        self.curr_action = None
        self.action_taken = False
        self.recent_actions = []  # Store recent actions to avoid loops
        self.recent_action_limit = 5
        self.commands = ['start', 'potion', 'play', 'end', 'proceed', 'return', 'choose', 'confirm', "leave"]
        self.action_space, self.actions = self.create_action_space()
        self.action_ids = {action.text: i for i, action in enumerate(self.actions)}
        self.actions_by_command = {}
        for i, action in enumerate(self.actions):
            self.actions_by_command.setdefault(action.command, []).append(i)

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
        return spaces.Discrete(len(actions)), [parse_action(action) for action in actions]

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
            high=np.array([100, 100, 1000, 20, screen_type_vocab.max_id]),  # Upper bounds
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
            player_observation = np.zeros((24,), dtype=np.float32)
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
        deck_observation = get_deck_observation(game_state)
        screen_observation = get_screen_observation(game_state)

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

        # Flatten the initial observation from the state
        observation = self.flatten_observation(self.state)

        # Return the observation and an empty dictionary (or any relevant reset info)
        return observation, {}

    def step(self, action):
        self.previous_action = self.curr_action
        self.curr_action = action

        # Add the action to the recent action list
        if len(self.recent_actions) >= self.recent_action_limit:
            self.recent_actions.pop(0)
        self.recent_actions.append(self.actions[action].text)

        if combat_turn(self.state) is not None and self.actions[action].command != "end":
            self.action_taken = True

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

    def update_game_state(self, state):
        self.previous_state = copy.deepcopy(self.state) 
        self.state = state
        if combat_turn(state) != combat_turn(self.previous_state):
            self.action_taken = False

    def calculate_reward(self):
        reward = 0
        # Check if previous_state and current state exist
        if self.previous_state is None or self.state is None or self.previous_action is None:
            return reward
        taken = self.actions[self.previous_action]

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
                    print("Monster Damage Reward ", taken.text)
                    max_hp = curr_monster.get('max_hp', 1)
                    health_diff = prev_monster.get('current_hp', 0) - curr_monster.get('current_hp', 0)
                    percentage_damage = health_diff / max_hp
                    reward += percentage_damage * 10
                    if curr_monster.get('current_hp', 0) == 0 and prev_monster.get('current_hp', 0) > 0:
                        print("Monster Kill Reward ", taken.text)
                        reward += 20

        if previous_game_state.get("screen_type") == "NONE" and current_game_state.get("screen_type") == "COMBAT_REWARD":
            print("Combat Ended Reward ")
            reward += 40

        # Penalty for taking damage
        previous_hp = previous_game_state.get('current_hp', 0)
        current_hp = current_game_state.get('current_hp', 0)
        if current_hp < previous_hp:
            print("HP Damage Penalty ", taken.text)
            reward -= (previous_hp - current_hp) * 3
                
        # Check for floor progression
        if current_game_state.get('floor', 0) > previous_game_state.get('floor', 0):
            print("Floor Climbing Reward ", taken.text)
            reward += 10
                
        # Additional reward for potion use
        if taken.command == "potion" and taken.use:
            print("Potion Use Reward ", taken.text)
            reward += 10

        if taken.command == "potion" and not taken.use:
            print("Potion Discard Penalty ", taken.text)
            reward -= 10

        # Reward for acquiring a relic
        previous_relics = previous_game_state.get('relics', [])
        current_relics = current_game_state.get('relics', [])
        if len(current_relics) > len(previous_relics):
            print("Relic taken reward ", taken.text)
            reward += 50  # Adjust the reward value as you see fit

        # Reward/Penalty for gold changes
        previous_gold = previous_game_state.get('gold', 0)
        current_gold = current_game_state.get('gold', 0)
        gold_difference = current_gold - previous_gold
        if gold_difference > 0:
            print("Gold Gained Reward ", taken.text)
            reward += (gold_difference / 10)  # 1 point for each 10 gold gained
        elif gold_difference < 0:
            print("Gold Lost Penalty ", taken.text)
            reward += (gold_difference * 0.05)  # -0.05 points for each gold lost

        # Reward for adding a card to the deck
        previous_deck = previous_game_state.get('deck', [])
        current_deck = current_game_state.get('deck', [])
        if len(current_deck) > len(previous_deck):
            new_card = current_deck[-1]  # Assuming the new card is added at the end
            rarity = new_card.get('rarity', 'COMMON').upper()  # Default to 'COMMON' if rarity is not found
            if rarity == 'COMMON':
                print("Common Card Reward ", taken.text)
                reward += 3
            elif rarity == 'UNCOMMON':
                print("Uncommon Card Reward", taken.text)
                reward += 4.3
            elif rarity == 'RARE':
                print("Rare Card Reward ", taken.text)
                reward += 10

        # Reward for removing CURSE cards from the deck
        previous_curse_count = sum(1 for card in previous_deck if card.get('rarity', '').upper() == 'CURSE')
        current_curse_count = sum(1 for card in current_deck if card.get('rarity', '').upper() == 'CURSE')
        if current_curse_count < previous_curse_count:
            print("Curse Removal Reward ", taken.text)
            reward += 15
        
        # Small penalty per action to encourage efficiency
        reward -= 0.1

        # Return the calculated reward
        return reward
    
    def get_valid_action_mask(self, state):
        """Boolean mask over self.actions, True where the action is legal in state.

        Legality comes from the game: the commands it offers, the cards, potions,
        targets and choices in the state. The RETURN loop guard and the rule against
        ending a turn before playing are preferences on top, dropped whenever they
        would leave no action legal.
        """
        valid = np.zeros(len(self.actions), dtype=bool)

        game_state = state.get("game_state")
        if not game_state:
            valid[self.actions_by_command["start"]] = True
            return valid

        combat_state = game_state.get("combat_state") or {}
        live_targets = {
            i for i, monster in enumerate(combat_state.get("monsters", []))
            if not monster.get("is_gone", False)
        }
        legality = {
            "potion": lambda action: potion_is_legal(action, game_state, live_targets),
            "play": lambda action: play_is_legal(action, combat_state, live_targets),
            "choose": lambda action: choice_is_legal(action, game_state),
        }

        for command in set(state.get("available_commands", [])):
            is_legal = legality.get(command, lambda action: True)
            for i in self.actions_by_command.get(command, []):
                valid[i] = is_legal(self.actions[i])

        discouraged = np.zeros_like(valid)
        if self.curr_action is not None:
            last_command = self.actions[self.curr_action].command
            if last_command in LOOP_BACK_AFTER:
                discouraged[self.actions_by_command["return"]] = True

        if not self.action_taken and valid[self.actions_by_command["play"]].any():
            discouraged[self.actions_by_command["end"]] = True

        preferred = valid & ~discouraged
        return preferred if preferred.any() else valid

    def check_if_done(self):
        game_state = self.state.get("game_state", None)
        if not game_state:
            return False
        return self.state['game_state'].get('screen_type') == "GAME_OVER"

