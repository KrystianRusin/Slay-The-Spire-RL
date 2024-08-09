import sys
import json
import time
def main():
     # Send ready signal to the communication mod (game)
    sys.stdout.write("ready\n")
    sys.stdout.flush()
    while True:
        game_state_json = sys.stdin.readline().strip()

        if game_state_json:
            # Generate a unique filename using the current timestamp
            filename = f"game_state_{int(time.time())}.json"
            
           # Write the game state to the file
            with open(filename, 'w') as file:
                json.dump(json.loads(game_state_json), file)



if __name__ == "__main__":
    main()


    # self.observation_space = spaces.Dict({
    #         'ready_for_command': spaces.Discrete(2),
    #         'in_game': spaces.Discrete(2),
    #         'screen_type': spaces.Discrete(10),
    #         'game_state': spaces.Dict({
    #             'choice_list': spaces.Sequence(spaces.Text(max_length=100)),
    #             'combat_state': spaces.Dict({
    #                 'draw_pile': spaces.Sequence(spaces.Dict({
    #                     'exhausts': spaces.Discrete(2),
    #                     'is_playable': spaces.Discrete(2),
    #                     'cost': spaces.Discrete(7),
    #                     'name': spaces.Text(max_length=100),
    #                     'id': spaces.Text(max_length=100),
    #                     'type': spaces.Discrete(5),
    #                     'ethereal': spaces.Discrete(2),
    #                     'uuid': spaces.Text(max_length=100),
    #                     'upgrades': spaces.Discrete(2),
    #                     'rarity': spaces.Discrete(2),
    #                     'has_target': spaces.Discrete(2)
    #                 })),
    #                 'discard_pile': spaces.Sequence(spaces.Dict({
    #                     'exhausts': spaces.Discrete(2),
    #                     'is_playable': spaces.Discrete(2),
    #                     'cost': spaces.Discrete(7),
    #                     'name': spaces.Text(max_length=100),
    #                     'id': spaces.Text(max_length=100),
    #                     'type': spaces.Discrete(5),
    #                     'ethereal': spaces.Discrete(2),
    #                     'uuid': spaces.Text(max_length=100),
    #                     'upgrades': spaces.Discrete(2),
    #                     'rarity': spaces.Discrete(2),
    #                     'has_target': spaces.Discrete(2)
    #                 })),
    #                 'exhaust_pile': spaces.Sequence(spaces.Dict({
    #                     'exhausts': spaces.Discrete(2),
    #                     'is_playable': spaces.Discrete(2),
    #                     'cost': spaces.Discrete(7),
    #                     'name': spaces.Text(max_length=100),
    #                     'id': spaces.Text(max_length=100),
    #                     'type': spaces.Discrete(5),
    #                     'ethereal': spaces.Discrete(2),
    #                     'uuid': spaces.Text(max_length=100),
    #                     'upgrades': spaces.Discrete(2),
    #                     'rarity': spaces.Discrete(2),
    #                     'has_target': spaces.Discrete(2)
    #                 })),
    #                 'cards_discarded_this_turn': spaces.Box(low=0, high=np.inf, shape=()),
    #                 'times_damaged': spaces.Box(low=0, high=np.inf, shape=()),
    #                 'monsters': spaces.Sequence(spaces.Dict({
    #                     'is_gone': spaces.Discrete(2),
    #                     'move_hits': spaces.Box(low=0, high=np.inf, shape=()),
    #                     'move_base_damage': spaces.Box(low=0, high=np.inf, shape=()),
    #                     'half_dead': spaces.Discrete(2),
    #                     'move_adjusted_damage': spaces.Box(low=-np.inf, high=np.inf, shape=()),
    #                     'max_hp': spaces.Box(low=0, high=np.inf, shape=()),
    #                     'intent': spaces.Discrete(10),
    #                     'move_id': spaces.Box(low=0, high=np.inf, shape=()),
    #                     'name': spaces.Text(max_length=100),
    #                     'current_hp': spaces.Box(low=0, high=np.inf, shape=()),
    #                     'block': spaces.Box(low=0, high=np.inf, shape=()),
    #                     'id': spaces.Text(max_length=100),
    #                     'powers': spaces.Sequence(spaces.Dict({
    #                         'name': spaces.Text(max_length=100),
    #                         'amount': spaces.Box(low=0, high=np.inf, shape=())
    #                     }))
    #                 })),
    #                 'turn': spaces.Box(low=0, high=np.inf, shape=()),
    #                 'limbo': spaces.Sequence(spaces.Dict({
    #                     'exhausts': spaces.Discrete(2),
    #                     'is_playable': spaces.Discrete(2),
    #                     'cost': spaces.Discrete(7),
    #                     'name': spaces.Text(max_length=100),
    #                     'id': spaces.Text(max_length=100),
    #                     'type': spaces.Discrete(5),
    #                     'ethereal': spaces.Discrete(2),
    #                     'uuid': spaces.Text(max_length=100),
    #                     'upgrades': spaces.Discrete(2),
    #                     'rarity': spaces.Discrete(2),
    #                     'has_target': spaces.Discrete(2)
    #                 })),
    #                 'hand': spaces.Sequence(spaces.Dict({
    #                     'exhausts': spaces.Discrete(2),
    #                     'is_playable': spaces.Discrete(2),
    #                     'cost': spaces.Discrete(7),
    #                     'name': spaces.Text(max_length=100),
    #                     'id': spaces.Text(max_length=100),
    #                     'type': spaces.Discrete(5),
    #                     'ethereal': spaces.Discrete(2),
    #                     'uuid': spaces.Text(max_length=100),
    #                     'upgrades': spaces.Discrete(2),
    #                     'rarity': spaces.Discrete(2),
    #                     'has_target': spaces.Discrete(2)
    #                 })),
    #                 'player': spaces.Dict({
    #                     'orbs': spaces.Sequence(spaces.Dict({
    #                         'name': spaces.Text(max_length=100),
    #                         'amount': spaces.Box(low=0, high=np.inf, shape=())
    #                     })),
    #                     'current_hp': spaces.Box(low=0, high=100, shape=()),
    #                     'block': spaces.Box(low=0, high=100, shape=()),
    #                     'max_hp': spaces.Box(low=0, high=100, shape=()),
    #                     'powers': spaces.Sequence(spaces.Dict({
    #                         'name': spaces.Text(max_length=100),
    #                         'amount': spaces.Box(low=0, high=np.inf, shape=())
    #                     })),
    #                     'energy': spaces.Box(low=0, high=10, shape=())
    #                 })
    #             }),
    #             'deck': spaces.Sequence(spaces.Dict({
    #                 'exhausts': spaces.Discrete(2),
    #                 'is_playable': spaces.Discrete(2),
    #                 'cost': spaces.Discrete(7),
    #                 'name': spaces.Text(max_length=100),
    #                 'id': spaces.Text(max_length=100),
    #                 'type': spaces.Discrete(5),
    #                 'ethereal': spaces.Discrete(2),
    #                 'uuid': spaces.Text(max_length=100),
    #                 'upgrades': spaces.Discrete(2),
    #                 'rarity': spaces.Discrete(2),
    #                 'has_target': spaces.Discrete(2)
    #             })),
    #             'relics': spaces.Sequence(spaces.Dict({
    #                 'name': spaces.Text(max_length=100),
    #                 'id': spaces.Text(max_length=100),
    #                 'counter': spaces.Box(low=-np.inf, high=np.inf, shape=())
    #             })),
    #             'max_hp': spaces.Box(low=0, high=100, shape=()),
    #             'current_hp': spaces.Box(low=0, high=100, shape=()),
    #             'gold': spaces.Box(low=0, high=np.inf, shape=()),
    #             'floor': spaces.Box(low=0, high=100, shape=()),
    #             'ascension_level': spaces.Box(low=0, high=20, shape=()),
    #             'act_boss': spaces.Discrete(5),
    #             'action_phase': spaces.Discrete(5),
    #             'act': spaces.Discrete(3),
    #             'screen_name': spaces.Discrete(5),
    #             'room_phase': spaces.Discrete(5),
    #             'is_screen_up': spaces.Discrete(2),
    #             'potions': spaces.Sequence(spaces.Dict({
    #                 'name': spaces.Text(max_length=100),
    #                 'requires_target': spaces.Discrete(2),
    #                 'can_use': spaces.Discrete(2),
    #                 'can_discard': spaces.Discrete(2)
    #             })),
    #             'class': spaces.Discrete(4),
    #             'map': spaces.Sequence(spaces.Dict({
    #                 'symbol': spaces.Text(max_length=100),
    #                 'children': spaces.Sequence(spaces.Dict({
    #                     'x': spaces.Box(low=0, high=20, shape=()),
    #                     'y': spaces.Box(low=0, high=20, shape=())
    #                 })),
    #                 'x': spaces.Box(low=0, high=20, shape=()),
    #                 'y': spaces.Box(low=0, high=20, shape=()),
    #                 'parents': spaces.Sequence(spaces.Dict({
    #                     'x': spaces.Box(low=0, high=20, shape=()),
    #                     'y': spaces.Box(low=0, high=20, shape=())
    #                 }))
    #             }))
    #         })
    #     })
