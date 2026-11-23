"""Watch the trained policy play the real, rendered Slay the Spire 2.

Needs the game running with the STS2MCP mod enabled (see docs/runbook.md),
which serves the game's state and accepts actions on localhost:15526. This
loads the newest learner checkpoint and plays: every decision the policy was
trained on is made by the policy, through the same observation and action
code as training (sts2/env.py); screens the headless trainer never shows it
(menus, reward claiming, chests, dialogue) are clicked through by sts2/live.py.

The terminal shows each decision: what the policy was looking at (HP,
energy, hand, enemies and their intents, or the options on offer), what it
chose, and the value it expects from there.

    python3 play_live.py [--character NAME] [--runs N] [--delay SECONDS] [--greedy]

Plays one run (--runs) as a random character (--character), then stops.
--greedy always takes the policy's most likely action instead of sampling.
"""
import argparse
import json
import random
import sys
import time
import urllib.error

import torch as th

from environment.run_env import act, to_tensors
from model import checkpoint
from model.policy import TrainingConfig, make_model
from sts2.env import (CONFIRM, DISCARD_BASE, END_TURN, OPTION_BASE, PLAY_BASE, POTION_BASE, SKIP, TARGETS,
                      Sts2Env)
from sts2.live import LiveGame
from sts2.vocab import CHARACTERS


def option_label(state, k, option):
    """A readable name for option k of a non-combat decision."""
    if option.card:
        return option.card.get("name") or "card"
    decision = state.get("decision")
    if decision == "map_select":
        choice = state["choices"][k]
        return f"{choice.get('type')} at col {choice.get('col')}"
    if decision in ("event_choice", "rest_site"):
        raw = state["options"][k]
        return raw.get("title") or raw.get("option_id") or f"option {k}"
    if decision == "shop":
        return option.kind.replace("shop_", "") + (f" {int(option.num[2] * 300)}g" if option.num[2] else "")
    return option.kind.replace("_", " ")


def describe(env, action, command):
    """What the policy chose, in words."""
    state = env.state
    if action == END_TURN:
        return "end turn"
    if PLAY_BASE <= action < POTION_BASE:
        h, t = divmod(action - PLAY_BASE, TARGETS)
        card = state["hand"][h]["name"]
        return f"play {card}" + (f" -> {state['enemies'][t - 1]['name']}" if t else "")
    if POTION_BASE <= action < DISCARD_BASE:
        p, t = divmod(action - POTION_BASE, TARGETS)
        return f"use potion slot {p}" + (f" -> {state['enemies'][t - 1]['name']}" if t else "")
    if DISCARD_BASE <= action < OPTION_BASE:
        return f"discard potion slot {action - DISCARD_BASE}"
    if OPTION_BASE <= action < SKIP:
        k = action - OPTION_BASE
        return f"pick {option_label(state, k, env.options[k])}"
    if action == CONFIRM:
        return "confirm selection"
    return command["action"].replace("_", " ") if command else "skip"


def intents(enemy):
    parts = []
    for intent in enemy.get("intents") or []:
        label = intent.get("type") or "?"
        if intent.get("damage"):
            label += f" {intent['damage']}" + (f"x{intent['hits']}" if intent.get("hits") else "")
        parts.append(label)
    return ", ".join(parts) or "?"


def situation(env):
    """The lines describing what the policy is looking at."""
    state = env.state
    player = state.get("player") or {}
    context = state.get("context") or {}
    lines = [f"act {context.get('act')} floor {context.get('floor')} | HP {player.get('hp')}/{player.get('max_hp')}"
             f" | block {player.get('block') or 0} | gold {player.get('gold')}"]
    if state.get("decision") == "combat_play":
        lines[0] += f" | energy {state.get('energy')}"
        hand = ", ".join(f"{c['name']}({c.get('cost')})" + ("" if c.get("can_play") else "x")
                         for c in state.get("hand") or [])
        lines.append(f"hand: {hand}")
        for e in state.get("enemies") or []:
            lines.append(f"enemy: {e['name']} {e['hp']}/{e['max_hp']} block {e.get('block') or 0} -> {intents(e)}")
    elif env.options:
        labels = [option_label(state, k, o) + ("" if o.enabled else " (can't)") for k, o in enumerate(env.options)]
        lines.append("options: " + ", ".join(labels))
    return lines


def wait_for_game(game):
    while True:
        try:
            return game.state()
        except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError):
            print("Waiting for the game: start it with the STS2MCP mod enabled (localhost:15526)...", flush=True)
            time.sleep(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", default="run/sts2-learner/checkpoints")
    parser.add_argument("--events-table", default="run/sts2-host/sts2-cli/localization_eng/events.json")
    parser.add_argument("--url", default="http://127.0.0.1:15526/api/v1/singleplayer")
    parser.add_argument("--delay", type=float, default=0.5, help="seconds to pause after each action, to watch")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--character", choices=CHARACTERS, help="default: a random one each run")
    parser.add_argument("--runs", type=int, default=1, help="runs to play before stopping")
    args = parser.parse_args()

    state = checkpoint.load_latest(args.checkpoints)
    if state is None:
        sys.exit(f"No checkpoints in {args.checkpoints}. Train first.")
    model = make_model(state["config"] if isinstance(state["config"], TrainingConfig) else TrainingConfig())
    model.policy.load_state_dict(state["policy"])
    model.policy.set_training_mode(False)
    print(f"Loaded policy v{state['version']} from {state['path']}", flush=True)

    game = LiveGame(args.url, delay=args.delay, events_table=args.events_table)
    env = Sts2Env()
    runs = 0
    character = args.character or random.choice(CHARACTERS)
    last_payload = None
    in_run = False

    while True:
        s = wait_for_game(game)
        kind = s.get("state_type")
        if kind == "game_over" and in_run:
            in_run = False
            runs += 1
            run = s.get("run") or {}
            print(f"\n=== Run over: {character}, act {run.get('act')}, floor {run.get('floor')} ===", flush=True)
            if runs >= args.runs:
                return
            character = args.character or random.choice(CHARACTERS)
        if kind == "menu":
            in_run = False
        if game.handle_automatically(s, character):
            continue
        payload = game.to_payload(s)
        if payload is None:
            print(f"  (no handler for screen {kind!r}; waiting)", flush=True)
            time.sleep(1)
            continue
        if not in_run:
            in_run = True
            # The game picks the ascension (whatever its character select is
            # set to; the mod can't change it), so read it back for the policy.
            ascension = int((s.get("run") or {}).get("ascension") or 0)
            env.begin_run(character, ascension)
            game.act_start.clear()
            print(f"\n=== New run: {character}, ascension {ascension}, policy v{state['version']} ===", flush=True)

        # The same screen again (e.g. after a refused action) keeps its refusals.
        if payload != last_payload:
            env.update_game_state(payload)
            last_payload = payload

        while True:
            mask = env.action_mask()
            commands = env.commands()
            action, value, _ = act(model, to_tensors(env.observe(), model.device), mask, deterministic=args.greedy)
            command = commands[action]
            print(f"\n[{payload['decision']}] " + "\n    ".join(situation(env)), flush=True)
            print(f"  -> {describe(env, action, command)}   (expects {value.item():+.2f})", flush=True)
            if command is not None:
                break
            env.apply_internal(action)  # one more card into a selection; pick again

        for request in game.to_requests(command):
            reply = game.post(request)
            if reply.get("status") != "ok":
                print(f"  refused: {reply.get('error')}", flush=True)
                env.refused.add(action)
                break
        else:
            last_payload = None  # accepted: whatever comes next is a new decision
            game.wait_for_change(s)


if __name__ == "__main__":
    th.set_num_threads(2)
    main()
