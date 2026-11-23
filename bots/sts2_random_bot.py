"""Play complete Slay the Spire 2 runs with uniformly random legal actions.

Checks the headless game works end to end and maps out its protocol. It starts
the sts2-cli host (a .NET process running the game's logic with no rendering)
and talks to it in JSON lines over the host's stdin/stdout:

    we send   {"cmd": "start_run", "character": "Ironclad", "seed": "s1"}
    it sends  {"type": "decision", "decision": "map_select", "choices": [...], ...}
    we send   {"cmd": "action", "action": "select_map_node", "args": {"col": 3, "row": 1}}
    ...until  {"type": "decision", "decision": "game_over", "victory": false, ...}

Every reply is the next decision the game is waiting on, and it lists what is
possible: playable cards carry "can_play", map choices are only the reachable
nodes, event options carry "is_locked", and so on. The bot builds its list of
legal actions from those fields and picks one at random. When the host refuses
an action anyway, the refusal is counted (the RL action mask will need the same
rules) and the bot tries another.

Env (defaults in brackets):
  STS2_HOST_DLL      the built host's Sts2Headless.dll (set by the entrypoint)
  STS2_RUNS          runs to play [10]
  STS2_CHARACTER     Ironclad, Silent, Defect, Regent, Necrobinder, or "random" [random]
  STS2_ASCENSION     [0]
  STS2_SEED_PREFIX   run i uses seed <prefix><i>, so runs are reproducible [spike-]
  STS2_TIMEOUT       seconds to wait for one reply before calling it a hang [30]
  STS2_MAX_STEPS     decisions per run before giving up [3000]
  STS2_FIXTURE_DIR   saves the first payload of each decision type here [./fixtures]
  STS2_TRACE         1 = print every action and decision

Writes summary.json (per-run results and speed) and host.log (the host's
stderr) in the working directory. Exits non-zero if any run failed to finish.
"""

import json
import os
import random
import sys
import time

from sts2.host import Host, HostError

CHARACTERS = ["Ironclad", "Silent", "Defect", "Regent", "Necrobinder"]
# Enemy-targeted cards and potions need a target_index; everything else doesn't.
TARGETED = {"AnyEnemy"}


def action(name, **args):
    command = {"cmd": "action", "action": name}
    if args:
        command["args"] = args
    return command


def live_enemies(state):
    return [e for e in state.get("enemies") or [] if e.get("hp", 0) > 0]


def legal_actions(state, rng):
    """Every action the payload says is possible right now.

    The host's index arguments are positions in the lists it sent ("index"
    fields), and target_index is a position in "enemies".
    """
    decision = state.get("decision")
    player = state.get("player") or {}
    gold = player.get("gold", 0)

    if decision == "map_select":
        return [action("select_map_node", col=c["col"], row=c["row"]) for c in state.get("choices", [])]

    if decision == "combat_play":
        actions = [action("end_turn")]
        targets = [e["index"] for e in live_enemies(state)]
        for card in state.get("hand", []):
            if not card.get("can_play"):
                continue
            if card.get("target_type") in TARGETED:
                actions += [action("play_card", card_index=card["index"], target_index=t) for t in targets]
            else:
                actions.append(action("play_card", card_index=card["index"]))
        for potion in player.get("potions") or []:
            if potion.get("target_type") in TARGETED:
                actions += [action("use_potion", potion_index=potion["index"], target_index=t) for t in targets]
            else:
                actions.append(action("use_potion", potion_index=potion["index"]))
            actions.append(action("discard_potion", potion_index=potion["index"]))
        return actions

    if decision == "card_reward":
        actions = [action("select_card_reward", card_index=c["index"]) for c in state.get("cards", [])]
        if state.get("can_skip", True):
            actions.append(action("skip_card_reward"))
        return actions

    if decision == "card_select":
        cards = state.get("cards", [])
        low = state.get("min_select", 1) or 0
        high = min(state.get("max_select", 1) or 0, len(cards))
        actions = []
        if low == 0:
            actions.append(action("skip_select"))
        if cards and high >= max(low, 1):
            # One random pick of a legal size; listing every subset would be huge.
            size = rng.randint(max(low, 1), high)
            picked = sorted(rng.sample([c["index"] for c in cards], size))
            actions.append(action("select_cards", indices=",".join(map(str, picked))))
        return actions

    if decision == "bundle_select":
        return [action("select_bundle", bundle_index=i) for i in range(len(state.get("bundles", [])))]

    if decision == "event_choice":
        actions = [action("choose_option", option_index=o["index"])
                   for o in state.get("options", []) if not o.get("is_locked")]
        return actions or [action("leave_room")]

    if decision == "rest_site":
        actions = [action("choose_option", option_index=o["index"])
                   for o in state.get("options", []) if o.get("is_enabled", True)]
        return actions or [action("leave_room")]

    if decision == "shop":
        actions = [action("leave_room")]
        for kind, verb in (("cards", "buy_card"), ("relics", "buy_relic"), ("potions", "buy_potion")):
            index_name = verb.split("_")[1] + "_index"
            for item in state.get(kind, []):
                if item.get("is_stocked", True) and item.get("cost", 0) <= gold:
                    actions.append(action(verb, **{index_name: item["index"]}))
        removal = state.get("card_removal_cost")
        if removal is not None and removal <= gold:
            actions.append(action("remove_card"))
        return actions

    # "unknown" covers rooms the host has no payload for (e.g. treasure).
    return [action("proceed")]


def choose(state, actions, rng):
    """A uniformly random legal action, except in shops.

    A uniform pick in a shop almost never leaves while gold lasts, so leaving
    gets half the probability there: shops stay explored without each one
    costing dozens of steps.
    """
    if state.get("decision") == "shop" and len(actions) > 1 and rng.random() < 0.5:
        return actions[0]
    return rng.choice(actions)


def save_fixture(fixture_dir, state, seen):
    """Keep the first payload of each decision type, as protocol samples."""
    key = state.get("decision") or state.get("type")
    if key in seen:
        return
    seen.add(key)
    os.makedirs(fixture_dir, exist_ok=True)
    with open(os.path.join(fixture_dir, f"{key}.json"), "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def play_run(host, character, seed, ascension, max_steps, rng, fixture_dir, seen, refusals):
    """Play one run to game over. Returns a result dict."""
    started = time.monotonic()
    state = host.send({"cmd": "start_run", "character": character, "seed": seed, "ascension": ascension})
    steps = 0
    decision_seconds = 0.0
    while steps < max_steps:
        if state.get("type") == "error":
            raise HostError(f"Run failed to start or continue: {state.get('message')}")
        save_fixture(fixture_dir, state, seen)
        if state.get("decision") == "game_over":
            return {
                "seed": seed, "character": character, "finished": True,
                "victory": bool(state.get("victory")), "act": state.get("act"),
                "floor": state.get("floor"), "steps": steps,
                "seconds": round(time.monotonic() - started, 2),
                "decision_seconds": round(decision_seconds, 2),
            }

        candidates = legal_actions(state, rng)
        while True:
            chosen = choose(state, candidates, rng)
            sent = time.monotonic()
            reply = host.send(chosen)
            decision_seconds += time.monotonic() - sent
            steps += 1
            if reply.get("type") != "error":
                break
            if reply.get("error_code") == "combat_turn_timeout":
                raise HostError(f"Engine stalled: {reply}")
            # Refused. Record what and why: these are the gaps between what the
            # payload advertises and what the game accepts.
            key = f"{state.get('decision')}/{chosen['action']}: {reply.get('message', '')[:100]}"
            refusals[key] = refusals.get(key, 0) + 1
            candidates = [a for a in candidates if a is not chosen]
            if not candidates:
                # Nothing listed is accepted. "proceed" is the host's generic
                # way past a screen; if that fails too the run is stuck.
                reply = host.send(action("proceed"))
                steps += 1
                if reply.get("type") == "error":
                    raise HostError(f"Stuck at {state.get('decision')}: {reply.get('message')}")
                break
        state = reply

    return {"seed": seed, "character": character, "finished": False, "error": "max_steps", "steps": steps}


def main():
    dll = os.environ["STS2_HOST_DLL"]
    runs = int(os.environ.get("STS2_RUNS", "10"))
    character_choice = os.environ.get("STS2_CHARACTER", "random")
    ascension = int(os.environ.get("STS2_ASCENSION", "0"))
    prefix = os.environ.get("STS2_SEED_PREFIX", "spike-")
    timeout = float(os.environ.get("STS2_TIMEOUT", "30"))
    max_steps = int(os.environ.get("STS2_MAX_STEPS", "3000"))
    fixture_dir = os.environ.get("STS2_FIXTURE_DIR", "fixtures")
    trace = os.environ.get("STS2_TRACE") == "1"

    results, refusals, seen = [], {}, set()
    with open("host.log", "a") as host_log:
        for i in range(1, runs + 1):
            seed = f"{prefix}{i}"
            # The bot's own choices are seeded too, so seed + bot = the same run.
            rng = random.Random(seed)
            character = rng.choice(CHARACTERS) if character_choice == "random" else character_choice
            # A fresh host per run: a crash or hang in one run can't leak into the next.
            launched = time.monotonic()
            host = None
            try:
                host = Host(dll, timeout, host_log, trace)
                startup = time.monotonic() - launched
                result = play_run(host, character, seed, ascension, max_steps, rng, fixture_dir, seen, refusals)
            except HostError as e:
                startup = time.monotonic() - launched
                result = {"seed": seed, "character": character, "finished": False, "error": str(e)}
            finally:
                if host:
                    host.close()
            result["startup_seconds"] = round(startup, 2)
            results.append(result)
            outcome = ("WIN" if result.get("victory") else "LOSS") if result["finished"] else f"FAILED ({result['error']})"
            rate = result["steps"] / result["decision_seconds"] if result.get("decision_seconds") else 0
            print(f"run {i}/{runs} {character:<11} seed={seed} {outcome} act={result.get('act')} "
                  f"floor={result.get('floor')} steps={result.get('steps')} "
                  f"startup={startup:.1f}s decisions/s={rate:.0f}", flush=True)

    finished = [r for r in results if r["finished"]]
    total_steps = sum(r["steps"] for r in finished)
    total_decision_seconds = sum(r["decision_seconds"] for r in finished)
    summary = {
        "runs": results,
        "finished": len(finished),
        "victories": sum(r["victory"] for r in finished),
        "decisions_per_second": round(total_steps / total_decision_seconds, 1) if total_decision_seconds else None,
        "mean_startup_seconds": round(sum(r["startup_seconds"] for r in results) / len(results), 2),
        "decision_types_seen": sorted(seen),
        "refused_actions": dict(sorted(refusals.items(), key=lambda kv: -kv[1])),
    }
    with open("summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=2))
    return 0 if len(finished) == runs else 1


if __name__ == "__main__":
    sys.exit(main())
