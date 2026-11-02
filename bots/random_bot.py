"""Random-action bot for testing the bridge and measuring raw game throughput.

Starts Ironclad runs and sends random valid commands. Every 10 seconds it logs
how many commands the game accepted (steps) and the steps per second, to
random_bot.log in the instance folder.
"""
import json
import os
import random
import select
import sys
import time

LOG_EVERY = 10.0
STALL_TIMEOUT = 10.0  # seconds of silence before asking the game to resend its state;
                     # long enough that a slow enemy turn doesn't trigger it


def choose_command(msg):
    """Pick a random valid command for this CommunicationMod message."""
    avail = msg.get("available_commands", [])
    if not msg.get("in_game"):
        return "start ironclad 0" if "start" in avail else "state"
    gs = msg.get("game_state", {})
    options = []
    if "choose" in avail and gs.get("choice_list"):
        options += [f"choose {random.randrange(len(gs['choice_list']))}"] * 3
    if "play" in avail:
        combat = gs.get("combat_state", {})
        monsters = [i for i, m in enumerate(combat.get("monsters", []))
                    if not m.get("is_gone") and m.get("current_hp", 0) > 0]
        for i, card in enumerate(combat.get("hand", []), start=1):
            if card.get("is_playable"):
                if card.get("has_target"):
                    if monsters:
                        options += [f"play {i} {random.choice(monsters)}"] * 2
                else:
                    options += [f"play {i}"] * 2
    for c in ("end", "proceed", "confirm", "leave", "skip", "cancel", "return"):
        if c in avail:
            options.append(c)
    return random.choice(options) if options else "wait 30"


def read_lines_with_timeout(fd, timeout):
    """Yield lines from fd, or None whenever `timeout` seconds pass with no input.

    Reads the raw fd instead of sys.stdin, because select() can't see lines
    already sitting in Python's read buffer.
    """
    buf = b""
    while True:
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            yield line.decode("utf-8", errors="replace")
        ready, _, _ = select.select([fd], [], [], timeout)
        if not ready:
            yield None
            continue
        chunk = os.read(fd, 65536)
        if not chunk:  # game closed the pipe
            return
        buf += chunk


def main():
    log = open("random_bot.log", "a", buffering=1)
    # RANDOM_BOT_TRACE=1 logs every message and reply as JSON lines (epoch-seconds
    # timestamps), for debugging stalls and timing.
    trace_file = open("random_bot_trace.jsonl", "a", buffering=1) if os.environ.get("RANDOM_BOT_TRACE") else None

    def trace(**event):
        if trace_file:
            trace_file.write(json.dumps({"t": round(time.time(), 3), **event}) + "\n")

    print("ready", flush=True)
    start = last = time.time()
    steps = errors = pokes = 0
    poked = False
    for line in read_lines_with_timeout(sys.stdin.fileno(), STALL_TIMEOUT):
        if line is None:
            # The game went quiet: it sometimes drops a command sent while a screen
            # is still animating in, then waits for us while we wait for it.
            # `state` makes CommunicationMod resend the current screen. Its reply is
            # always marked ready_for_command=False, so we act on it anyway.
            pokes += 1
            poked = True
            print("state", flush=True)
            trace(dir="out", cmd="state", poke=True)
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        gs = msg.get("game_state", {})
        trace(dir="in", ready=msg.get("ready_for_command"), screen=gs.get("screen_type"),
              phase=gs.get("room_phase"), floor=gs.get("floor"),
              avail=msg.get("available_commands"), error=msg.get("error"))
        answering_poke, poked = poked, False
        if "error" in msg:
            errors += 1
            cmd = "state"
        elif msg.get("ready_for_command") or answering_poke:
            cmd = choose_command(msg)
            steps += 1
        else:
            continue
        print(cmd, flush=True)
        trace(dir="out", cmd=cmd)
        now = time.time()
        if now - last >= LOG_EVERY:
            floor = msg.get("game_state", {}).get("floor")
            log.write(f"{time.strftime('%H:%M:%S')} steps={steps} errors={errors} pokes={pokes} "
                      f"steps/s={steps / (now - start):.2f} floor={floor}\n")
            last = now


if __name__ == "__main__":
    main()
