"""One headless STS2 game: the sts2-cli host process and the conversation with it.

The host reads one JSON command per line on stdin and answers each with one
JSON reply on stdout: the next decision the game is waiting on, or an error.

    {"cmd": "start_run", "character": "Ironclad", "seed": "s1", "ascension": 0}
    {"cmd": "action", "action": "play_card", "args": {"card_index": 0, "target_index": 0}}

A host plays one run: a second start_run in the same process fails, so a new
run needs a new Host.
"""
import json
import os
import queue
import subprocess
import threading
import time


class HostError(RuntimeError):
    """The host hung, exited, or broke the protocol. The run can't continue."""


class Host:
    def __init__(self, dll=None, timeout=30.0, log_file=None, trace=False):
        self.timeout = timeout
        self.trace = trace
        dll = dll or os.environ["STS2_HOST_DLL"]
        self.proc = subprocess.Popen(
            ["dotnet", dll],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log_file if log_file is not None else subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        # Reading stdout blocks, so a background thread does it and hands lines
        # over through a queue. The caller can then wait with a timeout and
        # notice a hung engine instead of blocking forever.
        self._lines = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        ready = self._read()
        if ready.get("type") != "ready":
            raise HostError(f"Expected a ready message, got {ready}")

    def _pump(self):
        for line in self.proc.stdout:
            self._lines.put(line)
        self._lines.put(None)  # end of stream: the process exited

    def _read(self):
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            try:
                line = self._lines.get(timeout=max(remaining, 0.001))
            except queue.Empty:
                raise HostError(f"No reply within {self.timeout}s") from None
            if line is None:
                raise HostError(f"Host exited with code {self.proc.wait()}")
            # The host's own logging can land on stdout too; replies are the
            # lines that are JSON objects.
            if line.startswith("{"):
                return json.loads(line)

    def send(self, command):
        if self.trace:
            print(f"  > {json.dumps(command)[:200]}", flush=True)
        try:
            self.proc.stdin.write(json.dumps(command) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise HostError(f"Host is gone: {e}") from None
        reply = self._read()
        if self.trace:
            print(f"  < {reply.get('type')}/{reply.get('decision', '')} {str(reply.get('message', ''))[:120]}",
                  flush=True)
        return reply

    def start_run(self, character, seed, ascension=0):
        return self.send({"cmd": "start_run", "character": character, "seed": seed, "ascension": ascension})

    def close(self):
        try:
            self.proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
            self.proc.wait()
