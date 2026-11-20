"""The real services, started for a test: the Go coordinator (its binary, as a
subprocess) and learners (in this process, on their own gRPC servers).

The coordinator binary is built into the game image at /opt/sts/bin/coordinator;
COORDINATOR_BIN points elsewhere. Tests that use the stack fixture skip without it.
"""
import os
import subprocess
import threading
import time

import pytest

from learner_server import LearnerService, serve
from rpc.client import LearnerClient
from rpc.publisher import WeightsPublisher

COORDINATOR_BIN = os.environ.get("COORDINATOR_BIN", "/opt/sts/bin/coordinator")
LONG_POLL_SECONDS = 0.5  # the coordinator's, so tests that wait one out stay quick
HEARTBEAT_SECONDS = 0.1  # the learner's check that the coordinator still holds its weights
WAIT_SECONDS = 30  # for things that should happen; only reached if they don't


class Coordinator:
    """One run of the coordinator binary. port=0 picks a free port."""

    def __init__(self, port=0):
        self.process = subprocess.Popen(
            [COORDINATOR_BIN], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            env={**os.environ, "COORDINATOR_PORT": str(port),
                 "COORDINATOR_LONG_POLL_SECONDS": str(LONG_POLL_SECONDS)})
        line = self.process.stdout.readline()
        assert "listening on port" in line, line
        self.port = int(line.rsplit(" ", 1)[1])
        self.address = f"localhost:{self.port}"
        # Read its log as it comes: a pipe nobody reads fills up, and then the
        # coordinator blocks on its next log line.
        self.log = []
        self._reader = threading.Thread(target=self._read_log, daemon=True)
        self._reader.start()

    def _read_log(self):
        for line in self.process.stdout:
            self.log.append(line)

    def wait(self, timeout):
        """Wait for it to exit; returns its exit code. Its whole log is in self.log after."""
        code = self.process.wait(timeout=timeout)
        self._reader.join(timeout=timeout)
        return code

    def kill(self):
        """As if it crashed: no chance to clean up, and everything it held is gone."""
        self.process.kill()
        self.wait(timeout=WAIT_SECONDS)


class Stack:
    def __init__(self):
        self.coordinator = None
        self._coordinator_port = 0
        self._learners = []  # (service, server)

    def start_coordinator(self):
        """Start the coordinator, on the same port as the last one if there was one."""
        self.coordinator = Coordinator(self._coordinator_port)
        self._coordinator_port = self.coordinator.port
        return self.coordinator

    def start_learner(self, config, num_actors=1, port=0):
        """Start a learner that publishes to the coordinator. Returns (service, its port).

        The learner can start before the coordinator: it publishes once the
        coordinator is up. Its training thread is up to the test.
        """
        if self._coordinator_port == 0:
            self.start_coordinator()
        publisher = WeightsPublisher(f"localhost:{self._coordinator_port}", heartbeat_seconds=HEARTBEAT_SECONDS)
        service = LearnerService(config, publisher, num_actors)
        server, bound = serve(service, port)
        self._learners.append((service, server))
        return service, bound

    def stop_learner(self, service):
        """As if the learner's process died: its server and its publisher both stop."""
        for entry in self._learners:
            if entry[0] is service:
                self._learners.remove(entry)
                entry[1].stop(grace=None).wait()
                service.publisher.close()
                return
        raise ValueError("not a running learner")

    def client(self, actor_id, learner_port):
        return LearnerClient(f"localhost:{self._coordinator_port}", f"localhost:{learner_port}", actor_id)

    def close(self):
        for service, _ in list(self._learners):
            self.stop_learner(service)
        if self.coordinator is not None:
            self.coordinator.kill()


def eventually(condition, message, timeout=WAIT_SECONDS):
    """Poll condition() until it is true; fail with message if it never is."""
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, message
        time.sleep(0.01)


def wait_until_published(service, version=None):
    """Block until the coordinator holds service's version (or the given one)."""
    version = service.version if version is None else version
    assert service.publisher.wait_until_confirmed(version, service.run_id, timeout=WAIT_SECONDS), \
        f"the coordinator never got v{version} from run {service.run_id}"


@pytest.fixture
def stack():
    if not os.path.exists(COORDINATOR_BIN):
        pytest.skip(f"no coordinator binary at {COORDINATOR_BIN}")
    running = Stack()
    yield running
    running.close()
