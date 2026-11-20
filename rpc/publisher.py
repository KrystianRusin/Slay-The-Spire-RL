"""The learner's side of the coordinator's Trainer service: keeps the coordinator
holding the learner's newest weights.

The learner hands each new version to publish(), which returns at once. A
background thread then makes the coordinator hold those weights, and keeps
checking that it still does: every HEARTBEAT_SECONDS it asks the coordinator
which version it has (a few bytes) and publishes again only if the answer is
wrong. The coordinator keeps weights only in memory, so after it restarts it
has none, and without the check nothing would ever send them again: actors
can't play without weights, so no rollouts arrive, so no round finishes, so
the learner never publishes.

Both cases are one check. A new version, a coordinator that restarted, and a
coordinator that couldn't be reached last time all show up as "the coordinator
doesn't have what I last published", and are fixed the same way.
"""
import sys
import threading

import grpc

from rpc import learner_pb2 as pb
from rpc import learner_pb2_grpc
from rpc.codec import GRPC_OPTIONS

HEARTBEAT_SECONDS = 5
CHECK_TIMEOUT = 5
PUBLISH_TIMEOUT = 30  # weights are a few MB


def log(message):
    sys.stdout.write(message + "\n")
    sys.stdout.flush()


class WeightsPublisher:
    def __init__(self, address, heartbeat_seconds=HEARTBEAT_SECONDS):
        self.address = address
        self.heartbeat_seconds = heartbeat_seconds
        self._channel = grpc.insecure_channel(address, options=GRPC_OPTIONS)
        self._stub = learner_pb2_grpc.TrainerStub(self._channel)
        self._lock = threading.Lock()
        self._changed = threading.Condition(self._lock)  # new weights to send, or closing
        self._latest = None  # the pb.Weights the coordinator should hold
        self._unchecked = False  # _latest changed since the thread last looked
        self._closed = False
        self.confirmed = None  # (version, run_id) the coordinator was last seen holding; None if unknown
        self.publishes = 0  # PublishWeights calls that succeeded
        self._failing = False  # so an outage is logged once, not every heartbeat
        self._thread = threading.Thread(target=self._run, name="publisher", daemon=True)
        self._thread.start()

    def publish(self, weights):
        """Make weights (a pb.Weights) the ones the coordinator should hold. Doesn't block."""
        with self._lock:
            self._latest = weights
            self._unchecked = True
            self._changed.notify_all()  # not notify(): it could wake a wait_until_confirmed instead

    def wait_until_confirmed(self, version, run_id, timeout):
        """Block until the coordinator is seen holding version from run_id. Returns whether it was."""
        with self._lock:
            return self._changed.wait_for(lambda: self.confirmed == (version, run_id), timeout=timeout)

    def close(self):
        """Stop the thread and drop the connection. Publishes nothing more."""
        with self._lock:
            self._closed = True
            self._changed.notify_all()
        self._thread.join(timeout=PUBLISH_TIMEOUT + CHECK_TIMEOUT)
        self._channel.close()

    def _run(self):
        while True:
            with self._lock:
                # Sleep until there are new weights, or for one heartbeat.
                self._changed.wait_for(lambda: self._unchecked or self._closed,
                                       timeout=self.heartbeat_seconds)
                if self._closed:
                    return
                weights, self._unchecked = self._latest, False
            if weights is not None:
                self._make_coordinator_hold(weights)

    def _make_coordinator_hold(self, weights):
        """Check what the coordinator has, and publish weights if it isn't them."""
        wanted = (weights.version, weights.run_id)
        try:
            held = self._stub.CurrentVersion(pb.CurrentVersionRequest(), timeout=CHECK_TIMEOUT)
            self._saw((held.version, held.run_id))
            if self.confirmed != wanted:
                if held.run_id == "":
                    log(f"publisher: coordinator at {self.address} has no weights, publishing v{weights.version}")
                self._stub.PublishWeights(weights, timeout=PUBLISH_TIMEOUT)
                self.publishes += 1
                self._saw(wanted)
                log(f"publisher: published v{weights.version}, {weights.ByteSize() / 1e6:.2f} MB")
        except grpc.RpcError as e:
            self._saw(None)
            if not self._failing:
                log(f"publisher: coordinator at {self.address} failed ({e.code().name}: {e.details()}), "
                    f"trying again every {self.heartbeat_seconds}s")
            self._failing = True
            return
        if self._failing:
            log(f"publisher: coordinator at {self.address} is answering again")
            self._failing = False

    def _saw(self, held):
        with self._lock:
            self.confirmed = held
            self._changed.notify_all()
