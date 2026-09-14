"""An actor's local policy, kept on the newest weights the learner has published."""

import logging

from model.policy_codec import decode_policy

WAIT_SECONDS = 10.0

logger = logging.getLogger(__name__)


class PolicyFollower:
    """Loads published policy weights into a local policy.

    source is anything with newest(timeout), returning the newest encoded
    weights published since the last call, or None if there are none.
    """

    def __init__(self, policy, source):
        self.policy = policy
        self.version = None
        self.learner_version = None
        self._source = source

    def wait_for_first(self):
        """Block until the learner has published weights, and load the newest."""
        while self.version is None:
            encoded = self._source.newest(WAIT_SECONDS)
            if encoded is None:
                logger.info("Waiting for the learner to publish policy weights")
            else:
                self._switch(encoded)

    def update(self):
        """Switch to the newest weights published since the last read, returning whether the policy changed."""
        encoded = self._source.newest()
        return encoded is not None and self._switch(encoded)

    def _switch(self, encoded):
        try:
            published = decode_policy(encoded)
            self.learner_version = published.version
            if published.version == self.version:
                return False
            published.apply_to(self.policy)
        except ValueError as error:
            logger.error("Keeping policy version %s, since the newest published weights cannot be loaded: %s", self.version, error)
            return False
        self.version = published.version
        return True
