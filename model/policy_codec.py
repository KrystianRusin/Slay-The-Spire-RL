"""Versioned byte encoding of policy weights, for publishing them to actors.

Layout, all integers little-endian:

    offset  size  field
    0       4     magic, the ASCII bytes "SPOL"
    4       2     schema version, uint16
    6       4     header length N, uint32
    10      N     header, UTF-8 JSON
    10 + N  rest  body, the policy's state dict as written by torch.save

The header is an object:

    {
      "policy_version": number of updates the learner had applied,
      "sha256": hex SHA-256 digest of the body
    }

A payload whose body does not match its digest is rejected whole, so weights
cut short in transfer can never be loaded.
"""

import hashlib
import io
import json
import struct

import torch as th

MAGIC = b"SPOL"
SCHEMA_VERSION = 1
FIXED_PREFIX = struct.Struct("<4sHI")


class PublishedPolicy:
    """Policy weights decoded from bytes, not yet loaded into any policy."""

    def __init__(self, version, state_dict):
        self.version = version
        self._state_dict = state_dict

    def apply_to(self, policy):
        """Load these weights into policy, all of them or, raising ValueError, none."""
        expected = {name: tensor.shape for name, tensor in policy.state_dict().items()}
        received = {name: tensor.shape for name, tensor in self._state_dict.items()}
        if received != expected:
            mismatched = sorted(name for name in expected.keys() | received.keys() if expected.get(name) != received.get(name))
            raise ValueError(f"Policy weights version {self.version} does not fit this policy: {', '.join(mismatched)}")
        policy.load_state_dict(self._state_dict)


def encode_policy(policy, version):
    """Encode a policy's weights to bytes, labelled with the learner's policy version."""
    buffer = io.BytesIO()
    th.save({name: tensor.detach().cpu() for name, tensor in policy.state_dict().items()}, buffer)
    body = buffer.getvalue()
    header = json.dumps({"policy_version": version, "sha256": hashlib.sha256(body).hexdigest()}).encode("utf-8")
    return FIXED_PREFIX.pack(MAGIC, SCHEMA_VERSION, len(header)) + header + body


def decode_policy(data):
    """Decode bytes from encode_policy into a PublishedPolicy.

    Raises ValueError for anything that is not a complete, intact payload.
    """
    if len(data) < FIXED_PREFIX.size or data[:len(MAGIC)] != MAGIC:
        raise ValueError("Data is not encoded policy weights")
    _, version, header_length = FIXED_PREFIX.unpack_from(data)
    if version != SCHEMA_VERSION:
        raise ValueError(f"Policy weights have schema version {version}; this reader only understands {SCHEMA_VERSION}")

    header_end = FIXED_PREFIX.size + header_length
    try:
        header = json.loads(data[FIXED_PREFIX.size:header_end].decode("utf-8"))
        policy_version, digest = header["policy_version"], header["sha256"]
    except (KeyError, TypeError, UnicodeDecodeError) as error:
        raise ValueError(f"Corrupt policy weights: {error!r}") from error
    body = data[header_end:]
    if hashlib.sha256(body).hexdigest() != digest:
        raise ValueError("Corrupt policy weights: body does not match its digest")
    state_dict = th.load(io.BytesIO(body), map_location="cpu", weights_only=True)
    return PublishedPolicy(policy_version, state_dict)
