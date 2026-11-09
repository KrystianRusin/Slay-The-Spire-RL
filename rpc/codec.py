"""Convert between our Python objects and the messages in learner.proto.

Rollouts: a CustomRolloutBuffer becomes a Rollout message, and a Rollout message
becomes a plain dict of numpy arrays. Weights: a policy state_dict becomes bytes
and back. Config: a TrainingConfig becomes a PolicyConfig message and back.
"""
import dataclasses
import hashlib
import io

import numpy as np
import torch

from model.policy import TrainingConfig
from rpc import learner_pb2 as pb

# A rollout is ~14 MB (mostly observations), over gRPC's 4 MB default.
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
GRPC_OPTIONS = [
    ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
    ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
]

# Rollout message field -> CustomRolloutBuffer attribute.
BUFFER_FIELDS = {
    "actions": "actions",
    "log_probs": "old_log_prob",
    "values": "values",
    "rewards": "rewards",
    "dones": "dones",
    "advantages": "advantages",
    "returns": "returns",
}


def array_to_tensor(array):
    array = np.ascontiguousarray(array)
    return pb.Tensor(shape=array.shape, dtype=array.dtype.name, data=array.tobytes())


def tensor_to_array(tensor):
    array = np.frombuffer(tensor.data, dtype=np.dtype(tensor.dtype))
    # frombuffer gives a read-only view of the message's bytes; copy so callers own it.
    return array.reshape(tuple(tensor.shape)).copy()


def rollout_to_proto(buffer, actor_id, policy_version):
    """Pack the filled part of a CustomRolloutBuffer into a Rollout message."""
    n = buffer.pos
    message = pb.Rollout(actor_id=actor_id, policy_version=policy_version, n_steps=n)
    for key, array in buffer.observations.items():
        message.observations[key].CopyFrom(array_to_tensor(array[:n]))
    for field, attribute in BUFFER_FIELDS.items():
        getattr(message, field).CopyFrom(array_to_tensor(getattr(buffer, attribute)[:n]))
    # The buffer doesn't record masks yet (M4); the field stays unset until it does.
    masks = getattr(buffer, "action_masks", None)
    if masks is not None:
        message.action_masks.CopyFrom(array_to_tensor(masks[:n]))
    return message


def rollout_from_proto(message):
    """Unpack a Rollout message into a dict of numpy arrays.

    Raises ValueError if any array's first dimension isn't n_steps, so a
    malformed rollout is rejected before anything trains on it.
    """
    n = message.n_steps
    rollout = {
        "actor_id": message.actor_id,
        "policy_version": message.policy_version,
        "n_steps": n,
        "observations": {key: tensor_to_array(t) for key, t in message.observations.items()},
        "action_masks": tensor_to_array(message.action_masks) if message.HasField("action_masks") else None,
    }
    for field in BUFFER_FIELDS:
        rollout[field] = tensor_to_array(getattr(message, field))

    arrays = {**{f"observations[{k}]": v for k, v in rollout["observations"].items()},
              **{field: rollout[field] for field in BUFFER_FIELDS}}
    if rollout["action_masks"] is not None:
        arrays["action_masks"] = rollout["action_masks"]
    for name, array in arrays.items():
        if array.ndim == 0 or array.shape[0] != n:
            raise ValueError(f"{name} has shape {array.shape}, expected {n} steps")
    return rollout


def encode_state_dict(state_dict):
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getvalue()


def decode_state_dict(data):
    # weights_only: refuse anything but tensors, so bytes off the network can't run code.
    return torch.load(io.BytesIO(data), weights_only=True, map_location="cpu")


def config_to_proto(config):
    return pb.PolicyConfig(**dataclasses.asdict(config))


def config_from_proto(message):
    values = {field.name: getattr(message, field.name) for field in dataclasses.fields(TrainingConfig)}
    values["policy_layers"] = tuple(values["policy_layers"])
    values["value_layers"] = tuple(values["value_layers"])
    return TrainingConfig(**values)


def state_dict_digest(state_dict):
    """Short hash of the weights' values, for checking two processes hold the same network."""
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        digest.update(key.encode())
        digest.update(state_dict[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:12]
