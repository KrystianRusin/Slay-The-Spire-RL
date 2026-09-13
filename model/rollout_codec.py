"""Versioned byte encoding of a completed rollout, for handing it to the learner.

Layout, all integers little-endian:

    offset  size  field
    0       4     magic, the ASCII bytes "SROL"
    4       2     schema version, uint16
    6       4     header length N, uint32
    10      N     header, UTF-8 JSON
    10 + N  rest  body, a single zlib stream

The header is an object:

    {
      "compression": "zlib",
      "rollout_id": string identifying the rollout, unique across actors,
      "steps": number of transitions,
      "arrays": [{"name": ..., "dtype": "float32", "shape": [...], "offset": ...}, ...]
    }

Decompressed, the body is every array's bytes in C order, concatenated. Each
"offset" is where that array starts in the decompressed body. "float32" is a
little-endian IEEE 754 single, 4 bytes, and is the only dtype in schema
version 2.

Arrays in schema version 2, each with a leading dimension of steps:

    observations/<key>  one per observation component, at that component's shape
    actions             (steps, 1)
    rewards, dones, values, old_log_prob, advantages, returns   (steps,)
"""

import json
import struct
import uuid
import zlib

import numpy as np

from model.custom_rollout_buffer import CustomRolloutBuffer

MAGIC = b"SROL"
SCHEMA_VERSION = 2
FIXED_PREFIX = struct.Struct("<4sHI")
COMPRESSION = "zlib"
DTYPES = {"float32": np.dtype("<f4")}
OBSERVATION_PREFIX = "observations/"
STEP_FIELDS = ["actions", "rewards", "dones", "values", "old_log_prob", "advantages", "returns"]


class UnsupportedSchemaVersion(ValueError):
    """The payload was written under a schema version this reader does not know."""

    def __init__(self, version):
        super().__init__(f"Rollout has schema version {version}; this reader only understands {SCHEMA_VERSION}")
        self.version = version


def encode_rollout(buffer, rollout_id=None):
    """Encode the filled steps of a rollout buffer to bytes.

    rollout_id is how the learner recognises a rollout delivered twice; a fresh
    one is generated when it is not given.
    """
    if rollout_id is None:
        rollout_id = uuid.uuid4().hex
    steps = buffer.pos
    arrays = {OBSERVATION_PREFIX + key: obs[:steps] for key, obs in buffer.observations.items()}
    arrays.update({name: getattr(buffer, name)[:steps] for name in STEP_FIELDS})

    entries, chunks, offset = [], [], 0
    for name, array in arrays.items():
        dtype = DTYPES[array.dtype.name]
        data = np.ascontiguousarray(array, dtype=dtype).tobytes()
        entries.append({"name": name, "dtype": array.dtype.name, "shape": list(array.shape), "offset": offset})
        chunks.append(data)
        offset += len(data)

    header = json.dumps({"compression": COMPRESSION, "rollout_id": rollout_id, "steps": steps, "arrays": entries}).encode("utf-8")
    body = zlib.compress(b"".join(chunks))
    return FIXED_PREFIX.pack(MAGIC, SCHEMA_VERSION, len(header)) + header + body


def decode_rollout(data, observation_space, action_space, device="cpu"):
    """Decode bytes from encode_rollout into a full CustomRolloutBuffer, with its rollout_id set.

    Raises UnsupportedSchemaVersion for a version this reader does not know,
    and ValueError for anything else it cannot decode exactly.
    """
    if len(data) < FIXED_PREFIX.size or data[:len(MAGIC)] != MAGIC:
        raise ValueError("Data is not an encoded rollout")
    _, version, header_length = FIXED_PREFIX.unpack_from(data)
    if version != SCHEMA_VERSION:
        raise UnsupportedSchemaVersion(version)

    header_end = FIXED_PREFIX.size + header_length
    try:
        header = json.loads(data[FIXED_PREFIX.size:header_end].decode("utf-8"))
        if header["compression"] != COMPRESSION:
            raise ValueError(f"Rollout compression {header['compression']!r} is not supported")
        body = _decompress(data[header_end:])
        rollout_id = header["rollout_id"]
        steps = header["steps"]
        arrays = {entry["name"]: _read_array(body, entry, steps) for entry in header["arrays"]}
    except (KeyError, TypeError, zlib.error) as error:
        raise ValueError(f"Corrupt rollout: {error!r}") from error

    missing = [name for name in STEP_FIELDS if name not in arrays]
    if missing:
        raise ValueError(f"Rollout is missing arrays: {', '.join(missing)}")
    _check_observation_layout(arrays, observation_space)

    buffer = CustomRolloutBuffer(steps, observation_space, action_space, device=device)
    for key in observation_space.spaces:
        buffer.observations[key] = arrays[OBSERVATION_PREFIX + key]
    for name in STEP_FIELDS:
        setattr(buffer, name, arrays[name])
    buffer.pos = steps
    buffer.full = True
    buffer.rollout_id = rollout_id
    return buffer


def _decompress(compressed):
    decompressor = zlib.decompressobj()
    body = decompressor.decompress(compressed)
    if not decompressor.eof or decompressor.unused_data:
        raise ValueError("Corrupt rollout: body is truncated or has trailing bytes")
    return body


def _read_array(body, entry, steps):
    name, shape, offset = entry["name"], tuple(entry["shape"]), entry["offset"]
    if entry["dtype"] not in DTYPES:
        raise ValueError(f"Rollout array {name} has unsupported dtype {entry['dtype']!r}")
    if not shape or shape[0] != steps:
        raise ValueError(f"Rollout array {name} has shape {shape}, but the header declares {steps} steps")
    dtype = DTYPES[entry["dtype"]]
    count = int(np.prod(shape))
    if offset < 0 or offset + count * dtype.itemsize > len(body):
        raise ValueError(f"Corrupt rollout: array {name} runs past the end of the body")
    return np.frombuffer(body, dtype=dtype, count=count, offset=offset).reshape(shape).astype(dtype.newbyteorder("="))


def _check_observation_layout(arrays, observation_space):
    encoded = {name[len(OBSERVATION_PREFIX):]: array.shape[1:]
               for name, array in arrays.items() if name.startswith(OBSERVATION_PREFIX)}
    expected = {key: space.shape for key, space in observation_space.spaces.items()}
    mismatched = sorted(key for key in encoded.keys() | expected.keys() if encoded.get(key) != expected.get(key))
    if mismatched:
        details = ", ".join(f"{key}: encoded {encoded.get(key)}, expected {expected.get(key)}" for key in mismatched)
        raise ValueError(f"Rollout observation layout does not match the observation space ({details})")
