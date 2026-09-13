"""The rollout wire format: encoding a completed rollout to bytes and back."""

import json
import struct
import zlib

import numpy as np
import pytest
from gymnasium import spaces

from model.rollout_codec import UnsupportedSchemaVersion, decode_rollout, encode_rollout

from tests.rollouts import ACTION_SPACE, fill, make_buffer

FIELDS = ["actions", "rewards", "dones", "values", "old_log_prob", "advantages", "returns"]


def test_a_rollout_decodes_to_an_equal_rollout(observation_space):
    rollout = fill(make_buffer(observation_space, size=16))

    decoded = decode_rollout(encode_rollout(rollout), observation_space, ACTION_SPACE)

    assert len(decoded) == 16
    assert decoded.full
    for name in FIELDS:
        np.testing.assert_array_equal(getattr(decoded, name), getattr(rollout, name), err_msg=name)
        assert getattr(decoded, name).dtype == getattr(rollout, name).dtype, name
    assert decoded.observations.keys() == rollout.observations.keys()
    for key, observations in rollout.observations.items():
        np.testing.assert_array_equal(decoded.observations[key], observations, err_msg=key)


def test_a_partly_filled_rollout_decodes_to_its_filled_steps(observation_space):
    rollout = make_buffer(observation_space, size=8)
    fill(rollout)
    rollout.pos = 5

    decoded = decode_rollout(encode_rollout(rollout), observation_space, ACTION_SPACE)

    assert len(decoded) == 5
    np.testing.assert_array_equal(decoded.rewards, rollout.rewards[:5])
    np.testing.assert_array_equal(decoded.observations["deck"], rollout.observations["deck"][:5])


def with_version(data, version):
    patched = bytearray(data)
    struct.pack_into("<H", patched, 4, version)
    return bytes(patched)


def test_a_rollout_keeps_the_id_it_was_encoded_with(observation_space):
    data = encode_rollout(fill(make_buffer(observation_space, size=4)), rollout_id="actor-3-rollout-7")

    assert decode_rollout(data, observation_space, ACTION_SPACE).rollout_id == "actor-3-rollout-7"


def test_each_rollout_encoded_without_an_id_is_given_a_distinct_one(observation_space):
    rollout = fill(make_buffer(observation_space, size=4))

    first, second = (decode_rollout(encode_rollout(rollout), observation_space, ACTION_SPACE) for _ in range(2))

    assert first.rollout_id and second.rollout_id
    assert first.rollout_id != second.rollout_id


@pytest.mark.parametrize("version", [0, 1, 3, 65535])
def test_an_unknown_schema_version_is_rejected(observation_space, version):
    data = with_version(encode_rollout(fill(make_buffer(observation_space, size=4))), version)

    with pytest.raises(UnsupportedSchemaVersion, match=str(version)):
        decode_rollout(data, observation_space, ACTION_SPACE)


@pytest.mark.parametrize("data", [b"", b"SRO", b"PK\x03\x04" + bytes(20)], ids=["empty", "truncated", "zip"])
def test_bytes_that_are_not_a_rollout_are_rejected(observation_space, data):
    with pytest.raises(ValueError, match="not an encoded rollout"):
        decode_rollout(data, observation_space, ACTION_SPACE)


def with_header(data, change):
    """Re-encode the header after change mutates it, keeping the body as it was."""
    _, version, header_length = struct.unpack_from("<4sHI", data)
    header = json.loads(data[10:10 + header_length])
    change(header)
    new_header = json.dumps(header).encode("utf-8")
    return struct.pack("<4sHI", b"SROL", version, len(new_header)) + new_header + data[10 + header_length:]


def set_entry(name, **fields):
    def change(header):
        next(entry for entry in header["arrays"] if entry["name"] == name).update(fields)
    return change


def drop_entry(name):
    def change(header):
        header["arrays"] = [entry for entry in header["arrays"] if entry["name"] != name]
    return change


@pytest.mark.parametrize("change, message", [
    (lambda header: header.update(compression="zstd"), "zstd"),
    (set_entry("observations/hand", dtype="float64", shape=[2, 10, 8]), "float64"),
    (lambda header: header.update(steps=3), "steps"),
    (drop_entry("returns"), "returns"),
    (lambda header: header.pop("rollout_id"), "rollout_id"),
], ids=["compression", "dtype", "steps", "missing-array", "missing-rollout-id"])
def test_a_header_this_reader_cannot_honour_is_rejected(observation_space, change, message):
    data = with_header(encode_rollout(fill(make_buffer(observation_space, size=4))), change)

    with pytest.raises(ValueError, match=message):
        decode_rollout(data, observation_space, ACTION_SPACE)


@pytest.mark.parametrize("corrupt", [
    lambda data: data[:-10],
    lambda data: data + b"\x00",
    lambda data: with_header(data, lambda header: header.pop("arrays")),
    lambda data: with_header(data, set_entry("rewards", offset=10**9)),
], ids=["truncated-body", "trailing-bytes", "header-without-arrays", "offset-past-body"])
def test_a_corrupt_rollout_is_rejected_as_corrupt(observation_space, corrupt):
    data = corrupt(encode_rollout(fill(make_buffer(observation_space, size=4))))

    with pytest.raises(ValueError, match="[Cc]orrupt"):
        decode_rollout(data, observation_space, ACTION_SPACE)


def changed_space(observation_space, **components):
    merged = {**observation_space.spaces, **components}
    return spaces.Dict({key: space for key, space in merged.items() if space is not None})


@pytest.mark.parametrize("components, component", [
    ({"hand": spaces.Box(0, 1, (10, 9), dtype=np.float32)}, "hand"),
    ({"relics": None}, "relics"),
    ({"orbs": spaces.Box(0, 1, (3,), dtype=np.float32)}, "orbs"),
], ids=["reshaped", "removed", "added"])
def test_an_observation_layout_the_learner_does_not_expect_is_rejected(observation_space, components, component):
    data = encode_rollout(fill(make_buffer(observation_space, size=4)))

    with pytest.raises(ValueError, match=component):
        decode_rollout(data, changed_space(observation_space, **components), ACTION_SPACE)


def read_without_project_code(data):
    """Parse an encoded rollout from the documented layout alone."""
    magic, version, header_length = struct.unpack_from("<4sHI", data)
    header = json.loads(data[10:10 + header_length])
    body = zlib.decompress(data[10 + header_length:])
    types = {"float32": "<f4"}
    arrays = {}
    for entry in header["arrays"]:
        count = int(np.prod(entry["shape"]))
        arrays[entry["name"]] = np.frombuffer(body, types[entry["dtype"]], count, entry["offset"]).reshape(entry["shape"])
    return magic, version, header, arrays


def test_the_encoding_is_readable_from_its_documented_layout(observation_space):
    rollout = fill(make_buffer(observation_space, size=6))

    magic, version, header, arrays = read_without_project_code(encode_rollout(rollout))

    assert (magic, version) == (b"SROL", 2)
    assert header["compression"] == "zlib"
    assert isinstance(header["rollout_id"], str)
    assert header["steps"] == 6
    expected_names = {f"observations/{key}" for key in observation_space.spaces} | set(FIELDS)
    assert set(arrays) == expected_names
    assert arrays["actions"].shape == (6, 1)
    np.testing.assert_array_equal(arrays["rewards"], rollout.rewards)
    np.testing.assert_array_equal(arrays["returns"], rollout.returns)
    np.testing.assert_array_equal(arrays["observations/deck"], rollout.observations["deck"])
