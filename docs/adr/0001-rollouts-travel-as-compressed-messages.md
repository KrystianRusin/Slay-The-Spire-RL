# Rollouts travel as compressed messages, not by claim-check

**Status:** accepted

## Context

A completed rollout is encoded by `model/rollout_codec.py`. At 2048 steps of
1,589 float32 observation values it is about 13MB before compression, against
Kafka's default 1MB message limit. The two options were to store the payload in
object storage and publish a pointer (claim-check), or to publish the payload
itself under raised size limits.

Measured sizes of one encoded 2048-step rollout:

| Observations                          | Size          |
| ------------------------------------- | ------------- |
| Any, arrays before compression        | 13,074,432 B  |
| Fixture game states, encoded          | 97,368 B      |
| Uniform random noise, encoded         | 11,727,369 B  |

The fixture row cycles seven game states, so it overstates compression. For
real play, where the deck, relics and map barely change between steps, a
conservative estimate is about 794,000B: each fixture observation compressed on
its own is about 360B, since roughly 320 of its 1,589 values are non-zero.
Encoding takes about 90ms and decoding about 30ms.

## Decision

Rollouts are published as the message payload itself.

- The encoding is always zlib-compressed, and the producer's own compression
  stays off, so the size checked against the limits is the size on the wire.
- The rollout topic sets `max.message.bytes` to 16MB, with producer
  `max.request.size` and consumer fetch sizes to match, all as committed topic
  and client configuration. 16MB clears the incompressible worst case, so
  delivery never depends on how well a rollout compresses.
- Published policy weights use the same approach; the 2.7MB model archive fits
  under the same limit.

## Reasoning

Rollouts arrive about once every few minutes per actor, so larger messages cost
nothing measurable in broker throughput, and real rollouts are expected to
compress to under 1MB. Claim-check would add an object store to deploy, secure
and monitor, and a garbage-collection policy that has to agree with topic
retention and with redelivery of uncommitted rollouts. A pointer whose object
has already been collected is a new failure mode; a message that is on the
topic always carries its data.

Revisit if rollouts grow by an order of magnitude (longer rollouts or image
observations), or if retention needs to hold far more data than the broker's
disks should.
