# Policy weights on a compacted topic, switched between rollouts

**Status:** accepted

## Context

Actors used to reload the learner's checkpoint from the working directory every
hundred rollouts. That only worked on a shared filesystem, could read an archive
the learner was still writing, and left actors up to a hundred rollouts behind.
Actors need the learner's newest weights over the broker, and PPO needs to know
how stale the policy that collected each rollout was.

## Decision

- **Weights are published as numbered versions.** The version is the number of
  updates the learner has applied. It is stored in the checkpoint, so it keeps
  counting across restarts. The learner publishes after each update is saved and
  before its rollout is committed, and republishes its current version on
  startup to cover a publish lost to a crash.
- **The payload is the policy's state dict**, encoded by
  `model/policy_codec.py`, at about 0.9MB. It travels as the message itself per
  ADR 0001. The optimizer state stays in the learner. The body is loaded with
  `weights_only=True` and checked against a SHA-256 digest in the header.
  Weights are loaded into the policy only if every tensor matches its shape, so
  a truncated or mismatched payload is rejected whole.
- **The policy topic has one partition and one key**, with `cleanup.policy`
  set to `compact` and one-hour segments. Only the newest version is guaranteed
  to be kept, and superseded ones are removed within about an hour.
- **Actors read without a consumer group.** A subscription jumps to the newest
  message on the topic, skipping versions it never used. A new actor therefore
  starts on the newest version, not the first.
- **Actors switch weights only at rollout boundaries**, after handing a rollout
  off. Each rollout is then collected by exactly one version. An actor waits
  for the first version before it plays at all.
- **Lag is reported per rollout.** At each hand-off the actor logs the version
  that collected the rollout and the newest version on the topic. The
  difference is how many updates the learner made while the rollout was
  collected.

## Reasoning

A compacted topic is the Kafka-native way to hold "the current value": the
newest weights are always there for an actor that starts late, and old versions
cannot pile up. One key on one partition keeps versions in publish order. The
one-hour segment matters because compaction never touches the active segment.
Under the default of a week, every version published that week would stay on
disk.

Shipping the state dict rather than the Stable Baselines3 archive leaves
unpickling out of the path from the broker into every actor. It also sends only
what an actor uses. The digest makes "partial weights are never loaded" a
property of the codec, whatever carries the bytes.

Jumping to the newest message, rather than reading every version in order,
keeps a slow actor from downloading versions it will discard. This matters once
many actors make the learner publish faster than one rollout completes.

Switching mid-rollout would give fresher weights, but a rollout collected by
several policies has no single version to measure staleness against. Actors
play at game speed, so the cost is at most one rollout of staleness. That
staleness is what the per-rollout lag report measures. A rollout collected
before any weights were published would come from a random policy with no
version, so actors wait for the first one instead.
