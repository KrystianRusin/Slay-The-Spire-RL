# At-least-once rollouts, deduplicated by ID, and lag accepted rather than throttled

**Status:** accepted

## Context

Kafka has no per-message acknowledgement. What the learner has consumed is
whatever offset it last committed, and anything after that offset is delivered
again when the learner restarts or its partitions are rebalanced. Separately,
actors publish whether or not the learner keeps up, so a learner that falls
behind builds up a backlog on the topic.

## Decision

**Delivery is at-least-once.** Auto-commit is off. The learner commits a
rollout's offset only after its update is in a saved checkpoint, so a learner
killed mid-update receives that rollout again on restart.

**Duplicates are skipped by rollout ID.** Each encoded rollout carries a
`rollout_id` in its header (schema version 2). The checkpoint stores the
learner's step count and the IDs of the last 10,000 rollouts it applied, as an
extra entry in the same archive as the weights, written to a temporary file
and moved into place. A rollout whose ID is already in the checkpoint is
committed without training on it. Rollouts that cannot be decoded are logged
and committed too, so one bad message cannot stop the learner. The exception
is a rollout from a newer schema version than the learner knows: the learner
stops without committing it, because the learner is the one out of date.

**Lag is accepted and measured, not throttled.** After every commit the
learner measures consumer lag, meaning the rollouts on its partitions not yet
committed. It logs the lag, and warns when it reaches 5. Actors never slow
down or drop rollouts because of lag.

## Reasoning

A redelivered rollout is always one the learner received but did not commit.
With a commit after every update, that is at most the rollout in flight when
the learner died, plus any whose commit failed during a rebalance. The 10,000
ID window is far beyond that, and stays small in the checkpoint. The window is
in the checkpoint rather than in memory so that it survives the restart that
causes the redelivery. Because it is saved in the same file as the weights,
the two cannot disagree about which rollouts have been applied.

Deduplicating on stored partition offsets was the alternative. It would have
needed no format change, but it breaks silently if the topic is recreated and
its offsets start again.

Throttling actors buys little here. Actors produce at the speed of a real game,
so pausing one leaves a game instance idle, and dropping a rollout at the
actor throws away minutes of play. One update takes a few seconds against a
rollout every few minutes per actor, so sustained lag means the learner was
down or is badly outnumbered, and the fix is operational. Staleness is better
judged at the learner, which knows its own policy version, by dropping
rollouts too many versions behind it. Each waiting rollout is one more update
the policy makes before that rollout is used, so the warning at 5 is where
staleness starts to cost PPO's clipped ratio, until a bound at the learner
takes over as the real control.

A restarted learner joins the group at once, but is not assigned partitions
until the broker's session timeout, 45 seconds by default, expires the member
that died. Nothing is lost or duplicated in that window. It only delays
training.
