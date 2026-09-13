# Rollout topic layout

**Status:** accepted

## Context

Actors publish encoded rollouts to one Kafka topic, and a single learner reads
it as a consumer group member. The topic is declared in `broker/topics.py` and
applied by `ensure_topics`, which both the learner and every actor run at
startup.

## Decision

- **6 partitions.** A consumer group gives each partition to at most one
  member, so this caps the learner at 6 parallel consumers. It does not affect
  throughput today: each actor sends a rollout every few minutes.
- **Keyed by actor id**, so each actor's rollouts stay in order within one
  partition.
- **Retention of 24 hours.** Rollouts published while the learner is down
  overnight are still on the topic when it returns.
- **`max.message.bytes` of 16MB**, per ADR 0001.

## Reasoning

Partitions can be added later but never removed, and adding them moves keys to
different partitions. Six is room for a handful of data-parallel learners, which
is as far as one training run realistically scales. Many more would only add
broker overhead and slower rebalances for no benefit. `ensure_topics` refuses to
continue if the partition count does not match, rather than repartitioning on
its own.

Rollouts go stale for on-policy training within minutes, so long retention does
not make them more useful. The 24 hours only has to outlast learner downtime.
At the ADR 0001 estimate of about 0.8MB a rollout, one actor fills less than
half a gigabyte a day.
