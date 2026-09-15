# Metrics scraped by Prometheus, lag measured from outside the consumer group

**Status:** accepted

## Context

Diagnostics were print statements and a chart the actor wrote to a PNG. Once
the learner and several actors run as separate processes, their output
interleaves and the chart shows one actor on one machine. The backpressure
decision in docs/adr/0003, accepting lag rather than throttling actors, needs
lag to be visible over time, not asserted.

## Decision

- **Each process serves its own metrics for Prometheus to scrape.** The
  learner serves `/metrics` on `LEARNER_METRICS_PORT` (8000 by default), and
  each actor on `--metrics-port` (8100 plus its env id by default). Metric
  names start `sts_learner_` or `sts_actor_`, and every actor metric carries an
  `actor_id` label.
- **Consumer lag is measured by the learner, from outside its group.** Every
  10 seconds a separate consumer reads the learner group's committed offsets
  and the rollout topic's watermarks, and exports the lag per partition. It
  never subscribes or commits, so it does not join the group. When a
  measurement fails, the lag series is dropped rather than left at its last
  value.
- **Rollout age comes from the Kafka message timestamp.** The learner records
  how long before an update each rollout was published.
- **Policy lag is measured by each actor.** When an actor publishes a rollout
  it records how many versions the learner had published past the one that
  collected it.
- **Logs are JSON lines on stderr**, one object per record with time, level,
  logger, message, the service, and for actors the actor id. `LOG_LEVEL`
  sets the level. Per-step detail, such as each reward component, is at DEBUG.
- **A Grafana dashboard is provisioned from `monitoring/`**, with Prometheus
  scraping the processes on the host.

## Reasoning

Pulling suits long-running processes: Prometheus notices a process that stops
answering, which a push model only sees as silence. A Pushgateway would also
keep the last value of a dead actor forever.

Measuring lag from the commit path, as before, updated it only when the
learner committed. A learner stuck in a slow update would show a flat lag
line exactly when lag was rising. A probe on its own thread keeps measuring
while the update runs. It uses its own consumer because the learner's
consumer is busy in the training loop. A standalone Kafka lag exporter would
also work, but would need network access to the broker from wherever
Prometheus runs, which the single-node local broker does not advertise.

The Kafka timestamp is set by the producer when the actor publishes, so age
needs no change to the rollout format. It compares clocks on two hosts, which
docs/adr/0005 already requires to agree within a few seconds.

The learner cannot compute policy lag per actor: a rollout does not carry the
version that collected it. The actor knows both versions at the moment it
publishes.

Static scrape targets mean a new actor beyond the fourth needs a line in
`monitoring/prometheus.yml`. Service discovery replaces this once the system
runs on Kubernetes.
