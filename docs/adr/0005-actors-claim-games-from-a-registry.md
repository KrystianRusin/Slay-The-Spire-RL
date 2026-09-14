# Actors claim game instances from a registry in the database

**Status:** accepted

## Context

Each middleman used to scan upward from port 9999 for the first free port, and
each actor was started with the port it should dial. Which game sat on which
port depended on the order the games started, and the scan tested a port by
binding and closing it before binding again, so another process could take the
port in between. Actors need to find a game of their own wherever it listens,
find it again when its middleman restarts, and give up when it is gone.

## Decision

- **Middlemen bind once and register.** A middleman binds `STS_MIDDLEMAN_PORT`,
  or any free port when that is unset, and records the game id, the host
  actors should dial (`STS_ADVERTISED_HOST`) and the port it actually got in
  the `game_instances` table. It re-registers every 10 seconds, and removes its
  row when the game closes its input.
- **A game's id outlives its middleman.** Unless `STS_GAME_ID` is set, the id
  is the host name and the id of the game process that started the middleman.
  A middleman restarted by the same game registers under the same id, with its
  new port, and any claim on the game is kept.
- **Actors claim a game.** An actor atomically claims a game registered within
  the last 60 seconds that no actor has renewed a claim on within 60 seconds,
  preferring one it already owns. It renews its claim every 10 seconds and
  releases it on exit. With no game free, it waits and polls.
- **Actors look the game up on every connection attempt.** A connection that
  drops or cannot be opened is retried with doubling waits, looking up the
  game's current address each time. The retry budget, 10 by default, starts
  over whenever a message arrives. Once it runs out the actor exits with an
  error. An actor whose claim has passed to another actor exits at its next
  message to or from the game.
- **The middleman hands an unanswered state to the next actor.** The game sends
  each state once and waits for a command. If the actor drops before answering,
  the middleman sends that state first to the actor that connects next.

## Reasoning

A claim has to be a compare-and-set, so that two actors checking the same free
game cannot both take it. Postgres gives that with one conditional `UPDATE`,
and actors already depend on it for game records. Kafka has no such operation:
a consumer group could hand out partitions as games, but the partition count
would have to match the number of games.

Heartbeats are needed because neither side can be trusted to clean up.
Communication Mod kills the middleman without warning when the game exits, and
an actor can die the same way. Expiry is on the database's rows but measured
by each process's clock, so hosts need clocks that agree to within a few
seconds, well inside the 60-second window.

The game id comes from the parent process so a game needs no configuration:
Communication Mod starts the middleman as a direct child of the game, and
starts a new one as a child of the same process. `STS_GAME_ID` covers a game
launched through a wrapper script, where the parent changes each time.

Ten retries at waits capped at 30 seconds give about three minutes, time
enough to restart a middleman by hand, before an actor gives up. An actor that
gives up releases its claim, so a supervisor can restart it and it claims
whichever game is running.

A database outage does not stop an actor that is already connected: the
address is only looked up to reconnect. If the outage outlasts the 60 seconds,
another waiting actor may claim the game and wait in the middleman's accept
queue. The first actor exits once its next renewal fails, and the middleman
hands the game, with any unanswered state, to the second.

The middleman reads the game's output on a background thread, so it notices
the game closing even while no actor is connected, and removes the game rather
than keeping a dead one registered.
