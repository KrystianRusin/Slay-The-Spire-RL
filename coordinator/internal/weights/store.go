// Package weights holds the latest policy weights and lets actors wait for new ones.
//
// The learner publishes a version; any number of actors wait for "at least
// version N" (a long poll). Every waiter wakes when a new version arrives.
package weights

import (
	"context"
	"sync"
	"time"

	"sts/coordinator/gen/stsv1"
)

// Store is safe to use from many goroutines at once. gRPC serves every call on
// its own goroutine, so many GetWeights calls and a PublishWeights can overlap.
type Store struct {
	mu      sync.Mutex     // guards the two fields below
	current *stsv1.Weights // nil until the learner first publishes
	// Closed (and replaced by a fresh one) on every publish. A closed channel
	// is ready to receive from for everyone, so this wakes every waiter at once.
	changed chan struct{}
}

func NewStore() *Store {
	return &Store{changed: make(chan struct{})}
}

// Publish makes w the current weights, unless it is the same version from the
// same learner run as the current ones. Returns whether it replaced them.
//
// Any version is accepted, even an older one: a learner that restarted without
// its checkpoints starts again at 0 with a new run ID, and actors must follow it.
func (s *Store) Publish(w *stsv1.Weights) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.current != nil && s.current.RunId == w.RunId && s.current.Version == w.Version {
		return false // a repeat; don't wake anyone
	}
	s.current = w
	close(s.changed)
	s.changed = make(chan struct{})
	return true
}

// Version returns the version and run ID of the current weights, or 0 and ""
// if nothing has been published yet.
func (s *Store) Version() (version int64, runID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.current == nil {
		return 0, ""
	}
	return s.current.Version, s.current.RunId
}

// Wait returns the current weights once their version is at least minVersion,
// waiting up to timeout. If the weights come from a different learner run than
// runID (and runID isn't empty), it returns them at once whatever the version:
// the version being waited for may have died with the old run.
//
// ok is false if the wait timed out. Then the returned message has no weights,
// only the current version and run ID (both empty if nothing was published), and
// the caller asks again.
func (s *Store) Wait(ctx context.Context, minVersion int64, runID string, timeout time.Duration) (w *stsv1.Weights, ok bool, err error) {
	timer := time.NewTimer(timeout)
	defer timer.Stop()

	for {
		s.mu.Lock()
		current, changed := s.current, s.changed
		s.mu.Unlock()

		if current != nil {
			restarted := runID != "" && runID != current.RunId
			if restarted || current.Version >= minVersion {
				return current, true, nil
			}
		}

		// Block until whichever happens first. The lock is not held here, so
		// Publish can run meanwhile.
		select {
		case <-changed: // something was published; check again
		case <-timer.C:
			if current == nil {
				return &stsv1.Weights{}, false, nil
			}
			return &stsv1.Weights{Version: current.Version, RunId: current.RunId}, false, nil
		case <-ctx.Done(): // the caller hung up or its deadline passed
			return nil, false, ctx.Err()
		}
	}
}
