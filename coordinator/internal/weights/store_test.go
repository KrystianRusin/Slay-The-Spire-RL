package weights

import (
	"context"
	"sync"
	"testing"
	"time"

	"sts/coordinator/gen/stsv1"
)

// A long timeout for waits that are meant to succeed: if one hangs, the test fails slowly instead of passing.
const forever = 10 * time.Second

func weightsAt(version int64, runID string) *stsv1.Weights {
	return &stsv1.Weights{Version: version, RunId: runID, StateDict: []byte("weights"), Config: &stsv1.PolicyConfig{NSteps: 16}}
}

func TestWaitReturnsAtOnceWhenTheVersionIsAlreadyThere(t *testing.T) {
	store := NewStore()
	store.Publish(weightsAt(3, "run-a"))

	w, ok, err := store.Wait(context.Background(), 2, "run-a", forever)

	if err != nil || !ok || w.Version != 3 {
		t.Fatalf("got version %d, ok=%v, err=%v; want version 3", w.GetVersion(), ok, err)
	}
}

func TestWaitTimesOutWithTheCurrentVersionButNoWeights(t *testing.T) {
	store := NewStore()
	store.Publish(weightsAt(3, "run-a"))

	w, ok, err := store.Wait(context.Background(), 4, "run-a", 50*time.Millisecond)

	if err != nil || ok {
		t.Fatalf("ok=%v err=%v, want a timeout", ok, err)
	}
	if w.Version != 3 || w.RunId != "run-a" || w.StateDict != nil || w.Config != nil {
		t.Fatalf("got %v, want only version 3 and run-a", w)
	}
}

func TestWaitTimesOutEmptyBeforeAnythingIsPublished(t *testing.T) {
	w, ok, _ := NewStore().Wait(context.Background(), 0, "", 50*time.Millisecond)

	if ok || w.Config != nil {
		t.Fatalf("got %v ok=%v, want an empty not-yet reply", w, ok)
	}
}

func TestEveryWaiterWakesWhenTheVersionIsPublished(t *testing.T) {
	store := NewStore()
	store.Publish(weightsAt(0, "run-a"))

	const waiters = 20
	got := make(chan int64, waiters) // room for every result, so no waiter blocks on sending
	var wg sync.WaitGroup
	for range waiters {
		wg.Add(1)
		go func() {
			defer wg.Done()
			w, _, _ := store.Wait(context.Background(), 1, "run-a", forever)
			got <- w.Version
		}()
	}
	time.Sleep(50 * time.Millisecond) // let them all start waiting
	store.Publish(weightsAt(1, "run-a"))
	wg.Wait()
	close(got)

	count := 0
	for version := range got {
		count++
		if version != 1 {
			t.Errorf("a waiter got version %d, want 1", version)
		}
	}
	if count != waiters {
		t.Fatalf("%d of %d waiters returned", count, waiters)
	}
}

func TestAWaiterFromAnEarlierLearnerRunGetsTheCurrentWeightsAtOnce(t *testing.T) {
	store := NewStore()
	store.Publish(weightsAt(5, "run-b")) // a restarted learner, back at version 5

	start := time.Now()
	w, ok, _ := store.Wait(context.Background(), 6, "run-a", forever) // still waiting for 6 from run-a

	if !ok || w.Version != 5 || w.RunId != "run-b" {
		t.Fatalf("got version %d run %q ok=%v, want version 5 from run-b", w.GetVersion(), w.GetRunId(), ok)
	}
	if time.Since(start) > time.Second {
		t.Fatalf("waited %v, want an immediate answer", time.Since(start))
	}
}

func TestAWaiterWakesWhenTheLearnerRestartsWhileItWaits(t *testing.T) {
	store := NewStore()
	store.Publish(weightsAt(5, "run-a"))
	done := make(chan *stsv1.Weights)
	go func() {
		w, _, _ := store.Wait(context.Background(), 6, "run-a", forever)
		done <- w
	}()
	time.Sleep(50 * time.Millisecond)

	store.Publish(weightsAt(5, "run-b"))

	select {
	case w := <-done:
		if w.RunId != "run-b" {
			t.Fatalf("got run %q, want run-b", w.RunId)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("the waiter never saw the restarted learner")
	}
}

func TestRepublishingTheSameVersionIsANoOp(t *testing.T) {
	store := NewStore()
	if !store.Publish(weightsAt(1, "run-a")) {
		t.Fatal("the first publish should replace nothing and count as replaced")
	}
	if store.Publish(weightsAt(1, "run-a")) {
		t.Fatal("the same version and run again should be a no-op")
	}
	if !store.Publish(weightsAt(1, "run-b")) {
		t.Fatal("the same version from a new run should replace")
	}
}

func TestWaitStopsWhenTheCallerHangsUp(t *testing.T) {
	store := NewStore()
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()

	_, ok, err := store.Wait(ctx, 1, "", forever)

	if ok || err != context.Canceled {
		t.Fatalf("ok=%v err=%v, want context.Canceled", ok, err)
	}
}
