// Package server implements the gRPC services from rpc/learner.proto that the
// coordinator serves: Actors (called by actors) and Trainer (called by the learner).
package server

import (
	"context"
	"log"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"sts/coordinator/gen/stsv1"
	"sts/coordinator/internal/weights"
)

// Actors serves the calls actors make.
type Actors struct {
	// Required by gRPC-Go: answers "unimplemented" for any RPC in the service
	// that this type doesn't define.
	stsv1.UnimplementedActorsServer
	Store *weights.Store
	// The longest a GetWeights call is held open before answering "not yet".
	LongPoll time.Duration
}

func (a *Actors) GetWeights(ctx context.Context, req *stsv1.GetWeightsRequest) (*stsv1.Weights, error) {
	w, ok, err := a.Store.Wait(ctx, req.MinVersion, req.RunId, a.LongPoll)
	if err != nil {
		// The actor's call ended (it hung up, or its deadline passed) while waiting.
		return nil, status.FromContextError(err).Err()
	}
	if ok {
		log.Printf("actor %s fetched weights v%d", req.ActorId, w.Version)
	}
	return w, nil
}

// Trainer serves the calls the learner makes.
type Trainer struct {
	stsv1.UnimplementedTrainerServer
	Store *weights.Store
}

func (t *Trainer) PublishWeights(ctx context.Context, w *stsv1.Weights) (*stsv1.PublishAck, error) {
	// Actors can't build a network without the config, or load empty weights.
	if w.Config == nil || len(w.StateDict) == 0 || w.RunId == "" {
		return nil, status.Error(codes.InvalidArgument, "weights need a config, a state_dict and a run_id")
	}
	replaced := t.Store.Publish(w)
	if replaced {
		log.Printf("published weights v%d from learner run %s, %.2f MB", w.Version, w.RunId, float64(len(w.StateDict))/1e6)
	}
	return &stsv1.PublishAck{Replaced: replaced}, nil
}

func (t *Trainer) CurrentVersion(ctx context.Context, _ *stsv1.CurrentVersionRequest) (*stsv1.WeightsVersion, error) {
	version, runID := t.Store.Version()
	return &stsv1.WeightsVersion{Version: version, RunId: runID}, nil
}
