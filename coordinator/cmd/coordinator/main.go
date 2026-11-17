// The coordinator sits between the actors and the learner.
//
// It distributes the policy: the learner publishes each new version of the
// weights here, and actors long-poll for them, so the learner sends the weights
// once however many actors there are.
//
// Env: COORDINATOR_PORT [50052; 0 picks a free port], COORDINATOR_LONG_POLL_SECONDS [30].
package main

import (
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"google.golang.org/grpc"

	"sts/coordinator/gen/stsv1"
	"sts/coordinator/internal/server"
	"sts/coordinator/internal/weights"
)

// Rollouts are ~14 MB, over gRPC's 4 MB default. Same limit as rpc/codec.py.
const maxMessageBytes = 64 * 1024 * 1024

// envOr returns the environment variable name, or fallback if it is unset or empty.
func envOr(name, fallback string) string {
	if value := os.Getenv(name); value != "" {
		return value
	}
	return fallback
}

func main() {
	log.SetPrefix("coordinator: ")
	log.SetFlags(0) // docker already timestamps each line

	port := envOr("COORDINATOR_PORT", "50052")
	longPollSeconds, err := strconv.ParseFloat(envOr("COORDINATOR_LONG_POLL_SECONDS", "30"), 64)
	if err != nil {
		log.Fatalf("COORDINATOR_LONG_POLL_SECONDS: %v", err)
	}

	listener, err := net.Listen("tcp", ":"+port)
	if err != nil {
		log.Fatalf("listen on port %s: %v", port, err)
	}

	store := weights.NewStore()
	grpcServer := grpc.NewServer(grpc.MaxRecvMsgSize(maxMessageBytes), grpc.MaxSendMsgSize(maxMessageBytes))
	stsv1.RegisterActorsServer(grpcServer, &server.Actors{
		Store:    store,
		LongPoll: time.Duration(longPollSeconds * float64(time.Second)),
	})
	stsv1.RegisterTrainerServer(grpcServer, &server.Trainer{Store: store})

	// Stop cleanly on Ctrl-C or `docker stop` (SIGTERM). Serve below blocks, so
	// the signal is waited for on a separate goroutine.
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		received := <-signals
		log.Printf("got %v, stopping", received)
		grpcServer.Stop()
	}()

	// The real port, so a test that asked for port 0 can read which one it got.
	fmt.Printf("coordinator: listening on port %d\n", listener.Addr().(*net.TCPAddr).Port)
	if err := grpcServer.Serve(listener); err != nil {
		log.Fatalf("serve: %v", err)
	}
}
