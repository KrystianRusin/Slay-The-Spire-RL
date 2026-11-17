# The Go coordinator. Three stages, picked with --target:
#   build    compiles it (and generates the gRPC code from rpc/learner.proto)
#   test     build + go test with the race detector:
#              docker build -f docker/coordinator/Dockerfile --target test .
#   runtime  (default) just the static binary on a minimal base image, ~20 MB
# Build context is the repo root, so the .proto is reachable.

FROM golang:1.26-bookworm AS build
# protoc turns the .proto into code; the two plugins make the Go message types
# and the gRPC service stubs. Pinned to the versions go.mod uses.
RUN apt-get update && apt-get install -y --no-install-recommends protobuf-compiler \
    && rm -rf /var/lib/apt/lists/* \
    && go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.36.12 \
    && go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.6.2

WORKDIR /src/coordinator
# Dependencies first, in their own layer, so editing Go code doesn't re-download them.
COPY coordinator/go.mod coordinator/go.sum ./
RUN go mod download

COPY rpc/learner.proto /src/rpc/learner.proto
COPY coordinator/ ./
# Generated code isn't committed (like the Python _pb2 files), so it can't drift from the .proto.
RUN protoc -I /src \
        --go_out=. --go_opt=module=sts/coordinator \
        --go-grpc_out=. --go-grpc_opt=module=sts/coordinator \
        rpc/learner.proto \
    && go vet ./... \
    # CGO_ENABLED=0: a fully static binary, so it runs on a base image with no C library.
    && CGO_ENABLED=0 go build -o /out/coordinator ./cmd/coordinator

FROM build AS test
RUN go test -race -count=1 ./...

FROM gcr.io/distroless/static-debian12:nonroot AS runtime
COPY --from=build /out/coordinator /coordinator
ENTRYPOINT ["/coordinator"]
