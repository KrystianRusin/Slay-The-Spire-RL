# Headless Slay the Spire 2. Two stages, picked with --target:
#
#   builder  .NET SDK + the sts2-cli source. Run once as a container, with the
#            game mounted read-only at /game: it copies the game's DLLs,
#            IL-patches sts2.dll, and compiles the headless host into /out.
#   runtime  (default) .NET runtime + Python + our bots. Mounts the builder's
#            /out and the game, and runs the host as a child process.
#
# Neither image contains any game file. The game is proprietary, so everything
# made from it (the copied and patched DLLs, the text extracted from the .pck)
# exists only in the builder's output folder on the host, never in a layer.

# --- builder --------------------------------------------------------------
FROM mcr.microsoft.com/dotnet/sdk:9.0-bookworm-slim AS builder

# python3 runs sts2-cli's script that pulls the English text tables out of the
# game's .pck; git fetches the source.
RUN apt-get update && apt-get install -y --no-install-recommends python3 git \
    && rm -rf /var/lib/apt/lists/*

# sts2-cli (MIT): a C# host that runs the game's logic with Godot replaced by
# stubs, driven by JSON lines on stdin/stdout. Pinned to one commit: its IL
# patches and adapters are written against specific game builds.
ARG STS2_CLI_REPO=https://github.com/wuhao21/sts2-cli.git
ARG STS2_CLI_COMMIT=fec723a
RUN git clone -q "$STS2_CLI_REPO" /src/sts2-cli \
    && cd /src/sts2-cli && git checkout -q "$STS2_CLI_COMMIT" \
    # The repo ships text tables extracted from the game. Drop them so this
    # image holds no game content; the build regenerates them from /game.
    && rm -rf .git localization_eng localization_zhs

# Warm the NuGet cache with the one package the patch step needs (Mono.Cecil),
# so building the host needs no network.
RUN mkdir /tmp/warm && cd /tmp/warm \
    && dotnet new console -o . --force >/dev/null \
    && dotnet add package Mono.Cecil --version 0.11.6 >/dev/null \
    && dotnet restore >/dev/null \
    && cd / && rm -rf /tmp/warm

COPY docker/sts2/build-host.sh /opt/sts2/build-host.sh
COPY docker/sts2/game-version.sh /opt/sts2/game-version.sh
ENV DOTNET_CLI_TELEMETRY_OPTOUT=1 \
    DOTNET_NOLOGO=1 \
    # NuGet's cache, from the warm-up above. The container runs as the host
    # UID, which can't write to /root, so point everything at the image's copy
    # and a writable home under /tmp.
    NUGET_PACKAGES=/root/.nuget/packages \
    HOME=/tmp/home
RUN chmod -R a+rwX /root
ENTRYPOINT ["/opt/sts2/build-host.sh"]

# --- runtime --------------------------------------------------------------
# Python 3.13 to match the repo; the .NET runtime is copied in from
# Microsoft's image (same Debian 12 base), like the Go binary below. One image
# for the actor (game + policy) and the learner (policy only).
FROM python:3.13-slim-bookworm AS runtime

# libicu: .NET's culture and string-comparison data, which the game uses.
RUN apt-get update && apt-get install -y --no-install-recommends libicu72 procps \
    && rm -rf /var/lib/apt/lists/*
COPY --from=mcr.microsoft.com/dotnet/runtime:9.0-bookworm-slim /usr/share/dotnet /usr/share/dotnet
RUN ln -s /usr/share/dotnet/dotnet /usr/local/bin/dotnet

# CPU-only torch: the actor only runs inference, and the CUDA build is ~5 GB.
# Installed first so the requirements below see it as already satisfied.
COPY requirements.txt /opt/sts/app/requirements.txt
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /opt/sts/app/requirements.txt

COPY . /opt/sts/app/
# Generate the gRPC code from the contract. Not committed, so it can't drift from the .proto.
RUN cd /opt/sts/app && python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. rpc/learner.proto
COPY docker/sts2/entrypoint.sh /opt/sts2/entrypoint.sh
COPY docker/sts2/game-version.sh /opt/sts2/game-version.sh
# The Go coordinator's binary, so the Python tests can start the real one. The
# "coordinator" build context is the coordinator image (see compose.yaml).
COPY --from=coordinator /coordinator /opt/sts/bin/coordinator

ENV STS2_GAME_DIR=/game \
    STS2_HOST_DIR=/sts2-host \
    STS2_INSTANCE_DIR=/instance \
    DOTNET_CLI_TELEMETRY_OPTOUT=1 \
    # Many games share the CPU: keep .NET from sizing its GC and thread pool
    # for the whole host.
    DOTNET_gcServer=0 \
    DOTNET_GCHeapCount=1 \
    PYTHONPATH=/opt/sts/app \
    PYTHONUNBUFFERED=1 \
    # The container runs as the host UID, which has no passwd entry in the
    # image; torch looks the username up (getpass) and crashes without this.
    USER=sts

ENTRYPOINT ["/opt/sts2/entrypoint.sh"]
