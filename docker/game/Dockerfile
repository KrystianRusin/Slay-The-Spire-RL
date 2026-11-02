# Headless Slay the Spire: Xvfb + ModTheSpire + CommunicationMod + a bot.
#
# The game itself is NOT in this image. It's proprietary, so it is bind-mounted
# read-only at /game at run time (see compose.yaml). The image only carries the
# system libraries the game needs, a virtual display, and our Python code.
FROM debian:12-slim

# xvfb             virtual X display the game renders into
# x11-xserver-utils  provides `xrandr`, which LWJGL 2 runs to list display modes
# libgl1-mesa-dri  software OpenGL (llvmpipe), since there is no GPU in the container
# libx*            X client libs LWJGL 2 loads at startup
# fontconfig/fonts  Java 8 AWT crashes (NPE in FontConfiguration) with no system fonts
# python3          runs the bot / middleman that CommunicationMod spawns
RUN apt-get update && apt-get install -y --no-install-recommends \
        xvfb x11-xserver-utils \
        libgl1 libgl1-mesa-dri \
        libxrandr2 libxcursor1 libxxf86vm1 libxi6 libxrender1 libxtst6 \
        fontconfig fonts-dejavu-core \
        python3 procps \
    && rm -rf /var/lib/apt/lists/* \
    # Xvfb runs as a normal user, so the socket dir must already exist with sticky perms.
    && mkdir -m 1777 -p /tmp/.X11-unix

COPY bots/ /opt/sts/bots/
COPY docker/game/entrypoint.sh /opt/sts/entrypoint.sh

# Per-run state (saves, prefs, logs) lives here; compose mounts a host dir over it.
ENV STS_GAME_DIR=/game \
    STS_INSTANCE_DIR=/instance \
    # Software render with one thread: many instances share the CPU.
    LP_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2

ENTRYPOINT ["/opt/sts/entrypoint.sh"]
