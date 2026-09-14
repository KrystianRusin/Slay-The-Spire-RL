import json
import socket
import struct
import time

# Each message on the socket is a 4-byte big-endian payload length, then that
# many bytes of UTF-8 text.
HEADER = struct.Struct(">I")
RECV_SIZE = 4096
MAX_MESSAGE_BYTES = 16 * 1024 * 1024
SOCKET_TIMEOUT_SECONDS = 10
MAX_RETRIES = 10


class FramedConnection:
    """Length-prefixed text messages over a stream socket.

    Bytes read past the end of one message are kept for the next receive, so
    messages coalesced into a single read or split across several both arrive
    intact.
    """

    def __init__(self, sock):
        self.sock = sock
        self._buffer = bytearray()

    def send(self, text):
        payload = text.encode("utf-8")
        self.sock.sendall(HEADER.pack(len(payload)) + payload)

    def receive(self):
        """Block until one whole message arrives and return it as text.

        Raises TimeoutError if the socket times out first, and ConnectionError
        if the peer closes the connection or announces an oversized message.
        After either, the stream position is unknown: discard the connection.
        """
        header = self._read_exactly(HEADER.size)
        (length,) = HEADER.unpack(header)
        if length > MAX_MESSAGE_BYTES:
            raise ConnectionError(f"Message of {length} bytes exceeds the {MAX_MESSAGE_BYTES} byte limit")
        return self._read_exactly(length).decode("utf-8")

    def receive_json(self):
        return json.loads(self.receive())

    def close(self):
        self.sock.close()

    def _read_exactly(self, size):
        while len(self._buffer) < size:
            part = self.sock.recv(RECV_SIZE)
            if not part:
                raise ConnectionError("Socket connection closed")
            self._buffer += part
        data = bytes(self._buffer[:size])
        del self._buffer[:size]
        return data


class RetriesExhausted(Exception):
    """A Backoff was asked to wait again after its last retry."""


class GameUnreachable(RuntimeError):
    """The actor could not reach its game within its retry budget."""


class Backoff:
    """Waits that double on each retry up to a ceiling, and start over once reset.

    With max_retries, a wait beyond that many since the last reset raises
    RetriesExhausted instead of sleeping.
    """

    def __init__(self, initial_seconds=1.0, max_seconds=30.0, max_retries=None, sleep=time.sleep):
        self.initial_seconds = initial_seconds
        self.max_seconds = max_seconds
        self.max_retries = max_retries
        self._sleep = sleep
        self._next = initial_seconds
        self._retries = 0

    def wait(self):
        if self.max_retries is not None and self._retries >= self.max_retries:
            raise RetriesExhausted(f"Gave up after {self._retries} retries")
        self._retries += 1
        self._sleep(self._next)
        self._next = min(self._next * 2, self.max_seconds)

    def reset(self):
        self._next = self.initial_seconds
        self._retries = 0


class GameConnection:
    """Framed messages to a game's middleman, reconnecting with backoff whenever the connection drops.

    locate returns the (host, port) the middleman listens on, or None while
    it is not known, and is called again for every connection attempt, so a
    middleman restarted on a new port is found. Whatever locate raises stops
    the connection.

    receive blocks until a message arrives, reconnecting as often as the
    backoff allows, and raises GameUnreachable once its retries run out. A
    send that fails raises ConnectionError, since the game never got the
    message, and the next receive reconnects. Once abandoned, every send and
    receive raises the error given.
    """

    def __init__(self, locate, backoff=None, connect=None):
        self._locate = locate
        self.backoff = backoff or Backoff()
        self._connect = connect or (lambda address: socket.create_connection(address, timeout=SOCKET_TIMEOUT_SECONDS))
        self._connection = None
        self._address = None
        self._abandoned = None

    def send(self, text):
        self._check_abandoned()
        try:
            self._connected().send(text)
        except OSError as error:
            self._drop(error)
            raise ConnectionError(f"Could not send to the game at {self._address}") from error

    def receive(self):
        self._check_abandoned()
        while True:
            try:
                message = self._connected().receive()
            except OSError as error:
                self._drop(error)
                continue
            self.backoff.reset()
            return message

    def receive_json(self):
        return json.loads(self.receive())

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def abandon(self, error):
        """Make the next send or receive close the connection and raise error. Safe to call from another thread."""
        self._abandoned = error

    def _check_abandoned(self):
        if self._abandoned is not None:
            self.close()
            raise self._abandoned

    def _connected(self):
        while self._connection is None:
            self._address = self._locate()
            if self._address is None:
                print("The game is not registered as running")
                self._retry()
                continue
            try:
                self._connection = FramedConnection(self._connect(self._address))
            except OSError as error:
                print(f"Could not connect to the game at {self._address}: {error}")
                self._retry()
        return self._connection

    def _drop(self, error):
        print(f"Lost the connection to the game at {self._address}: {error}")
        self.close()
        self._retry()

    def _retry(self):
        try:
            self.backoff.wait()
        except RetriesExhausted as error:
            raise GameUnreachable(f"Could not reach the game, last registered at {self._address}: {error}") from error


def handle_end_of_episode(connection):
    """
    Handles the end-of-episode scenario by sending the necessary commands
    to navigate through the game over screen and start a new game.
    """
    commands = ["PROCEED", "PROCEED"]

    for command in commands:
        try:
            connection.send(command)
            print(f"Sent '{command}' command")
            connection.receive_json()
            print(f"Game state received after '{command}'")

        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON after '{command}': {e}")
            return
        except ConnectionError as e:
            print(f"Connection error after '{command}': {e}")
            return
