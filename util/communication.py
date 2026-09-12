import json
import struct

# Each message on the socket is a 4-byte big-endian payload length, then that
# many bytes of UTF-8 text.
HEADER = struct.Struct(">I")
RECV_SIZE = 4096
MAX_MESSAGE_BYTES = 16 * 1024 * 1024


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

    def _read_exactly(self, size):
        while len(self._buffer) < size:
            part = self.sock.recv(RECV_SIZE)
            if not part:
                raise ConnectionError("Socket connection closed")
            self._buffer += part
        data = bytes(self._buffer[:size])
        del self._buffer[:size]
        return data


def handle_end_of_episode(connection):
    """
    Handles the end-of-episode scenario by sending the necessary commands
    to navigate through the game over screen and start a new game.
    """
    commands = ["PROCEED", "PROCEED"]

    for command in commands:
        connection.send(command)
        print(f"Sent '{command}' command")

        try:
            connection.receive_json()
            print(f"Game state received after '{command}'")

        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON after '{command}': {e}")
            return
        except ConnectionError as e:
            print(f"Connection error after '{command}': {e}")
            return
