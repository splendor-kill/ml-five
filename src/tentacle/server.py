import copy
import random
import socket
import struct
from threading import Thread

from tentacle.config import cfg
from tentacle.board import Board
from tentacle.checkpoint import latest_checkpoint
from tentacle.strategy_dnn import StrategyDNN


HOST = ""  # Symbolic name, meaning all available interfaces
PORT = 10000  # Arbitrary non-privileged port
MAX_MESSAGE_SIZE = 1024 * 1024


try:
    ConnectionResetError = ConnectionResetError
except NameError:

    class ConnectionResetError(Exception):
        """
        A HTTP connection was unexpectedly reset.
        """


def send_one_message(sock, data):
    length = len(data)
    #     print('send:', data)
    sock.sendall(struct.pack("!I", length))
    sock.sendall(data)


def recv_one_message(sock):
    lengthbuf = recvall(sock, 4)
    if lengthbuf is None:
        return None
    (length,) = struct.unpack("!I", lengthbuf)
    if length > MAX_MESSAGE_SIZE:
        raise ValueError("message too large")
    return recvall(sock, length)


def recvall(sock, count):
    buf = b""
    while count:
        newbuf = sock.recv(count)
        if not newbuf:
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def create_strategy():
    file = latest_checkpoint(cfg.RL_BRAIN_DIR)
    return StrategyDNN(from_file=file, part_vars=True)


class SessionState:
    def __init__(self):
        self.board = None
        self.s1 = create_strategy()
        self.first_query = True
        self.who_first = None


def dispose_msg(msg, msg_queue, state):
    # print('recv:', msg)

    ans = None
    seq = msg.split(" ")
    if len(seq) == 0 or seq[0] == "":
        return "ERROR: empty message"

    if seq[0] == "START:":
        if len(seq) != 2:
            return "ERROR: protocol inconsistent"
        board_size = int(seq[1])
        if board_size != Board.BOARD_SIZE:
            return "ERROR: board size mismatch"
        Board.set_board_size(board_size)
        state.board = Board()
        state.first_query = True
        state.who_first = None
        ans = "START: OK"
        if msg_queue is not None:
            msg_queue.put(("start",))
        state.s1.absorb("?")
        state.s1.on_episode_start()
    elif seq[0] == "MOVE:":
        if state.board is None:
            return "ERROR: game not started"
        if len(seq) < 4:
            return "ERROR: protocol inconsistent"
        old_board = copy.deepcopy(state.board)
        x, y = int(seq[1]), int(seq[2])
        who = Board.STONE_BLACK if int(seq[3]) == 1 else Board.STONE_WHITE
        if state.who_first is None:
            state.who_first = who
            print("who first?", state.who_first)
        if state.board.is_legal(x, y):
            state.board.move(x, y, who)

        state.s1.swallow(who, old_board, state.board)
        if msg_queue is not None:
            msg_queue.put(("move", who, x * Board.BOARD_SIZE + y))
    elif seq[0] == "WIN:":
        if state.board is None:
            return "ERROR: game not started"
        if len(seq) != 3:
            return "ERROR: protocol inconsistent"
        x, y = int(seq[1]), int(seq[2])
        who = state.board.get(x, y)
        print("player %d win the game" % (who,))
    elif seq[0] == "UNDO:":
        ans = "UNDO: unsupported yet"
    elif seq[0] == "WHERE:":
        if state.board is None:
            return "ERROR: game not started"
        if state.who_first is None:
            state.who_first = Board.STONE_BLACK
            print("who first?", state.who_first)
        if state.first_query:
            state.s1.stand_for = state.board.query_stand_for(state.who_first)
            print("i stand for:", state.s1.stand_for)
            state.first_query = False
        assert state.s1.stand_for is not None
        x, y = state.s1.preferred_move(state.board)
        ans = "HERE: %d %d" % (x, y)
    elif seq[0] == "END:":
        # s1.close()
        ans = "END: OK"
    else:
        ans = "ERROR: unknown command"

    return ans


class ClientThread(Thread):
    def __init__(self, conn, msg_queue):
        Thread.__init__(self)
        self.conn = conn
        self.msg_queue = msg_queue
        self.state = SessionState()

    def run(self):
        try:
            msg = "TOKEN: %d" % (random.randint(1, 1 << 30),)
            send_one_message(self.conn, msg.encode("ascii"))

            while True:
                msg = recv_one_message(self.conn)
                if msg is None:
                    break
                msg = msg.decode("ascii")
                ans = dispose_msg(msg, self.msg_queue, self.state)
                if ans is not None:
                    send_one_message(self.conn, ans.encode("ascii"))
        except (ConnectionResetError, BrokenPipeError):
            self.conn.close()
        except (ValueError, UnicodeDecodeError) as ex:
            send_one_message(self.conn, ("ERROR: %s" % (str(ex),)).encode("ascii"))
        finally:
            self.conn.close()


def net(msg_queue=None):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Socket created")

    # Bind socket to local host and port
    try:
        s.bind((HOST, PORT))
    except socket.error as msg:
        print("Bind failed. %s" % (str(msg),))
        raise

    print("Socket bind complete")

    # Start listening on socket
    s.listen(5)
    print("Socket now listening")

    # now keep talking with the client
    while True:
        # wait to accept a connection - blocking call
        conn, addr = s.accept()
        print("Connected with " + addr[0] + ":" + str(addr[1]))
        thread = ClientThread(conn, msg_queue)
        thread.start()

    s.close()


def main():
    net()


if __name__ == "__main__":
    main()
