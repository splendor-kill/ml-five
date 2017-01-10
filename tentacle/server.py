import copy
import random
import socket
import struct
import sys
from threading import Thread

from tentacle.board import Board
from tentacle.strategy_dnn import StrategyDNN


HOST = ''  # Symbolic name, meaning all available interfaces
PORT = 10000  # Arbitrary non-privileged port


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
    sock.sendall(struct.pack('!I', length))
    sock.sendall(data)

def recv_one_message(sock):
    lengthbuf = recvall(sock, 4)
    length, = struct.unpack('!I', lengthbuf)
    return recvall(sock, length)

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def dispose_msg(msg, msg_queue):
#     print('recv:', msg)

    global board
    global s1
    global first_query
    global who_first

    ans = None
    seq = msg.split(' ')
    if seq[0] == 'START:':
        board_size = int(seq[1])
        Board.set_board_size(board_size)
        board = Board()
        if s1 is None:
            s1 = StrategyDNN()
        first_query = True
        who_first = None
        ans = 'START: OK'
        if msg_queue is not None:
            msg_queue.put(('start',))
        s1.absorb('?')
        s1.on_episode_start()
    elif seq[0] == 'MOVE:':
        assert len(seq) >= 4, 'protocol inconsistent'
        old_board = copy.deepcopy(board)
        x, y = int(seq[1]), int(seq[2])
        who = Board.STONE_BLACK if int(seq[3]) == 1 else Board.STONE_WHITE
        if who_first is None:
            who_first = who
            print('who first?', who_first)
        if board.is_legal(x, y):
            board.move(x, y, who)

        s1.swallow(who, old_board, board)
        if msg_queue is not None:
            msg_queue.put(('move', who, x * Board.BOARD_SIZE + y))
    elif seq[0] == 'WIN:':
        assert len(seq) == 3, 'protocol inconsistent'
        x, y = int(seq[1]), int(seq[2])
        who = board.get(x, y)
        print('player %d win the game' % (who,))
    elif seq[0] == 'UNDO:':
        ans = 'UNDO: unsupported yet'
    elif seq[0] == 'WHERE:':
        if who_first is None:
            who_first = Board.STONE_BLACK
            print('who first?', who_first)
        if first_query:
            s1.stand_for = board.query_stand_for(who_first)
            print('i stand for:', s1.stand_for)
            first_query = False
        assert s1.stand_for is not None
        x, y = s1.preferred_move(board)
        ans = 'HERE: %d %d' % (x, y)
    elif seq[0] == 'END:':
#             s1.close()
        ans = 'END: OK'

    return ans

class ClientThread(Thread):
    def __init__(self, conn, msg_queue):
        Thread.__init__(self)
        self.conn = conn
        self.msg_queue = msg_queue

    def run(self):
        try:
            msg = 'TOKEN: %d' % (random.randint(1, 1 << 30),)
            send_one_message(self.conn, msg.encode('ascii'))

            while True:
                msg = recv_one_message(self.conn)
                if msg is not None:
                    msg = msg.decode('ascii')
                    ans = dispose_msg(msg, self.msg_queue)
                    if ans is not None:
                        send_one_message(self.conn, ans.encode('ascii'))
        except ConnectionResetError:
            self.conn.close()
        finally:
            self.conn.close()


def net(msg_queue=None):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('Socket created')

    # Bind socket to local host and port
    try:
        s.bind((HOST, PORT))
    except socket.error as msg:
        print('Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1])
        sys.exit()

    print('Socket bind complete')

    # Start listening on socket
    s.listen(5)
    print('Socket now listening')

    # now keep talking with the client
    while True:
        # wait to accept a connection - blocking call
        conn, addr = s.accept()
        print('Connected with ' + addr[0] + ':' + str(addr[1]))
        thread = ClientThread(conn, msg_queue)
        thread.start()

    s.close()


s1 = None
if __name__ == '__main__':
    net()
