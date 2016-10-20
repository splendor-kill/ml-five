from _thread import start_new_thread
import random
import socket
import struct
import sys

from tentacle.board import Board
from tentacle.strategy_dnn import StrategyDNN


HOST = ''  # Symbolic name, meaning all available interfaces
PORT = 10000  # Arbitrary non-privileged port


def send_one_message(sock, data):
    length = len(data)
    print('send:', data)
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

def dispose_msg(conn, msg):
    print('recv:', msg)
    global board
    global s1
    global first_query
    global who_first

    seq = msg.split(' ')
    if seq[0] == 'START:':
#          clear board
        board_size = int(seq[1])
        board = Board(board_size)
        s1 = StrategyDNN(is_train=True)
        first_query = True
        who_first = None
        ans = 'START: OK'
        send_one_message(conn, ans.encode('ascii'))
    elif seq[0] == 'MOVE:':
        assert len(seq) == 4, 'protocol inconsistent'
        x, y = int(seq[1]), int(seq[2])
        who = Board.STONE_BLACK if int(seq[3]) == 1 else Board.STONE_WHITE
        if who_first is None:
            who_first = who
            print('first:', who_first)
        if board.is_legal(x, y):
            board.move(x, y, who)
    elif seq[0] == 'WIN:':
        assert len(seq) == 3, 'protocol inconsistent'
        x, y = int(seq[1]), int(seq[2])
        who = board.get(x, y)
        print('player %d win the game' % (who,))
    elif seq[0] == 'UNDO:':
        ans = 'UNDO: unsupported yet'
        send_one_message(conn, ans.encode('ascii'))
    elif seq[0] == 'WHERE:':
        if who_first is None:
            who_first = Board.STONE_BLACK
            print('first:', who_first)
        if first_query:
            s1.stand_for = board.query_stand_for(who_first)
            print('me:', s1.stand_for)
            first_query = False
        assert s1.stand_for is not None
        x, y = s1.preferred_move(board)
        board.move(x, y, s1.stand_for)
        ans = 'HERE: %d %d' % (x, y)
        send_one_message(conn, ans.encode('ascii'))
    elif seq[0] == 'END:':
        s1.close()
        ans = 'END: OK'
        send_one_message(conn, ans.encode('ascii'))

def clientthread(conn):
    try:
        # Sending message to connected client
        msg = 'TOKEN: %d' % (random.randint(1, 1 << 30),)
        send_one_message(conn, msg.encode('ascii'))

        # infinite loop so that function do not terminate and thread do not end.
        while True:
            msg = recv_one_message(conn)
            if msg is not None:
                msg = msg.decode('ascii')
                dispose_msg(conn, msg)

    finally:
        # came out of loop
        conn.close()


def net():
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
        start_new_thread(clientthread, (conn,))

    s.close()


if __name__ == '__main__':
    net()
