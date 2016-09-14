from _thread import start_new_thread
import random
import socket
import struct
import sys

from tentacle.board import Board


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
    seq = msg.split(' ')
    if seq[0] == 'START:':
#          clear board
        board_size = int(seq[1])
        board = Board(board_size)
        ans = 'START: OK'
        send_one_message(conn, ans.encode('ascii'))
    elif seq[0] == 'MOVE:':
        assert len(seq) == 4, 'protocol inconsistent'
        x, y = int(seq[1]), int(seq[2])
        who = Board.STONE_BLACK if int(seq[3]) == 1 else Board.STONE_WHITE
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
        x = random.randint(0, Board.BOARD_SIZE - 1)
        y = random.randint(0, Board.BOARD_SIZE - 1)
        board.move(x, y, Board.STONE_BLACK)
        ans = 'HERE: %d %d' % (x, y)
        send_one_message(conn, ans.encode('ascii'))
    elif seq[0] == 'END:':
        ans = 'END: unsupported yet'
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


class Server(object):
    def __init__(self):
        pass


if __name__ == '__main__':
    net()
