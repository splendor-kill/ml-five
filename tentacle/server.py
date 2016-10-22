from threading import Thread
from queue import Queue
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
                    self.msg_queue.put(msg)

                    self.msg_queue.join()
                    ans = self.msg_queue.get()
                    send_one_message(self.conn, ans.encode('ascii'))
                    self.msg_queue.task_done()

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


if __name__ == '__main__':
    net()
