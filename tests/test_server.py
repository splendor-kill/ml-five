"""server 模块单元测试。"""

import socket
import struct
import threading

import pytest

from tentacle import server


class FakeStrategy:
    def __init__(self):
        self.stand_for = None
        self.started = False

    def absorb(self, _token):
        return None

    def on_episode_start(self):
        self.started = True

    def swallow(self, _who, _old_board, _new_board):
        return None

    def preferred_move(self, _board):
        return (0, 0)


@pytest.fixture(autouse=True)
def _patch_strategy_factory(monkeypatch):
    monkeypatch.setattr(server, 'create_strategy', lambda: FakeStrategy())
    yield


def _recv_framed(sock):
    header = sock.recv(4)
    assert len(header) == 4
    length, = struct.unpack('!I', header)
    payload = b''
    while len(payload) < length:
        payload += sock.recv(length - len(payload))
    return payload


def test_recv_one_message_returns_none_after_peer_close():
    srv, cli = socket.socketpair()
    try:
        cli.close()
        assert server.recv_one_message(srv) is None
    finally:
        srv.close()


def test_recv_one_message_raises_for_oversized_payload():
    srv, cli = socket.socketpair()
    try:
        cli.sendall(struct.pack('!I', server.MAX_MESSAGE_SIZE + 1))
        with pytest.raises(ValueError, match='message too large'):
            server.recv_one_message(srv)
    finally:
        srv.close()
        cli.close()


def test_dispose_msg_move_before_start_returns_protocol_error():
    state = server.SessionState()
    ans = server.dispose_msg('MOVE: 0 0 1', None, state)
    assert ans == 'ERROR: game not started'


def test_dispose_msg_start_then_where_returns_here():
    state = server.SessionState()
    ans1 = server.dispose_msg(f'START: {server.Board.BOARD_SIZE}', None, state)
    ans2 = server.dispose_msg('WHERE:', None, state)

    assert ans1 == 'START: OK'
    assert ans2 == 'HERE: 0 0'
    assert state.s1.started


def test_dispose_msg_states_are_isolated_between_clients():
    state1 = server.SessionState()
    state2 = server.SessionState()

    ans1 = server.dispose_msg(f'START: {server.Board.BOARD_SIZE}', None, state1)
    ans2 = server.dispose_msg('MOVE: 0 0 1', None, state1)
    ans3 = server.dispose_msg('MOVE: 0 0 1', None, state2)

    assert ans1 == 'START: OK'
    assert ans2 is None
    assert ans3 == 'ERROR: game not started'


def test_client_thread_exits_after_peer_disconnect():
    srv, cli = socket.socketpair()
    t = None
    try:
        t = threading.Thread(target=server.ClientThread(srv, None).run)
        t.start()

        token = _recv_framed(cli).decode('ascii')
        assert token.startswith('TOKEN: ')

        cli.close()
        t.join(timeout=1.0)
        assert not t.is_alive()
    finally:
        if t is not None and t.is_alive():
            t.join(timeout=1.0)
