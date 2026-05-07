"""强化学习主路径单元测试。"""

import numpy as np
import pytest

from tentacle.board import Board
from tentacle.dnn import Pre
from tentacle.main import _rl_win_ratio, run_reinforce
from tentacle.rl_policy import Brain, NUM_ACTIONS, Transformer


def _one_move_game():
    st0 = Board()
    st1 = Board()
    st1.stones = st0.stones.copy()
    st1.stones[0] = Board.STONE_BLACK
    return st0, st1


def test_rl_win_ratio_treats_zero_losses_as_decisive():
    assert _rl_win_ratio(0, 0) == 1.0
    assert _rl_win_ratio(1, 0) == float("inf")
    assert _rl_win_ratio(3, 2) == 1.5


def test_run_reinforce_rejects_invalid_iterations_before_setup():
    with pytest.raises(ValueError, match="iterations must be positive"):
        run_reinforce(iterations=0)


def test_absorb_draw_records_replay_with_zero_terminal_reward():
    brain = Pre(is_train=False, is_revive=False, is_rl=True)
    st0, st1 = _one_move_game()
    brain.observation = [(Board.STONE_BLACK, st0, st1, False)]

    trained = brain.absorb(Board.STONE_EMPTY, stand_for=Board.STONE_BLACK)

    assert not trained
    assert len(brain.replay_memory_games.indexes) == 1
    game = brain.replay_memory_games.data[0]
    assert game["states"].shape[0] == 1
    assert game["actions"].shape == (1, Pre.NUM_ACTIONS)
    np.testing.assert_allclose(game["rewards"], np.array([0.0], dtype=np.float32))
    np.testing.assert_allclose(game["policy_weights"], np.array([1.0], dtype=np.float32))


def test_rl_train_rejects_illegal_policy_action(monkeypatch):
    monkeypatch.setattr(Pre, "REPLAY_MEMORY_CAPACITY", 1)
    brain = Pre(is_train=False, is_revive=False, is_rl=True)
    _, st1 = _one_move_game()
    state, _ = brain.adapt_state(st1.stones)
    illegal_action = np.zeros(Pre.NUM_ACTIONS, dtype=np.float32)
    illegal_action[0] = 1.0
    game = brain._pack_rl_game([state], [illegal_action], [0.0], [1.0])
    brain.replay_memory_games.append(game)

    with pytest.raises(ValueError, match="illegal policy action"):
        brain.rl_train(policy_games=[game], opt_policy_only=False)


def test_legacy_rl_policy_rejects_illegal_policy_action(tmp_path):
    transformer = Transformer()
    brain = Brain(transformer.get_input_shape, str(tmp_path), str(tmp_path))
    _, st1 = _one_move_game()
    state, _ = transformer.adapt_state(st1.stones)
    illegal_action = np.zeros(NUM_ACTIONS, dtype=np.float32)
    illegal_action[0] = 1.0

    with pytest.raises(ValueError, match="illegal policy action"):
        brain.reinforce(
            np.array([state], dtype=np.float32),
            np.array([illegal_action], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
        )
