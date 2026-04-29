"""命令行入口：监督学习、强化学习、图形界面等。"""

from __future__ import annotations

import argparse
import sys


def main_gui_only() -> None:
    """兼容旧入口 ``ml-five-main``：等价于 ``ml-five gui ...``。"""
    sys.argv = [sys.argv[0], "gui", *sys.argv[1:]]
    main()


def _cmd_supervised(args: argparse.Namespace) -> None:
    resume = args.resume
    from_file = args.from_file
    part_vars = not args.no_part_vars

    if args.network == "pre":
        from tentacle.dnn import Pre

        model = Pre(is_train=True, is_revive=resume, is_rl=False)
    else:
        from tentacle.dnn3 import DCNN3

        model = DCNN3(is_train=True, is_revive=resume, is_rl=False)

    model.run(
        from_file=from_file,
        part_vars=part_vars,
        arena_games_per_side=args.arena_games_per_side,
    )


def _cmd_reinforce(args: argparse.Namespace) -> None:
    from tentacle.main import run_reinforce

    run_reinforce(
        resume=not args.no_resume,
        opponent=args.opponent,
        iterations=args.iterations,
        episodes_per_iter=args.episodes_per_iter,
        eval_games_per_side=args.eval_games_per_side,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        minmax_curriculum=not args.no_curriculum,
        minmax_curriculum_iters=args.curriculum_iters,
    )


def _cmd_eval_supervised(args: argparse.Namespace) -> None:
    from tentacle.dnn import Pre

    model = Pre(is_train=False, is_revive=True, is_rl=False)
    model._ensure_net()
    model.load_from_vat(from_file=args.from_file, part_vars=not args.no_part_vars)
    metrics = model.evaluate_fixed_splits(
        train_file=args.train_file,
        valid_file=args.valid_file,
        test_file=args.test_file,
    )
    for split in ("train", "valid", "test"):
        m = metrics[split]
        print(
            "%s: top1=%.4f top3=%.4f top5=%.4f"
            % (split, m["top1"], m["top3"], m["top5"])
        )


def _cmd_gui(args: argparse.Namespace) -> None:
    from tentacle.main import launch_gui

    if args.ckpt and args.strategy != "dnn":
        raise ValueError("--ckpt 仅当 --strategy 为 dnn 时可用")
    launch_gui(opponent_strategy=args.strategy, opponent_checkpoint=args.ckpt)


def _cmd_replay_dataset(args: argparse.Namespace) -> None:
    from tentacle.dataset_replay import launch_dataset_replay_gui

    max_rows = None if args.max_rows == 0 else args.max_rows
    launch_dataset_replay_gui(
        args.file,
        step_sec=args.step_sec,
        max_rows=max_rows,
        chain=args.chain,
    )


def _cmd_match(args: argparse.Namespace) -> None:
    from tentacle.main import run_model_match

    if args.black_ckpt and args.black != "dnn":
        raise ValueError("--black-ckpt 仅当 --black 为 dnn 时可用")
    if args.white_ckpt and args.white != "dnn":
        raise ValueError("--white-ckpt 仅当 --white 为 dnn 时可用")

    stats = run_model_match(
        args.black,
        args.white,
        args.games_per_side,
        black_ckpt=args.black_ckpt,
        white_ckpt=args.white_ckpt,
        random_start=not args.empty_start,
    )
    total = stats["total"]
    print(
        "%s vs %s：共 %d 局（每名称各执黑 %d 局）；%s 胜 %d，%s 胜 %d，和 %d"
        % (
            stats["black_strategy"],
            stats["white_strategy"],
            total,
            args.games_per_side,
            stats["black_strategy"],
            stats["wins_black"],
            stats["white_strategy"],
            stats["wins_white"],
            stats["draws"],
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ml-five",
        description="五子棋 Alphago 风格训练与对弈入口。",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", help="子命令")

    gui_p = sub.add_parser("gui", help="matplotlib 人机对弈（human 对一个命令行指定的程序策略）")
    gui_p.add_argument(
        "-s",
        "--strategy",
        choices=["rand", "minmax", "td", "dnn"],
        required=True,
        help="程序对手策略；启动后 human 默认执黑直接开局；F2 执黑重开，F3 执白重开",
    )
    gui_p.add_argument(
        "--ckpt",
        default=None,
        metavar="PATH",
        help="DNN checkpoint 目录或具体 .pt 文件（仅 --strategy dnn 时有效；省略则优先 rl_brain/ 再 zero/）",
    )
    gui_p.set_defaults(func=_cmd_gui)

    replay_p = sub.add_parser(
        "replay-dataset",
        help="matplotlib 回放监督 CSV：每步展示局面与标注落子，可设间隔（默认打乱数据为逐条样本；--chain 合并相邻可衔接步）",
    )
    replay_p.add_argument(
        "file",
        metavar="PATH",
        help="数据集 CSV 路径（如 data/alphagomoku/dataset_gomocup15/train.txt）",
    )
    replay_p.add_argument(
        "--step-sec",
        type=float,
        default=1.0,
        metavar="SEC",
        help="每步总时长（秒）；约前 45%% 仅局面，后 55%% 标出落点，默认 1.0",
    )
    replay_p.add_argument(
        "--max-rows",
        type=int,
        default=50_000,
        metavar="N",
        help="最多加载行数，避免超大文件占满内存；0 表示不限制",
    )
    replay_p.add_argument(
        "--chain",
        action="store_true",
        help="若相邻行「下一行局面 == 本行落子后」，则合并为同一局顺序回放",
    )
    replay_p.set_defaults(func=_cmd_replay_dataset)

    match_p = sub.add_parser(
        "match",
        help="双方程序对弈统计（无 GUI；DNN 可用 --black-ckpt / --white-ckpt 指定目录或 .pt）",
    )
    match_p.add_argument(
        "-b",
        "--black",
        choices=["rand", "minmax", "td", "dnn"],
        required=True,
        metavar="STRAT",
        help="执「黑方名」一侧的策略（统计中的黑方指该名称，非棋盘颜色）",
    )
    match_p.add_argument(
        "-w",
        "--white",
        choices=["rand", "minmax", "td", "dnn"],
        required=True,
        metavar="STRAT",
        help="执「白方名」一侧的策略",
    )
    match_p.add_argument(
        "--games-per-side",
        type=int,
        default=2,
        metavar="N",
        help="每方各执黑 N 局（总局数 2N；默认 2，与 evaluate_vs_opponents 一致）",
    )
    match_p.add_argument(
        "--black-ckpt",
        default=None,
        metavar="PATH",
        help="黑方 DNN 的 checkpoint 目录或前缀（仅 --black dnn 时有效；省略则与 GUI 相同规则）",
    )
    match_p.add_argument(
        "--white-ckpt",
        default=None,
        metavar="PATH",
        help="白方 DNN 的 checkpoint（仅 --white dnn 时有效）",
    )
    match_p.add_argument(
        "--empty-start",
        action="store_true",
        help="每局从空盘开始（默认随机合法前缀局面）",
    )
    match_p.set_defaults(func=_cmd_match)

    sl_p = sub.add_parser(
        "supervised",
        help="监督学习：用棋谱训练策略网络（原 dnn*.py / dnn.Pre）",
    )
    sl_p.add_argument(
        "--network",
        choices=["dnn3", "pre"],
        default="dnn3",
        help="网络变体：dnn3 为默认 DCNN3；pre 为 dnn.Pre",
    )
    sl_p.add_argument(
        "--resume",
        action="store_true",
        help="从 checkpoint 恢复（is_revive=True，会尝试加载已有权重）",
    )
    sl_p.add_argument(
        "--from-file",
        default=None,
        metavar="PATH",
        help="checkpoint 搜索目录或路径前缀（传给 Pre.run）",
    )
    sl_p.add_argument(
        "--no-part-vars",
        action="store_true",
        help="加载权重时不做部分变量恢复（part_vars=False）",
    )
    sl_p.add_argument(
        "--arena-games-per-side",
        type=int,
        default=2,
        metavar="N",
        help="每次记录 vs_rand/vs_minmax 时，每个先后手各对弈 N 局（总计每对手 2N 局）",
    )
    sl_p.set_defaults(func=_cmd_supervised)

    rl_p = sub.add_parser(
        "reinforce",
        help="强化学习：self-play 风格 reinforce 循环（仅 CLI，无 GUI）",
    )
    rl_p.add_argument(
        "--no-resume",
        action="store_true",
        help="忽略 rl_brain 中已有 checkpoint，从监督学习权重起步",
    )
    rl_p.add_argument(
        "--opponent",
        choices=["selfplay", "minmax"],
        default="selfplay",
        help="强化学习对手：selfplay 为 DNN 对手池，minmax 为固定 MinMax 对手",
    )
    rl_p.add_argument(
        "--iterations",
        type=int,
        default=100,
        metavar="N",
        help="强化学习外层迭代次数",
    )
    rl_p.add_argument(
        "--episodes-per-iter",
        type=int,
        default=None,
        metavar="N",
        help="每个迭代采样局数；默认使用 config.REINFORCE_PERIOD",
    )
    rl_p.add_argument(
        "--eval-games-per-side",
        type=int,
        default=2,
        metavar="N",
        help="评估时 DNN 先后手各对 MinMax 下 N 局；0 表示不评估",
    )
    rl_p.add_argument(
        "--eval-interval",
        type=int,
        default=5,
        metavar="N",
        help="每 N 个迭代做一次无探索 MinMax 评估；0 表示不评估",
    )
    rl_p.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        metavar="N",
        help="MinMax 强化学习每 N 个迭代保存一次 checkpoint；0 表示只在结束保存",
    )
    rl_p.add_argument(
        "--no-curriculum",
        action="store_true",
        help="关闭 MinMax 课程训练，采样阶段始终使用固定 MinMax",
    )
    rl_p.add_argument(
        "--curriculum-iters",
        type=int,
        default=100,
        metavar="N",
        help="MinMax 课程训练从混合随机对手过渡到纯 MinMax 的迭代数",
    )
    rl_p.set_defaults(func=_cmd_reinforce)

    eval_p = sub.add_parser(
        "eval-supervised",
        help="固定数据文件评估监督学习模型（top1/top3/top5）",
    )
    eval_p.add_argument(
        "--from-file",
        default=None,
        metavar="PATH",
        help="checkpoint 搜索目录或路径前缀（默认使用 zero 目录）",
    )
    eval_p.add_argument(
        "--no-part-vars",
        action="store_true",
        help="加载权重时不做部分变量恢复（part_vars=False）",
    )
    eval_p.add_argument(
        "--train-file",
        default=None,
        metavar="PATH",
        help="训练集文件路径（默认 config.py 的 DATA_SET_TRAIN）",
    )
    eval_p.add_argument(
        "--valid-file",
        default=None,
        metavar="PATH",
        help="验证集文件路径（默认 config.py 的 DATA_SET_VALID）",
    )
    eval_p.add_argument(
        "--test-file",
        default=None,
        metavar="PATH",
        help="测试集文件路径（默认 config.py 的 DATA_SET_TEST）",
    )
    eval_p.set_defaults(func=_cmd_eval_supervised)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        parser.exit(status=2)

    args.func(args)


if __name__ == "__main__":
    main()
