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
    elif args.network == "dnn1":
        from tentacle.dnn1 import DCNN1

        model = DCNN1(is_train=True, is_revive=resume, is_rl=False)
    elif args.network == "dnn2":
        from tentacle.dnn2 import DCNN2

        model = DCNN2(is_train=True, is_revive=resume, is_rl=False)
    else:
        from tentacle.dnn3 import DCNN3

        model = DCNN3(is_train=True, is_revive=resume, is_rl=False)

    model.run(from_file=from_file, part_vars=part_vars)


def _cmd_reinforce(args: argparse.Namespace) -> None:
    from tentacle.main import run_reinforce

    run_reinforce(resume=not args.no_resume)


def _cmd_gui(args: argparse.Namespace) -> None:
    from tentacle.main import launch_gui

    launch_gui(black_strategy=args.black_strategy, white_strategy=args.white_strategy)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ml-five",
        description="五子棋 Alphago 风格训练与对弈入口。",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", help="子命令")

    gui_p = sub.add_parser("gui", help="打开 matplotlib 图形界面（原 ml-five-main）")
    gui_p.add_argument(
        "-b",
        "--black-strategy",
        choices=["rand", "minmax", "td", "dnn"],
        default=None,
        help="黑方策略",
    )
    gui_p.add_argument(
        "-w",
        "--white-strategy",
        choices=["rand", "minmax", "td", "dnn"],
        default=None,
        help="白方策略",
    )
    gui_p.set_defaults(func=_cmd_gui)

    sl_p = sub.add_parser(
        "supervised",
        help="监督学习：用棋谱训练策略网络（原 dnn*.py / dnn.Pre）",
    )
    sl_p.add_argument(
        "--network",
        choices=["dnn1", "dnn2", "dnn3", "pre"],
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
    sl_p.set_defaults(func=_cmd_supervised)

    rl_p = sub.add_parser(
        "reinforce",
        help="强化学习：self-play 风格 reinforce 循环（原 GUI F4）",
    )
    rl_p.add_argument(
        "--no-resume",
        action="store_true",
        help="忽略 rl_brain 中已有 checkpoint，从监督学习权重起步",
    )
    rl_p.set_defaults(func=_cmd_reinforce)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        parser.exit(status=2)

    args.func(args)


if __name__ == "__main__":
    main()
