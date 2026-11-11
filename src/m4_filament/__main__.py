"""Command line for training, refining and evaluating M4 from frozen E5 artifacts."""
import argparse
from pathlib import Path

from .data import evaluate, prepare
from .pipeline import components, refine, train
from .recipes import RULES, SEED, TRIALS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    for name, action in (("prepare", prepare), ("refine", refine)):
        stage = commands.add_parser(name)
        for flag in ("images", "m1", "m2", "coarse", "output"):
            stage.add_argument("--" + flag, type=Path, required=True)
        if name == "prepare":
            stage.add_argument("--annotations", type=Path, required=True)
        else:
            stage.add_argument("--checkpoint", type=Path, required=True)
            stage.add_argument("--device", default="cpu")
            stage.add_argument("--rule", choices=RULES)
        stage.set_defaults(action=action)

    stage = commands.add_parser("train")
    stage.add_argument("--dataset", type=Path, required=True)
    stage.add_argument("--recipe", choices=TRIALS, required=True)
    stage.add_argument("--output", type=Path, required=True)
    stage.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    stage.add_argument("--seed", type=int, default=SEED)
    stage.add_argument("--stop-after", type=int, choices=(2,4), default=4)
    stage.add_argument("--exclude", type=Path, help="JSON list of development observations to exclude by date group")
    stage.add_argument("--threshold", type=float, default=.5)
    stage.set_defaults(action=train)

    stage = commands.add_parser("components")
    stage.add_argument("--model", choices=("m1","m2"), required=True)
    for flag in ("images", "probabilities", "output"):
        stage.add_argument("--" + flag, type=Path, required=True)
    stage.set_defaults(action=components)

    stage = commands.add_parser("score")
    stage.add_argument("--prediction", action="append", metavar="NAME=JSONL", required=True)
    stage.add_argument("--annotations", type=Path, required=True)
    stage.add_argument("--observations", type=Path, help="JSON list of development observation IDs. Defaults to the frozen holdout.")
    stage.add_argument("--output", type=Path, required=True)
    stage.set_defaults(action=evaluate)
    args = parser.parse_args()
    if args.command == "train" and not 0 < args.threshold < 1:
        parser.error("--threshold must be strictly between zero and one")
    args.action(args)


if __name__ == "__main__":
    main()
