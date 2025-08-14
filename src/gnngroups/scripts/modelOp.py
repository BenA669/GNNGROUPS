import argparse
from gnngroups.model import *

def parse_args():
    parser = argparse.ArgumentParser(
        description="Dataset Operations"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-t", "--train", action="store_true", help="Train and validate model")
    group.add_argument("-e", "--eval", action="store_true", help="Evaluate model")

    args = parser.parse_args()

    if not (args.train or args.eval):
        parser.error("At least one of --train or --eval is required.")

    return args

def main():
    args = parse_args()

    if (args.train):
        train()
        exit()
    
    if (args.eval):
        evaluate()
        exit()


if __name__ == "__main__":
    main()