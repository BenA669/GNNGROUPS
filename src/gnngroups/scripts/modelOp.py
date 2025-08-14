import argparse
from gnngroups.dataset import *
from gnngroups.utils import *

def parse_args():
    parser = argparse.ArgumentParser(
        description="Dataset Operations"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-m", "--make", action="store_true", help="Make and display episode")
    group.add_argument("-b", "--bulk", action="store_true", help="Generate episodes in bulk")
    group.add_argument("-s", "--sample", action="store_true", help="Sample episode")

    args = parser.parse_args()

    if not (args.make or args.bulk or args.sample):
        parser.error("At least one of --make, --bulk, or --sample is required.")

    return args

def main():
    args = parse_args()

    if (args.make):
        anchor_ratio = dataset_cfg["anchor_node_ratio"]
        distance_threshold = dataset_cfg["distance_threshold"]

        positions_t_n_xy = genDataset()

        new_positions_t_n_xy, \
        Xhat_t_n_n, \
        A_t_n_n, \
        anchor_pos_t_n_xy, \
        anchor_indices_n = genAnchors(positions_t_n_xy, anchor_ratio, distance_threshold)

        animatev2(new_positions_t_n_xy, A_t_n_n, anchor_indices_n)

        exit()
    
    if (args.bulk):
        genBulkDataset()
        exit()

    if (args.sample):
        sampleDataset()
        exit()


if __name__ == "__main__":
    main()