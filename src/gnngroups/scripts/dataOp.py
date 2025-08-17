import argparse

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
        from gnngroups.dataset import makeEpisode
        makeEpisode()
        exit()
    
    if (args.bulk):
        from gnngroups.dataset import genBulkDataset
        genBulkDataset()
        exit()

    if (args.sample):
        from gnngroups.dataset import sampleDataset
        sampleDataset()
        exit()






if __name__ == "__main__":
    main()