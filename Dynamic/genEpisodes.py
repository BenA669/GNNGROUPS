from makeEpisode import makeDatasetDynamic
import torch
from tqdm import tqdm

if __name__ == "__main__":
    # You can adjust these parameters as you like
    time_steps = 20
    group_amt = 4
    node_amt = 400
    distance_threshold = 2
    intra_prob = 0.05
    inter_prob = 0.001

    NUM_SAMPLES_TEST = 1000
    NUM_SAMPLES_VAL = 1000

    test_data = []
    val_data = []

    # Generate test data
    for _ in tqdm(range(NUM_SAMPLES_TEST)):
        positions, adjacency = makeDatasetDynamic(
            node_amt=node_amt,
            group_amt=group_amt,
            time_steps=time_steps,
            distance_threshold=distance_threshold,
            intra_prob=intra_prob,
            inter_prob=inter_prob
        )
        test_data.append((positions, adjacency))

    # Generate validation data
    for _ in tqdm(range(NUM_SAMPLES_VAL)):
        positions, adjacency = makeDatasetDynamic(
            node_amt=node_amt,
            group_amt=group_amt,
            time_steps=time_steps,
            distance_threshold=distance_threshold,
            intra_prob=intra_prob,
            inter_prob=inter_prob
        )
        val_data.append((positions, adjacency))

    # Save datasets to file
    torch.save(test_data, "test_data.pt")
    torch.save(val_data, "val_data.pt")

    print("Finished generating and saving test_data.pt and val_data.pt")