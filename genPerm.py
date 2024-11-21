from itertools import permutations
import numpy as np

def generate_swapped_sequences(original_sequence):
    # Step 1: Identify the distinct numbers in the original sequence
    distinct_elements = list(set(original_sequence))
    
    # Step 2: Generate all permutations of the distinct elements
    distinct_permutations = list(permutations(distinct_elements))
    
    # Step 3: For each permutation, replace elements in the original list
    swapped_sequences = []
    for perm in distinct_permutations:
        mapping = dict(zip(distinct_elements, perm))
        swapped_sequence = [mapping[element] for element in original_sequence]
        swapped_sequences.append(tuple(swapped_sequence))
    
    return swapped_sequences

def findRightPerm(predicted_labels, labels):
    best_accuracy = 0.0
    best_permutation = None

    permutations = generate_swapped_sequences(predicted_labels)

    for perm in permutations:
        correct_predictions = np.sum(perm == labels)
        if correct_predictions > best_accuracy:
            best_accuracy = correct_predictions
            best_permutation = perm

    return np.array(best_permutation), best_accuracy

if __name__ == "__main__":
    # Example usage
    original_sequence = (0, 2, 0, 1, 1, 0)
    result = generate_swapped_sequences(original_sequence)

    # Print all the swapped sequences
    for seq in result:
        print(seq)