import matplotlib.pyplot as plt

# Define the epochs
epochs = [1, 2, 3, 4, 5]

# PrePad losses
prepad_train = [0.5641, 0.5545, 0.5580, 0.5652, 0.5672]
prepad_val   = [0.5627, 0.5629, 0.5604, 0.5727, 0.5654]

# PostPad losses
postpad_train = [0.5370, 0.4683, 0.4519, 0.4541, 0.4467]
postpad_val   = [0.4824, 0.4722, 0.4658, 0.4613, 0.4597]

# PackedSequence losses
packed_train = [0.5661, 0.4994, 0.4884, 0.4776, 0.4673]
packed_val   = [0.5260, 0.4929, 0.4823, 0.4816, 0.4831]

# Create a new figure
plt.figure(figsize=(10, 6))

# Plot PrePad losses
plt.plot(epochs, prepad_train, marker='o', linestyle='-', color='red', label='PrePad Train')
plt.plot(epochs, prepad_val, marker='o', linestyle='--', label='PrePad Val', color='red')

# Plot PostPad losses
plt.plot(epochs, postpad_train, marker='s', linestyle='-', label='PostPad Train', color='green')
plt.plot(epochs, postpad_val, marker='s', linestyle='--', label='PostPad Val', color='green')

# Plot PackedSequence losses
plt.plot(epochs, packed_train, marker='^', linestyle='-', label='PackedSequence Train', color='blue')
plt.plot(epochs, packed_val, marker='^', linestyle='--', label='PackedSequence Val', color='blue')

# Label the axes and add a title
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss per Epoch for Different Padding Methods')

# Add a legend and grid for clarity
plt.legend()
plt.grid(True)

# Display the plot
plt.show()
