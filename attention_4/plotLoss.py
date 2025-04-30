import re
import matplotlib.pyplot as plt

def parse_log(file_path):
    epochs = []
    train_losses = []
    val_losses = []
    # Updated to handle files with non-utf-8 characters
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = re.search(r"Epoch \[(\d+)/\d+\]\s+Train Loss: ([\d\.]+)\s+Val Loss: ([\d\.]+)", line)
            if m:
                epochs.append(int(m.group(1)))
                train_losses.append(float(m.group(2)))
                val_losses.append(float(m.group(3)))
    return epochs, train_losses, val_losses

# Paths to your log files
logs = {
    'Attention Model': 'C:/Users/Benja/Documents/Projects/GNN/GNNGROUPS/attention_4/attentionvalLoss.txt',
    'Dynamic Graph Model': 'C:/Users/Benja/Documents/Projects/GNN/GNNGROUPS/attention_4/dynamicgraphValLoss.txt',
    'GCN-LSTM Model': 'C:/Users/Benja/Documents/Projects/GNN/GNNGROUPS/attention_4/GCNLSTMLoss.txt',
    'GCN Only': 'C:/Users/Benja/Documents/Projects/GNN/GNNGROUPS/attention_4/gcnOnly.txt',
    'LSTM Only': 'C:/Users/Benja/Documents/Projects/GNN/GNNGROUPS/attention_4/lstmOnly.txt',
}

# Line style settings
line_styles = {
    'GCN-LSTM Model': '-',
    'Attention Model': '-',
}

plt.figure(figsize=(10, 6))
for name, path in logs.items():
    epochs, _, val_losses = parse_log(path)
    style = line_styles.get(name, '--')
    plt.plot(epochs, val_losses, linestyle=style, label=name)

plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Validation Loss Comparison')
plt.legend()
plt.grid(True, linestyle=':', linewidth=0.5)
plt.tight_layout()
plt.show()
