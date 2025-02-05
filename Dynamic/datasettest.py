import torch
from torch.utils.data import Dataset, DataLoader

# Define a simple dataset
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data  # Store data
    
    def __len__(self):
        return len(self.data)  # Return the number of samples
    
    def __getitem__(self, index):
        return self.data[index]  # Fetch a sample by index

# Create dataset
data = torch.arange(10)  # Sample data from 0 to 9
dataset = MyDataset(data)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=0)

# Iterate through DataLoader
for batch_idx, batch in enumerate(dataloader):
    print(batch_idx)
    print(batch)
