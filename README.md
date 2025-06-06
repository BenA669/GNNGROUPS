# GNNGROUPS

Framework for dynamic group detection and clustering using Graph Convolutional Networks (GCNs) on synthetic datasets with temporal and spatial structure.

## Usage

### 1. Configuration
Edit `config.ini` to set dataset, model, and training parameters. 

### Model parameters

- **input_dim**  
  Number of raw features per node (before any embedding).  
  - In the synthetic setup, each node has 2D coordinates (x, y), so `input_dim = 2`.

- **hidden_dim**  
  Dimension of hidden layers or intermediate GCN feature size.  
  - For example, the first `GCNConv` maps from `input_dim` → `hidden_dim` (e.g., 2 → 64).

- **hidden_dim_2**  
  A second bottleneck dimension (used after temporal/LSTM or attention layers).
  
- **output_dim**  
  Final per‑node embedding size after all GNN layers.  
  - This `output_dim` is what the InfoNCE contrastive loss uses as z‑vectors for each node.

- **num_heads**  
  Number of attention heads in multi‑head attention.

- **model_type**  
  The name of the class in `model.py` to instantiate.  
  - Supported values:
    - `LSTMOnly`
    - `GCNOnly`
    - `DynamicGraphNN`
    - `TemporalGCN`
    - `AttentionGCN`
  - Must match a class name in `model.py`.

---

### Dataset parameters

- **nodes**  
  Number of nodes in each synthetic episode.  
  - Each “episode” contains exactly `nodes` nodes moving around in 2D space.

- **timesteps**  
  Number of time steps per episode.  

- **groups**  
  Number of ground‑truth groups in each episode.  
  - Nodes are randomly assigned to one of `groups` groups at t=0, then they move together with Perlin noise + group‑specific tilt.

- **min_groups**  
  Minimum number of distinct groups required within an ego‑subgraph for selection as a pivot.  
  - When extracting an ego‑network in `getEgo`, randomly pick a node but require its neighborhood (within `hops`) to contain at least `min_groups` different ground‑truth labels.

- **mixed**  
  Boolean (True / False) indicating whether node‑to‑group assignments can vary in size each episode.  
  - If `False`, we evenly partition `nodes` across exactly `groups`. If `True`, the code can sample a random number of groups (up to the maximum) and assign nodes non‑uniformly.

- **perlin_offset_amt**  
  Amount to shift the Perlin noise coordinate at each time step (e.g., 0.75).  
  - After computing Perlin gradients for group g at time t, increment that group’s noise offset by `(perlin_offset_amt, perlin_offset_amt)` so that noise evolves over time.

- **noise_scale**  
  Multiplier for the Perlin noise input coordinates (affects “frequency”).  
  - A smaller `noise_scale` (e.g., 0.05) means “smoother” noise (larger features), while a larger value yields more rapid oscillations.

- **noise_strength**  
  Scales how strongly each node follows the Perlin gradient (i.e., the “step size” due to noise).  
  - If `noise_strength = 2`, then each node’s next step is `2 × ∇f<sub>Perlin</sub>(x, y)` (plus tilt).

- **tilt_strength**  
  A small, constant vector bias (added to each node’s Perlin‑gradient step) that tends to push each entire group in a fixed direction.  
  - Encourages group “drift” rather than pure random noise.

- **std_dev**  
  Standard deviation for the initial position sampling around each group’s “seed” at t=0.  

- **boundary**  
  Half‑width of a square bounding region.  
  - If a group’s centroid crosses boundary along x or y, we flip that group’s tilt vector so nodes bounce back into the region.

- **distance_threshold**  
  Radius threshold (Euclidean) for connecting two nodes with an edge at any given time t.  
  - If `||(x_i,y_i) − (x_j,y_j)|| < distance_threshold`, we set `adjacency[i,j] = 1`.

- **hops**  
  Hop‑distance parameter for extracting an Ego network.  
  - In `getEgo()`, we form the “union” adjacency matrix across all timesteps, then take up to `hops` hops away from a random pivot node to define the ego‑mask.

- **dir_path**  
  Directory where generated datasets and saved models will be stored (e.g., `./Models_Datasets/attention_4/`).  
  - `configReader.py` uses `dir_path + dataset_name + '_train.pt' / '_val.pt'` to construct full file paths.

- **dataset_name**  
  Base name for dataset files (e.g., `attn_4`).  
  - Combined with `dir_path`, the scripts automatically set:
    - `train_path = dir_path + dataset_name + "_train.pt"`
    - `val_path = dir_path + dataset_name + "_val.pt"`

- **samples**  
  Number of episodes (samples) to generate per split.  
  - In `genEpisodeBatches.py`, we loop `samples` times, each time calling `makeDatasetDynamicPerlin()` and saving the resulting episodes to either the training list or validation list.

---

### Training parameters

- **batch_size**  
  Number of episodes per minibatch when training (e.g., 4).  
  - Each episode contains its own full time‑series of node positions and adjacencies.

- **epochs**  
  Number of full passes over the training dataset (e.g., 100).

- **temp**  
  Temperature parameter for the InfoNCE (contrastive) loss (e.g., 0.1).  

- **learning_rate**  
  Learning rate for the optimizer (Adam α in `Adam(model.parameters(), lr=learning_rate)`).

- **model_name_pt**  
  Filename under `dir_path` where the best‑performing model’s `state_dict` is saved (e.g., `AttentionGCN_BEST.pt`).

- **demo**  
  Boolean (True / False) to trigger demo mode in scripts:  
  - If `demo = True`, then after each evaluation or in `makeEpisode.py`/`evaluate.py`, the code will call `plot_faster(...)` to animate node movements, group labels, and predicted clusters. If `False`, it skips plotting.

### 2. Training
Train a model using:
```bash
python train.py
```
Model checkpoints will be saved as specified in `config.ini`.

### 3. Evaluation & Visualization
Evaluate a model using:
```bash
python evaluate.py
```
This will give the accuracy of the model on the validation datqaset.

### 4. Custom Dataset Generation & Visualization
You can generate new synthetic episodes or datasets by modifying and running `makeEpisode.py` or related scripts.

## File Structure
- `train.py` — Model training script
- `evaluate.py` — Evaluation scripts
- `model.py` — GNN model architectures
- `makeEpisode.py`, `genEpisodeBatches.py` — Dataset generation utilities
- `datasetEpisode.py` — Dataset and dataloader definitions
- `config.ini` — Main configuration file
- `requirements.txt` — Python dependencies
- `animate.py` — Additional plotting/animation utilities
