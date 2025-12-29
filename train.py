"""
train.py
--------
Training Loop for TINO-Rocket.
Loads synthetic real-gas data and trains the FNO model.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from model import TINO_FNO1d
import time

# --- CONFIG ---
MODES = 16
WIDTH = 64
BATCH_SIZE = 32
EPOCHS = 500  # We want a very good fit
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"--- TINO TRAINING START ---")
print(f"Device: {DEVICE}")

# 1. Load Data
print("Loading Data...")
data_dir = 'data'
# Inputs
train_q = np.load(os.path.join(data_dir, 'train_q.npy'))
train_cond = np.load(os.path.join(data_dir, 'train_cond.npy'))
# Targets
train_T = np.load(os.path.join(data_dir, 'train_T.npy'))

test_q = np.load(os.path.join(data_dir, 'test_q.npy'))
test_cond = np.load(os.path.join(data_dir, 'test_cond.npy'))
test_T = np.load(os.path.join(data_dir, 'test_T.npy'))

# 2. Data Preprocessing (Normalization is Key)
# Normalize inputs to [0,1] or [-1,1] ranges roughly
q_norm_scale = 1e7
T_norm_scale = 1000.0 # Normalize Temp by dividing by 1000K

X_train_list = []
Y_train_list = []

print("Preprocessing inputs...")
# We need to stack inputs into shape [Batch, Grid, Channels]
# Channels: [x, q, P, T_in, mdot]
grid_size = train_q.shape[1]
x_grid = np.linspace(0, 1, grid_size)

def prepare_tensors(q_arr, cond_arr, T_arr):
    X_list = []
    Y_list = []
    for i in range(len(q_arr)):
        # Create grid channels
        x_channel = x_grid
        q_channel = q_arr[i] / q_norm_scale
        
        # Create constant channels for scalars
        P_val = cond_arr[i][0] / 20.0 # Norm by max pressure
        Tin_val = cond_arr[i][1] / 600.0
        mdot_val = cond_arr[i][2] / 0.1
        
        P_channel = np.full(grid_size, P_val)
        Tin_channel = np.full(grid_size, Tin_val)
        mdot_channel = np.full(grid_size, mdot_val)
        
        # Stack: [Grid, 5]
        sample_input = np.stack([x_channel, q_channel, P_channel, Tin_channel, mdot_channel], axis=1)
        X_list.append(sample_input)
        
        # Target
        Y_list.append(T_arr[i] / T_norm_scale)
        
    return torch.tensor(np.array(X_list), dtype=torch.float32), torch.tensor(np.array(Y_list), dtype=torch.float32)

train_X, train_Y = prepare_tensors(train_q, train_cond, train_T)
test_X, test_Y = prepare_tensors(test_q, test_cond, test_T)

train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(test_X, test_Y), batch_size=BATCH_SIZE, shuffle=False)

# 3. Model Setup
model = TINO_FNO1d(modes=MODES, width=WIDTH).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
criterion = nn.MSELoss()

# 4. Training Loop
start_time = time.time()
loss_history = []

for epoch in range(EPOCHS):
    model.train()
    train_mse = 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        optimizer.zero_grad()
        out = model(x)
        
        # Standard MSE Loss (We will upgrade this to Physics-Loss later)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_mse += loss.item()
    
    scheduler.step()
    train_mse /= len(train_loader)
    loss_history.append(train_mse)
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss = {train_mse:.8f}")

# 5. Save & Test
torch.save(model.state_dict(), 'tino_model.pth')
print(f"Training Complete in {time.time()-start_time:.1f}s. Model saved.")

# Visualization
model.eval()
with torch.no_grad():
    # Pick a random test sample
    idx = 10
    test_sample = test_X[idx].unsqueeze(0).to(DEVICE)
    prediction = model(test_sample).cpu().numpy()[0] * T_norm_scale
    ground_truth = test_Y[idx].cpu().numpy() * T_norm_scale
    
    plt.figure(figsize=(10,6))
    plt.plot(ground_truth, label='CFD (Ground Truth)', color='black', linewidth=2)
    plt.plot(prediction, label='TINO (AI Prediction)', color='red', linestyle='--')
    plt.title(f"TINO Prediction vs Real-Gas CFD (Test Sample {idx})")
    plt.legend()
    plt.grid(True)
    plt.savefig('tino_result.png')
    plt.show()