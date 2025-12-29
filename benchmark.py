"""
benchmark_batch.py
------------------
Performance Analysis: Batch Throughput (The Real-Time Control Standard).
Compares solving 100 Scenarios (Trajectory Optimization Task).
"""
import time
import numpy as np
import torch
from solver import ChannelSolver
from model import TINO_FNO1d

# --- SETUP ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Benchmarking on: {DEVICE}")

# Load AI
model = TINO_FNO1d(modes=16, width=64).to(DEVICE)
try:
    model.load_state_dict(torch.load('tino_model.pth', map_location=DEVICE))
except:
    model.load_state_dict(torch.load('tino_model.pth', map_location=torch.device('cpu')))
model.eval()

# Load Data
test_q = np.load('data/test_q.npy')[0]
test_cond = np.load('data/test_cond.npy')[0]

# --- THE CHALLENGE: SOLVE 100 FLIGHT SCENARIOS ---
BATCH_SIZE = 100
print(f"\n--- CHALLENGE: PREDICT {BATCH_SIZE} FUTURE SCENARIOS ---")

# 1. CFD SOLVER (Sequential)
print("1. CFD Solver (Running sequentially, please wait)...")
solver = ChannelSolver()
start_time = time.time()

# CFD must run in a loop (It cannot parallelize physics easily)
# We run 5 iterations and extrapolate to save you waiting 30 seconds
for i in range(5): 
    _ = solver.solve(mdot=test_cond[2], P_in_MPa=test_cond[0], 
                     T_in_K=test_cond[1], q_flux_profile=test_q)
    print(f"   Solved {i+1}/5...", end='\r')

cfd_time_per_sample = (time.time() - start_time) / 5
total_cfd_time = cfd_time_per_sample * BATCH_SIZE
print(f"\n   -> CFD Time for {BATCH_SIZE} cases: {total_cfd_time:.2f} seconds")

# 2. TINO AI (Batch Parallel)
print("\n2. TINO AI Model (Parallel Inference)...")

# Prepare a BATCH of 100 inputs
grid_size = 200
x_grid = np.linspace(0, 1, grid_size)
single_input = np.stack([
    x_grid, test_q / 1e7,
    np.full(grid_size, test_cond[0]/20.0),
    np.full(grid_size, test_cond[1]/600.0),
    np.full(grid_size, test_cond[2]/0.1)
], axis=1)

# Stack them into shape [100, 5, 200]
batch_input = np.tile(single_input, (BATCH_SIZE, 1, 1))
batch_tensor = torch.tensor(batch_input, dtype=torch.float32).permute(0, 2, 1).to(DEVICE)
# Note: Permute needed because model expects [Batch, 5, Grid] or similar depending on implementation
# Let's fix dimension to match model: [Batch, Grid, 5]
batch_tensor = torch.tensor(batch_input, dtype=torch.float32).to(DEVICE)

# Warmup
with torch.no_grad():
    _ = model(batch_tensor)

start_time = time.time()
with torch.no_grad():
    # ONE pass solves all 100
    _ = model(batch_tensor)
total_ai_time = time.time() - start_time
print(f"   -> AI Time for {BATCH_SIZE} cases:  {total_ai_time:.4f} seconds")

# --- THE VERDICT ---
speedup = total_cfd_time / total_ai_time
print(f"\n==========================================")
print(f"FINAL BATCH SPEEDUP: {speedup:.1f}x FASTER")
print(f"==========================================")