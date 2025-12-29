"""
plot_paper_figure.py
--------------------
Generates "Figure 1" for the AIP Physics of Fluids submission.
Features:
- Dual Y-Axis (Temperature vs. Specific Heat)
- Error Subplot
- Professional Formatting (300 DPI)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from model import TINO_FNO1d

# --- SETUP ---
# Use clean, professional fonts
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load Data & Model
print("Loading Model and Data...")
model = TINO_FNO1d(modes=16, width=64).to(DEVICE)
try:
    model.load_state_dict(torch.load('tino_model.pth', map_location=DEVICE))
except:
    model.load_state_dict(torch.load('tino_model.pth', map_location=torch.device('cpu')))
model.eval()

# Load Test Dataset
# We want a sample with a distinct spike. Let's scan for the one with the highest Cp.
test_q = np.load('data/test_q.npy')
test_cond = np.load('data/test_cond.npy')
test_T = np.load('data/test_T.npy')
test_Cp = np.load('data/test_Cp.npy') # We need this for the physics overlay

# Find the index of the sample with the maximum Cp spike (The "Hardest" case)
max_cp_idx = np.argmax([np.max(cp) for cp in test_Cp])
print(f"Plotting Sample {max_cp_idx} (Hardest Physics Case)...")

# 2. Run Inference on that sample
grid_size = 200
x_grid = np.linspace(0, 0.5, grid_size) # 0.5m length

# Prepare input
sample_input = np.stack([
    np.linspace(0, 1, grid_size), # Normalized x
    test_q[max_cp_idx] / 1e7,
    np.full(grid_size, test_cond[max_cp_idx][0]/20.0),
    np.full(grid_size, test_cond[max_cp_idx][1]/600.0),
    np.full(grid_size, test_cond[max_cp_idx][2]/0.1)
], axis=1)

input_tensor = torch.tensor(sample_input, dtype=torch.float32).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    prediction_norm = model(input_tensor).cpu().numpy()[0]
    
# Denormalize
T_pred = prediction_norm * 1000.0
T_true = test_T[max_cp_idx]
Cp_true = test_Cp[max_cp_idx] / 1000.0 # Convert to kJ/kg.K for cleaner axis

# 3. Create the Master Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

# --- TOP PANEL: Temperature & Physics ---
# Plot CFD vs AI
line1, = ax1.plot(x_grid, T_true, 'k-', linewidth=2.5, label='CFD Ground Truth (Real Gas)')
line2, = ax1.plot(x_grid, T_pred, 'r--', linewidth=2.5, label='TINO Prediction (Ours)')

ax1.set_ylabel('Wall Temperature (K)', fontsize=14, fontweight='bold')
ax1.tick_params(axis='y', labelsize=12)
ax1.grid(True, which='major', alpha=0.3)

# Create Twin Axis for Cp (The Physics)
ax1b = ax1.twinx()
fill = ax1b.fill_between(x_grid, 0, Cp_true, color='blue', alpha=0.15, label='Specific Heat ($C_p$)')
line3, = ax1b.plot(x_grid, Cp_true, 'b:', linewidth=1.5)

ax1b.set_ylabel('Specific Heat $C_p$ (kJ/kg$\cdot$K)', color='blue', fontsize=14)
ax1b.tick_params(axis='y', labelcolor='blue', labelsize=12)
ax1b.set_ylim(0, np.max(Cp_true)*1.2) # Give it headroom

# Legend
lines = [line1, line2, fill]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', fontsize=11, frameon=True, framealpha=0.9)
ax1.set_title(f'TINO Performance in Pseudo-Boiling Regime (Sample {max_cp_idx})', fontsize=16, pad=15)

# --- BOTTOM PANEL: Error Residuals ---
error = np.abs(T_true - T_pred)
ax2.plot(x_grid, error, 'k-', linewidth=1.5)
ax2.fill_between(x_grid, 0, error, color='gray', alpha=0.3)
ax2.set_ylabel('Abs. Error (K)', fontsize=12)
ax2.set_xlabel('Channel Position (m)', fontsize=14, fontweight='bold')
ax2.set_ylim(0, max(np.max(error)*1.1, 1.0)) # Auto scale
ax2.grid(True, alpha=0.3)

# Add Annotations
max_err = np.max(error)
mean_err = np.mean(error)
ax2.text(0.02, 0.8, f'Max Error: {max_err:.2f} K\nMean Error: {mean_err:.2f} K', 
         transform=ax2.transAxes, fontsize=11, bbox=dict(facecolor='white', alpha=0.8))

# Save
plt.tight_layout()
plt.savefig('Figure1_TINO_Publication.png', dpi=300)
print("Figure 1 Generated: 'Figure1_TINO_Publication.png'")
plt.show()