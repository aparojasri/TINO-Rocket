"""
data_gen.py
-----------
Mass production script for Project TINO.
Generates 1,000 Training Samples and 200 Test Samples.

Features:
- Random Heat Flux Profiles (Gaussian, Step, Ramp)
- Regime Coverage: Sub-critical, Trans-critical, Super-critical
- Saves data as .npy (NumPy binary) for fast loading
"""

import numpy as np
import os
from solver import ChannelSolver
from tqdm import tqdm  # Progress bar

# --- CONFIGURATION ---
NUM_TRAIN = 800
NUM_TEST = 200
SAVE_DIR = 'data'

# Physics Constraints (Pushing into the Danger Zone)
P_MIN, P_MAX = 6.0, 15.0  # MPa (Critical is ~1.8 MPa)
T_IN_MIN, T_IN_MAX = 300, 500  # K
Q_BASE = 2e6  # 2 MW/m2
Q_VAR = 6e6   # Up to 8 MW/m2 (This ensures we hit T_crit)

def generate_random_profile(nodes=200):
    """Generates a realistic engine heat flux profile."""
    x = np.linspace(0, 1, nodes)
    
    # Base: A smooth sine wave (typical engine profile)
    profile = Q_BASE + Q_VAR * np.sin(np.pi * x)
    
    # Add random localized "Hot Spots" (Simulating injector streaks)
    if np.random.rand() > 0.5:
        loc = np.random.randint(20, 180)
        width = np.random.randint(5, 20)
        profile[loc-width:loc+width] += 2e6  # 2 MW/m2 spike
        
    # Add Sensor Noise (1%)
    noise = np.random.normal(0, 0.01 * np.max(profile), nodes)
    return profile + noise

def run_campaign(count, mode='train'):
    print(f"--- Generating {count} {mode} samples ---")
    
    inputs_q = []  # Heat Flux (Input)
    inputs_cond = [] # Conditions [P_in, T_in, mdot]
    outputs_T = [] # Wall Temperature (Target)
    outputs_Cp = [] # Specific Heat (For Physics Loss)
    
    solver = ChannelSolver()
    
    for i in tqdm(range(count)):
        # Random Operating Conditions
        P_in = np.random.uniform(P_MIN, P_MAX)
        T_in = np.random.uniform(T_IN_MIN, T_IN_MAX)
        mdot = np.random.uniform(0.03, 0.06) # kg/s
        
        # Generate Profile
        q_flux = generate_random_profile(solver.N)
        
        # Run Solver
        try:
            df = solver.solve(mdot, P_in, T_in, q_flux)
            
            # Check for Solver Failure (NaNs)
            if df.isnull().values.any():
                print(f"Sample {i} Failed (Solver Crash). Skipping.")
                continue
                
            # Store Data
            inputs_q.append(q_flux)
            inputs_cond.append([P_in, T_in, mdot])
            outputs_T.append(df['T_K'].values)
            outputs_Cp.append(df['Cp'].values)
            
        except Exception as e:
            print(f"Sample {i} Error: {e}")
            continue

    # Save to Disk
    np.save(os.path.join(SAVE_DIR, f'{mode}_q.npy'), np.array(inputs_q))
    np.save(os.path.join(SAVE_DIR, f'{mode}_cond.npy'), np.array(inputs_cond))
    np.save(os.path.join(SAVE_DIR, f'{mode}_T.npy'), np.array(outputs_T))
    np.save(os.path.join(SAVE_DIR, f'{mode}_Cp.npy'), np.array(outputs_Cp))
    
    print(f"Saved {len(inputs_q)} valid samples to {SAVE_DIR}/")

if __name__ == "__main__":
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    # You might need to install tqdm: pip install tqdm
    run_campaign(NUM_TRAIN, 'train')
    run_campaign(NUM_TEST, 'test')