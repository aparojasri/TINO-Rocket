"""
solver.py
---------
1D Quasi-Steady Finite Difference Solver for Regenerative Cooling Channels.
Solves the Euler Equations with Heat Addition and Friction.

Project: TINO (Thermodynamic-Informed Neural Operator)
Method: Spatial Marching (Upstream -> Downstream)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from physics_engine import RealGasProperty

class ChannelSolver:
    def __init__(self, length=0.5, diameter=0.003, nodes=200):
        """
        Args:
            length: Channel length [m] (Default: 0.5m)
            diameter: Hydraulic diameter [m] (Default: 3mm)
            nodes: Spatial resolution (Default: 200 points)
        """
        self.L = length
        self.D = diameter
        self.N = nodes
        self.dx = length / nodes
        self.x_grid = np.linspace(0, length, nodes)
        
        # Instantiate Physics Engine
        self.fluid_model = RealGasProperty('n-Dodecane')

    def solve(self, mdot, P_in_MPa, T_in_K, q_flux_profile):
        """
        Executes the spatial marching simulation.
        
        Args:
            mdot: Mass flow rate [kg/s]
            P_in_MPa: Inlet Pressure [MPa] (Must be > P_crit for supercritical)
            T_in_K: Inlet Temperature [K]
            q_flux_profile: Array of heat flux values [W/m2] matching node size
        """
        # 1. Initialize Field Arrays
        P = np.zeros(self.N)
        H = np.zeros(self.N)
        T = np.zeros(self.N)
        rho = np.zeros(self.N)
        u = np.zeros(self.N)
        cp = np.zeros(self.N) # Tracking Cp to detect the spike
        
        # 2. Boundary Conditions (Inlet)
        P[0] = P_in_MPa * 1e6
        H[0] = self.fluid_model.get_initial_enthalpy(P[0], T_in_K)
        
        # Get initial state
        state0 = self.fluid_model.get_properties(P[0], H[0])
        T[0] = state0['T']
        rho[0] = state0['rho']
        cp[0] = state0['cp']
        # u = mdot / (rho * Area)
        area = np.pi * (self.D/2)**2
        u[0] = mdot / (rho[0] * area)
        
        # 3. Spatial Marching Loop
        for i in range(self.N - 1):
            # A. Energy Conservation
            # Enthalpy_next = Enthalpy_current + (Heat_Added / Mass_Flow)
            # Heat_Added = q_flux * Surface_Area_Segment
            heat_input = q_flux_profile[i] * (np.pi * self.D * self.dx)
            H[i+1] = H[i] + (heat_input / mdot)
            
            # B. Momentum Conservation (Pressure Drop)
            # dP = - Friction - Acceleration
            # Simplified Darcy-Weisbach: f ~ 0.02 (approx for rough channel)
            f = 0.02 
            dP_friction = -f * (self.dx / self.D) * (rho[i] * u[i]**2) / 2
            P[i+1] = P[i] + dP_friction
            
            # C. State Update (The Physics Call)
            state = self.fluid_model.get_properties(P[i+1], H[i+1])
            
            if state['status'] == 'FAIL':
                print(f"[Solver] Crash at Node {i} (x={self.x_grid[i]:.3f}m): {state['error']}")
                # Fill remaining with NaNs to indicate failure
                T[i+1:] = np.nan
                break
                
            T[i+1] = state['T']
            rho[i+1] = state['rho']
            cp[i+1] = state['cp']
            u[i+1] = mdot / (rho[i+1] * area)
            
        # 4. Package Results
        return pd.DataFrame({
            'x_m': self.x_grid,
            'P_MPa': P/1e6,
            'T_K': T,
            'H_Jkg': H,
            'rho': rho,
            'u': u,
            'Cp': cp,
            'q_applied': q_flux_profile
        })

# --- EXPERIMENTAL RUN (Verification) ---
if __name__ == "__main__":
    # Create Solver Instance
    solver = ChannelSolver(length=0.5, diameter=0.003, nodes=200)
    
    # Define Conditions
    # Scenario: High Heat Flux Ramp causing Pseudo-Boiling
    q_flux = np.linspace(2e6, 8e6, 200) # 2 MW/m2 -> 8 MW/m2
    
    print("--- STARTING TINO VERIFICATION RUN ---")
    results = solver.solve(mdot=0.04, P_in_MPa=8, T_in_K=300, q_flux_profile=q_flux)
    
    # Visualization
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Channel Position (m)')
    ax1.set_ylabel('Fluid Temperature (K)', color=color)
    ax1.plot(results['x_m'], results['T_K'], color=color, linewidth=2, label='Temperature')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Specific Heat Cp (J/kg/K)', color=color)
    ax2.plot(results['x_m'], results['Cp'], color=color, linestyle='--', linewidth=2, label='Cp Spike')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('TINO Verification: Pseudo-Boiling Detection (Cp Spike)')
    fig.tight_layout()
    plt.show()
    
    print("Simulation Complete. Check the plot for the 'Cp Spike'.")