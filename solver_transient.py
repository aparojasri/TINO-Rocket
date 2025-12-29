"""
solver_transient.py
-------------------
Unsteady (Time-Dependent) 1D Solver for Regenerative Cooling.
Solves: d(rho*h)/dt + d(rho*u*h)/dx = Q_wall
Target: Capture "Ignition Transients" and "Throttle Step Responses".
"""
import numpy as np
import pandas as pd
from physics_engine import RealGasProperty

class TransientSolver:
    def __init__(self, length=0.5, nodes=100, dt=0.001):
        self.L = length
        self.N = nodes
        self.dx = length / nodes
        self.dt = dt # Time step (1 ms)
        self.fluid = RealGasProperty('n-Dodecane')
        
    def solve_scenario(self, time_steps, q_flux_func, P_in, T_in, mdot):
        """
        Simulates a full 1-second throttle event.
        q_flux_func: Function that returns Heat Flux at time t
        """
        # Initialize Field (Cold Start)
        H = np.ones(self.N) * self.fluid.get_initial_enthalpy(P_in, 300)
        T = np.ones(self.N) * 300
        results = []
        
        # Time Marching (The "Movie")
        for t in range(time_steps):
            time_sec = t * self.dt
            
            # Get Instantaneous Heat Flux (e.g., Engine throttling up)
            q_current = q_flux_func(time_sec) 
            
            # Explicit Euler Update (Simplified for stability)
            # H_new = H_old - u * dH/dx * dt + Q * dt
            # Note: We assume u is roughly constant for the thermal transient (valid for liquid)
            
            H_new = np.copy(H)
            for i in range(1, self.N):
                # Upwind Scheme for Advection
                dH_dx = (H[i] - H[i-1]) / self.dx
                
                # Source Term (Heat Addition)
                # Q = (4 * q_w) / (rho * D)
                state = self.fluid.get_properties(P_in, H[i])
                rho = state['rho']
                u = mdot / (rho * 0.000007) # Approx Area
                
                source = (4 * q_current[i]) / (0.003 * rho)
                
                # PDE Update
                H_new[i] = H[i] - u * dH_dx * self.dt + source * self.dt
            
            H = H_new
            
            # Save Snapshot every 10ms
            if t % 10 == 0:
                # Decode T from H
                T_profile = [self.fluid.get_properties(P_in, h_val)['T'] for h_val in H]
                results.append(T_profile)
                
        return np.array(results)

# Verification
if __name__ == "__main__":
    solver = TransientSolver()
    
    # Define a "Throttle Up" scenario (Heat rises from 0 to 5MW in 0.2s)
    def throttle_profile(t):
        power = min(t / 0.2, 1.0) * 5e6 
        return np.ones(100) * power
        
    print("Simulating Transient Ignition...")
    data = solver.solve_scenario(500, throttle_profile, 10e6, 300, 0.05)
    print(f"Generated {data.shape[0]} time snapshots.")