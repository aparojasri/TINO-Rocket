"""
physics_engine.py
-----------------
Module for handling Real Gas thermophysical properties using CoolProp.
Target Fluid: n-Dodecane (Scientific Surrogate for RP-1).

Project: TINO (Thermodynamic-Informed Neural Operator)
Protocol: NIST RefProp Logic via CoolProp
"""

import CoolProp.CoolProp as CP
import numpy as np

class RealGasProperty:
    def __init__(self, fluid_name='n-Dodecane'):
        self.fluid = fluid_name
        try:
            self.T_crit = CP.PropsSI('Tcrit', self.fluid)
            self.P_crit = CP.PropsSI('Pcrit', self.fluid)
            # Pre-calculate Critical Enthalpy for stability checks
            self.H_crit = CP.PropsSI('H', 'P', self.P_crit, 'T', self.T_crit, self.fluid)
            print(f"[TINO-Physics] Initialized {self.fluid}. P_crit={self.P_crit/1e6:.2f} MPa")
        except Exception as e:
            print(f"[Error] Could not initialize fluid {fluid_name}. Check CoolProp installation.")
            raise e

    def get_properties(self, P_Pa, H_Jkg):
        """
        Returns fluid state based on Pressure (Pa) and Enthalpy (J/kg).
        
        Why Enthalpy (H)? 
        Near the pseudo-critical point, Temperature (T) is extremely sensitive 
        to energy changes. H is conserved and monotonic, making the solver stable.
        """
        try:
            # Core State Update (Inverting H,P -> T,Rho)
            T = CP.PropsSI('T', 'P', P_Pa, 'H', H_Jkg, self.fluid)
            rho = CP.PropsSI('D', 'P', P_Pa, 'H', H_Jkg, self.fluid)
            
            # Transport & Thermal Properties
            mu = CP.PropsSI('V', 'P', P_Pa, 'H', H_Jkg, self.fluid) # Viscosity [Pa.s]
            k = CP.PropsSI('L', 'P', P_Pa, 'H', H_Jkg, self.fluid)  # Conductivity [W/m/K]
            cp = CP.PropsSI('C', 'P', P_Pa, 'H', H_Jkg, self.fluid) # Specific Heat [J/kg/K]
            
            return {
                'T': T,
                'rho': rho,
                'mu': mu,
                'k': k,
                'cp': cp,
                'status': 'OK'
            }
        except Exception as e:
            # Catch singularities (rare, but happens if solver steps exactly on P_crit)
            return {'status': 'FAIL', 'error': str(e)}

    def get_initial_enthalpy(self, P_Pa, T_K):
        """Helper to convert Inlet Temperature to Inlet Enthalpy."""
        return CP.PropsSI('H', 'P', P_Pa, 'T', T_K, self.fluid)

# Unit Test
if __name__ == "__main__":
    engine = RealGasProperty()
    # Test a Supercritical Point (10 MPa, 500K)
    test_H = engine.get_initial_enthalpy(10e6, 500)
    print(f"Test State (10 MPa, 500K): {engine.get_properties(10e6, test_H)}")