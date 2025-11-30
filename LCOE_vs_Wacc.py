import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
import os

# ---------------- Setup ----------------
np.random.seed(0)
num_simulations = 10000
hours_per_year = 8760

construction_period = 8  # years
plant_lifetime = 60      # years

output_folder = "figures"
os.makedirs(output_folder, exist_ok=True)

# ---------------- Random Distribution Functions ----------------
def triangular(low, mode, high):
    return np.random.triangular(low, mode, high)

def beta_pert(low, mode, high, lamb=4):
    alpha = 1 + lamb * ((mode - low) / (high - low))
    beta = 1 + lamb * ((high - mode) / (high - low))
    return np.random.beta(alpha, beta) * (high - low) + low

def uniform_continuous(low, high):
    return np.random.uniform(low, high)

# ---------------- Storage ----------------
lcoe_values = np.zeros(num_simulations)
wacc_values = np.zeros(num_simulations)

# ---------------- Simulation ----------------
for i in range(num_simulations):

    # ---- Random inputs ----
    occ = triangular(5200, 5800, 6500)                # $/kW
    capacity_mw = triangular(1070, 1818, 1944)        # MW
    fuel_cost = triangular(0.0045, 0.0060, 0.0075)    # $/kWh
    fixed_om = triangular(0.0065, 0.0081, 0.0145)     # $/kWh
    variable_om = triangular(0.00095, 0.0011, 0.0014) # $/kWh
    capacity_factor = beta_pert(0.85, 0.90, 0.92)
    decom_cost_per_kw = triangular(468, 696, 975)     # $/kW
    wacc = uniform_continuous(0.03, 0.07)

    # Back-end costs (converted to $/kWh)
    interim_storage = triangular(0.20, 0.40, 0.60)    # $/kWh
    waste_management = triangular(0.40, 0.80, 1.20)   # $/kWh
    final_disposal = triangular(0.50, 1.00, 1.50)     # $/kWh
    back_end_cost_kwh = (interim_storage + waste_management + final_disposal) / 1000  # $/kWh

    # ---- Derived quantities ----
    capacity_kw = capacity_mw * 1000                   # kW
    occ_total = occ * capacity_kw                      # $ total construction
    annual_generation = capacity_mw * hours_per_year * capacity_factor * 1000  # kWh

    annual_cost = annual_generation * (fuel_cost + fixed_om + variable_om + back_end_cost_kwh)
    decom_total = decom_cost_per_kw * capacity_kw

    # ---- Discounted cost and energy ----
    discounted_cost = 0
    discounted_energy = 0

    # Construction cost (spread over construction period)
    for t in range(1, construction_period + 1):
        discounted_cost += (occ_total / construction_period) / ((1 + wacc)**t)

    # Operating cost, energy, and decommissioning
    for t in range(1, plant_lifetime + 1):
        year = construction_period + t
        discounted_cost += annual_cost / ((1 + wacc)**year)

        # Decommissioning in final 5 years
        if t > plant_lifetime - 5:
            discounted_cost += (decom_total / 5) / ((1 + wacc)**year)

        discounted_energy += annual_generation / ((1 + wacc)**year)

    # ---- LCOE in $/MWh ----
    lcoe = (discounted_cost / discounted_energy) * 1000  # convert from $/kWh to $/MWh

    lcoe_values[i] = lcoe
    wacc_values[i] = wacc

# ---------------- Plot LCOE vs WACC ----------------
plt.figure(figsize=(10,6))
plt.scatter(wacc_values * 100, lcoe_values, s=8, alpha=0.25)
plt.xlabel("WACC (%)")
plt.ylabel("LCOE ($/MWh)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f"{output_folder}/lcoe_vs_wacc.png", dpi=300)
plt.show()
