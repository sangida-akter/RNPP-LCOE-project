import numpy as np
import matplotlib.pyplot as plt
from SALib.sample import sobol
from SALib.analyze import sobol as sobol_analyze

# ---------- LCOE function ----------
hours_per_year = 8760
construction_period = 8
plant_lifetime = 60

def compute_lcoe(occ, capacity_mw, fuel_cost, fixed_om, variable_om,
                 capacity_factor, decom_per_kW, wacc, back_end_kwh):
    annual_gen_mwh = capacity_mw * hours_per_year * capacity_factor
    annual_gen_kwh = annual_gen_mwh * 1000

    annual_om_fuel = annual_gen_kwh * (fuel_cost + fixed_om + variable_om + back_end_kwh)
    capacity_kw = capacity_mw * 1000
    total_occ = occ * capacity_kw
    capex_annual_nominal = total_occ / construction_period
    decom_total = decom_per_kW * capacity_kw
    decom_annual = decom_total / 5.0
    discount = lambda t: 1.0 / ((1.0 + wacc) ** t)

    capex_disc = sum(capex_annual_nominal * discount(t) for t in range(1, construction_period + 1))
    omfuel_disc = sum(annual_om_fuel * discount(t) for t in range(construction_period + 1,
                                                                  construction_period + plant_lifetime + 1))
    decom_disc = sum(decom_annual * discount(t) for t in range(construction_period + plant_lifetime - 4,
                                                                construction_period + plant_lifetime + 1))

    total_discounted_cost = capex_disc + omfuel_disc + decom_disc
    discounted_energy_mwh = sum(annual_gen_mwh * discount(t) for t in range(construction_period + 1,
                                                                             construction_period + plant_lifetime + 1))
    return total_discounted_cost / discounted_energy_mwh

# ---------- Parameter ranges ----------
params = {
    "occ":               (5200.0, 6500.0),        # $/kW
    "capacity_mw":       (1070.0, 1944.0),        # MW
    "fuel_cost":         (0.0045, 0.0075),        # $/kWh
    "fixed_om":          (0.0065, 0.0145),        # $/kWh
    "variable_om":       (0.00095, 0.0014),       # $/kWh
    "capacity_factor":   (0.85, 0.92),            # -
    "decom_per_kW":      (468.0, 975.0),          # $/kW
    "wacc":              (0.03, 0.07),            # discount rate
    "back_end_kwh":      (0.0011, 0.0033)         # $/kWh
}

param_names = list(params.keys())
param_bounds = [list(params[k]) for k in param_names]

# ---------- Sobol problem ----------
problem = {
    'num_vars': len(param_names),
    'names': param_names,
    'bounds': param_bounds
}

# ---------- Generate Sobol samples ----------
N = 4096  # base sample size, power of 2 recommended
param_values = sobol.sample(problem, N, calc_second_order=False)

# ---------- Evaluate LCOE ----------
Y = np.array([compute_lcoe(*X) for X in param_values])

# ---------- Sobol sensitivity analysis ----------
Si = sobol_analyze.analyze(problem, Y, calc_second_order=False, print_to_console=True)

# ---------- Plot Tornado-style Sobol indices ----------
S1 = Si['S1']
ST = Si['ST']

# Sort parameters by total-order index
sorted_idx = np.argsort(ST)[::-1]
names_sorted = [param_names[i] for i in sorted_idx]
S1_sorted = S1[sorted_idx]
ST_sorted = ST[sorted_idx]

y_pos = np.arange(len(names_sorted))

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(y_pos - 0.15, S1_sorted, height=0.3, label='First-order (S1)', color='steelblue')
ax.barh(y_pos + 0.15, ST_sorted, height=0.3, label='Total-order (ST)', color='firebrick')

ax.set_yticks(y_pos)
ax.set_yticklabels(names_sorted)
ax.set_xlabel('Sobol Sensitivity Index')
ax.legend()
ax.grid(axis='x', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
