import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ----------------- LCOE Function -----------------
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

# ----------------- Parameter Table -----------------
params = {
    "occ":               (5200.0, 5800.0, 6500.0),
    "capacity_mw":       (1070.0, 1818.0, 1944.0),
    "fuel_cost":         (0.0045, 0.0060, 0.0075),
    "fixed_om":          (0.0065, 0.0081, 0.0145),
    "variable_om":       (0.00095, 0.0011, 0.0014),
    "capacity_factor":   (0.85, 0.90, 0.92),
    "decom_per_kW":      (468.0, 696.0, 975.0),
    "wacc":              (0.03, 0.05, 0.07),
    "back_end_kwh":      (0.0011, 0.0022, 0.0033)
}

pretty = {
    "occ": "OCC ($/kW)",
    "capacity_mw": "Plant Capacity (MW)",
    "fuel_cost": "Fuel Cost ($/kWh)",
    "fixed_om": "Fixed O&M ($/kWh)",
    "variable_om": "Variable O&M ($/kWh)",
    "capacity_factor": "Capacity Factor",
    "decom_per_kW": "Decommissioning ($/kW)",
    "wacc": "WACC",
    "back_end_kwh": "Back-end ($/kWh)"
}

# ----------------- Base Case -----------------
base = {k: v[1] for k, v in params.items()}
LCOE_base = compute_lcoe(**base)

# ----------------- Sensitivity Sweep -----------------
results = {}
print("Sensitivity Effects on LCOE (% change from base):\n")
for pname, (low, mode, high) in params.items():
    args_low = base.copy()
    args_low[pname] = low
    L_low = compute_lcoe(**args_low)

    args_high = base.copy()
    args_high[pname] = high
    L_high = compute_lcoe(**args_high)

    pct_low = (L_low - LCOE_base) / LCOE_base * 100
    pct_high = (L_high - LCOE_base) / LCOE_base * 100
    max_effect = max(abs(pct_low), abs(pct_high))

    results[pname] = {
        "L_base": LCOE_base,
        "pct_low": pct_low,
        "pct_high": pct_high,
        "max_effect": max_effect
    }

    print(f"{pretty[pname]}: decrease = {pct_low:.2f}%, increase = {pct_high:.2f}%, max effect = {max_effect:.2f}%")

# ----------------- Prepare DataFrame -----------------
df = pd.DataFrame([
    {"Parameter": pretty[k], **v} for k, v in results.items()
])
df = df.sort_values('max_effect', ascending=True)

# ----------------- Tornado Chart -----------------
fig, ax = plt.subplots(figsize=(8,6))
ax.barh(df['Parameter'], df['pct_low'], color='tomato', label='Decrease')
ax.barh(df['Parameter'], df['pct_high'], color='limegreen', label='Increase')
ax.axvline(0, color='gray', linestyle='--', label='Base LCOE')
ax.set_xlabel('LCOE Change (%)')
ax.set_ylabel('Parameter')
ax.legend()
plt.tight_layout()
plt.show()

# ----------------- Spider (Radar) Plot -----------------
labels = df['Parameter'].tolist()
num_vars = len(labels)
values_low = df['pct_low'].tolist()
values_high = df['pct_high'].tolist()

# Close the loop
values_low += values_low[:1]
values_high += values_high[:1]
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
ax.plot(angles, values_high, color='limegreen', linewidth=2, label='Best Case')
ax.fill(angles, values_high, color='limegreen', alpha=0.25)
ax.plot(angles, values_low, color='tomato', linewidth=2, label='Worst Case')
ax.fill(angles, values_low, color='tomato', alpha=0.25)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_title("LCOE Sensitivity Spider Plot (%)", y=1.1)
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
max_range = max(max(values_high), abs(min(values_low))) * 1.1
ax.set_ylim(-max_range, max_range)
plt.show()
