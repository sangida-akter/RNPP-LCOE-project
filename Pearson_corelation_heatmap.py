import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------- LCOE FUNCTION -----------------
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
    omfuel_disc = sum(annual_om_fuel * discount(t)
                      for t in range(construction_period + 1,
                                     construction_period + plant_lifetime + 1))
    decom_disc = sum(decom_annual * discount(t)
                     for t in range(construction_period + plant_lifetime - 4,
                                    construction_period + plant_lifetime + 1))

    total_discounted_cost = capex_disc + omfuel_disc + decom_disc
    discounted_energy_mwh = sum(annual_gen_mwh * discount(t)
                                for t in range(construction_period + 1,
                                               construction_period + plant_lifetime + 1))

    return total_discounted_cost / discounted_energy_mwh

# ----------------- PARAMETER RANGES -----------------
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

# ----------------- MONTE CARLO SIMULATION -----------------
N = 5000
data = {k: [] for k in params.keys()}
data["LCOE"] = []

def sample_param(low, mode, high):
    return np.random.triangular(low, mode, high)

for _ in range(N):
    sample = {k: sample_param(*v) for k, v in params.items()}
    L = compute_lcoe(**sample)
    for k, v in sample.items():
        data[k].append(v)
    data["LCOE"].append(L)

df_mc = pd.DataFrame(data)

# ----------------- CORRELATION MATRICES -----------------
# Pearson correlation
corr_pearson = df_mc.corr()

# Spearman rank correlation
corr_spearman = df_mc.corr(method="spearman")

# ----------------- PRINT RESULTS -----------------
print("\n=== Pearson Correlation with LCOE ===")
print(corr_pearson["LCOE"].sort_values(ascending=False))

print("\n=== Spearman Rank Correlation with LCOE ===")
print(corr_spearman["LCOE"].sort_values(ascending=False))

# ----------------- HEATMAP (Pearson) -----------------
plt.figure(figsize=(11, 8))
sns.heatmap(corr_pearson, cmap="coolwarm", center=0, annot=True)
plt.show()
