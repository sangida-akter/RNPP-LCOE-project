import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
import os

# ---------------- Setup ----------------
hours_per_year = 8760

# Construction & plant constants
construction_period = 8
plant_lifetime = 60
grace_period = 10
loan_term = 30
loan_interest = 0.04

national_equity = 1.27e9
russian_loan = 11.38e9

# Plant parameters (deterministic)
capacity_mw = 1818
occ = 5800  # $/kW
fuel_cost = 0.006  # $/kWh
fixed_om = 0.0081  # $/kWh
variable_om = 0.0011  # $/kWh
decom_cost_per_kw = 696
wacc = 0.05
elec_price = 100  # $/MWh
back_end_cost_kwh = (0.40 + 0.80 + 1.00) / 1000  # $/kWh
capacity_factor = 0.90

# Output folder
output_folder = "figures"
os.makedirs(output_folder, exist_ok=True)

# ---------------- Derived Values ----------------
annual_generation_mwh = capacity_mw * hours_per_year * capacity_factor
annual_generation_kwh = annual_generation_mwh * 1000

annual_fuel = annual_generation_kwh * fuel_cost
annual_fixed_om = annual_generation_kwh * fixed_om
annual_variable_om = annual_generation_kwh * variable_om
annual_back_end = annual_generation_kwh * back_end_cost_kwh

# Decommissioning
decom_total = decom_cost_per_kw * capacity_mw * 1000
decom_annual = decom_total / 5

# Loan repayment
amort_years = loan_term - grace_period
loan_payment = (russian_loan * (loan_interest * (1 + loan_interest) ** amort_years) /
                ((1 + loan_interest) ** amort_years - 1)) if amort_years > 0 else 0.0

equity_per_year = national_equity / construction_period
loan_draw_per_year = russian_loan / construction_period

# ---------------- Deterministic Cash Flows ----------------
cash_flows = []

# Construction phase
for t in range(1, construction_period + 1):
    cash_flows.append(-(equity_per_year + loan_draw_per_year))

# Operation phase
for t in range(1, plant_lifetime + 1):
    project_year = construction_period + t
    net_cash = annual_generation_mwh * elec_price - (annual_fuel + annual_fixed_om + annual_variable_om + annual_back_end)

    # Loan repayment during term (after grace period)
    repay_start = construction_period + grace_period + 1
    repay_end = construction_period + grace_period + amort_years
    if repay_start <= project_year <= repay_end:
        net_cash -= loan_payment

    # Decommissioning in last 5 years
    if t > plant_lifetime - 5:
        net_cash -= decom_annual

    cash_flows.append(net_cash)

# ---------------- Deterministic NPV & IRR ----------------
npv_det = npf.npv(wacc, cash_flows)
irr_det = npf.irr(cash_flows)

print(f"Deterministic NPV: {npv_det / 1e9:.2f} Billion USD")
print(f"Deterministic IRR: {irr_det * 100:.2f} %")

# ---------------- Deterministic LCOE ----------------
discount_factor = lambda t: 1 / ((1 + wacc) ** t)

# CapEx (discounted)
capex_discounted = sum((equity_per_year + loan_draw_per_year) * discount_factor(t)
                       for t in range(1, construction_period + 1))

# Fuel, O&M, Back-end (discounted)
fuel_discounted = sum(annual_fuel * discount_factor(t) for t in range(construction_period + 1, construction_period + plant_lifetime + 1))
fixed_om_discounted = sum(annual_fixed_om * discount_factor(t) for t in range(construction_period + 1, construction_period + plant_lifetime + 1))
variable_om_discounted = sum(annual_variable_om * discount_factor(t) for t in range(construction_period + 1, construction_period + plant_lifetime + 1))
back_end_discounted = sum(annual_back_end * discount_factor(t) for t in range(construction_period + 1, construction_period + plant_lifetime + 1))

# Decommissioning (discounted)
decom_discounted = sum(decom_annual * discount_factor(t) for t in range(construction_period + plant_lifetime - 4, construction_period + plant_lifetime + 1))

# Total discounted cost for standard (economic) LCOE
total_discounted_cost = capex_discounted + fuel_discounted + fixed_om_discounted + variable_om_discounted + back_end_discounted + decom_discounted

# Discounted electricity
discounted_energy_mwh = sum(annual_generation_mwh * discount_factor(t)
                            for t in range(construction_period + 1, construction_period + plant_lifetime + 1))

lcoe_det = total_discounted_cost / discounted_energy_mwh
print(f"Deterministic LCOE: {lcoe_det:.2f} $/MWh")



# ------------------------------------------
# LCOE results to plot (in $/MWh)
# ------------------------------------------
lcoe_values = {
    "Deterministic Baseline": 73,   # fixed expected inputs
    "Monte Carlo Mean":       84,    # mean of 10 000 simulations
    "Best Case":             51,    # favourable input set
    "Worst Case":            147    # conservative (high-cost) input set
}

labels  = list(lcoe_values.keys())
values  = list(lcoe_values.values())
colors  = ['steelblue', 'orange', 'forestgreen', 'firebrick']

# ------------------------------------------
# Create bar chart
# ------------------------------------------
plt.figure(figsize=(9, 5))
bars = plt.bar(labels, values, color=colors, edgecolor='black')

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2,
             height + 5,               # slight offset above bar
             f'{height:.1f}',
             ha='center', va='bottom', fontsize=9)

# Axis labels and formatting
plt.ylabel('LCOE ($/MWh)')
plt.ylim(0, max(values) * 1.15)        # add head-room for labels
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# Countries and their baseline LCOE values in $/MWh
countries = ['Russia', 'Belarus', 'Bangladesh', 'Turkey', 'Egypt', 'Hungary']
lcoe_values = [ 65, 70, 84, 115, 100, 105]

# Plot
plt.figure(figsize=(9, 5))
bars = plt.bar(countries, lcoe_values, color='cornflowerblue')

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 2, f'{height:.0f}',
             ha='center', va='bottom', fontsize=10)

# Labels and title
plt.ylabel('LCOE ($/MWh)')
plt.ylim(0, max(lcoe_values) + 20)
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
