import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt

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

# Fixed deterministic inputs
capacity_mw = 1818
occ = 5800
fuel_cost = 0.006
fixed_om = 0.0081
variable_om = 0.0011
decom_cost_per_kw = 696
wacc_det = 0.05
back_end_cost_kwh = (0.40 + 0.80 + 1.00) / 1000  # $/kWh

# Derived values
annual_generation_mwh = capacity_mw * hours_per_year * 0.90
annual_generation_kwh = annual_generation_mwh * 1000

annual_fuel = annual_generation_kwh * fuel_cost
annual_fixed_om = annual_generation_kwh * fixed_om
annual_variable_om = annual_generation_kwh * variable_om
annual_back_end = annual_generation_kwh * back_end_cost_kwh

# Decommissioning
decom_total = decom_cost_per_kw * capacity_mw * 1000
decom_annual = decom_total / 5

# Loan payments
amort_years = loan_term - grace_period
loan_payment = (russian_loan * (loan_interest * (1 + loan_interest) ** amort_years) /
                ((1 + loan_interest) ** amort_years - 1) if amort_years > 0 else 0.0)

# CapEx per year
equity_per_year = national_equity / construction_period
loan_draw_per_year = russian_loan / construction_period

# ---------------- Range of Tariffs ----------------
tariffs = np.linspace(10, 200, 100)  # $/MWh
npv_list = []
irr_list = []

for elec_price in tariffs:
    cash_flows = []

    # Construction phase
    for t in range(1, construction_period + 1):
        cash_flows.append(-(equity_per_year + loan_draw_per_year))

    # Operation phase
    for t in range(1, plant_lifetime + 1):
        project_year = construction_period + t
        annual_revenue = annual_generation_mwh * elec_price
        annual_cost = annual_fuel + annual_fixed_om + annual_variable_om + annual_back_end
        net_cash = annual_revenue - annual_cost

        repay_start = construction_period + grace_period + 1
        repay_end = construction_period + grace_period + amort_years
        if repay_start <= project_year <= repay_end:
            net_cash -= loan_payment
        if t > plant_lifetime - 5:
            net_cash -= decom_annual

        cash_flows.append(net_cash)

    npv_list.append(npf.npv(wacc_det, cash_flows))
    irr_value = npf.irr(cash_flows)
    irr_list.append(irr_value * 100 if irr_value is not None else np.nan)

# ---------------- Plot NPV and IRR vs Tariff ----------------
fig, ax1 = plt.subplots(figsize=(10, 6))

# NPV curve (solid blue)
color1 = 'tab:blue'
ax1.set_xlabel('Electricity Tariff ($/MWh)')
ax1.set_ylabel('NPV (Billion USD)', color=color1)
line1, = ax1.plot(tariffs, np.array(npv_list)/1e9, color=color1, label='NPV', linewidth=2)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, linestyle='--', alpha=0.5)

# IRR curve (dashed red)
ax2 = ax1.twinx()
color2 = 'tab:red'
line2, = ax2.plot(tariffs, irr_list, color=color2, linestyle='--', label='IRR', linewidth=2)
ax2.set_ylabel('IRR (%)', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

# Add legend
lines = [line1, line2]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper left')

fig.tight_layout()
plt.show()