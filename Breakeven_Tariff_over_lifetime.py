import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
import os

# ---------------- Setup ----------------
np.random.seed(0)
num_simulations = 10000
hours_per_year = 8760

# Construction & plant constants
construction_period = 8
plant_lifetime = 60
grace_period = 10
loan_term = 30
loan_interest = 0.04

national_equity = 1.27e9
russian_loan = 11.38e9

# Output folder
output_folder = "figures"
os.makedirs(output_folder, exist_ok=True)

# ---------------- Distributions ----------------
def triangular(low, mode, high):
    return np.random.triangular(low, mode, high)

def beta_pert(low, mode, high, lamb=4):
    alpha = 1 + lamb * ((mode - low) / (high - low))
    beta = 1 + lamb * ((high - mode) / (high - low))
    return np.random.beta(alpha, beta) * (high - low) + low

def uniform_continuous(low, high):
    return np.random.uniform(low, high)

# ---------------- Monte Carlo Storage ----------------
lcoe_values = np.zeros(num_simulations)
npv_values = np.zeros(num_simulations)
irr_values = np.zeros(num_simulations)

# ---------------- Monte Carlo Simulation ----------------
for i in range(num_simulations):
    # Randomized parameters
    occ = triangular(5200, 5800, 6500)  # $/kW
    capacity_mw = triangular(1070, 1818, 1944)
    fuel_cost = triangular(0.0045, 0.0060, 0.0075)  # $/kWh
    fixed_om = triangular(0.0065, 0.0081, 0.0145)  # $/kWh
    variable_om = triangular(0.00095, 0.0011, 0.0014)  # $/kWh
    capacity_factor = beta_pert(0.85, 0.90, 0.92)
    decom_cost_per_kw = triangular(468, 696, 975)
    wacc = uniform_continuous(0.03, 0.07)
    elec_price = triangular(95, 100, 130)  # $/MWh

    # Back-end fuel cycle costs (once-through)
    interim_storage = triangular(0.20, 0.40, 0.60)
    waste_management = triangular(0.40, 0.80, 1.20)
    final_disposal = triangular(0.50, 1.00, 1.50)
    back_end_cost_kwh = (interim_storage + waste_management + final_disposal) / 1000.0

    # Derived values
    total_capex = occ * capacity_mw * 1000
    annual_generation_mwh = capacity_mw * hours_per_year * capacity_factor
    annual_generation_kwh = annual_generation_mwh * 1000
    annual_om_fuel = annual_generation_kwh * (fuel_cost + fixed_om + variable_om + back_end_cost_kwh)
    annual_revenue = annual_generation_mwh * elec_price

    # Decommissioning
    decom_total = decom_cost_per_kw * capacity_mw * 1000
    decom_annual = decom_total / 5

    # Loan payments
    amort_years = loan_term - grace_period
    loan_payment = (russian_loan * (loan_interest * (1 + loan_interest) ** amort_years) /
                    ((1 + loan_interest) ** amort_years - 1)) if amort_years > 0 else 0.0

    equity_per_year = national_equity / construction_period
    loan_draw_per_year = russian_loan / construction_period

    # Discounted cost calculation
    discount_factor = lambda t: 1 / ((1 + wacc) ** t)
    capex_discounted = sum((equity_per_year + loan_draw_per_year) * discount_factor(t)
                           for t in range(1, construction_period + 1))
    omfuel_discounted = sum(annual_om_fuel * discount_factor(t)
                            for t in range(construction_period + 1, construction_period + plant_lifetime + 1))
    decom_discounted = sum(decom_annual * discount_factor(t)
                           for t in range(construction_period + plant_lifetime - 4,
                                          construction_period + plant_lifetime + 1))

    total_discounted_cost = capex_discounted + omfuel_discounted + decom_discounted
    discounted_energy_mwh = sum(annual_generation_mwh * discount_factor(t)
                                for t in range(construction_period + 1, construction_period + plant_lifetime + 1))
    lcoe_values[i] = total_discounted_cost / discounted_energy_mwh

    # Cash flows for NPV & IRR
    cash_flows = []
    for t in range(1, construction_period + 1):
        cash_flows.append(-(equity_per_year + loan_draw_per_year))
    for t in range(1, plant_lifetime + 1):
        project_year = construction_period + t
        net_cash = annual_revenue - annual_om_fuel
        repay_start = construction_period + grace_period + 1
        repay_end = construction_period + grace_period + amort_years
        if repay_start <= project_year <= repay_end:
            net_cash -= loan_payment
        if t > plant_lifetime - 5:
            net_cash -= decom_annual
        cash_flows.append(net_cash)

    npv_values[i] = npf.npv(wacc, cash_flows)
    try:
        irr_values[i] = npf.irr(cash_flows)
    except:
        irr_values[i] = np.nan

# ---------------- Probabilistic Summary ----------------
mean_lcoe = np.nanmean(lcoe_values)
ci_lcoe = np.nanpercentile(lcoe_values, [2.5, 97.5])
best_lcoe = np.min(lcoe_values)
worst_lcoe = np.max(lcoe_values)

print("----- Probabilistic (Monte Carlo) Results -----")
print(f"Mean LCOE: {mean_lcoe:.2f} $/MWh")
print(f"LCOE 95% CI: {ci_lcoe[0]:.2f} – {ci_lcoe[1]:.2f} $/MWh")
print(f"Best-case LCOE: {best_lcoe:.2f} $/MWh")
print(f"Worst-case LCOE: {worst_lcoe:.2f} $/MWh")
print(f"Mean NPV: {np.nanmean(npv_values) / 1e9:.2f} Billion USD")
print(f"Mean IRR: {np.nanmean(irr_values[~np.isnan(irr_values)])*100:.2f} %")

# ---------------- Deterministic Inputs ----------------
capacity_mw_det = 1818
occ_det = 5800
fuel_cost_det = 0.006
fixed_om_det = 0.0081
variable_om_det = 0.0011
decom_cost_per_kw_det = 696
wacc_det = 0.05
elec_price_det = 100
back_end_cost_kwh_det = (0.40 + 0.80 + 1.00) / 1000

annual_generation_mwh = capacity_mw_det * hours_per_year * 0.90
annual_generation_kwh = annual_generation_mwh * 1000

annual_fuel = annual_generation_kwh * fuel_cost_det
annual_fixed_om = annual_generation_kwh * fixed_om_det
annual_variable_om = annual_generation_kwh * variable_om_det
annual_back_end = annual_generation_kwh * back_end_cost_kwh_det

equity_per_year = national_equity / construction_period
loan_draw_per_year = russian_loan / construction_period

# Decommissioning
decom_total = decom_cost_per_kw_det * capacity_mw_det * 1000
decom_annual = decom_total / 5

# Loan repayment
amort_years = loan_term - grace_period
loan_payment = (russian_loan * (loan_interest * (1 + loan_interest) ** amort_years) /
                ((1 + loan_interest) ** amort_years - 1) if amort_years > 0 else 0.0)

# ---------------- Tariff over Lifetime ----------------
tariff_per_year = []

for year in range(1, plant_lifetime + 1):
    project_year = construction_period + year
    net_cost = 0
    # Loan repayment
    repay_start = construction_period + grace_period + 1
    repay_end = construction_period + grace_period + amort_years
    if repay_start <= project_year <= repay_end:
        net_cost += loan_payment
    # O&M + fuel + back-end
    net_cost += annual_fuel + annual_fixed_om + annual_variable_om + annual_back_end
    # Decommissioning last 5 years
    if year > plant_lifetime - 5:
        net_cost += decom_annual
    tariff_per_year.append(net_cost / annual_generation_mwh)  # $/MWh

tariff_per_year = np.array(tariff_per_year)

# ---------------- Print Yearly Tariff ----------------
print("\n----- Yearly Tariff Over Lifetime -----")
for year, tariff in enumerate(tariff_per_year, start=1):
    print(f"Year {year}: {tariff:.2f} $/MWh")

# ---------------- Plot Tariff ----------------
plt.figure(figsize=(12, 6))
plt.plot(range(1, plant_lifetime + 1), tariff_per_year, color='blue', linewidth=2)
plt.xlabel("Year")
plt.ylabel("Tariff ($/MWh)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "tariff_over_lifetime.png"), dpi=600)
plt.savefig(os.path.join(output_folder, "tariff_over_lifetime.pdf"))
plt.show()
