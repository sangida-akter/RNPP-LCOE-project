import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
import os

# ---------------- Setup ----------------
np.random.seed(0)
num_simulations = 10000
hours_per_year = 8760

# Plant & financial constants
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


# ---------------- Monte Carlo Simulation ----------------
construction_periods = [8, 12, 14, 16]
mean_lcoes = []
cis = []

for construction_period in construction_periods:
    lcoe_values = np.zeros(num_simulations)

    for i in range(num_simulations):
        # Randomized inputs
        occ = triangular(5200, 5800, 6500)  # $/kW
        capacity_mw = triangular(1070, 1818, 1944)
        fuel_cost = triangular(0.0045, 0.0060, 0.0075)  # $/kWh
        fixed_om = triangular(0.0065, 0.0081, 0.0145)  # $/kWh
        variable_om = triangular(0.00095, 0.0011, 0.0014)  # $/kWh
        capacity_factor = beta_pert(0.85, 0.90, 0.92)
        decom_cost_per_kw = triangular(468, 696, 975)
        wacc = uniform_continuous(0.03, 0.07)
        elec_price = triangular(95, 100, 130)  # $/MWh

        # Back-end costs
        interim_storage = triangular(0.20, 0.40, 0.60)
        waste_management = triangular(0.40, 0.80, 1.20)
        final_disposal = triangular(0.50, 1.00, 1.50)
        back_end_cost_kwh = (interim_storage + waste_management + final_disposal) / 1000.0

        # Derived values
        total_capex = occ * capacity_mw * 1000
        annual_generation_mwh = capacity_mw * hours_per_year * capacity_factor
        annual_generation_kwh = annual_generation_mwh * 1000
        annual_om_fuel = annual_generation_kwh * (fuel_cost + fixed_om + variable_om + back_end_cost_kwh)

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

    mean_lcoe = np.mean(lcoe_values)
    ci_lcoe = [np.percentile(lcoe_values, 2.5), np.percentile(lcoe_values, 97.5)]

    mean_lcoes.append(mean_lcoe)
    cis.append(ci_lcoe)

    print(
        f"Construction Period {construction_period} yr: Mean LCOE = {mean_lcoe:.2f} $/MWh, 95% CI = {ci_lcoe[0]:.2f}-{ci_lcoe[1]:.2f}")

# ---------------- Plot with 95% CI ----------------
scenarios = [f"{yr} yr" for yr in construction_periods]
error_bars = np.array([[mean - ci[0], ci[1] - mean] for mean, ci in zip(mean_lcoes, cis)]).T

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(scenarios, mean_lcoes, yerr=error_bars, color='indianred', edgecolor='black', capsize=5)

# Annotate mean LCOE
for bar, val in zip(bars, mean_lcoes):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{val:.2f}", ha='center', fontsize=10)

ax.set_ylabel("Mean LCOE ($/MWh)", fontsize=12)
ax.set_xlabel("Construction Period", fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
