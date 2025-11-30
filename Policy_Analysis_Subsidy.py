import numpy as np
import numpy_financial as npf

# -------------------- Setup --------------------
np.random.seed(0)
num_simulations = 10000
hours_per_year = 8760

# Plant and financial constants
construction_period = 8
plant_lifetime = 60
grace_period = 10
loan_term = 30
loan_interest = 0.04

# Equity / loan fraction (used for cost split)
national_equity_fraction = 1.27e9 / (1.27e9 + 11.38e9)
russian_loan_fraction = 1 - national_equity_fraction

# -------------------- Distribution Functions --------------------
def triangular(low, mode, high):
    if low == mode == high:
        return low
    return np.random.triangular(low, mode, high)

def beta_pert(low, mode, high, lamb=4):
    alpha = 1 + lamb * ((mode - low) / (high - low))
    beta = 1 + lamb * ((high - mode) / (high - low))
    return np.random.beta(alpha, beta) * (high - low) + low

def uniform_continuous(low, high):
    return np.random.uniform(low, high)

# -------------------- Monte Carlo Storage --------------------
lcoe_values = np.zeros(num_simulations)
npv_values = np.zeros(num_simulations)
irr_values = np.zeros(num_simulations)

# -------------------- OCC Distribution (Original) --------------------
OCC_LOW = 6000      # $/kW
OCC_MODE = 6500
OCC_HIGH = 7000
SUBSIDY = 0.8       # 20% CAPEX subsidy

# -------------------- Monte Carlo Simulation --------------------
for i in range(num_simulations):
    # --- Sample OCC and apply 20% subsidy ---
    occ_no_subsidy = triangular(OCC_LOW, OCC_MODE, OCC_HIGH)
    occ = occ_no_subsidy * SUBSIDY

    # --- Sample other parameters ---
    capacity_mw = triangular(1070, 1818, 1944)
    fuel_cost = triangular(0.0045, 0.0060, 0.0075)  # $/kWh
    fixed_om = triangular(0.0065, 0.0081, 0.0145)  # $/kWh
    variable_om = triangular(0.00095, 0.0011, 0.0014)  # $/kWh
    capacity_factor = beta_pert(0.85, 0.90, 0.92)
    decom_cost_per_kw = triangular(468, 696, 975)  # $/kW
    wacc = uniform_continuous(0.03, 0.07)
    elec_price = triangular(95, 100, 130)  # $/MWh

    # --- Back-end fuel costs ($/kWh) ---
    interim_storage = triangular(0.20, 0.40, 0.60)
    waste_management = triangular(0.40, 0.80, 1.20)
    final_disposal = triangular(0.50, 1.00, 1.50)
    back_end_cost_kwh = (interim_storage + waste_management + final_disposal) / 1000.0

    # --- CAPEX, equity & loan ---
    total_capex = occ * capacity_mw * 1000
    equity_total = total_capex * national_equity_fraction
    loan_total = total_capex * russian_loan_fraction

    equity_per_year = equity_total / construction_period
    loan_draw_per_year = loan_total / construction_period

    # --- Annual generation & O&M + fuel ---
    annual_generation_mwh = capacity_mw * hours_per_year * capacity_factor
    annual_generation_kwh = annual_generation_mwh * 1000
    annual_om_fuel = annual_generation_kwh * (fuel_cost + fixed_om + variable_om + back_end_cost_kwh)
    annual_revenue = annual_generation_mwh * elec_price

    # --- Decommissioning ---
    decom_total = decom_cost_per_kw * capacity_mw * 1000
    decom_annual = decom_total / 5

    # --- Loan amortization ---
    amort_years = loan_term - grace_period
    loan_payment = (loan_total * (loan_interest * (1 + loan_interest) ** amort_years) /
                    ((1 + loan_interest) ** amort_years - 1)) if amort_years > 0 else 0.0

    # --- Discount function ---
    df = lambda t: 1 / ((1 + wacc) ** t)

    # --- Discounted cost calculation ---
    capex_discounted = sum((equity_per_year + loan_draw_per_year) * df(t)
                           for t in range(1, construction_period + 1))
    omfuel_discounted = sum(annual_om_fuel * df(t)
                            for t in range(construction_period + 1,
                                           construction_period + plant_lifetime + 1))
    decom_discounted = sum(decom_annual * df(t)
                           for t in range(construction_period + plant_lifetime - 4,
                                          construction_period + plant_lifetime + 1))
    total_discounted_cost = capex_discounted + omfuel_discounted + decom_discounted

    discounted_energy_mwh = sum(annual_generation_mwh * df(t)
                                for t in range(construction_period + 1,
                                               construction_period + plant_lifetime + 1))
    lcoe_values[i] = total_discounted_cost / discounted_energy_mwh

    # --- Cash flows for NPV / IRR ---
    cash_flows = []

    # Construction years
    for t in range(1, construction_period + 1):
        cash_flows.append(-(equity_per_year + loan_draw_per_year))

    # Operation years
    repay_start = construction_period + grace_period + 1
    repay_end = repay_start + amort_years - 1
    for t in range(1, plant_lifetime + 1):
        abs_year = construction_period + t
        net_cash = annual_revenue - annual_om_fuel

        if repay_start <= abs_year <= repay_end:
            net_cash -= loan_payment
        if t > plant_lifetime - 5:
            net_cash -= decom_annual

        cash_flows.append(net_cash)

    # NPV
    npv_values[i] = npf.npv(wacc, cash_flows)

    # IRR
    try:
        irr_values[i] = npf.irr(cash_flows)
        if np.isnan(irr_values[i]) or np.isinf(irr_values[i]):
            irr_values[i] = np.nan
    except:
        irr_values[i] = np.nan

# -------------------- Probabilistic Summary --------------------
mean_lcoe = np.nanmean(lcoe_values)
ci_lcoe = np.nanpercentile(lcoe_values, [2.5, 97.5])

mean_npv = np.nanmean(npv_values)
ci_npv = np.nanpercentile(npv_values, [2.5, 97.5])

mean_irr = np.nanmean(irr_values[~np.isnan(irr_values)])
ci_irr = np.nanpercentile(irr_values[~np.isnan(irr_values)], [2.5, 97.5])

print("----- Monte Carlo Results with 20% CAPEX Subsidy -----")
print(f"Mean LCOE: {mean_lcoe:.2f} $/MWh")
print(f"LCOE 95% CI: {ci_lcoe[0]:.2f} – {ci_lcoe[1]:.2f} $/MWh")
print(f"Mean NPV: {mean_npv / 1e9:.2f} Billion USD")
print(f"NPV 95% CI: {ci_npv[0] / 1e9:.2f} – {ci_npv[1] / 1e9:.2f} Billion USD")
print(f"Mean IRR: {mean_irr * 100:.2f} %")
print(f"IRR 95% CI: {ci_irr[0] * 100:.2f} – {ci_irr[1] * 100:.2f} %")
