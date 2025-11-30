import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------
# LCOE data ($/MWh) for three power sources under four scenarios
# ------------------------------------------------------------------
scenarios = ["Baseline",
             "20% CAPEX\nSubsidy (Nuclear)",
             "2% Concessional\nFinancing (Nuclear)",
             "Carbon Pricing\n$50/t CO2"]

# values in order: Nuclear, Coal, Gas
lcoe_nuclear = [84, 62, 50, 84]

# baseline; subsidy; concessional; carbon
lcoe_coal    = [105,   105,  105,  150]       # unchanged except carbon pricing
lcoe_gas     = [65,     65,   65,   87.5]     # unchanged except carbon pricing

# ------------------------------------------------------------------
# Plotting parameters
# ------------------------------------------------------------------
x = np.arange(len(scenarios))           # scenario positions
bar_width = 0.25

plt.figure(figsize=(10, 5))

# Plot bars for each technology
plt.bar(x - bar_width,  lcoe_nuclear, width=bar_width,
        color="steelblue", label="Nuclear (Rooppur)")
plt.bar(x,              lcoe_coal,    width=bar_width,
        color="dimgray", label="Coal (imported)")
plt.bar(x + bar_width,  lcoe_gas,     width=bar_width,
        color="seagreen", label="Gas (domestic)")

# Add data labels on top of each bar
for idx, vals in enumerate([lcoe_nuclear, lcoe_coal, lcoe_gas]):
    for pos, val in zip(x + (idx - 1) * bar_width, vals):
        plt.text(pos, val + 3, f"{val:.1f}", ha="center", va="bottom", fontsize=8)

# Axis formatting
plt.xticks(x, scenarios, rotation=15, ha="right")
plt.ylabel("LCOE ($/MWh)")
plt.ylim(0, max(lcoe_coal + lcoe_nuclear + lcoe_gas) * 1.15)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()
