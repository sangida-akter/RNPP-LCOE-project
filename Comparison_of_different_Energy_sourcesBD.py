import matplotlib.pyplot as plt
import numpy as np

# LCOE values (in $/MWh) for each power source
sources = [
    "Nuclear (Rooppur)",
    "Coal (imported)",
    "Gas (domestic)",
    "HFO (oil-based)",
    "Solar PV",
    "Onshore Wind"
]
lcoe = [84, 105, 65, 145, 58, 80]

# Assign colors for visual clarity
colors = [
    "steelblue",  # Nuclear
    "dimgray",    # Coal
    "seagreen",   # Gas
    "firebrick",  # HFO
    "gold",       # Solar
    "mediumturquoise"  # Wind
]

# Create bar chart
plt.figure(figsize=(10, 5))
bars = plt.bar(sources, lcoe, color=colors, edgecolor="black")

# Annotate each bar with its value
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 3,                   # offset for label
        f"{height:.0f}",
        ha="center", va="bottom", fontsize=9
    )

# Chart formatting
plt.ylabel("LCOE ($/MWh)")
plt.ylim(0, max(lcoe) * 1.15)
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()
