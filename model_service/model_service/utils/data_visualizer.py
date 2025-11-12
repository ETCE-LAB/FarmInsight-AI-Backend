import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/optimized_data.csv", parse_dates=["date"])

print(df.isna().sum())

print(df.describe())

corr = df[["rain_amount", "soil_moisture", "water_level"]].corr()

print("\nKorrelationsmatrix:")
print(corr)

sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Korrelationen")
plt.show()

df.set_index("date", inplace=True)
df.plot(subplots=True, figsize=(10, 6), title=["Regen", "Bodenfeuchte", "Wasserstand"])
plt.tight_layout()
plt.show()

print(df.isna().sum())

missing_ratio = df.isna().mean().sort_values(ascending=False)
print("\nAnteil fehlender Werte (%):")
print(missing_ratio * 100)

missing_ratio.plot(kind="bar", title="Anteil fehlender Werte pro Spalte")
plt.ylabel("Prozent fehlend")
plt.tight_layout()
plt.show()
