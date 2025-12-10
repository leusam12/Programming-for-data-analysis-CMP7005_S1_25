import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os



print(">>> RUNNING NEW SCRIPT VERSION <<<")

file_path = r"C:\Users\Jide\Desktop\New folder (2)\StaticsStreamlit\Assessment Data-20251028\cleaned_dataset.csv"
df = pd.read_csv(file_path)

# Output folder for images
output_folder = "images"
os.makedirs(output_folder, exist_ok=True)

# ----------------------------
# 1. Missing Values Heatmap
# ----------------------------
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Values Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "missing_values_heatmap.png"))
plt.close()

# ----------------------------
# 2. Correlation Heatmap
# ----------------------------
plt.figure(figsize=(14, 8))
numeric_df = df.select_dtypes(include=["number"])
sns.heatmap(numeric_df.corr(), annot=False, cmap="viridis")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "correlation_heatmap.png"))
plt.close()

# ----------------------------
# 3. Pollutant Distribution Example (PM2.5)
# ----------------------------
if "PM2.5" in df.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(df["PM2.5"], kde=True)
    plt.title("PM2.5 Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "pollutant_distribution.png"))
    plt.close()
