import pandas as pd

# Load your CSV
df = pd.read_csv("c:/VERGER/REVENUE_TRACKER/forecasting_data.csv")

# Clean and standardize the Month column
df["Month"] = df["Month"].str.strip().str.title()

# Convert Month names to numbers
month_map = {
    'January': 1, 'February': 2, 'March': 3,
    'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9,
    'October': 10, 'November': 11, 'December': 12
}
df["Month_No"] = df["Month"].map(month_map)

# Create Sort_Index to track chronological order
df["Sort_Index"] = df["Year"] * 100 + df["Month_No"]

# Define grouping keys
group_cols = ["Account", "Product", "Type"]

# Sort by group and time
df = df.sort_values(by=group_cols + ["Sort_Index"]).reset_index(drop=True)

# Create lag features
df["Lag_Qty_1"] = df.groupby(group_cols)["Quantity"].shift(1)
df["Lag_Qty_2"] = df.groupby(group_cols)["Quantity"].shift(2)
df["Lag_Rev_1"] = df.groupby(group_cols)["Revenue"].shift(1)
df["Lag_Rev_2"] = df.groupby(group_cols)["Revenue"].shift(2)

# Create rolling mean features (excluding current row)
df["Rolling_Qty_3"] = (
    df.groupby(group_cols)["Quantity"]
    .apply(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
    .reset_index(drop=True)
)

df["Rolling_Rev_3"] = (
    df.groupby(group_cols)["Revenue"]
    .apply(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
    .reset_index(drop=True)
)

# Optional: restore final sort by original time
df = df.sort_values(by="Sort_Index").reset_index(drop=True)

# Save the updated file
df.to_csv("c:/VERGER/REVENUE_TRACKER/forecasting_data_with_features.csv", index=False)

print("✅ Lag and rolling features added with correct time alignment.")
