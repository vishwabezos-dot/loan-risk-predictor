import pandas as pd

df = pd.read_csv("Loan_Default.csv", on_bad_lines='skip', engine='python')

# only needed columns
df = df[['loan_amount','rate_of_interest','income','status']]

# reduce rows (IMPORTANT ↓)
df_small = df.sample(2000, random_state=42)   # 👈 change 5000 → 2000

df_small.to_csv("Loan_Default_small.csv", index=False)

print("Small file created!")