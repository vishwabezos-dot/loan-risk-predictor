import pandas as pd

# Read safely
df = pd.read_csv("Loan_Default.csv", on_bad_lines='skip', engine='python')

# Keep only required columns
df = df[['loan_amount', 'rate_of_interest', 'income', 'status']]

# Take very small sample
df_small = df.sample(n=500, random_state=42)

# Save compressed (optional but useful)
df_small.to_csv("Loan_Default_small.csv", index=False)

print("Done. Small file created.")