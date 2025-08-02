# src/preprocess.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import pickle

# Load cleaned data
df = pd.read_csv("data/cleaned_data.csv")

# Ensure InvoiceDate is datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Calculate TotalAmount if not present
if 'TotalAmount' not in df.columns:
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

# Reference date for recency
reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

# Compute RFM
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',                                     # Frequency
    'TotalAmount': 'sum'                                        # Monetary
})
rfm.columns = ['Recency', 'Frequency', 'Monetary']

# Remove non-positive monetary values
rfm = rfm[rfm['Monetary'] > 0]

# Scale RFM
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# Create scaled RFM DataFrame
rfm_scaled_df = pd.DataFrame(rfm_scaled, index=rfm.index, columns=rfm.columns)

# Save the scaled RFM data
os.makedirs("data", exist_ok=True)
rfm_scaled_df.to_csv("data/rfm_scaled.csv")
print("✅ Scaled RFM data saved at data/rfm_scaled.csv")

# Save the scaler
os.makedirs("models", exist_ok=True)
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("✅ Scaler saved at models/scaler.pkl")
