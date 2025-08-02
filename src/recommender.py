import pandas as pd
import pickle
import os

# Paths
RFM_PATH = "data/rfm_scaled.csv"
MODEL_PATH = "models/clustering_model.pkl"
OUTPUT_PATH = "data/customer_segments.csv"

# Load scaled RFM data
print("📥 Loading scaled RFM data...")
rfm_scaled = pd.read_csv(RFM_PATH, index_col=0)

# Load clustering model
print("🤖 Loading clustering model...")
with open(MODEL_PATH, 'rb') as f:
    clustering_model = pickle.load(f)

# Predict cluster for each customer
print("🔍 Predicting customer segments...")
rfm_scaled['Cluster'] = clustering_model.predict(rfm_scaled)

# Add customer ID as column (if needed)
rfm_scaled['CustomerID'] = rfm_scaled.index

# Rearranging columns
segment_df = rfm_scaled[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Cluster']]

# Save to CSV
os.makedirs("data", exist_ok=True)
segment_df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Customer segments saved at {OUTPUT_PATH}")

# Optional: Cluster-wise summary
print("\n📊 Cluster Summary:")
print(segment_df.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean().round(2))
