import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# 📁 STEP 0: Load cleaned data
data_path = "data/cleaned_data.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ File not found: {data_path}\nMake sure the file exists in the 'data/' folder.")

df = pd.read_csv(data_path)
print(f"✅ Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

# 🧮 STEP 0.1: Add TotalAmount column
if 'TotalAmount' not in df.columns:
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    print("➕ Added 'TotalAmount' column")

# 📅 Convert InvoiceDate to datetime
if not np.issubdtype(df['InvoiceDate'].dtype, np.datetime64):
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# 🧠 STEP 1: RFM Feature Engineering
print("🔄 Generating RFM features...")
rfm_df = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (df['InvoiceDate'].max() - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalAmount': 'sum'
}).reset_index()
rfm_df.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
print(f"✅ RFM table created: {rfm_df.shape}")

# 🧪 STEP 2: Feature Scaling
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])
print("✅ RFM features scaled")

# 🔗 STEP 3: KMeans Clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(rfm_scaled)
print("✅ KMeans clustering model trained")

# 💾 STEP 4: Save scaler & clustering model
os.makedirs("models", exist_ok=True)
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("models/clustering_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)
print("💾 Saved: models/scaler.pkl & models/clustering_model.pkl")

# 🎯 STEP 5: Recommender System via cosine similarity
print("🔄 Building item-customer matrix...")
item_cust_matrix = df.pivot_table(index='Description', columns='CustomerID', values='Quantity', fill_value=0)
print(f"✅ Item-Customer matrix created: {item_cust_matrix.shape}")

similarity_matrix = cosine_similarity(item_cust_matrix)
recommender_df = pd.DataFrame(similarity_matrix, index=item_cust_matrix.index, columns=item_cust_matrix.index)

# 💾 STEP 6: Save recommender model
with open("models/recommender_model.pkl", "wb") as f:
    pickle.dump(recommender_df, f)
print("💾 Saved: models/recommender_model.pkl")

print("\n🎉 All models trained and saved successfully!")
