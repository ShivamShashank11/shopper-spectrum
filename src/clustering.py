# src/clustering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle
import os

# Paths
RAW_DATA_PATH = 'data/online_retail.csv'
CLEANED_DATA_PATH = 'data/cleaned_data.csv'
CLUSTER_MODEL_PATH = 'models/clustering_model.pkl'
SCALER_PATH = 'models/scaler.pkl'

def load_and_clean_data():
    if not os.path.exists(RAW_DATA_PATH):
        print(f"❌ File not found: {RAW_DATA_PATH}")
        return None

    print("📥 Loading raw data...")
    df = pd.read_csv(RAW_DATA_PATH, encoding='ISO-8859-1')

    print("🧹 Cleaning data...")
    df.dropna(subset=['CustomerID', 'InvoiceDate', 'Quantity', 'UnitPrice'], inplace=True)
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]

    # Create TotalSum column
    df['TotalSum'] = df['Quantity'] * df['UnitPrice']

    # Save cleaned data for reference
    os.makedirs('data', exist_ok=True)
    df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"✅ Cleaned data saved at {CLEANED_DATA_PATH}")
    return df

def calculate_rfm(df):
    print("📆 Converting InvoiceDate...")
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    print("📊 Calculating RFM values...")
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalSum': 'sum'
    }).reset_index()

    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    return rfm.set_index('CustomerID')

def train_clustering_model():
    df = load_and_clean_data()
    if df is None:
        return

    rfm = calculate_rfm(df)

    print("📏 Scaling RFM values...")
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    print("🔁 Training KMeans clustering model...")
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(rfm_scaled)

    print("💾 Saving models...")
    os.makedirs('models', exist_ok=True)
    with open(CLUSTER_MODEL_PATH, 'wb') as f:
        pickle.dump(kmeans, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"✅ Clustering model saved at {CLUSTER_MODEL_PATH}")
    print(f"✅ Scaler saved at {SCALER_PATH}")

# ✅ Run this file directly to execute training
if __name__ == "__main__":
    train_clustering_model()
