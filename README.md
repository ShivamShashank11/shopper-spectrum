# 🛍️ Shopper Segmentation & Product Recommendation System

A complete end-to-end machine learning project for segmenting shoppers based on their purchasing behavior using K-Means clustering and recommending products using collaborative filtering.

---

## 📌 Project Highlights

- Customer Segmentation using **K-Means Clustering**
- Product Recommendation using **Item-Based Collaborative Filtering**
- Interactive **data visualizations** using Matplotlib & Seaborn
- Clean, modular **Jupyter Notebook** for full analysis
- Ready-to-deploy **Streamlit App** (optional UI)

---

## 📂 Project Structure

shopper-spectrum/
├── data/
│ └── online_retail.csv # dataset here
├── models/
│ ├── clustering_model.pkl
│ └── recommender_model.pkl
├── notebooks/
│ └── analysis.ipynb
├── src/
│ ├── preprocess.py
│ ├── clustering.py
│ ├── recommend.py
├── streamlit_app/
│ └── app.py
├── outputs/
│ └── figures/
└── README.md

---

## 📊 Dataset Information

- **Source**: UCI Machine Learning Repository
- **File**: `Online Retail.csv`
- **Features**:
  - `InvoiceNo`, `StockCode`, `Description`
  - `Quantity`, `UnitPrice`, `CustomerID`, `Country`
- **Size**: ~500K rows
- **Use Case**: Analyze transactions to discover customer behavior and preferences.

---

## 🚀 Features Implemented

### 1. Data Preprocessing

- Null handling
- Duplicate removal
- Country filtering (`United Kingdom`)
- Feature engineering: `TotalAmount`

### 2. Exploratory Data Analysis (EDA)

- Sales trends
- Country-wise distribution
- Top-selling products
- Purchase value distribution

### 3. Customer Segmentation (Clustering)

- RFM Analysis (Recency, Frequency, Monetary)
- Normalization using MinMaxScaler
- K-Means clustering (Elbow Method + Silhouette Score)
- Cluster visualization

### 4. Product Recommendation (Recommender System)

- Customer-product interaction matrix
- Cosine similarity for item-based filtering
- Recommend top 5 items per customer
- Recommendations shown using tables and charts

---

## 📷 Visualizations

| Chart Type   | Purpose            |
| ------------ | ------------------ |
| Pie Chart    | Sales by Country   |
| Bar Plot     | Top Products       |
| Heatmap      | Correlation Matrix |
| Cluster Plot | Customer Segments  |

All visualizations are created using **Matplotlib**, **Seaborn**, and optionally displayed in **Streamlit**.

---

## 🧠 Machine Learning Models

| Model                              | Purpose            | Library          |
| ---------------------------------- | ------------------ | ---------------- |
| K-Means Clustering                 | Segment customers  | Scikit-learn     |
| Item-Based Collaborative Filtering | Recommend products | Sklearn + Pandas |

Models are trained and saved using `pickle`.

---

## 📈 Sample Results

### 📍 Customer Segments

- Cluster 0: High value, frequent shoppers
- Cluster 1: Infrequent, low spenders
- Cluster 2: Medium frequency, high spenders

### 🎯 Sample Recommendations

| Customer ID | Top Recommended Products       |
| ----------- | ------------------------------ |
| 12345       | White Mug, Gift Bag, Red Pen   |
| 67890       | Alarm Clock, Lunch Box, Candle |

---

## 💻 How to Run

### ▶️ Run Notebook

jupyter notebook notebook/shopper_segmentation.ipynb

## Run Streamlit App

streamlit run streamlit_app/app.py

🔮 Future Improvements
Time-based product trends

Deep Learning-based recommender

Deploy on HuggingFace or Streamlit Cloud

Google BigQuery or Snowflake for large datasets

📌 Author
Shivam Shashank
Machine Learning & Data Enthusiast
📧 [shivamshashank961@gmail.com]
🔗 LinkedIn: www.linkedin.com/in/shivam-shashank-616957213
🔗 GitHub: https://github.com/ShivamShashank11
