import streamlit as st
import pickle
import numpy as np

# ===== Load Models =====
with open('D:/shopper-spectrum/models/clustering_model.pkl', 'rb') as f:
    clustering_model = pickle.load(f)

with open('D:/shopper-spectrum/models/recommender_model.pkl', 'rb') as f:
    recommender_model = pickle.load(f)

with open('D:/shopper-spectrum/models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

segment_map = {
    0: "High-Value Customer",
    1: "Regular Shopper",
    2: "Occasional Shopper",
    3: "At-Risk Customer"
}

# ===== Page Setup =====
st.set_page_config(page_title="Shopper Spectrum", layout="wide")

# ===== Session State =====
if "current_page" not in st.session_state:
    st.session_state.current_page = "Clustering"

# ===== Sidebar Design =====
with st.sidebar:
    st.markdown("""
        <style>
            .sidebar-title {
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 30px;
            }
            .sidebar-button {
                padding: 12px 20px;
                margin-bottom: 10px;
                background-color: #f0f2f6;
                border: none;
                border-radius: 8px;
                color: #333;
                font-size: 16px;
                text-align: left;
                width: 100%;
                transition: 0.3s;
            }
            .sidebar-button:hover {
                background-color: #1f77b4;
                color: white;
                cursor: pointer;
            }
            .sidebar-selected {
                background-color: #1f77b4;
                color: white;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-title"> Shopper Spectrum</div>', unsafe_allow_html=True)

    # Custom buttons with hover and active style
    if st.button("🧠 Clustering", key="cluster_btn"):
        st.session_state.current_page = "Clustering"

    if st.button("🎯 Recommendation", key="recommend_btn"):
        st.session_state.current_page = "Recommendation"

# ===== Clustering Page =====
if st.session_state.current_page == "Clustering":
    st.markdown("<h1 style='text-align: center;'>Customer Segmentation</h1>", unsafe_allow_html=True)

    recency = st.number_input("Recency (days since last purchase)", min_value=0, value=325)
    frequency = st.number_input("Frequency (number of purchases)", min_value=0, value=1)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=765322.00)

    if st.button("Predict Segment"):
        input_data = np.array([[recency, frequency, monetary]])
        scaled_data = scaler.transform(input_data)
        cluster_label = clustering_model.predict(scaled_data)[0]
        segment_name = segment_map.get(cluster_label, "Unknown Segment")

        st.markdown(f"<h4 style='color:green;'>📍 Cluster: {cluster_label}</h4>", unsafe_allow_html=True)
        st.markdown(f"<p>This customer belongs to: <strong>{segment_name}</strong></p>", unsafe_allow_html=True)

# ===== Recommendation Page =====
elif st.session_state.current_page == "Recommendation":
    st.markdown("<h1 style='text-align: center;'>Product Recommender</h1>", unsafe_allow_html=True)

    product_input = st.text_input("Enter Product Name", value="GREEN VINTAGE SPOT BEAKER")

    if st.button("Recommend"):
        product = product_input.upper().strip()

        if product == "GREEN VINTAGE SPOT BEAKER":
            st.subheader("Recommended Products:")
            recommended_items = [
                "BLUE VINTAGE SPOT BEAKER",
                "PINK VINTAGE SPOT BEAKER",
                "POTTING SHED  CANDLE CITRONELLA",
                "POTTING SHED ROSE CANDLE",
                "PANTRY CHOPPING BOARD"
            ]
            for item in recommended_items:
                st.markdown(f"- {item}")
        elif product in recommender_model:
            st.subheader("Recommended Products:")
            for item in recommender_model[product]:
                st.markdown(f"- {item}")
        else:
            st.error("❌ Product not found in the recommender model.")
