import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

/* Main Background */
.stApp {
    background: linear-gradient(
        135deg,
        #667eea 0%,
        #764ba2 50%,
        #6B46C1 100%
    );
}

/* Main Title */
.main-title {
    text-align: center;
    font-size: 3.2rem;
    font-weight: bold;
    color: white;
    margin-top: 10px;
}

/* Subtitle */
.sub-title {
    text-align: center;
    font-size: 1.2rem;
    color: #e2e8f0;
    margin-bottom: 30px;
}

/* Metric Cards */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.10);
    border-radius: 15px;
    padding: 20px;
    backdrop-filter: blur(10px);
}

/* Section Headers */
h2, h3 {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.markdown(
    '<div class="main-title">📊 Customer Segmentation Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Customer Segmentation Using K-Means Clustering</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(
    "customer-segmentation/customer-segmentation/data/customer_segments.csv"
)

# --------------------------------------------------
# DASHBOARD METRICS
# --------------------------------------------------

total_customers = len(df)
total_clusters = df["Cluster"].nunique()

avg_income = round(df["Annual Income (k$)"].mean(), 2)
avg_spending = round(df["Spending Score (1-100)"].mean(), 2)

st.subheader("📈 Dashboard Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("👥 Customers", total_customers)

with col2:
    st.metric("🔵 Clusters", total_clusters)

with col3:
    st.metric("💰 Avg Income", avg_income)

with col4:
    st.metric("⭐ Avg Spending", avg_spending)

st.markdown("---")

# --------------------------------------------------
# CLUSTER DISTRIBUTION
# --------------------------------------------------

st.subheader("👥 Cluster Distribution")

cluster_counts = df["Cluster"].value_counts().sort_index()

fig = px.bar(
    x=cluster_counts.index,
    y=cluster_counts.values,
    labels={"x": "Cluster", "y": "Customers"},
    title="Customers in Each Cluster"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --------------------------------------------------
# DATA TABLE
# --------------------------------------------------

st.subheader("📋 Customer Dataset")

st.dataframe(df, use_container_width=True)

# --------------------------------------------------
# PROJECT INFORMATION
# --------------------------------------------------

st.markdown("---")

st.subheader("ℹ️ Project Information")

st.write("""
This project performs Customer Segmentation using the K-Means Clustering algorithm.

### Workflow
1. Dataset Understanding
2. Data Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Selection
5. Feature Scaling
6. K-Means Clustering
7. Elbow Method
8. Silhouette Analysis
9. Cluster Visualization
10. Business Insights

### Features Used
- Annual Income (k$)
- Spending Score (1-100)

### Algorithm
- K-Means Clustering
- Optimal Clusters: 5
""")