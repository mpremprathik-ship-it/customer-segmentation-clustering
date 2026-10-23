
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.preprocessing import StandardScaler


# -------------------- PAGE CONFIGURATION --------------------

st.set_page_config(
    page_title="Customer Intelligence | Segmentation",
    page_icon="🛍️",
    layout="wide",
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "Mall_Customers.csv"

FEATURES = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)",
]


# -------------------- STYLING --------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f7fb;
    }
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #16324f;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e1e8f0;
        padding: 16px;
        border-radius: 12px;
    }
    div[data-testid="stTabs"] button {
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------- LOAD AND VALIDATE DATA --------------------

@st.cache_data
def load_data(path_string):
    data = pd.read_csv(path_string)
    data.columns = data.columns.str.strip()

    required = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    missing = [column for column in required if column not in data.columns]

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Available columns: {list(data.columns)}"
        )

    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=required).copy()
    return data


try:
    df = load_data(str(DATA_PATH))
except Exception as error:
    st.error(f"Unable to load the dataset: {error}")
    st.info(
        "Check that Mall_Customers.csv exists inside "
        "D:\\customer-segmentation\\data\\raw"
    )
    st.stop()

if len(df) < 3:
    st.error("At least three valid customer records are required.")
    st.stop()

numeric_data = df[FEATURES].copy()
scaler = StandardScaler()
X = scaler.fit_transform(numeric_data)


# -------------------- SIDEBAR --------------------

st.sidebar.title("🛍️ Customer Intelligence")
st.sidebar.caption("Customer Segmentation Using Clustering")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Data Exploration",
        "Model Comparison",
        "Customer Segments",
        "New Customer",
    ],
)

st.sidebar.divider()
st.sidebar.subheader("Clustering settings")

k = st.sidebar.slider(
    "Number of clusters (K)",
    min_value=2,
    max_value=min(10, len(df) - 1),
    value=min(5, len(df) - 1),
)

eps = st.sidebar.slider(
    "DBSCAN eps",
    min_value=0.1,
    max_value=2.0,
    value=0.5,
    step=0.1,
    help="Maximum neighborhood distance in standardized feature space.",
)

min_samples = st.sidebar.slider(
    "DBSCAN minimum samples",
    min_value=2,
    max_value=10,
    value=5,
)

st.sidebar.caption(
    "All algorithms use the same three standardized numeric features. "
    "Gender is retained for profiling when available."
)


# -------------------- FIT MODELS --------------------

@st.cache_data
def compare_models(data_values, cluster_count, dbscan_eps, dbscan_min_samples):
    values = np.asarray(data_values, dtype=float)

    models = {
        "K-Means": KMeans(
            n_clusters=cluster_count,
            random_state=42,
            n_init=10,
        ),
        "Agglomerative": AgglomerativeClustering(
            n_clusters=cluster_count,
            linkage="ward",
        ),
        "DBSCAN": DBSCAN(
            eps=dbscan_eps,
            min_samples=dbscan_min_samples,
        ),
    }

    labels_by_model = {}
    rows = []

    for name, model in models.items():
        labels = model.fit_predict(values)
        labels_by_model[name] = labels

        if name == "DBSCAN":
            # Evaluate DBSCAN on non-noise records only.
            valid_mask = labels != -1
            evaluation_values = values[valid_mask]
            evaluation_labels = labels[valid_mask]
            noise_count = int(np.sum(labels == -1))
        else:
            evaluation_values = values
            evaluation_labels = labels
            noise_count = 0

        cluster_count_found = len(np.unique(labels[labels != -1]))
        evaluation_cluster_count = len(np.unique(evaluation_labels))

        silhouette = np.nan
        davies_bouldin = np.nan
        calinski_harabasz = np.nan

        # Internal metrics require at least 2 clusters and fewer
        # clusters than evaluated samples.
        if (
            len(evaluation_values) >= 3
            and 2 <= evaluation_cluster_count < len(evaluation_values)
        ):
            silhouette = silhouette_score(
                evaluation_values, evaluation_labels
            )
            davies_bouldin = davies_bouldin_score(
                evaluation_values, evaluation_labels
            )
            calinski_harabasz = calinski_harabasz_score(
                evaluation_values, evaluation_labels
            )

        rows.append(
            {
                "Model": name,
                "Clusters": cluster_count_found,
                "Noise Points": noise_count,
                "Silhouette Score": silhouette,
                "Davies-Bouldin Index": davies_bouldin,
                "Calinski-Harabasz Score": calinski_harabasz,
            }
        )

    return pd.DataFrame(rows), labels_by_model


comparison, labels_by_model = compare_models(
    X,
    k,
    eps,
    min_samples,
)


# -------------------- HEADER --------------------

st.title("Customer Intelligence Dashboard")
st.write(
    "Discover customer groups, compare clustering algorithms, "
    "and turn customer data into actionable insights."
)

st.caption("Dataset: Mall Customers | Learning approach: Unsupervised ML")

st.divider()


# -------------------- OVERVIEW --------------------

if page == "Overview":
    st.subheader("Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Customers", f"{len(df):,}")
    col2.metric("Features", len(FEATURES))
    col3.metric("Clustering Models", "3")
    col4.metric("Missing Values", f"{int(df[FEATURES].isna().sum().sum())}")

    st.markdown("### Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    left, right = st.columns(2)

    with left:
        st.markdown("### Annual Income Distribution")
        fig = px.histogram(
            df,
            x="Annual Income (k$)",
            nbins=20,
            color_discrete_sequence=["#3478c7"],
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("### Spending Score Distribution")
        fig = px.histogram(
            df,
            x="Spending Score (1-100)",
            nbins=20,
            color_discrete_sequence=["#12a594"],
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Project Workflow")
    st.write(
        "Data validation → Feature scaling → Clustering → "
        "Model evaluation → Customer profiling"
    )


# -------------------- DATA EXPLORATION --------------------

elif page == "Data Exploration":
    st.subheader("Explore Customer Behaviour")

    selected_x = st.selectbox("X-axis", FEATURES, index=1)
    selected_y = st.selectbox("Y-axis", FEATURES, index=2)

    color_column = "Gender" if "Gender" in df.columns else None

    fig = px.scatter(
        df,
        x=selected_x,
        y=selected_y,
        color=color_column,
        hover_data=[
            column for column in ["CustomerID", "Age", "Gender"]
            if column in df.columns
        ],
        title=f"{selected_x} vs {selected_y}",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Feature Correlation")
    corr = df[FEATURES].corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation Between Numeric Features",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Descriptive Statistics")
    st.dataframe(df[FEATURES].describe().round(2), use_container_width=True)


# -------------------- MODEL COMPARISON --------------------

elif page == "Model Comparison":
    st.subheader("Compare Clustering Algorithms")

    st.write(
        "The algorithms use identical standardized features. "
        "Adjust K and DBSCAN settings in the sidebar to explore "
        "how the results change."
    )

    st.markdown("### Evaluation Results")

    display_comparison = comparison.copy()

    for column in [
        "Silhouette Score",
        "Davies-Bouldin Index",
        "Calinski-Harabasz Score",
    ]:
        display_comparison[column] = display_comparison[column].round(4)

    st.dataframe(
        display_comparison,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Silhouette: higher is better. Davies–Bouldin: lower is better. "
        "Calinski–Harabasz: higher is better. "
        "A dash/NaN means the metric could not be calculated reliably."
    )

    valid_silhouette = comparison.dropna(subset=["Silhouette Score"])

    if not valid_silhouette.empty:
        best_row = valid_silhouette.loc[
            valid_silhouette["Silhouette Score"].idxmax()
        ]
        st.success(
            f"Highest Silhouette Score in this run: {best_row['Model']} "
            f"({best_row['Silhouette Score']:.4f})."
        )
        st.write(
            "This is a metric-based result, not proof that the model is "
            "best for every business objective. Review all metrics and "
            "the usefulness of the resulting customer groups."
        )
    else:
        st.warning(
            "No model produced a valid Silhouette Score for this run. "
            "Try adjusting DBSCAN parameters."
        )

    metric_choice = st.selectbox(
        "Visualize metric",
        [
            "Silhouette Score",
            "Davies-Bouldin Index",
            "Calinski-Harabasz Score",
        ],
    )

    chart_data = comparison.dropna(subset=[metric_choice])

    if not chart_data.empty:
        fig = px.bar(
            chart_data,
            x="Model",
            y=metric_choice,
            color="Model",
            text_auto=".4f",
            title=f"Model Comparison: {metric_choice}",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No valid values are available for this metric.")

    with st.expander("How should I interpret these results?"):
        st.write(
            "- K-Means: forms groups around centroids.\n"
            "- Agglomerative: merges nearby groups hierarchically.\n"
            "- DBSCAN: finds dense groups and labels noise as -1.\n\n"
            "DBSCAN metrics are calculated on non-noise records only. "
            "Its scores therefore use a different evaluated subset when "
            "noise is present; compare them with that limitation in mind."
        )


# -------------------- CUSTOMER SEGMENTS --------------------

elif page == "Customer Segments":
    st.subheader("Visualize Customer Groups")

    chosen_model = st.selectbox(
        "Choose clustering algorithm",
        ["K-Means", "Agglomerative", "DBSCAN"],
    )

    result_df = df.copy()
    result_df["Cluster"] = labels_by_model[chosen_model].astype(str)

    result_df["Cluster"] = result_df["Cluster"].replace(
        {"-1": "Noise"}
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(
            result_df,
            x="Annual Income (k$)",
            y="Spending Score (1-100)",
            color="Cluster",
            hover_data=[
                column for column in ["CustomerID", "Age", "Gender"]
                if column in result_df.columns
            ],
            title=f"{chosen_model}: Income vs Spending",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            result_df,
            x="Age",
            y="Spending Score (1-100)",
            color="Cluster",
            title=f"{chosen_model}: Age vs Spending",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Segment Profiles")

    profile = result_df[result_df["Cluster"] != "Noise"].groupby(
        "Cluster", observed=True
    )[FEATURES].mean().round(2)

    st.dataframe(profile, use_container_width=True)

    sizes = result_df["Cluster"].value_counts().rename_axis(
        "Cluster"
    ).reset_index(name="Customers")

    st.markdown("### Customers per Group")
    fig = px.bar(
        sizes,
        x="Cluster",
        y="Customers",
        color="Cluster",
        text="Customers",
        title="Cluster Sizes",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(template="plotly_white", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Download Segmented Customer Data",
        data=result_df.to_csv(index=False).encode("utf-8"),
        file_name=f"customer_segments_{chosen_model.lower()}.csv",
        mime="text/csv",
    )


# -------------------- NEW CUSTOMER --------------------

elif page == "New Customer":
    st.subheader("Assign a New Customer to a K-Means Cluster")

    st.write(
        "Enter numeric customer attributes. The existing dataset's "
        "scaler and a K-Means model fitted on the existing customers "
        "will be used to assign a cluster."
    )

    with st.form("new_customer_form"):
        age = st.number_input(
            "Age",
            min_value=1,
            max_value=100,
            value=30,
        )
        income = st.number_input(
            "Annual Income (k$)",
            min_value=0.0,
            max_value=1000.0,
            value=60.0,
        )
        spending = st.slider(
            "Spending Score (1-100)",
            min_value=1,
            max_value=100,
            value=50,
        )

        submitted = st.form_submit_button("Assign Customer Cluster")

    if submitted:
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )
        model.fit(X)

        new_customer = pd.DataFrame(
            [[age, income, spending]],
            columns=FEATURES,
        )
        scaled_customer = scaler.transform(new_customer)
        predicted_cluster = int(model.predict(scaled_customer)[0])

        st.success(f"Assigned to Customer Cluster {predicted_cluster}")

        st.write(
            "This cluster number identifies a group, not a ranking. "
            "Review the Customer Segments page to understand the group's "
            "average age, income, and spending behaviour."
        )

        st.caption(
            "This is an unsupervised cluster assignment, not a prediction "
            "of spending, revenue, or future customer behaviour."
        )


# -------------------- FOOTER --------------------

st.divider()
st.caption(
    "Customer Segmentation Using Clustering | "
    "K-Means • Agglomerative Clustering • DBSCAN"
)
