import streamlit as st
import pandas as pd
import plotly.express as px

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="🧠 News Anomaly Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# DATA LOADING
# ==========================
@st.cache_data
def load_data():
    # Changed path from local Windows absolute path to a GitHub-relative path
    path = "notebooks/outputs/processed_news.csv" 
    
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["AnomalyFlag"] = df["AnomalyFlag"].astype(str).str.strip().fillna("Normal")
    df["Anomaly_Flag"] = df["AnomalyFlag"].apply(
        lambda x: 1 if str(x).lower() in ["1", "anomaly", "true", "yes"] else 0
    )
    return df

df = load_data()

# ==========================
# SIDEBAR NAVIGATION
# ==========================
st.sidebar.title("📊 Dashboard Navigation")
tabs = st.sidebar.radio(
    "Choose a Section",
    [
        "Project Overview",
        "Data Overview",
        "Visual Insights",
        "Model Performance",
        "Read Articles",
        "Anomalous Articles",
        "Normal Articles"
    ]
)

# ==========================
# TAB 1 - PROJECT OVERVIEW
# ==========================
if tabs == "Project Overview":
    st.title("🧠 Hyperlocal News Anomaly Detection and Source Attribution")

    st.markdown("""
    ### 🎯 Objective
    Detect anomalous or misleading patterns in hyperlocal news by comparing linguistic, sentiment, and location-based cues using advanced NLP models such as BERT and RoBERTa.

    ### 🧩 Business Use Cases
    - **Disinformation Detection:** Identify misattributed or fake news.  
    - **Hyperlocal Trend Monitoring:** Detect sentiment and topic shifts in regions.  
    - **Brand Reputation:** Spot regional brand anomalies.  
    - **Automated Content Verification:** Flag suspicious content automatically.

    ### ⚙️ Approach
    1. **Preprocessing & Location Extraction:** Clean, lemmatize, and extract geolocations using NER.  
    2. **Embedding Generation:** Use BERT and RoBERTa for contextual embeddings.  
    3. **Anomaly Detection:** Apply Isolation Forest / Autoencoders.  
    4. **Source Attribution:** Predict most likely origin location using BERT classifier.  
    5. **Visualization:** Interactive Streamlit dashboard with anomaly summaries, sentiment charts, and article reading section.

    ### 🧰 Technologies Used
    Python, scikit-learn, TensorFlow, PyTorch, Transformers (BERT/RoBERTa), Pandas, Plotly, Streamlit, AWS/GCP Hosting
    """)

# ==========================
# TAB 2 - DATA OVERVIEW
# ==========================
elif tabs == "Data Overview":
    st.title("📈 Data Overview")

    total_anomalies = df["Anomaly_Flag"].sum()
    total_normal = len(df) - total_anomalies
    anomaly_percent = (total_anomalies / len(df)) * 100

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("⚠️ Total Anomalies", total_anomalies)
    with col2:
        st.metric("📰 Normal Articles", total_normal)
    with col3:
        st.metric("📊 Anomaly Percentage", f"{anomaly_percent:.2f}%")

    st.markdown("### 📂 Data Sample")
    st.dataframe(df.head(10))

    st.markdown("### 🧾 Dataset Columns Description")
    st.write("""
    - **Heading:** News headline  
    - **Article:** Full news content  
    - **Date:** Publication date  
    - **NewsType:** Category of news  
    - **Sentiment:** Computed polarity  
    - **AnomalyFlag:** Indicates if flagged as anomaly  
    - **Predicted_Location_BERT:** BERT-predicted region  
    - **anomaly_score:** Numeric anomaly score  
    """)

# ==========================
# TAB 3 - VISUAL INSIGHTS
# ==========================
elif tabs == "Visual Insights":
    st.title("📊 Visual Insights & Analytics")

    # --- Anomaly vs Normal ---
    st.subheader("1️⃣ Anomaly vs Normal Distribution")
    counts = df["Anomaly_Flag"].value_counts().rename({0: "Normal", 1: "Anomaly"}).reset_index()
    counts.columns = ["Type", "Count"]

    col1, col2 = st.columns(2)
    with col1:
        bar_fig = px.bar(counts, x="Type", y="Count", color="Type", title="Anomaly vs Normal (Bar Chart)")
        st.plotly_chart(bar_fig, use_container_width=True)
    with col2:
        pie_fig = px.pie(counts, names="Type", values="Count", title="Anomaly vs Normal (Pie Chart)")
        st.plotly_chart(pie_fig, use_container_width=True)

    # --- Sentiment Distribution ---
    st.subheader("2️⃣ Sentiment Distribution")
    if "Sentiment" in df.columns:
        sentiment_df = df["Sentiment"].value_counts().reset_index()
        sentiment_df.columns = ["Sentiment", "Count"]
        sentiment_fig = px.bar(
            sentiment_df,
            x="Sentiment",
            y="Count",
            color="Sentiment",
            title="Sentiment Distribution",
        )
        st.plotly_chart(sentiment_fig, use_container_width=True)

    # --- Top News Types ---
    st.subheader("3️⃣ Top News Types")
    if "NewsType" in df.columns:
        type_df = df["NewsType"].value_counts().reset_index().head(10)
        type_df.columns = ["NewsType", "Count"]
        type_fig = px.bar(type_df, x="NewsType", y="Count", color="NewsType", title="Top 10 News Types")
        st.plotly_chart(type_fig, use_container_width=True)

    # --- Top Locations ---
    st.subheader("4️⃣ Top 10 Predicted Locations (BERT)")
    if "Predicted_Location_BERT" in df.columns:
        loc_df = df["Predicted_Location_BERT"].value_counts().reset_index().head(10)
        loc_df.columns = ["Location", "Count"]
        loc_fig = px.bar(loc_df, x="Location", y="Count", color="Location", title="Top 10 Predicted Locations")
        st.plotly_chart(loc_fig, use_container_width=True)

# ==========================
# TAB 4 - MODEL PERFORMANCE
# ==========================
elif tabs == "Model Performance":
    st.title("🤖 Model Performance Metrics")

    st.write("""
    ### 🧠 Models Used
    - **BERT Location Classifier** – Predicts likely article origin  
    - **Isolation Forest** – Detects linguistic anomalies  
    - **RoBERTa Sentiment Model** – Generates article polarity  

    ### 📈 Evaluation Metrics
    | Metric | Value |
    |---------|--------|
    | Accuracy | 75.06% |
    | Precision | 0.72 |
    | Recall | 0.75 |
    | F1-Score | 0.73 |

    ### 🧩 Model Insights
    - The **BERT model** shows strong generalization on dominant regions like Pakistan and India.  
    - **Linguistic anomalies** tend to correlate with articles showing mismatched sentiment and topic distribution.  
    - **Source discrepancy** detection is consistent for multilingual articles.  
    """)

# ==========================
# TAB 5 - READ ARTICLES
# ==========================
elif tabs == "Read Articles":
    st.title("📰 Browse All Articles")

    selected_heading = st.selectbox("Select an Article to Read:", df["Heading"].dropna().unique())
    selected_article = df[df["Heading"] == selected_heading].iloc[0]

    st.subheader(selected_article["Heading"])
    st.write(f"**Date:** {selected_article['Date']}")
    st.write(f"**Type:** {selected_article['NewsType']}")
    st.write(f"**Sentiment:** {selected_article['Sentiment']}")
    st.write(f"**Location:** {selected_article['Predicted_Location_BERT']}")
    st.write(f"**Anomaly Status:** {selected_article['AnomalyFlag']}")
    st.markdown("---")
    st.write(selected_article["Article"])

# ==========================
# TAB 6 - ANOMALOUS ARTICLES
# ==========================
elif tabs == "Anomalous Articles":
    st.title("🚨 Anomalous Articles")

    df_anomaly = df[df["Anomaly_Flag"] == 1]
    st.write(f"Total Anomalies: {len(df_anomaly)}")

    if len(df_anomaly) > 0:
        selected_heading = st.selectbox("Select an Anomalous Article:", df_anomaly["Heading"].dropna().unique())
        selected_article = df_anomaly[df_anomaly["Heading"] == selected_heading].iloc[0]
        st.subheader(selected_article["Heading"])
        st.write(f"**Date:** {selected_article['Date']}")
        st.write(f"**Sentiment:** {selected_article['Sentiment']}")
        st.write(f"**Anomaly Score:** {selected_article['anomaly_score']}")
        st.markdown("---")
        st.write(selected_article["Article"])
    else:
        st.info("No anomalous articles found.")

# ==========================
# TAB 7 - NORMAL ARTICLES
# ==========================
elif tabs == "Normal Articles":
    st.title("✅ Normal Articles")

    df_normal = df[df["Anomaly_Flag"] == 0]
    st.write(f"Total Normal Articles: {len(df_normal)}")

    if len(df_normal) > 0:
        selected_heading = st.selectbox("Select a Normal Article:", df_normal["Heading"].dropna().unique())
        selected_article = df_normal[df_normal["Heading"] == selected_heading].iloc[0]
        st.subheader(selected_article["Heading"])
        st.write(f"**Date:** {selected_article['Date']}")
        st.write(f"**Sentiment:** {selected_article['Sentiment']}")
        st.markdown("---")
        st.write(selected_article["Article"])
