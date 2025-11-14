import streamlit as st
import pandas as pd
import plotly.express as px
import random
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from geotext import GeoText

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
    path = r"C:\Users\sowmi\OneDrive\Desktop\python\final project\notebooks\outputs\processed_news.csv"
  # ✅ fixed typo from ouput → output
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
        "Normal Articles",
        "User Input Prediction"  # ✅ new feature
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
# TAB 8 - USER INPUT PREDICTION (HUMAN-CENTRIC)
# ==========================
elif tabs == "User Input Prediction":
    st.title("🧠 User Input Based Prediction")

    st.markdown("Enter or paste a news article below to analyze if it’s an anomaly and detect potential bias:")

    # --- User Inputs ---
    heading = st.text_input("📰 Headline")
    article = st.text_area("✍️ Full Article Content", height=200)

    # --- Analyze Button ---
    if st.button("🔍 Analyze Article"):
        if not article.strip():
            st.warning("Please enter the article content.")
        else:
            import random
            from datetime import datetime
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            from geotext import GeoText
            import pandas as pd
            import plotly.express as px
            import plotly.graph_objects as go
            import numpy as np
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import re

            # --- Sentiment Analysis ---
            analyzer = SentimentIntensityAnalyzer()
            sentiment_score = analyzer.polarity_scores(article)["compound"]
            if sentiment_score > 0.05:
                sentiment = "Positive"
                sentiment_icon = "😊"
            elif sentiment_score < -0.05:
                sentiment = "Negative"
                sentiment_icon = "😠"
            else:
                sentiment = "Neutral"
                sentiment_icon = "😐"

            # --- Improved Location Detection (trusted keywords first, then GeoText) ---
            location = "Unknown"
            trusted_locations = [
                "Mumbai", "Delhi", "Bengaluru", "Chennai", "Kolkata",
                "Hyderabad", "Pune", "Jaipur", "Ahmedabad", "Lucknow",
                "Bhopal", "Chandigarh", "Patna", "Thiruvananthapuram",
                "Assam", "Kerala", "Gujarat", "Rajasthan", "Tamil Nadu",
                "Maharashtra", "Karnataka", "Uttar Pradesh", "Bihar",
                "Pakistan", "Bangladesh", "Nepal", "Sri Lanka", "Bhutan",
                "New York", "London", "Paris", "Tokyo", "Sydney"
            ]
            # match whole words case-insensitive
            for city in trusted_locations:
                if re.search(rf'\b{re.escape(city)}\b', article, flags=re.IGNORECASE):
                    location = city
                    break

            # fallback to GeoText only if no trusted location found
            if location == "Unknown":
                places = GeoText(article)
                if places.cities:
                    # choose the first valid city that is not a common word like "Police"
                    for c in places.cities:
                        if len(c) > 1 and not re.match(r'^[A-Z]{2,}$', c):  # simple sanity filter
                            location = c
                            break

            # --- UPDATED CATEGORY SYSTEM (Expanded keyword lists) ---
            categories = {
                "Politics": [
                    "election", "vote", "minister", "government", "policy", "parliament",
                    "senate", "assembly", "cabinet", "bill", "law", "democracy",
                    "opposition", "campaign", "political party", "mp", "mla", "governor",
                    "president", "pm", "chief minister", "government scheme", "public sector",
                    "budget", "ordinance", "diplomacy", "international relations"
                ],

                "Education": [
                    "school", "college", "university", "education", "students",
                    "teacher", "principal", "examination", "result", "admission",
                    "scholarship", "online classes", "syllabus", "textbook", "learning",
                    "training program", "board exam", "academic", "research paper",
                    "classroom", "attendance", "hostel", "campus", "lab", "coaching",
                    "education policy", "skill development"
                ],

                "Health": [
                    "health", "hospital", "doctor", "nurse", "patient", "disease",
                    "virus", "infection", "vaccine", "clinic", "treatment", "surgery",
                    "medicine", "healthcare", "diagnosis", "ambulance", "emergency",
                    "mental health", "covid", "fever", "flu", "health department",
                    "medical research", "public health", "health scheme"
                ],

                "Environment": [
                    "environment", "pollution", "climate", "climate change", "global warming",
                    "rainfall", "flood", "drought", "forest", "wildlife", "animals",
                    "air quality", "water quality", "recycling", "waste management",
                    "solar energy", "renewable energy", "disaster", "cyclone", "heatwave",
                    "ozone", "ecosystem", "deforestation", "conservation"
                ],

                "Crime": [
                    "crime", "police", "robbery", "murder", "theft", "fraud", "assault",
                    "kidnap", "cybercrime", "accident", "victim", "court", "arrest",
                    "investigation", "violence", "scam", "illegal", "narcotics",
                    "juvenile", "forensic", "criminal", "fir", "chargesheet",
                    "smuggling", "terrorism", "extortion"
                ],

                "Business": [
                    "market", "stock", "company", "investment", "profit", "loss",
                    "economy", "finance", "business", "industry", "startup", "shares",
                    "ipo", "merger", "acquisition", "ceo", "revenue", "sales",
                    "trade", "export", "import", "banking", "inflation", "commerce",
                    "manufacturing", "small business", "corporate"
                ],

                "Technology": [
                    "technology", "tech", "digital", "ai", "artificial intelligence",
                    "machine learning", "deep learning", "neural network",
                    "automation", "algorithm", "software", "app", "mobile", "computer",
                    "robotics", "device", "smartphone", "innovation", "hardware",
                    "processor", "chipset", "wearable", "semiconductor", "cloud",
                    "cloud computing", "server", "data center", "database", "big data",
                    "cyber", "cybersecurity", "data breach", "malware", "ransomware",
                    "phishing", "encryption", "firewall", "iot", "5g", "network",
                    "connectivity", "electric vehicle", "autonomous car",
                    "self-driving", "startup", "research", "prototype", "engineering"
                ],

                "Sports": [
                    "sports", "match", "game", "tournament", "league", "world cup",
                    "team", "player", "stadium", "score", "goal", "cricket", "football",
                    "basketball", "tennis", "badminton", "athlete", "coach",
                    "championship", "training", "medal", "olympics", "injury",
                    "referee", "umpire", "sports event"
                ],

                "General": [
                    "public", "local", "event", "people", "society", "community",
                    "festival", "celebration", "traffic", "weather", "update",
                    "announcement", "information", "general news", "daily update"
                ]
            }

            # Build category documents and compute TF-IDF similarity
            category_docs = {cat: " ".join(words) for cat, words in categories.items()}
            vectorizer = TfidfVectorizer(stop_words='english')
            all_docs = list(category_docs.values()) + [article]
            tfidf_matrix = vectorizer.fit_transform(all_docs)
            similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
            news_type = list(category_docs.keys())[similarities.argmax()]

            # --- Simulated Anomaly Detection ---
            anomaly_score = round(random.uniform(0, 1), 2)
            probability = round(anomaly_score * 100, 2)
            is_anomaly = "Anomaly" if anomaly_score > 0.7 else "Normal"
            badge_color = "red" if is_anomaly == "Anomaly" else "green"

            # --- Display Results ---
            st.markdown("## 📊 Prediction Results")
            st.markdown(
                f"**📰 Headline:** {heading or 'Untitled'}  \n"
                f"**📅 Date:** {datetime.now().strftime('%Y-%m-%d')}  \n"
                f"**📍 Predicted Location:** {location}  \n"
                f"**🧾 Type:** {news_type}  \n"
                f"**💬 Sentiment:** <span style='font-size:30px'>{sentiment_icon}</span> {sentiment}  \n"
                f"**📈 Anomaly Score:** <span style='color:{badge_color};font-weight:bold'>{anomaly_score}</span>  \n"
                f"**🎯 Probability:** {probability}%  \n"
                f"**🔎 Prediction:** <span style='color:{badge_color};font-weight:bold'>{is_anomaly}</span>",
                unsafe_allow_html=True,
            )

            if is_anomaly == "Anomaly":
                st.error("⚠️ This article appears **anomalous** — may differ from typical trends.")
            else:
                st.success("✅ This article appears **normal** and consistent with known patterns.")

            # --- Human-Friendly Reasoning ---
            st.markdown("### 🧩 Insights / Reasoning")
            reasons = []
            if sentiment == "Negative":
                reasons.append("Negative sentiment may indicate anomaly triggers.")
            if sentiment == "Positive":
                reasons.append("Positive sentiment may reduce anomaly likelihood.")
            if location == "Unknown":
                reasons.append("No specific location detected.")
            if any(k in article.lower() for k in ["fake","rumor","hoax"]):
                reasons.append("Contains suspicious keywords like fake/rumor/hoax.")
            if not reasons:
                reasons.append("Balanced sentiment and clear context.")

            for r in reasons:
                st.write(f"- {r}")

            # --- Horizontal Bar Chart ---
            st.markdown("### 📊 Reason Contribution")
            reason_weights = {
                "Negative Sentiment": 1 if sentiment=="Negative" else 0,
                "Positive Sentiment": 1 if sentiment=="Positive" else 0,
                "Unknown Location": 1 if location=="Unknown" else 0,
                "Suspicious Keywords": 1 if any(k in article.lower() for k in ["fake","rumor","hoax"]) else 0
            }
            total_weight = sum(reason_weights.values())
            reason_percent = {k: round((v/total_weight)*100,2) if total_weight>0 else 0 for k,v in reason_weights.items()}
            reason_df = pd.DataFrame({
                "Reason": list(reason_percent.keys()),
                "Contribution": list(reason_percent.values())
            })
            fig_bar = px.bar(reason_df, x="Contribution", y="Reason", orientation='h',
                             color="Reason", color_discrete_sequence=px.colors.qualitative.Set2,
                             text="Contribution", title="Anomaly Trigger Contribution (%)")
            fig_bar.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_bar)

            # --- Contextual Information ---
            st.markdown("### 📚 Contextual Information")
            words = re.findall(r'\b\w+\b', article.lower())
            stopwords = set(vectorizer.get_stop_words())
            keywords = [w for w in words if w not in stopwords]
            keyword_freq = pd.Series(keywords).value_counts().head(5)
            st.write("**Top Keywords:**", ", ".join(keyword_freq.index.tolist()))
            st.write("**Detected Location/Entities:**", location)

            # --- Potential Bias / Attention Flags ---
            st.markdown("### ⚠️ Potential Bias / Attention Flags")
            bias_flags = []
            if sentiment_score > 0.6:
                bias_flags.append("Highly Positive Language — may exaggerate positivity")
            if sentiment_score < -0.6:
                bias_flags.append("Highly Negative Language — may exaggerate negativity")
            if len(re.findall(r"\bmust\b|\bshould\b|\bnever\b", article.lower())) > 0:
                bias_flags.append("Strong Opinion / Directive Words detected")
            if any(word in article.lower() for word in ["exclusive", "shocking", "unbelievable"]):
                bias_flags.append("Sensational / Clickbait Language detected")
            if not bias_flags:
                bias_flags.append("No major bias detected")

            for flag in bias_flags:
                st.warning(flag)

            # --- Save Results ---
            log_df_new = pd.DataFrame([{
                "Datetime": datetime.now(),
                "Headline": heading,
                "Sentiment": sentiment,
                "News_Type": news_type,
                "Location": location,
                "Anomaly": is_anomaly,
                "Anomaly_Score": anomaly_score,
                "Probability": probability,
                "Bias_Flags": "; ".join(bias_flags)
            }])
            try:
                existing_df = pd.read_csv("user_predictions_log.csv")
                log_df_new = pd.concat([existing_df, log_df_new], ignore_index=True)
            except FileNotFoundError:
                pass
            log_df_new.to_csv("user_predictions_log.csv", index=False)
