# Hyperlocal-News-Anomaly-Detection

🧠 Hyperlocal News Anomaly Detection and Source Attribution

🔍 Overview

This project detects anomalies in hyperlocal news articles using advanced Natural Language Processing (NLP) and Machine Learning models. It leverages BERT and 
RoBERTa embeddings for contextual understanding, combined with Isolation Forest and Autoencoder models for anomaly detection.

An interactive Streamlit dashboard enables users to visualize anomaly patterns, sentiment trends, and geographic source attributions in real time.

🚀 Features

🧩 Anomaly Detection: Detect misleading or misattributed regional news.

🌍 Source Attribution: Predict article origin using BERT-based classifiers.

💬 Sentiment Analysis: Identify article tone (positive, negative, neutral).

📊 Interactive Dashboard: Explore anomalies, sentiments, and insights visually.

⚙️ Model Evaluation: View accuracy, precision, recall, and F1 metrics.

🧰 Tech Stack

Languages: Python

Frameworks & Libraries: Streamlit, Pandas, Plotly, Scikit-learn, PyTorch, TensorFlow

NLP Models: BERT, RoBERTa (via Hugging Face Transformers)

Visualization: Plotly, Streamlit

Version Control: Git, GitHub

📂 Project Structure
├── notebooks/     
# Model training & data processing notebooks  
├── new.py   
# Streamlit dashboard app  
├── requirements.txt 
# Dependencies list  
├── processed_news.csv 
# Preprocessed dataset  
├── README.md  
# Project overview  
└── docs/   
# Supporting documentation and screenshots

📈 Model Performance

Metric	Value

Accuracy	75.06%

Precision	0.72

Recall	0.75

F1-Score	0.73

🔮 Future Enhancements

Multilingual support with mBERT / XLM-R

Real-time anomaly monitoring with live news feeds

Automated retraining pipelines

Integration with government/media verification systems

📘 References

BERT Paper (Devlin et al., 2019)

Hugging Face Transformers

Streamlit Documentation

Plotly Documentation
