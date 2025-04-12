# **Fake News Detection using Machine Learning**

### 🚀 **Project Overview**
**This repository contains a Fake News Detection system built with Machine Learning using DistilBERT. The goal of this project is to automatically classify news articles into two categories: Real News and Fake News.**

**With the rise of misinformation and fake news on social media and news platforms, this tool aims to provide a reliable way to verify news articles, helping users filter out unverified content.**

**The model is trained using a large dataset of over 60,000 news articles, including data from Kaggle and Indian news sources. It leverages the DistilBERT architecture, a smaller and faster version of BERT, for high-quality text classification.**

### 📋 **Table of Contents**
- **[Project Description](#project-description)**
- **[Model Architecture](#model-architecture)**
- **[Technologies Used](#technologies-used)**
- **[How It Works](#how-it-works)**
- **[Setup & Installation](#setup--installation)**
- **[Training the Model](#training-the-model)**
- **[API Endpoint](#api-endpoint)**
- **[Demo](#demo)**
- **[Feedback & Contributions](#feedback--contributions)**

### 📖 **Project Description**
**This project utilizes DistilBERT, a transformer-based language model, to detect fake news articles based on their content. The model has been trained with data from multiple sources, including Kaggle's fake and true news datasets and Indian news data.**

#### **Features:**
- **Classifies news articles as either Real or Fake.**
- **User-friendly web interface to paste news and get the classification result.**
- **Feedback collection from users to improve the model's predictions.**

---

### ⚙️ **Model Architecture**
**The model used in this project is based on DistilBERT (a distilled version of BERT) which is trained to predict fake and real news articles:**

- **DistilBERT**: **A smaller, faster, and lighter version of the BERT model that maintains 97% of BERT’s performance while being 60% faster and requiring 40% fewer parameters.**

---

### 💻 **Technologies Used**
- **Python**
- **Flask**: **For serving the model via an API.**
- **PyTorch**: **For training the model.**
- **Hugging Face Transformers**: **For accessing DistilBERT and training it on the dataset.**
- **Scikit-Learn**: **For data preprocessing and model evaluation.**
- **Bootstrap 5**: **For building the responsive web UI.**
- **NewsAPI**: **To fetch the latest news articles for real-time testing.**
- **GitHub**: **Version control and collaboration.**

---

### 🚀 **How It Works**
1. **Training Phase**:
   - **We used a large dataset that includes True and Fake news articles.**
   - **The dataset was preprocessed and tokenized using DistilBERT's tokenizer.**
   - **The model was fine-tuned on this dataset to distinguish between fake and real news.**

2. **Prediction Phase**:
   - **Users paste a news article's content into the web application.**
   - **The application sends the news to a Flask-based API, which passes it through the trained DistilBERT model.**
   - **The result is classified as either Real News or Fake News, based on the model's prediction.**

3. **Feedback**:
   - **Users can submit feedback on the predictions, allowing us to continuously improve the model.**

4. **Current News**
 -**Users can see the latest real-time news fetched from a news API and check whether the articles are real or fake using the model.**


