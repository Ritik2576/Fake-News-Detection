# Fake-News-Detection
A machine learning model for detecting fake news using DistilBERT. The system classifies news articles as Real or Fake based on content and context.
Fake News Detection using Machine Learning
🚀 Project Overview
This repository contains a Fake News Detection system built with Machine Learning using DistilBERT. The goal of this project is to automatically classify news articles into two categories: Real News and Fake News.
0
With the rise of misinformation and fake news on social media and news platforms, this tool aims to provide a reliable way to verify news articles, helping users filter out unverified content.

The model is trained using a large dataset of over 60,000 news articles, including data from Kaggle and Indian news sources. It leverages the DistilBERT architecture, a smaller and faster version of BERT, for high-quality text classification.

📋 Table of Contents
Project Description

Model Architecture

Technologies Used

How It Works

Setup & Installation

Training the Model

API Endpoint

Demo

Feedback & Contributions

📖 Project Description
This project utilizes DistilBERT, a transformer-based language model, to detect fake news articles based on their content. The model has been trained with data from multiple sources, including Kaggle's fake and true news datasets and Indian news data.

Features:
Classifies news articles as either Real or Fake.

User-friendly web interface to paste news and get the classification result.

Feedback collection from users to improve the model's predictions.

⚙️ Model Architecture
The model used in this project is based on DistilBERT (a distilled version of BERT) which is trained to predict fake and real news articles:

DistilBERT: A smaller, faster, and lighter version of the BERT model that maintains 97% of BERT’s performance while being 60% faster and requiring 40% fewer parameters.

💻 Technologies Used
Python

Flask: For serving the model via an API.

PyTorch: For training the model.

Hugging Face Transformers: For accessing DistilBERT and training it on the dataset.

Scikit-Learn: For data preprocessing and model evaluation.

Bootstrap 5: For building the responsive web UI.

NewsAPI: To fetch the latest news articles for real-time testing.

GitHub: Version control and collaboration.

🚀 How It Works
Training Phase:

We used a large dataset that includes True and Fake news articles.

The dataset was preprocessed and tokenized using DistilBERT's tokenizer.

The model was fine-tuned on this dataset to distinguish between fake and real news.

Prediction Phase:

Users paste a news article's content into the web application.

The application sends the news to a Flask-based API, which passes it through the trained DistilBERT model.

The result is classified as either Real News or Fake News, based on the model's prediction.

Feedback:

Users can submit feedback on the predictions, allowing us to continuously improve the model.

🛠️ Setup & Installation
Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/Fake-News-Detection.git
cd Fake-News-Detection
Install Required Dependencies
Make sure you have Python 3.7+ installed. You can create a virtual environment (optional but recommended).

bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
🧑‍💻 Training the Model
The model is trained on the dataset using DistilBERT and PyTorch. To train the model, you can run the following script:

bash
Copy
Edit
python train_model.py
This script will:

Load and preprocess the data.

Train the DistilBERT model.

Save the model and tokenizer for future use.

🌐 API Endpoint
The Flask application provides a simple API endpoint to check news articles. The main endpoint is:

POST /predict

Input: JSON with the key text containing the news article content.

Output: A JSON with the result of the prediction: Real News or Fake News.

Example request:

bash
Copy
Edit
curl -X POST -H "Content-Type: application/json" \
  -d '{"text": "Is the Earth flat?"}' \
  http://127.0.0.1:5000/predict
🎬 Demo
You can try out the Fake News Detection system directly by running the Flask app locally:

bash
Copy
Edit
python app.py
Visit http://127.0.0.1:5000 in your browser and paste a news article into the input box to check whether it's real or fake.

💬 Feedback & Contributions
We welcome contributions to this project! If you have ideas for improving the system or want to help enhance the model, feel free to open an issue or create a pull request.

How to Contribute:
Fork the repository.

Create a new branch (git checkout -b feature-name).

Make your changes.

Commit your changes (git commit -am 'Add new feature').

Push to the branch (git push origin feature-name).

Open a pull request.

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

✨ Acknowledgements
The DistilBERT model by Hugging Face.

Kaggle for providing datasets to train the model.

Flask for creating the API and web server.

