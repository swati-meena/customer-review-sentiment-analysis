import streamlit as st
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
dataset = pd.read_csv(r"/home/hp/Downloads/november/5th, 6th - NLP project/4.CUSTOMERS REVIEW DATASET/Restaurant_Reviews.tsv", delimiter='\t', quoting=3)

# Text Preprocessing Function
nltk.download('stopwords')

def preprocess_text(review):
    ps = PorterStemmer()
    review = re.sub('[^a-zA-Z]', ' ', review)  # Remove special characters
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    return ' '.join(review)

# Prepare Data
corpus = [preprocess_text(review) for review in dataset['Review']]
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Train Model
classifier = RandomForestClassifier(n_estimators=100, random_state=0)
classifier.fit(X, y)

# Save the Model and Vectorizer
joblib.dump(classifier, "sentiment_model.pkl")
joblib.dump(cv, "vectorizer.pkl")

# Load Model
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit UI
st.title("Restaurant Review Sentiment Analysis")
st.write("Enter a restaurant review to predict its sentiment.")

# User Input
user_input = st.text_area("Enter your review here:", "")

if st.button("Predict Sentiment"):
    if user_input:
        processed_review = preprocess_text(user_input)
        review_vector = vectorizer.transform([processed_review]).toarray()
        prediction = model.predict(review_vector)

        if prediction[0] == 1:
            st.success("Positive Review 😊")
        else:
            st.error("Negative Review 😞")
    else:
        st.warning("Please enter a review.")

