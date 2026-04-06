import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add category column
fake['category'] = "Fake"
true['category'] = "Real"

# Combine datasets
data = pd.concat([fake, true])

# Features & target
X = data['text']
y = data['category']

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("Fake News Detection App")

user_input = st.text_area("Enter News Text")

if st.button("Predict"):
    if user_input.strip() != "":
        input_data = vectorizer.transform([user_input])
        prediction = model.predict(input_data)

        st.write("Result:", prediction[0])
    else:
        st.warning("Please enter news text.")
