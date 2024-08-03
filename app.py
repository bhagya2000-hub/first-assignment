import streamlit as st
import time
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Set the title of the application
st.title("Streamlit Machine Learning App")

# Load data
st.subheader("Step 1: Load Data")
with st.spinner("Loading data..."):
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    time.sleep(2)
st.success("Data loaded successfully!")

# Display the data
st.write("Here's a preview of the dataset:")
st.dataframe(df.head())

# Split data
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
st.subheader("Step 2: Train Model")
progress_bar = st.progress(0)
clf = RandomForestClassifier()

with st.spinner("Training model..."):
    for i in range(1, 101):
        time.sleep(0.05)
        progress_bar.progress(i)
    clf.fit(X_train, y_train)
st.success("Model trained successfully!")

# Predictions
st.subheader("Step 3: Make Predictions")
st.write("Let's make predictions using the trained model.")

# Input features for prediction
sepal_length = st.slider("Sepal Length", float(X.min()[0]), float(X.max()[0]), float(X.mean()[0]))
sepal_width = st.slider("Sepal Width", float(X.min()[1]), float(X.max()[1]), float(X.mean()[1]))
petal_length = st.slider("Petal Length", float(X.min()[2]), float(X.max()[2]), float(X.mean()[2]))
petal_width = st.slider("Petal Width", float(X.min()[3]), float(X.max()[3]), float(X.mean()[3]))

# Make prediction
prediction_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
with st.spinner("Predicting..."):
    prediction = clf.predict(prediction_input)
    time.sleep(2)

# Display prediction
st.success(f"The predicted class is: {data.target_names[prediction][0]}")

# Visualize
st.subheader("Visualize Predictions")
st.write("Visualizing the prediction with a bar chart:")

# Bar chart for prediction confidence
prediction_proba = clf.predict_proba(prediction_input)[0]
st.bar_chart(pd.DataFrame({
    "Species": data.target_names,
    "Probability": prediction_proba
}).set_index("Species"))

st.write("All tasks completed successfully!")
