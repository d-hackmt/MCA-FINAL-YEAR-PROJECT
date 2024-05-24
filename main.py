import streamlit as st
st.set_page_config(layout="wide")
from util import set_background
import random
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

extra_trees_model_path = "models/extra_trees_model.pkl"
random_forest_model_path = "models/random_forest_model.pkl"
transformer_model_path = "models/transformer_model.h5"

# Load the CSV file containing combined questions and answers
csv_file = "Thunderstrom_Project_Questions_Answers.csv"
df = pd.read_csv(csv_file)

# Set the title
# st.title("Music Generation & LLM Chatbot")
def preprocess_and_predict(model, geopotential, specific_humidity, air_temperature, eastward_wind, northward_wind):
    # Convert inputs to a numpy array and reshape it
    inputs = np.array([
        float(geopotential), 
        float(specific_humidity), 
        float(air_temperature), 
        float(eastward_wind), 
        float(northward_wind)
    ]).reshape(1, -1)
    
    # Normalize inputs (assuming scaler is fitted on the same columns during training)
    scaler = MinMaxScaler()
    # Fit the scaler (normally you would save and load the scaler fitted on your training data)
    scaler.fit(inputs)
    inputs_scaled = scaler.transform(inputs)

    # Make prediction
    prediction = model.predict(inputs_scaled)
    return prediction

# Function to preprocess text data
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    return text

st.markdown(
    """
    <style>
    .centered-title {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.write("")
st.write("")
st.markdown('<h1 class="centered-title">Thunderstrom Forecasting </h1>', unsafe_allow_html=True)
st.markdown('<h1 class="centered-title">and Satellite Imagery</h1>', unsafe_allow_html=True)
st.write("")
st.write("")
st.write("")
st.write("")
st.markdown('<div style="text-align:center; font-size: xx-large;">'
            '<a href="https://docs.google.com/spreadsheets/d/1Q7fi6MRq-m0a8O8gS8Jfju1PrMLkTThM5MPURj-1RSM/edit?usp=sharing" style="margin: 0px 20px;">2D Dataset</a>'
            '<a href="https://drive.google.com/drive/folders/1ApxOrlXec4WZDfJx6G49c2r0PN50FJyI?usp=sharing" style="margin: 0px 20px;">Satellite Dataset</a>'
            '<a href="https://www.canva.com/design/DAGGEoR0wpk/_z_4xphpl9pTvKbi4LFg_A/edit?utm_content=DAGGEoR0wpk&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton" style="margin: 0px 20px;">LITERATURE SURVEY</a>'
            '<a href="https://www.canva.com/design/DAGGEspKsV4/xre--uL-oK5UwohLfEHefQ/edit?utm_content=DAGGEspKsV4&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton" style="margin: 0px 20px;">REPORT</a>'
            '</div>',
            unsafe_allow_html=True)
st.write("")
st.write("")
st.write("")
# Define the background image or set_background function if required

set_background('pix/3.png')

# Preprocess the questions
df['Question'] = df['Question'].apply(preprocess_text)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Question'])

col1, col2, col3 = st.columns(3)

# Left column for Chatbot
with col1:
    st.header("NLP Chatbot 🤖")
    query = st.text_area('ASK ME ANYTHING ABOUT THE PROJECT REPORT', height=70)
    submit = st.button("Ask")

    if submit:
        # Preprocess the user's query
        query = preprocess_text(query)

        # Transform the query using the TF-IDF vectorizer
        query_vectorized = vectorizer.transform([query])

        # Calculate cosine similarity between query and questions
        similarities = cosine_similarity(query_vectorized, X)

        # Get the index of the most similar question
        most_similar_index = similarities.argmax()

        # Retrieve the corresponding answer
        answer = df.loc[most_similar_index, 'Answer']

        # Display the answer
        st.write("Answer:", answer)

# Right column for Music Generation
with col2:
    st.header("Thunderstrom Forecasting ")
    # Add a button to generate a random seed from predefined seeds
    # Add a text input area for the seed
    geopotential = st.text_input("Geopotential ")
    specific_humidity = st.text_input("Specific Humidity", "")
    air_temperature = st.text_input("Air Temperature ", "")
    eastward_wind = st.text_input("Eastward Wind", "")
    northward_wind = st.text_input("Northward Wind ", "")

    # Buttons for prediction
    if st.button("Predict Using ML"):
        # Load the Extra Trees model
        extra_trees_model = joblib.load(extra_trees_model_path)
        # Make prediction
        prediction = preprocess_and_predict(extra_trees_model, geopotential, specific_humidity, air_temperature, eastward_wind, northward_wind)
        st.write(f"Prediction using Machine Learning : {prediction[0]}")

    if st.button("Predict Using Neural Network"):
        # Load the Random Forest model
        random_forest_model = joblib.load(random_forest_model_path)
        # Make prediction
        prediction = preprocess_and_predict(random_forest_model, geopotential, specific_humidity, air_temperature, eastward_wind, northward_wind)
        st.write(f"Prediction using Neural Network: {prediction[0]}")


with col3:
    st.header("Previous + Future Work 🔮")

 
    if st.button("Forecast Using Ground Observatories"):
        
        st.markdown('<a href="https://thunderstrom.streamlit.app" target="_blank">2D Dataset</a>', unsafe_allow_html=True)

    if st.button("Thunderstorm Segmentation Using Satellite Image"):

        st.markdown('<a href="https://kolkatathunderstrom.streamlit.app" target="_blank">Thunderstorm Segmentation</a>', unsafe_allow_html=True)
    
    if st.button("Generate ALERT !! "):
        
        st.write("Sent Alert TO Farmers , Avaition Industry , New Channels")
    
    if st.button("Forecast Upcoming Values"):
        
        st.write("Forecasting ...")

    

    
