import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Load Model and Scaler ---

# Define the file paths for the saved artifacts
MODEL_PATH = 'AiroX.pkl'
SCALER_PATH = 'RobustScaler.pkl'


@st.cache_resource
def load_artifacts():
    """Loads the pre-trained model and scaler using joblib."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except FileNotFoundError:
        st.error(f"FATAL ERROR: Model or Scaler file not found.")
        st.error(f"Please ensure '{MODEL_PATH}' and '{SCALER_PATH}' exist in the same directory as this app.")
        return None, None


model, scaler = load_artifacts()

# --- 2. Define Constants ---

# IMPORTANT: Ensure this order matches the X_train columns used for model training
FEATURE_NAMES = [
    'Temperature', 'Humidity', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO',
    'Proximity_to_Industrial_Areas', 'Population_Density'
]

# Map the numeric prediction (0-3) back to the original categorical labels
QUALITY_MAPPING = {
    0: ('Good', '✅', 'green'),
    1: ('Moderate', '⚠️', 'orange'),
    2: ('Poor', '🔥', 'red'),
}

# --- 3. Custom CSS for Styling ---

# Hex codes: Black (#000000), Gold Text (#f9e79f), Gold Highlight (#e0c168)
custom_css = """
<style>
/* 1. Main Background: Black */
.stApp {
    background-color: black;
}

/* 1.5 Global Text Color: Gold */
/* This targets all primary Streamlit text elements */
.stApp, .stApp label, .stApp p, .stApp h1, .stApp h2, .stApp h3 {
    color: #f9e79f !important; /* Gold text color */
}

/* 2. Center All Headings: Gold Text */
h1, h2, h3, h4, h5, h6 {
    text-align: center;
    color: #e0c168 !important; /* Gold highlight color for headings */
}

/* 3. Style the Sidebar Containers: Black Background */
/* Target the sidebar container classes */
section[data-testid="stSidebar"] {
    background-color: black !important;
    border-right: 2px solid #e0c168; /* Gold separation line */
}

/* 3.1 Sidebar Text: Gold (must be gold for contrast on black sidebar) */
div[data-testid="stSidebar"] * {
    color: #f9e79f !important;
}

/* 4. Color the Slider Track and Thumbs: Gold (on black sidebar) */
/* Apply gold to the slider track */
div[data-testid="stSidebar"] .stSlider > div > div > div > div {
    background: #e0c168; /* Gold track color */
}

/* Apply gold to the slider thumb (the draggable part) */
div[data-testid="stSidebar"] .stSlider > div > div > div > div > div {
    background: #f9e79f; /* Lighter gold thumb */
    border: 1px solid #e0c168; 
}

/* 5. Prediction Result Box: Black Background */
.prediction-box {
    background-color: black; /* Black background */
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    border: 3px solid #e0c168; /* Gold border */
    box-shadow: 0 4px 12px rgba(224, 193, 104, 0.4); /* Gold shadow */
}

/* 6. Prediction Button Styling (NEW) */

/* Default state: Black background, Gold text/border */
div.stButton > button {
    background-color: black;
    color: #f9e79f !important; 
    border: 2px solid #e0c168 !important;
    transition: background-color 0.3s, color 0.3s, border-color 0.3s;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 16px;
    font-weight: bold;
}

/* Hover state: Gold background, Black text/border */
div.stButton > button:hover {
    background-color: #e0c168; /* Gold on hover */
    color: black !important; /* Black text on hover */
    border-color: black !important;
}


</style>
"""

# Inject the CSS
st.markdown(custom_css, unsafe_allow_html=True)

# --- 4. Streamlit UI and Input ---

st.set_page_config(page_title="💨 AIROX", layout="wide")

st.title("💨 AIROX")
st.markdown("### Input Sensor Readings and Location Data to Predict Air Quality Category")

if model is None or scaler is None:
    st.stop()  # Stop the app if model or scaler loading failed

# --- User Input Sidebar ---
with st.sidebar:
    st.header("Sensor Inputs")

    # Pollutants (Using the corrected FEATURE_NAMES order later for the DataFrame)
    input_pm25 = st.slider('PM2.5 (μg/m³)', 0.0, 500.0, 50.0, 0.1)
    input_pm10 = st.slider('PM10 (μg/m³)', 0.0, 600.0, 100.0, 0.1)
    input_no2 = st.slider('NO₂ (ppb)', 0.0, 100.0, 25.0, 0.1)
    input_so2 = st.slider('SO₂ (ppb)', 0.0, 50.0, 5.0, 0.1)
    input_co = st.slider('CO (ppm)', 0.0, 10.0, 1.0, 0.01)

    st.header("Meteorological & Urban Factors")

    # Weather
    input_temp = st.slider('Temperature (°C)', 0.0, 60.0, 25.0, 0.1)
    input_humidity = st.slider('Humidity (%)', 0.0, 100.0, 60.0, 0.1)

    # Urban Factors
    input_proximity = st.slider('Proximity to Industrial Areas (km)', 0.0, 30.0, 2.5, 0.1)
    input_density = st.number_input('Population Density (ppl/km²)', 100, 20000, 5000, 100)

# --- 5. Data Processing and Prediction ---

# Combine inputs into a DataFrame, ensuring the order matches FEATURE_NAMES
input_data = pd.DataFrame([[
    input_temp, input_humidity, input_pm25, input_pm10, input_no2, input_so2, input_co,
    input_proximity, input_density
]], columns=FEATURE_NAMES)

# Center the DataFrame display
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Scaling the input
scaled_data = scaler.transform(input_data)
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1.1, 0.5, 1])
with col2:
    # Prediction button
    if st.button('Predict Air Quality'):
        with st.spinner('Calculating prediction...'):
            # Predict returns a NumPy array
            prediction_array = model.predict(scaled_data)

            # FIX: Extract the scalar integer from the array
            prediction_numeric = int(prediction_array[0])

            # Map the result to the descriptive label
            label, emoji, color = QUALITY_MAPPING.get(prediction_numeric, ('Unknown', '❓', 'gray'))

            st.subheader("Prediction Result")
            st.markdown(
                # Using the new .prediction-box class for styling
                f"<div class='prediction-box'>"
                f"<h2 style='color: #f9e79f;'>The Predicted Air Quality is:</h2>"
                f"<h1 style='color: {color}; font-size: 60px;'>{emoji} {label}</h1>"
                f"</div>",
                unsafe_allow_html=True
            )

            st.markdown("---")
