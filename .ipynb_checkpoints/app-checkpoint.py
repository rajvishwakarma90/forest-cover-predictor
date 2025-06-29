import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open('forest_cover_model.pkl', 'rb'))
scaler = pickle.load(open('forest_cover_scaler.pkl', 'rb'))

st.set_page_config(page_title="Forest Cover Type Prediction", layout="wide")
st.title("🌲 Forest Cover Type Prediction App")
st.write("Predict the type of forest cover for a land patch in Roosevelt National Forest based on environmental and soil factors.")

# Sidebar info
with st.sidebar:
    st.header("📊 About Project")
    st.write("""
    This machine learning model predicts one of 7 forest cover types using
    continuous and categorical attributes from the UCI Roosevelt National Forest dataset.
    The prediction is based on terrain, hydrology, roads, wilderness area, and soil type data.
    """)
    st.markdown("---")
    st.write("Made by Raj 🔥")

# Continuous Inputs (first 10 features)
st.subheader("📝 Environmental Attributes")
col1, col2, col3 = st.columns(3)

with col1:
    elevation = st.slider('Elevation (m)', 1800, 4000, 2700)
    slope = st.slider('Slope (degrees)', 0, 70, 20)
    hillshade_9am = st.slider('Hillshade at 9am', 0, 255, 200)

with col2:
    aspect = st.slider('Aspect (degrees)', 0, 360, 180)
    horizontal_distance_to_hydrology = st.slider('Horizontal Distance to Hydrology (m)', 0, 500, 50)
    hillshade_noon = st.slider('Hillshade at Noon', 0, 255, 220)

with col3:
    vertical_distance_to_hydrology = st.slider('Vertical Distance to Hydrology (m)', -300, 300, 0)
    horizontal_distance_to_roadways = st.slider('Horizontal Distance to Roadways (m)', 0, 7000, 2000)
    hillshade_3pm = st.slider('Hillshade at 3pm', 0, 255, 150)

horizontal_distance_to_fire_points = st.slider('Horizontal Distance to Fire Points (m)', 0, 7000, 1500)

# Wilderness Area selection (one-hot encoded)
st.subheader("🏞️ Wilderness Area")
wilderness_area = st.selectbox('Select Wilderness Area', ['Area 1', 'Area 2', 'Area 3', 'Area 4'])
wilderness_vector = [0, 0, 0, 0]
wilderness_vector[int(wilderness_area[-1])-1] = 1

# Soil Type selection (one-hot encoded)
st.subheader("🧱 Soil Type")
soil_type = st.slider('Soil Type Number (1-40)', 1, 40, 10)
soil_vector = [0] * 40
soil_vector[soil_type - 1] = 1

# Prepare final input array
input_data = np.array([[elevation, aspect, slope,
                        horizontal_distance_to_hydrology, vertical_distance_to_hydrology,
                        horizontal_distance_to_roadways, hillshade_9am, hillshade_noon, hillshade_3pm,
                        horizontal_distance_to_fire_points] + wilderness_vector + soil_vector])

# Scale continuous features only (first 10 columns scaled)
input_data_scaled = scaler.transform(input_data)

# Predict button
if st.button("🌲 Predict Cover Type"):
    prediction = model.predict(input_data_scaled)
    prediction_proba = model.predict_proba(input_data_scaled)

    st.success(f"**Predicted Forest Cover Type:** {prediction[0]}")
    st.write(f"Prediction confidence: {np.max(prediction_proba)*100:.2f}%")

    st.info("""
    **Cover Type Classes:**
    1 = Spruce/Fir  
    2 = Lodgepole Pine  
    3 = Ponderosa Pine  
    4 = Cottonwood/Willow  
    5 = Aspen  
    6 = Douglas-fir  
    7 = Krummholz  
    """)

st.caption("Made by Raj 🔥")
