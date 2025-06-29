import streamlit as st
import numpy as np
import pickle
import time

# Load model and scaler
model = pickle.load(open('forest_cover_model.pkl', 'rb'))
scaler = pickle.load(open('forest_cover_scaler.pkl', 'rb'))

# Page config
st.set_page_config(page_title="🌲 Forest Cover Type Prediction", layout="wide")
st.title("🌲 Forest Cover Type Prediction App")

# Add instructions
with st.expander("📌 About This App"):
    st.write("""
    Predict the forest cover type for a land patch in Roosevelt National Forest  
    based on environmental, terrain and soil attributes using a trained ML model.
    """)

# Continuous Inputs
with st.expander("📝 Environmental Attributes", expanded=True):
    col1, col2, col3 = st.columns(3)

    with col1:
        elevation = st.slider('🌄 Elevation (meters above sea level)', 1800, 4000, 2700)
        slope = st.slider('⛰️ Slope (degrees)', 0, 70, 20)
        hillshade_9am = st.slider('☀️ Hillshade at 9am (0-255)', 0, 255, 200)

    with col2:
        aspect = st.slider('🧭 Aspect (azimuth degrees)', 0, 360, 180)
        horizontal_distance_to_hydrology = st.slider('💧 Horizontal Dist to Hydrology (m)', 0, 500, 50)
        hillshade_noon = st.slider('☀️ Hillshade at Noon (0-255)', 0, 255, 220)

    with col3:
        vertical_distance_to_hydrology = st.slider('💧 Vertical Dist to Hydrology (m)', -300, 300, 0)
        horizontal_distance_to_roadways = st.slider('🛣️ Horizontal Dist to Roadways (m)', 0, 7000, 2000)
        hillshade_3pm = st.slider('☀️ Hillshade at 3pm (0-255)', 0, 255, 150)

    horizontal_distance_to_fire_points = st.slider('🔥 Horizontal Dist to Fire Points (m)', 0, 7000, 1500)

# Wilderness and Soil Type
with st.expander("🏞️ Wilderness & 🧱 Soil Type", expanded=True):
    wilderness_area = st.selectbox('🏞️ Select Wilderness Area', ['Area 1', 'Area 2', 'Area 3', 'Area 4'])
    wilderness_vector = [0, 0, 0, 0]
    wilderness_vector[int(wilderness_area[-1])-1] = 1

    soil_type = st.slider('🧱 Select Soil Type (1-40)', 1, 40, 10)
    soil_vector = [0]*40
    soil_vector[soil_type-1] = 1

# Prepare final input array
input_data = np.array([[elevation, aspect, slope,
                        horizontal_distance_to_hydrology, vertical_distance_to_hydrology,
                        horizontal_distance_to_roadways, hillshade_9am, hillshade_noon, hillshade_3pm,
                        horizontal_distance_to_fire_points] + wilderness_vector + soil_vector])

# Scale continuous features only (first 10 columns)
input_data_scaled = scaler.transform(input_data)

# Prediction with interactivity
if st.button("🌳 Predict Cover Type"):
    with st.spinner("🔍 Analyzing data and predicting..."):
        progress = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            progress.progress(percent_complete + 1)

        time.sleep(0.3)

    prediction = model.predict(input_data_scaled)
    prediction_proba = model.predict_proba(input_data_scaled)

    st.success(f"🌲 **Predicted Forest Cover Type:** {prediction[0]}")
    st.write(f"Prediction confidence: **{np.max(prediction_proba)*100:.2f}%**")

    st.balloons()  # 🎈 Animation after result

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

st.caption("Made with ❤️ by Raj 🔥")
