import requests
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from datetime import datetime
import folium
from streamlit_folium import st_folium
import pickle
from transformers import pipeline
from PIL import Image
import torch
import google.generativeai as genai
import os

# Set page config
st.set_page_config(
    page_title="Vriksha Rakshak",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with improved background and UI
st.markdown(
    """
    <style>
    /* Main background with gradient */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4f0fb 100%);
        background-attachment: fixed;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2E8B57 0%, #3CB371 100%) !important;
        border-right: 1px solid #ffffff20;
    }
    
    .sidebar .sidebar-content {
        background: transparent !important;
    }
    
    /* Sidebar text color */
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Main content area */
    .main .block-container {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        padding: 2rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Headers */
    h1 {
        color: #2E8B57;
        border-bottom: 2px solid #3CB371;
        padding-bottom: 0.3em;
    }
    
    h2 {
        color: #3CB371;
    }
    
    h3 {
        color: #4682B4;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #3CB371 0%, #2E8B57 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Cards */
    .card {
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border-left: 4px solid #3CB371;
    }
    
    /* Input fields */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select {
        border-radius: 8px !important;
        border: 1px solid #ddd !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 8px 8px 0 0 !important;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: #3CB371 !important;
        color: white !important;
    }
    
    /* Radio buttons */
    [data-testid="stRadio"] label {
        padding: 8px 12px;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stRadio"] label:hover {
        background: #f0f8ff;
    }
    
    [data-testid="stRadio"] [aria-checked="true"]+label {
        background: #3CB371 !important;
        color: white !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #3CB371;
        border-radius: 12px;
        padding: 2rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1rem;
        color: #666;
        font-size: 0.9rem;
    }
    
    /* Animation for important elements */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animated {
        animation: fadeIn 0.5s ease-out;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar with improved visual cues
# st.sidebar.image("farmer.jpg", use_column_width=True)
st.sidebar.markdown("")

app_mode = st.sidebar.radio(
    "Navigate",
    ["Home", "Disease Recognition", "Crop Recommendation", "Insect Detection", "Weather", "Organic Farming", "Soil Health", "About"],
    index=0,
    help="Use the navigation to explore the app!",
    horizontal=False
)

# Adding more descriptive and visually appealing sidebar descriptions
if app_mode == "Home":
    st.sidebar.markdown("🏠 *Home:* Overview of the app and main features.")
elif app_mode == "Disease Recognition":
    st.sidebar.markdown("🦠 *Disease Recognition:* Identify plant diseases using images.")
    st.sidebar.markdown("Upload photos of plant leaves to get instant diagnosis.")
elif app_mode == "Crop Recommendation":
    st.sidebar.markdown("🌾 *Crop Recommendation:* Get suggestions on what to plant based on your conditions.")
    st.sidebar.markdown("Enter soil parameters to find the perfect crop match.")
elif app_mode == "Insect Detection":
    st.sidebar.markdown("🐛 *Insect Detection:* Identify harmful insects affecting your crops.")
    st.sidebar.markdown("Upload insect images for identification and control methods.")
elif app_mode == "Weather":
    st.sidebar.markdown("🌦 *Weather:* Check the local weather for better farming decisions.")
    st.sidebar.markdown("Real-time weather data and forecasts for your location.")
elif app_mode == "Organic Farming":
    st.sidebar.markdown("🌍 *Organic Farming:* Explore eco-friendly farming practices and sustainable solutions.")
    st.sidebar.markdown("Step-by-step guides for growing crops naturally.")
elif app_mode == "Soil Health":
    st.sidebar.markdown("🌱 *Soil Health:* Learn about soil quality and health indicators.")
    st.sidebar.markdown("AI-powered soil analysis and improvement recommendations.")
elif app_mode == "About":
    st.sidebar.markdown("📖 *About:* Learn more about the project and its objectives.")
    st.sidebar.markdown("Our mission and the technology behind Vriksha Rakshak.")

# Main content
if app_mode == "Home":
    st.markdown('<div class="animated">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="card">
        <h1>Vriksha Rakshak</h1>
        <h3>Transforming Agriculture with AI</h3>
        <p>Your all-in-one farming assistant powered by artificial intelligence to help you grow better crops with less effort.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
        <h3>Key Features</h3>
        <ul>
          <li>🌿 Identify plant diseases with image recognition</li>
          <li>🌾 Get personalized crop recommendations</li>
          <li>🐛 Detect harmful insects and get control methods</li>
          <li>⛅ Check hyperlocal weather forecasts</li>
          <li>🌱 Learn organic farming techniques</li>
          <li>🧪 Analyze your soil health</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.image("home_page.jpg", use_column_width=True)
        
        st.markdown("""
        <div class="card">
        <h3>Getting Started</h3>
        <ol>
          <li>Select a feature from the sidebar</li>
          <li>Provide required inputs (images, parameters, etc.)</li>
          <li>Get instant AI-powered recommendations</li>
           <li>Implement the suggestions for better yields</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

elif app_mode == "Disease Recognition":
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")

    def model_prediction(test_image):
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # convert single image to batch
        predictions = model.predict(input_arr)
        return np.argmax(predictions)  # return index of max element

    st.markdown('<div class="animated">', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <h1>Plant Disease Recognition</h1>
    <p>Upload a photo of your plant to get instant diagnosis and treatment recommendations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="card">
        <h3>How to take good photos:</h3>
        <ul>
          <li>Take photos in natural daylight</li>
          <li>Focus on affected leaves or stems</li>
          <li>Include both close-up and whole plant views</li>
          <li>Ensure the image is clear and in focus</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        test_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], 
                                     help="Upload a clear image of the affected plant part")

    if test_image:
        st.markdown("""
        <div class="card">
        <h3>Uploaded Image Preview</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(test_image, use_column_width=True)
        
        with col2:
            if st.button("🔍 Analyze Plant", help="Click to predict disease"):
                with st.spinner("Analyzing your plant image... Please wait."):
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown('<div class="animated">', unsafe_allow_html=True)
                    
                    result_index = model_prediction(test_image)
                    
                    class_name = [
                        'Apple: Apple scab',
                        'Apple: Black rot',
                        'Apple: Cedar apple rust',
                        'Apple: Healthy',
                        'Blueberry: Healthy',
                        'Cherry (including sour): Powdery mildew',
                        'Cherry (including sour): Healthy',
                        'Corn (maize): Cercospora leaf spot, Gray leaf spot',
                        'Corn (maize): Common rust',
                        'Corn (maize): Northern Leaf Blight',
                        'Corn (maize): Healthy',
                        'Grape: Black rot',
                        'Grape: Esca (Black Measles)',
                        'Grape: Leaf blight (Isariopsis Leaf Spot)',
                        'Grape: Healthy',
                        'Orange: Huanglongbing (Citrus greening)',
                        'Peach: Bacterial spot',
                        'Peach: Healthy',
                        'Pepper (bell): Bacterial spot',
                        'Pepper (bell): Healthy',
                        'Potato: Early blight',
                        'Potato: Late blight',
                        'Potato: Healthy',
                        'Raspberry: Healthy',
                        'Soybean: Healthy',
                        'Squash: Powdery mildew',
                        'Strawberry: Leaf scorch',
                        'Strawberry: Healthy',
                        'Tomato: Bacterial spot',
                        'Tomato: Early blight',
                        'Tomato: Late blight',
                        'Tomato: Leaf Mold',
                        'Tomato: Septoria leaf spot',
                        'Tomato: Spider mites (Two-spotted spider mite)',
                        'Tomato: Target Spot',
                        'Tomato: Tomato Yellow Leaf Curl Virus',
                        'Tomato: Tomato mosaic virus',
                        'Tomato: Healthy'
                        ]
                    plant_disease_treatments = {
                        "Apple: Apple scab": "Apply fungicides during the early season. Remove and destroy fallen leaves and infected fruit. Use resistant apple varieties.",
                        "Apple: Black rot": "Prune and destroy infected twigs, branches, and fruits. Apply fungicides preventatively. Maintain tree health with proper fertilization and watering.",
                        "Apple: Cedar apple rust": "Apply fungicides early in the season. Remove nearby cedar trees if possible, as they are alternate hosts.",
                        "Apple: Healthy": "No treatment needed. Maintain regular care to keep the plant healthy.",
                        "Blueberry: Healthy": "No treatment needed. Ensure proper soil conditions and adequate watering.",
                        "Cherry (including sour): Powdery mildew": "Apply sulfur-based or fungicidal sprays. Increase air circulation and avoid overhead watering.",
                        "Cherry (including sour): Healthy": "No treatment needed. Continue regular disease prevention practices.",
                        "Corn (maize): Cercospora leaf spot, Gray leaf spot": "Use resistant varieties. Apply fungicides if necessary and practice crop rotation.",
                        "Corn (maize): Common rust": "Plant resistant varieties and apply fungicides if needed. Maintain good field hygiene.",
                        "Corn (maize): Northern Leaf Blight": "Use resistant seeds and rotate crops. Apply fungicides if required.",
                        "Corn (maize): Healthy": "No treatment needed. Maintain regular care.",
                        "Grape: Black rot": "Prune and remove infected vines. Apply fungicides in early spring. Ensure good air circulation.",
                        "Grape: Esca (Black Measles)": "Remove infected wood. Ensure vines are not stressed and avoid trunk or root injuries.",
                        "Grape: Leaf blight (Isariopsis Leaf Spot)": "Apply fungicides and prune affected leaves. Increase ventilation around vines.",
                        "Grape: Healthy": "No treatment needed. Maintain proper care.",
                        "Orange: Huanglongbing (Citrus greening)": "No known cure. Remove and destroy infected trees. Control psyllid vectors and plant disease-free stock.",
                        "Peach: Bacterial spot": "Apply copper-based bactericides. Use resistant varieties if available. Prune and remove affected parts.",
                        "Peach: Healthy": "No treatment needed. Ensure good tree health.",
                        "Pepper (bell): Bacterial spot": "Apply copper-based sprays. Practice crop rotation and use certified disease-free seeds.",
                        "Pepper (bell): Healthy": "No treatment needed. Maintain regular care.",
                        "Potato: Early blight": "Apply fungicides and remove infected plant debris. Practice crop rotation.",
                        "Potato: Late blight": "Use resistant varieties. Apply fungicides and destroy infected plants.",
                        "Potato: Healthy": "No treatment needed. Continue regular monitoring.",
                        "Raspberry: Healthy": "No treatment needed. Maintain proper care and disease prevention practices.",
                        "Soybean: Healthy": "No treatment needed. Practice regular crop management.",
                        "Squash: Powdery mildew": "Apply fungicides. Improve air circulation and avoid overhead watering.",
                        "Strawberry: Leaf scorch": "Remove and destroy infected leaves. Apply appropriate fungicides.",
                        "Strawberry: Healthy": "No treatment needed. Maintain regular care.",
                        "Tomato: Bacterial spot": "Use copper sprays and disease-resistant seeds. Practice crop rotation.",
                        "Tomato: Early blight": "Apply fungicides and remove affected plant parts.",
                        "Tomato: Late blight": "Apply fungicides and remove and destroy infected plants. Use resistant varieties.",
                        "Tomato: Leaf Mold": "Increase air circulation and apply fungicides. Prune lower leaves.",
                        "Tomato: Septoria leaf spot": "Apply fungicides and remove infected leaves. Ensure good ventilation.",
                        "Tomato: Spider mites (Two-spotted spider mite)": "Use insecticidal soap or horticultural oil. Increase humidity around plants.",
                        "Tomato: Target Spot": "Apply fungicides and remove infected plant debris.",
                        "Tomato: Tomato Yellow Leaf Curl Virus": "Control whitefly populations. Use resistant varieties and remove infected plants.",
                        "Tomato: Tomato mosaic virus": "Remove and destroy infected plants. Sanitize tools and avoid tobacco products.",
                        "Tomato: Healthy": "No treatment needed. Maintain regular plant care."
                    }
                    
                    st.markdown(f"""
                    <div class="card" style="background-color: #f0fff0;">
                    <h2>Diagnosis Results</h2>
                    <div style="padding: 1rem; background-color: white; border-radius: 8px; margin: 1rem 0;">
                    <h4>Identified Condition:</h4>
                    <p style="font-size: 1.2rem; font-weight: bold; color: #2E8B57;">{class_name[result_index]}</p>
                    </div>
                    
                    <div style="padding: 1rem; background-color: white; border-radius: 8px;">
                    <h4>Recommended Treatment:</h4>
                    <p>{plant_disease_treatments[class_name[result_index]]}</p>
                    </div>
                    </div>
                    """, unsafe_allow_html=True)

elif app_mode == "Soil Health":

    genai.configure(api_key="AIzaSyDPRI0VEuHeZ1A2CDL7_AqqgExvb9Go4qY")
    
    st.markdown('<div class="animated">', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <h1>Soil Health Analysis</h1>
    <p>Get AI-powered soil quality assessment and improvement recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
        <h3>Crop History</h3>
        """, unsafe_allow_html=True)
        crop_year_1 = st.text_input("Year 1 Crop", placeholder="Wheat")
        crop_year_2 = st.text_input("Year 2 Crop", placeholder="Soybean")
        crop_year_3 = st.text_input("Year 3 Crop", placeholder="Corn")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
        <h3>Soil Properties</h3>
        """, unsafe_allow_html=True)
        soil_texture = st.selectbox(
            "Soil Texture", 
            [
                "Sandy", "Loamy sand", "Sandy loam", "Loam", 
                "Silt loam", "Silty clay loam", "Clay loam", 
                "Silty clay", "Clay"
            ]
        )
        soil_ph = st.slider("Soil pH", 0.0, 14.0, 6.5, 0.1,
                           help="Ideal range for most crops: 6.0-7.0")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
        <h3>Farming Practices</h3>
        """, unsafe_allow_html=True)
        tillage_practices = st.selectbox(
            "Tillage Practices", 
            ["No-till", "Reduced tillage", "Conventional tillage"]
        )
        fertilization_practices = st.selectbox(
            "Fertilization", 
            ["Organic", "Inorganic", "Both", "None"]
        )
        irrigation_practices = st.selectbox(
            "Irrigation", 
            ["Drip", "Sprinkler", "Flood", "Furrow", "None"]
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
        <h3>Location</h3>
        """, unsafe_allow_html=True)


        # Load state and city data
        with open('weatherdata.json', 'r') as file:
            data = json.load(file)

        states = data.keys()

        state_name = st.selectbox("Select State", options=states)
        if state_name:
            cities = data[state_name]
            city_name = st.selectbox("Select City", options=cities)

        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔍 Analyze Soil Health", help="Click to get soil health assessment"):
        if crop_year_1 and crop_year_2 and crop_year_3:
            with st.spinner("Analyzing soil health... Please wait."):
                prompt = f"""
                Based on the following agricultural data:

                Crop History:
                - Year 1: {crop_year_1}
                - Year 2: {crop_year_2}
                - Year 3: {crop_year_3}
                
                Farming Practices:
                - Tillage: {tillage_practices}
                - Fertilization: {fertilization_practices}
                - Irrigation: {irrigation_practices}
                
                Soil Properties:
                - Texture: {soil_texture}
                - pH: {soil_ph}
                
                state name: {state_name}
                city name: {city_name}

                Provide a detailed soil health assessment including:
                1. Current soil health rating (1-10)
                2. Key strengths and weaknesses
                3. Three specific improvement recommendations
                4. Best crops for these conditions
                """

                try:
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    response = model.generate_content(prompt)
                    
                    st.markdown(f"""
                    <div class="card" style="background-color: #f0f8ff;">
                    <h2>Soil Health Report</h2>
                    <div style="padding: 1rem; background-color: white; border-radius: 8px; margin: 1rem 0;">
                    <p>{response.text}<p/>
                    </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please provide crop history for all three years.")



elif app_mode == "Insect Detection":
    st.markdown('<div class="animated">', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <h1>Insect Detection & Management</h1>
    <p>Upload insect images for identification and get control recommendations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="card">
        <h3>How to take good photos:</h3>
        <ul>
          <li>Take photos in natural daylight</li>
          <li>Capture from multiple angles</li>
          <li>Include a scale reference if possible</li>
          <li>Focus on the insect's distinguishing features</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], 
                                       help="Upload a clear image of the insect")

    if uploaded_file:
        st.markdown("""
        <div class="card">
        <h3>Uploaded Image Preview</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        
        with col2:
            if st.button("🔍 Identify Insect", help="Click to identify the insect"):
                with st.spinner("Identifying insect... Please wait."):
                    # Load and run the model
                    try:
                        model_name = "insects_models/"  # Replace with your actual model path
                        pipe = pipeline('image-classification', model=model_name, device=0)
                        predictions = pipe(image)
                        top_prediction = max(predictions, key=lambda x: x['score'])
                        insect_label = top_prediction['label']
                        confidence = top_prediction['score']
                    except Exception as e:
                        st.error(f"Error loading/running model: {str(e)}")
                        insect_label = "Not Found"  # Fallback to mock data
                        confidence = 0.00
                    
                    st.markdown('<div class="animated">', unsafe_allow_html=True)
                    
                    insect_control_data = {
                        "Western Corn Rootworms": {
                            "description": "Yellowish beetles with black stripes that feed on corn roots and silks. Larvae are white with brown heads and feed on corn roots.",
                            "risk_level": "High - Can cause significant yield loss",
                            "affected_crops": ["Corn"],
                            "pesticides": ["Bifenthrin", "Chlorantraniliprole", "Thiamethoxam"],
                            "organic_options": ["Crop rotation", "Beneficial nematodes", "Diatomaceous earth"],
                            "damage": "Root pruning, lodged plants, reduced nutrient uptake. Adults feed on silks, interfering with pollination.",
                            "lifecycle": "Eggs overwinter in soil, larvae feed on roots in spring, adults emerge in summer to lay eggs.",
                            "tips": [
                                "Rotate crops annually to break the pest's life cycle",
                                "Plant resistant corn varieties",
                                "Use beneficial nematodes in the soil to control larvae",
                                "Monitor fields with yellow sticky traps",
                                "Time planting to avoid peak beetle activity"
                            ]
                        },
                        "Tomato Hornworms": {
                            "description": "Large green caterpillars with white diagonal stripes and a horn-like tail. They feed voraciously on tomato plants and can quickly defoliate them.",
                            "risk_level": "High - Can defoliate plants quickly",
                            "affected_crops": ["Tomato", "Pepper", "Eggplant", "Potato"],
                            "pesticides": ["Spinosad", "Bacillus thuringiensis (Bt)", "Pyrethrin"],
                            "organic_options": ["Hand picking", "Neem oil", "Beneficial insects (wasps)"],
                            "damage": "Chewed leaves, missing foliage, dark green droppings on leaves. Severe infestations can strip plants completely.",
                            "lifecycle": "Complete metamorphosis: Eggs hatch in 4-5 days, larval stage lasts 3-4 weeks, pupates in soil for 2-4 weeks, adult moths emerge to lay eggs.",
                            "tips": [
                                "Handpick worms off plants in the early morning or evening",
                                "Encourage natural predators like parasitic wasps",
                                "Use Bacillus thuringiensis (Bt) for biological control",
                                "Rotate crops to disrupt lifecycle",
                                "Till soil in fall to expose pupae to predators"
                            ]
                        },
                        "Thrips": {
                            "description": "Tiny, slender insects with fringed wings that feed by puncturing plant cells. They can transmit plant viruses.",
                            "risk_level": "Moderate-High (can transmit viruses)",
                            "affected_crops": ["Various vegetables", "Fruits", "Ornamentals"],
                            "pesticides": ["Spinosad", "Abamectin", "Imidacloprid"],
                            "organic_options": ["Insecticidal soap", "Neem oil", "Predatory mites"],
                            "damage": "Silvery speckling on leaves, distorted growth, black fecal spots. May cause flower buds to abort.",
                            "lifecycle": "Eggs inserted into plant tissue, nymphs feed for 1-3 weeks, pupate in soil or on plant.",
                            "tips": [
                                "Use sticky traps to monitor and reduce their population",
                                "Remove and destroy infested plant parts",
                                "Apply insecticidal soap or neem oil sprays",
                                "Introduce predatory mites or minute pirate bugs",
                                "Avoid over-fertilizing with nitrogen"
                            ]
                        },
                        "Spider Mites": {
                            "description": "Tiny arachnids that create fine webbing on plants. They thrive in hot, dry conditions.",
                            "risk_level": "High in dry conditions",
                            "affected_crops": ["Various crops", "Ornamentals", "Fruit trees"],
                            "pesticides": ["Abamectin", "Bifenazate", "Spiromesifen"],
                            "organic_options": ["Horticultural oil", "Insecticidal soap", "Predatory mites"],
                            "damage": "Yellow stippling on leaves, fine webbing, leaf drop. Severe infestations can kill plants.",
                            "lifecycle": "Complete lifecycle in 5-20 days depending on temperature. Eggs hatch into six-legged larvae.",
                            "tips": [
                                "Spray plants with water to dislodge mites",
                                "Introduce predatory mites like Phytoseiulus persimilis",
                                "Avoid over-fertilizing as it encourages outbreaks",
                                "Increase humidity around plants",
                                "Remove heavily infested leaves"
                            ]
                        },
                        "Fruit Flies": {
                            "description": "Small flies attracted to fermenting fruits and vegetables. Larvae feed within ripe or rotting produce.",
                            "risk_level": "Moderate (more nuisance than damage)",
                            "affected_crops": ["Various fruits", "Vegetables"],
                            "pesticides": ["Malathion", "Spinosad", "Pyrethrin"],
                            "organic_options": ["Vinegar traps", "Cultural control", "Sanitation"],
                            "damage": "Larvae tunneling in fruits, premature fruit drop, secondary rot organisms.",
                            "lifecycle": "Egg to adult in 7-30 days depending on temperature. Females lay eggs in ripe fruit.",
                            "tips": [
                                "Remove overripe or rotting fruits promptly",
                                "Use vinegar or fruit-based traps",
                                "Clean drains and other breeding areas",
                                "Harvest fruits as soon as they ripen",
                                "Store produce in refrigerator"
                            ]
                        },
                        "Fall Armyworms": {
                            "description": "Caterpillars with distinctive inverted Y on head. Highly mobile and can quickly defoliate plants.",
                            "risk_level": "Very High (rapid defoliators)",
                            "affected_crops": ["Corn", "Grasses", "Various vegetables"],
                            "pesticides": ["Bacillus thuringiensis (Bt)", "Spinetoram", "Lambda-Cyhalothrin"],
                            "organic_options": ["Hand picking", "Biological controls", "Trap crops"],
                            "damage": "Ragged leaf feeding, defoliation, frass (excrement) visible. May feed on fruits in heavy infestations.",
                            "lifecycle": "Complete lifecycle in 30 days in summer. Cannot survive cold winters.",
                            "tips": [
                                "Inspect plants regularly for egg masses and larvae",
                                "Encourage natural predators like birds",
                                "Apply Bt when larvae are small",
                                "Use pheromone traps to monitor adults",
                                "Plant early to avoid peak populations"
                            ]
                        },
                        "Corn Earworms": {
                            "description": "Variable colored caterpillars (green, pink, brown) that feed on corn ears and other crops.",
                            "risk_level": "High (direct damage to ears)",
                            "affected_crops": ["Corn", "Tomato", "Cotton", "Soybeans"],
                            "pesticides": ["Carbaryl", "Permethrin", "Bacillus thuringiensis (Bt)"],
                            "organic_options": ["Mineral oil on silks", "Trichogramma wasps", "Early planting"],
                            "damage": "Holes in leaves, feeding on silks and kernels, frass in ear tips.",
                            "lifecycle": "Multiple generations per year. Pupae overwinter in soil.",
                            "tips": [
                                "Apply mineral oil to corn silk to prevent larvae entry",
                                "Encourage beneficial insects like lacewings",
                                "Use row covers to protect young plants",
                                "Plant early-maturing varieties",
                                "Destroy crop residues after harvest"
                            ]
                        },
                        "Corn Borers": {
                            "description": "Pale caterpillars with dark spots that tunnel into corn stalks and ears.",
                            "risk_level": "High (stalk weakening)",
                            "affected_crops": ["Corn", "Peppers", "Potatoes"],
                            "pesticides": ["Chlorantraniliprole", "Lambda-Cyhalothrin", "Bacillus thuringiensis (Bt)"],
                            "organic_options": ["Bt corn varieties", "Parasitic wasps", "Crop rotation"],
                            "damage": "Holes in leaves, stalk tunneling, ear damage, lodging of plants.",
                            "lifecycle": "1-4 generations per year depending on climate. Overwinter as larvae in stalks.",
                            "tips": [
                                "Plant early-maturing corn varieties",
                                "Remove crop residues after harvest",
                                "Use pheromone traps to monitor activity",
                                "Plant Bt corn varieties where legal",
                                "Encourage parasitic wasps"
                            ]
                        },
                        "Colorado Potato Beetles": {
                            "description": "Yellow-orange beetles with black stripes. Both adults and larvae feed on foliage.",
                            "risk_level": "Very High (rapid defoliators)",
                            "affected_crops": ["Potato", "Eggplant", "Tomato", "Pepper"],
                            "pesticides": ["Spinosad", "Imidacloprid", "Azadirachtin (Neem Oil)"],
                            "organic_options": ["Hand picking", "Crop rotation", "Floating row covers"],
                            "damage": "Skeletonized leaves, complete defoliation possible. Can kill young plants.",
                            "lifecycle": "1-3 generations per year. Adults overwinter in soil.",
                            "tips": [
                                "Handpick beetles and larvae from plants",
                                "Rotate crops to reduce populations",
                                "Use floating row covers on young plants",
                                "Plant trap crops like eggplant",
                                "Delay planting to avoid first generation"
                            ]
                        },
                        "Citrus Canker": {
                            "description": "Bacterial disease causing raised lesions on leaves, stems and fruit. Spread by wind-driven rain.",
                            "risk_level": "High (quarantine significant)",
                            "affected_crops": ["Citrus trees"],
                            "pesticides": ["Copper-Based Fungicides", "Streptomycin Sulfate"],
                            "organic_options": ["Pruning", "Copper sprays", "Resistant varieties"],
                            "damage": "Raised corky lesions on leaves, fruit drop, twig dieback. Makes fruit unmarketable.",
                            "lifecycle": "Bacteria enter through stomata or wounds. Spread by rain, wind, equipment.",
                            "tips": [
                                "Prune and destroy infected branches",
                                "Avoid overhead watering",
                                "Apply copper sprays as preventive",
                                "Disinfect tools between trees",
                                "Plant resistant varieties where available"
                            ]
                        },
                        "Cabbage Loopers": {
                            "description": "Green caterpillars that move with a characteristic looping motion. Feed on brassica crops.",
                            "risk_level": "Moderate-High",
                            "affected_crops": ["Cabbage", "Broccoli", "Cauliflower", "Kale"],
                            "pesticides": ["Bacillus thuringiensis (Bt)", "Spinosad", "Permethrin"],
                            "organic_options": ["Row covers", "Hand picking", "Beneficial insects"],
                            "damage": "Irregular holes in leaves, frass present. Can bore into heads in heavy infestations.",
                            "lifecycle": "Complete lifecycle in about 30 days. Multiple generations per year.",
                            "tips": [
                                "Cover plants with floating row covers",
                                "Handpick caterpillars from leaves",
                                "Use Bt to target larvae specifically",
                                "Encourage parasitic wasps",
                                "Interplant with strong-smelling herbs"
                            ]
                        },
                        "Brown Marmorated Stink Bugs": {
                            "description": "Shield-shaped bugs that emit foul odor when disturbed. Feed on many fruits and vegetables.",
                            "risk_level": "High (cosmetic damage to fruit)",
                            "affected_crops": ["Various fruits", "Vegetables", "Legumes"],
                            "pesticides": ["Dinotefuran", "Bifenthrin", "Pyrethroid Insecticides"],
                            "organic_options": ["Kaolin clay", "Physical barriers", "Trapping"],
                            "damage": "Cat-facing on fruits, discolored spots, seed abortion. May introduce pathogens.",
                            "lifecycle": "1-2 generations per year. Adults overwinter in structures.",
                            "tips": [
                                "Seal cracks in buildings to prevent entry",
                                "Use pheromone traps to monitor",
                                "Encourage natural predators",
                                "Handpick in early morning",
                                "Apply kaolin clay as repellent"
                            ]
                        },
                        "Armyworms": {
                            "description": "Green or brown striped caterpillars that march in large numbers between fields.",
                            "risk_level": "Very High (rapid defoliators)",
                            "affected_crops": ["Grasses", "Corn", "Small grains", "Vegetables"],
                            "pesticides": ["Bacillus thuringiensis (Bt)", "Chlorantraniliprole", "Spinosad"],
                            "organic_options": ["Biological controls", "Trap crops", "Early harvest"],
                            "damage": "Skeletonized leaves, complete defoliation. May move en masse to new fields.",
                            "lifecycle": "Multiple generations per year. Cannot survive cold winters in northern areas.",
                            "tips": [
                                "Encourage natural enemies like ground beetles",
                                "Mow and remove weeds to eliminate egg sites",
                                "Apply insecticides early when larvae are small",
                                "Use trap crops to concentrate populations",
                                "Monitor field edges where infestations often start"
                            ]
                        },
                        "Aphids": {
                            "description": "Small, soft-bodied insects that cluster on new growth. Many species exist with different host preferences.",
                            "risk_level": "Moderate (can transmit viruses)",
                            "affected_crops": ["Almost all crops"],
                            "pesticides": ["Imidacloprid", "Acetamiprid", "Flupyradifurone"],
                            "organic_options": ["Insecticidal soap", "Neem oil", "Beneficial insects"],
                            "damage": "Curled leaves, sticky honeydew, sooty mold growth. May transmit plant viruses.",
                            "lifecycle": "Complex with sexual and asexual generations. Can reproduce without mating.",
                            "tips": [
                                "Spray plants with strong water stream",
                                "Introduce ladybugs or lacewings",
                                "Apply neem oil or insecticidal soap",
                                "Plant companion plants that repel aphids",
                                "Use reflective mulches to deter them"
                            ]
                        },
                        "Africanized Honey Bees (Killer Bees)": {
                            "description": "Aggressive hybrid honey bees that defend nests vigorously. Similar appearance to European honey bees.",
                            "risk_level": "High (stinging hazard)",
                            "affected_crops": ["N/A (pollinators but dangerous)"],
                            "pesticides": ["Pyrethrins", "Synthetic Pyrethroids"],
                            "organic_options": ["Professional removal", "Exclusion", "Prevention"],
                            "damage": "Not crop pests but dangerous to humans. May outcompete native pollinators.",
                            "lifecycle": "Similar to European honey bees but more prolific swarming.",
                            "tips": [
                                "Never disturb a bee nest",
                                "Seal holes in walls and trees",
                                "Contact professional beekeepers for removal",
                                "Wear light-colored clothing when working outside",
                                "Have an escape plan when working in areas with bees"
                            ]
                        }
                    }
                    
                    st.markdown(f"""
                    <div class="card" style="background-color: #fffaf0;">
                    <h2>Identification Results</h2>
                    <div style="padding: 1rem; background-color: white; border-radius: 8px; margin: 1rem 0;">
                    <h4>Identified Insect:</h4>
                    <p style="font-size: 1.2rem; font-weight: bold; color: #2E8B57;">{insect_label}</p>
                    <p>Confidence: {confidence:.0%}</p>
                    </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if insect_label in insect_control_data:
                        insect_data = insect_control_data[insect_label]
                        
                        tab1, tab2, tab3, tab4 = st.tabs(["📋 Description", "⚗ Control Methods", "🌱 Damage & Lifecycle", "📌 Management Tips"])
                        
                        with tab1:
                            st.markdown(f"""
                            <div class="card">
                            <h4>Description</h4>
                            <p>{insect_data['description']}</p>
                            
                            <h4>Risk Level</h4>
                            <p>{insect_data['risk_level']}</p>
                            
                            <h4>Commonly Affected Crops</h4>
                            <ul>
                            {''.join([f'<li>{crop}</li>' for crop in insect_data['affected_crops']])}
                            </ul>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with tab2:
                            st.markdown(f"""
                            <div class="card">
                            <h4>Chemical Control Options</h4>
                            <ul>
                            {''.join([f'<li>{pesticide}</li>' for pesticide in insect_data['pesticides']])}
                            </ul>
                            
                            <h4>Organic Control Options</h4>
                            <ul>
                            {''.join([f'<li>{option}</li>' for option in insect_data['organic_options']])}
                            </ul>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with tab3:
                            st.markdown(f"""
                            <div class="card">
                            <h4>Damage Symptoms</h4>
                            <p>{insect_data['damage']}</p>
                            
                            <h4>Lifecycle Information</h4>
                            <p>{insect_data['lifecycle']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with tab4:
                            st.markdown(f"""
                            <div class="card">
                            <h4>Integrated Pest Management Tips</h4>
                            <ol>
                            {''.join([f'<li>{tip}</li>' for tip in insect_data['tips']])}
                            </ol>
                            </div>
                            """, unsafe_allow_html=True)

                    else:
                        st.warning(f"Insect '{insect_label}' not found in our database. Please consult a local agricultural extension service.")


elif app_mode == "Organic Farming":
    organic_crops_steps = {
        "Rice": {
        "tips": [
            "Use organic compost or manure to enrich soil fertility. This improves soil structure, water retention, and nutrient availability.",
            "Rotate rice with leguminous crops like beans or peas to improve nitrogen levels in the soil naturally. Legumes fix nitrogen from the atmosphere, reducing the need for synthetic fertilizers.",
            "Use natural pesticides like neem oil to protect against pests. Neem oil is a broad-spectrum insecticide, fungicide, and miticide that is safe for humans and the environment.",
            "Practice proper water management to conserve water and reduce waterlogging, which can lead to diseases.",
            "Select rice varieties suitable for your region and climate for optimal yield.",
            "Ensure proper spacing between rice seedlings to allow for healthy growth and reduce competition for resources.",
            "Control weeds effectively through manual weeding, mechanical methods, or organic mulches to prevent them from competing with rice for nutrients and sunlight."
        ],
        "steps": {
            "June": [
            "Prepare the soil by plowing and mixing in organic manure or compost. This creates a loose, well-aerated soil structure that is conducive to root growth.",
            "Sow pre-soaked rice seeds in well-watered soil. Soaking the seeds helps them germinate faster and more uniformly.",
            "Alternatively, you can transplant rice seedlings that have been grown in a nursery. This method allows for better control over seedling growth and reduces the time it takes for the rice to mature in the field."
            ],
            "July": [
            "Ensure proper irrigation during the growing phase, applying natural fertilizers periodically. Rice requires a consistent supply of water, especially during the tillering and flowering stages.",
            "Weed the field manually or use organic mulch to suppress weeds. Weeds can significantly reduce rice yields if they are not controlled effectively.",
            "Monitor for pests and diseases regularly and take appropriate action if necessary. Early detection and treatment can prevent major crop losses."
            ],
            "August": [
            "Continue irrigation and maintain weed control. As the rice plants grow taller, they become more resistant to weeds.",
            "Monitor for pests and apply neem oil if necessary. Neem oil is most effective when applied preventatively or at the first sign of pest infestation.",
            "Ensure that the rice plants receive adequate sunlight for photosynthesis and healthy growth."
            ],
            "September": [
            "Harvest when grains turn golden and dry under the sun naturally. This indicates that the rice is mature and ready for harvest.",
            "Thresh the harvested rice to separate the grains from the stalks.",
            "Dry the rice grains thoroughly to prevent mold growth and ensure their long-term storage."
            ]
        },
        "weather": "Rice grows well in warm, tropical climates with temperatures ranging from 25°C to 32°C and requires abundant water. High humidity is also beneficial for rice growth.",
        "month_to_grow": "June to August (Monsoon season)",
        "month_to_grow_in_india": "June to September (Monsoon season), with variations depending on the specific region."
        },
        "Wheat": {
        "tips": [
            "Sow wheat seeds after legume crops to improve soil nitrogen naturally. This reduces the need for nitrogen fertilizers.",
            "Apply organic mulch to retain moisture and suppress weeds. Mulch also helps regulate soil temperature and prevent soil erosion.",
            "Use cow dung manure to boost soil nutrients organically. Cow dung is a rich source of essential plant nutrients.",
            "Choose wheat varieties that are resistant to common diseases and pests in your area.",
            "Ensure proper seedbed preparation by plowing and harrowing the field to create a fine tilth.",
            "Sow wheat seeds at the recommended depth and spacing to ensure uniform germination and growth.",
            "Irrigate the wheat crop at critical growth stages, such as tillering, flowering, and grain filling."
        ],
        "steps": {
            "November": [
            "Plow the field and apply organic compost to enrich the soil. Plowing helps to aerate the soil and incorporate organic matter.",
            "Sow wheat seeds evenly in rows during the cooler season. This ensures uniform germination and growth of the wheat crop."
            ],
            "December": [
            "Irrigate using rainwater or minimal artificial water sources. Wheat requires less water than rice, but adequate moisture is essential, especially during the early stages of growth.",
            "Remove weeds by hand or apply organic mulches to suppress them. Weeds can compete with wheat for resources and reduce yields."
            ],
            "January": [
            "Ensure the crop gets enough moisture and control weeds manually. This is a crucial period for wheat growth, and adequate moisture is essential for healthy development.",
            "Monitor for pests and diseases and take appropriate action if necessary."
            ],
            "February": [
            "Harvest when wheat turns golden and dry in natural sunlight. This indicates that the wheat is mature and ready for harvest.",
            "Thresh the harvested wheat to separate the grains from the stalks.",
            "Dry the wheat grains thoroughly to prevent mold growth and ensure their long-term storage."
            ]
        },
        "weather": "Wheat grows best in temperate climates with moderate rainfall and temperatures between 12°C and 22°C. Cool temperatures during the growing season are essential for proper grain development.",
        "month_to_grow": "November to February (Winter season)",
        "month_to_grow_in_india": "November to March (Winter season), with regional variations."
    },

        "Maize": {
        "tips": [
            "Apply farmyard manure (FYM) or compost before planting to enrich the soil. FYM improves soil structure, water-holding capacity, and nutrient availability.",
            "Intercrop maize with legumes like beans, cowpeas, or groundnuts to enhance soil fertility naturally. Legumes fix nitrogen from the atmosphere, benefiting the maize crop.",
            "Control pests using natural repellents like garlic spray, neem oil, or companion planting (e.g., marigold). These methods are environmentally friendly and reduce the risk of chemical residues.",
            "Ensure proper spacing between maize plants to allow for adequate airflow, sunlight penetration, and nutrient uptake. This helps to prevent overcrowding and reduces the risk of diseases.",
            "Practice crop rotation to break pest and disease cycles and improve soil health. Rotating maize with other crops can also help to improve nutrient balance in the soil.",
            "Provide adequate irrigation, especially during critical growth stages like tasseling and silking. Maize is sensitive to water stress, and insufficient moisture can significantly reduce yields."
        ],
        "steps": {
            "March": [
            "Prepare the soil by plowing, harrowing, and adding compost or organic manure. This creates a loose, well-aerated soil structure that is ideal for maize growth.",
            "Ensure proper irrigation setup if using drip or sprinkler irrigation systems. This will help to conserve water and deliver it directly to the roots of the maize plants."
            ],
            "April": [
            "Sow maize seeds with appropriate spacing (typically 60cm x 25cm) to ensure airflow and prevent overcrowding. The recommended spacing may vary depending on the maize variety and local conditions.",
            "Water consistently but avoid waterlogging. Maize needs adequate moisture for germination and growth, but excessive water can lead to root rot and other diseases."
            ],
            "May": [
            "Weed the field manually or use organic mulch to suppress weeds and retain soil moisture. Weeds compete with maize for nutrients, water, and sunlight, so effective weed control is essential.",
            "Thin out excess seedlings if necessary to maintain the desired plant spacing."
            ],
            "June": [
            "Continue irrigation and monitor for pests like stem borers, aphids, and fall armyworms. Take appropriate action if necessary, using natural pest control methods whenever possible.",
            "Apply top dressing of fertilizer if needed, based on soil test results and crop requirements."
            ],
            "July": [
            "Harvest maize when the ears are fully mature and the silks have turned brown and dry. The kernels should be plump and milky when squeezed.",
            "Dry the harvested maize cobs naturally in the sun or using artificial drying methods to reduce moisture content and prevent spoilage."
            ]
        },
        "weather": "Maize thrives in warm weather with temperatures between 21°C to 30°C and requires moderate rainfall or irrigation. It is a C4 plant, which means it is efficient at converting sunlight into energy, making it well-suited to warm climates.",
        "month_to_grow": "March to June (Summer season)",
        "month_to_grow_in_india": "March to July (Summer season), with variations depending on the specific region."
        },
        "Bajra": {
        "tips": [
            "Use organic manure, such as farmyard manure or compost, to enhance soil texture and fertility. This improves water retention, nutrient availability, and overall soil health.",
            "Ensure proper crop rotation with legumes like cowpeas or groundnuts to maintain soil health and improve nitrogen levels. Legumes fix nitrogen from the atmosphere, which benefits subsequent crops.",
            "Adopt natural pest control methods like neem oil spray, intercropping with repellent plants (e.g., marigold), or using biopesticides. These methods are environmentally friendly and sustainable.",
            "Bajra is a drought-tolerant crop, but providing supplemental irrigation during critical growth stages can significantly improve yields.",
            "Choose bajra varieties that are resistant to common diseases and pests in your area. This can help to reduce crop losses and the need for chemical interventions.",
            "Practice timely sowing and harvesting to optimize yields and minimize losses due to pests, diseases, or adverse weather conditions."
        ],
        "steps": {
            "June": [
            "Plow the field and incorporate compost or other organic matter for soil enrichment. This improves soil structure, aeration, and water-holding capacity.",
            "Sow bajra seeds with proper spacing (typically 45cm x 15cm) during the rainy season. Proper spacing ensures adequate airflow and reduces competition for resources."
            ],
            "July": [
            "Use organic mulch, such as straw or crop residues, to retain moisture and suppress weeds. Mulch also helps to regulate soil temperature and prevent soil erosion.",
            "Ensure regular weeding to remove competing weeds and allow bajra plants to access nutrients, water, and sunlight. Manual weeding or mechanical methods can be used."
            ],
            "August": [
            "Continue weeding and monitoring for pests and diseases. Bajra is relatively resistant to pests and diseases, but regular monitoring is still important.",
            "Thin out excess seedlings if necessary to maintain the desired plant spacing."
            ],
            "September": [
            "Harvest bajra when the grain heads turn golden and dry. This indicates that the crop is mature and ready for harvest.",
            "Thresh the harvested bajra to separate the grains from the stalks.",
            "Dry the bajra grains thoroughly to reduce moisture content and prevent spoilage during storage."
            ]
        },
        "weather": "Bajra is a drought-resistant crop that grows well in hot and dry climates with temperatures between 25°C to 40°C. It is well-suited to arid and semi-arid regions with limited rainfall.",
        "month_to_grow": "June to September (Monsoon season)",
        "month_to_grow_in_india": "June to September (Monsoon season), with regional variations."
    },

        "Jowar": {
        "tips": [
            "Apply green manure crops like sunhemp or dhaincha to increase organic matter in the soil. Green manure improves soil structure, water-holding capacity, and nutrient availability.",
            "Use crop residue mulching with sorghum stover or other organic materials to retain moisture, suppress weed growth, and regulate soil temperature. Mulching also helps to prevent soil erosion.",
            "Adopt biological pest control methods like releasing beneficial insects (e.g., ladybugs, lacewings) or using biopesticides (e.g., neem-based products) to manage pests in an environmentally friendly way.",
            "Practice intercropping with legumes like pigeon pea or cowpea to improve soil fertility and provide additional income. Legumes fix nitrogen from the atmosphere, benefiting the sorghum crop.",
            "Choose jowar varieties that are resistant to common diseases and pests in your region. This can help to reduce crop losses and the need for chemical interventions.",
            "Ensure timely sowing and harvesting to optimize yields and minimize losses due to pests, diseases, or adverse weather conditions."
        ],
        "steps": {
            "June": [
            "Enrich the soil with organic manure or compost before sowing. This improves soil fertility and provides essential nutrients for the jowar crop.",
            "Sow jowar seeds in rows with enough space (typically 45cm x 15cm) to grow. Proper spacing ensures adequate airflow and reduces competition for resources."
            ],
            "July": [
            "Apply water sparingly as jowar is drought-resistant. Overwatering can lead to root rot and other diseases. Irrigate only when necessary, especially during critical growth stages like flowering and grain filling.",
            "Weed the field regularly to remove competing weeds and allow jowar plants to access nutrients, water, and sunlight."
            ],
            "August": [
            "Weed manually and use natural pest control methods if necessary. Monitor for pests like shoot fly, stem borer, and aphids and take appropriate action if needed.",
            "Thin out excess seedlings if necessary to maintain the desired plant spacing."
            ],
            "September": [
            "Harvest when the grain heads turn brown and fully dry. This indicates that the jowar is mature and ready for harvest.",
            "Thresh the harvested jowar to separate the grains from the stalks.",
            "Dry the jowar grains thoroughly to reduce moisture content and prevent spoilage during storage."
            ]
        },
        "weather": "Jowar grows best in hot and dry climates with temperatures between 25°C to 35°C and requires minimal rainfall. It is a drought-tolerant crop and well-suited to arid and semi-arid regions.",
        "month_to_grow": "June to September (Monsoon season)",
        "month_to_grow_in_india": "June to September (Monsoon season), with regional variations."
        },
        "Ragi": {
        "tips": [
            "Use organic compost to improve soil fertility and moisture retention. Compost enhances soil structure, provides essential nutrients, and improves water-holding capacity.",
            "Plant ragi with leguminous crops like cowpea or green gram for nitrogen fixation in the soil. Legumes fix nitrogen from the atmosphere, which benefits the ragi crop.",
            "Control weeds with organic mulching methods using straw, crop residues, or other organic materials. Mulching suppresses weed growth, conserves moisture, and regulates soil temperature.",
            "Ragi is a drought-tolerant crop, but providing supplemental irrigation during critical growth stages can improve yields.",
            "Choose ragi varieties that are resistant to common diseases and pests in your area. This can help to reduce crop losses and the need for chemical interventions.",
            "Ensure timely sowing and harvesting to optimize yields and minimize losses due to pests, diseases, or adverse weather conditions."
        ],
        "steps": {
            "June": [
            "Prepare the soil with compost or green manure. This improves soil fertility and provides essential nutrients for the ragi crop.",
            "Sow ragi seeds in rows or broadcast them evenly. Proper spacing is important for good growth and yield."
            ],
            "July": [
            "Irrigate sparingly as ragi is drought-tolerant. Overwatering can be harmful. Irrigate only when necessary, especially during critical growth stages.",
            "Weed the field regularly to remove competing weeds and allow ragi plants to access nutrients, water, and sunlight."
            ],
            "August": [
            "Weed manually or use mulch to control weed growth. Weeds can significantly reduce ragi yields if they are not controlled effectively.",
            "Monitor for pests and diseases and take appropriate action if necessary."
            ],
            "September": [
            "Harvest when grains are mature and dry. This indicates that the ragi is ready for harvest.",
            "Thresh the harvested ragi to separate the grains from the stalks.",
            "Dry the ragi grains thoroughly to reduce moisture content and prevent spoilage during storage."
            ]
        },
        "weather": "Ragi grows best in semi-arid regions with temperatures between 20°C to 35°C and minimal rainfall. It is a highly drought-tolerant crop and well-suited to dry and marginal lands.",
        "month_to_grow": "June to September (Monsoon season)",
        "month_to_grow_in_india": "June to September (Monsoon season), with regional variations."
    },

        "Barley": {
        "tips": [
            "Incorporate organic compost to enhance soil fertility. Compost improves soil structure, water retention, and nutrient availability, creating a healthy environment for barley growth.",
            "Rotate barley with pulses (legumes) like peas, beans, or lentils to improve soil structure and nutrients naturally. Pulses fix nitrogen in the soil, benefiting subsequent barley crops and reducing the need for synthetic fertilizers.",
            "Control pests using natural insect repellents like neem oil, garlic spray, or companion planting (e.g., marigolds). These methods are environmentally friendly and reduce the risk of chemical residues on the barley grains.",
            "Choose barley varieties that are well-suited to your local climate and growing conditions. Some varieties are more resistant to diseases and pests, while others are better adapted to specific soil types.",
            "Ensure proper seedbed preparation by plowing and harrowing the field to create a fine tilth. This helps to ensure good seed-to-soil contact and uniform germination.",
            "Sow barley seeds at the recommended depth and spacing to optimize plant growth and yield. Proper spacing allows for adequate airflow and reduces competition for resources."
        ],
        "steps": {
            "October": [
            "Plow the field and apply compost for better soil structure. Plowing helps to aerate the soil and incorporate organic matter, while compost provides essential nutrients for barley growth.",
            "Sow barley seeds in rows during the cooler season. The optimal sowing time may vary depending on the specific region and climate."
            ],
            "November": [
            "Water sparingly but ensure moisture retention through mulch. Barley is relatively drought-tolerant, but adequate moisture is important, especially during germination and early growth stages. Mulching helps to conserve moisture and suppress weed growth.",
            "Use natural pest control methods to manage insects. Monitor the barley crop regularly for signs of pests and take appropriate action if necessary."
            ],
            "December": [
            "Harvest when the crop turns golden and the grains are dry. This indicates that the barley is mature and ready for harvest.",
            "Thresh the harvested barley to separate the grains from the stalks.",
            "Dry the barley grains thoroughly to reduce moisture content and prevent spoilage during storage."
            ]
        },
        "weather": "Barley grows best in cool temperate climates with temperatures between 15°C to 20°C and moderate rainfall. It is a winter crop in many regions.",
        "month_to_grow": "October to December (Winter season)",
        "month_to_grow_in_india": "November to January (Winter season), with regional variations."
        },
        "Chickpea": {
        "tips": [
            "Use crop rotation with cereals like wheat or barley to enhance soil nitrogen levels naturally. This reduces the need for synthetic nitrogen fertilizers and improves soil health.",
            "Apply compost and well-decomposed manure to increase soil fertility. These organic materials provide essential nutrients and improve soil structure, water-holding capacity, and drainage.",
            "Control fungal diseases like wilt and blight with organic sprays like neem oil, garlic spray, or by using disease-resistant chickpea varieties. Early detection and treatment are crucial for managing fungal diseases.",
            "Choose chickpea varieties that are suitable for your local climate and soil conditions. Some varieties are more resistant to diseases and pests, while others are better adapted to specific soil types.",
            "Ensure proper seedbed preparation by plowing and harrowing the field to create a fine tilth. This helps to ensure good seed-to-soil contact and uniform germination.",
            "Sow chickpea seeds at the recommended depth and spacing to optimize plant growth and yield. Proper spacing allows for adequate airflow and reduces competition for resources."
        ],
        "steps": {
            "October": [
            "Prepare the soil with compost and manure for better fertility. This provides essential nutrients for the chickpea crop and improves soil structure.",
            "Sow chickpea seeds with proper spacing in rows. The recommended spacing may vary depending on the chickpea variety and local conditions."
            ],
            "November": [
            "Irrigate moderately, avoiding waterlogging. Chickpeas are relatively drought-tolerant, but adequate moisture is important, especially during germination and pod development. Overwatering can lead to root rot and other diseases.",
            "Weed the field regularly to remove competing weeds and allow chickpea plants to access nutrients, water, and sunlight."
            ],
            "December": [
            "Weed manually and control pests with natural methods if necessary. Monitor for pests like pod borer and aphids and take appropriate action if needed.",
            "Monitor for diseases and take appropriate action if necessary. Early detection and treatment are crucial for managing diseases."
            ],
            "January": [
            "Harvest when the pods dry and turn brown. This indicates that the chickpeas are mature and ready for harvest.",
            "Thresh the harvested chickpeas to separate the seeds from the pods.",
            "Dry the chickpea seeds thoroughly to reduce moisture content and prevent spoilage during storage."
            ]
        },
        "weather": "Chickpeas grow well in dry, semi-arid climates with temperatures between 20°C to 30°C. They are a winter crop in many regions.",
        "month_to_grow": "October to February (Winter season)",
        "month_to_grow_in_india": "October to February (Winter season), with regional variations."
    },

        "Pigeon Pea": {
        "tips": [
            "Use compost or farmyard manure (FYM) for better soil health. These organic materials improve soil structure, water retention, and nutrient availability, creating a favorable environment for pigeon pea growth.",
            "Practice intercropping with cereals like sorghum or maize for better nitrogen fixation. Pigeon pea is a legume that fixes nitrogen from the atmosphere, benefiting the cereal crop and reducing the need for synthetic nitrogen fertilizers.",
            "Apply organic pest control solutions like neem extract, garlic spray, or other biopesticides. These methods are environmentally friendly and reduce the risk of chemical residues on the pigeon peas.",
            "Choose pigeon pea varieties that are resistant to common diseases and pests in your region. This can help to reduce crop losses and the need for chemical interventions.",
            "Ensure proper spacing between pigeon pea plants to allow for adequate airflow, sunlight penetration, and nutrient uptake. This helps to prevent overcrowding and reduces the risk of diseases.",
            "Pigeon pea is relatively drought-tolerant, but providing supplemental irrigation during critical growth stages (e.g., flowering and pod development) can significantly improve yields."
        ],
        "steps": {
            "June": [
            "Enrich the soil with organic compost before sowing. This provides essential nutrients for the pigeon pea crop and improves soil structure.",
            "Sow pigeon pea seeds with adequate spacing (typically 60cm x 30cm) for airflow. Proper spacing ensures that plants have enough room to grow and prevents overcrowding."
            ],
            "July": [
            "Apply organic mulch to retain moisture and suppress weeds. Mulch helps to conserve water, reduce weed competition, and regulate soil temperature.",
            "Thin out excess seedlings if necessary to maintain the desired plant spacing."
            ],
            "August": [
            "Weed regularly to remove competing weeds and allow pigeon pea plants to access nutrients, water, and sunlight. Weeds can significantly reduce yields if they are not controlled effectively.",
            "Control pests using neem oil or garlic spray. Monitor for pests like pod borer and aphids and take appropriate action if needed."
            ],
            "September": [
            "Harvest when the pods dry and mature. This indicates that the pigeon peas are ready for harvest.",
            "Thresh the harvested pigeon peas to separate the seeds from the pods.",
            "Dry the pigeon pea seeds thoroughly to reduce moisture content and prevent spoilage during storage."
            ]
        },
        "weather": "Pigeon pea grows well in hot, dry climates with temperatures between 25°C to 35°C and low to moderate rainfall. It is a drought-tolerant crop and well-suited to semi-arid and arid regions.",
        "month_to_grow": "June to September (Monsoon season)",
        "month_to_grow_in_india": "June to September (Monsoon season), with regional variations."
        },
        "Potato": {
        "tips": [
            "Use well-decomposed compost or manure to improve soil fertility. These organic materials provide essential nutrients for the potato crop and improve soil structure, drainage, and water-holding capacity.",
            "Rotate crops to avoid pests and improve soil health. Rotating potatoes with other crops helps to break pest and disease cycles and improves soil fertility.",
            "Control pests like aphids, potato tuber moth, and early blight using neem oil, insecticidal soap, or other natural sprays. Early detection and treatment are crucial for managing potato pests and diseases.",
            "Choose potato varieties that are well-suited to your local climate and growing conditions. Some varieties are more resistant to diseases and pests, while others are better adapted to specific soil types.",
            "Ensure proper soil drainage to prevent waterlogging, which can lead to potato rot and other diseases. Potatoes prefer well-drained soil.",
            "Hill up the potato plants regularly to encourage tuber development and protect the tubers from sunlight, which can cause them to turn green."
        ],
        "steps": {
            "October": [
            "Prepare the soil with organic compost or farmyard manure. This provides essential nutrients for the potato crop and improves soil structure.",
            "Plant seed potatoes (tubers with eyes) in well-drained soil with organic mulch. The seed potatoes should be planted at a depth of about 10-15 cm and spaced appropriately."
            ],
            "November": [
            "Water regularly to keep the soil consistently moist, but avoid overwatering. Potatoes need adequate moisture, especially during tuber development, but waterlogging can be harmful.",
            "Control weeds with mulch or manual weeding. Weeds compete with potatoes for nutrients, water, and sunlight, so effective weed control is essential."
            ],
            "December": [
            "Control pests naturally using neem oil or insecticidal soap. Monitor for pests regularly and take appropriate action if necessary.",
            "Hill up the potato plants by adding soil around the stems. This encourages tuber development and protects the tubers from sunlight."
            ],
            "January": [
            "Harvest potatoes when the vines begin to die back. This indicates that the potatoes are mature and ready for harvest.",
            "Dig up the potatoes carefully to avoid damaging the tubers.",
            "Cure the potatoes by allowing them to dry in a cool, dark, and well-ventilated place for a few days. This helps to toughen the skin and prevent spoilage during storage."
            ]
        },
        "weather": "Potatoes grow best in cool climates with temperatures between 15°C to 20°C and require moderate rainfall. They are a cool-season crop in many regions.",
        "month_to_grow": "October to December (Winter season)",
        "month_to_grow_in_india": "October to December (Winter season), with regional variations."
    },

        "Tomato": {
        "tips": [
            "Use compost to enrich soil before planting. Compost improves soil structure, water retention, and nutrient availability, creating a healthy environment for tomato plants.",
            "Use organic mulch (e.g., straw, wood chips, or grass clippings) to retain soil moisture and suppress weeds. Mulch also helps to regulate soil temperature and prevent soil erosion.",
            "Control pests like aphids and caterpillars using neem oil, insecticidal soap, or other natural sprays. Early detection and treatment are crucial for managing tomato pests.",
            "Provide support for tomato plants using stakes, cages, or trellises. This helps to prevent the plants from sprawling on the ground, which can lead to fruit rot and other problems.",
            "Prune tomato plants regularly to remove suckers (side shoots that grow between the main stem and branches). Pruning helps to improve airflow, reduce disease risk, and encourage fruit production.",
            "Water tomato plants deeply and consistently, especially during flowering and fruiting. Avoid overhead watering, as this can lead to fungal diseases."
        ],
        "steps": {
            "March": [
            "Prepare the soil with compost for nutrient enrichment. This provides essential nutrients for tomato plants and improves soil structure.",
            "Plant tomato seedlings with proper spacing (typically 45-60 cm apart) in well-drained soil. The seedlings should be planted slightly deeper than they were in their original containers."
            ],
            "April": [
            "Water consistently but avoid waterlogging, using mulch for moisture retention. Tomatoes need consistent moisture, especially during flowering and fruiting, but overwatering can be harmful.",
            "Provide support for tomato plants using stakes, cages, or trellises. This helps to prevent the plants from sprawling on the ground."
            ],
            "May": [
            "Control pests using natural sprays and weed manually. Monitor for pests regularly and take appropriate action if necessary.",
            "Prune tomato plants regularly to remove suckers and improve airflow."
            ],
            "June": [
            "Harvest tomatoes when fully ripe. Ripe tomatoes are firm, have a deep color, and are slightly soft to the touch.",
            "Continue to monitor for pests and diseases and take appropriate action if necessary."
            ]
        },
        "weather": "Tomatoes thrive in warm climates with temperatures between 18°C to 30°C and moderate rainfall. They are a warm-season crop.",
        "month_to_grow": "March to May (Summer season)",
        "month_to_grow_in_india": "March to May (Summer season), with regional variations."
        },
        "Cabbage": {
        "tips": [
            "Apply organic manure to enrich soil and improve cabbage growth. Manure provides essential nutrients and improves soil structure, water-holding capacity, and drainage.",
            "Control weeds with organic mulching methods using straw, wood chips, or other organic materials. Mulch helps to suppress weed growth, conserve moisture, and regulate soil temperature.",
            "Use neem oil to manage common pests like aphids, cabbage worms, and diamondback moths. Neem oil is a natural insecticide that is effective against a wide range of pests.",
            "Choose cabbage varieties that are well-suited to your local climate and growing conditions. Some varieties are more resistant to diseases and pests, while others are better adapted to specific soil types.",
            "Ensure proper spacing between cabbage plants to allow for adequate airflow, sunlight penetration, and nutrient uptake. This helps to prevent overcrowding and reduces the risk of diseases.",
            "Water cabbage plants regularly, especially during head formation. Cabbages need consistent moisture for healthy growth and head development."
        ],
        "steps": {
            "October": [
            "Prepare the soil by applying compost and organic manure. This provides essential nutrients for the cabbage crop and improves soil structure.",
            "Plant cabbage seedlings in well-drained soil with proper spacing (typically 45-60 cm apart). The seedlings should be planted slightly deeper than they were in their original containers."
            ],
            "November": [
            "Use organic mulch to control weeds and retain moisture. Mulch helps to suppress weed growth, conserve water, and regulate soil temperature.",
            "Water cabbage plants regularly, especially during head formation."
            ],
            "December": [
            "Control pests using neem oil or natural sprays. Monitor for pests regularly and take appropriate action if necessary.",
            "Fertilize cabbage plants with a balanced fertilizer if needed, based on soil test results and crop requirements."
            ],
            "January": [
            "Harvest cabbage heads when they are firm and fully grown. The heads should be compact and heavy for their size.",
            "Cut the cabbage heads off at the base, leaving a few outer leaves attached to protect the head.",
            "Store cabbage in a cool, dry place for short-term storage."
            ]
        },
        "weather": "Cabbage grows best in cool weather with temperatures between 15°C to 20°C and moderate rainfall. It is a cool-season crop.",
        "month_to_grow": "October to February (Winter season)",
        "month_to_grow_in_india": "October to February (Winter season), with regional variations."
        },
        "Soybean": {
        "tips": [
            "Use organic manure, such as compost or farmyard manure (FYM), to enrich soil fertility. Organic matter improves soil structure, water retention, and nutrient availability, creating a healthy environment for soybean growth.",
            "Practice crop rotation with cereals like wheat or maize to maintain soil health and break pest and disease cycles. Crop rotation also helps to improve soil structure and nutrient balance.",
            "Control pests using neem oil, garlic spray, or other natural pest control methods. Early detection and treatment are crucial for managing soybean pests like aphids, pod borers, and whiteflies.",
            "Choose soybean varieties that are well-suited to your local climate and growing conditions. Some varieties are more resistant to diseases and pests, while others are better adapted to specific soil types.",
            "Ensure proper seedbed preparation by plowing and harrowing the field to create a fine tilth. This helps to ensure good seed-to-soil contact and uniform germination.",
            "Soybeans are legumes, so they can fix nitrogen from the atmosphere. However, inoculation with rhizobium bacteria can further enhance nitrogen fixation and improve yields."
        ],
        "steps": {
            "June": [
            "Prepare the soil with compost before sowing. This provides essential nutrients for the soybean crop and improves soil structure.",
            "Sow soybean seeds with proper spacing (typically 45-60 cm between rows and 10-15 cm between plants) in well-drained soil. Proper spacing ensures adequate airflow and reduces competition for resources."
            ],
            "July": [
            "Water moderately but avoid waterlogging. Soybeans need adequate moisture, especially during flowering and pod development, but overwatering can be harmful.",
            "Weed the field regularly to remove competing weeds and allow soybean plants to access nutrients, water, and sunlight."
            ],
            "August": [
            "Weed manually and control pests using natural methods if necessary. Monitor for pests regularly and take appropriate action if needed.",
            "Monitor for diseases and take appropriate action if necessary. Early detection and treatment are crucial for managing diseases."
            ],
            "September": [
            "Harvest when the pods turn brown and dry naturally. This indicates that the soybeans are mature and ready for harvest.",
            "Thresh the harvested soybeans to separate the seeds from the pods.",
            "Dry the soybean seeds thoroughly to reduce moisture content and prevent spoilage during storage."
            ]
        },
        "weather": "Soybean grows well in warm climates with temperatures between 20°C to 30°C and moderate rainfall. It is a warm-season crop.",
        "month_to_grow": "June to September (Monsoon season)",
        "month_to_grow_in_india": "June to September (Monsoon season), with regional variations."
        },
        "Groundnut": {
        "tips": [
            "Use farmyard manure (FYM) to improve soil fertility. FYM provides essential nutrients for the groundnut crop and improves soil structure, water-holding capacity, and drainage.",
            "Practice crop rotation with cereals like sorghum or maize for better soil health. Crop rotation helps to break pest and disease cycles and improves soil fertility.",
            "Control pests using neem oil, biological pest control methods, or other natural sprays. Groundnuts are susceptible to pests like aphids, leaf miners, and termites.",
            "Choose groundnut varieties that are well-suited to your local climate and soil conditions. Some varieties are more resistant to diseases and pests, while others are better adapted to specific soil types.",
            "Ensure proper seedbed preparation by plowing and harrowing the field to create a fine tilth. This helps to ensure good seed-to-soil contact and uniform germination.",
            "Groundnuts are legumes, so they can fix nitrogen from the atmosphere. However, inoculation with rhizobium bacteria can further enhance nitrogen fixation and improve yields."
        ],
        "steps": {
            "June": [
            "Plow the field and apply compost for soil enrichment. This provides essential nutrients for the groundnut crop and improves soil structure.",
            "Sow groundnut seeds in well-drained soil with proper spacing (typically 45-60 cm between rows and 15-20 cm between plants). Proper spacing ensures adequate airflow and reduces competition for resources."
            ],
            "July": [
            "Water moderately, avoiding waterlogging. Groundnuts need adequate moisture, especially during flowering and pegging (when the pegs grow down into the soil to form the pods), but overwatering can be harmful.",
            "Weed the field regularly to remove competing weeds and allow groundnut plants to access nutrients, water, and sunlight."
            ],
            "August": [
            "Weed manually and apply organic mulch to suppress weeds. Mulch also helps to conserve moisture and regulate soil temperature.",
            "Earth up the groundnut plants by adding soil around the base of the plants. This helps to encourage pod development and protect the pods from sunlight."
            ],
            "September": [
            "Harvest when the leaves turn yellow and the pods are dry. This indicates that the groundnuts are mature and ready for harvest.",
            "Dig up the groundnut plants carefully to avoid damaging the pods.",
            "Dry the groundnut pods thoroughly in the sun to reduce moisture content and prevent spoilage during storage."
            ]
        },
        "weather": "Groundnut thrives in hot, semi-arid climates with temperatures between 25°C to 35°C and moderate rainfall. It is a warm-season crop.",
        "month_to_grow": "June to September (Monsoon season)",
        "month_to_grow_in_india": "June to September (Monsoon season), with regional variations."
        },  
        "Mustard": {
        "tips": [
            "Apply compost and organic manure for better growth. These organic materials improve soil structure, water retention, and nutrient availability, creating a healthy environment for mustard plants.",
            "Use crop rotation with legumes like peas, beans, or lentils to improve nitrogen levels in the soil naturally. Legumes fix nitrogen from the atmosphere, which benefits subsequent mustard crops and reduces the need for synthetic fertilizers.",
            "Control pests using neem oil, garlic spray, or other organic pest control methods. Mustard plants are susceptible to pests like aphids, sawflies, and cabbage worms.",
            "Choose mustard varieties that are well-suited to your local climate and growing conditions. Some varieties are more resistant to diseases and pests, while others are better adapted to specific soil types.",
            "Ensure proper seedbed preparation by plowing and harrowing the field to create a fine tilth. This helps to ensure good seed-to-soil contact and uniform germination.",
            "Sow mustard seeds at the recommended depth and spacing to optimize plant growth and yield. Proper spacing allows for adequate airflow and reduces competition for resources."
        ],
        "steps": {
            "September": [
            "Prepare the soil with organic compost. This provides essential nutrients for the mustard crop and improves soil structure.",
            "Sow mustard seeds in well-drained soil. The seeds should be sown at a shallow depth (about 1-2 cm) and spaced appropriately."
            ],
            "October": [
            "Water consistently to keep the soil moist, but avoid overwatering. Mustard plants need adequate moisture, especially during germination and early growth stages.",
            "Use organic mulch to control weeds. Mulch helps to suppress weed growth, conserve moisture, and regulate soil temperature."
            ],
            "November": [
            "Weed manually to remove any remaining weeds. Weeds can compete with mustard plants for nutrients, water, and sunlight.",
            "Apply neem oil or other organic pest control methods if necessary. Monitor for pests regularly and take appropriate action if needed."
            ],
            "December": [
            "Harvest when mustard seeds turn brown and dry. This indicates that the mustard is mature and ready for harvest.",
            "Cut the mustard plants near the base and allow them to dry further in the sun.",
            "Thresh the mustard plants to separate the seeds from the pods."
            ]
        },
        "weather": "Mustard grows best in cool climates with temperatures between 10°C to 25°C and requires minimal rainfall. It is a winter crop in many regions.",
        "month_to_grow": "October to February (Winter season)",
        "month_to_grow_in_india": "October to February (Winter season), with regional variations."
        },
        "Sunflower": {
        "tips": [
            "Use organic manure, such as compost or farmyard manure (FYM), to enhance soil fertility. Organic matter improves soil structure, water retention, and nutrient availability, creating a healthy environment for sunflower growth.",
            "Plant sunflowers with proper spacing (typically 45-60 cm apart) for better growth. Proper spacing allows for adequate airflow, sunlight penetration, and nutrient uptake, preventing overcrowding and reducing the risk of diseases.",
            "Control pests like aphids, sunflower moths, and birds using neem oil, insecticidal soap, or other organic pest control methods. Early detection and treatment are crucial for managing sunflower pests.",
            "Choose sunflower varieties that are well-suited to your local climate and growing conditions. Some varieties are more resistant to diseases and pests, while others are better adapted to specific soil types.",
            "Provide support for sunflower plants, especially tall varieties, using stakes or trellises. This helps to prevent the plants from lodging (falling over) due to wind or heavy rain.",
            "Water sunflower plants deeply and consistently, especially during flowering and seed development. Sunflowers need adequate moisture for healthy growth and seed production."
        ],
        "steps": {
            "March": [
            "Prepare the soil by mixing in compost. This provides essential nutrients for the sunflower crop and improves soil structure.",
            "Plant sunflower seeds in well-drained soil. The seeds should be planted at a depth of about 2-3 cm."
            ],
            "April": [
            "Water regularly but avoid over-watering. Sunflowers need consistent moisture, especially during flowering and seed development, but overwatering can be harmful.",
            "Thin out excess seedlings if necessary to maintain the desired plant spacing."
            ],
            "May": [
            "Control weeds and pests using organic methods like neem oil. Weeds can compete with sunflowers for nutrients, water, and sunlight.",
            "Provide support for sunflower plants if necessary."
            ],
            "June": [
            "Harvest when sunflower heads are fully mature and the backs of the heads turn brown. The seeds should be plump and the petals should start to fall off.",
            "Cut the sunflower heads off the stalks, leaving a portion of the stem attached.",
            "Dry the sunflower heads in a well-ventilated area until the backs are completely brown and the seeds are dry."
            ]
        },
        "weather": "Sunflowers thrive in warm, sunny weather with temperatures between 20°C to 30°C and moderate rainfall. They are a warm-season crop.",
        "month_to_grow": "March to June (Summer season)",
        "month_to_grow_in_india": "March to June (Summer season), with regional variations."
        },

        "Cotton": {
        "tips": [
            "Use farmyard manure (FYM) to improve soil health. FYM provides essential nutrients for the cotton crop and improves soil structure, water-holding capacity, and drainage.",
            "Rotate cotton crops with legumes like groundnut, soybean, or pigeon pea to improve nitrogen levels in the soil naturally. Legumes fix nitrogen from the atmosphere, benefiting subsequent cotton crops and reducing the need for synthetic nitrogen fertilizers.",
            "Control pests using neem oil, garlic spray, or other natural insecticides. Cotton is susceptible to a variety of pests, including bollworms, aphids, and whiteflies. Early detection and treatment are crucial for managing these pests.",
            "Choose cotton varieties that are well-suited to your local climate and growing conditions. Some varieties are more resistant to diseases and pests, while others are better adapted to specific soil types.",
            "Ensure proper seedbed preparation by plowing and harrowing the field to create a fine tilth. This helps to ensure good seed-to-soil contact and uniform germination.",
            "Cotton requires adequate moisture, especially during flowering and boll development. However, overwatering can be harmful, so it's important to ensure proper drainage."
        ],
        "steps": {
            "April": [
            "Prepare the soil with compost and organic manure. This provides essential nutrients for the cotton crop and improves soil structure.",
            "Sow cotton seeds in well-drained soil with proper spacing (typically 60-90 cm between rows and 30-45 cm between plants). Proper spacing ensures adequate airflow and reduces competition for resources."
            ],
            "May": [
            "Water regularly but ensure proper drainage. Cotton needs consistent moisture, especially during flowering and boll development, but overwatering can be harmful.",
            "Weed the field regularly to remove competing weeds and allow cotton plants to access nutrients, water, and sunlight."
            ],
            "June": [
            "Weed the field and apply neem oil for pest control. Monitor for pests regularly and take appropriate action if needed.",
            "Thin out excess seedlings if necessary to maintain the desired plant spacing."
            ],
            "July": [
            "Harvest cotton when the bolls open and the cotton is dry. This indicates that the cotton is mature and ready for harvest.",
            "Pick the cotton from the open bolls, leaving the burrs (outer coverings) behind.",
            "Dry the harvested cotton thoroughly to reduce moisture content and prevent spoilage during storage."
            ]
        },
        "weather": "Cotton grows well in warm climates with temperatures between 25°C to 35°C and moderate rainfall. It is a warm-season crop.",
        "month_to_grow": "April to July (depending on the variety and region)",
        "month_to_grow_in_india": "June to October (Monsoon season), with regional variations."
        },
        "Sesame": {
        "tips": [
            "Use compost to enhance soil fertility. Compost improves soil structure, water retention, and nutrient availability, creating a healthy environment for sesame growth.",
            "Sow sesame seeds in well-drained, sandy soils. Sesame prefers light-textured soils that are well-drained.",
            "Control pests using neem oil, garlic spray, or other natural pest control methods. Sesame is susceptible to pests like aphids, leafhoppers, and sesame gall midge.",
            "Choose sesame varieties that are well-suited to your local climate and growing conditions. Some varieties are more resistant to diseases and pests, while others are better adapted to specific soil types.",
            "Ensure proper seedbed preparation by plowing and harrowing the field to create a fine tilth. This helps to ensure good seed-to-soil contact and uniform germination.",
            "Sesame is relatively drought-tolerant, but providing supplemental irrigation during critical growth stages (e.g., flowering and capsule development) can significantly improve yields."
        ],
        "steps": {
            "June": [
            "Prepare the soil with compost and ensure proper drainage. This provides essential nutrients for the sesame crop and improves soil structure.",
            "Sow sesame seeds in rows with proper spacing (typically 30-45 cm between rows and 10-15 cm between plants). Proper spacing ensures adequate airflow and reduces competition for resources."
            ],
            "July": [
            "Water regularly but avoid over-watering. Sesame needs adequate moisture, especially during flowering and capsule development, but overwatering can be harmful.",
            "Weed the field regularly to remove competing weeds and allow sesame plants to access nutrients, water, and sunlight."
            ],
            "August": [
            "Weed manually and apply neem oil for pest control. Monitor for pests regularly and take appropriate action if needed.",
            "Thin out excess seedlings if necessary to maintain the desired plant spacing."
            ],
            "September": [
            "Harvest when sesame pods turn dry and split open. This indicates that the sesame seeds are mature and ready for harvest.",
            "Cut the sesame plants near the base and allow them to dry further in the sun.",
            "Thresh the sesame plants to separate the seeds from the pods."
            ]
        },
        "weather": "Sesame thrives in warm, semi-arid climates with temperatures between 25°C to 35°C. It is a drought-tolerant crop and well-suited to regions with limited rainfall.",
        "month_to_grow": "June to September (Monsoon season)",
        "month_to_grow_in_india": "June to September (Monsoon season), with regional variations."
        },

        
        "Sugarcane": {
        "tips": [
            "Use organic fertilizers, such as compost, farmyard manure (FYM), or vermicompost, to maintain soil health. Organic matter improves soil structure, water retention, and nutrient availability, creating a favorable environment for sugarcane growth.",
            "Plant sugarcane in rows for better growth and airflow. Row planting allows for easier access for weeding, irrigation, and harvesting, and it also helps to prevent the spread of diseases and pests.",
            "Control pests using organic insecticides like neem oil, garlic spray, or other biopesticides. Sugarcane is susceptible to various pests, including borers, scales, and whiteflies. Early detection and treatment are crucial for managing these pests.",
            "Choose sugarcane varieties that are well-suited to your local climate and growing conditions. Some varieties are more resistant to diseases and pests, while others are better adapted to specific soil types.",
            "Ensure proper seedbed preparation by plowing and harrowing the field to create a fine tilth. This helps to ensure good seed-to-soil contact and uniform germination of the sugarcane setts (seed pieces).",
            "Provide adequate irrigation, especially during the early stages of growth and during the grand growth period. Sugarcane requires a consistent supply of water for optimal growth and yield."
        ],
        "steps": {
            "March": [
            "Prepare the soil with compost and organic manure. This provides essential nutrients for the sugarcane crop and improves soil structure.",
            "Plant sugarcane stalks (setts) in well-drained soil. The setts should be planted at a depth of about 25-30 cm and spaced appropriately (typically 60-90 cm between rows and 30-45 cm between setts)."
            ],
            "April": [
            "Water regularly, but avoid waterlogging. Sugarcane needs consistent moisture, especially during the early stages of growth, but overwatering can be harmful.",
            "Weed the field regularly to remove competing weeds and allow sugarcane plants to access nutrients, water, and sunlight."
            ],
            "May": [
            "Weed the field and control pests with neem oil. Monitor for pests regularly and take appropriate action if needed.",
            "Apply top dressing of fertilizer if needed, based on soil test results and crop requirements."
            ],
            "June": [
            "Harvest when sugarcane stalks are thick and mature. The maturity of sugarcane is typically determined by the Brix level (sugar content), which is measured using a refractometer.",
            "Cut the sugarcane stalks near the base and remove the leaves.",
            "Transport the harvested sugarcane to the sugar mill for processing as soon as possible to prevent sugar loss."
            ]
        },
        "weather": "Sugarcane requires a warm tropical climate with temperatures between 20°C to 35°C and an ample water supply. It is a long-duration crop, typically taking 10-18 months to mature.",
        "month_to_grow": "Planting time varies depending on the region, but in many parts of India, it is done between January and March (Spring season).",
        "month_to_grow_in_india": "Planting time varies depending on the region, but in many parts of India, it is done between January and March (Spring season)."
        },
        "Turmeric": {
        "tips": [
            "Use organic manure, such as compost or farmyard manure (FYM), to enhance soil fertility. Organic matter improves soil structure, water retention, and nutrient availability, creating a healthy environment for turmeric growth.",
            "Ensure proper drainage for turmeric, as it doesn't like waterlogging. Waterlogged conditions can lead to rhizome rot and other diseases.",
            "Control pests using neem oil, garlic spray, or other natural pest control methods. Turmeric is susceptible to pests like shoot borer, rhizome scale, and nematodes.",
            "Choose turmeric varieties that are well-suited to your local climate and growing conditions. Some varieties are more resistant to diseases and pests, while others are better adapted to specific soil types.",
            "Provide shade for turmeric plants, especially during hot and dry periods. Turmeric prefers partial shade and can be damaged by direct sunlight.",
            "Mulch turmeric plants with organic materials like straw or dried leaves to conserve moisture, suppress weed growth, and regulate soil temperature."
        ],
        "steps": {
            "April": [
            "Prepare the soil by adding compost or organic manure. This provides essential nutrients for the turmeric crop and improves soil structure.",
            "Plant turmeric rhizomes (seed rhizomes) in well-drained soil. The rhizomes should be planted at a depth of about 5-7 cm and spaced appropriately (typically 30-45 cm between rows and 25-30 cm between plants)."
            ],
            "May": [
            "Water regularly and ensure proper drainage. Turmeric needs consistent moisture, especially during the growing season, but waterlogging can be harmful.",
            "Weed the field regularly to remove competing weeds and allow turmeric plants to access nutrients, water, and sunlight."
            ],
            "June": [
            "Control weeds and pests using organic methods like neem oil. Monitor for pests regularly and take appropriate action if needed.",
            "Provide shade for turmeric plants if necessary."
            ],
            "July": [
            "Harvest turmeric after 7-9 months when the leaves start to dry and turn yellow. This indicates that the turmeric is mature and ready for harvest.",
            "Dig up the turmeric rhizomes carefully, avoiding damage to the rhizomes.",
            "Clean the rhizomes and cure them by boiling them in water and then drying them in the sun."
            ]
        },
        "weather": "Turmeric prefers warm, humid climates with temperatures between 20°C to 30°C. It is a shade-loving plant and thrives in areas with moderate rainfall.",
        "month_to_grow": "Planting time varies depending on the region, but in many parts of India, it is done between April and May.",
        "month_to_grow_in_india": "Planting time varies depending on the region, but in many parts of India, it is done between April and May."
        },
    }


    st.markdown('<div class="animated">', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <h1>Organic Farming Guide</h1>
    <p>Step-by-step instructions for growing crops organically.</p>
    </div>
    """, unsafe_allow_html=True)

    sorted_plants = sorted(organic_crops_steps.keys())
    plant = st.selectbox("Choose a Crop", sorted_plants)

    if plant:
        crop_info = organic_crops_steps[plant]
        
        st.markdown(f"""
        <div class="card">
        <h2>Organic {plant} Cultivation</h2>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📅 Growing Guide", "ℹ Basic Info", "💡 Tips"])
        
        with tab1:
            st.markdown(f"""
            <div class="card">
            <h3>Monthly Growing Guide</h3>
            """, unsafe_allow_html=True)
            
            for month, steps in crop_info["steps"].items():
                st.markdown(f"**{month}**")
                for step in steps:
                    st.write(f"- {step}")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with tab2:
            st.markdown(f"""
            <div class="card">
            <h3>Crop Information</h3>
            <p><strong>Best Growing Season:</strong> {crop_info['month_to_grow_in_india']}</p>
            <p><strong>Ideal Weather:</strong> {crop_info['weather']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown(f"""
            <div class="card">
            <h3>Organic Farming Tips</h3>
            <ul>
            """, unsafe_allow_html=True)
            
            for tip in crop_info["tips"]:
                st.write(f"- {tip}")
            
            st.markdown("""
            </ul>
            </div>
            """, unsafe_allow_html=True)

elif app_mode == "About":
    st.markdown('<div class="animated">', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <h1>About Vriksha Rakshak</h1>
    <p>AI-powered agricultural assistant for sustainable farming.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    <h2>Our Mission</h2>
    <p>Vriksha Rakshak aims to empower farmers with AI tools to improve crop yields, reduce losses, and promote sustainable farming practices through accessible technology.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    <h2>Technology Stack</h2>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
    <div class="card" style="padding: 1rem;">
    <h4>Machine Learning</h4>
    <ul>
    <li>TensorFlow/Keras</li>
    <li>Scikit-learn</li>
    <li>Transformers</li>
    </ul>
    </div>
    <div class="card" style="padding: 1rem;">
    <h4>Web Framework</h4>
    <ul>
    <li>Streamlit</li>
    <li>Folium</li>
    </ul>
    </div>
    <div class="card" style="padding: 1rem;">
    <h4>APIs</h4>
    <ul>
    <li>OpenWeatherMap</li>
    <li>Google Gemini</li>
    </ul>
    </div>
    <div class="card" style="padding: 1rem;">
    <h4>Data</h4>
    <ul>
    <li>PlantVillage Dataset</li>
    <li>FAO Soil Data</li>
    </ul>
    </div>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    <h2>Developed</h2>
    <p>Vriksha Rakshak was developed by a team of AI engineers from Bennett University committed to bridging technology and farming.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <h2>Development Team</h2>
    <ul>
    <li>Advitya Singh</li>
    <li>Shivanshi</li>
    <li>Vinith Reddy</li>
    <li>Yaswanth Kumar</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

elif app_mode == "Crop Recommendation":
    genai.configure(api_key="AIzaSyDPRI0VEuHeZ1A2CDL7_AqqgExvb9Go4qY")
    st.markdown('<div class="animated">', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <h1>Crop Recommendation System</h1>
    <p>Get AI-powered suggestions for the best crops to grow based on your soil and climate conditions.</p>
    </div>
    """, unsafe_allow_html=True)

        # Location input at the top
    location = st.text_input("📍 Farm Location (Country/Region)", 
                           help="Enter your country or specific region for more accurate recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
        <h3>Soil Parameters</h3>
        """, unsafe_allow_html=True)
        N = st.number_input("Nitrogen Content (N)", min_value=0.0, max_value=100.0, value=50.0, 
                           help="Typical range: 10-100 ppm")
        P = st.number_input("Phosphorus Content (P)", min_value=0.0, max_value=100.0, value=50.0,
                           help="Typical range: 10-50 ppm")
        K = st.number_input("Potassium Content (K)", min_value=0.0, max_value=100.0, value=50.0,
                           help="Typical range: 5-50 ppm")
        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=7.0,
                            help="Optimal range for most crops: 6.0-7.5")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
        <h3>Climate Parameters</h3>
        """, unsafe_allow_html=True)
        temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0,
                              help="Average temperature during growing season")
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0,
                                 help="Average relative humidity")
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0,
                                 help="Total rainfall during growing season")
        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🌾 Find Best Crop", help="Click to get crop recommendation"):
        with st.spinner("Analyzing your parameters..."):
            try:
                # Create the prompt for Gemini
                prompt = f"""
                Based on the following agricultural parameters, recommend the top 3 most suitable crops with detailed explanations:
                
                Location: {location if location else 'india'}

                Soil Conditions:
                - Nitrogen (N): {N} ppm
                - Phosphorus (P): {P} ppm
                - Potassium (K): {K} ppm
                - Soil pH: {ph}

                Climate Conditions:
                - Temperature: {temp}°C
                - Humidity: {humidity}%
                - Rainfall: {rainfall} mm

                Please provide:
                1. Three recommended crops in order of suitability
                2. Specific growing tips for each recommended crop

                give very breifly
                """

                # Initialize Gemini model
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt)
                
                # Display results in a formatted card
                st.markdown(f"""
                <div class="content-card" style="background-color: #f0fff0; border-left: 5px solid #2E8B57; padding: 15px; margin-top: 20px;">
                <h2 style="color: #2E8B57;">🌱 AI-Powered Crop Recommendations</h2>
                <div style="padding: 15px; background-color: white; border-radius: 8px; margin-top: 10px;">
                <p>{response.text}<p/>
                </div>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error getting recommendations: {str(e)}")
                st.markdown("""
                <div class="card" style="background-color: #fff0f0;">
                <h3>Alternative Recommendation</h3>
                <p>While we work on fixing the AI model, here are some general recommendations based on your inputs:</p>
                <ul>
                <li><strong>For high rainfall ({rainfall}mm):</strong> Consider rice or taro</li>
                <li><strong>For moderate temperatures ({temp}°C):</strong> Wheat or maize may be suitable</li>
                <li><strong>For your soil pH ({ph}):</strong> Most crops prefer pH 6-7.5</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

# elif app_mode == "Crop Recommendation":
#     model = pickle.load(open('model.pkl', 'rb'))
#     sc = pickle.load(open('standscaler.pkl', 'rb'))
#     mx = pickle.load(open('minmaxscaler.pkl', 'rb'))

#     crop_dict = {
#         1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
#         8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
#         14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
#         19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
#     }

#     st.markdown('<div class="animated">', unsafe_allow_html=True)
#     st.markdown("""
#     <div class="card">
#     <h1>Crop Recommendation System</h1>
#     <p>Get AI-powered suggestions for the best crops to grow based on your soil and climate conditions.</p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown("""
#         <div class="card">
#         <h3>Soil Parameters</h3>
#         """, unsafe_allow_html=True)
#         N = st.number_input("Nitrogen Content (N)", min_value=0.0, max_value=100.0, value=50.0, 
#                            help="Typical range: 10-100 ppm")
#         P = st.number_input("Phosphorus Content (P)", min_value=0.0, max_value=100.0, value=50.0,
#                            help="Typical range: 10-50 ppm")
#         K = st.number_input("Potassium Content (K)", min_value=0.0, max_value=100.0, value=50.0,
#                            help="Typical range: 5-50 ppm")
#         ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=7.0,
#                             help="Optimal range for most crops: 6.0-7.5")
#         st.markdown("</div>", unsafe_allow_html=True)
    
#     with col2:
#         st.markdown("""
#         <div class="card">
#         <h3>Climate Parameters</h3>
#         """, unsafe_allow_html=True)
#         temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0,
#                               help="Average temperature during growing season")
#         humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0,
#                                  help="Average relative humidity")
#         rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0,
#                                  help="Total rainfall during growing season")
#         st.markdown("</div>", unsafe_allow_html=True)

#     if st.button("🌾 Find Best Crop", help="Click to get crop recommendation"):

#         # with st.spinner("Analyzing your parameters..."):
#         #     feature_list = [N, P, K, temp, humidity, ph, rainfall]
#         #     single_pred = np.array(feature_list).reshape(1, -1)

#         #     mx_features = mx.transform(single_pred)
#         #     sc_mx_features = sc.transform(mx_features)
#         #     prediction = model.predict(sc_mx_features)
            
#         #     if prediction[0] in crop_dict:
#         #         crop = crop_dict[prediction[0]]
#         #         st.markdown(f"""
#         #         <div class="card" style="background-color: #f0fff0;">
#         #         <h2>Recommendation Results</h2>
#         #         <div style="text-align: center; padding: 2rem;">
#         #         <p style="font-size: 1.5rem;">The optimal crop for your conditions is:</p>
#         #         <p style="font-size: 2rem; font-weight: bold; color: #2E8B57;">{crop}</p>
#         #         </div>
#         #         </div>
#         #         """, unsafe_allow_html=True)
                
#         #         # Add crop-specific tips
#         #         crop_tips = {
#         #             "Rice": "Requires standing water, grows best in clayey soil with good water retention.",
#         #             "Maize": "Needs well-drained soil and moderate rainfall, sensitive to waterlogging.",
#         #             "Cotton": "Prefers warm climate and well-drained soil, requires long growing season.",
#         #             "Wheat": "Grows best in cool climates with moderate rainfall, prefers loamy soil."
#         #             # Add more crops as needed
#         #         }
                
#         #         if crop in crop_tips:
#         #             st.markdown(f"""
#         #             <div class="card">
#         #             <h3>Growing Tips for {crop}</h3>
#         #             <p>{crop_tips[crop]}</p>
#         #             </div>
#         #             """, unsafe_allow_html=True)
#         #     else:
#         #         st.error("Sorry, we could not determine the best crop to be cultivated with the provided data.")


elif app_mode == "Weather":
    with open('weatherdata.json', 'r') as file:
        data = json.load(file)

    states = data.keys()
    
    st.markdown('<div class="animated">', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <h1>Weather Forecast</h1>
    <p>Get real-time weather data and forecasts for better farming decisions.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            state_name = st.selectbox("Select State", options=states)
        with col2:
            if state_name:
                cities = data[state_name]
                city_name = st.selectbox("Select City", options=cities)

    if city_name:
        api_key = "f09c8818b978dbb75c0a83da4c21767b"
        current_weather_url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name},{state_name}&appid={api_key}&units=metric"
        forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?q={city_name},{state_name}&appid={api_key}&units=metric"

        current_response = requests.get(current_weather_url)
        forecast_response = requests.get(forecast_url)

        if current_response.status_code == 200:
            current_data = current_response.json()

            st.markdown(f"""
            <div class="card">
            <h2>Current Weather in {city_name}, {state_name}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            sunrise = datetime.utcfromtimestamp(current_data['sys']['sunrise']).strftime('%H:%M:%S')
            sunset = datetime.utcfromtimestamp(current_data['sys']['sunset']).strftime('%H:%M:%S')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="🌡 Temperature", value=f"{current_data['main']['temp']}°C")
                st.metric(label="💧 Humidity", value=f"{current_data['main']['humidity']}%")
            with col2:
                st.metric(label="🌬 Wind Speed", value=f"{current_data['wind']['speed']} m/s")
                st.metric(label="☁ Condition", value=f"{current_data['weather'][0]['main']}")
            with col3:
                st.metric(label="🌅 Sunrise", value=f"{sunrise} UTC")
                st.metric(label="🌇 Sunset", value=f"{sunset} UTC")

            lat, lon = current_data["coord"]["lat"], current_data["coord"]["lon"]
            m = folium.Map(location=[lat, lon], zoom_start=10)
            
            folium.TileLayer(
                tiles=f"http://maps.openweathermap.org/maps/2.0/weather/TA2/{{z}}/{{x}}/{{y}}?date=1552861800&opacity=0.5&fill_bound=true&arrow_step=10&palette=0:FF0000;10:00FF00;20:0000FF&appid={api_key}",
                attr="OpenWeatherMap",
                name="Weather Overlay",
                overlay=True
            ).add_to(m)

            folium.Marker(
                location=[lat, lon],
                popup=f"{city_name}: {current_data['weather'][0]['description']}",
                icon=folium.Icon(color="blue", icon="cloud"),
            ).add_to(m)

            st.markdown("""
            <div class="card">
            <h3>Location Map with Weather</h3>
            </div>
            """, unsafe_allow_html=True)
            st_folium(m, width=700, height=500)

            if forecast_response.status_code == 200:
                forecast_data = forecast_response.json()
                forecast_list = forecast_data["list"][:16]  # Next 16 entries (2 days)
                
                daily_forecast = {}
                for forecast in forecast_list:
                    forecast_time = datetime.utcfromtimestamp(forecast["dt"])
                    date_str = forecast_time.strftime('%d-%m-%Y')
                    time_str = forecast_time.strftime('%H:%M:%S')
                    
                    if date_str not in daily_forecast:
                        daily_forecast[date_str] = []
                    daily_forecast[date_str].append({
                        "Time (UTC)": time_str,
                        "Condition": forecast["weather"][0]["description"].capitalize(),
                        "Temp (°C)": forecast["main"]["temp"],
                        "Humidity (%)": forecast["main"]["humidity"],
                        "Wind (m/s)": forecast["wind"]["speed"],
                    })
                
                st.markdown("""
                <div class="card">
                <h2>2-Day Weather Forecast</h2>
                </div>
                """, unsafe_allow_html=True)
                
                tabs = st.tabs(list(daily_forecast.keys()))
                for tab, (date, forecasts) in zip(tabs, daily_forecast.items()):
                    with tab:
                        st.table(forecasts)
            else:
                st.error("Error fetching forecast data. Please try again.")
        else:
            st.error("Error fetching current weather data. Please try again.")

# Footer
st.markdown("""
<div class="footer">
<p>Vriksha Rakshak - AI for Sustainable Agriculture</p>
</div>
""", unsafe_allow_html=True)
