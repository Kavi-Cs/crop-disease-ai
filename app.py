import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# වෙබ් පිටුවේ සැකසුම්
st.set_page_config(page_title="Crop Disease Tracker", page_icon="🌱", layout="centered")

# --- UI එක ලස්සන කරන CSS කේතය (Background, Fonts & Sidebar Styling) ---
st.markdown("""
    <style>
    /* මුළු පිටුවේම පසුබිම සඳහා තේ වත්තක පින්තූරයක් එක් කිරීම */
    [data-testid="stAppViewContainer"] {
        background-image: url("https://images.unsplash.com/photo-1597432480301-a141d33309f3?q=80&w=2070&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* පිටුවේ අකුරු කියවීමට හැකිවන සේ Overlay එකක් එක් කිරීම */
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        background-color: rgba(0, 0, 0, 0.6); /* අඳුරු පසුබිමක් */
        z-index: -1;
    }

    /* පැති මෙනුව (Sidebar) ලස්සන කිරීම */
    [data-testid="stSidebar"] {
        background-color: rgba(25, 40, 25, 0.95) !important;
        border-right: 2px solid #4CAF50;
    }
    
    .sidebar-text {
        font-size: 1.5rem !important;
        font-weight: bold;
        color: #4CAF50 !important;
        margin-bottom: 10px;
    }

    /* මාතෘකා සහ අනෙකුත් කොටස් Styling */
    .main-title {
        text-align: center;
        color: #4CAF50;
        font-family: 'Arial', sans-serif;
        font-size: 3.5rem; /* අකුරු ගොඩක් ලොකු කළා */
        font-weight: bold;
        text-shadow: 2px 2px 4px #000000;
        padding-bottom: 10px;
    }
    .sub-title {
        text-align: center;
        font-size: 1.8rem;
        margin-bottom: 25px;
        color: #ffffff;
        font-weight: 500;
    }
    .prediction-frame {
        border: 3px solid #4CAF50;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        background-color: rgba(0, 50, 0, 0.7);
        margin-bottom: 15px;
    }
    .confidence-frame {
        border: 3px solid #2196F3;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        background-color: rgba(0, 30, 60, 0.7);
        margin-bottom: 15px;
    }
    .treatment-frame {
        border: 3px solid #FFC107;
        border-radius: 15px;
        padding: 25px;
        background-color: rgba(60, 50, 0, 0.8);
        margin-bottom: 30px;
    }
    .big-text {
        font-size: 1.8rem !important;
        font-weight: bold;
        color: #ffffff;
    }
    .treatment-text {
        font-size: 1.5rem !important;
        line-height: 1.6;
        color: #f1f1f1;
    }
    </style>
""", unsafe_allow_html=True)

# 🌍 පැති මෙනුව (Sidebar) තුළ භාෂාව තේරීම
st.sidebar.markdown('<p class="sidebar-text">🌐 භාෂාව තෝරන්න <br> Select Language</p>', unsafe_allow_html=True)
language = st.sidebar.selectbox("", ["සිංහල", "English"], label_visibility="collapsed")

# UI එකේ පෙන්වන වචන 
ui = {
    "සිංහල": {
        "title": "🌱 කෘෂිකාර්මික බෝග රෝග හඳුනාගැනීම",
        "upload_msg": "ඔබේ වගාවේ රෝගී වූ කොළයක පින්තූරයක් මෙතනට Upload කරන්න.",
        "uploader": "පින්තූරය තෝරන්න (JPG/PNG)...",
        "caption": "ඔබ Upload කළ පින්තූරය",
        "analyzing": "පරීක්ෂා කරමින් පවතී... 🔍",
        "condition": "හඳුනාගත් තත්වය",
        "confidence": "නිවැරදි වීමේ සම්භාවිතාවය",
        "treatment_title": "💊 නිර්දේශිත පිළියම් සහ උපදෙස්:",
        "community_title": "Featured Community Stories: Thriving Sri Lankan Agriculture"
    },
    "English": {
        "title": "🌱 Crop Disease Detection",
        "upload_msg": "Upload a picture of a diseased leaf from your crop.",
        "uploader": "Choose an image (JPG/PNG)...",
        "caption": "Uploaded Image",
        "analyzing": "Analyzing... 🔍",
        "condition": "Detected Condition",
        "confidence": "Confidence Level",
        "treatment_title": "💊 Recommended Treatments & Advice:",
        "community_title": "Featured Community Stories: Thriving Sri Lankan Agriculture"
    }
}

t = ui[language]

# --- Banner පින්තූරය ---
try:
    banner = Image.open("support-us-banner-2-877x470.jpg")
    st.image(banner, use_container_width=True)
except:
    pass 

# මාතෘකා
st.markdown(f"<h1 class='main-title'>{t['title']}</h1>", unsafe_allow_html=True)
st.markdown(f"<p class='sub-title'>{t['upload_msg']}</p>", unsafe_allow_html=True)

# Model එක Load කරගැනීම
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('super_crop_disease_model.keras', compile=False)
        return model
    except Exception as e:
        return str(e)

model = load_model()

# රෝග ප්‍රතිකාර (treatments dict එක කලින් වගේමයි...)
treatments = {
    'Pepper__bell___Bacterial_spot': {"සිංහල": "කොපර් (Copper) අඩංගු දිලීර නාශකයක් භාවිතා කරන්න.", "English": "Use a copper-based fungicide."},
    'Pepper__bell___healthy': {"සිංහල": "ශාකය නිරෝගී තත්වයේ පවතී!", "English": "This plant is healthy!"},
    'Potato___Early_blight': {"සිංහල": "මැන්කොසෙබ් (Mancozeb) අඩංගු දිලීර නාශකයක් යොදන්න.", "English": "Apply a fungicide containing Mancozeb."},
    'Potato___Late_blight': {"සිංහල": "වහාම මෙටලැක්සිල් (Metalaxyl) අඩංගු දිලීර නාශකයක් යොදන්න.", "English": "Immediately apply a fungicide containing Metalaxyl."},
    'Potato___healthy': {"සිංහල": "නිරෝගී අල ශාකයකි.", "English": "Healthy potato plant."},
    'Tomato_Bacterial_spot': {"සිංහල": "කොපර් දිලීර නාශක භාවිතා කරන්න.", "English": "Use copper-based fungicides."},
    'Tomato_Early_blight': {"සිංහල": "මැන්කොසෙබ් (Mancozeb) භාවිතා කරන්න.", "English": "Use Mancozeb fungicide."},
    'Tomato_Late_blight': {"සිංහල": "මෙටලැක්සිල් (Metalaxyl) වහාම යොදන්න.", "English": "Immediately apply Metalaxyl."},
    'Tomato_Leaf_Mold': {"සිංහල": "වාතාශ්‍රය ලැබෙන්නට ඉඩ හරින්න.", "English": "Ensure good ventilation."},
    'Tomato_Septoria_leaf_spot': {"සිංහල": "ක්ලෝරෝතැලොනිල් අඩංගු දිලීර නාශකයක් යොදන්න.", "English": "Apply Chlorothalonil fungicide."},
    'Tomato_Spider_mites_Two_spotted_spider_mite': {"සිංහල": "ඇබමෙක්ටින් (Abamectin) භාවිතා කරන්න.", "English": "Apply Abamectin miticide."},
    'Tomato__Target_Spot': {"සිංහල": "තඹ අඩංගු දිලීර නාශක භාවිතා කරන්න.", "English": "Use copper-based fungicides."},
    'Tomato__Tomato_YellowLeaf__Curl_Virus': {"සිංහල": "සුදු මැස්සන් මර්දනයට ඉමිඩාක්ලෝප්‍රිඩ් යොදන්න.", "English": "Apply Imidacloprid for whiteflies."},
    'Tomato__Tomato_mosaic_virus': {"සිංහල": "රෝගී ගස් ගලවා පුළුස්සා දමන්න.", "English": "Uproot and burn infected plants."},
    'Tomato_healthy': {"සිංහල": "නිරෝගී තක්කාලි ශාකයකි!", "English": "Healthy tomato plant!"}
}

if isinstance(model, str):
    st.error(f"🚨 Error loading model: {model}")
else:
    uploaded_file = st.file_uploader(t["uploader"], type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.markdown("<div style='border: 4px solid #4CAF50; border-radius: 15px; padding: 10px; background-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
        st.image(image, caption=t["caption"], use_container_width=True)
        st.markdown("</div><br>", unsafe_allow_html=True)
        
        with st.spinner(t["analyzing"]):
            img = image.convert('RGB').resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions)
            confidence = round(np.max(predictions) * 100, 2)

            class_names = list(treatments.keys())
            predicted_disease = class_names[predicted_class_index]

        # ප්‍රතිඵල පෙන්වීම
        st.markdown(f"<div class='prediction-frame'><span class='big-text'>🟢 {t['condition']}: <br><span style='color: #90EE90;'>{predicted_disease}</span></span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='confidence-frame'><span class='big-text'>📊 {t['confidence']}: <span style='color: #87CEEB;'>{confidence}%</span></span></div>", unsafe_allow_html=True)
        
        disease_info = treatments[predicted_disease].get(language, treatments[predicted_disease]["සිංහල"])
        st.markdown(f"<div class='treatment-frame'><span class='big-text' style='color: #FFD700;'>{t['treatment_title']}</span><br><p class='treatment-text'>{disease_info}</p></div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown(f"<h2 style='text-align: center; color: white; text-shadow: 2px 2px 4px #000;'>{t['community_title']}</h2><br>", unsafe_allow_html=True)

# පහළ පින්තූර 3
col1, col2, col3 = st.columns(3)
try:
    with col1: st.image("food_security_1.jpg", use_container_width=True, caption="හරිත වගාවන්")
    with col2: st.image("Image-3-Farmers-EDITED-1200x800.jpg", use_container_width=True, caption="ශ්‍රී ලාංකීය ගොවියා")
    with col3: st.image("istockphoto-470248962-612x612.jpg", use_container_width=True, caption="සාරවත් තේ වතු")
except:
    pass

st.markdown("<br><div style='text-align: center; color: #4CAF50; font-size: 1.5rem; font-weight: bold; background-color: rgba(0,0,0,0.7); padding: 10px; border-radius: 10px;'>👨‍💻 Developed by Kaveesha Induwara</div>", unsafe_allow_html=True)
