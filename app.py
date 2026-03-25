import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import base64

# වෙබ් පිටුවේ සැකසුම්
st.set_page_config(page_title="Crop Disease Tracker", page_icon="🌱", layout="centered")

# --- 🚀 විශේෂ විසඳුම: Error එක මඟහැරීම සඳහා Custom Layer එකක් සෑදීම ---
class SafeDense(tf.keras.layers.Dense):
    def __init__(self, **kwargs):
        # කරදරකාරී 'quantization_config' කොටස කේතයෙන් ඉවත් කිරීම
        kwargs.pop('quantization_config', None)
        super().__init__(**kwargs)

# --- පසුබිම් පින්තූරය සැකසීම ---
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

try:
    # ඔබේ තේ වත්තේ පින්තූරය මෙතනට සම්බන්ධ වේ
    image_base64 = get_base64_image("istockphoto-470248962-612x612.jpg")
    
    st.markdown(f"""
        <style>
        /* මුළු පිටුවේම පසුබිම */
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/jpg;base64,{image_base64}");
            background-size: cover;
            background-attachment: fixed;
        }}
        [data-testid="stAppViewContainer"]::before {{
            content: "";
            position: absolute;
            top: 0; left: 0; width: 100%; height: 100%;
            background-color: rgba(0, 0, 0, 0.65); 
            z-index: -1;
        }}

        /* Sidebar එක සහ භාෂාව තෝරන කොටස ලස්සන කිරීම */
        [data-testid="stSidebar"] {{
            background-color: rgba(10, 30, 10, 0.9) !important;
            border-right: 2px solid #4CAF50;
        }}
        
        .sidebar-label {{
            font-size: 1.6rem !important;
            font-weight: bold;
            color: #4CAF50;
            text-shadow: 0px 0px 10px rgba(76, 175, 80, 0.5);
            margin-bottom: 10px;
            display: block;
        }}

        /* මාතෘකා සහ Frames */
        .main-title {{
            text-align: center;
            color: #4CAF50;
            font-size: 3.5rem;
            font-weight: bold;
            text-shadow: 2px 2px 10px #000;
        }}
        .large-text {{
            font-size: 1.6rem !important;
            font-weight: bold;
            color: white;
        }}
        .prediction-box {{
            border: 3px solid #4CAF50;
            background-color: rgba(0, 50, 0, 0.5);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
        }}
        </style>
    """, unsafe_allow_html=True)
except FileNotFoundError:
    st.error("පසුබිම් පින්තූරය සොයාගත නොහැක.")

# --- Sidebar එකේ භාෂාව තෝරන්න ---
st.sidebar.markdown('<span class="sidebar-label">🌐 භාෂාව තෝරන්න <br> Select Language</span>', unsafe_allow_html=True)
language = st.sidebar.selectbox("", ["සිංහල", "English"], label_visibility="collapsed")

# UI භාෂා සැකසුම
ui_text = {
    "සිංහල": {
        "title": "🌱 කෘෂිකාර්මික බෝග රෝග හඳුනාගැනීම",
        "msg": "ඔබේ වගාවේ රෝගී වූ කොළයක පින්තූරයක් Upload කරන්න.",
        "btn": "පින්තූරය තෝරන්න...",
        "result": "හඳුනාගත් තත්වය",
        "conf": "නිවැරදි වීමේ සම්භාවිතාවය",
        "treat": "💊 නිර්දේශිත පිළියම්:",
        "analyzing": "පරීක්ෂා කරමින් පවතී... 🔍"
    },
    "English": {
        "title": "🌱 Crop Disease Detection",
        "msg": "Upload a picture of a diseased leaf from your crop.",
        "btn": "Choose an image...",
        "result": "Detected Condition",
        "conf": "Confidence Level",
        "treat": "💊 Recommended Treatments:",
        "analyzing": "Analyzing... 🔍"
    }
}

t = ui_text[language]

# Banner පින්තූරය
try:
    st.image("support-us-banner-2-877x470.jpg", use_container_width=True)
except:
    pass

st.markdown(f"<h1 class='main-title'>{t['title']}</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center; color:white; font-size:1.5rem;'>{t['msg']}</p>", unsafe_allow_html=True)

# 🚀 Model එක Load කිරීම (Error එක හදපු විදිහ)
@st.cache_resource
def load_model():
    try:
        # custom_objects හරහා අපි හදපු SafeDense එක දෙනවා
        model = tf.keras.models.load_model(
            'super_crop_disease_model.keras', 
            compile=False,
            custom_objects={'Dense': SafeDense}
        )
        return model
    except Exception as e:
        return str(e)

model = load_model()

# රෝග ප්‍රතිකාර
treatments = {
    'Tomato_Late_blight': {
        "සිංහල": "මෙටලැක්සිල් (Metalaxyl) අඩංගු දිලීර නාශකයක් වහාම යොදන්න.",
        "English": "Immediately apply a fungicide containing Metalaxyl."
    },
    'Tomato_healthy': {
        "සිංහල": "ශාකය නිරෝගී තත්වයේ පවතී!",
        "English": "The plant is in a healthy condition!"
    }
}

if isinstance(model, str):
    st.error(f"🚨 Error loading model: {model}")
else:
    # පින්තූරය Upload කිරීම
    up_file = st.file_uploader(t["btn"], type=["jpg", "png", "jpeg"])

    if up_file:
        img = Image.open(up_file)
        st.image(img, caption=t["msg"], use_container_width=True)
        
        with st.spinner(t["analyzing"]):
            # AI සැකසුම්
            processed_img = img.convert('RGB').resize((224, 224))
            img_arr = tf.keras.preprocessing.image.img_to_array(processed_img) / 255.0
            img_arr = np.expand_dims(img_arr, axis=0)
            
            preds = model.predict(img_arr)
            idx = np.argmax(preds)
            conf = round(np.max(preds) * 100, 2)
            
            class_names = ['Tomato_Late_blight', 'Tomato_healthy']
            res = class_names[idx]

        # ප්‍රතිඵල පෙන්වීම
        st.markdown(f"""
            <div class='prediction-box'>
                <p class='large-text'>🟢 {t['result']}: <span style="color: #90EE90;">{res}</span></p>
                <p class='large-text'>📊 {t['conf']}: <span style="color: #87CEEB;">{conf}%</span></p>
            </div>
        """, unsafe_allow_html=True)

        if res in treatments:
            info = treatments[res][language]
            st.markdown(f"""
                <div style='border:2px solid #FFC107; padding:20px; margin-top:20px; background:rgba(50,50,0,0.5); border-radius:15px;'>
                    <p class='large-text' style='color:#FFC107;'>{t['treat']}</p>
                    <p style='color:white; font-size:1.3rem;'>{info}</p>
                </div>
            """, unsafe_allow_html=True)

st.markdown("<br><hr><center style='color:white; font-size:1.2rem;'>👨‍💻 Developed by Kaveesha Induwara</center>", unsafe_allow_html=True)
