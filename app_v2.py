import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2  

# වෙබ් පිටුවේ සැකසුම්
st.set_page_config(page_title="Crop Disease Tracker v3.3", page_icon="🌱", layout="wide")

# ඔබට අවශ්‍ය වෙනත් වීඩියෝවක් (MP4 link එකක්) මෙතැනට දැමිය හැක:
background_video_url = "https://assets.mixkit.co/videos/preview/mixkit-beautiful-view-of-a-misty-rice-field-during-a-morning-28821-large.mp4"

# --- UI CSS (Video Background & Glassmorphism with Enhanced Header) ---
st.markdown(f"""
    <style>
    /* Streamlit App එකේ පසුබිම transparent කිරීම (වීඩියෝව පෙනීමට) */
    [data-testid="stAppViewContainer"] {{
        background-color: transparent !important;
    }}
    [data-testid="stHeader"] {{
        background-color: transparent !important;
    }}

    /* Fullscreen Video Background */
    #bg-video {{
        position: fixed;
        right: 0;
        bottom: 0;
        min-width: 100%;
        min-height: 100%;
        z-index: -2;
        object-fit: cover;
    }}
    
    /* වීඩියෝව මත අඳුරු පාරදෘශ්‍ය overlay එකක් (අකුරු පැහැදිලිව පෙනීමට) */
    .video-overlay {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(10, 30, 15, 0.7); /* අඳුරු කොළ/කළු Tint එකක් */
        z-index: -1;
    }}

    /* 👇👇👇 Enhanced Head Topic Section (Header කොටස) 👇👇👇 */
    .nature-header-box {{
        background: linear-gradient(135deg, rgba(34, 139, 34, 0.7), rgba(0, 100, 0, 0.6)); /* Forest Green Gradient */
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 2px solid rgba(168, 224, 99, 0.5); /* Lime Green border */
        padding: 40px;
        text-align: center;
        margin-bottom: 40px;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.5);
    }}

    .nature-main-title {{
        color: #FFFFFF;
        font-size: 3.8rem;
        font-weight: 900;
        margin-bottom: 10px;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.7);
        font-family: 'Arial', sans-serif;
    }}

    .nature-subtitle {{
        background: -webkit-linear-gradient(45deg, #FFE082, #A8E063); /* Yellow to Lime Gradient */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.4rem;
        font-weight: 600;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }}
    /* 👆👆👆 End of Head Topic Section 👆👆👆 */

    /* Glass Cards for results */
    .glass-card {{
        background: rgba(20, 50, 25, 0.45); 
        backdrop-filter: blur(15px); 
        -webkit-backdrop-filter: blur(15px);
        border-radius: 20px; 
        border: 1px solid rgba(255, 255, 255, 0.15); 
        padding: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3); 
        margin-bottom: 20px; 
        color: white;
    }}
    
    .result-value {{ font-size: 1.8rem; font-weight: bold; }}
    .color-green {{ color: #A8E063; }} .color-blue {{ color: #81D4FA; }}
    
    [data-testid="stFileUploadDropzone"] {{
        background-color: rgba(255, 255, 255, 0.1) !important; 
        border: 2px dashed #A8E063 !important; 
        border-radius: 15px;
    }}
    </style>

    <video autoplay loop muted playsinline id="bg-video">
        <source src="{background_video_url}" type="video/mp4">
    </video>
    <div class="video-overlay"></div>
""", unsafe_allow_html=True)

# භාෂා සැකසුම
translations = {
    "සිංහල": {
        "title": "🌿 කෘෂිකාර්මික බෝග රෝග විශ්ලේෂක",
        "subtitle": "ඔබේ වගාවේ රෝගී වූ කොළයක පින්තූරයක් අප්ලෝඩ් කර ස්මාර්ට් තාක්ෂණයෙන් විසඳුම් ලබාගන්න.",
        "upload_title": "📤 පින්තූරය අප්ලෝඩ් කරන්න",
        "upload_desc": "රෝගී කොළයක පැහැදිලි පින්තූරයක් තෝරන්න (JPG/PNG)",
        "result_card_title": "📊 රෝග අනාවැකිය",
        "condition": "හඳුනාගත් තත්ත්වය", "confidence": "නිරවද්‍යතාවය", "severity": "අවදානම් මට්ටම",
        "treatment_title": "💊 නිර්දේශිත පිළියම", "explainable_ai": "🔍 AI අවධානය (Explainable AI)"
    },
    "English": {
        "title": "🌿 Smart Crop Disease Analyzer",
        "subtitle": "Upload an image of a diseased leaf and get data-driven solutions.",
        "upload_title": "📤 Upload Image", "upload_desc": "Choose a clear image of a diseased leaf (JPG/PNG)",
        "result_card_title": "📊 Disease Forecast",
        "condition": "Detected Condition", "confidence": "Confidence", "severity": "Severity Level",
        "treatment_title": "💊 Recommended Treatment", "explainable_ai": "🔍 AI Attention Map"
    }
}

# Sidebar
st.sidebar.markdown("<p style='color: #A8E063; font-weight: bold; font-size: 1.2rem;'>🌍 Language / භාෂාව</p>", unsafe_allow_html=True)
language = st.sidebar.radio("", ["සිංහල", "English"])
t = translations[language]

# 👇👇👇 Enhanced Header Section (මුලින්මHeader එක පෙන්වීම) 👇👇👇
st.markdown(f"""
    <div class="nature-header-box">
        <div class="nature-main-title">{t['title']}</div>
        <div class="nature-subtitle">{t['subtitle']}</div>
    </div>
""", unsafe_allow_html=True)
# 👆👆👆 End of Header Section 👆👆👆

# Model Load
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('super_crop_disease_model.keras', compile=False)

model = load_model()

# රෝග සහ ප්‍රතිකර්ම
treatments = {
    'Bell_pepper__Bacterial_spot': {"සිංහල": "කොපර් (Copper) අඩංගු දිලීර නාශකයක් භාවිතා කරන්න.", "English": "Use a copper-based fungicide."},
    'Bell_pepper__healthy': {"සිංහල": "ශාකය නිරෝගී තත්වයේ පවතී!", "English": "This plant is healthy!"},
    'Potato__Early_blight': {"සිංහල": "මැන්කොසෙබ් (Mancozeb) අඩංගු දිලීර නාශකයක් යොදන්න.", "English": "Apply a fungicide containing Mancozeb."},
    'Potato__Late_blight': {"සිංහල": "වහාම මෙටලැක්සිල් (Metalaxyl) අඩංගු දිලීර නාශකයක් යොදන්න.", "English": "Immediately apply a fungicide containing Metalaxyl."},
    'Potato__healthy': {"සිංහල": "නිරෝගී අල ශාකයකි.", "English": "Healthy potato plant."},
    'Tomato__Bacterial_spot': {"සිංහල": "කොපර් දිලීර නාශක භාවිතා කරන්න.", "English": "Use copper-based fungicides."},
    'Tomato__Early_blight': {"සිංහල": "මැන්කොසෙබ් (Mancozeb) භාවිතා කරන්න.", "English": "Use Mancozeb fungicide."},
    'Tomato__Late_blight': {"සිංහල": "මෙටලැක්සිල් (Metalaxyl) වහාම යොදන්න.", "English": "Immediately apply Metalaxyl."},
    'Tomato__Leaf_Mold': {"සිංහල": "වාතාශ්‍රය ලැබෙන්නට ඉඩ හරින්න.", "English": "Ensure good ventilation."},
    'Tomato__Septoria_leaf_spot': {"සිංහල": "ක්ලෝරෝතැලොනිල් අඩංගු දිලීර නාශකයක් යොදන්න.", "English": "Apply Chlorothalonil fungicide."},
    'Tomato__Spider_mites': {"සිංහල": "ඇබමෙක්ටින් (Abamectin) භාවිතා කරන්න.", "English": "Apply Abamectin miticide."},
    'Tomato__Target_Spot': {"සිංහල": "තඹ අඩංගු දිලීර නාශක භාවිතා කරන්න.", "English": "Use copper-based fungicides."},
    'Tomato__Tomato_Yellow_Leaf_Curl_Virus': {"සිංහල": "සුදු මැස්සන් මර්දනයට ඉමිඩාක්ලෝප්‍රිඩ් යොදන්න.", "English": "Apply Imidacloprid for whiteflies."},
    'Tomato__Tomato_mosaic_virus': {"සිංහල": "රෝගී ගස් ගලවා පුළුස්සා දමන්න.", "English": "Uproot and burn infected plants."},
    'Tomato__healthy': {"සිංහල": "නිරෝගී තක්කාලි ශාකයකි!", "English": "Healthy tomato plant!"}
}

class_names = list(treatments.keys())

# --- Auto-Detect Grad-CAM Function ---
def get_gradcam_heatmap(img_array, model):
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break
    if last_conv_layer_name is None:
        raise ValueError("Convolutional Layer Not Found.")
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(img, heatmap, alpha=0.6):
    img = np.array(img)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown(f"<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='margin-bottom:5px; color:white;'>{t['upload_title']}</h3><p style='margin-top:0; color:#E8F5E9;'>{t['upload_desc']}</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        original_image = Image.open(uploaded_file).convert('RGB')
        st.image(original_image, use_container_width=True, caption=uploaded_file.name)
        analyze_button = st.button("🔍 රෝගය පරීක්ෂා කරන්න", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    if uploaded_file is not None and analyze_button:
        with st.spinner("🌿 Analyzing structure & detecting pathogens..."):
            image_resized = original_image.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(image_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions)
            predicted_disease = class_names[predicted_class_index]
            confidence_score = round(100 * np.max(predictions), 2)

            st.markdown(f"<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color:white; margin-bottom:15px;'>{t['result_card_title']}</h3>", unsafe_allow_html=True)
            
            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown(f"**{t['condition']}:**<br><span class='result-value color-green'>{predicted_disease}</span>", unsafe_allow_html=True)
            with rc2:
                st.markdown(f"**{t['confidence']}:**<br><span class='result-value color-blue'>{confidence_score}%</span>", unsafe_allow_html=True)
            
            st.markdown("<hr style='border-color: rgba(255,255,255,0.2);'>", unsafe_allow_html=True)

            try:
                heatmap = get_gradcam_heatmap(img_array, model) 
                cam_image = overlay_heatmap(image_resized, heatmap)

                if "healthy" in predicted_disease.lower():
                    severity_score = 0.0
                else:
                    severity_score = min(np.mean(heatmap > 0.3) * 100 * 2.5, 100.0) 
                
                if language == "සිංහල":
                    if "healthy" in predicted_disease.lower():
                        severity_text = "අවදානමක් නැත"
                    else:
                        severity_text = "බරපතලයි (Severe)" if severity_score > 60 else ("මධ්‍යම (Moderate)" if severity_score > 25 else "අවම බලපෑමක් (Mild)")
                else:
                    if "healthy" in predicted_disease.lower():
                        severity_text = "No Risk"
                    else:
                        severity_text = "Severe" if severity_score > 60 else ("Moderate" if severity_score > 25 else "Mild")
                    
                sev_color = "#FF5252" if severity_score > 60 else ("#FFE082" if severity_score > 25 else "#A8E063")

                st.markdown(f"**{t['severity']}:** <span style='color: {sev_color}; font-size:1.2rem; font-weight:bold;'>{severity_text} ({severity_score:.1f}%)</span>", unsafe_allow_html=True)
                st.progress(int(severity_score))
                
            except Exception: pass

            disease_info = treatments[predicted_disease].get(language, treatments[predicted_disease]["සිංහල"])
            st.markdown(f"<br>**{t['treatment_title']}**<br><div style='background:rgba(0,0,0,0.3); padding:15px; border-radius:10px; border-left:4px solid #FFE082;'><span style='font-size: 1.2rem; color: #FFF;'>{disease_info}</span></div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

            if "healthy" not in predicted_disease.lower():
                st.markdown(f"<div class='glass-card' style='text-align:center;'><h4>{t['explainable_ai']}</h4>", unsafe_allow_html=True)
                try:
                    st.image(cam_image, width=300, caption="AI Attention Heatmap")
                except: st.warning("Heatmap Error.")
                st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br><div style='text-align: center; color: #A8E063; font-size: 1rem; font-weight: bold; padding: 20px;'>👨‍💻 Developed by Kaveesha Induwara</div>", unsafe_allow_html=True)