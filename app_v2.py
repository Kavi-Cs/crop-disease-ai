import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2  # අලුතින් එකතු කළා (Grad-CAM සඳහා)

# වෙබ් පිටුවේ සැකසුම්
st.set_page_config(page_title="Crop Disease Tracker v2.0", page_icon="🌱", layout="centered")

# --- UI එක ලස්සන කරන CSS කේතය ---
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url("https://images.unsplash.com/photo-1597432480301-a141d33309f3?q=80&w=2070&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        background-color: rgba(0, 0, 0, 0.7); 
        z-index: -1;
    }
    .main-title {
        text-align: center; color: #4CAF50; font-family: 'Arial', sans-serif; font-size: 3.5rem; font-weight: bold; text-shadow: 2px 2px 4px #000;
    }
    .sub-title {
        text-align: center; color: #E8F5E9; font-size: 1.5rem; margin-bottom: 30px; text-shadow: 1px 1px 2px #000;
    }
    .prediction-frame {
        background-color: rgba(0, 0, 0, 0.6); padding: 20px; border-radius: 10px; border: 2px solid #4CAF50; text-align: center; margin-bottom: 20px;
    }
    .heatmap-frame {
        background-color: rgba(0, 0, 0, 0.6); padding: 15px; border-radius: 10px; border: 2px solid #FFA500; text-align: center; margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# භාෂා සැකසුම (Language Dictionary)
translations = {
    "සිංහල": {
        "title": "🌱 කෘෂිකාර්මික බෝග රෝග හඳුනාගැනීම",
        "subtitle": "ඔබේ වගාවේ රෝගී වූ කොළයක පින්තූරයක් Upload කරන්න.",
        "upload_prompt": "Upload image (JPG, PNG)...",
        "button": "🔍 රෝගය හඳුනාගන්න",
        "condition": "හඳුනාගත් තත්ත්වය",
        "confidence": "නිරවද්‍ය වීමේ සම්භාවිතාවය",
        "treatment_title": "💊 නිර්දේශිත පිළියම සහ උපදෙස්:",
        "heatmap_title": "🔍 AI අවධානය යොමු කළ ස්ථානය (Explainable AI):",
        "heatmap_desc": "රතු/කහ පැහැයෙන් පෙන්වන්නේ AI ආකෘතිය විසින් රෝගය හඳුනාගැනීම සඳහා විශේෂයෙන් නිරීක්ෂණය කළ ප්‍රදේශයයි."
    },
    "English": {
        "title": "🌱 Crop Disease Detection",
        "subtitle": "Upload an image of a diseased leaf from your crop.",
        "upload_prompt": "Upload image (JPG, PNG)...",
        "button": "🔍 Detect Disease",
        "condition": "Predicted Condition",
        "confidence": "Confidence Level",
        "treatment_title": "💊 Recommended Treatment:",
        "heatmap_title": "🔍 AI Attention Map (Explainable AI):",
        "heatmap_desc": "The red/yellow areas highlight the specific regions the AI focused on to detect the disease."
    }
}

# Sidebar එක
st.sidebar.markdown("<p class='sidebar-text'>Language / භාෂාව</p>", unsafe_allow_html=True)
language = st.sidebar.radio("", ["සිංහල", "English"])
t = translations[language]

st.markdown(f"<h1 class='main-title'>{t['title']}</h1>", unsafe_allow_html=True)
st.markdown(f"<h2 class='sub-title'>{t['subtitle']}</h2>", unsafe_allow_html=True)

# Model එක Load කිරීම 
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('super_crop_disease_model.keras')

model = load_model()

# රෝග නම් ලැයිස්තුව
class_names = [
    'Bell_pepper__Bacterial_spot', 'Bell_pepper__healthy', 'Potato__Early_blight', 
    'Potato__Late_blight', 'Potato__healthy', 'Tomato__Bacterial_spot', 
    'Tomato__Early_blight', 'Tomato__Late_blight', 'Tomato__Leaf_Mold', 
    'Tomato__Septoria_leaf_spot', 'Tomato__Spider_mites', 'Tomato__Target_Spot', 
    'Tomato__Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato__healthy'
]

# (මෙතැනට ඔයාගේ කලින් තිබුණු treatments dictionary එක සම්පූර්ණයෙන්ම දාන්න)
# treatments = { 'Bell_pepper__Bacterial_spot': { "සිංහල": "...", "English": "..." }, ... }
# මම කේතය දිග වැඩි වෙන නිසා treatments ටික මෙතනින් skip කළා. ඔයාගේ පරණ එකේ තියෙන ටිකම දාන්න.

# --- GRAD-CAM කේතය (අලුතින් එකතු කළ කොටස) ---
def get_gradcam_heatmap(img_array, model, last_conv_layer_name="conv2d_5"):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
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

def overlay_heatmap(img, heatmap, alpha=0.5):
    img = np.array(img)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    superimposed_img = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)
    return superimposed_img

uploaded_file = st.file_uploader(t['upload_prompt'], type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # පින්තූරය පෙන්වීම
    original_image = Image.open(uploaded_file).convert('RGB')
    st.image(original_image, caption="Uploaded Image", width=300)

    if st.button(t['button']):
        with st.spinner("Analyzing..."):
            # පින්තූරය model එකට ගැළපෙන සේ සකස් කිරීම
            image_resized = original_image.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(image_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # අනාවැකිය (Prediction) ලබා ගැනීම
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions)
            predicted_disease = class_names[predicted_class_index]
            confidence = round(100 * np.max(predictions), 2)

            # Grad-CAM Heatmap එක සෑදීම
            heatmap = get_gradcam_heatmap(img_array, model, "conv2d_5") # Layer එකේ නම මෙතන දුන්නා
            cam_image = overlay_heatmap(image_resized, heatmap)

            # ප්‍රතිඵල පෙන්වීම
            st.markdown(f"<div class='prediction-frame'><h3>🟢 {t['condition']}: {predicted_disease}</h3><h3>📊 {t['confidence']}: {confidence}%</h3></div>", unsafe_allow_html=True)
            
            # (treatments ටික පෙන්වන කේතය - ඔයාගේ පරණ එකේ වගේම දාන්න)
            # disease_info = treatments[predicted_disease].get(language, treatments[predicted_disease]["සිංහල"])
            # st.success(f"{t['treatment_title']} \n\n {disease_info}")

            # අලුත් Heatmap එක පෙන්වීම
            st.markdown(f"<div class='heatmap-frame'><h4>{t['heatmap_title']}</h4><p style='font-size: 14px;'>{t['heatmap_desc']}</p></div>", unsafe_allow_html=True)
            st.image(cam_image, caption="AI Attention Heatmap (Grad-CAM)", width=350)