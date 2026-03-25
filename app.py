import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# වෙබ් පිටුවේ මාතෘකාව සහ හැඳින්වීම
st.set_page_config(page_title="Crop Disease Tracker", page_icon="🌱")
st.title("🌱 කෘෂිකාර්මික බෝග රෝග හඳුනාගැනීම")
st.write("ඔබේ වගාවේ රෝගී වූ කොළයක පින්තූරයක් මෙතනට Upload කරන්න.")

# Model එක Load කරගැනීම (අපි අලුතින් හදපු keras file එක)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('crop_disease_model.h5', compile=False)
    return model

model = load_model()

# පින්තූරය Upload කිරීමට අවකාශය හැදීම
uploaded_file = st.file_uploader("පින්තූරය තෝරන්න (JPG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Upload කරපු පින්තූරය වෙබ් එකේ පෙන්වීම
    image = Image.open(uploaded_file)
    st.image(image, caption='ඔබ Upload කළ පින්තූරය', use_container_width=True)
    st.write("පරීක්ෂා කරමින් පවතී... 🔍")

    # පින්තූරය Model එකට අවශ්‍ය විදිහට සකස් කිරීම
    image = image.convert('RGB') 
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction එක ලබා ගැනීම
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    confidence = round(np.max(predictions) * 100, 2)

    # ඔයාගේ Dataset එකේ තිබුණු රෝග 15 හි නම් ලැයිස්තුව
    class_names = [
        'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 
        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
        'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 
        'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
        'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 
        'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 
        'Tomato_healthy'
    ] 

    # ප්‍රතිඵලය ලස්සනට වෙබ් එකේ පෙන්වීම
    st.success(f"**හඳුනාගත් තත්වය:** {class_names[predicted_class_index]}")
    st.info(f"**නිවැරදි වීමේ සම්භාවිතාවය (Confidence):** {confidence}%")
