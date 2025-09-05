import streamlit as st
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

st.set_page_config(
    layout="centered",
    page_title="TUGAS LAS WEEK 5",
    page_icon="📖"
)

# --- Muat Model dan Definisikan Kelas ---
@st.cache_resource
def load_dogcat_model():
    try:
        interpreter = tf.lite.Interpreter('models/inception_dogcat_optimized.tflite')
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error: {e}")
        return None

@st.cache_resource
def load_food_model():
    try:
        model = tf.keras.models.load_model('models/mn2_food_model.h5')
        return model
    except Exception as e:
        st.error(f"Error: {e}")
        return None

@st.cache_resource
def load_sentiment_model_and_tokenizer():
    try:
        tokenizer = AutoTokenizer.from_pretrained('valdeez/indobertweet_sentiment_optimized')
        model_path = hf_hub_download(repo_id='valdeez/indobertweet_sentiment_optimized', filename='model_quantized.onnx')
        ort_session = ort.InferenceSession(model_path)

        return ort_session, tokenizer
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None

# Inisialisasi model dan tokenizer
INTERPRETER = load_dogcat_model()
FOOD_MODEL = load_food_model()
ORT_SESSION, SENTIMENT_TOKENIZER = load_sentiment_model_and_tokenizer()

# Inisialisasi kelas
FOOD_CLASS = ['Baby Back Ribs', 'Beef Carpaccio', 'French Toast', 'Guacamole', 'Lobster Bisque', 'Macarons', 'Mussels', 'Pork Chop', 'Poutine', 'Prime Rib']
SENTIMENT_CLASS = ['Sadness', 'Anger', 'Support', 'Hope', 'Disappointment']

# --- Fungsi Prediksi ---
def predict_dogcat(img):
    # Dapatkan detail input model TFLite
    input_details = INTERPRETER.get_input_details()
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    img = img.resize((128, 128))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.image.resize(img_array, (input_shape[1], input_shape[2]))
    img_array /= 255.0
    
    # Jika model kuantisasi mengharapkan INT8 atau UINT8, konversi tipe datanya
    if input_dtype == np.int8:
        img_array = (img_array * 255 - 128).astype(np.int8)
    elif input_dtype == np.uint8:
        img_array = (img_array * 255).astype(np.uint8)

    # Jalankan inferensi
    INTERPRETER.set_tensor(input_details[0]['index'], img_array)
    INTERPRETER.invoke()
    
    # Dapatkan hasil prediksi
    output_details = INTERPRETER.get_output_details()
    prediction_tensor = INTERPRETER.get_tensor(output_details[0]['index'])
    prediction = prediction_tensor[0][0]
        
    # Tentukan hasil prediksi berdasarkan label 0 (kucing) dan 1 (anjing)
    if prediction > 0.5:
        return "Anjing", prediction
    else:
        return "Kucing", 1 - prediction

def predict_food(img):
    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
      
    predictions = FOOD_MODEL.predict(img_array)[0]
    
    # Ambil indeks dengan probabilitas tertinggi
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = FOOD_CLASS[predicted_class_index]
    confidence = predictions[predicted_class_index]
        
    return predicted_class_name, confidence

def predict_sentiment(text):
    encoding = SENTIMENT_TOKENIZER(
        text,
        return_tensors='np',
        padding='max_length',
        truncation=True,
        max_length=100
    )

    ort_inputs = {
        ORT_SESSION.get_inputs()[0].name: encoding['input_ids'],
        ORT_SESSION.get_inputs()[1].name: encoding['attention_mask'],
        'token_type_ids': np.zeros(encoding['input_ids'].shape, dtype=np.int64)
    }

    # Jalankan inferensi
    ort_outputs = ORT_SESSION.run(None, ort_inputs)
    logits = ort_outputs[0]

    # Perhitungan softmax dengan numpy
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Dapatkan indeks kelas dengan probabilitas tertinggi
    predicted_class_id = np.argmax(probabilities, axis=1)[0]
    
    confidence_score = probabilities[0, predicted_class_id]
    predicted_label = SENTIMENT_CLASS[predicted_class_id]
    
    return predicted_label, confidence_score

# --- Fungsi Halaman Utama ---
def main_page():
    st.title("Hi, Selamat Datang!")
    st.info("Silahkan pilih model yang ingin kamu coba")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("images/pets.png", use_container_width=True)
        st.subheader("Dog vs Cat Image Classification")
        if st.button("Coba Model Anjing vs Kucing", key="btn_dogcat"):
            st.query_params["page"] = "dogcat"
            st.rerun()

    with col2:
        st.image("images/foods.png", use_container_width=True)
        st.subheader("Food 101 Image Classification")
        if st.button("Coba Model 101 Makanan", key="btn_food"):
            st.query_params["page"] = "food"
            st.rerun()

    with col3:
        st.image("images/emotions.png", use_container_width=True)
        st.subheader("Text Sentiment Classification")
        if st.button("Coba Model Sentimen Teks", key="btn_sentiment"):
            st.query_params["page"] = "sentiment"
            st.rerun()

    st.write("---")
    st.write("by Muhammad Daffa Izzati")

# --- Fungsi Halaman Model ---
def dogcat_page():
    st.title("🐾Klasifikasi Anjing vs Kucing")
    st.markdown("Aplikasi ini menggunakan model deep learning untuk mengklasifikasikan gambar anjing dan kucing.")

    with st.expander("Tentang Model & Dataset"):
        st.markdown("""
        Model ini dikembangkan menggunakan arsitektur **InceptionV3**, sebuah model Convolutional Neural Network (CNN) yang sangat efisien dan telah dilatih sebelumnya pada dataset ImageNet.
        Model dilatih dan di-fine-tune menggunakan dataset **Kaggle Dogs vs. Cats**, yang berisi lebih dari 25.000 gambar.
        """)
    st.write("Unggah gambar anjing atau kucing, dan model akan memprediksinya!")

    uploaded_file = st.file_uploader("Pilih sebuah gambar...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_to_predict = Image.open(uploaded_file)
        st.image(image_to_predict, caption="Gambar yang Diunggah", use_container_width=True)
        st.write("Sedang memprediksi...")

        result, probability = predict_dogcat(image_to_predict)
        if result:
            st.markdown(f"**Prediksi:** `{result}`")
            st.markdown(f"**Probabilitas:** `{probability * 100:.2f}%`")
            st.balloons()
    
    st.markdown("---")
    if st.button("Kembali ke Halaman Utama"):
        st.query_params.clear()
        st.rerun()

def food_page():
    st.title("🍽️Klasifikasi 101 Makanan")
    st.markdown("Aplikasi ini menggunakan model deep learning untuk mengidentifikasi jenis makanan.")

    with st.expander("Tentang Model & Dataset"):
        st.markdown("""
        Model ini dikembangkan menggunakan arsitektur **MobileNetV2**, sebuah model Convolutional Neural Network (CNN) yang efisien dan cocok untuk aplikasi *web* dan *mobile*. Model ini dilatih menggunakan teknik **transfer learning** pada dataset **Food 101**.

        Dataset **Food 101** berisi 101 kategori makanan yang berbeda, dengan total 101.000 gambar. Model MobileNetV2 yang sudah terlatih pada ImageNet diadaptasi dan di-fine-tune dengan dataset Food 101 untuk dapat mengenali berbagai jenis makanan dengan akurat.
        """)
    st.write("Unggah gambar makanan, dan model akan memprediksi nama makanannya!")

    uploaded_file = st.file_uploader("Pilih sebuah gambar...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_to_predict = Image.open(uploaded_file)
        st.image(image_to_predict, caption="Gambar yang Diunggah", use_container_width=True)
        st.write("Sedang memprediksi...")

        result, probability = predict_food(image_to_predict)
        
        st.markdown(f"**Prediksi:** `{result}`")
        st.markdown(f"**Probabilitas:** `{probability * 100:.2f}%`")

    st.markdown("---")
    if st.button("Kembali ke Halaman Utama"):
        st.query_params.clear()
        st.rerun()

def sentiment_page():
    st.title("💬Klasifikasi Sentimen Teks")
    st.markdown("Aplikasi ini memprediksi sentimen dari teks dengan menggunakan model berbasis Transformers.")

    with st.expander("Tentang Model & Dataset"):
        st.markdown("""
        Model ini menggunakan arsitektur **IndoBertweet**, sebuah model *deep learning* yang dikembangkan khusus untuk bahasa Indonesia dan telah dilatih pada data dari Twitter. Model ini sangat efektif dalam memahami konteks dan nuansa bahasa gaul serta singkatan yang sering digunakan di media sosial.

        Model dilatih pada dataset komentar terkait kasus Tom Lembong. Dataset ini memungkinkan model untuk secara spesifik mengidentifikasi sentimen seperti **Sadness, Anger, Hope, Support, dan Disappointment** dari konteks komentar yang relevan.
        """)
    st.write("Masukkan teks, dan model akan memprediksi sentimennya!")

    user_text = st.text_area("Masukkan teks di sini:", height=150)
    if st.button("Prediksi Sentimen"):
        if user_text:
            result, probability = predict_sentiment(user_text)

            st.markdown(f"**Prediksi:** `{result}`")
            st.markdown(f"**Probabilitas:** `{probability * 100:.2f}%`")
        else:
            st.warning("Mohon masukkan teks terlebih dahulu.")

    st.markdown("---")
    if st.button("Kembali ke Halaman Utama"):
        st.query_params.clear()
        st.rerun()

# --- Logika Utama Aplikasi ---
query_params = st.query_params
page = query_params.get("page", "main")

if page == "dogcat":
    dogcat_page()
elif page == "food":
    food_page()
elif page == "sentiment":
    sentiment_page()
else:

    main_page()



